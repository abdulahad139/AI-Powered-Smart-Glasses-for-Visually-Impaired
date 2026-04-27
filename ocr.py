# =============================================================================
# OCR — Standalone OCR pipeline using Google Drive (no server)
# =============================================================================
#
# Handles OCR using Google Drive API:
#   1. Compress image → 2. Upload to Drive as Google Doc (triggers OCR)
#   3. Poll for text → 4. Clean text → 5. Smart TTS (Urdu/English split)
#
# Also handles device setup verification (requires server for initial setup).
#
# =============================================================================

import io
import os
import re
import tempfile
import time
import threading
import requests
from urllib.parse import unquote

from PIL import Image
import cv2
from gtts import gTTS
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from config import (
    SERVER_URL,
    DEVICE_CODE,
    MAX_IMAGE_WIDTH,
    MAX_IMAGE_HEIGHT,
    JPEG_QUALITY,
)
from audio import _speak_blocking, _play_audio


# =============================================================================
# GOOGLE DRIVE AUTHENTICATION
# =============================================================================

# OAuth scopes required for Drive file operations
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Token and credentials file paths
TOKEN_FILE       = 'token.json'
CREDENTIALS_FILE = 'credentials.json'

# You need to set these from Google Cloud Console
CLIENT_ID     = ""  # Set from Google Cloud Console
CLIENT_SECRET = ""  # Set from Google Cloud Console


def authenticate_google_drive():
    """
    Authenticate with Google Drive using OAuth 2.0.
    
    First run  : opens browser for Google login → saves token.json
    Future runs: silently refreshes token — works headlessly
    
    Requires:
    - credentials.json from Google Cloud Console (OAuth 2.0 client ID)
    - Or set CLIENT_ID and CLIENT_SECRET directly above
    """
    creds = None

    # Load existing token if available
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # If no valid credentials, need to authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Refresh expired token
            creds.refresh(Request())
        else:
            # Need new credentials
            if not CLIENT_ID or not CLIENT_SECRET:
                # Try loading from credentials.json
                if os.path.exists(CREDENTIALS_FILE):
                    from google_auth_oauthlib.flow import Flow
                    flow = Flow.from_client_secrets_file(
                        CREDENTIALS_FILE, scopes=SCOPES)
                    # For headless operation, use port token
                    flow.redirect_uri = 'http://localhost:1/'
                    auth_url, _ = flow.authorization_url(
                        access_type='offline', prompt='consent')
                    print(f"\n🔗 Please visit this URL to authorize:")
                    print(f"   {auth_url}")
                    print("\n📋 Or if you have credentials.json, run:")
                    print("   python -c \"from google_auth_oauthlib.flow import Flow; "
                          "flow = Flow.from_client_secrets_file('credentials.json', "
                          "['https://www.googleapis.com/auth/drive.file']); "
                          "flow.run_local_server(port=1)\"")
                    print("\n⚠️  For headless Pi, set CLIENT_ID and CLIENT_SECRET above")
                    print("    and use the device code flow.\n")
                    return None
                else:
                    print("❌ No credentials.json found and CLIENT_ID/CLIENT_SECRET not set.")
                    print("   Download from Google Cloud Console → APIs → Credentials → OAuth 2.0")
                    return None

            # Use client ID/secret directly
            from google_auth_oauthlib.flow import Flow
            client_config = {
                "web": {
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "redirect_uris": ["http://localhost:1/"]
                }
            }
            flow = Flow.from_client_config(client_config, scopes=SCOPES)
            flow.redirect_uri = 'http://localhost:1/'
            
            # For headless, try device flow or ask for auth code
            print("\n🔐 Google Drive authentication required.")
            print("   On a computer with browser, visit:")
            auth_url, _ = flow.authorization_url(access_type='offline')
            print(f"   {auth_url}")
            code = input("   Enter the authorization code: ")
            flow.fetch_token(code=code)
            creds = flow.credentials

            # Save token for future runs
            with open(TOKEN_FILE, 'w') as f:
                creds.to_json()

    return build('drive', 'v3', credentials=creds)


# Lazy-loaded Drive service
_drive_service = None


def get_drive_service():
    """Lazy-load Google Drive service."""
    global _drive_service
    if _drive_service is None:
        _drive_service = authenticate_google_drive()
    return _drive_service


# =============================================================================
# LANGUAGE DETECTION + TEXT SPLITTING
# =============================================================================

# Urdu Unicode ranges: Arabic block + extended Arabic + presentation forms
URDU_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeFF]+")
ENG_RE  = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\s,.\-']*")


def split_by_language(text: str) -> list:
    """
    Split mixed Urdu/English text into language-tagged segments.
    
    Returns:
        List of (text, lang) tuples where lang is 'ur' or 'en'
    """
    if not text:
        return []

    segments = []
    current_lang = None
    current_text = []

    # Split into Urdu and English runs
    urdu_matches = list(URDU_RE.finditer(text))
    eng_matches = list(ENG_RE.finditer(text))

    # Merge and sort all matches by position
    all_matches = []
    for m in urdu_matches:
        all_matches.append((m.start(), m.end(), 'ur', m.group()))
    for m in eng_matches:
        all_matches.append((m.start(), m.end(), 'en', m.group()))
    all_matches.sort(key=lambda x: x[0])

    # Build segments
    for start, end, lang, segment_text in all_matches:
        segment_text = segment_text.strip()
        if not segment_text:
            continue

        if current_lang == lang:
            current_text.append(segment_text)
        else:
            if current_text:
                segments.append((' '.join(current_text), current_lang))
            current_lang = lang
            current_text = [segment_text]

    if current_text:
        segments.append((' '.join(current_text), current_lang))

    return segments


# =============================================================================
# OCR TEXT CLEANING
# =============================================================================

def _clean_ocr_text(raw: str) -> str:
    """
    Clean OCR output — remove artifacts, control chars, noise.
    """
    if not raw:
        return ""

    # Remove null bytes and control characters (except newlines/tabs)
    cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', raw)

    # Remove page numbers and common OCR junk patterns
    cleaned = re.sub(r'Page \d+ of \d+', '', cleaned)
    cleaned = re.sub(r'-\s*-\s*-\s*-\s*-\s*', '', cleaned)

    # Normalize whitespace
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)

    # Remove orphan single characters (likely OCR errors)
    lines = cleaned.split('\n')
    filtered_lines = []
    for line in lines:
        line = line.strip()
        # Keep lines with meaningful content
        if len(line) > 1 or line.isdigit():
            filtered_lines.append(line)
        elif line and any(c in line for c in 'آأؤإابةتثجحخدذرزسشصضطظعغفقكلمنهوي'):
            # Keep Urdu characters even if single
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


# =============================================================================
# SMART TTS — URDU/ENGLISH SPLITTING
# =============================================================================

def _gtts_segment(text: str, lang: str) -> bytes:
    """Convert a single text segment to MP3 bytes using gTTS."""
    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _concatenate_mp3s(mp3_chunks: list) -> io.BytesIO:
    """
    Concatenate multiple MP3 byte chunks into one BytesIO buffer.
    Simple byte concatenation works for MP3 — each chunk is a valid
    MP3 frame sequence.
    """
    buf = io.BytesIO()
    for chunk in mp3_chunks:
        buf.write(chunk)
    buf.seek(0)
    return buf


def text_to_speech_smart(text: str) -> io.BytesIO:
    """
    Convert mixed Urdu/English text to a single MP3 stream.

    Strategy:
    1. Split text into language-tagged segments via split_by_language()
    2. Call gTTS separately for each segment with the correct lang code
    3. Concatenate all MP3 chunks into one buffer
    4. Fall back to full-text Urdu TTS if splitting fails entirely
    """
    segments = split_by_language(text)

    if not segments:
        segments = [(text, "ur")]

    print(f"🗣️  TTS segments ({len(segments)} total):")
    for seg_text, lang in segments:
        preview = seg_text[:40] + ("..." if len(seg_text) > 40 else "")
        print(f"   [{lang}] {preview}")

    mp3_chunks = []
    errors     = []

    for seg_text, lang in segments:
        try:
            chunk = _gtts_segment(seg_text, lang)
            mp3_chunks.append(chunk)
        except Exception as e:
            errors.append(f"[{lang}] '{seg_text[:30]}': {e}")
            # Try the other language as fallback for this segment
            fallback = "en" if lang == "ur" else "ur"
            try:
                chunk = _gtts_segment(seg_text, fallback)
                mp3_chunks.append(chunk)
                print(f"   ⚠️  Fell back to [{fallback}] for segment")
            except Exception:
                print(f"   ❌ Skipping segment — both langs failed: {seg_text[:30]}")

    if not mp3_chunks:
        # Last resort — full text as Urdu
        print("⚠️  All segments failed, trying full text as Urdu...")
        try:
            chunk = _gtts_segment(text, "ur")
            mp3_chunks.append(chunk)
        except Exception as e:
            print(f"❌ TTS completely failed: {e}")
            return io.BytesIO()

    if errors:
        print(f"⚠️  TTS had {len(errors)} segment error(s) (recovered)")

    return _concatenate_mp3s(mp3_chunks)


# =============================================================================
# AUTH CACHE (for offline capability)
# =============================================================================

AUTH_CACHE_PATH = "/opt/aisee/.auth_cache"

# =============================================================================
# IMAGE COMPRESSION
# =============================================================================

def compress_image_bytes(image_bytes: bytes) -> bytes:
    """
    Resize and compress image bytes before uploading to Google Drive.
    Reduces file size by ~70%, cutting upload time significantly.
    """
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode in ('RGBA', 'P', 'L'):
        img = img.convert('RGB')

    original_kb = len(image_bytes) / 1024
    img.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.LANCZOS)

    out = io.BytesIO()
    img.save(out, 'JPEG', quality=JPEG_QUALITY, optimize=True)
    compressed = out.getvalue()

    print(f"🖼️  Compressed: {original_kb:.0f} KB → {len(compressed)/1024:.0f} KB "
          f"({(1 - len(compressed)/len(image_bytes))*100:.0f}% reduction)")

    return compressed


# =============================================================================
# GOOGLE DRIVE OCR OPERATIONS
# =============================================================================

OCR_POLL_INTERVAL = 1    # Seconds between retries
OCR_MAX_RETRIES   = 30   # Max wait = OCR_POLL_INTERVAL × OCR_MAX_RETRIES


def _upload_to_drive_with_ocr(drive_service, image_path: str) -> str:
    """
    Upload compressed image to Google Drive and trigger OCR.
    Setting mimeType to a Google Doc tells Drive to run OCR on the image.
    Returns the Drive file ID.
    """
    file_metadata = {
        'name':     f"AiSee_OCR_{int(time.time())}",
        'mimeType': 'application/vnd.google-apps.document',   # Triggers OCR
    }
    media = MediaFileUpload(image_path, mimetype='image/jpeg', resumable=True)

    print("📤 Uploading to Google Drive (OCR enabled)...")
    uploaded = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        ocrLanguage='ur',           # 'ur' covers both Urdu + English in practice
        fields='id'
    ).execute()

    file_id = uploaded.get('id')
    print(f"✓ Upload complete — Drive file ID: {file_id}")
    return file_id


def _poll_for_ocr_result(drive_service, file_id: str) -> str | None:
    """
    Adaptively poll Google Drive until OCR text is available.
    Returns cleaned text string, or None if timed out.
    """
    print(f"⏳ Polling for OCR result (every {OCR_POLL_INTERVAL}s, max {OCR_MAX_RETRIES} attempts)...")

    for attempt in range(1, OCR_MAX_RETRIES + 1):
        try:
            request = drive_service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            raw_text = fh.getvalue().decode('utf-8')
            cleaned  = _clean_ocr_text(raw_text)

            if cleaned.strip():
                print(f"✓ OCR complete after ~{attempt} second(s)\n")
                return cleaned

        except Exception:
            # Drive returns errors while OCR is still processing — this is normal
            pass

        print(f"   Attempt {attempt}/{OCR_MAX_RETRIES} — still processing...")
        time.sleep(OCR_POLL_INTERVAL)

    print("❌ OCR timed out — no text returned.")
    return None


def _delete_drive_file(drive_service, file_id: str):
    """Delete the temporary Google Doc from Drive. Runs in a background thread."""
    try:
        drive_service.files().delete(fileId=file_id).execute()
        print(f"🗑️  Drive temp file deleted (ID: {file_id})")
    except Exception as e:
        print(f"⚠️  Could not delete Drive file: {e}")


# =============================================================================
# MAIN OCR PIPELINE
# =============================================================================

def run_ocr_pipeline(frame):
    """
    Full standalone OCR pipeline — no server involved.

    Flow:
        Capture frame
             ↓
        Compress image  (~70% size reduction for faster upload)
             ↓
        Upload to Google Drive  (mimeType=Google Doc triggers OCR)
             ↓
        Adaptive poll until text is ready
             ↓
        Clean OCR text  (remove artifacts, control chars, noise)
             ↓
        Split by language  (Urdu segments → gTTS 'ur', English → gTTS 'en')
             ↓
        Concatenate MP3 chunks → save to temp file → play
             ↓
        Delete Drive file in background
    """
    print(f"\n{'='*60}")
    print("🔘 OCR pipeline started (standalone — no server)...")
    print(f"{'='*60}\n")

    # Get Drive service (authenticates if needed)
    drive_service = get_drive_service()
    if not drive_service:
        _speak_blocking("Google Drive not authenticated. Please set up credentials.")
        return

    tmp_img          = None
    tmp_drive_file_id = None
    audio_path       = None

    try:
        # STEP 1 — Encode and compress frame
        _, img_encoded  = cv2.imencode('.jpg', frame)
        image_bytes     = img_encoded.tobytes()
        compressed_bytes = compress_image_bytes(image_bytes)

        # STEP 2 — Write compressed bytes to a temp file for Drive upload
        tmp_img = os.path.join(tempfile.gettempdir(), f'aisee_ocr_{int(time.time())}.jpg')
        with open(tmp_img, 'wb') as f:
            f.write(compressed_bytes)

        # STEP 3 — Upload to Drive and trigger OCR
        tmp_drive_file_id = _upload_to_drive_with_ocr(drive_service, tmp_img)

        # STEP 4 — Poll until text is ready
        extracted_text = _poll_for_ocr_result(drive_service, tmp_drive_file_id)

        if not extracted_text:
            _speak_blocking("No text found in the image. Please try again.")
            return

        # STEP 5 — Print extracted text to terminal
        print("\n" + "=" * 60)
        print("📄  EXTRACTED TEXT:")
        print("=" * 60)
        print(extracted_text)
        print("=" * 60 + "\n")

        # STEP 6 — Delete Drive file in background (don't block TTS)
        threading.Thread(
            target=_delete_drive_file,
            args=(drive_service, tmp_drive_file_id),
            daemon=True
        ).start()
        tmp_drive_file_id = None   # Already being handled

        # STEP 7 — Smart language-split TTS
        print("🎙️  Generating smart Urdu/English speech...")
        audio_buffer = text_to_speech_smart(extracted_text)

        # STEP 8 — Save MP3 to temp file and play
        audio_path = os.path.join(tempfile.gettempdir(), 'aisee_ocr_result.mp3')
        with open(audio_path, 'wb') as f:
            f.write(audio_buffer.read())

        _play_audio(audio_path)

        print(f"\n{'='*60}")
        print("✓ OCR DONE — Press Button 2 to scan again, Button 1 for OD mode")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ OCR pipeline error: {e}")
        import traceback
        traceback.print_exc()
        _speak_blocking("An error occurred during OCR. Please try again.")

    finally:
        # Clean up temp image file
        if tmp_img and os.path.exists(tmp_img):
            try:
                os.remove(tmp_img)
            except Exception:
                pass

        # Clean up audio file
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

        # If Drive file was never cleaned (pipeline crashed before background thread)
        if tmp_drive_file_id and drive_service:
            try:
                drive_service.files().delete(fileId=tmp_drive_file_id).execute()
            except Exception:
                pass


def run_ocr_async(frame, ocr_processing: threading.Event):
    """
    Run OCR pipeline in a background thread.
    Sets the ocr_processing event before starting and clears it when done.
    """
    def pipeline():
        try:
            run_ocr_pipeline(frame)
        finally:
            ocr_processing.clear()

    ocr_processing.set()
    threading.Thread(target=pipeline, daemon=True).start()