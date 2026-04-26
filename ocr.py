# =============================================================================
# OCR — AiSee server OCR pipeline and device setup
# =============================================================================
#
# Handles communication with the AiSee server for OCR processing,
# image compression, and device setup verification.
#
# =============================================================================

import io
import os
import tempfile
import base64
import time
import threading
import requests
from urllib.parse import unquote

from PIL import Image
import cv2

from config import (
    SERVER_URL,
    DEVICE_CODE,
    MAX_IMAGE_WIDTH,
    MAX_IMAGE_HEIGHT,
    JPEG_QUALITY,
)
from audio import _speak_blocking, _play_audio
import requests


# =============================================================================
# AUTH CACHE (for offline capability)
# =============================================================================

AUTH_CACHE_PATH = "/opt/aisee/.auth_cache"


def save_auth_cache():
    """Save verified auth state to disk after successful server check."""
    import json
    import os
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(AUTH_CACHE_PATH), exist_ok=True)
        with open(AUTH_CACHE_PATH, "w") as f:
            json.dump({"claimed": True, "timestamp": time.time()}, f)
        print("✓ Auth state cached locally.")
    except Exception as e:
        print(f"⚠️  Could not save auth cache: {e}")


def load_auth_cache() -> bool:
    """Return True if device was previously verified, False if no cache exists."""
    import json
    try:
        with open(AUTH_CACHE_PATH, "r") as f:
            data = json.load(f)
            return data.get("claimed", False)
    except Exception:
        return False


def is_server_reachable(timeout: float = 3.0) -> bool:
    """Check if the server is reachable."""
    try:
        requests.get(SERVER_URL, timeout=timeout)
        return True
    except Exception:
        return False


def check_device_setup() -> bool:
    """
    Check if this Pi's DEVICE_CODE has been claimed by a user.
    Called on boot — if not set up, announces instructions and waits.
    """
    try:
        resp = requests.get(
            f"{SERVER_URL}/ocr/status/{DEVICE_CODE}",
            timeout=10
        )
        print(f"🔍 Status check → HTTP {resp.status_code}")
        print(f"🔍 Response body: {resp.text}")

        if resp.status_code == 200:
            data    = resp.json()
            claimed = data.get("claimed", False)
            active  = data.get("active", False)
            print(f"🔍 claimed={claimed}, active={active}")
            return claimed and active
    except requests.ConnectionError:
        print("⚠️  Cannot reach server — check internet connection")
    except Exception as e:
        print(f"⚠️  Status check error: {e}")
    return False


def wait_for_setup():
    """
    Device not yet claimed. Announce device code and poll until user sets up
    via aisee.com/setup. Replaces the old browser-based OAuth flow.
    """
    spaced = " ".join(DEVICE_CODE)   # "A I S - 4 8 2 9"

    print(f"\n{'='*50}")
    print(f"  Device not set up yet!")
    print(f"  Visit aisee.com/setup and enter: {DEVICE_CODE}")
    print(f"{'='*50}\n")

    _speak_blocking(
        f"Welcome to AiSee. "
        f"Please visit aisee dot com slash setup, sign in with Google, "
        f"and enter your device code: {spaced}. "
        f"I will check every 15 seconds."
    )

    print("Polling server every 15 seconds...")
    while True:
        time.sleep(15)
        if check_device_setup():
            print("✓ Device claimed! Starting up...")
            _speak_blocking("Setup complete! AiSee is ready.")
            return
        print("   Still waiting for setup...")


def compress_image_bytes(image_bytes: bytes) -> bytes:
    """Resize + compress image bytes before sending to server."""
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


def run_ocr_pipeline(frame):
    """
    OCR pipeline — sends image to AiSee server.

    Flow:
        Pi → AiSee server (with DEVICE_CODE)
           → server looks up user's Google token in DB
           → server runs OCR via Google
           → server generates MP3 via gTTS
           → MP3 streamed back to Pi
           → Pi plays audio
    """
    print(f"\n{'='*60}")
    print("🔘 OCR pipeline started (via AiSee server)...")
    print(f"{'='*60}\n")

    try:
        # Step 1 — Convert OpenCV frame to JPEG bytes
        _, img_encoded = cv2.imencode('.jpg', frame)
        image_bytes = img_encoded.tobytes()

        # Step 2 — Compress before sending
        compressed_bytes = compress_image_bytes(image_bytes)

        # Step 3 — Send to server
        print(f"📤 Sending to AiSee server ({SERVER_URL})...")
        image_b64 = base64.b64encode(compressed_bytes).decode()

        resp = requests.post(
            f"{SERVER_URL}/ocr/process",
            json={
                "device_code": DEVICE_CODE,
                "image_data":  image_b64,
            },
            timeout=60,
            stream=True,
        )

        # Step 4 — Handle response codes
        if resp.status_code == 403:
            print("❌ Device not set up or revoked.")
            _speak_blocking("Please set up your AiSee glasses before use.")
            return

        if resp.status_code == 422:
            print("❌ No text found in image.")
            _speak_blocking("No text found in the image. Please try again.")
            return

        if resp.status_code != 200:
            print(f"❌ Server error: {resp.status_code}")
            _speak_blocking("Server error. Please try again.")
            return

        # Step 5 — Print extracted text to terminal
        extracted_text = unquote(resp.headers.get("X-Extracted-Text", ""))
        if extracted_text:
            print("\n" + "=" * 60)
            print("📄  EXTRACTED TEXT:")
            print("=" * 60)
            print(extracted_text)
            print("=" * 60 + "\n")
        else:
            print("⚠️  No extracted text header received.")

        # Step 6 — Save and play the MP3 streamed back from server
        audio_path = os.path.join(tempfile.gettempdir(), 'ocr_result.mp3')
        with open(audio_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print("✓ Audio received — playing...")
        _play_audio(audio_path)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        print(f"\n{'='*60}")
        print("✓ OCR DONE — Press Button 2 to scan again, Button 1 to return to OD")
        print(f"{'='*60}\n")

    except requests.ConnectionError:
        print("❌ Cannot connect to server.")
        _speak_blocking("Cannot connect to server. Please check internet.")
    except requests.Timeout:
        print("❌ Request timed out.")
        _speak_blocking("Request timed out. Please try again.")
    except Exception as e:
        print(f"\n❌ OCR pipeline error: {e}")
        import traceback
        traceback.print_exc()
        _speak_blocking("An error occurred. Please try again.")


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