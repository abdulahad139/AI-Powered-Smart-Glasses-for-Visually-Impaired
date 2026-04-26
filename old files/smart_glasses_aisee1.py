# =============================================================================
# Smart Glasses — AiSee Integrated
# Object Detection (YOLO) + Urdu OCR via AiSee server (no local token.json)
# =============================================================================
#
# WHAT CHANGED FROM YOUR ORIGINAL:
#   OLD: authenticate_google_drive() → reads token.json on the Pi
#   NEW: get_drive_service()         → calls YOUR server with DEVICE_CODE
#                                      server holds the token in DB
#                                      Pi never touches Google directly for OCR
#
# MODES:
#   Object Detection (default) — YOLO runs continuously on camera feed
#   OCR                        — Live preview, capture on Button 2, read text
#
# GPIO BUTTONS:
#   Button 1 (GPIO 16) — Toggle mode (OD ↔ OCR) with audio feedback
#   Button 2 (GPIO 26) — Capture image for OCR (ignored if in OD mode)
#   ESC key            — Quit (keyboard fallback for testing)
#
# SETUP REQUIRED (one time):
#   1. Visit aisee.com/setup → Sign in with Google → enter device code
#   2. That's it. No token.json, no credentials.json needed on Pi.
#
# =============================================================================

import os
import io
import time
import sys
import base64
import threading
import tempfile
import requests
from queue import Queue

try:
    from PIL import Image
    import cv2
    import pyttsx3
    from ultralytics import YOLO
    import RPi.GPIO as GPIO
except ImportError:
    print("Installing required packages...")
    os.system(
        'pip install --quiet '
        'requests gTTS Pillow opencv-python '
        'pyttsx3 ultralytics RPi.GPIO'
    )
    print("✓ Packages installed! Please run the script again.\n")
    sys.exit(0)


# =============================================================================
# CONFIGURATION — Only change these two lines per Pi
# =============================================================================

SERVER_URL  = "https://aisee.onrender.com"  # Your Railway backend URL
DEVICE_CODE = "AIS-4829"                        # Unique per Pi — matches DB + printed card

# --- Camera ---
CAMERA_INDEX         = 0
CAMERA_WARMUP_FRAMES = 10

# --- Image compression (reduces upload time ~70%) ---
MAX_IMAGE_WIDTH  = 1600
MAX_IMAGE_HEIGHT = 1200
JPEG_QUALITY     = 75

# --- Object Detection ---
MODEL_PATH           = r"best(7)_ncnn_model"
CONFIDENCE_THRESHOLD = 0.45
EXCLUDED_CLASSES     = {'glasses', 'calculator', 'pencil'}
OD_FRAME_SKIP        = 5      # Run YOLO every Nth frame
SPEECH_COOLDOWN      = 3      # Seconds before same object is announced again

# --- GPIO Button Pins (BCM numbering) ---
# Button 1 — Mode toggle  (GPIO 16, Physical Pin 36, GND Pin 34)
# Button 2 — Capture OCR  (GPIO 26, Physical Pin 37, GND Pin 39)
BUTTONS = {
    "Button 1": 16,   # Toggle OD ↔ OCR
    "Button 2": 26,   # Capture (OCR mode only)
}
BUTTON_BOUNCETIME = 300  # ms debounce


# =============================================================================
# SHARED STATE
# =============================================================================

current_mode   = ['od']           # 'od' or 'ocr'
ocr_processing = threading.Event()
speech_queue   = Queue()
last_spoken    = {}


# =============================================================================
# ── SERVER / OCR FUNCTIONS (replaces authenticate_google_drive + local OCR)
# =============================================================================

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
        print(f"🔍 Response body: {resp.text}")   # ← ADD THIS

        if resp.status_code == 200:
            data = resp.json()
            claimed = data.get("claimed", False)
            active  = data.get("active", False)
            print(f"🔍 claimed={claimed}, active={active}")  # ← AND THIS
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

    # Speak instructions
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
    NEW OCR pipeline — sends image to YOUR server instead of Google directly.

    OLD FLOW (your original code):
        Pi → Google Drive API (using local token.json) → poll → gTTS → play

    NEW FLOW (AiSee):
        Pi → YOUR server (with DEVICE_CODE) → server looks up DB
           → server uses user's Google token → OCR → gTTS → MP3 stream back
           → Pi plays audio

    Everything else (compression, audio playback) is the same.
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

        # Step 3 — Send to server — server does Drive upload + OCR + gTTS
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

        # Step 4 — Handle response
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
        from urllib.parse import unquote
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

        print(f"✓ Audio received — playing...")
        _play_audio(audio_path)

        # Cleanup
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


# =============================================================================
# ── AUDIO HELPERS
# =============================================================================

def _play_audio(path: str):
    """Play MP3 — tries mpg123, ffplay, vlc in order."""
    print("🔊 Playing audio...")
    for cmd in [f'mpg123 -q "{path}"',
                f'ffplay -nodisp -autoexit -loglevel quiet "{path}"',
                f'vlc --play-and-exit --quiet "{path}"']:
        player = cmd.split()[0].replace('"', '')
        if os.system(f'which {player} > /dev/null 2>&1') == 0:
            os.system(cmd)
            return
    print("⚠️  No audio player found. Run: sudo apt install mpg123")


def _speak_blocking(text: str, lang: str = "en"):
    """Speak English text immediately using pyttsx3 (blocking)."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 235)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine
    except Exception as e:
        print(f"⚠️  TTS error: {e}")


# =============================================================================
# ── SPEECH WORKER (for OD + mode announcements)
# =============================================================================

def speak_worker():
    """
    Single TTS worker thread — handles ALL speech for OD + mode announcements.

    Queue item formats:
      OD detection  : ('od',       label,   position)
      Announcement  : ('announce', message, None)
      Shutdown      : None
    """
    while True:
        try:
            if not speech_queue.empty():
                item = speech_queue.get(timeout=1)
                if item is None:
                    break

                msg_type = item[0]
                current_time = time.time()

                if msg_type == 'announce':
                    _, message, _ = item
                    print(f"📢 Announcing: {message}")

                elif msg_type == 'od':
                    _, label, position = item

                    if label in last_spoken and \
                            current_time - last_spoken[label] < SPEECH_COOLDOWN:
                        continue

                    message = (f"{label} detected in front of you"
                               if position == "center"
                               else f"{label} detected to your {position}")
                    print(f"🔈 OD: {message}")
                    last_spoken[label] = current_time

                    # Clear stale OD backlog
                    while not speech_queue.empty():
                        try:
                            peeked = speech_queue.queue[0]
                            if peeked is not None and peeked[0] == 'od':
                                speech_queue.get_nowait()
                            else:
                                break
                        except Exception:
                            break
                else:
                    continue

                # Speak
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 235)
                    engine.setProperty('volume', 1.0)
                    engine.say(message)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                except Exception as e:
                    print(f"⚠️  TTS error: {e}")

            else:
                time.sleep(0.1)

        except Exception as e:
            print(f"⚠️  Speech worker error: {e}")
            time.sleep(0.1)


# =============================================================================
# ── OBJECT DETECTION HELPERS
# =============================================================================

def get_position(frame_width, x_center):
    if x_center < frame_width // 3:
        return "left"
    elif x_center < 2 * (frame_width // 3):
        return "center"
    else:
        return "right"


def blur_person(image, box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    top = image[y1:y1 + int(0.08 * (y2 - y1)), x1:x2]
    image[y1:y1 + int(0.08 * (y2 - y1)), x1:x2] = cv2.GaussianBlur(top, (15, 15), 0)
    return image


def run_od_on_frame(frame, model):
    try:
        results = model.predict(frame, verbose=False)
        result  = results[0]

        for box in result.boxes:
            label      = result.names[box.cls[0].item()]
            confidence = box.conf[0].item()

            if label.lower() in EXCLUDED_CLASSES:
                continue
            if confidence <= CONFIDENCE_THRESHOLD:
                continue

            print(f"Detected: '{label}' ({confidence:.2f})")

            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            x_center = (x1 + x2) // 2
            color = (255, 0, 0)

            if label in ("people", "fullchair"):
                try:
                    frame = blur_person(frame, box)
                    color = (0, 255, 0)
                except Exception as e:
                    print(f"⚠️  Blur error: {e}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            position = get_position(frame.shape[1], x_center)
            if speech_queue.qsize() < 2:
                speech_queue.put(('od', label, position))

    except Exception as e:
        print(f"⚠️  Detection error: {e}")

    return frame


# =============================================================================
# ── OVERLAY
# =============================================================================

def draw_overlay(frame, mode, ocr_busy):
    display = frame.copy()
    h, w    = display.shape[:2]
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    if mode == 'od':
        text  = "MODE: Object Detection  |  BTN1: Toggle to OCR  |  ESC: Quit"
        color = (0, 200, 255)
    elif ocr_busy:
        text  = "MODE: OCR  |  Processing... Please wait"
        color = (0, 0, 255)
    else:
        text  = "MODE: OCR  |  BTN2: Scan  |  BTN1: Toggle to OD  |  ESC: Quit"
        color = (0, 220, 0)

    cv2.putText(display, text, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return display


# =============================================================================
# ── MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  AiSee Smart Glasses — Object Detection + Urdu OCR")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # STEP 1 — Check device setup (replaces authenticate_google_drive)
    # ------------------------------------------------------------------
    print(f"🔍 Checking device setup (code: {DEVICE_CODE})...")
    if not check_device_setup():
        wait_for_setup()
    else:
        print("✓ Device is set up and active!\n")

    # ------------------------------------------------------------------
    # STEP 2 — Load YOLO model
    # ------------------------------------------------------------------
    print(f"🤖 Loading YOLO model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print("✓ Model loaded!\n")
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 3 — Open camera
    # ------------------------------------------------------------------
    print(f"📷 Opening camera (index {CAMERA_INDEX})...")
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cam.isOpened():
        print("❌ Could not open camera.")
        sys.exit(1)

    print(f"   Warming up ({CAMERA_WARMUP_FRAMES} frames)...")
    for _ in range(CAMERA_WARMUP_FRAMES):
        cam.read()
    print("✓ Camera ready!\n")

    # ------------------------------------------------------------------
    # STEP 4 — Start TTS worker thread
    # ------------------------------------------------------------------
    tts_thread = threading.Thread(target=speak_worker, daemon=True)
    tts_thread.start()

    # ------------------------------------------------------------------
    # STEP 5 — Startup announcement
    # ------------------------------------------------------------------
    _speak_blocking("System activated. Object detection running.")

    # ------------------------------------------------------------------
    # STEP 6 — Display window
    # ------------------------------------------------------------------
    cv2.namedWindow("AiSee Smart Glasses", cv2.WINDOW_NORMAL)

    print("=" * 60)
    print("  ✅ System ready!")
    print(f"  Button 1 (GPIO {BUTTONS['Button 1']}) — Toggle OD ↔ OCR")
    print(f"  Button 2 (GPIO {BUTTONS['Button 2']}) — Capture (OCR mode only)")
    print("  ESC key            — Quit")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # STEP 7 — GPIO setup
    # ------------------------------------------------------------------

    # Shared latest frame — GPIO callback reads from here
    latest_frame = [None]

    def announce(message):
        speech_queue.put(('announce', message, None))

    def on_mode_toggle(channel=None):
        """Button 1 — Toggle between OD and OCR."""
        if current_mode[0] == 'od':
            current_mode[0] = 'ocr'
            while not speech_queue.empty():
                try:
                    speech_queue.get_nowait()
                except Exception:
                    break
            print("\n🔀 Switched to OCR mode — press Button 2 to scan\n")
            announce("OCR mode activated")
        else:
            if ocr_processing.is_set():
                print("⚠️  OCR still running — please wait.\n")
                announce("Please wait, scan in progress")
                return
            current_mode[0] = 'od'
            print("\n🔀 Switched to Object Detection mode\n")
            announce("Object detection resumed")

    def on_capture(channel=None):
        """Button 2 — Capture frame and send to AiSee server for OCR."""
        if current_mode[0] != 'ocr':
            return

        if ocr_processing.is_set():
            print("⚠️  Already processing — please wait.\n")
            announce("Already processing, please wait")
            return

        frame = latest_frame[0]
        if frame is None:
            print("⚠️  No frame available yet.")
            return

        ocr_processing.set()
        announce("Image captured, processing")

        def pipeline():
            try:
                run_ocr_pipeline(frame.copy())
            finally:
                ocr_processing.clear()

        threading.Thread(target=pipeline, daemon=True).start()

    # GPIO init
    GPIO.setmode(GPIO.BCM)
    for name, pin in BUTTONS.items():
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"✓ {name} registered on GPIO {pin}")

    GPIO.add_event_detect(BUTTONS["Button 1"], GPIO.FALLING,
                          callback=on_mode_toggle, bouncetime=BUTTON_BOUNCETIME)
    GPIO.add_event_detect(BUTTONS["Button 2"], GPIO.FALLING,
                          callback=on_capture, bouncetime=BUTTON_BOUNCETIME)
    print("✓ GPIO buttons active\n")

    # ------------------------------------------------------------------
    # STEP 8 — Main frame loop
    # ------------------------------------------------------------------
    frame_count = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("❌ Lost camera feed.")
                break

            frame_count += 1
            latest_frame[0] = frame.copy()

            if current_mode[0] == 'od':
                if frame_count % OD_FRAME_SKIP == 0:
                    frame = run_od_on_frame(frame, model)

            display_frame = draw_overlay(frame, current_mode[0], ocr_processing.is_set())
            cv2.imshow("AiSee Smart Glasses", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                print("\n🛑 ESC pressed — shutting down...")
                break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")

    finally:
        print("Cleaning up...")
        speech_queue.put(None)
        time.sleep(0.5)
        GPIO.cleanup()
        cam.release()
        cv2.destroyAllWindows()
        print("✓ Done. Goodbye!\n")


if __name__ == "__main__":
    main()
