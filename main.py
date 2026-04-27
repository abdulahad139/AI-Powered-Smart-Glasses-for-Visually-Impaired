# =============================================================================
# Main — Smart Glasses entry point with GPIO and camera loop
# =============================================================================
#
# Integrates all modules: config, detection, tracking, distance, OCR, audio.
# Handles GPIO buttons, camera capture, and the main processing loop.
#
# OCR uses Google Drive API directly on the Pi (no AiSee server needed).
# OD always starts regardless of internet or Drive auth state.
#
# =============================================================================

import os
import sys
import time
import socket
import threading

# Try imports, install if missing
try:
    import cv2
    import RPi.GPIO as GPIO
except ImportError:
    print("Installing required packages...")
    os.system(
        'pip install --quiet '
        'requests gTTS Pillow opencv-python '
        'pyttsx3 ultralytics RPi.GPIO '
        'google-api-python-client google-auth-oauthlib'
    )
    print("✓ Packages installed! Please run the script again.\n")
    sys.exit(0)

from config import (
    CAMERA_INDEX,
    CAMERA_WARMUP_FRAMES,
    BUTTONS,
    BUTTON_BOUNCETIME,
    OD_FRAME_SKIP,
)
from detection import run_dual_detection, draw_detections
from tracking import (
    expire_lost_tracks,
    process_detections,
    clear_track_states,
    trigger_manual_scan,
)
from ocr import get_drive_service, run_ocr_pipeline
from audio import (
    start_speech_worker,
    stop_speech_worker,
    announce,
    speech_queue,
    _speak_blocking,
)


# =============================================================================
# SHARED STATE
# =============================================================================

current_mode   = ['od']   # 'od' or 'ocr'
ocr_processing = threading.Event()
ocr_available  = [False]  # True only if Google Drive authenticated successfully


# =============================================================================
# CONNECTIVITY HELPER
# =============================================================================

def is_google_reachable(timeout: float = 3.0) -> bool:
    """
    Check if Google APIs are reachable.
    Used before OCR capture to catch mid-session connectivity loss.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.connect(("www.googleapis.com", 443))
        socket.close()
        return True
    except OSError:
        return False


# =============================================================================
# OVERLAY
# =============================================================================

def draw_overlay(frame, mode, ocr_busy, ocr_ok):
    """Draw HUD bar at bottom of frame showing current mode and controls."""
    display = frame.copy()
    h, w    = display.shape[:2]
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    if mode == 'od':
        ocr_hint = "BTN1: OCR" if ocr_ok else "BTN1: OCR (unavailable)"
        text  = f"MODE: Dual OD  |  [P]=Pretrained  [C]=Custom  |  {ocr_hint}  |  BTN2: Scan  |  ESC: Quit"
        color = (0, 200, 255)
    elif ocr_busy:
        text  = "MODE: OCR  |  Processing... Please wait"
        color = (0, 0, 255)
    else:
        text  = "MODE: OCR  |  BTN2: Scan  |  BTN1: Toggle to OD  |  ESC: Quit"
        color = (0, 220, 0)

    cv2.putText(display, text, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
    return display


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  AiSee Smart Glasses — Dual Model OD + Edge OCR + Distance")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # STEP 1 — Check Google Drive authentication
    # OD always starts regardless of this result.
    # OCR is only enabled if Drive auth succeeds.
    # ------------------------------------------------------------------
    print("🔐 Checking Google Drive authentication...")
    drive_service = get_drive_service()

    if drive_service:
        ocr_available[0] = True
        print("✓ Google Drive authenticated. OCR available.\n")
    else:
        ocr_available[0] = False
        print("⚠️  Google Drive not authenticated. OCR unavailable.\n")
        _speak_blocking(
            "Text reading unavailable. "
            "Google Drive credentials not set up. "
            "Object detection is running."
        )

    # ------------------------------------------------------------------
    # STEP 2 — Open camera
    # ------------------------------------------------------------------
    print(f"📷 Opening camera (index {CAMERA_INDEX})...")
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        print("❌ Could not open camera.")
        sys.exit(1)

    print(f"   Warming up ({CAMERA_WARMUP_FRAMES} frames)...")
    for _ in range(CAMERA_WARMUP_FRAMES):
        cam.read()
    print("✓ Camera ready!\n")

    # ------------------------------------------------------------------
    # STEP 3 — Start TTS worker thread
    # ------------------------------------------------------------------
    print("🎤 Starting speech worker...")
    start_speech_worker()
    print("✓ Speech worker started!\n")

    # ------------------------------------------------------------------
    # STEP 4 — Startup announcement
    # ------------------------------------------------------------------
    _speak_blocking("System activated. Object detection running.")

    # ------------------------------------------------------------------
    # STEP 5 — Display window
    # ------------------------------------------------------------------
    cv2.namedWindow("AiSee Smart Glasses", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AiSee Smart Glasses", 640, 480)

    print("=" * 60)
    print("  ✅ System ready!")
    print(f"  Button 1 (GPIO {BUTTONS['Button 1']}) — Toggle OD ↔ OCR")
    print(f"  Button 2 (GPIO {BUTTONS['Button 2']}) — Manual scan (OD) / Capture (OCR)")
    print(f"  OCR available: {ocr_available[0]}")
    print("  ESC key — Quit")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # STEP 6 — GPIO setup
    # ------------------------------------------------------------------

    latest_frame    = [None]
    last_detections = []

    def on_mode_toggle(channel=None):
        """
        Button 1 — Toggle between OD and OCR.
        If Google Drive auth failed or no internet, stays in OD and tells user.
        """
        if current_mode[0] == 'od':
            # Block switch if OCR is not available
            if not ocr_available[0]:
                print("⚠️  OCR unavailable — Google Drive not authenticated.")
                _speak_blocking(
                    "Text reading is unavailable. "
                    "Please set up Google Drive credentials and restart."
                )
                return

            # OCR available but check connectivity before switching
            if not is_google_reachable():
                print("⚠️  OCR unavailable — no internet connection.")
                _speak_blocking(
                    "Text reading is unavailable. "
                    "Please connect to Wi-Fi to use this feature."
                )
                return

            current_mode[0] = 'ocr'
            while not speech_queue.empty():
                try:
                    speech_queue.get_nowait()
                except Exception:
                    break
            clear_track_states()
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
        """Button 2 — Manual scan in OD mode, OCR capture in OCR mode."""
        if current_mode[0] == 'od':
            print("\n🔘 Manual scan triggered (Button 2)")
            announce("Scanning")
            frame_width = latest_frame[0].shape[1] if latest_frame[0] is not None else 640
            if last_detections:
                trigger_manual_scan(last_detections, frame_width)
            elif latest_frame[0] is not None:
                temp_dets = run_dual_detection(latest_frame[0])
                trigger_manual_scan(temp_dets, frame_width)
            else:
                announce("No objects currently detected")
            return

        # OCR mode
        if ocr_processing.is_set():
            print("⚠️  Already processing — please wait.\n")
            announce("Already processing, please wait")
            return

        # Re-check Google connectivity at capture time
        # (connection may have dropped mid-session after mode switch)
        if not is_google_reachable():
            print("❌ No internet — OCR unavailable.")
            _speak_blocking(
                "No internet connection. "
                "Please connect to Wi-Fi to use text reading."
            )
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
    # STEP 7 — Main frame loop
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
                    last_detections = run_dual_detection(frame)
                    expire_lost_tracks()
                    process_detections(last_detections, frame.shape[1])
            else:
                last_detections = []

            frame         = draw_detections(frame, last_detections)
            display_frame = draw_overlay(
                frame, current_mode[0], ocr_processing.is_set(), ocr_available[0]
            )
            cv2.imshow("AiSee Smart Glasses", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("\n🛑 ESC pressed — shutting down...")
                break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")

    finally:
        print("Cleaning up...")
        stop_speech_worker()
        GPIO.cleanup()
        cam.release()
        cv2.destroyAllWindows()
        print("✓ Done. Goodbye!\n")


if __name__ == "__main__":
    main()