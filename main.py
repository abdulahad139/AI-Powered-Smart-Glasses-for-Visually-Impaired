# =============================================================================
# Main — Smart Glasses entry point with GPIO and camera loop
# =============================================================================
#
# Integrates all modules: config, detection, tracking, distance, OCR, audio.
# Handles GPIO buttons, camera capture, and the main processing loop.
#
# =============================================================================

import os
import sys
import time
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
        'pyttsx3 ultralytics RPi.GPIO'
    )
    print("✓ Packages installed! Please run the script again.\n")
    sys.exit(0)

# Import all modules
from config import (
    SERVER_URL,
    DEVICE_CODE,
    CAMERA_INDEX,
    CAMERA_WARMUP_FRAMES,
    BUTTONS,
    BUTTON_BOUNCETIME,
    OD_FRAME_SKIP,
)
from detection import run_dual_detection, draw_detections
from tracking import expire_lost_tracks, process_detections, clear_track_states, trigger_manual_scan
from ocr import check_device_setup, wait_for_setup, run_ocr_pipeline
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

current_mode   = ['od']           # 'od' or 'ocr'
ocr_processing = threading.Event()


# =============================================================================
# OVERLAY
# =============================================================================

def draw_overlay(frame, mode, ocr_busy):
    """Draw HUD bar at bottom of frame showing current mode and controls."""
    display = frame.copy()
    h, w    = display.shape[:2]
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    if mode == 'od':
        text  = "MODE: Dual OD  |  [P]=Pretrained  [C]=Custom  [T]=Tracked  |  BTN1: OCR  |  ESC: Quit"
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
    print("  AiSee Smart Glasses — Dual Model OD + Urdu OCR + Distance")
    print("  (Modular version)")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # STEP 1 — Check device setup
    # ------------------------------------------------------------------
    print(f"🔍 Checking device setup (code: {DEVICE_CODE})...")
    if not check_device_setup():
        wait_for_setup()
    else:
        print("✓ Device is set up and active!\n")

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
    tts_thread = start_speech_worker()
    print("✓ Speech worker started!\n")

    # ------------------------------------------------------------------
    # STEP 4 — Startup announcement
    # ------------------------------------------------------------------
    _speak_blocking("System activated. Dual model object detection with distance estimation running.")

    # ------------------------------------------------------------------
    # STEP 5 — Display window
    # ------------------------------------------------------------------
    cv2.namedWindow("AiSee Smart Glasses", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AiSee Smart Glasses", 640, 480)

    print("=" * 60)
    print("  ✅ System ready!")
    print(f"  Button 1 (GPIO {BUTTONS['Button 1']}) — Toggle OD ↔ OCR")
    print(f"  Button 2 (GPIO {BUTTONS['Button 2']}) — Manual scan (OD) / Capture (OCR)")
    print("  ESC key            — Quit")
    print("  Bounding boxes:    Blue=[P] Pretrained   Orange=[C] Custom   Tracked=[T]")
    print("  Distance shown in label when calibrated, e.g. '[C] chair (0.72) 1.4m'")
    print("  Settled tracks go silent after 3 announcements until distance changes")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # STEP 6 — GPIO setup
    # ------------------------------------------------------------------

    latest_frame = [None]

    def on_mode_toggle(channel=None):
        """Button 1 — Toggle between OD and OCR."""
        if current_mode[0] == 'od':
            current_mode[0] = 'ocr'
            # Clear speech queue when switching modes
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
        """Button 2 — Manual scan in OD mode, OCR capture in OCR mode."""
        if current_mode[0] == 'od':
            # OD mode: trigger manual scan
            print("\n🔘 Manual scan triggered (Button 2)")
            announce("Manual scan")
            # Get current detections from the last inference
            if last_detections:
                trigger_manual_scan(last_detections, 640)  # frame width is 640
            else:
                # Run inference immediately if no recent detections
                temp_dets = run_dual_detection(latest_frame[0])
                trigger_manual_scan(temp_dets, 640)
            return

        # OCR mode: capture frame and send to server
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
    # STEP 7 — Main frame loop
    # ------------------------------------------------------------------
    frame_count     = 0
    last_detections = []   # Reused between skipped frames — prevents box flicker

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
                    # Run both models with tracking
                    last_detections = run_dual_detection(frame)

                    # Expire stale tracks, then decide what to announce
                    expire_lost_tracks()
                    process_detections(last_detections, frame.shape[1])
            else:
                # In OCR mode — clear detections so boxes don't linger
                last_detections = []

            # Draw boxes every frame for smooth visuals
            frame = draw_detections(frame, last_detections)

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
        stop_speech_worker()
        GPIO.cleanup()
        cam.release()
        cv2.destroyAllWindows()
        print("✓ Done. Goodbye!\n")


if __name__ == "__main__":
    main()