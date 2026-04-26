import pyttsx3
from threading import Thread
from queue import Queue
import cv2
import time
from ultralytics import YOLO

# =====================================================================
#  PATH CONFIGURATION
# =====================================================================
PRETRAINED_MODEL_PATH = "yolov8n.pt"          # Downloads automatically if not present
CUSTOM_MODEL_PATH     = "best(7)_ncnn_model"  # Your custom trained model path

CAMERA_INDEX = 0

# =====================================================================
#  CLASS FILTER — YOLOv8 pretrained model ALLOWLIST
#  Only these classes will be kept from the pretrained model.
#  Everything else is ignored automatically.
# =====================================================================
ALLOWED_PRETRAINED_CLASSES = {
    'backpack',
    'handbag',
    'bottle',
    'book',
    'laptop',
    'keyboard',
}

# =====================================================================
#  CUSTOM MODEL CLASSES
#  These are handled by the custom model, so if the pretrained model
#  also detects them, they are dropped to avoid duplicates.
# =====================================================================
CUSTOM_MODEL_CLASSES = {
    'people',
    'whiteboard',
    'chair', # empty chair
    'desk',
    'fullchair',
    'stairs',
    'door',
}

# =====================================================================
#  CONFIDENCE THRESHOLDS
# =====================================================================
PRETRAINED_CONF_THRESHOLD = 0.50
CUSTOM_CONF_THRESHOLD     = 0.45

# =====================================================================
#  SPEECH SETTINGS
# =====================================================================
SPEECH_COOLDOWN = 3     # seconds between repeat announcements for same object
MAX_QUEUE_SIZE  = 2     # max queued speech items before dropping

# =====================================================================
#  PROCESSING SETTINGS
# =====================================================================
FRAME_SKIP = 5          # run inference every Nth frame

# =====================================================================
#  BOUNDING BOX COLORS
#  Pretrained = Blue, Custom = Orange
# =====================================================================
COLOR_PRETRAINED = (255, 100,   0)
COLOR_CUSTOM     = (  0, 165, 255)

# ---------------------------------------------------------------------

last_spoken  = {}
speech_queue = Queue()


def speak_worker():
    """Background TTS thread. Creates a fresh engine per message to avoid conflicts."""
    while True:
        try:
            if not speech_queue.empty():
                item = speech_queue.get(timeout=1)
                if item is None:          # shutdown signal
                    break

                label, position = item
                current_time = time.time()

                # Cooldown check
                if label in last_spoken and current_time - last_spoken[label] < SPEECH_COOLDOWN:
                    print(f"[TTS] Skipping '{label}' — cooldown active")
                    continue

                # Build announcement
                if position == "center":
                    message = f"{label} ahead"
                else:
                    message = f"{label} on your {position}"

                print(f"[TTS] Speaking: {message}")

                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 235)
                    engine.setProperty('volume', 1.0)
                    engine.say(message)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                except Exception as e:
                    print(f"[TTS] Engine error: {e}")

                last_spoken[label] = current_time

                # Drain backlog after speaking
                while not speech_queue.empty():
                    try:
                        speech_queue.get_nowait()
                        print("[TTS] Cleared backlogged item")
                    except Exception:
                        break
            else:
                time.sleep(0.1)

        except Exception as e:
            print(f"[TTS] Thread error: {e}")
            time.sleep(0.1)


def get_position(frame_width, x_center):
    """Divide frame into left / center / right thirds."""
    if x_center < frame_width // 3:
        return "left"
    elif x_center < 2 * (frame_width // 3):
        return "center"
    else:
        return "right"


def run_pretrained_model(model, frame):
    """
    Run YOLOv8 pretrained model and return filtered detections.
    Returns list of dicts: {label, confidence, x1, y1, x2, y2}
    """
    detections = []
    try:
        results = model.predict(frame, verbose=False)
        for box in results[0].boxes:
            label      = results[0].names[int(box.cls[0].item())]
            confidence = box.conf[0].item()

            # Apply allowlist — only keep classes we explicitly want
            if label.lower() not in ALLOWED_PRETRAINED_CLASSES:
                continue
            # Drop anything the custom model already handles
            if label.lower() in CUSTOM_MODEL_CLASSES:
                continue
            if confidence < PRETRAINED_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "label":      label,
                "confidence": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "source":     "pretrained",
            })
    except Exception as e:
        print(f"[Pretrained] Inference error: {e}")
    return detections


def run_custom_model(model, frame):
    """
    Run custom trained YOLO model and return filtered detections.
    Returns list of dicts: {label, confidence, x1, y1, x2, y2}
    """
    detections = []
    try:
        results = model.predict(frame, verbose=False)
        for box in results[0].boxes:
            label      = results[0].names[int(box.cls[0].item())]
            confidence = box.conf[0].item()

            if confidence < CUSTOM_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "label":      label,
                "confidence": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "source":     "custom",
            })
    except Exception as e:
        print(f"[Custom] Inference error: {e}")
    return detections


def merge_detections(pretrained_detections, custom_detections):
    """
    Combine detections from both models.
    Custom model detections take priority — if the same label appears
    in both, only the custom result is kept to avoid duplicate announcements.
    """
    custom_labels = {d["label"].lower() for d in custom_detections}

    merged = list(custom_detections)  # always keep all custom detections

    for det in pretrained_detections:
        if det["label"].lower() not in custom_labels:
            merged.append(det)        # only add pretrained if not already covered

    return merged


def draw_detections(frame, detections):
    """Draw bounding boxes and labels. Blue = pretrained, Orange = custom."""
    for det in detections:
        color = COLOR_CUSTOM if det["source"] == "custom" else COLOR_PRETRAINED
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label      = det["label"]
        confidence = det["confidence"]
        source_tag = "[C]" if det["source"] == "custom" else "[P]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{source_tag} {label} ({confidence:.2f})",
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    return frame


def announce_detections(frame_width, detections):
    """Push detected objects into the TTS speech queue."""
    for det in detections:
        if speech_queue.qsize() < MAX_QUEUE_SIZE:
            x_center = (det["x1"] + det["x2"]) // 2
            position = get_position(frame_width, x_center)
            speech_queue.put((det["label"], position))
            print(f"[Queue] {det['label']} ({det['source']}) → {position}")
        else:
            print("[Queue] Full — dropping announcement")


def main():
    # ── Startup announcement ──────────────────────────────────────────
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 235)
        engine.say("System activated")
        engine.runAndWait()
        engine.stop()
        del engine
    except Exception as e:
        print(f"[Startup] TTS error: {e}")

    # ── Start TTS background thread ───────────────────────────────────
    tts_thread = Thread(target=speak_worker, daemon=True)
    tts_thread.start()

    # ── Load models ───────────────────────────────────────────────────
    print("[Init] Loading pretrained YOLOv8 model...")
    try:
        pretrained_model = YOLO(PRETRAINED_MODEL_PATH)
    except Exception as e:
        print(f"[Init] Failed to load pretrained model: {e}")
        return

    print("[Init] Loading custom model...")
    try:
        custom_model = YOLO(CUSTOM_MODEL_PATH)
    except Exception as e:
        print(f"[Init] Failed to load custom model: {e}")
        return

    print("[Init] Both models loaded successfully.")

    # ── Open camera ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[Init] Error: Could not open camera.")
        return

    # ── Video output ──────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        'dual_model_output.avi', fourcc, 20.0,
        (int(cap.get(3)), int(cap.get(4))),
    )

    frame_count = 0
    paused      = False
    last_detections = []   # reuse between skipped frames so boxes don't flicker

    print("[Main] Running — press 'q' to quit, 'p' to pause.")

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[Main] Failed to read frame — exiting.")
                    break

                frame_count += 1

                if frame_count % FRAME_SKIP == 0:
                    # ── Run both models ───────────────────────────────
                    pretrained_dets = run_pretrained_model(pretrained_model, frame)
                    custom_dets     = run_custom_model(custom_model, frame)

                    # ── Merge ─────────────────────────────────────────
                    last_detections = merge_detections(pretrained_dets, custom_dets)

                    # ── Announce ──────────────────────────────────────
                    announce_detections(frame.shape[1], last_detections)

                # ── Draw (every frame for smooth visuals) ────────────
                frame = draw_detections(frame, last_detections)

                cv2.imshow("Audio World — Dual Model", frame)
                out.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Main] Quit signal received.")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"[Main] {'Paused' if paused else 'Resumed'}")

    except KeyboardInterrupt:
        print("[Main] Interrupted by user.")
    finally:
        print("[Main] Shutting down...")
        speech_queue.put(None)
        time.sleep(1)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("[Main] Cleanup complete.")


if __name__ == "__main__":
    main()