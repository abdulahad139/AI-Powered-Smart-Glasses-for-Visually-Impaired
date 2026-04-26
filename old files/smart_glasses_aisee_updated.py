# =============================================================================
# Smart Glasses — AiSee Integrated
# Dual Model Object Detection (Pretrained YOLOv8 + Custom YOLO) + Urdu OCR
# + Distance Estimation (monocular, per-class width ratio)
# =============================================================================
#
# OBJECT DETECTION:
#   Two YOLO models run in parallel every Nth frame:
#     1. Pretrained YOLOv8 (yolov8n_ncnn_model) — detects general objects
#        filtered to an explicit ALLOWLIST (backpack, bottle, laptop, etc.)
#     2. Custom model (best(7)_ncnn_model)       — detects domain-specific
#        objects (people, chairs, stairs, doors, etc.)
#   Results are merged — custom model takes priority to avoid duplicates.
#   Bounding boxes: Blue = pretrained, Orange = custom.
#
# DISTANCE ESTIMATION:
#   Uses per-class width_ratio (calibrated via distance_calibration.py).
#   Formula: distance_m = FOCAL_LENGTH / (pixel_width * width_ratio)
#   Only classes present in CLASS_WIDTH_RATIOS get a distance estimate.
#   Distance is spoken aloud and shown in the bounding box label.
#
# OCR MODE:
#   Captures frame on Button 2 press, sends to AiSee server.
#   Server does Google OCR + gTTS, streams MP3 audio back to Pi.
#
# GPIO BUTTONS:
#   Button 1 (GPIO 16) — Toggle mode (OD ↔ OCR) with audio feedback
#   Button 2 (GPIO 26) — Capture image for OCR (ignored in OD mode)
#   ESC key            — Quit (keyboard fallback for testing)
#
# CALIBRATION:
#   Run distance_calibration.py to measure width_ratio values for each class.
#   Then paste the results into CLASS_WIDTH_RATIOS below.
#
# SETUP REQUIRED (one time):
#   Visit aisee.com/setup → Sign in with Google → enter device code
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
# CONFIGURATION
# =============================================================================

SERVER_URL  = "https://aisee.onrender.com"  # AiSee backend URL
DEVICE_CODE = "AIS-4829"                    # Unique per Pi — matches DB + printed card

# --- Camera ---
CAMERA_INDEX         = 0
CAMERA_WARMUP_FRAMES = 10

# --- Image compression (reduces upload time ~70%) ---
MAX_IMAGE_WIDTH  = 1600
MAX_IMAGE_HEIGHT = 1200
JPEG_QUALITY     = 75

# --- Model paths ---
PRETRAINED_MODEL_PATH = "yolov8n_ncnn_model"   # Downloads automatically if absent
CUSTOM_MODEL_PATH     = "best(7)_ncnn_model"   # Your custom trained model

# --- Pretrained model ALLOWLIST ---
# Only these classes are announced from the pretrained model.
# Everything else is silently ignored.
ALLOWED_PRETRAINED_CLASSES = {
    'backpack',
    'handbag',
    'bottle',
    'book',
    'laptop',
    'keyboard',
}

# --- Custom model classes ---
# These are owned by the custom model. If the pretrained model also detects
# any of these, the pretrained result is dropped to avoid duplicate speech.
CUSTOM_MODEL_CLASSES = {
    'people',
    'whiteboard',
    'chair',
    'desk',
    'fullchair',
    'stairs',
    'door',
}

# --- Confidence thresholds ---
PRETRAINED_CONF_THRESHOLD = 0.50
CUSTOM_CONF_THRESHOLD     = 0.45

# --- Bounding box colors (BGR) ---
COLOR_PRETRAINED = (255, 100,   0)   # Blue
COLOR_CUSTOM     = (  0, 165, 255)   # Orange

# --- Processing ---
OD_FRAME_SKIP   = 5    # Run inference every Nth frame
SPEECH_COOLDOWN = 3    # Seconds before same label is announced again
MAX_QUEUE_SIZE  = 2    # Max queued speech items before dropping

# --- GPIO Button Pins (BCM numbering) ---
# Button 1 — Mode toggle  (GPIO 16, Physical Pin 36, GND Pin 34)
# Button 2 — Capture OCR  (GPIO 26, Physical Pin 37, GND Pin 39)
BUTTONS = {
    "Button 1": 16,
    "Button 2": 26,
}
BUTTON_BOUNCETIME = 300  # ms debounce

# =============================================================================
# DISTANCE ESTIMATION CONFIGURATION
# =============================================================================

# Pre-computed focal length for your camera (pixels).
# Measured at frame_width=640, horizontal FOV ~70°.
# To recalculate: FOCAL_LENGTH = (frame_width / 2) / tan(FOV_deg / 2 * pi/180)
FOCAL_LENGTH = 457.007

# Per-class width ratios — produced by distance_calibration.py.
# Formula used at runtime: distance_m = FOCAL_LENGTH / (pixel_width * width_ratio)
#
# HOW TO POPULATE:
#   1. Run distance_calibration.py with your camera.
#   2. Copy the printed "width_ratio" values here.
#
# Classes not listed here will NOT receive a distance estimate
# (they are still detected and announced normally).
CLASS_WIDTH_RATIOS = {
    # --- Pretrained model classes ---
    "backpack":   {"width_ratio": 2.59},   
    "bottle":     {"width_ratio": 10.62},
    # "book":       {"width_ratio": 0.15},
    "laptop":     {"width_ratio": 2.84},
    # "keyboard":   {"width_ratio": 0.35},
    # "handbag":    {"width_ratio": 0.28},
    "cellphone":  {"width_ratio": 8.69},

    # --- Custom model classes ---
    "chair":    {"width_ratio": 2.08},
    "door":     {"width_ratio": 0.75},
    "people":   {"width_ratio": 1.90},
    "fullchair": {"width_ratio": 1.45},
    "whiteboard": {"width_ratio": 0.57},
    "stairs": {"width_ratio": 0.65},
    "desk": {"width_ratio": 1.31},
}

# Distance thresholds for proximity warnings (metres).
# Spoken announcement changes when the object is within these ranges.
DISTANCE_VERY_CLOSE = 1.0   # "right in front of you"
DISTANCE_CLOSE      = 2.5   # "nearby"
DISTANCE_FAR        = 5.0   # "ahead" / "on your left/right"
# Beyond DISTANCE_FAR the object is announced normally with the numeric distance.


# =============================================================================
# ANNOUNCEMENT TUNING
# =============================================================================

# How many seconds before the SAME track can be announced again.
# Prevents constant repetition of a stationary object.
ANNOUNCE_COOLDOWN = 6.0

# If a track disappears for this many seconds, it is forgotten.
# When it re-enters the frame it will be announced fresh.
TRACK_EXPIRY_SECONDS = 4.0

# A track is re-announced when its distance changes by this much (metres).
# e.g. someone walking toward you crosses from 3m → 1.5m → triggers a new alert.
DISTANCE_REANNOUNCE_DELTA = 1.2

# Classes treated as "hazards" — announced with urgency language and a shorter
# cooldown when they are within DISTANCE_VERY_CLOSE.
HAZARD_CLASSES = {'people', 'stairs', 'door'}

# Shortened cooldown (seconds) for hazards that are very close.
HAZARD_CLOSE_COOLDOWN = 2.5

# =============================================================================
# SHARED STATE
# =============================================================================

current_mode   = ['od']           # 'od' or 'ocr'
ocr_processing = threading.Event()
speech_queue   = Queue()

# track_id → TrackState dict
# Keys: label, last_announced, last_seen, last_distance, last_position
track_states: dict = {}


# =============================================================================
# DISTANCE CALCULATION HELPERS
# =============================================================================

def estimate_distance(label: str, pixel_width: float) -> float | None:
    """
    Return estimated distance in metres for a detected object.

    Uses the monocular formula:
        distance = FOCAL_LENGTH / (pixel_width * width_ratio)

    Returns None if the class is not calibrated or pixel_width is invalid.
    """
    if pixel_width <= 0:
        return None
    entry = CLASS_WIDTH_RATIOS.get(label.lower())
    if entry is None:
        return None
    ratio = entry.get("width_ratio", 0)
    if ratio <= 0:
        return None
    return FOCAL_LENGTH / (pixel_width * ratio)


def format_distance_spoken(distance_m: float | None) -> str:
    """
    Return a natural spoken distance phrase.

    Examples:
        0.6  → "less than one metre away"
        1.2  → "about one metre away"
        2.4  → "about 2 metres away"
        6.1  → "about 6 metres away"
    """
    if distance_m is None:
        return ""
    if distance_m < 1.0:
        return "less than one metre away"
    rounded = round(distance_m)
    if rounded == 1:
        return "about one metre away"
    return f"about {rounded} metres away"


def format_distance_label(distance_m: float | None) -> str:
    """Short on-screen label, e.g. '1.2m'. Used in bounding box text only."""
    if distance_m is None:
        return ""
    return f"{distance_m:.1f}m"


# =============================================================================
# ANNOUNCEMENT PHRASE BUILDER
# =============================================================================

def build_announcement(label: str, position: str,
                        distance_m: float | None,
                        is_new: bool,
                        approaching: bool) -> str | None:
    """
    Build a natural, descriptive spoken phrase for a detected object.

    Returns None if the object should be silently skipped this cycle.

    Logic priority:
      1. Hazard + very close  → urgent warning regardless of cooldown
      2. Approaching          → "X is getting closer, now Y metres away"
      3. New track            → full introduction with position + distance
      4. Periodic reminder    → brief re-mention with current distance
    """
    is_hazard  = label.lower() in HAZARD_CLASSES
    dist_text  = format_distance_spoken(distance_m)

    # Position phrase
    if position == "center":
        pos_phrase = "directly ahead"
    elif position == "left":
        pos_phrase = "to your left"
    else:
        pos_phrase = "to your right"

    # --- Urgent hazard very close ---
    if is_hazard and distance_m is not None and distance_m <= DISTANCE_VERY_CLOSE:
        if label.lower() == "stairs":
            return f"Caution — stairs {pos_phrase}, {dist_text}"
        if label.lower() == "door":
            return f"Door {pos_phrase}, {dist_text}"
        return f"Caution — person {pos_phrase}, {dist_text}"

    # --- Object is approaching ---
    if approaching and distance_m is not None:
        return f"{label} approaching {pos_phrase}, {dist_text}"

    # --- First time seeing this track ---
    if is_new:
        if dist_text:
            return f"{label} {pos_phrase}, {dist_text}"
        else:
            return f"{label} {pos_phrase}"

    # --- Periodic reminder (distance changed significantly or cooldown elapsed) ---
    if dist_text:
        return f"{label} still {pos_phrase}, {dist_text}"
    return f"{label} {pos_phrase}"


# =============================================================================
# SERVER / OCR FUNCTIONS
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


# =============================================================================
# AUDIO HELPERS
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
# SPEECH WORKER (for OD announcements + mode changes)
# =============================================================================

def speak_worker():
    """
    Single TTS worker thread — handles ALL speech for OD + mode announcements.

    Queue item formats:
      OD detection  : ('od',       message_str, None, None)
      Announcement  : ('announce', message_str, None, None)
      Shutdown      : None

    All cooldown / filtering logic happens in process_detections() before
    items reach this queue, so the worker just speaks whatever arrives.
    """
    while True:
        try:
            if not speech_queue.empty():
                item = speech_queue.get(timeout=1)
                if item is None:
                    break

                msg_type = item[0]
                message  = item[1]

                if msg_type not in ('od', 'announce') or not message:
                    continue

                print(f"🔈 {'OD' if msg_type == 'od' else 'SYS'}: {message}")

                # Clear any stale OD items that piled up while we were speaking
                if msg_type == 'od':
                    while not speech_queue.empty():
                        try:
                            peeked = speech_queue.queue[0]
                            if peeked is not None and peeked[0] == 'od':
                                speech_queue.get_nowait()
                                print("[TTS] Dropped backlogged OD item")
                            else:
                                break
                        except Exception:
                            break

                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 220)
                    engine.setProperty('volume', 1.0)
                    engine.say(message)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                except Exception as e:
                    print(f"⚠️  TTS error: {e}")

            else:
                time.sleep(0.05)

        except Exception as e:
            print(f"⚠️  Speech worker error: {e}")
            time.sleep(0.1)


# =============================================================================
# TRACK STATE MANAGEMENT + SMART ANNOUNCEMENT LOGIC
# =============================================================================

def expire_lost_tracks():
    """
    Remove tracks that haven't been seen recently.
    Called every inference cycle so re-entering objects are treated as new.
    """
    now     = time.time()
    expired = [tid for tid, state in track_states.items()
               if now - state['last_seen'] > TRACK_EXPIRY_SECONDS]
    for tid in expired:
        print(f"[Track] Expired: {tid} ({track_states[tid]['label']})")
        del track_states[tid]


def process_detections(detections: list, frame_width: int):
    """
    Core announcement logic. For each detection:

      1. Update or create its TrackState.
      2. Decide whether to speak using these rules:
           - Hazard very close       → always speak (short cooldown)
           - Approaching             → speak when delta > DISTANCE_REANNOUNCE_DELTA
           - New track               → always speak
           - Cooldown elapsed        → speak periodic reminder
           - Otherwise               → silent

    Queues a single pre-built string per announcement into speech_queue.
    """
    now = time.time()

    for det in detections:
        tid        = det.get("track_id")
        label      = det["label"]
        distance_m = det.get("distance_m")
        x_center   = (det["x1"] + det["x2"]) / 2
        position   = get_position(frame_width, x_center)
        is_hazard  = label.lower() in HAZARD_CLASSES

        # Determine per-object cooldown
        cooldown = (HAZARD_CLOSE_COOLDOWN
                    if is_hazard and distance_m is not None and distance_m <= DISTANCE_VERY_CLOSE
                    else ANNOUNCE_COOLDOWN)

        if tid is None:
            # No track ID — fall back to label-keyed cooldown only
            state = track_states.get(f"noid_{label}")
            if state and now - state['last_announced'] < cooldown:
                continue
            is_new      = state is None
            approaching = False
            track_states[f"noid_{label}"] = {
                'label':          label,
                'last_announced': now,
                'last_seen':      now,
                'last_distance':  distance_m,
                'last_position':  position,
            }
        else:
            state  = track_states.get(tid)
            is_new = state is None

            if state:
                state['last_seen'] = now

                # Detect approach: distance decreased by more than threshold
                prev_dist   = state.get('last_distance')
                approaching = (
                    distance_m is not None and
                    prev_dist  is not None and
                    prev_dist - distance_m >= DISTANCE_REANNOUNCE_DELTA
                )

                # Check if silent this cycle
                time_ok  = now - state['last_announced'] >= cooldown
                urgent   = is_hazard and distance_m is not None and distance_m <= DISTANCE_VERY_CLOSE

                if not (time_ok or approaching or urgent):
                    state['last_distance'] = distance_m
                    state['last_position'] = position
                    continue  # nothing new to say

                state['last_announced'] = now
                state['last_distance']  = distance_m
                state['last_position']  = position

            else:
                # Brand new track
                approaching = False
                track_states[tid] = {
                    'label':          label,
                    'last_announced': now,
                    'last_seen':      now,
                    'last_distance':  distance_m,
                    'last_position':  position,
                }

        # Build and queue the phrase
        phrase = build_announcement(label, position, distance_m, is_new, approaching)
        if phrase and speech_queue.qsize() < MAX_QUEUE_SIZE:
            speech_queue.put(('od', phrase, None, None))
            print(f"[Announce] {phrase}")


# =============================================================================
# DUAL MODEL — OBJECT DETECTION
# =============================================================================

def run_pretrained_model(model, frame):
    """
    Run pretrained YOLOv8 model with tracking and return filtered detections.
    Only returns classes in ALLOWED_PRETRAINED_CLASSES.
    Drops any class already owned by the custom model.
    Returns list of dicts with track_id and distance_m.
    """
    detections = []
    try:
        results = model.track(frame, persist=True, verbose=False)
        for box in results[0].boxes:
            label      = results[0].names[int(box.cls[0].item())]
            confidence = box.conf[0].item()
            track_id   = int(box.id[0].item()) if box.id is not None else None

            if label.lower() not in ALLOWED_PRETRAINED_CLASSES:
                continue
            if label.lower() in CUSTOM_MODEL_CLASSES:
                continue
            if confidence < PRETRAINED_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            pixel_width     = x2 - x1
            distance_m      = estimate_distance(label, pixel_width)

            detections.append({
                "label":      label,
                "confidence": confidence,
                "track_id":   track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "source":     "pretrained",
                "distance_m": distance_m,
            })
    except Exception as e:
        print(f"[Pretrained] Inference error: {e}")
    return detections


def run_custom_model(model, frame):
    """
    Run custom trained YOLO model with tracking and return filtered detections.
    Returns list of dicts with track_id and distance_m.
    """
    detections = []
    try:
        results = model.track(frame, persist=True, verbose=False)
        for box in results[0].boxes:
            label      = results[0].names[int(box.cls[0].item())]
            confidence = box.conf[0].item()
            track_id   = int(box.id[0].item()) if box.id is not None else None

            if confidence < CUSTOM_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            pixel_width     = x2 - x1
            distance_m      = estimate_distance(label, pixel_width)

            detections.append({
                "label":      label,
                "confidence": confidence,
                "track_id":   track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "source":     "custom",
                "distance_m": distance_m,
            })
    except Exception as e:
        print(f"[Custom] Inference error: {e}")
    return detections


def merge_detections(pretrained_detections, custom_detections):
    """
    Combine detections from both models.
    Custom model detections always take priority — if the same label appears
    in both, only the custom result is kept to avoid duplicate announcements.
    Adjust track_ids to avoid conflicts: custom as 'c_{id}', pretrained as 'p_{id}'.
    """
    custom_labels = {d["label"].lower() for d in custom_detections}
    merged = []
    for det in custom_detections:
        if det["track_id"] is not None:
            det["track_id"] = f"c_{det['track_id']}"
        merged.append(det)
    for det in pretrained_detections:
        if det["label"].lower() not in custom_labels:
            if det["track_id"] is not None:
                det["track_id"] = f"p_{det['track_id']}"
            merged.append(det)
    return merged


def blur_person(image, det):
    """Blur the upper face region of a detected person or fullchair box."""
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    top = image[y1:y1 + int(0.08 * (y2 - y1)), x1:x2]
    image[y1:y1 + int(0.08 * (y2 - y1)), x1:x2] = cv2.GaussianBlur(top, (15, 15), 0)
    return image


def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on frame.
    Blue  [P] = pretrained model result
    Orange[C] = custom model result
    People / fullchair boxes are also face-blurred.
    Distance is shown in the label when available, e.g. "bottle (0.92) 1.2m".
    """
    for det in detections:
        label      = det["label"]
        source     = det["source"]
        color      = COLOR_CUSTOM if source == "custom" else COLOR_PRETRAINED
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        tag        = "[T]" if det.get('source') == 'tracked' else ("[C]" if source == "custom" else "[P]")
        dist_str   = format_distance_label(det.get("distance_m"))
        dist_label = f" {dist_str}" if dist_str else ""

        if label.lower() in ("people", "fullchair"):
            try:
                frame = blur_person(frame, det)
            except Exception as e:
                print(f"⚠️  Blur error: {e}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{tag} {label} ({det['confidence']:.2f}){dist_label}",
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    return frame


# =============================================================================
# POSITION HELPER
# =============================================================================

def get_position(frame_width, x_center):
    """Divide frame into left / center / right thirds."""
    if x_center < frame_width // 3:
        return "left"
    elif x_center < 2 * (frame_width // 3):
        return "center"
    else:
        return "right"




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
    # STEP 2 — Load both YOLO models
    # ------------------------------------------------------------------
    print(f"🤖 Loading pretrained YOLOv8 model from {PRETRAINED_MODEL_PATH}...")
    try:
        pretrained_model = YOLO(PRETRAINED_MODEL_PATH)
        print("✓ Pretrained model loaded!")
    except Exception as e:
        print(f"❌ Failed to load pretrained model: {e}")
        sys.exit(1)

    print(f"🤖 Loading custom model from {CUSTOM_MODEL_PATH}...")
    try:
        custom_model = YOLO(CUSTOM_MODEL_PATH)
        print("✓ Custom model loaded!\n")
    except Exception as e:
        print(f"❌ Failed to load custom model: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 3 — Open camera
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
    # STEP 4 — Start TTS worker thread
    # ------------------------------------------------------------------
    tts_thread = threading.Thread(target=speak_worker, daemon=True)
    tts_thread.start()

    # ------------------------------------------------------------------
    # STEP 5 — Startup announcement
    # ------------------------------------------------------------------
    _speak_blocking("System activated. Dual model object detection with distance estimation running.")

    # ------------------------------------------------------------------
    # STEP 6 — Display window
    # ------------------------------------------------------------------
    cv2.namedWindow("AiSee Smart Glasses", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AiSee Smart Glasses", 640, 480)

    print("=" * 60)
    print("  ✅ System ready!")
    print(f"  Button 1 (GPIO {BUTTONS['Button 1']}) — Toggle OD ↔ OCR")
    print(f"  Button 2 (GPIO {BUTTONS['Button 2']}) — Capture (OCR mode only)")
    print("  ESC key            — Quit")
    print("  Bounding boxes:    Blue=[P] Pretrained   Orange=[C] Custom   Tracked=[T]")
    print("  Distance shown in label when calibrated, e.g. '[C] chair (0.72) 1.4m'")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # STEP 7 — GPIO setup
    # ------------------------------------------------------------------

    latest_frame = [None]

    def announce(message):
        speech_queue.put(('announce', message, None, None))

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
                    pretrained_dets = run_pretrained_model(pretrained_model, frame)
                    custom_dets     = run_custom_model(custom_model, frame)

                    # Merge — custom takes priority
                    last_detections = merge_detections(pretrained_dets, custom_dets)

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
        speech_queue.put(None)
        time.sleep(0.5)
        GPIO.cleanup()
        cam.release()
        cv2.destroyAllWindows()
        print("✓ Done. Goodbye!\n")


if __name__ == "__main__":
    main()