# =============================================================================
# Configuration — All constants and tuning values in one place
# =============================================================================
#
# This module contains all configuration constants used across the system.
# Modify these values to tune the behavior of the smart glasses.
#
# =============================================================================

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================

SERVER_URL  = "https://aisee.onrender.com"  # AiSee backend URL
DEVICE_CODE = "AIS-4829"                    # Unique per Pi — matches DB + printed card

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================

CAMERA_INDEX         = 0
CAMERA_WARMUP_FRAMES = 10

# --- Image compression (reduces upload time ~70%) ---
MAX_IMAGE_WIDTH  = 1600
MAX_IMAGE_HEIGHT = 1200
JPEG_QUALITY     = 75

# =============================================================================
# MODEL PATHS
# =============================================================================

PRETRAINED_MODEL_PATH = "yolov8n_ncnn_model"   # Downloads automatically if absent
CUSTOM_MODEL_PATH     = "best(7)_ncnn_model"   # Your custom trained model

# =============================================================================
# CLASS FILTERS
# =============================================================================

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

# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================

PRETRAINED_CONF_THRESHOLD = 0.50
CUSTOM_CONF_THRESHOLD     = 0.45

# =============================================================================
# BOUNDING BOX COLORS (BGR)
# =============================================================================

COLOR_PRETRAINED = (255, 100,   0)   # Blue
COLOR_CUSTOM     = (  0, 165, 255)   # Orange

# =============================================================================
# PROCESSING SETTINGS
# =============================================================================

OD_FRAME_SKIP   = 5    # Run inference every Nth frame
SPEECH_COOLDOWN = 3    # Seconds before same label is announced again
MAX_QUEUE_SIZE  = 2    # Max queued speech items before dropping

# =============================================================================
# GPIO BUTTON PINS (BCM numbering)
# =============================================================================

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
SETTLE_AFTER_SECONDS  = 5.0    # Silence track after this many seconds in frame
TRACK_EXPIRY_SECONDS  = 10.0   # Forget track after this many seconds out of frame

# A track is re-announced when its distance changes by this much (metres).
# e.g. someone walking toward you crosses from 3m → 1.5m → triggers a new alert.
DISTANCE_REANNOUNCE_DELTA = 1.2

# Classes treated as "hazards" — announced with urgency language and a shorter
# cooldown when they are within DISTANCE_VERY_CLOSE.
HAZARD_CLASSES = {'people', 'stairs', 'door'}

# Shortened cooldown (seconds) for hazards that are very close.
HAZARD_CLOSE_COOLDOWN = 2.5

# =============================================================================
# SETTLE TRACK CONFIGURATION
# =============================================================================

# Number of times a track must be announced before it becomes "settled".
# Once settled, the track goes silent until a wakeup event occurs.
# Set to 0 to disable settling (always announce).
SETTLE_ANNOUNCE_COUNT = 3

# Minimum distance change (metres) to wake up a settled track.
# If a settled track's distance changes by more than this, it becomes active again.
SETTLE_WAKEUP_DELTA = 0.8