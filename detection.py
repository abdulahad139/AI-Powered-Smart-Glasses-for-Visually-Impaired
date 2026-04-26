# =============================================================================
# Detection — Dual Model YOLO object detection, merge, draw, blur
# =============================================================================
#
# Handles both YOLO models (pretrained and custom), detection merging,
# bounding box drawing, and person face blurring.
#
# =============================================================================

import cv2

from config import (
    PRETRAINED_MODEL_PATH,
    CUSTOM_MODEL_PATH,
    ALLOWED_PRETRAINED_CLASSES,
    CUSTOM_MODEL_CLASSES,
    PRETRAINED_CONF_THRESHOLD,
    CUSTOM_CONF_THRESHOLD,
    COLOR_PRETRAINED,
    COLOR_CUSTOM,
)
from distance import estimate_distance, format_distance_label


# Lazy-loaded models
_pretrained_model = None
_custom_model = None


def get_pretrained_model():
    """Lazy-load the pretrained YOLO model."""
    global _pretrained_model
    if _pretrained_model is None:
        from ultralytics import YOLO
        _pretrained_model = YOLO(PRETRAINED_MODEL_PATH)
    return _pretrained_model


def get_custom_model():
    """Lazy-load the custom YOLO model."""
    global _custom_model
    if _custom_model is None:
        from ultralytics import YOLO
        _custom_model = YOLO(CUSTOM_MODEL_PATH)
    return _custom_model


def run_pretrained_model(frame):
    """
    Run pretrained YOLOv8 model with tracking and return filtered detections.
    Only returns classes in ALLOWED_PRETRAINED_CLASSES.
    Drops any class already owned by the custom model.
    Returns list of dicts with track_id and distance_m.
    """
    detections = []
    model = get_pretrained_model()
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


def run_custom_model(frame):
    """
    Run custom trained YOLO model with tracking and return filtered detections.
    Returns list of dicts with track_id and distance_m.
    """
    detections = []
    model = get_custom_model()
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


def run_dual_detection(frame):
    """
    Run both models and merge detections.
    Convenience function for the main loop.
    """
    pretrained_dets = run_pretrained_model(frame)
    custom_dets     = run_custom_model(frame)
    return merge_detections(pretrained_dets, custom_dets)