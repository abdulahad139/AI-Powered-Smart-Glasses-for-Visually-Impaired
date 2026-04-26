# =============================================================================
# Tracking — Track state management and smart announcement logic
# =============================================================================
#
# Manages object tracking states, expiry, and builds announcement phrases.
#
# =============================================================================

import time

from config import (
    HAZARD_CLASSES,
    DISTANCE_VERY_CLOSE,
    ANNOUNCE_COOLDOWN,
    HAZARD_CLOSE_COOLDOWN,  
    TRACK_EXPIRY_SECONDS,
    DISTANCE_REANNOUNCE_DELTA,
    SETTLE_ANNOUNCE_COUNT,
    SETTLE_WAKEUP_DELTA,
)
from distance import format_distance_spoken, is_very_close
from audio import queue_od_message


# Track_id → TrackState dict
# Keys: label, last_announced, last_seen, last_distance, last_position, settled, announce_count
track_states: dict = {}


def get_position(frame_width, x_center):
    """Divide frame into left / center / right thirds."""
    if x_center < frame_width // 3:
        return "left"
    elif x_center < 2 * (frame_width // 3):
        return "center"
    else:
        return "right"


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


def process_detections(detections: list, frame_width: int, force_announce: bool = False):
    """
    Core announcement logic. For each detection:

      1. Update or create its TrackState.
      2. Decide whether to speak using these rules:
           - force_announce=True    → always speak (manual scan)
           - Hazard very close       → always speak (short cooldown)
           - Approaching             → speak when delta > DISTANCE_REANNOUNCE_DELTA
           - New track               → always speak
           - Settled track           → silent unless wakeup event
           - Cooldown elapsed        → speak periodic reminder
           - Otherwise               → silent

    Queues a single pre-built string per announcement into speech_queue.
    """
    now = time.time()

    for det in detections:
        tid         = det.get("track_id")
        label       = det["label"]
        distance_m  = det.get("distance_m")
        x_center    = (det["x1"] + det["x2"]) / 2
        position    = get_position(frame_width, x_center)
        is_hazard   = label.lower() in HAZARD_CLASSES

        # Determine per-object cooldown
        cooldown = (HAZARD_CLOSE_COOLDOWN
                    if is_hazard and distance_m is not None and distance_m <= DISTANCE_VERY_CLOSE
                    else ANNOUNCE_COOLDOWN)

        # Build the track ID (use noid_ prefix for tracks without IDs)
        track_key = tid if tid is not None else f"noid_{label}"

        if tid is None:
            # No track ID — fall back to label-keyed cooldown only
            state = track_states.get(track_key)
            if state and now - state['last_announced'] < cooldown:
                continue
            is_new       = state is None
            approaching  = False
            settled      = False
            announce_count = 0
            track_states[track_key] = {
                'label':          label,
                'last_announced': now,
                'last_seen':      now,
                'last_distance':  distance_m,
                'last_position':  position,
                'settled':        False,
                'announce_count': 0,
            }
        else:
            state  = track_states.get(tid)
            is_new = state is None

            if state:
                state['last_seen'] = now

                # Get settled state
                settled = state.get('settled', False)
                announce_count = state.get('announce_count', 0)

                # Detect approach: distance decreased by more than threshold
                prev_dist   = state.get('last_distance')
                approaching = (
                    distance_m is not None and
                    prev_dist  is not None and
                    prev_dist - distance_m >= DISTANCE_REANNOUNCE_DELTA
                )

                # Detect wakeup: significant distance change (even if approaching didn't trigger)
                # This wakes up settled tracks
                distance_change = 0.0
                if prev_dist is not None and distance_m is not None:
                    distance_change = abs(distance_m - prev_dist)
                wakeup = settled and distance_change >= SETTLE_WAKEUP_DELTA

                # Check if silent this cycle
                time_ok  = now - state['last_announced'] >= cooldown
                urgent   = is_hazard and distance_m is not None and distance_m <= DISTANCE_VERY_CLOSE

                # Force announce (manual scan) always wins
                # Otherwise: check normal conditions
                should_announce = force_announce or time_ok or approaching or urgent or wakeup

                # Hazard very close should never be settled
                if urgent:
                    settled = False
                    announce_count = 0

                if not should_announce:
                    # Track is settled or in cooldown — update state but don't announce
                    state['last_distance'] = distance_m
                    state['last_position'] = position
                    continue  # nothing new to say

                # We're announcing — increment count and check settling
                if not force_announce:  # Don't count manual scans toward settling
                    announce_count += 1
                    if announce_count >= SETTLE_ANNOUNCE_COUNT:
                        settled = True
                    state['announce_count'] = announce_count
                    state['settled'] = settled

                state['last_announced'] = now
                state['last_distance']  = distance_m
                state['last_position']  = position

            else:
                # Brand new track
                approaching = False
                settled = False
                announce_count = 0
                track_states[tid] = {
                    'label':          label,
                    'last_announced': now,
                    'last_seen':      now,
                    'last_distance':  distance_m,
                    'last_position':  position,
                    'settled':        False,
                    'announce_count': 0,
                }

        # Build and queue the phrase
        phrase = build_announcement(label, position, distance_m, is_new, approaching)
        if phrase:
            queue_od_message(phrase)
            print(f"[Announce] {phrase}")


def clear_track_states():
    """Clear all track states (useful when switching modes)."""
    global track_states
    track_states = {}


def trigger_manual_scan(detections: list, frame_width: int):
    """
    Manual scan — announce all currently visible tracks regardless of settled state.
    
    Called when Button 2 is pressed in OD mode.
    - Announces all detected objects sorted by distance (closest first)
    - Resets settled=False and announce_count=0 on every track
    - If no objects detected, speaks "No objects currently detected"
    
    Args:
        detections: List of detection dicts from run_dual_detection()
        frame_width: Width of the frame for position calculation
    """
    if not detections:
        queue_od_message("No objects currently detected")
        print("[Manual Scan] No objects detected")
        return

    # Sort by distance (closest first)
    sorted_dets = sorted(
        detections,
        key=lambda d: d.get('distance_m', float('inf'))
    )

    # Reset all tracks and announce each one
    for det in sorted_dets:
        tid = det.get("track_id")
        label = det["label"]
        distance_m = det.get("distance_m")
        x_center = (det["x1"] + det["x2"]) / 2
        position = get_position(frame_width, x_center)

        # Reset track state
        track_key = tid if tid is not None else f"noid_{label}"
        if track_key in track_states:
            track_states[track_key]['settled'] = False
            track_states[track_key]['announce_count'] = 0
            track_states[track_key]['last_announced'] = 0  # Force announce

        # Build announcement (is_new=True to get full introduction)
        phrase = build_announcement(label, position, distance_m, is_new=True, approaching=False)
        if phrase:
            queue_od_message(phrase)
            print(f"[Manual Scan] {phrase}")

    print(f"[Manual Scan] Announced {len(sorted_dets)} objects")