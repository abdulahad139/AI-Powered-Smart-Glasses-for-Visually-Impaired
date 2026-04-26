import pyttsx3
from threading import Thread
from queue import Queue
import cv2
import numpy as np
import time
from ultralytics import YOLO
import concurrent.futures

# Define absolute paths
MODEL_PATH = r"gpModel.pt"
VIDEO_PATH = r"video2.mp4"

# Optimized settings for real-time performance
last_spoken = {}
last_distances = {}
speech_cooldown = 2  # Reduced cooldown for more responsive feedback
max_speech_queue_size = 1  # Keep only latest message

# Queue for speech - smaller queue for real-time
speech_queue = Queue(maxsize=2)

def speak_worker_optimized():
    """Optimized TTS worker with pre-initialized engine and faster settings"""
    # Pre-initialize TTS engine with faster settings
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 280)  # Faster speech rate
    tts_engine.setProperty('volume', 1.0)
    
    while True:
        try:
            # Non-blocking get with timeout
            item = speech_queue.get(timeout=0.1)
            if item is None:  # Shutdown signal
                break
                
            label, distance, position = item
            current_time = time.time()
            
            # Quick cooldown check
            if label in last_spoken and current_time - last_spoken[label] < speech_cooldown:
                continue

            # Simplified distance formatting (faster)
            distance_str = f"{distance:.1f}" if distance >= 1 else f"{distance:.2f}"
            
            # Determine motion quickly
            prev_distance = last_distances.get(label)
            if prev_distance is not None:
                if distance < prev_distance - 0.5:  # Increased threshold for quicker detection
                    motion = "approaching"
                elif distance > prev_distance + 0.5:
                    motion = "moving away"
                else:
                    motion = ""
            else:
                motion = ""
            
            # Very close override
            if distance <= 2:
                motion = "very close"
            
            last_distances[label] = distance
            
            # Create shorter, quicker message
            if motion:
                message = f"{label} {distance_str} meters {position}, {motion}"
            else:
                message = f"{label} {distance_str} meters {position}"
            
            # Speak immediately - reuse engine for speed
            tts_engine.say(message)
            tts_engine.runAndWait()
            
            last_spoken[label] = current_time
            
            # Clear any waiting messages to stay current
            try:
                while True:
                    speech_queue.get_nowait()
            except:
                pass
                
        except:
            time.sleep(0.01)  # Very short sleep

def calculate_distance_fast(box, frame_width, label):
    """Optimized distance calculation"""
    # Get box coordinates directly as numpy array
    box_coords = box.xyxy[0].cpu().numpy()
    object_width = box_coords[2] - box_coords[0]  # x2 - x1
    width_ratio = class_avg_sizes.get(label, {}).get("width_ratio", 1.0)
    object_width *= width_ratio
    distance = (frame_width * 0.5) / np.tan(np.radians(35)) / (object_width + 1e-6)
    return round(distance, 1)

def get_position_fast(frame_width, box_center_x):
    """Faster position calculation"""
    third = frame_width // 3
    if box_center_x < third:
        return "left"
    elif box_center_x < 2 * third:
        return "center"
    else:
        return "right"

def blur_person_fast(image, box):
    """Optimized blurring"""
    coords = box.xyxy[0].cpu().numpy().astype(int)
    x, y, x2, y2 = coords
    h = y2 - y
    blur_height = max(int(0.08 * h), 5)  # Minimum blur height
    
    if y + blur_height < image.shape[0]:
        region = image[y:y+blur_height, x:x2]
        image[y:y+blur_height, x:x2] = cv2.GaussianBlur(region, (9, 9), 0)  # Smaller blur kernel
    return image

# Simplified object size ratios (only essential objects)
class_avg_sizes = {
    "person": {"width_ratio": 2.5},
    "car": {"width_ratio": 0.4},
    "bicycle": {"width_ratio": 2.0},
    "motorcycle": {"width_ratio": 2.0},
    "bus": {"width_ratio": 0.3},
}

def main():
    # Quick startup message
    try:
        startup_engine = pyttsx3.init()
        startup_engine.setProperty('rate', 300)
        startup_engine.say("System ready")
        startup_engine.runAndWait()
        startup_engine.stop()
        del startup_engine
    except:
        print("Startup TTS failed")
    
    # Start optimized TTS thread
    tts_thread = Thread(target=speak_worker_optimized, daemon=True)
    tts_thread.start()
    
    # Load model
    try:
        model = YOLO(MODEL_PATH)
        # Optimize model for speed
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)  # Warm up
        cap = cv2.VideoCapture(VIDEO_PATH)
    except Exception as e:
        print(f"Error loading: {e}")
        return
    
    if not cap.isOpened():
        print("Cannot open video")
        return
    
    # Optimize video capture
    cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)  # Reduce buffer to minimize delay
    
    # Output video setup (optional - disable for max performance)
    write_video = False  # Set to False for maximum real-time performance
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_with_boxes.avi', fourcc, 30.0, 
                             (int(cap.get(3)), int(cap.get(4))))
    
    pause = False
    frame_count = 0
    last_detection_time = 0
    detection_interval = 0.1  # Process detection every 100ms
    
    try:
        while cap.isOpened():
            if not pause:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                frame_count += 1
                
                # Time-based processing instead of frame-based for consistent real-time performance
                if current_time - last_detection_time >= detection_interval:
                    last_detection_time = current_time
                    
                    try:
                        # Fast inference with smaller image size
                        small_frame = cv2.resize(frame, (416, 416))  # Smaller for speed
                        results = model.predict(small_frame, verbose=False, conf=0.5)  # Higher confidence threshold
                        result = results[0]
                        
                        # Scale factor for coordinates
                        scale_x = frame.shape[1] / 416
                        scale_y = frame.shape[0] / 416
                        
                        nearest_object = None
                        min_distance = float('inf')
                        
                        for box in result.boxes:
                            label = result.names[box.cls[0].item()]
                            
                            # Skip objects not in our optimized list
                            if label not in class_avg_sizes:
                                continue
                            
                            # Scale coordinates back to original frame
                            scaled_box = box.xyxy[0].cpu().numpy()
                            cords = [int(x) for x in scaled_box.tolist()]
                            
                            distance = calculate_distance_fast(type('', (), {'xyxy': [scaled_box]})(), frame.shape[1], label)
                            
                            if distance < min_distance and distance <= 15:  # Only consider objects within 15m
                                min_distance = distance
                                box_center_x = (cords[0] + cords[2]) // 2
                                nearest_object = (label, distance, box_center_x)
                            
                            # Draw on original frame
                            color = (0, 255, 0) if label == "person" else (0, 255, 255) if label == "car" else (255, 0, 0)
                            
                            if label == "person":
                                # Quick blur (optional - can be disabled for max speed)
                                pass  # Disabled for speed
                            
                            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), color, 2)
                            cv2.putText(frame, f"{label} {distance:.1f}m", 
                                      (cords[0], cords[1] - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Quick speech queue update
                        if nearest_object and nearest_object[1] <= 12:
                            position = get_position_fast(frame.shape[1], nearest_object[2])
                            
                            # Non-blocking queue put - drop if full
                            try:
                                speech_queue.put_nowait((nearest_object[0], nearest_object[1], position))
                            except:
                                pass  # Queue full, skip this message
                    
                    except Exception as e:
                        print(f"Detection error: {e}")
                
                cv2.imshow('Audio World', frame)
                if write_video:
                    out.write(frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                pause = not pause
                
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        print("Shutting down...")
        speech_queue.put(None)
        cap.release()
        if write_video:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()