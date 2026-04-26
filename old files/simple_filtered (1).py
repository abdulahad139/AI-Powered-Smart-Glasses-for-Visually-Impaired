# this file is implementation of our first simple script with camera input
# no distance estimation feature
# Added class filtering to exclude unwanted objects
# Added confidence threshold filtering

import pyttsx3
from threading import Thread
from queue import Queue
import cv2
import time
from ultralytics import YOLO

# Define absolute paths
MODEL_PATH = r"best(7)_ncnn_model"
CAMERA_INDEX = 0 

# ===== CLASS FILTER CONFIGURATION =====
# Add class names you want to EXCLUDE from detection here
EXCLUDED_CLASSES = {
    'glasses',
    'calculator',
    'pencil',
}
# ======================================

# ===== CONFIDENCE THRESHOLD =====
CONFIDENCE_THRESHOLD = 0.45  
# ================================

# Cooldown tracking
last_spoken = {}
speech_cooldown = 3  # seconds between announcements for same object

# Queue for speech
speech_queue = Queue()

def speak_worker():
    """TTS worker thread - creates new engine for each message"""
    while True:
        try:
            if not speech_queue.empty():
                item = speech_queue.get(timeout=1)
                if item is None:  # Shutdown signal
                    break
                    
                label, position = item
                current_time = time.time()

                # Cooldown to prevent flooding
                if label in last_spoken and current_time - last_spoken[label] < speech_cooldown:
                    print(f"Skipping {label} due to cooldown")
                    continue

                # Create message and speak it
                if position == "center":
                    message = f"{label} detected in front of you"
                else:
                    message = f"{label} detected to your {position}"
                print(f"Speaking: {message}")
                
                # Create new TTS engine for each message
                try:
                    tts_engine = pyttsx3.init()
                    tts_engine.setProperty('rate', 235)
                    tts_engine.setProperty('volume', 1.0)
                    tts_engine.say(message)
                    tts_engine.runAndWait()
                    tts_engine.stop()
                    del tts_engine
                except Exception as tts_error:
                    print(f"TTS Error: {tts_error}")

                last_spoken[label] = current_time

                # Clear any backlog in queue
                while not speech_queue.empty():
                    try:
                        speech_queue.get_nowait()
                        print("Cleared backlogged message")
                    except:
                        break
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"Speaker thread error: {e}")
            time.sleep(0.1)

# Get object position
def get_position(frame_width, x_center):
    if x_center < frame_width // 3:
        return "left"
    elif x_center < 2 * (frame_width // 3):
        return "center"
    else:
        return "right"

# Blur region (optional privacy feature)
def blur_person(image, box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    top_region = image[y1:y1+int(0.08 * (y2-y1)), x1:x2]
    blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
    image[y1:y1+int(0.08 * (y2-y1)), x1:x2] = blurred_top_region
    return image

def main():
    # Initialize startup TTS
    try:
        startup_engine = pyttsx3.init()
        startup_engine.setProperty('rate', 235)
        startup_engine.setProperty('volume', 1.0)
        startup_engine.say("System activated")
        startup_engine.runAndWait()
        startup_engine.stop()
        del startup_engine
    except Exception as e:
        print(f"Startup TTS error: {e}")
    
    
    # Start TTS thread
    tts_thread = Thread(target=speak_worker, daemon=True)
    tts_thread.start()
    
    # Load model and video
    try:
        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(CAMERA_INDEX)
        # Optional: Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception as e:
        print(f"Error loading model or camera: {e}")
        return
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Output video setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_with_boxes.avi', fourcc, 20.0, 
                         (int(cap.get(3)), int(cap.get(4))))
    
    pause = False
    frame_count = 0

    try:
        while cap.isOpened():
            if not pause:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or failed to read frame")
                    break
                
                frame_count += 1
                
                # Process every 5th frame to reduce load
                if frame_count % 5 == 0:
                    try:
                        results = model.predict(frame, verbose=False)
                        result = results[0]

                        for box in result.boxes:
                            label = result.names[box.cls[0].item()]
                            confidence = box.conf[0].item()
                            
                            # ===== FILTER EXCLUDED CLASSES =====
                            if label.lower() in EXCLUDED_CLASSES:
                                continue  # Skip this detection
                            # ===================================
                            
                            # ===== FILTER LOW CONFIDENCE DETECTIONS =====
                            if confidence <= CONFIDENCE_THRESHOLD:
                                #print(f"Filtered out: '{label}' (confidence: {confidence:.2f} <= {CONFIDENCE_THRESHOLD})")
                                continue  # Skip low-confidence detection
                            # ============================================
                            
                            print(f"Detected: '{label}' (confidence: {confidence:.2f})")
                            
                            # Get bounding box coordinates from model
                            cords = [round(x) for x in box.xyxy[0].tolist()]
                            x1, y1, x2, y2 = cords
                            
                            # Calculate center point for position detection
                            x_center = (x1 + x2) // 2
                            
                            # Default color for bounding box
                            color = (255, 0, 0)  # Blue

                            # Special handling for privacy-sensitive objects
                            if label == "people" or label == "fullchair":
                                try:
                                    frame = blur_person(frame, box)
                                    color = (0, 255, 0)  # Green if blur succeeds
                                except Exception as blur_error:
                                    print(f"Blur error for {label}: {blur_error}")

                            # Draw bounding box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                        (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Add to speech queue for announcement
                            position = get_position(frame.shape[1], x_center)
                            
                            # Only add if queue is not too full
                            if speech_queue.qsize() < 2:
                                speech_queue.put((label, position))
                                print(f"Added to speech queue: {label} on {position}")
                            else:
                                print("Speech queue full, skipping")
                                
                    except Exception as e:
                        print(f"Detection error: {e}")
                        
                cv2.imshow('Audio World', frame)
                out.write(frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('p'):
                pause = not pause
                print(f"Paused: {pause}")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        print("Shutting down...")
        speech_queue.put(None)  # Signal TTS thread to stop
        time.sleep(1)  # Give it time to finish
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
