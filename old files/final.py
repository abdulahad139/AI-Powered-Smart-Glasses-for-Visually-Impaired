import pyttsx3
from threading import Thread
from queue import Queue
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Define absolute paths
MODEL_PATH = r"best.pt"
VIDEO_PATH = r"fahad.mp4"

# Cooldown and distance tracking
last_spoken = {}
last_distances = {}
speech_cooldown = 3  # Reduced to 3 seconds for testing
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
                    
                label, distance, position = item
                current_time = time.time()
                rounded_distance = round(distance * 2) / 2
                distance_str = str(int(rounded_distance)) if rounded_distance.is_integer() else str(rounded_distance)

                # Cooldown to prevent flooding
                if label in last_spoken and current_time - last_spoken[label] < speech_cooldown:
                    print(f"Skipping {label} due to cooldown")
                    continue

                # Determine motion direction
                prev_distance = last_distances.get(label, None)
                if prev_distance is not None:
                    if distance < prev_distance - 0.3:
                        motion = "approaching"
                    elif distance > prev_distance + 0.3:
                        motion = "going away"
                    else:
                        motion = "ahead"
                else:
                    motion = "ahead"

                if distance <= 2:
                    motion = "very close"

                last_distances[label] = distance

                # Create message and speak it
                message = f"{label} is {distance_str} meters on your {position}, {motion}"
                print(f"Speaking: {message}")
                
                # Create new TTS engine for each message (fixes pyttsx3 threading issues)
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

# Calculate distance
def calculate_distance(box, frame_width, label):
    object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()
    if label in class_avg_sizes:
        object_width *= class_avg_sizes[label]["width_ratio"]
    distance = (frame_width * 0.5) / np.tan(np.radians(70 / 2)) / (object_width + 1e-6)
    return round(distance, 2)

# Get object position
def get_position(frame_width, box):
    if box[0] < frame_width // 3:
        return "left"
    elif box[0] < 2 * (frame_width // 3):
        return "center"
    else:
        return "right"

# Blur region
def blur_person(image, box):
    x, y, w, h = box.xyxy[0].cpu().numpy().astype(int)
    top_region = image[y:y+int(0.08 * h), x:x+w]
    blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
    image[y:y+int(0.08 * h), x:x+w] = blurred_top_region
    return image

# Object widths
class_avg_sizes = {
    "people": {"width_ratio": 2.5},
    "door": {"width_ratio": 0.4},
    "Whiteboard": {"width_ratio": 0.5},
    "desk": {"width_ratio": 0.5},
    "fullchair": {"width_ratio": 2.4},
    "emptychair": {"width_ratio": 2.4},
}

def main():
    # Initialize startup TTS. This is an offline text-to-speech library.
    try:
        startup_engine = pyttsx3.init()
        startup_engine.setProperty('rate', 235) # This is the speaking speed
        startup_engine.setProperty('volume', 1.0) 
        startup_engine.say("System activated") # Annoucement that the system is ready
        startup_engine.runAndWait() # This will wait for all says. Blocking function call.
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
        cap = cv2.VideoCapture(VIDEO_PATH)
    except Exception as e:
        print(f"Error loading model or video: {e}")
        return
    
    if not cap.isOpened():
        print("Error: Could not open video file")
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
                        nearest_object = None
                        min_distance = float('inf')

                        for box in result.boxes:
                            label = result.names[box.cls[0].item()]
                            print(f"Detected: '{label}'")
                            cords = [round(x) for x in box.xyxy[0].tolist()]
                            distance = calculate_distance(box, frame.shape[1], label)

                            # Track nearest object
                            if distance < min_distance:
                                min_distance = distance
                                nearest_object = (label, round(distance, 1), cords)
        
                            # ALWAYS set default color FIRST (before any operations that might fail)
                            color = (255, 0, 0) # Blue for everything by default

                            # Special handling for privacy-sensitive objects
                            if label == "people" or label == "fullchair":
                                try:
                                    frame = blur_person(frame, box)
                                    color = (0, 255, 0)  # Only change color if blur succeeds
                                except Exception as blur_error:
                                    print(f"Blur error for {label}: {blur_error}")
                                    # Keep default blue color if blur fails

                            # Draw bounding box and label (color is guaranteed to exist)
                            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), color, 2)
                            cv2.putText(frame, f"{label} - {distance:.1f}m", 
                                        (cords[0], cords[1] - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                        # Add to speech queue if object is close enough
                        if nearest_object and nearest_object[1] <= 12.5:
                            position = get_position(frame.shape[1], nearest_object[2])
                            print(f"Detected: {nearest_object[0]} at {nearest_object[1]}m on {position}")
                            
                            # Only add if queue is not too full
                            if speech_queue.qsize() < 2:
                                speech_queue.put((nearest_object[0], nearest_object[1], position))
                                print(f"Added to speech queue")
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

    main()