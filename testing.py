import pyttsx3
from threading import Thread
from queue import Queue
import time

# Test TTS in thread - Fixed version
def test_tts():
    speech_queue = Queue()
    
    def tts_worker():
        while True:
            if not speech_queue.empty():
                message = speech_queue.get()
                if message is None:
                    break
                
                # Create new engine instance for each message
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 235)
                    engine.setProperty('volume', 1.0)
                    print(f"Speaking: {message}")
                    engine.say(message)
                    engine.runAndWait()
                    engine.stop()  # Important: stop the engine
                    del engine  # Clean up
                except Exception as e:
                    print(f"TTS Error: {e}")
            else:
                time.sleep(0.1)
    
    # Start TTS thread
    tts_thread = Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    
    # Test messages
    print("Adding test message 1")
    speech_queue.put("Test message 1")
    time.sleep(4)
    
    print("Adding test message 2")
    speech_queue.put("Test message 2")
    time.sleep(4)
    
    print("Adding test message 3")
    speech_queue.put("Test message 3")
    time.sleep(4)
    
    speech_queue.put(None)  # Stop signal
    tts_thread.join()  # Wait for thread to finish

if __name__ == "__main__":
    test_tts()