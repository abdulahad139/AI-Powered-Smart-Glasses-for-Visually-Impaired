# =============================================================================
# Audio — Text-to-speech and audio playback utilities
# =============================================================================
#
# Handles all audio output: TTS engine, speech worker thread, and audio playback.
#
# =============================================================================

import os
import time
import wave
import tempfile
import threading
import subprocess
from queue import Queue

from piper.voice import PiperVoice

# =============================================================================
# PIPER TTS CONFIGURATION
# =============================================================================

PIPER_MODEL_PATH = "/home/ab/Desktop/final script/en_US-amy-medium.onnx"
BT_DEVICE        = "bluez_output.DF_F0_93_9C_67_B3.1"

print("🔊 Loading Piper TTS voice model...")
_piper_voice = PiperVoice.load(PIPER_MODEL_PATH)
print("✓ Piper TTS ready!\n")


def _piper_speak(text: str):
    """Synthesize text with Piper and play via Bluetooth. Blocking."""
    tmp_path = None
    try:
        chunks = list(_piper_voice.synthesize(text))
        if not chunks:
            print("⚠️  Piper produced no audio chunks.")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        with wave.open(tmp_path, "wb") as wav_file:
            wav_file.setnchannels(chunks[0].sample_channels)
            wav_file.setsampwidth(chunks[0].sample_width)
            wav_file.setframerate(chunks[0].sample_rate)
            for chunk in chunks:
                wav_file.writeframes(chunk.audio_int16_bytes)

        subprocess.run(
            ["paplay", f"--device={BT_DEVICE}", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"⚠️  Piper TTS error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# Shared speech queue for cross-module communication
speech_queue = Queue()


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
    """Speak English text immediately using Piper TTS (blocking)."""
    print(f"🔊 Speaking: {text}")
    _piper_speak(text)


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
    from config import MAX_QUEUE_SIZE

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

                _piper_speak(message)

            else:
                time.sleep(0.05)

        except Exception as e:
            print(f"⚠️  Speech worker error: {e}")
            time.sleep(0.1)


def start_speech_worker() -> threading.Thread:
    """Start the TTS worker thread and return the thread object."""
    tts_thread = threading.Thread(target=speak_worker, daemon=True)
    tts_thread.start()
    return tts_thread


def announce(message: str):
    """Queue a system announcement message for speech."""
    speech_queue.put(('announce', message, None, None))


def queue_od_message(message: str):
    """Queue an object detection message for speech."""
    from config import MAX_QUEUE_SIZE
    if speech_queue.qsize() < MAX_QUEUE_SIZE:
        speech_queue.put(('od', message, None, None))


def stop_speech_worker():
    """Signal the speech worker to stop."""
    speech_queue.put(None)
    time.sleep(0.5)