# =============================================================================
# Audio — Text-to-speech and audio playback utilities
# =============================================================================
#
# Handles all audio output: TTS engine (Piper), speech worker thread, and audio playback.
#
# =============================================================================

import os
import time
import threading
import subprocess
import tempfile
from queue import Queue

from config import PIPER_MODEL_PATH, BT_DEVICE


# Shared speech queue for cross-module communication
speech_queue = Queue()

# Lazy-loaded Piper voice
_piper_voice = None


def _get_piper_voice():
    """Lazy-load the Piper voice."""
    global _piper_voice
    if _piper_voice is None:
        from piper.voice import PiperVoice
        _piper_voice = PiperVoice.load(PIPER_MODEL_PATH)
    return _piper_voice


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


def _speak_piper(text: str):
    """
    Speak text using Piper TTS.
    Synthesizes speech to a temporary WAV file and plays it via paplay.
    """
    try:
        voice = _get_piper_voice()
        
        # Synthesize to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        # Synthesize audio
        chunks = list(voice.synthesize(text))
        
        # Write WAV file
        import wave
        with wave.open(tmp_path, "wb") as wav_file:
            wav_file.setnchannels(chunks[0].sample_channels)
            wav_file.setsampwidth(chunks[0].sample_width)
            wav_file.setframerate(chunks[0].sample_rate)
            for chunk in chunks:
                wav_file.writeframes(chunk.audio_int16_bytes)

        # Play via paplay (with optional Bluetooth device)
        if BT_DEVICE:
            subprocess.run(["paplay", f"--device={BT_DEVICE}", tmp_path], check=True)
        else:
            subprocess.run(["paplay", tmp_path], check=True)

        # Cleanup
        os.unlink(tmp_path)
        
    except Exception as e:
        print(f"⚠️  Piper TTS error: {e}")


# Backward compatibility alias
def _speak_blocking(text: str, lang: str = "en"):
    """Speak English text immediately using Piper (blocking)."""
    _speak_piper(text)


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

                # Use Piper TTS
                try:
                    _speak_piper(message)
                except Exception as e:
                    print(f"⚠️  TTS error: {e}")

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