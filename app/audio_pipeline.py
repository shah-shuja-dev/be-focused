"""
audio_pipeline.py -- TTS with wav caching.

pyttsx3:  offline, instant, robotic but functional.
coqui:    voice cloning from a 6s reference clip. Better quality.
          Swap by setting TTS_ENGINE=coqui in .env.
          Requires: pip install TTS  (large download ~2GB, one time)
          And: audio_cache\mf_reference.wav (6s clean Morgan Freeman clip)

Generated wav files are cached by content hash — identical quotes
never re-synthesize, so repeated violations are instant.
"""

import os, hashlib, threading, time
from dotenv import load_dotenv

load_dotenv(r"config\.env")

TTS_ENGINE        = os.getenv("TTS_ENGINE",        "pyttsx3")
QUOTE_STYLE       = os.getenv("QUOTE_STYLE",       "motivational")
TRIGGER_MODE      = os.getenv("TRIGGER_MODE",      "on_violation")
REMINDER_INTERVAL = float(os.getenv("REMINDER_INTERVAL_SECS", "3600"))
CACHE_DIR         = "audio_cache"

os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


class AudioPipeline:

    def __init__(self):
        self._last_spoke = 0
        self._lock       = threading.Lock()
        self._engine_name = TTS_ENGINE

        if TTS_ENGINE == "pyttsx3":
            import pyttsx3
            self._tts = pyttsx3.init()
            self._tts.setProperty("rate", 155)
            # Prefer a male voice
            voices = self._tts.getProperty("voices")
            for v in voices:
                name = v.name.lower()
                if "david" in name or "mark" in name or "male" in name:
                    self._tts.setProperty("voice", v.id)
                    break
            print("TTS: pyttsx3 ready")

        elif TTS_ENGINE == "coqui":
            from TTS.api import TTS
            self._tts    = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._ref_wav = os.path.join(CACHE_DIR, "mf_reference.wav")
            if not os.path.exists(self._ref_wav):
                print(f"WARNING: Coqui needs audio_cache\\mf_reference.wav")
                print(f"  Download a 6-second clean Morgan Freeman clip and save it there.")
            print("TTS: Coqui XTTS-v2 ready")

        else:
            raise ValueError(f"Unknown TTS_ENGINE: {TTS_ENGINE}")

    def speak(self, text: str, force: bool = False):
        """
        Speak text in a background thread (non-blocking).
        force=True skips the 10s debounce.
        """
        now = time.time()
        with self._lock:
            if not force and (now - self._last_spoke) < 10:
                return
            self._last_spoke = now
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text: str):
        key      = _cache_key(text)
        wav_path = os.path.join(CACHE_DIR, f"{key}.wav")
        try:
            if self._engine_name == "pyttsx3":
                if not os.path.exists(wav_path):
                    self._tts.save_to_file(text, wav_path)
                    self._tts.runAndWait()
                self._play(wav_path)

            elif self._engine_name == "coqui":
                if not os.path.exists(wav_path):
                    self._tts.tts_to_file(
                        text        = text,
                        speaker_wav = self._ref_wav,
                        language    = "en",
                        file_path   = wav_path,
                    )
                self._play(wav_path)

        except Exception as e:
            print(f"TTS error: {e}")

    def _play(self, wav_path: str):
        try:
            import playsound
            playsound.playsound(wav_path)
        except Exception:
            try:
                import winsound
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            except Exception as e:
                print(f"Playback error: {e}")

    def on_violation(self, violation: dict):
        """Direct callback for ViolationDetector."""
        from app.quotes import get_quote
        quote = get_quote(QUOTE_STYLE)
        print(f"[AUDIO] Violation #{violation['id']} -> speaking")
        self.speak(quote, force=True)

    def start_reminder_loop(self, is_unfocused_fn):
        """
        If TRIGGER_MODE is 'periodic' or 'both', start a background thread
        that speaks every REMINDER_INTERVAL seconds while still unfocused.
        """
        if TRIGGER_MODE not in ("periodic", "both"):
            return

        def _loop():
            while True:
                time.sleep(REMINDER_INTERVAL)
                if is_unfocused_fn():
                    from app.quotes import get_quote
                    self.speak(get_quote(QUOTE_STYLE))

        threading.Thread(target=_loop, daemon=True).start()
        print(f"Reminder loop active (every {REMINDER_INTERVAL}s while unfocused)")