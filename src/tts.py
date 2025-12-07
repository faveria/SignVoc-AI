import os
import torch
import numpy as np
import threading
import winsound
import soundfile as sf
import tempfile
import time

# Set espeak path BEFORE importing TTS
# This is crucial for Windows
try:
    from src.config import CONFIG
    if os.path.exists(CONFIG.PHONEMIZER_ESPEAK_PATH):
        # 1. Set Library Path
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = CONFIG.PHONEMIZER_ESPEAK_PATH
        
        # 2. Add Directory to PATH (Crucial for executables)
        espeak_dir = os.path.dirname(CONFIG.PHONEMIZER_ESPEAK_PATH)
        os.environ["PATH"] += os.pathsep + espeak_dir
        os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_dir
        
        print(f"Set eSpeak config: {espeak_dir}")
    else:
        print(f"Warning: eSpeak dll not found at {CONFIG.PHONEMIZER_ESPEAK_PATH}")
except ImportError:
     # Fallback if config not ready yet
     default_path = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
     if os.path.exists(default_path):
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = default_path
        espeak_dir = os.path.dirname(default_path)
        os.environ["PATH"] += os.pathsep + espeak_dir

from TTS.api import TTS

class TextToSpeech:
    """
    Text-to-Speech Engine using Coqui TTS.
    """
    def __init__(self, model_name: str = "tts_models/en/vctk/vits", gpu: bool = False):
        """
        Initialize TTS engine.
        
        Args:
            model_name (str): Name of the Coqui TTS model to load.
            gpu (bool): Whether to use GPU.
        """
        print(f"Initializing TTS Engine (Model: {model_name})... This might take a while on first run.")
        self.gpu = gpu and torch.cuda.is_available()
        # Initialize TTS
        self.tts = TTS(model_name=model_name, progress_bar=False, gpu=self.gpu)
        print("TTS Engine Ready.")
        
        self.lock = threading.Lock()
        self.is_speaking = False

    def _play_audio(self, wav_data, sample_rate):
        """Play audio waveform using Windows winsound."""
        try:
             # Create a temp file
             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                 temp_path = f.name
             
             # Write wav to file
             sf.write(temp_path, wav_data, sample_rate)
             
             # Play using winsound (Windows only, built-in)
             winsound.PlaySound(temp_path, winsound.SND_FILENAME)
             
             # Cleanup
             try:
                os.remove(temp_path)
             except:
                pass
                
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
             with self.lock:
                 self.is_speaking = False

    def speak(self, text: str):
        """
        Convert text to speech and play it in a separate thread.
        
        Args:
            text (str): Text to speak.
        """
        with self.lock:
            if self.is_speaking:
                return
            self.is_speaking = True

        def run_thread():
            try:
                # Generate speech
                # Check for multi-speaker
                speaker = None
                if self.tts.is_multi_speaker:
                    # VCTK has many speakers. p225 is a standard English male, p226 is English Male
                    # p225, p226, p227 are good. Let's use p226 (deeper voice) or p225 (standard).
                    # Or 'p225' is often default example.
                    speaker = 'p225' 

                wav = self.tts.tts(text=text, speaker=speaker)
                wav_np = np.array(wav)
                
                # VITS VCTK usually 22050Hz but let's be dynamic if possible, currently hardcoded in logic usually is 22050
                sample_rate = 22050 
                
                self._play_audio(wav_np, sample_rate)
                
            except Exception as e:
                print(f"TTS Error: {e}")
                with self.lock:
                    self.is_speaking = False

        threading.Thread(target=run_thread, daemon=True).start()

if __name__ == "__main__":
    # Test
    tts = TextToSpeech()
    print("Speaking 'Hello'...")
    tts.speak("Hello")
    time.sleep(2)
    print("Speaking 'World'...")
    tts.speak("World")
    time.sleep(2)
