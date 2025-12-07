# SignVoc-AI: Real-Time Sign Language Recognition with Premium TTS

A high-performance Sign Language Recognition system that translates ASL capability into spoken English using **VITS (VCTK)**, a state-of-the-art Text-to-Speech model.

## Features
- **Real-Time Recognition**: Uses MediaPipe Holistic and a custom TFLite model to detect 250+ signs.
- **Natural TTS**: Integrated Coqui TTS (VITS VCTK) for human-like voice output.
- **Smart Pause**: Intelligent sentence construction that waits for natural pauses before speaking.
- **Optimized Performance**: Multi-threaded webcam capture and background audio processing for smooth UI (15-25+ FPS).
- **GPU Acceleration**: Fully supports CUDA for accelerated TTS inference.

## Prerequisites
- **OS**: Windows 10/11
- **Python**: 3.11 (Recommended)
- **Hardware**: NVIDIA GPU (Optional but recommended for smoother TTS)
- **System Dependency**: `espeak-ng` (Required for TTS)

## Installation Guide

### 1. Install System Dependency (Crucial!)
The high-quality TTS model requires `espeak-ng`.
1. Download the installer: [espeak-ng-X64.msi](https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi)
2. Install it (Standard installation).
3. **Restart your computer** or VS Code to ensure the system detects it.

### 2. Clone and Setup Environment
```bash
git clone https://github.com/yourusername/signvoc-ai.git
cd signvoc-ai

# Create Virtual Environment (Python 3.11)
python -m venv venv

# Activate Environment
.\venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```
*Note: If you have an NVIDIA GPU, ensure you have CUDA 11.8 installed for best performance.*

## Usage

Simply run the main script:
```bash
python main.py
```

- **First Run**: The system will download the VITS model (~100MB). This happens only once.
- **Operation**: 
    1. Stand in front of the camera.
    2. Perform signs.
    3. The system captures words in a buffer (displayed on screen).
    4. Stop signing for **3 seconds** to trigger the "Smart Pause".
    5. The system will speak the sentence aloud.
- **Controls**: Press `q` to exit.

## Project Structure
- `src/backbone.py`: Custom Keras/TFLite model definitions.
- `src/landmarks_extraction.py`: MediaPipe logic for converting frames to landmark vectors.
- `src/tts.py`: Audio engine handling VITS model and background playback.
- `src/config.py`: Configuration for model hyperparameters and paths.
- `main.py`: Main application loop integrating UI, Detection, and TTS.

## Troubleshooting
- **No Sound?**: Ensure `espeak-ng` is installed. The application tries to auto-detect it in `C:\Program Files\eSpeak NG`.
- **Low FPS?**: Ensure your GPU is being used. Only `face_landmarks` are disabled by default to boost performance.

## Credits
Powered by [MediaPipe](https://google.github.io/mediapipe/), [Coqui TTS](https://github.com/coqui-ai/TTS), and TensorFlow.
