# SignVoc-AI: Hybrid Edge-Cloud Sign Language Translation

SignVoc-AI is a high-performance **Hybrid Edge-Cloud** system that bridges the gap between Sign Language and spoken English. It leverages local efficiency for computer vision and the unlimited power of Cloud AI for semantic understanding.

## 🧠 System Architecture

This project utilizes a **Smart Local** architecture:

1.  **Vision (Edge)**:  
    Running locally on your machine, **MediaPipe Holistic** detects hand landmarks in real-time, and a custom **TFLite Model** recognizes glosses (words) with low latency.
2.  **Intelligence (Cloud)**:  
    Detected glosses are sent to **Groq Cloud** running **Llama-3.3-70b-versatile**. The LLM contextually refines raw glosses into grammatically correct English sentences (e.g., "ME HUNGRY" -> "I am hungry").
3.  **Voice (Edge)**:  
    The refined sentence is synthesized back into speech locally using **Coqui TTS (VITS VCTK)**, ensuring high-quality, natural-sounding audio without latency-heavy audio streaming.

## ✨ Features

- **Real-Time Recognition**: Detects 250+ distinct ASL signs using lightweight TFLite inference.
- **Context-Aware Translation**: Uses Large Language Models (LLM) to fix grammar, pronouns, and sentence structure.
- **Natural Speech Synthesis**: Human-like voice output running locally via VITS.
- **Smart Pause**: Automatically detects when the user stops signing to finalize and speak the sentence.
- **Privacy-First**: Video feeds never leave your device; only text glosses are sent to the cloud.

## 📋 Prerequisites

- **OS**: Windows 10/11
- **Python**: 3.10+
- **System Audio**: `espeak-ng` (Required for TTS phonemization)
- **API Key**: A valid [Groq Cloud](https://console.groq.com/) API Key (Free tier available).

## 🛠️ Installation

### 1. Install System Dependencies (Crucial)
The TTS engine requires `espeak-ng` to work.
1. Download installer: [espeak-ng-X64.msi](https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi)
2. Install it.
3. **Restart your computer** to ensure system paths are updated.

### 2. Clone Repository
```bash
git clone https://github.com/faveria/SignVoc-AI.git
cd SignVoc-AI
```

### 3. Setup Python Environment
```bash
# Create Virtual Environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 4. Configure Secrets
Create a `.env` file in the root directory and add your Groq API Key:
```ini
GROQ_API_KEY=gsk_your_actual_api_key_here
```
*(Note: Never commit your .env file to GitHub!)*

## 🚀 Usage

Run the main application:
```bash
python main.py
```

### Operational Guide:
1.  **Sign**: Perform ASL signs in front of the camera.
2.  **Monitor**: Recognized glosses (words) will appear in the buffer at the top of the screen.
3.  **Pause**: Stop signing for **3 seconds**.
4.  **Listen**: The system will automatically translate the glosses and speak the sentence.

**Controls**:
- Press `q` to exit safely.

## 📂 Project Structure

- `main.py`: The central orchestrator managing the vision loop, thread management, and logic flow.
- `src/backbone.py`: Definition of the Keras/TFLite model architecture.
- `src/llm_client.py`: Interface for communicating with Groq API (Llama-3).
- `src/tts.py`: Local audio engine wrapper for Coqui TTS (VITS).
- `src/landmarks_extraction.py`: Optimized MediaPipe logic for landmark processing.
- `src/config.py`: Centralized configuration management.

## 🤝 Credits
Powered by **MediaPipe**, **TensorFlow**, **Groq Cloud**, and **Coqui TTS**.
