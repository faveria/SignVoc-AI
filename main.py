"""
Real-time ASL (American Sign Language) Recognition

This script uses a pre-trained TFLite model to perform real-time ASL recognition using webcam feed.
It utilizes the MediaPipe library for hand tracking and landmark extraction.

Author: 209sontung
Date: May 2023 [Refactored Dec 2025]
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from collections import deque
from typing import List, Optional, Tuple, Any

from src.backbone import TFLiteModel, get_model
from src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file
from src.config import CONFIG
from src.utils import Preprocess
from src.tts import TextToSpeech 

class SignLanguageDetector:
    """
    Class to handle the sign language detection logic.
    """
    def __init__(self, models_path: List[str], map_path: str = "src/sign_to_prediction_index_map.json"):
        """
        Initialize the detector.

        Args:
            models_path (List[str]): List of paths to the model weights.
            map_path (str): Path to the sign-to-prediction index map JSON.
        """
        self.models_path = models_path
        self.map_data = load_json_file(map_path)
        self.s2p_map = {k.lower(): v for k, v in self.map_data.items()}
        self.p2s_map = {v: k for k, v in self.map_data.items()}
        
        self.models = self._load_models()
        self.tflite_keras_model = TFLiteModel(islr_models=self.models)
        
        # Sliding Window Buffer (deque automatically handles popping old frames)
        self.sequence_data = deque(maxlen=CONFIG.SEQ_LEN)
        
        # Buffer for results (detected sentences)
        self.sentence: List[str] = []
        
        # Mediapipe Holistic
        self.mp_holistic = mp.solutions.holistic
        
        
        # Prediction Frequency Control
        self.frame_counter = 0
        self.prediction_frequency = 5 # Predict every 5 frames
        
        # Initialize TTS
        try:
            self.tts = TextToSpeech(gpu=True)
        except Exception as e:
            print(f"Warning: TTS failed to initialize. Audio will be disabled. Error: {e}")
            self.tts = None
            
        # Smart Pause / Sentence Building
        self.sentence_buffer: List[str] = []
        self.last_detection_time = time.time()
        self.pause_threshold = 3.0 # seconds

    def _load_models(self) -> List[Any]:
        """Load and initialize models."""
        models = [get_model() for _ in self.models_path]
        for model, path in zip(models, self.models_path):
            try:
                model.load_weights(path)
                print(f"Successfully loaded model weights from {path}")
            except Exception as e:
                print(f"Error loading model {path}: {e}")
        return models

    def decode_sign(self, index: int) -> Optional[str]:
        """Convert prediction index to sign string."""
        return self.p2s_map.get(index)

    def process_frame(self, frame: np.ndarray, holistic_model: Any) -> Tuple[np.ndarray, Optional[str]]:
        """
        Process a single frame: detect landmarks, draw, and predict if sequence is full.

        Args:
            frame (np.ndarray): Input video frame.
            holistic_model: Initialized MediaPipe holistic model.

        Returns:
            Tuple[np.ndarray, Optional[str]]: Processed image with drawings, and detected sign (if any).
        """
        image, results = mediapipe_detection(frame, holistic_model)
        draw(image, results)
        
        landmarks = extract_coordinates(results)
        self.sequence_data.append(landmarks)
        self.frame_counter += 1
        
        detected_sign = None
        
        # Check if buffer is full (maxlen reached) AND it's time to predict
        if len(self.sequence_data) == CONFIG.SEQ_LEN and (self.frame_counter % self.prediction_frequency == 0):
             # Prepare input batch [1, SEQ_LEN, 543*3]
             input_seq = np.array(list(self.sequence_data), dtype=np.float32)
             
             prediction = self.tflite_keras_model(input_seq)["outputs"]
             max_prob = np.max(prediction.numpy(), axis=-1)[0]
             print(f"Max Prob: {max_prob:.2f}")

             if max_prob > CONFIG.THRESH_HOLD:
                 sign_idx = int(np.argmax(prediction.numpy(), axis=-1)[0])
                 detected_sign = self.decode_sign(sign_idx)
                 
                 # Optional: Reset sequence if a strong detection is found to avoid double counting?
                 # No, in sliding window we usually keep going. But we might want simple debounce.
                 # For now, let's keep it pure sliding window.
        
        return image, detected_sign

class WebcamStream:
    """Threaded webcam capture for higher FPS."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # optimize text
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # Lower resolution slightly for inference speed if needed (optional)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.stopped, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    """Main application loop."""
    models_path = [
        './models/islr-fp16-192-8-seed_all42-foldall-last.h5',
    ]
    
    try:
        detector = SignLanguageDetector(models_path=models_path)
    except FileNotFoundError as e:
        print(f"Critical Error: {e}")
        return

    # Use threaded camera
    cap = WebcamStream(0).start()
    
    # Allow camera to warm up
    time.sleep(1.0)
    
    # Dummy check (threaded stream doesn't have isOpened in same way, but read will return valid frame)
    if not cap.grabbed:
         print("Error: Could not open webcam.")
         cap.stop()
         return

    # Set up MediaPipe Holistic
    with detector.mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=0
    ) as holistic:
        
        # FPS Calculation
        prev_time = 0
        
        print("Starting real-time recognition... Press 'q' to quit.")
        
        while True:
            stopped, frame = cap.read()
            if stopped or frame is None:
                # If frame is None (maybe initializing), continue or break
                if frame is None: continue 
                print("Failed to grab frame.")
                break
            
            # FPS Logic
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Process frame
            image, sign = detector.process_frame(frame, holistic)
            
            # Update Sentence Logic
            if sign:
                # Basic debounce: don't add if it's the exact same sign detected immediately? 
                # For now, let's allow it but maybe prevent flooding if it predicts every 5 frames.
                # Let's simple check if the last added word is different OR if enough time appeared.
                # Actually, given sliding window, we might get multiple detections of same sign.
                # Simple logic: Only add if distinct from the *very last* item in buffer?
                # Or just trust the sliding window and threshold?
                
                # Let's prevent duplicate consecutive words for stability
                # Let's prevent duplicate consecutive words for stability
                if not detector.sentence_buffer or detector.sentence_buffer[-1] != sign:
                     detector.sentence_buffer.append(sign)
                     detector.last_detection_time = time.time()
                     print(f"Buffer: {detector.sentence_buffer}") # Debug
            
            # --- Smart Pause Logic ---
            # If buffer has words AND user hasn't signed for N seconds -> Speak
            if detector.sentence_buffer and (time.time() - detector.last_detection_time > detector.pause_threshold):
                full_sentence = " ".join(detector.sentence_buffer)
                print(f"Speaking Sentence: {full_sentence}")
                
                if detector.tts:
                    detector.tts.speak(full_sentence)
                
                # Add to history (detector.sentence) for display
                detector.sentence.insert(0, full_sentence)
                
                # Clear buffer
                detector.sentence_buffer = []
            
            # --- UI Rendering ---
            image = cv2.flip(image, 1) # Flip for selfie view
            
            # Draw FPS
            cv2.putText(image, f"FPS: {int(fps)}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Draw Prediction Status
            is_predicting = len(detector.sequence_data) == CONFIG.SEQ_LEN and (detector.frame_counter % detector.prediction_frequency == 0)
            if is_predicting:
                 cv2.circle(image, (width - 30, 30), 10, (0, 0, 255), -1) # Red dot when predicting
            
            # Display buffer size visually? Or just "Recording..."
            cv2.putText(image, f"Buff: {len(detector.sequence_data)}/{CONFIG.SEQ_LEN}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
             # Create a white banner at the top for text
            height, width = image.shape[0], image.shape[1]
            banner_height = height // 8
            white_banner = np.ones((banner_height, width, 3), dtype='uint8') * 255
            
            final_image = np.concatenate((white_banner, image), axis=0)

            # Display sentence
            # Show "Building: [ ... ]"
            current_buffer_text = " ".join(detector.sentence_buffer)
            if current_buffer_text:
                cv2.putText(final_image, f"Building: {current_buffer_text}...", (10, banner_height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)

            # Show Last Spoken sentence
            last_spoken = detector.sentence[0] if detector.sentence else ""
            cv2.putText(final_image, f"Spoken: {last_spoken}", (10, banner_height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('SignVoc-AI', final_image)
            
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
                
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()