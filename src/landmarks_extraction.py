"""
ASL Recognition Utility Functions

This module contains utility functions for ASL recognition using the Mediapipe library.
"""

from .config import ROWS_PER_FRAME, SEQ_LEN
import json
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

class CFG:
    """
    Configuration class for ASL recognition.

    Attributes:
        sequence_length (int): Length of the sequence used for recognition.
        rows_per_frame (int): Number of rows per frame in the image.
    """
    sequence_length: int = SEQ_LEN
    rows_per_frame: int = ROWS_PER_FRAME


mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities

def mediapipe_detection(image: np.ndarray, model: mp.solutions.holistic.Holistic) -> Tuple[np.ndarray, Any]:
    """
    Perform landmark detection using the Mediapipe library.

    Args:
        image (numpy.ndarray): Input image (BGR).
        model (mp.solutions.holistic.Holistic): Mediapipe holistic model instance.

    Returns:
        tuple: A tuple containing the processed image (BGR) and the prediction results.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion
    image.flags.writeable = False # img no longer writeable
    pred = model.process(image) # make landmark prediction
    image.flags.writeable = True  # img now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color reconversion
    return image, pred

def draw(image: np.ndarray, results: Any) -> None:
    """
    Draw landmarks on the image.

    Args:
        image (numpy.ndarray): Input image to draw on.
        results: Prediction results containing the landmarks from MediaPipe.
    """
    # Visualization Optimization: Render only hand landmarks to minimize CPU usage.
    
    # Draw Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
                             
    # Draw Right Hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_coordinates(results: Any) -> np.ndarray:
    """
    Extract coordinates from the prediction results.

    Args:
        results: Prediction results containing the landmarks.

    Returns:
        numpy.ndarray: Array of extracted coordinates concatenated [face, lh, pose, rh].
                       Shape is (N, 3) where N = 468+21+33+21 = 543.
    """
    try:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results and results.face_landmarks else np.zeros((468, 3)) * np.nan
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results and results.pose_landmarks else np.zeros((33, 3)) * np.nan
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results and results.left_hand_landmarks else np.zeros((21, 3)) * np.nan
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results and results.right_hand_landmarks else np.zeros((21, 3)) * np.nan
        
        return np.concatenate([face, lh, pose, rh])
    except Exception as e:
        # In case of malformed results or other unexpected errors, return a NaN array of expected shape
        # This is safer than the original bare except which masked all errors including NameErrors
        # However, for robustness in production, logging this error would be ideal.
        # print(f"Error extracting coordinates: {e}") 
        return np.zeros((468 + 21 + 33 + 21, 3)) * np.nan
    
def load_json_file(json_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return it as a dictionary.
    
    Args:
        json_path (str): Path to the JSON file
    
    Returns: 
        dict: Dictionary of loaded JSON content.
    """
    with open(json_path, 'r') as f:
        sign_map = json.load(f)
    return sign_map

if __name__ == '__main__':
    pass