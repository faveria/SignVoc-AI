import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    SEQ_LEN: int = 30
    ROWS_PER_FRAME: int = 543
    MAX_LEN: int = 384
    CROP_LEN: int = 384
    NUM_CLASSES: int = 250
    PAD: float = -100.0
    THRESH_HOLD: float = 0.4
    
    # Espeak Path
    PHONEMIZER_ESPEAK_PATH: str = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    
    CHANNELS: int = field(init=False)
    NUM_NODES: int = field(init=False)

    def update_from_landmarks(self, landmarks_config):
        self.NUM_NODES = len(landmarks_config.POINT_LANDMARKS)
        self.CHANNELS = 6 * self.NUM_NODES

@dataclass
class LandmarksConfig:
    """Configuration for MediaPipe Landmarks Indices"""
    NOSE: List[int] = field(default_factory=lambda: [1, 2, 98, 327])
    LNOSE: List[int] = field(default_factory=lambda: [98])
    RNOSE: List[int] = field(default_factory=lambda: [327])
    LIP: List[int] = field(default_factory=lambda: [
        0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])
    LLIP: List[int] = field(default_factory=lambda: [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82])
    RLIP: List[int] = field(default_factory=lambda: [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312])
    
    POSE: List[int] = field(default_factory=lambda: [500, 502, 504, 501, 503, 505, 512, 513])
    LPOSE: List[int] = field(default_factory=lambda: [513,505,503,501])
    RPOSE: List[int] = field(default_factory=lambda: [512,504,502,500])
    
    REYE: List[int] = field(default_factory=lambda: [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173,
    ])
    LEYE: List[int] = field(default_factory=lambda: [
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398,
    ])
    
    LHAND: List[int] = field(default_factory=lambda: np.arange(468, 489).tolist())
    RHAND: List[int] = field(default_factory=lambda: np.arange(522, 543).tolist())
    
    POINT_LANDMARKS: List[int] = field(init=False)

    def __post_init__(self):
        self.POINT_LANDMARKS = self.LIP + self.LHAND + self.RHAND + self.NOSE + self.REYE + self.LEYE

# Global Instances to maintain backward compatibility (to some extent) or easy access
LANDMARKS = LandmarksConfig()
CONFIG = ModelConfig()
CONFIG.update_from_landmarks(LANDMARKS)

# Export variables to match old config.py interface for minimal breakage during refactor
THRESH_HOLD = CONFIG.THRESH_HOLD
SEQ_LEN = CONFIG.SEQ_LEN
ROWS_PER_FRAME = CONFIG.ROWS_PER_FRAME
MAX_LEN = CONFIG.MAX_LEN
CROP_LEN = CONFIG.CROP_LEN
NUM_CLASSES = CONFIG.NUM_CLASSES
PAD = CONFIG.PAD

NOSE = LANDMARKS.NOSE
LNOSE = LANDMARKS.LNOSE
RNOSE = LANDMARKS.RNOSE
LIP = LANDMARKS.LIP
LLIP = LANDMARKS.LLIP
RLIP = LANDMARKS.RLIP
POSE = LANDMARKS.POSE
LPOSE = LANDMARKS.LPOSE
RPOSE = LANDMARKS.RPOSE
REYE = LANDMARKS.REYE
LEYE = LANDMARKS.LEYE
LHAND = LANDMARKS.LHAND
RHAND = LANDMARKS.RHAND
POINT_LANDMARKS = LANDMARKS.POINT_LANDMARKS

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES

if __name__ == "__main__":
    print(f"NUM_NODES: {NUM_NODES}")
    print(f"CHANNELS: {CHANNELS}")
