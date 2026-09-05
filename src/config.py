"""Paths and dataset constants shared across the project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

LANDMARKS_CSV = PROCESSED_DIR / "landmarks.csv"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"

# MediaPipe >= 1.0 removed the legacy `mp.solutions.hands` API; the Tasks API
# used instead loads its weights from an external .task bundle.
HAND_LANDMARKER_TASK = MODELS_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_FINGER_MCP = 9

CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

META_COLUMNS = ["path", "label", "detected", "handedness", "hand_score"]
# Interleaved as x0,y0,z0,x1,... so a row slice reshapes straight to (21, 3).
LANDMARK_COLUMNS = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]
WORLD_LANDMARK_COLUMNS = [
    f"w{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]
CSV_COLUMNS = META_COLUMNS + LANDMARK_COLUMNS + WORLD_LANDMARK_COLUMNS
