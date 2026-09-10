"""Geometric normalisation of hand landmarks.

Shared by the offline extraction path and the real-time demo, so that training
and serving features cannot drift apart.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .config import (
    LANDMARK_COLUMNS,
    MIDDLE_FINGER_MCP,
    NUM_LANDMARKS,
    WORLD_LANDMARK_COLUMNS,
    WRIST,
)

FeatureMode = Literal["image", "world"]

_MIN_SCALE = 1e-8


def normalise_landmarks(
    landmarks: np.ndarray, aspect_ratio: float = 1.0
) -> np.ndarray:
    """Turn (21, 3) landmarks into a translation- and scale-invariant (60,) vector.

    `aspect_ratio` is the source image's width / height, and 1.0 for world
    landmarks, which are already metric. Raises ValueError on a degenerate hand.
    """
    points = np.asarray(landmarks, dtype=np.float64).copy()

    # z shares x's scale.
    points[:, [0, 2]] *= aspect_ratio

    points -= points[WRIST]

    # Near-rigid, so it holds when fingers bend. In 3D: the 2D projection
    # collapses when the hand points at the camera.
    scale = float(np.linalg.norm(points[MIDDLE_FINGER_MCP]))
    if scale < _MIN_SCALE:
        raise ValueError("degenerate hand: wrist and middle-finger MCP coincide")
    points /= scale

    # The wrist is (0, 0, 0) after re-centring.
    return np.delete(points, WRIST, axis=0).ravel()


def features_from_dataframe(
    df: pd.DataFrame, mode: FeatureMode = "image", aspect_ratio: float = 1.0
) -> np.ndarray:
    """Build the (n, 60) feature matrix from `landmarks.csv` rows.

    Rows with no detected hand must have been dropped already.
    """
    if mode not in ("image", "world"):
        raise ValueError(f"unknown feature mode: {mode!r}")

    columns = LANDMARK_COLUMNS if mode == "image" else WORLD_LANDMARK_COLUMNS
    ratio = aspect_ratio if mode == "image" else 1.0
    raw = df[columns].to_numpy(dtype=np.float64).reshape(-1, NUM_LANDMARKS, 3)

    # Loop, not vectorised: same code path as the webcam.
    return np.stack([normalise_landmarks(points, ratio) for points in raw])
