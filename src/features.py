"""Geometric normalisation of hand landmarks.

Shared by the offline extraction path and the real-time demo. Two copies that
drifted apart would show a healthy validation score while the webcam predicted
nonsense, with nothing failing in between.
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

FEATURE_DIM = (NUM_LANDMARKS - 1) * 3
_MIN_SCALE = 1e-8


def normalise_image_landmarks(
    landmarks: np.ndarray, aspect_ratio: float = 1.0
) -> np.ndarray:
    """Normalise image-space landmarks into a translation- and scale-invariant vector.

    Args:
        landmarks: (21, 3) array as returned by MediaPipe, with x and y in [0, 1].
        aspect_ratio: image width divided by image height.

    Returns:
        A (60,) vector: 20 points, the wrist dropped once it becomes the origin.

    Raises:
        ValueError: the hand has no usable reference length.
    """
    points = np.asarray(landmarks, dtype=np.float64).copy()

    # x is normalised by image width and y by image height, so one unit of each
    # spans a different distance unless the frame is square. z shares x's scale.
    points[:, [0, 2]] *= aspect_ratio

    points -= points[WRIST]

    # Wrist -> middle-finger MCP is near-rigid, so it holds when fingers bend.
    # Measured in 2D because z is a noisy monocular estimate, and noise in a
    # divisor would spread to all 60 features.
    scale = float(np.linalg.norm(points[MIDDLE_FINGER_MCP, :2]))
    if scale < _MIN_SCALE:
        raise ValueError("degenerate hand: wrist and middle-finger MCP coincide")
    points /= scale

    return np.delete(points, WRIST, axis=0).ravel()


def normalise_world_landmarks(world_landmarks: np.ndarray) -> np.ndarray:
    """Normalise MediaPipe's metric landmarks into a scale-invariant vector.

    World landmarks are already in metres and centred on the hand, so neither the
    aspect-ratio correction nor an image-dependent origin applies.

    Args:
        world_landmarks: (21, 3) array of metric coordinates.

    Returns:
        A (60,) vector: 20 points, the wrist dropped once it becomes the origin.

    Raises:
        ValueError: the hand has no usable reference length.
    """
    points = np.asarray(world_landmarks, dtype=np.float64).copy()

    # Re-centred on the wrist rather than MediaPipe's hand centre, so both
    # variants share an origin and stay comparable.
    points -= points[WRIST]

    # Taken in 3D, z being a genuine depth here. Dividing by it removes hand-size
    # differences between signers, a shortcut the classifier would otherwise take
    # instead of learning the letters.
    scale = float(np.linalg.norm(points[MIDDLE_FINGER_MCP]))
    if scale < _MIN_SCALE:
        raise ValueError("degenerate hand: wrist and middle-finger MCP coincide")
    points /= scale

    return np.delete(points, WRIST, axis=0).ravel()


def features_from_dataframe(
    df: pd.DataFrame, mode: FeatureMode = "image", aspect_ratio: float = 1.0
) -> np.ndarray:
    """Build the (n, 60) feature matrix from rows of `landmarks.csv`.

    Args:
        df: rows holding the landmark columns; rows with no detected hand are
            expected to have been dropped already.
        mode: which landmark representation to normalise. The two are benchmarked
            rather than chosen a priori, MediaPipe not specifying the exact scale
            of its monocular z.
        aspect_ratio: width / height of the source images, for "image" mode.
    """
    if mode not in ("image", "world"):
        raise ValueError(f"unknown feature mode: {mode!r}")

    columns = LANDMARK_COLUMNS if mode == "image" else WORLD_LANDMARK_COLUMNS
    raw = df[columns].to_numpy(dtype=np.float64).reshape(-1, NUM_LANDMARKS, 3)

    # Looping costs a few seconds over the whole dataset, and buys the guarantee
    # that training features come from the same code as the webcam's.
    if mode == "image":
        rows = [normalise_image_landmarks(points, aspect_ratio) for points in raw]
    else:
        rows = [normalise_world_landmarks(points) for points in raw]
    return np.stack(rows)
