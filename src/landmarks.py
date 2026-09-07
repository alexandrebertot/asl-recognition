"""MediaPipe Hand Landmarker wrapper.

The only module that talks to MediaPipe, so an API change stays contained here
-- as already happened when 1.0 dropped the legacy `mp.solutions.hands`.
"""

from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import mediapipe as mp
import numpy as np

from .config import HAND_LANDMARKER_TASK, HAND_LANDMARKER_URL

_BaseOptions = mp.tasks.BaseOptions
_HandLandmarker = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode


@dataclass(frozen=True)
class HandDetection:
    """A single detected hand, as NumPy arrays so no MediaPipe type escapes."""

    landmarks: np.ndarray  # (21, 3), image-normalised
    world_landmarks: np.ndarray  # (21, 3), metric, centred on the hand
    handedness: str  # "Left" or "Right"
    score: float


def ensure_model(
    path: Path = HAND_LANDMARKER_TASK, url: str = HAND_LANDMARKER_URL
) -> Path:
    """Return the landmarker bundle, downloading it on first use."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    # Rename once complete: an interrupted download must not leave a truncated
    # bundle, which MediaPipe loads before failing much later.
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as f:
        shutil.copyfileobj(response, f)
    partial.replace(path)
    return path


class HandLandmarkExtractor:
    """Detects one hand per frame and returns its landmarks.

    Building the landmarker loads ~8 MB of weights, so an instance is meant to be
    reused across frames. `video_mode` selects MediaPipe's VIDEO mode, which
    tracks the hand between frames: right for a webcam, wrong for a folder of
    unrelated images.
    """

    def __init__(
        self,
        *,
        video_mode: bool = False,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self.video_mode = video_mode
        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=_RunningMode.VIDEO if video_mode else _RunningMode.IMAGE,
            num_hands=num_hands,  # fingerspelling is one-handed
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = _HandLandmarker.create_from_options(options)

    def detect(
        self, bgr_image: np.ndarray, timestamp_ms: int | None = None
    ) -> HandDetection | None:
        """Detect a hand in a BGR image, or return None if there is none.

        A missing hand is an ordinary outcome rather than an error, which keeps
        the 87k-image extraction loop free of a try/except. `timestamp_ms` is
        required in video mode, where MediaPipe uses it to order its tracking.
        """
        if self.video_mode and timestamp_ms is None:
            raise ValueError("timestamp_ms is required in video mode")

        # cv2 decodes to BGR, MediaPipe expects RGB.
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.video_mode:
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        else:
            result = self._landmarker.detect(image)

        if not result.hand_landmarks:
            return None

        handedness = result.handedness[0][0]
        return HandDetection(
            landmarks=_to_array(result.hand_landmarks[0]),
            world_landmarks=_to_array(result.hand_world_landmarks[0]),
            handedness=handedness.category_name,
            score=handedness.score,
        )

    def close(self) -> None:
        """Release the native resources held by MediaPipe."""
        self._landmarker.close()

    def __enter__(self) -> HandLandmarkExtractor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _to_array(landmarks) -> np.ndarray:
    """Stack MediaPipe landmark objects into a (21, 3) array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
