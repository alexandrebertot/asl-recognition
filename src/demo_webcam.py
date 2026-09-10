"""Real-time ASL letter recognition from a webcam."""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from .config import CLASSIFIER_PATH, HAND_CONNECTIONS
from .features import normalise_landmarks
from .landmarks import HandDetection, HandLandmarkExtractor

QUIT_KEYS = frozenset({ord("q"), 27})
GREEN = (80, 220, 80)
ORANGE = (40, 140, 255)
WHITE = (255, 255, 255)


def stable_letter(recent: deque[str | None]) -> str | None:
    """Return the letter holding a majority over the recent frames, else None."""
    votes = Counter(letter for letter in recent if letter is not None)
    if not votes:
        return None
    letter, count = votes.most_common(1)[0]
    return letter if count > recent.maxlen // 2 else None


def draw_hand(frame: np.ndarray, detection: HandDetection) -> None:
    """Draw the landmark skeleton, which tells a bad detection from a bad guess."""
    height, width = frame.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in detection.landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], GREEN, 2)
    for point in points:
        cv2.circle(frame, point, 4, ORANGE, -1)


def draw_overlay(
    frame: np.ndarray, letter: str | None, confidence: float, fps: float
) -> None:
    cv2.putText(
        frame, letter or "-", (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 3.0, WHITE, 6
    )
    caption = f"{confidence:.0%}" if letter else "no hand"
    cv2.putText(
        frame, caption, (24, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2
    )
    cv2.putText(
        frame,
        f"{fps:.0f} fps  |  q to quit",
        (24, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=CLASSIFIER_PATH)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="below this, no letter is shown rather than a confident guess",
    )
    parser.add_argument("--smoothing", type=int, default=5, help="frames voting")
    args = parser.parse_args()

    model = joblib.load(args.model)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    recent: deque[str | None] = deque(maxlen=args.smoothing)
    frame_times: deque[float] = deque(maxlen=30)
    start, last_timestamp = time.perf_counter(), -1

    with HandLandmarkExtractor(video_mode=True) as extractor:
        while True:
            read, frame = capture.read()
            if not read:
                break
            tick = time.perf_counter()

            # MediaPipe requires strictly increasing timestamps, and two frames
            # can land in the same millisecond.
            timestamp = max(int((tick - start) * 1000), last_timestamp + 1)
            last_timestamp = timestamp
            detection = extractor.detect(frame, timestamp)

            confidence = 0.0
            if detection is None:
                recent.append(None)
            else:
                height, width = frame.shape[:2]
                # Webcam frames are not square, unlike the training images.
                features = normalise_landmarks(detection.landmarks, width / height)
                probabilities = model.predict_proba(features.reshape(1, -1))[0]
                best = int(probabilities.argmax())
                confidence = float(probabilities[best])
                recent.append(
                    model.classes_[best] if confidence >= args.min_confidence else None
                )
                draw_hand(frame, detection)

            frame_times.append(time.perf_counter() - tick)
            draw_overlay(
                frame, stable_letter(recent), confidence, 1 / np.mean(frame_times)
            )
            cv2.imshow("ASL alphabet", frame)
            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
