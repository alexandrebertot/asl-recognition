"""Extract MediaPipe hand landmarks from a folder of images into a CSV.

A full pass over the dataset takes tens of minutes, so its output is materialised
once and every later step reads the CSV instead of the images.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    CSV_COLUMNS,
    IMAGE_SUFFIXES,
    LANDMARK_COLUMNS,
    LANDMARKS_CSV,
    WORLD_LANDMARK_COLUMNS,
)
from .landmarks import HandLandmarkExtractor


def iter_images(
    input_dir: Path, limit_per_class: int | None = None
) -> Iterator[tuple[Path, str]]:
    """Yield (image path, label) pairs from a directory holding one folder per class.

    Sorted at both levels so that two runs produce identical CSVs; a dataset whose
    row order shifts would make model comparisons meaningless.
    """
    for class_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        images = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        yield from ((path, class_dir.name) for path in images[:limit_per_class])


def build_row(
    path: Path, label: str, root: Path, extractor: HandLandmarkExtractor
) -> dict[str, object]:
    """Build one CSV row, landmark columns left empty when no hand was found.

    Undetected images are kept rather than dropped: they are what makes the
    detection rate measurable, per class as well as overall. Training filters them
    out when it loads the CSV.

    Args:
        root: paths are stored relative to it, keeping the CSV portable.
    """
    row: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "label": label,
        "detected": 0,
        "handedness": "",
        "hand_score": np.nan,
    }

    image = cv2.imread(str(path))
    if image is None:
        row["width"] = row["height"] = 0
        return row

    row["height"], row["width"] = image.shape[:2]
    detection = extractor.detect(image)
    if detection is None:
        return row

    row["detected"] = 1
    row["handedness"] = detection.handedness
    row["hand_score"] = detection.score
    row.update(zip(LANDMARK_COLUMNS, detection.landmarks.ravel()))
    row.update(zip(WORLD_LANDMARK_COLUMNS, detection.world_landmarks.ravel()))
    return row


def report(df: pd.DataFrame) -> None:
    """Print the detection rate, overall and for the classes that fare worst."""
    print(f"\nhand detected on {df['detected'].mean():.1%} of {len(df)} images")
    per_class = df.groupby("label")["detected"].mean().sort_values()
    print("\nlowest detection rates")
    for label, rate in per_class.head(5).items():
        print(f"  {label:>7}  {rate:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="directory holding one subdirectory per class",
    )
    parser.add_argument("--output", type=Path, default=LANDMARKS_CSV)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        help="images per class; use a small value to smoke-test the run in seconds",
    )
    args = parser.parse_args()

    pairs = list(iter_images(args.input_dir, args.limit_per_class))
    if not pairs:
        raise SystemExit(f"no images found under {args.input_dir}")

    with HandLandmarkExtractor() as extractor:
        rows = [
            build_row(path, label, args.input_dir, extractor)
            for path, label in tqdm(pairs, unit="img")
        ]

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nwrote {args.output}")
    report(df)


if __name__ == "__main__":
    main()
