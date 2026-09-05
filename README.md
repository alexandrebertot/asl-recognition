# ASL Alphabet Recognition

Real-time American Sign Language alphabet recognition from a webcam. MediaPipe hand
landmarks feed a lightweight classifier, running on CPU.

> 🚧 Work in progress — sections marked 🚧 will be filled with measured numbers.

<!-- demo.gif -->

## Problem

Fingerspelling is how ASL signers spell names and words that have no dedicated sign,
which makes it the entry point to any sign language interface. This project classifies
the 26 static letters (plus `space`, `del` and `nothing`) from a single frame, at
interactive frame rates, on CPU.

## Approach

```
                     ┌──────────────────┐     ┌────────────┐
  image / webcam ──> │    MediaPipe     │ ──> │ normalised │ ──> classifier ──> letter
      frame          │ Hand Landmarker  │     │  features  │
                     └──────────────────┘     └────────────┘
                       21 × (x, y, z)            63 values
                          (frozen)                (shared)
```

MediaPipe is used as a **frozen feature extractor**: each frame is reduced to 21 hand
landmarks, and only the final classifier is trained.

| | CNN on raw pixels | MediaPipe landmarks |
| --- | --- | --- |
| Model input | 200×200×3 = 120,000 values | **63 values** |
| What the model sees | hand **+ background + lighting + skin tone** | hand geometry only |
| Hardware | GPU, hours | **CPU, minutes** |

The middle row is the decisive one. The training images share a single signer and a
single background, so a pixel-based model would partly learn the *scene* and collapse on
a different webcam. Landmarks discard everything that is not hand geometry by
construction — exactly the invariance this task needs.

Landmark extraction over the full dataset takes ~30 minutes, so it is materialised once
into `landmarks.csv` and training then runs in seconds. Feature normalisation lives in a
single function shared by the offline and real-time paths, which rules out
training/serving skew by construction.

## Dataset

[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Kaggle) —
87,000 images of 200×200 pixels across 29 classes.

The images come from a single signer and are near-consecutive video frames, so a random
train/test split places near-duplicates on both sides and badly overstates accuracy.
Results below are therefore reported on an independent set of signers.

## Results

🚧 Planned: accuracy and macro F1, per-class scores, confusion matrix, hand detection
rate, and inference latency.

## Limitations

- **Static letters only.** `J` and `Z` involve motion and cannot be recognised from a
  single frame.
- **MediaPipe is a hard ceiling.** Being frozen, it cannot be fine-tuned, so any pose it
  mis-estimates is unrecoverable. Letters that tuck the thumb under the fingers (`M`,
  `N`, `S`, `T`) are occluded and expected to dominate the confusions.
- **Single-signer training data**, with the generalisation consequences described above.

## Installation

Requires Python 3.14.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

MediaPipe weights are downloaded automatically on first run.

## Usage

```bash
python -m src.build_dataset --input-dir data/raw/asl_alphabet_train
python -m src.train --model mlp
python -m src.demo_webcam
```

## Layout

```
src/
├── config.py           paths, dataset constants, CSV schema
├── landmarks.py        MediaPipe wrapper: image -> 21 landmarks
├── features.py         landmarks -> normalised feature vector (shared)
├── build_dataset.py    images -> data/processed/landmarks.csv
├── train.py            landmarks.csv -> classifier + reports/
└── demo_webcam.py      real-time webcam demo
```

`reports/` is tracked so metrics stay readable without cloning the dataset; `data/` and
model binaries are not.

## Roadmap

- [ ] Landmark extraction pipeline
- [ ] Training and evaluation, MLP vs histogram gradient boosting
- [ ] Real-time webcam demo
- [ ] Cross-dataset evaluation on unseen signers
- [ ] *Optional:* temporal model for the dynamic letters `J` and `Z`
- [ ] *Optional:* packaging via `pyproject.toml`, warranted once tests are added
