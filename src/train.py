"""Train a classifier on the extracted hand landmarks.

Compares the two landmark representations on a split holding out whole signers,
then re-runs the winner on a random split to show what the latter overstates.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CLASSIFIER_PATH,
    CONFUSION_MATRIX_PATH,
    LANDMARKS_CSV,
    METRICS_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    STATIC_LETTERS,
)
from .features import features_from_dataframe

FEATURE_MODES = ("image", "world")
VALIDATION_SIGNERS = 2
TEST_SIGNERS = 2
MAX_EPOCHS = 300
PATIENCE = 15
LATENCY_SAMPLES = 200


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the landmark table, keeping detected rows of the static letters."""
    # Digit class folders make labels look numeric, so the dtype is pinned.
    df = pd.read_csv(path, dtype={"path": str, "label": str})
    df = df[(df["detected"] == 1) & (df["label"].isin(STATIC_LETTERS))].copy()
    df["signer"] = df["path"].str.extract(r"(?:^|/)P(\d+)_", expand=False).astype(int)
    return df.reset_index(drop=True)


def split_by_signer(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Assign whole signers to train, validation and test."""
    signers = np.random.default_rng(seed).permutation(np.unique(df["signer"]))
    held = TEST_SIGNERS + VALIDATION_SIGNERS
    groups = {
        "test": signers[:TEST_SIGNERS],
        "validation": signers[TEST_SIGNERS:held],
        "train": signers[held:],
    }
    print(
        "signers  "
        + "  ".join(f"{k}={sorted(int(s) for s in v)}" for k, v in groups.items())
    )
    return {k: np.flatnonzero(df["signer"].isin(v)) for k, v in groups.items()}


def split_at_random(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Split rows at random, ignoring who signed them."""
    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}

    for _, group in df.groupby("label", sort=True):
        positions = rng.permutation(group.index.to_numpy())
        first = round(len(positions) * 0.6)
        second = round(len(positions) * 0.8)
        parts["train"].append(positions[:first])
        parts["validation"].append(positions[first:second])
        parts["test"].append(positions[second:])

    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


SPLITS = {"signer": split_by_signer, "random": split_at_random}


def fit_mlp(X_train, y_train, X_val, y_val, seed: int) -> Pipeline:
    """Fit an MLP, early-stopping on the held-out validation set."""
    scaler = StandardScaler().fit(X_train)
    train_scaled, val_scaled = scaler.transform(X_train), scaler.transform(X_val)
    # One epoch per fit, so early stopping can score on our validation signers
    # rather than on sklearn's random slice of the training set.
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128), max_iter=1, warm_start=True, random_state=seed
    )

    best_score, best_weights, waited = -np.inf, None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(MAX_EPOCHS):
            mlp.fit(train_scaled, y_train)
            score = f1_score(y_val, mlp.predict(val_scaled), average="macro")
            if score > best_score:
                best_score, waited = score, 0
                best_weights = (
                    [w.copy() for w in mlp.coefs_],
                    [b.copy() for b in mlp.intercepts_],
                )
            else:
                waited += 1
                if waited >= PATIENCE:
                    break

    mlp.coefs_, mlp.intercepts_ = best_weights
    return Pipeline([("scaler", scaler), ("model", mlp)])


def evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predictions = model.predict(X)

    # Timed one row at a time, which is the webcam's regime rather than a batch.
    sample = X[:LATENCY_SAMPLES]
    start = time.perf_counter()
    for row in sample:
        model.predict(row.reshape(1, -1))
    latency = (time.perf_counter() - start) / len(sample) * 1000

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "latency_ms": round(latency, 2),
    }


def run(
    df: pd.DataFrame, X: np.ndarray, features: str, strategy: str, seed: int
) -> tuple[dict, Pipeline, np.ndarray, np.ndarray]:
    """Train one configuration and score it on validation and test."""
    y = df["label"].to_numpy()
    index = SPLITS[strategy](df, seed)
    model = fit_mlp(
        X[index["train"]],
        y[index["train"]],
        X[index["validation"]],
        y[index["validation"]],
        seed,
    )

    result = {
        "features": features,
        "split": strategy,
        "n_train": len(index["train"]),
        "validation": evaluate(model, X[index["validation"]], y[index["validation"]]),
        "test": evaluate(model, X[index["test"]], y[index["test"]]),
    }
    return result, model, y[index["test"]], model.predict(X[index["test"]])


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'features':<9} {'split':<8} {'val F1':>7} {'test F1':>8} {'test acc':>9}")
    for row in rows:
        print(
            f"{row['features']:<9} {row['split']:<8}"
            f" {row['validation']['macro_f1']:>7.3f}"
            f" {row['test']['macro_f1']:>8.3f}"
            f" {row['test']['accuracy']:>9.3f}"
        )


def save_confusion_matrix(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=STATIC_LETTERS,
        normalize="true",
        include_values=False,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Normalised confusion matrix, unseen signers")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LANDMARKS_CSV)
    parser.add_argument("--features", choices=FEATURE_MODES, help="restrict to one mode")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_dataset(args.input)
    print(f"{len(df)} rows, {df['label'].nunique()} classes, {df['signer'].nunique()} signers")

    ratios = np.unique(df["width"] / df["height"])
    if len(ratios) > 1:
        raise SystemExit("mixed aspect ratios: features_from_dataframe takes only one")

    modes = [args.features] if args.features else list(FEATURE_MODES)
    selection, best = [], None
    for mode in modes:
        X = features_from_dataframe(df, mode=mode, aspect_ratio=float(ratios[0]))
        result, model, y_true, y_pred = run(df, X, mode, "signer", args.seed)
        selection.append(result)
        # Chosen on validation; test stays untouched until the winner is known.
        if best is None or result["validation"]["macro_f1"] > best[0]["validation"]["macro_f1"]:
            best = (result, model, y_true, y_pred, X)

    print_table("landmark representation (unseen signers)", selection)

    result, model, y_true, y_pred, X = best
    random_result, *_ = run(df, X, result["features"], "random", args.seed)
    print_table("winning representation, both splits", [result, random_result])

    print("\nper-class report, unseen signers")
    print(classification_report(y_true, y_pred, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CLASSIFIER_PATH)
    METRICS_PATH.write_text(
        json.dumps({"selection": selection, "splits": [result, random_result]}, indent=2)
    )
    save_confusion_matrix(y_true, y_pred, CONFUSION_MATRIX_PATH)
    print(f"\nsaved {CLASSIFIER_PATH.name}, {METRICS_PATH.name}, {CONFUSION_MATRIX_PATH.name}")


if __name__ == "__main__":
    main()
