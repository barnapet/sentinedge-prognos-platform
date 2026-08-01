"""Leave-one-experiment-out (LOEO) evaluation harness.

Implements the protocol committed to in `docs/evaluation_protocol.md` (Issue #69),
which was written before any model existed precisely so the split and the metrics could
not be chosen to flatter a result. This module is the executable form of that document
and deliberately adds no evaluation decisions of its own:

- `loeo_folds` implements Section 1's split: for each experiment, train on the other
  two experiments' rows in full, test on the held-out one. No row-level splitting
  within an experiment.
- `fold_metrics` computes Section 4's committed metrics: per-class recall and
  precision (headlined by `Critical` recall), macro-F1, and the full 3x3 confusion
  matrix.
- `aggregate` implements Section 5: all three fold values individually, plus mean and
  range -- explicitly *not* a standard deviation or confidence interval, which three
  points cannot support.

Feature columns are `FEATURE_MATRIX_COLUMNS` below. `rms_ratio` is included, unchanged:
whether its circularity with the labeling rule should exclude it is the Step 4 ablation
question (Issue #67 Task 3), not this module's to pre-empt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.labeling import LABELS

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATASET_PATH = REPO_ROOT / "data" / "processed" / "training_dataset.parquet"

# The model input columns, from the training dataset's feature half. `experiment` is the
# fold key (not a feature -- feeding it would hand the model the very grouping LOEO
# holds out), `file_index`/`timestamp` are positional/temporal identifiers that would let
# a model key on "how far into this run are we" rather than on signal content, and the
# label columns are the target.
FEATURE_MATRIX_COLUMNS = ["rms", "rms_ratio", "kurtosis", "skewness", "skewness_smoothed"]

EXPERIMENTS = ["1st_test", "2nd_test", "3rd_test"]


@dataclass(frozen=True)
class Fold:
    """One LOEO fold: everything but `held_out` trains, `held_out` tests."""

    held_out: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_training_dataset(path: Path = TRAINING_DATASET_PATH) -> pd.DataFrame:
    """Read the Issue #67 training dataset (`build_training_dataset.py`'s output)."""
    return pd.read_parquet(path)


def loeo_folds(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> list[Fold]:
    """Yield the three LOEO folds from `docs/evaluation_protocol.md` Section 1.

    Each experiment is wholly train or wholly test in a given fold, never both -- the
    filtering is exactly the document's `df[df.experiment != held_out]` /
    `df[df.experiment == held_out]`, using the `experiment` column Issue #43 added for
    this purpose.

    Args:
        df: The training dataset, with an `experiment` column and a `label` column.
        feature_columns: Model input columns; defaults to `FEATURE_MATRIX_COLUMNS`.
            Exposed so the Step 4 `rms_ratio` ablation can drop a column without
            reimplementing the split.

    Returns:
        One `Fold` per experiment, in `EXPERIMENTS` order.
    """
    feature_columns = feature_columns or FEATURE_MATRIX_COLUMNS
    folds = []
    for held_out in EXPERIMENTS:
        is_held_out = df["experiment"] == held_out
        train, test = df.loc[~is_held_out], df.loc[is_held_out]
        folds.append(
            Fold(
                held_out=held_out,
                X_train=train[feature_columns].to_numpy(),
                y_train=train["label"].astype(str).to_numpy(),
                X_test=test[feature_columns].to_numpy(),
                y_test=test["label"].astype(str).to_numpy(),
            )
        )
    return folds


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute `docs/evaluation_protocol.md` Section 4's committed metrics for one fold.

    `zero_division=0` throughout: a model that predicts a class zero times has undefined
    precision for it, and scoring that as 0 rather than raising is the honest reading --
    "never predicted Critical" is a real, reportable failure, not a missing value. The
    confusion matrix is returned in full (Section 4 asks for it explicitly) with rows =
    true, columns = predicted, in `LABELS` order.
    """
    per_class_recall = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_class_precision = precision_score(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    return {
        "recall": {label: float(v) for label, v in zip(LABELS, per_class_recall)},
        "precision": {label: float(v) for label, v in zip(LABELS, per_class_precision)},
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "support": {label: int((y_true == label).sum()) for label in LABELS},
    }


def aggregate(values: list[float]) -> dict:
    """Summarise a metric across the three folds, per Section 5.

    Mean and range (min/max) only -- deliberately no standard deviation or confidence
    interval. Section 5's reasoning: over three points those imply a sampling
    distribution three points cannot characterise, the same argument
    `docs/uncertainty_quantification.md` makes for the n=3 M4 serving fallback. Callers
    are expected to report `per_fold` alongside this, never the summary alone.
    """
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }
