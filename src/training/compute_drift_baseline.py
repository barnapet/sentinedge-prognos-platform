"""Compute and persist the drift-detection baseline (Issue #90, `docs/monitoring_design.md`).

`docs/monitoring_design.md` Sections 1-2 decided what this module produces: per-feature
(mean, std) pairs, computed once offline from the full pooled
`data/processed/training_dataset.parquet` (#67) -- every experiment, every label, together
-- for the four features Section 2 monitors for input drift: `rms`, `kurtosis`, `skewness`,
`skewness_smoothed`. `rms_ratio` is deliberately excluded: it is already normalized to that
bearing's own first-50-file baseline (`docs/feature_windowing_decision.md`), so a second,
population-level normalization on top would measure between-bearing severity differences
rather than sensor/environment drift -- Section 2's full reasoning.

Mirrors `src/training/train_serving_model.py`'s shape: a small, auditable, committed
artifact alongside `models/serving_model.joblib`, produced by a `main()` this project
already knows how to run and reproduce, not read out of a notebook cell. Unlike that
module, this one does not build a full provenance hash chain (`serving_code_hash` +
`combined_hash`) -- eight floats are directly human-diffable in the committed JSON, so the
proportionate check is "does this match its recorded source dataset version", not a
byte-level integrity hash the way an opaque pickled model needs (`docs/serving_model_artifact.md`
Section 3's reasoning does not transfer to a plain-text artifact this small).

Reproducing:

    python -m src.features.build_training_dataset   # Issue #67, if not already built
    python -m src.training.compute_drift_baseline
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.training.evaluation import TRAINING_DATASET_PATH, load_training_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
BASELINE_PATH = MODELS_DIR / "drift_baseline.json"
TRAINING_DATASET_MANIFEST_PATH = TRAINING_DATASET_PATH.parent / "training_dataset_manifest.json"

# docs/monitoring_design.md Section 2: rms/kurtosis/skewness/skewness_smoothed are
# unnormalized measurements, so a population baseline answers "does this look like
# training data". rms_ratio is excluded -- already bearing-relative by construction, see
# that section's full reasoning.
MONITORED_FEATURES = ["rms", "kurtosis", "skewness", "skewness_smoothed"]


def compute_feature_baseline(
    df: pd.DataFrame, features: list[str] = MONITORED_FEATURES
) -> dict[str, dict[str, float]]:
    """Per-feature (mean, std) over every row, every experiment, every label pooled together.

    `docs/monitoring_design.md` Section 1's "baseline: pooled training distribution, all
    labels included" decision -- `Degrading`/`Critical` rows are not excluded, because
    drift means "unlike anything training saw", not "unlike Normal" (that distinction is
    the classifier's own job, per Section 1's reasoning, not this baseline's).
    """
    return {
        feature: {"mean": float(df[feature].mean()), "std": float(df[feature].std())}
        for feature in features
    }


def read_training_dataset_version(manifest_path: Path = TRAINING_DATASET_MANIFEST_PATH) -> str:
    """Reused, not recomputed -- same pattern as
    `train_serving_model.read_training_dataset_version`."""
    return json.loads(manifest_path.read_text())["combined_hash"]


def build_baseline_manifest(
    df: pd.DataFrame,
    training_dataset_version: str | None = None,
    generated_at: datetime | None = None,
) -> dict:
    """What produced this baseline, and the eight numbers themselves.

    `training_dataset_version` defaults to reading Issue #67's manifest, and is injectable
    so tests can build one without `data/processed/`, which is gitignored and absent when
    CI's unit-test step runs -- the same constraint `tests/test_train_serving_model.py`
    works around.
    """
    if training_dataset_version is None:
        training_dataset_version = read_training_dataset_version()
    generated_at = generated_at or datetime.now(timezone.utc)
    return {
        "generated_at": generated_at.isoformat(),
        "training_dataset_version": training_dataset_version,
        "features": list(MONITORED_FEATURES),
        "n_rows": int(len(df)),
        "baseline": compute_feature_baseline(df),
    }


def persist_baseline(
    df: pd.DataFrame,
    path: Path = BASELINE_PATH,
    training_dataset_version: str | None = None,
) -> dict:
    """Write the baseline manifest, returning it."""
    manifest = build_baseline_manifest(df, training_dataset_version=training_dataset_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_drift_baseline(path: Path = BASELINE_PATH) -> dict[str, tuple[float, float]]:
    """The committed baseline, as `{feature: (mean, std)}` -- the shape
    `src.serving.drift` needs at request time."""
    manifest = json.loads(path.read_text())
    return {
        feature: (stats["mean"], stats["std"]) for feature, stats in manifest["baseline"].items()
    }


def format_report(manifest: dict) -> str:
    lines = [f"Computed drift baseline from {manifest['n_rows']} pooled training rows:"]
    for feature, stats in manifest["baseline"].items():
        lines.append(f"  {feature:<18} mean={stats['mean']:.6f}  std={stats['std']:.6f}")
    lines.append(f"  training_dataset_version: {manifest['training_dataset_version'][:16]}...")
    return "\n".join(lines)


def main() -> None:
    df = load_training_dataset()
    manifest = persist_baseline(df)
    print(format_report(manifest))
    print(f"\nwrote {BASELINE_PATH}")


if __name__ == "__main__":
    main()
