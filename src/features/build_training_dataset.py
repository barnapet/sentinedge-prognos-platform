"""Join versioned feature parquets with labels into one M3-ready training dataset.

CLI entrypoint for Issue #67. Run from the repo root:

    python -m src.features.build_training_dataset

For each experiment already extracted by `build_dataset.py` (#41), reads
`data/processed/<name>_features.parquet`, derives that experiment's `critical_multiple`
from the parquet's own `rms_ratio` column via `src.labeling.derive_critical_multiple`
(#65), labels every row with `src.labeling.assign_labels` (#19/#20), and concatenates
all three into:

    data/processed/training_dataset.parquet
    data/processed/training_dataset_manifest.json

No dependency on `data/raw/`: `derive_critical_multiple` only needs each experiment's
peak `rms_ratio`, which the feature parquet already carries -- confirmed in #65's
pre-work check and re-confirmed here (see `docs/training_dataset_versioning.md`
Section 1). `rms_ratio` is kept as an ordinary output column, unchanged -- whether it
should be excluded from model training as circular with its own labeling basis is a
question for M3's training/ablation step, not this join (Issue #67 Task 3;
`docs/training_dataset_versioning.md` Section 3).

See `docs/training_dataset_versioning.md` Section 2 for why this output gets its own
manifest, deliberately independent of `src/features/versioning.py`'s
`GENERATING_CODE_FILES` (which stays scoped to `extraction.py`/`versioning.py`, per
Issue #65's constraint -- this module does not touch it).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features.extraction import EXPERIMENTS, FEATURE_COLUMNS
from src.features.versioning import compute_combined_hash, write_manifest
from src.labeling import LABELS, assign_labels, derive_critical_multiple

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TRAINING_DATASET_COLUMNS = FEATURE_COLUMNS + ["label", "label_pre_override", "override_applied"]

# The code whose content determines what this join *means*, independent of
# src/features/versioning.py's GENERATING_CODE_FILES (which governs the upstream
# feature parquets only -- see docs/training_dataset_versioning.md Section 2 for why
# these are kept as two separate hash chains rather than one merged one).
LABELING_CODE_FILES: tuple[Path, ...] = (
    REPO_ROOT / "src" / "labeling.py",
    REPO_ROOT / "src" / "features" / "build_training_dataset.py",
)


def compute_labeling_code_hash(code_files: tuple[Path, ...] = LABELING_CODE_FILES) -> str:
    """SHA-256 over the sorted, concatenated bytes of the given source files.

    Same scheme as `src.features.versioning.compute_code_hash`, over a different,
    independent file set -- see that function's docstring for why sorting first makes
    the result independent of argument order.
    """
    hasher = hashlib.sha256()
    for path in sorted(code_files):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def compute_upstream_feature_version(feature_manifests: dict[str, dict]) -> str:
    """SHA-256 fingerprint of exactly which upstream feature-parquet versions fed this join.

    Hashes `f"{experiment}:{combined_hash}"` for each experiment's feature manifest,
    sorted by experiment name. Reusing each upstream `combined_hash` (rather than
    re-reading `raw_dataset_version`/`code_hash` separately, or re-hashing the parquet
    bytes) means this fingerprint changes if *either* the extraction code or the raw
    dataset changed upstream (#41), without this module needing to know anything about
    how that upstream hash was computed, and without a second read pass over the
    parquet content.
    """
    hasher = hashlib.sha256()
    for experiment in sorted(feature_manifests):
        combined_hash = feature_manifests[experiment]["combined_hash"]
        hasher.update(f"{experiment}:{combined_hash}\n".encode())
    return hasher.hexdigest()


def build_training_dataset(processed_dir: Path = PROCESSED_DIR) -> Path:
    """Label and concatenate all three experiments' feature parquets into one dataset.

    Returns the path to the written parquet file. Writes an accompanying
    `training_dataset_manifest.json` alongside it (see module docstring and
    `docs/training_dataset_versioning.md`).
    """
    frames = []
    critical_multiples: dict[str, float] = {}
    feature_manifests: dict[str, dict] = {}
    per_experiment_n_files: dict[str, int] = {}

    for name in EXPERIMENTS:
        df = pd.read_parquet(processed_dir / f"{name}_features.parquet")
        feature_manifests[name] = json.loads(
            (processed_dir / f"{name}_features_manifest.json").read_text()
        )

        critical_multiple = derive_critical_multiple(df["rms_ratio"].max())
        critical_multiples[name] = critical_multiple

        labelled = assign_labels(df, critical_multiple)
        per_experiment_n_files[name] = len(labelled)
        frames.append(labelled)

    training_df = pd.concat(frames, ignore_index=True)

    parquet_path = processed_dir / "training_dataset.parquet"
    processed_dir.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(parquet_path, index=False)

    labeling_code_hash = compute_labeling_code_hash()
    upstream_feature_version = compute_upstream_feature_version(feature_manifests)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labeling_code_hash": labeling_code_hash,
        "upstream_feature_version": upstream_feature_version,
        "combined_hash": compute_combined_hash(labeling_code_hash, upstream_feature_version),
        "critical_multiple": critical_multiples,
        "labels": LABELS,
        "columns": TRAINING_DATASET_COLUMNS,
        "n_files": per_experiment_n_files,
        "n_files_total": len(training_df),
    }
    write_manifest(manifest, processed_dir / "training_dataset_manifest.json")

    return parquet_path


def main() -> None:
    path = build_training_dataset()
    df = pd.read_parquet(path)
    print(f"wrote {path} ({len(df)} rows)")
    for name, group in df.groupby("experiment", sort=False):
        counts = group["label"].value_counts().reindex(LABELS).to_dict()
        print(f"  {name}: {counts}")


if __name__ == "__main__":
    main()
