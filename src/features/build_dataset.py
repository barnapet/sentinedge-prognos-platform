"""Build versioned RMS/kurtosis feature parquet files for all three experiments.

CLI entrypoint for Issue #41. Run from the repo root:

    python -m src.features.build_dataset

For each experiment in `EXPERIMENTS`, reads `data/raw/<name>/`, computes features
via `extract_experiment_features`, and writes:

    data/processed/<name>_features.parquet
    data/processed/<name>_features_manifest.json

Re-running with unchanged code and an unchanged raw dataset reproduces the same
`combined_hash` in the manifest (see `src/features/versioning.py`).
"""
from __future__ import annotations

from pathlib import Path

from src.features.extraction import EXPERIMENTS, FEATURE_COLUMNS, extract_experiment_features
from src.features.versioning import (
    build_manifest,
    compute_code_hash,
    compute_raw_dataset_version,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def build_experiment(name: str, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Extract features for one experiment and write its parquet + manifest.

    Returns the path to the written parquet file.
    """
    cfg = EXPERIMENTS[name]
    experiment_raw_dir = raw_dir / name

    df = extract_experiment_features(experiment_raw_dir, name, cfg.channel_idx)

    parquet_path = processed_dir / f"{name}_features.parquet"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    manifest = build_manifest(
        experiment=name,
        code_hash=compute_code_hash(),
        raw_dataset_version=compute_raw_dataset_version(experiment_raw_dir),
        feature_columns=FEATURE_COLUMNS,
        n_files=len(df),
    )
    write_manifest(manifest, processed_dir / f"{name}_features_manifest.json")

    return parquet_path


def main() -> None:
    for name in EXPERIMENTS:
        path = build_experiment(name)
        print(f"{name}: wrote {path}")


if __name__ == "__main__":
    main()
