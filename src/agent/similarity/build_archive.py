"""Build `models/trajectory_archive.parquet` and its manifest (Issue #140,
`docs/agent_design.md` Section 12).

Section 12's reasoning for committing this at all, in one line: the trajectories
`find_similar_historical_pattern` compares against live in `data/processed/`, which is
gitignored and needs the 6.2 GB raw dataset to regenerate -- the same obstacle #86 solved by
committing real signal rather than optimizing a download.

**Source.** `data/processed/training_dataset.parquet` (#67), not the three
`data/processed/<exp>_features.parquet` files Issue #140's task text named. Those three
carry no `label` column -- their schema is `experiment`/`file_index`/`timestamp` plus the
five features -- and the training dataset *is* their labelled join, the same 9,464 rows,
already manifest-hashed. Re-deriving labels here would create a second labelling path that
could drift from the committed one; this reuses #67's, exactly as
`src/training/compute_drift_baseline.py` does.

**Two hashes, deliberately.** `archive_sha256` is over the parquet file's bytes and answers
"has the committed artifact been altered" -- the same integrity role
`models/serving_model_manifest.json`'s `model_sha256` plays. `content_sha256` is over a
canonical text serialization of the rows themselves and answers "is this the same *data*",
which the file hash cannot: parquet embeds the writer's version string, so a re-encode on a
different `pyarrow` build changes the file bytes without changing a single number. The
citable `source_id` Section 6 verifies against is derived from the content hash for that
reason -- an id that moved when someone re-encoded the file would make every prior citation
unverifiable for no real change.

Reproducing:

    python -m src.features.build_training_dataset     # Issue #67, if not already built
    python -m src.agent.similarity.build_archive
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.labeling import LABELS
from src.training.evaluation import (
    EXPERIMENTS,
    FEATURE_MATRIX_COLUMNS,
    TRAINING_DATASET_PATH,
    load_training_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"
ARCHIVE_PATH = MODELS_DIR / "trajectory_archive.parquet"
ARCHIVE_MANIFEST_PATH = MODELS_DIR / "trajectory_archive_manifest.json"
TRAINING_DATASET_MANIFEST_PATH = TRAINING_DATASET_PATH.parent / "training_dataset_manifest.json"

# Issue #140's task text: `experiment`, `file_index`, `label`, and the five feature columns.
# `timestamp` is dropped -- DTW compares by position in the sequence, and the archive's
# purpose is shape, not wall-clock.
ARCHIVE_COLUMNS = ["experiment", "file_index", "label", *FEATURE_MATRIX_COLUMNS]


def build_archive_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The archive's rows: every experiment, every file, in `(experiment, file_index)` order.

    Sorted explicitly rather than trusting the source's order, because DTW reads these rows
    as a *sequence* -- a shuffled archive would still load, still produce numbers, and be
    wrong in a way no shape check catches.

    Raw `rms` is carried even though Section 12's metric never reads it (that section's
    single most important exclusion: raw RMS amplitude does not transfer between bearings,
    so a distance dominated by it reports a scale finding as a shape finding). The archive
    is the trajectory record, and `archive.py` enforces the channel selection at query time
    -- one place, tested -- rather than by omitting a column here.
    """
    missing = [column for column in ARCHIVE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"training dataset is missing {missing}")
    out = df[ARCHIVE_COLUMNS].copy()
    out["label"] = out["label"].astype(str)
    out["experiment"] = out["experiment"].astype(str)
    return out.sort_values(["experiment", "file_index"], kind="stable").reset_index(drop=True)


def content_hash(frame: pd.DataFrame) -> str:
    """SHA-256 over a canonical text serialization of `frame`.

    `repr()` of a float is its shortest round-tripping decimal form, so this depends on the
    values alone -- not on `pyarrow`'s encoding, not on `pandas`' `to_string` formatting,
    and not on the dtype backing an integer column.
    """
    digest = hashlib.sha256()
    digest.update(("\t".join(ARCHIVE_COLUMNS) + "\n").encode())
    for row in frame.itertuples(index=False, name=None):
        digest.update(("\t".join(repr(value) for value in row) + "\n").encode())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_training_dataset_version(path: Path = TRAINING_DATASET_MANIFEST_PATH) -> str:
    """Reused, not recomputed -- same pattern as
    `compute_drift_baseline.read_training_dataset_version`."""
    return json.loads(path.read_text())["combined_hash"]


def build_manifest(
    frame: pd.DataFrame,
    archive_sha256: str,
    training_dataset_version: str | None = None,
    generated_at: datetime | None = None,
) -> dict:
    """What produced this archive, and what is in it.

    `training_dataset_version` is injectable so tests can build a manifest without
    `data/processed/`, which is gitignored and absent when CI's unit-test step runs -- the
    same constraint `tests/test_train_serving_model.py` and
    `tests/test_compute_drift_baseline.py` already work around.
    """
    if training_dataset_version is None:
        training_dataset_version = read_training_dataset_version()
    generated_at = generated_at or datetime.now(timezone.utc)
    return {
        "generated_at": generated_at.isoformat(),
        "source": str(TRAINING_DATASET_PATH.relative_to(REPO_ROOT)),
        "training_dataset_version": training_dataset_version,
        "archive_sha256": archive_sha256,
        "content_sha256": content_hash(frame),
        "columns": list(ARCHIVE_COLUMNS),
        "labels": list(LABELS),
        "n_rows": int(len(frame)),
        "n_rows_per_experiment": {
            experiment: int((frame["experiment"] == experiment).sum())
            for experiment in EXPERIMENTS
        },
        "class_support_per_experiment": {
            experiment: {
                label: int(((frame["experiment"] == experiment) & (frame["label"] == label)).sum())
                for label in LABELS
            }
            for experiment in EXPERIMENTS
        },
    }


def persist_archive(
    frame: pd.DataFrame,
    archive_path: Path = ARCHIVE_PATH,
    manifest_path: Path = ARCHIVE_MANIFEST_PATH,
    training_dataset_version: str | None = None,
) -> dict:
    """Write the parquet and the manifest beside it, returning the manifest.

    The parquet is written first so its hash goes into the manifest describing it, rather
    than the manifest claiming a hash for bytes that were never written.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(archive_path, index=False)
    manifest = build_manifest(
        frame,
        archive_sha256=file_hash(archive_path),
        training_dataset_version=training_dataset_version,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def format_report(manifest: dict, archive_path: Path = ARCHIVE_PATH) -> str:
    lines = [f"Wrote {manifest['n_rows']} archived trajectory rows:"]
    for experiment, count in manifest["n_rows_per_experiment"].items():
        support = manifest["class_support_per_experiment"][experiment]
        breakdown = "  ".join(f"{label}={support[label]}" for label in manifest["labels"])
        lines.append(f"  {experiment:<10} {count:>5} rows   {breakdown}")
    lines.append(f"  content_sha256:  {manifest['content_sha256'][:16]}...")
    lines.append(f"  archive_sha256:  {manifest['archive_sha256'][:16]}...")
    lines.append(f"  size on disk:    {archive_path.stat().st_size / 1024:.1f} KB")
    return "\n".join(lines)


def main() -> None:
    frame = build_archive_frame(load_training_dataset())
    manifest = persist_archive(frame)
    print(format_report(manifest))
    print(f"\nwrote {ARCHIVE_PATH}\nwrote {ARCHIVE_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
