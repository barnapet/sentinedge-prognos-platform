"""Combined code+raw-dataset versioning for `src/features` parquet outputs (Issue #41).

Implements the "rough direction" flagged in `docs/PRD.md` Section 12: hash the
generating code + raw data version, and keep a manifest recording what produced each
cache. Both pieces are combined here rather than choosing one -- see
`docs/feature_extraction_versioning.md` for the full rationale behind the specific
hash scheme and manifest schema chosen (in particular: why the raw-dataset "version"
fingerprints filenames+sizes rather than full file content, and why labels are not
part of this manifest/output).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Source files whose content changes the meaning of a feature output. Sorted before
# hashing so the result doesn't depend on argument order.
GENERATING_CODE_FILES: tuple[Path, ...] = (
    REPO_ROOT / "src" / "features" / "extraction.py",
    REPO_ROOT / "src" / "features" / "versioning.py",
)


def compute_code_hash(code_files: tuple[Path, ...] = GENERATING_CODE_FILES) -> str:
    """SHA-256 over the sorted, concatenated bytes of the given source files."""
    hasher = hashlib.sha256()
    for path in sorted(code_files):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def compute_raw_dataset_version(raw_dir: Path) -> str:
    """SHA-256 fingerprint of a raw experiment directory's contents.

    Fingerprints (filename, size in bytes) for every file in `raw_dir`, sorted by
    filename -- not full file content. See `docs/feature_extraction_versioning.md`
    for why: this is a fixed, publicly archived, immutable dataset (NASA IMS), and
    the realistic ways it could "change" (wrong file count, truncated download,
    wrong archive layout) all change either the file listing or a file's size. A
    same-size silent content mutation would not be caught by this fingerprint; that
    trade-off is deliberate, not an oversight, given a full content hash of this
    dataset's ~6 GB would cost several minutes per run.
    """
    hasher = hashlib.sha256()
    for path in sorted(raw_dir.iterdir(), key=lambda p: p.name):
        hasher.update(f"{path.name}:{path.stat().st_size}\n".encode())
    return hasher.hexdigest()


def compute_combined_hash(code_hash: str, raw_dataset_version: str) -> str:
    """Single hash identifying "this exact code, run against this exact raw data"."""
    return hashlib.sha256(f"{code_hash}:{raw_dataset_version}".encode()).hexdigest()


def build_manifest(
    experiment: str,
    code_hash: str,
    raw_dataset_version: str,
    feature_columns: list[str],
    n_files: int,
    generated_at: datetime | None = None,
) -> dict:
    """Build the manifest dict recording how a feature parquet was produced.

    Fields cover Issue #41's acceptance criteria: the combined hash, generation
    timestamp, source dataset version, and the feature columns included in the
    accompanying parquet output.
    """
    generated_at = generated_at or datetime.now(timezone.utc)
    return {
        "experiment": experiment,
        "generated_at": generated_at.isoformat(),
        "code_hash": code_hash,
        "raw_dataset_version": raw_dataset_version,
        "combined_hash": compute_combined_hash(code_hash, raw_dataset_version),
        "feature_columns": list(feature_columns),
        "n_files": n_files,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
