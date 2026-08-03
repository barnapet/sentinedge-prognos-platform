"""Regenerate the committed demo sample from the full raw dataset (Issue #86).

Run only when the sample needs rebuilding -- its output is committed, so a fresh clone
never runs this:

    python -m demo.build_sample        # requires data/raw/ and data/processed/

Same provenance discipline as `src/features/build_dataset.py` (#41) and
`src/training/train_serving_model.py` (#80): the artifact is written alongside a manifest
recording exactly what produced it, including the SHA-256 of the bytes written, so a
committed binary is auditable rather than opaque. What the sample *is* and why it was cut
this way: `demo/sample.py`'s module docstring.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from demo.sample import (
    MANIFEST_PATH,
    SAMPLE_EXPERIMENT,
    SAMPLE_PATH,
    SAMPLE_STEP,
    REPO_ROOT,
)
from src.features.extraction import EXPERIMENTS, list_snapshot_files, load_channel

RAW_DIR = REPO_ROOT / "data" / "raw"
TRAINING_DATASET_PATH = REPO_ROOT / "data" / "processed" / "training_dataset.parquet"


def select_sample_files(raw_dir: Path, step: int) -> list[Path]:
    """Every `step`-th snapshot, in chronological order, starting at file 0.

    Starting at file 0 is required, not cosmetic: `docs/serving_design.md` Section 1 has
    the server infer a bearing's position from arrival order, so a replay that began
    mid-stream would build its baseline from the wrong files.
    """
    return list_snapshot_files(raw_dir)[::step]


def read_labels(experiment: str, file_indices: np.ndarray) -> np.ndarray:
    """Committed ground-truth labels for the sampled files (display-only, see `sample.py`)."""
    df = pd.read_parquet(TRAINING_DATASET_PATH)
    labels = (
        df[df["experiment"] == experiment]
        .set_index("file_index")["label"]
        .astype(str)
        .loc[file_indices]
    )
    # Fixed-width unicode, not pandas' object dtype: an object array would force
    # `allow_pickle=True` on load, and a committed binary that can only be read by
    # unpickling is exactly the kind of artifact this repo avoids shipping.
    return labels.to_numpy().astype("U")


def build_sample(experiment: str = SAMPLE_EXPERIMENT, step: int = SAMPLE_STEP) -> dict:
    """Write the sample `.npz` and its manifest; return the manifest."""
    config = EXPERIMENTS[experiment]
    files = select_sample_files(RAW_DIR / experiment, step)
    file_indices = np.arange(0, len(list_snapshot_files(RAW_DIR / experiment)), step, dtype=np.int32)

    signals = np.stack([load_channel(path, config.channel_idx) for path in files])
    labels = read_labels(experiment, file_indices)

    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        experiment=experiment,
        channel_idx=config.channel_idx,
        signals=signals,
        file_indices=file_indices,
        filenames=np.array([path.name for path in files], dtype="U"),
        labels=labels,
    )
    data = payload.getvalue()

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_bytes(data)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "bearing": config.bearing_label,
        "channel_idx": config.channel_idx,
        "failure_mode": config.failure_mode,
        "step": step,
        "n_files_sampled": int(len(files)),
        "n_files_in_experiment": int(len(list_snapshot_files(RAW_DIR / experiment))),
        "samples_per_file": int(signals.shape[1]),
        "dtype": str(signals.dtype),
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "label_counts": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "source": "NASA IMS bearing dataset (see data/README.md)",
        "note": (
            "Every {step}th snapshot of {experiment}, tracked channel only. Each snapshot is "
            "the complete, unmodified 20,480-point recording as src/features/extraction.py's "
            "load_channel returns it -- only which files are present is reduced, never their "
            "contents. Labels are ground truth for display by demo/playback.py and are never "
            "sent to the server."
        ).format(step=step, experiment=experiment),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    manifest = build_sample()
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {SAMPLE_PATH} ({manifest['size_bytes'] / 1e6:.1f} MB)")
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
