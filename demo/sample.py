"""The committed demo sample: what it is, and how to load it (Issue #86).

`docs/PRD.md` Section 10 asks for "fresh `git clone` -> running demo in under 15 minutes."
The raw dataset cannot serve that: it is a 1.1 GB download that expands to 6.2 GB, needs
`unrar` and `py7zr` (neither ships by default), and `data/*` is fully gitignored --
`docs/serving_model_artifact.md` already flagged that committing the model artifact clears
only one of the two blockers. This module is the other half: a small, committed slice of
real raw signal, enough to drive a full, honest demo without downloading anything.

**What the sample is.** Every 5th snapshot of `2nd_test`, tracked channel only (channel 0,
`EXPERIMENTS["2nd_test"]`), 197 of 984 files. Each snapshot is the **complete, unmodified
20,480-point recording** as `src/features/extraction.py`'s `load_channel` returns it --
same `float32` dtype, same values, nothing downsampled, truncated, or synthesised within a
window. What the decimation changes is only *which* snapshots are present, never their
contents.

**Why `2nd_test`.** It is the smallest experiment (984 files) and has by far the best label
balance for a demo: 66% `Normal` / 31% `Degrading` / 2% `Critical`, versus `3rd_test`'s 97%
`Normal` (`docs/eda_findings.md`, and the counts in this sample's manifest). A replay
therefore shows a real progression rather than a flat line. It is also one of the two
experiments the model is measured to handle (`Critical` recall 0.913 under LOEO,
`docs/model_training_decision.md` Section 6).

**Why no `1st_test` sample**, despite it being the experiment whose failure mode
`docs/serving_design.md` Section 4's disclosure is about: decimating it would manufacture a
misleading result. Measured while choosing this sample (see the PR for Issue #86), replaying
`1st_test` at full resolution has the pooled model catching 16 of its 17 `Critical` files,
but at 1-in-10 decimation that collapses to 0 of 2 -- because that bearing's degradation is
impulsive and spiky (`docs/feature_windowing_decision.md` Section 2), so skipping files
throws away the very transients kurtosis keys on. A reviewer watching that would see what
looks like the documented `1st_test` limitation reproducing live, when it is really an
artifact of the sampling. The limitation is disclosed on every single response through
`model_notes` (Section 4's decision), which is the honest channel for it; anyone wanting to
watch a real `1st_test` replay can point `demo/playback.py` at the full dataset
(`--raw-dir`), which is exactly what that flag is for.

Regenerate with `python -m demo.build_sample` (requires the full raw dataset).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "demo" / "sample_data"
SAMPLE_PATH = SAMPLE_DIR / "2nd_test_sample.npz"
MANIFEST_PATH = SAMPLE_DIR / "2nd_test_sample_manifest.json"

SAMPLE_EXPERIMENT = "2nd_test"
# Every 5th file: 197 snapshots, ~6 MB compressed, and a replay whose predicted-label
# progression still tracks the full-resolution one (98.5% vs 99.0% agreement with the
# committed labels -- measured, see the PR for Issue #86).
SAMPLE_STEP = 5


@dataclass(frozen=True)
class SampleRun:
    """One bearing's replayable history: the signals, and what they are.

    `labels` are the committed ground-truth health states for these files
    (`src/labeling.py`, via `data/processed/training_dataset.parquet`). They are carried
    **for client-side display only** -- `demo/playback.py` prints them next to the
    prediction so a viewer can see where the model agrees and where it does not. They are
    never sent to the server; `docs/serving_design.md` Section 1's payload is a raw signal
    and a `bearing_id`, nothing else.
    """

    experiment: str
    channel_idx: int
    signals: np.ndarray  # (n_files, 20480) float32
    file_indices: np.ndarray  # index within the full experiment
    filenames: np.ndarray  # original snapshot filenames (timestamps)
    labels: np.ndarray  # ground truth, display-only

    def __len__(self) -> int:
        return len(self.signals)


def load_sample(path: Path = SAMPLE_PATH) -> SampleRun:
    """Load the committed sample. No dataset download required."""
    with np.load(path, allow_pickle=False) as data:
        return SampleRun(
            experiment=str(data["experiment"]),
            channel_idx=int(data["channel_idx"]),
            signals=data["signals"],
            file_indices=data["file_indices"],
            filenames=data["filenames"],
            labels=data["labels"],
        )
