"""Crest factor: a time-domain feature evaluated and rejected in Issue #23.

`docs/eda_findings.md` Section 4 flagged crest factor "plausible, low priority" and
already suspected redundancy with kurtosis (`corr` 0.57-0.88 in the Degrading+Critical
window). Issue #23 confirmed this: crest factor correlates 0.56-0.88 with kurtosis
across all three experiments, and where that correlation is *not* high (`2nd_test`,
`3rd_test`), its own univariate separability across health states is far weaker than
kurtosis's -- see `docs/skewness_crestfactor_decision.md` for the full analysis.

**Decision: evaluated, not used.** Not wired into `src/features/extraction.py`'s
`FEATURE_COLUMNS`/`extract_experiment_features`, and not part of the versioned parquet
output. Kept here, tested, in case a future issue re-evaluates it (e.g. against a
different feature set or a different failure mode) -- per Issue #23's own instruction
not to delete evaluated-but-unused computation code.

(Skewness was evaluated alongside crest factor in the same issue but *confirmed*
useful -- it lives in `src/features/extraction.py` instead, alongside RMS/kurtosis.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.extraction import (
    ROLLING_WINDOW,
    compute_rms,
    list_snapshot_files,
    load_channel,
    parse_timestamp,
)

CREST_FACTOR_COLUMNS = ["experiment", "file_index", "timestamp", "crest_factor"]


def compute_crest_factor(signal: np.ndarray) -> float:
    """Peak absolute amplitude divided by RMS. `nan` for an all-zero signal (RMS == 0),
    matching `03_feature_candidate_screening.ipynb`'s `peak / rms if rms > 0 else nan`."""
    rms = compute_rms(signal)
    if rms == 0:
        return float("nan")
    return float(np.max(np.abs(signal)) / rms)


def add_rolling_crest_factor(df: pd.DataFrame, rolling_window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add `crest_factor_smoothed`: a `rolling_window`-file rolling mean of
    `crest_factor`, `min_periods=1`.

    Not used by `extract_crest_factor`'s default output -- `docs/feature_windowing_decision.md`
    (Issue #40) treats crest factor as unwindowed by default, and Issue #23
    (`docs/skewness_crestfactor_decision.md`) found smoothing roughly doubles its
    separability but does not close the gap to kurtosis's, nor remove its redundancy
    with kurtosis in the experiment (`1st_test`) where it's otherwise most separable.
    Kept as a function, not deleted, alongside the rest of this evaluated-but-unused
    module.
    """
    out = df.copy()
    out["crest_factor_smoothed"] = out["crest_factor"].rolling(rolling_window, min_periods=1).mean()
    return out


def extract_crest_factor(
    raw_dir: Path,
    experiment: str,
    channel_idx: int,
) -> pd.DataFrame:
    """Compute per-file crest factor for one experiment.

    Same shape/style as `extraction.extract_experiment_features`, scoped to the one
    candidate feature evaluated and rejected in Issue #23.
    """
    files = list_snapshot_files(raw_dir)
    if not files:
        raise ValueError(f"No snapshot files found in {raw_dir}")

    records = []
    for i, f in enumerate(files):
        sig = load_channel(f, channel_idx)
        records.append(
            {
                "file_index": i,
                "timestamp": parse_timestamp(f),
                "crest_factor": compute_crest_factor(sig),
            }
        )

    df = pd.DataFrame.from_records(records)
    df["experiment"] = experiment
    return df[CREST_FACTOR_COLUMNS]
