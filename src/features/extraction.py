"""RMS, kurtosis, and skewness feature extraction for bearing vibration snapshots.

Extracted as an importable module (Issue #41, same pattern as `src/labeling.py` from
Issue #19) so M3 training can consume it directly rather than re-deriving features
from a notebook. See `notebooks/01_vibration_signal_evolution.ipynb` for the original
derivation narrative and `docs/feature_windowing_decision.md` (Issue #40) for why RMS,
kurtosis, and skewness are windowed the way they are here:

- RMS: same 10-file rolling mean, `min_periods=1`, ratio to a per-experiment baseline,
  as `src/labeling.py` consumes (`rms_ratio`). Recomputing it identically means the
  feature column and the label are always derived from the same signal.
- Kurtosis: raw per-file, no rolling window, no baseline ratio. Deliberate --
  smoothing would blunt exactly the sharp per-file spikes that make kurtosis
  informative for `1st_test`'s impulsive inner-race failure (see the windowing
  decision doc, Section 2).
- Skewness: raw per-file value plus a 10-file rolling mean (`skewness_smoothed`,
  same window as RMS), absolute value not ratio-to-baseline (baseline |skewness| ~
  0.03 in all three experiments -- indistinguishable from zero). Confirmed useful in
  Issue #23 (`docs/skewness_crestfactor_decision.md`): `skewness_smoothed` separates
  health states at least as strongly as kurtosis in every experiment, without being
  collinear with it. Crest factor was evaluated alongside it and found redundant with
  kurtosis -- see `src/features/candidate_features.py`, kept there as
  evaluated-but-unused rather than folded in here.

Per-experiment tracked-bearing channel selection matches
`docs/eda_findings.md` Section 1 / `01_vibration_signal_evolution.ipynb`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import skew as _scipy_skew

# Carried over from notebooks/01_vibration_signal_evolution.ipynb /
# 02_health_state_labeling.ipynb -- do not change without re-running that analysis.
ROLLING_WINDOW = 10
BASELINE_N_FILES = 50

FEATURE_COLUMNS = [
    "experiment",
    "file_index",
    "timestamp",
    "rms",
    "rms_ratio",
    "kurtosis",
    "skewness",
    "skewness_smoothed",
]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    channel_idx: int  # 0-based column index of the tracked/failed bearing's channel
    bearing_label: str
    failure_mode: str


# Matches docs/eda_findings.md Section 1 and the EXPERIMENTS dict in
# notebooks/01_vibration_signal_evolution.ipynb.
EXPERIMENTS: dict[str, ExperimentConfig] = {
    "1st_test": ExperimentConfig("1st_test", 4, "Bearing 3, Ch 5", "inner race defect"),
    "2nd_test": ExperimentConfig("2nd_test", 0, "Bearing 1, Ch 1", "outer race failure"),
    "3rd_test": ExperimentConfig("3rd_test", 2, "Bearing 3, Ch 3", "outer race failure"),
}


def list_snapshot_files(test_dir: Path) -> list[Path]:
    """Snapshot filenames are timestamps, so lexicographic sort == chronological order."""
    return sorted(test_dir.iterdir(), key=lambda p: p.name)


def parse_timestamp(path: Path) -> datetime:
    return datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S")


def load_channel(path: Path, channel_idx: int) -> np.ndarray:
    """Read a single channel column from a snapshot file without loading the rest."""
    return (
        pd.read_csv(path, sep="\t", header=None, usecols=[channel_idx], dtype=np.float32)
        .iloc[:, 0]
        .to_numpy()
    )


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def compute_kurtosis(signal: np.ndarray) -> float:
    """Standard (Pearson) kurtosis, not Fisher's excess kurtosis -- Gaussian == 3.

    Matches `kurtosis(sig, fisher=False)` in
    `notebooks/01_vibration_signal_evolution.ipynb`, and the absolute-threshold
    rationale in `docs/feature_windowing_decision.md` (baseline kurtosis ~3.4 across
    all three experiments, not ~0 -- Fisher's excess form would shift that zero-point
    without changing the underlying evidence).
    """
    return float(_scipy_kurtosis(signal, fisher=False))


def compute_skewness(signal: np.ndarray) -> float:
    """Sample skewness (scipy default, `bias=True`).

    Matches `skew(sig)` in `03_feature_candidate_screening.ipynb` and
    `notebooks/01_vibration_signal_evolution.ipynb`. Confirmed useful in Issue #23
    (`docs/skewness_crestfactor_decision.md`) -- see `add_rolling_skewness` for the
    smoothed series that's actually separable across health states.
    """
    return float(_scipy_skew(signal))


def add_rolling_rms_ratio(
    df: pd.DataFrame,
    rolling_window: int = ROLLING_WINDOW,
    baseline_n_files: int = BASELINE_N_FILES,
) -> pd.DataFrame:
    """Add `rms_ratio`: an `rolling_window`-file rolling mean of `rms`, divided by the
    mean `rms` of the first `baseline_n_files` rows (chronological order assumed).

    Identical computation to `notebooks/02_health_state_labeling.ipynb`'s
    `load_stats` and to what `src.labeling.assign_labels` expects as input --
    see `docs/feature_windowing_decision.md` (Issue #40). `min_periods=1` means no
    row is ever `NaN`, including the first `rolling_window - 1` rows (Section 3 of
    that doc).
    """
    out = df.copy()
    baseline_rms = out["rms"].head(baseline_n_files).mean()
    out["rms_ratio"] = out["rms"].rolling(rolling_window, min_periods=1).mean() / baseline_rms
    return out


def add_rolling_skewness(df: pd.DataFrame, rolling_window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add `skewness_smoothed`: a `rolling_window`-file rolling mean of `skewness`,
    `min_periods=1` (no NaNs, same convention as `add_rolling_rms_ratio` -- see
    `docs/feature_windowing_decision.md` Section 3).

    Unlike `rms_ratio`, this is **not** a ratio-to-baseline -- `docs/eda_findings.md`
    /`docs/feature_windowing_decision.md` found baseline `|skewness|` ~ 0 in all three
    experiments, so a baseline-relative ratio would be meaningless (a near-zero
    denominator). Confirmed in Issue #23 (`docs/skewness_crestfactor_decision.md`) that
    smoothing materially improves skewness's separability across health states,
    validating this window empirically rather than by analogy alone.
    """
    out = df.copy()
    out["skewness_smoothed"] = out["skewness"].rolling(rolling_window, min_periods=1).mean()
    return out


def extract_experiment_features(
    raw_dir: Path,
    experiment: str,
    channel_idx: int,
    rolling_window: int = ROLLING_WINDOW,
    baseline_n_files: int = BASELINE_N_FILES,
) -> pd.DataFrame:
    """Compute per-file RMS/kurtosis/skewness and their rolling series for one experiment.

    Returns one row per snapshot file, in chronological order, with columns
    `FEATURE_COLUMNS` (`experiment`, `file_index`, `timestamp`, `rms`, `rms_ratio`,
    `kurtosis`, `skewness`, `skewness_smoothed`). `experiment` is a constant tag
    (e.g. `"1st_test"`), not derived from any per-file computation -- it lets the
    three experiments' parquet outputs be concatenated and grouped/filtered by test
    set without relying on filenames (Issue #43).

    Args:
        raw_dir: Directory containing one file per snapshot, named
            `YYYY.MM.DD.HH.MM.SS` (see `data/README.md`).
        experiment: Name of the experiment this `raw_dir` belongs to (e.g. one of
            the keys of `EXPERIMENTS`), stamped onto every output row.
        channel_idx: 0-based column index of the tracked bearing's channel within
            each snapshot file (see `EXPERIMENTS`).
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
                "rms": compute_rms(sig),
                "kurtosis": compute_kurtosis(sig),
                "skewness": compute_skewness(sig),
            }
        )

    df = pd.DataFrame.from_records(records)
    df = add_rolling_rms_ratio(
        df, rolling_window=rolling_window, baseline_n_files=baseline_n_files
    )
    df = add_rolling_skewness(df, rolling_window=rolling_window)
    df["experiment"] = experiment
    return df[FEATURE_COLUMNS]
