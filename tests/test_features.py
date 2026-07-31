import math

import numpy as np
import pandas as pd
import pytest

from src.features.extraction import (
    EXPERIMENTS,
    FEATURE_COLUMNS,
    add_rolling_rms_ratio,
    compute_kurtosis,
    compute_rms,
    extract_experiment_features,
)
from src.features.versioning import (
    build_manifest,
    compute_code_hash,
    compute_combined_hash,
    compute_raw_dataset_version,
)


def write_snapshot(path, columns):
    """Write a tab-separated snapshot file with the given per-column value lists."""
    n_rows = len(columns[0])
    with open(path, "w") as f:
        for row in range(n_rows):
            f.write("\t".join(str(col[row]) for col in columns) + "\n")


# --- compute_rms / compute_kurtosis -------------------------------------------

def test_compute_rms_matches_manual_calculation():
    signal = np.array([3.0, 4.0])
    assert compute_rms(signal) == pytest.approx(math.sqrt((9.0 + 16.0) / 2))


def test_compute_kurtosis_uses_standard_not_fisher_convention():
    """fisher=False: a symmetric two-point-mass signal has standard kurtosis 1.0,
    not scipy's default Fisher (excess) form -- Gaussian is 3, not 0, in this
    convention, matching docs/feature_windowing_decision.md's absolute-threshold
    rationale (baseline kurtosis ~3.4, not ~0)."""
    signal = np.array([-1.0, -1.0, 1.0, 1.0])
    assert compute_kurtosis(signal) == pytest.approx(1.0)


# --- add_rolling_rms_ratio ------------------------------------------------------

def test_add_rolling_rms_ratio_uses_10_file_window_and_baseline():
    """Same computation as notebooks/02_health_state_labeling.ipynb's load_stats:
    10-file rolling mean of rms, min_periods=1, divided by the mean rms of the first
    50 files (docs/feature_windowing_decision.md, Issue #40)."""
    rms_values = [1.0, 2.0, 3.0] + [1.0] * 20  # first 20 rows form the baseline window
    df = pd.DataFrame({"rms": rms_values})

    result = add_rolling_rms_ratio(df, rolling_window=10, baseline_n_files=20)

    baseline = pd.Series(rms_values).head(20).mean()
    expected_rolling = pd.Series(rms_values).rolling(10, min_periods=1).mean()
    expected_ratio = expected_rolling / baseline

    assert result["rms_ratio"].tolist() == pytest.approx(expected_ratio.tolist())
    # min_periods=1 means no NaNs anywhere, including the first (window - 1) rows.
    assert not result["rms_ratio"].isna().any()


# --- extract_experiment_features (end-to-end on a synthetic raw dir) -----------

def test_extract_experiment_features_end_to_end(tmp_path):
    raw_dir = tmp_path / "synthetic_test"
    raw_dir.mkdir()

    # Three tiny 4-point, single-channel snapshots, named as real IMS timestamps are.
    filenames = ["2003.10.22.12.00.00", "2003.10.22.12.10.00", "2003.10.22.12.20.00"]
    signals = [[1.0, -1.0, 1.0, -1.0], [2.0, -2.0, 2.0, -2.0], [0.5, 0.5, -0.5, -0.5]]
    for name, sig in zip(filenames, signals):
        write_snapshot(raw_dir / name, [sig])

    df = extract_experiment_features(raw_dir, channel_idx=0, rolling_window=2, baseline_n_files=1)

    assert list(df.columns) == FEATURE_COLUMNS
    assert df["file_index"].tolist() == [0, 1, 2]
    # Filenames sort chronologically, so row order must match the listed order above.
    assert df["rms"].tolist() == pytest.approx(
        [compute_rms(np.array(sig)) for sig in signals]
    )
    assert df["kurtosis"].tolist() == pytest.approx(
        [compute_kurtosis(np.array(sig)) for sig in signals]
    )
    assert not df["rms_ratio"].isna().any()


@pytest.mark.parametrize("experiment_name", ["1st_test", "2nd_test", "3rd_test"])
def test_extract_experiment_features_reads_documented_tracked_channel(tmp_path, experiment_name):
    """Each experiment's tracked-bearing channel (docs/eda_findings.md Section 1) must
    be the column actually read -- not an arbitrary/default one. Build an 8-column
    synthetic file where every column has a distinct, known signal, and confirm the
    extracted rms/kurtosis match the documented channel_idx's column only."""
    cfg = EXPERIMENTS[experiment_name]
    raw_dir = tmp_path / experiment_name
    raw_dir.mkdir()

    n_channels = 8
    # Column j gets values [j, -j, j, -j] (well-defined, non-constant kurtosis, and
    # distinguishable between columns) so a wrong channel_idx would be caught.
    columns = [[j, -j, j, -j] for j in range(1, n_channels + 1)]
    write_snapshot(raw_dir / "2003.10.22.12.00.00", columns)

    df = extract_experiment_features(raw_dir, channel_idx=cfg.channel_idx)

    expected_signal = np.array(columns[cfg.channel_idx], dtype=np.float32)
    assert df["rms"].iloc[0] == pytest.approx(compute_rms(expected_signal))
    assert df["kurtosis"].iloc[0] == pytest.approx(compute_kurtosis(expected_signal))


# --- versioning: code hash -------------------------------------------------------

def test_compute_code_hash_is_deterministic(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")

    assert compute_code_hash((f,)) == compute_code_hash((f,))


def test_compute_code_hash_changes_with_content(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")
    original = compute_code_hash((f,))

    f.write_text("x = 2\n")
    changed = compute_code_hash((f,))

    assert original != changed


def test_compute_code_hash_independent_of_argument_order(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("a = 1\n")
    b.write_text("b = 2\n")

    assert compute_code_hash((a, b)) == compute_code_hash((b, a))


# --- versioning: raw dataset version ---------------------------------------------

def test_compute_raw_dataset_version_is_deterministic(tmp_path):
    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    (raw_dir / "2003.10.22.12.00.00").write_text("data\n")
    (raw_dir / "2003.10.22.12.10.00").write_text("more data\n")

    assert compute_raw_dataset_version(raw_dir) == compute_raw_dataset_version(raw_dir)


def test_compute_raw_dataset_version_changes_with_file_size(tmp_path):
    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    f = raw_dir / "2003.10.22.12.00.00"
    f.write_text("data\n")
    original = compute_raw_dataset_version(raw_dir)

    f.write_text("more data than before\n")
    changed = compute_raw_dataset_version(raw_dir)

    assert original != changed


def test_compute_raw_dataset_version_changes_with_file_count(tmp_path):
    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    (raw_dir / "2003.10.22.12.00.00").write_text("data\n")
    original = compute_raw_dataset_version(raw_dir)

    (raw_dir / "2003.10.22.12.10.00").write_text("data\n")
    changed = compute_raw_dataset_version(raw_dir)

    assert original != changed


# --- versioning: combined hash and manifest --------------------------------------

def test_compute_combined_hash_deterministic_and_sensitive_to_either_input():
    h1 = compute_combined_hash("code-a", "data-a")
    assert h1 == compute_combined_hash("code-a", "data-a")
    assert h1 != compute_combined_hash("code-b", "data-a")
    assert h1 != compute_combined_hash("code-a", "data-b")


def test_build_manifest_contains_required_fields():
    manifest = build_manifest(
        experiment="1st_test",
        code_hash="deadbeef",
        raw_dataset_version="cafef00d",
        feature_columns=FEATURE_COLUMNS,
        n_files=2156,
    )

    assert manifest["experiment"] == "1st_test"
    assert manifest["code_hash"] == "deadbeef"
    assert manifest["raw_dataset_version"] == "cafef00d"
    assert manifest["combined_hash"] == compute_combined_hash("deadbeef", "cafef00d")
    assert manifest["feature_columns"] == FEATURE_COLUMNS
    assert manifest["n_files"] == 2156
    assert "generated_at" in manifest


# --- reproducibility: same code + same raw data -> same hash ---------------------

def test_reproducibility_same_code_and_data_yield_same_combined_hash(tmp_path):
    code_file = tmp_path / "extraction_copy.py"
    code_file.write_text("# pretend generating code\n")

    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    (raw_dir / "2003.10.22.12.00.00").write_text("0.1\t0.2\n")
    (raw_dir / "2003.10.22.12.10.00").write_text("0.3\t0.4\n")

    def run():
        code_hash = compute_code_hash((code_file,))
        raw_version = compute_raw_dataset_version(raw_dir)
        manifest = build_manifest(
            experiment="1st_test",
            code_hash=code_hash,
            raw_dataset_version=raw_version,
            feature_columns=FEATURE_COLUMNS,
            n_files=2,
        )
        return manifest["combined_hash"]

    assert run() == run()


def test_reproducibility_breaks_if_raw_data_changes(tmp_path):
    code_file = tmp_path / "extraction_copy.py"
    code_file.write_text("# pretend generating code\n")

    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    (raw_dir / "2003.10.22.12.00.00").write_text("0.1\t0.2\n")

    code_hash = compute_code_hash((code_file,))
    before = compute_combined_hash(code_hash, compute_raw_dataset_version(raw_dir))

    (raw_dir / "2003.10.22.12.10.00").write_text("0.3\t0.4\n")  # dataset changed
    after = compute_combined_hash(code_hash, compute_raw_dataset_version(raw_dir))

    assert before != after
