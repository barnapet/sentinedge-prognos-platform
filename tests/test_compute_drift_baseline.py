"""Tests for the drift-detection baseline artifact (Issue #90).

Mirrors `tests/test_train_serving_model.py`'s shape: a synthetic training-dataset-shaped
frame (`data/processed/` is gitignored and absent when CI's unit-test step runs, before the
notebook step populates it), plus a set of tests against the real committed
`models/drift_baseline.json`, skipped rather than failed where it is absent.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.training.compute_drift_baseline import (
    BASELINE_PATH,
    MONITORED_FEATURES,
    _compensated_mean_std,
    build_baseline_manifest,
    compute_feature_baseline,
    load_drift_baseline,
    persist_baseline,
)
from tests.test_train_baseline import ROWS_PER_EXPERIMENT, make_dataset

# Stand-in for Issue #67's `training_dataset_manifest.json` `combined_hash`, not on disk in
# CI. Any fixed string works -- what is under test is that it is recorded, not its value.
FAKE_DATASET_VERSION = "1" * 64


@pytest.fixture
def df():
    return make_dataset()


@pytest.fixture
def persisted(tmp_path, df):
    """Persist a baseline into `tmp_path`, never the repo's `models/`."""
    path = tmp_path / "drift_baseline.json"
    manifest = persist_baseline(df, path=path, training_dataset_version=FAKE_DATASET_VERSION)
    return {"df": df, "manifest": manifest, "path": path}


# --- MONITORED_FEATURES excludes rms_ratio ------------------------------------------


def test_monitored_features_are_exactly_the_four_unnormalized_features():
    """docs/monitoring_design.md Section 2: rms/kurtosis/skewness/skewness_smoothed, and
    critically, NOT rms_ratio -- it is already normalized per-bearing, so a population
    baseline for it would measure between-bearing severity, not sensor drift."""
    assert MONITORED_FEATURES == ["rms", "kurtosis", "skewness", "skewness_smoothed"]
    assert "rms_ratio" not in MONITORED_FEATURES


# --- compute_feature_baseline --------------------------------------------------------


def test_baseline_matches_plain_pandas_mean_and_std(df):
    """No hidden weighting or filtering: the baseline is exactly `df[feature].mean()`/
    `.std()`, pooled over every row regardless of experiment or label."""
    baseline = compute_feature_baseline(df)

    for feature in MONITORED_FEATURES:
        assert baseline[feature]["mean"] == pytest.approx(df[feature].mean())
        assert baseline[feature]["std"] == pytest.approx(df[feature].std())


def test_baseline_pools_every_label_not_just_normal(df):
    """docs/monitoring_design.md Section 1: the baseline must differ from a Normal-only
    baseline, because pooling every label (including Degrading/Critical) is the deliberate
    decision -- if these matched, the fixture would not actually be testing pooling."""
    baseline = compute_feature_baseline(df)
    normal_only = compute_feature_baseline(df[df["label"] == "Normal"])

    assert baseline["rms"]["mean"] != pytest.approx(normal_only["rms"]["mean"])


def test_baseline_covers_every_monitored_feature(df):
    assert set(compute_feature_baseline(df)) == set(MONITORED_FEATURES)


def test_compensated_mean_recovers_precision_pandas_mean_loses():
    """Regression test for Issue #93: a real CI-observed drift between the committed
    `models/drift_baseline.json` and a locally recomputed baseline, traced to plain pandas
    `.mean()`/`.std()` (Issue #90's original implementation) losing precision on pooled sums
    that largely cancel toward a near-zero value -- exactly the shape of three of the four
    monitored features (`docs/monitoring_design.md` Section 1).

    `[1e16, 1.0, -1e16]` is the textbook catastrophic-cancellation case: the `1.0` is far
    below the precision of `1e16`, so any summation that adds the huge values first loses it
    entirely. Measured directly (not assumed) against this repo's pinned versions
    (`numpy==2.4.6`, `pandas==3.0.5`): `pandas.Series.mean()` and `numpy.mean()` both return
    `0.0` here, silently discarding the `1.0`, while `math.fsum` -- what
    `src.serving.state.window_mean` already uses for the same reason (Issue #82) -- recovers
    the exact `1.0` regardless of numpy/pandas version, because its result is a property of
    the input alone.
    """
    values = pd.Series([1e16, 1.0, -1e16])

    assert values.mean() == 0.0, "pandas must still lose this -- otherwise the test proves nothing"

    mean, _ = _compensated_mean_std(values)
    assert mean == pytest.approx(1 / 3)


# --- manifest shape -------------------------------------------------------------------


def test_manifest_records_the_dataset_version_and_row_count(df):
    manifest = build_baseline_manifest(df, training_dataset_version=FAKE_DATASET_VERSION)

    assert manifest["training_dataset_version"] == FAKE_DATASET_VERSION
    assert manifest["n_rows"] == len(df)
    assert manifest["features"] == MONITORED_FEATURES


def test_manifest_is_written_as_readable_json(persisted):
    """The manifest is the human-auditable artifact itself -- eight floats, not an opaque
    binary (docs/monitoring_design.md Section 3), so it must round-trip as plain JSON."""
    on_disk = json.loads(persisted["path"].read_text())

    assert on_disk == persisted["manifest"]


def test_persist_baseline_creates_parent_directories(tmp_path, df):
    nested_path = tmp_path / "nested" / "drift_baseline.json"

    persist_baseline(df, path=nested_path, training_dataset_version=FAKE_DATASET_VERSION)

    assert nested_path.exists()


# --- load_drift_baseline round-trip ----------------------------------------------------


def test_load_drift_baseline_returns_mean_std_pairs(persisted):
    loaded = load_drift_baseline(persisted["path"])

    assert set(loaded) == set(MONITORED_FEATURES)
    for feature in MONITORED_FEATURES:
        mean, std = loaded[feature]
        assert mean == pytest.approx(persisted["manifest"]["baseline"][feature]["mean"])
        assert std == pytest.approx(persisted["manifest"]["baseline"][feature]["std"])


# --- the artifact committed to the repo -------------------------------------------------


def test_committed_baseline_matches_the_real_training_dataset_if_both_are_present():
    """The repo's own `models/drift_baseline.json` (committed, Issue #90) should reproduce
    exactly if `data/processed/training_dataset.parquet` is also present -- both skipped,
    not failed, on a clone that has neither (same convention as
    `tests/test_train_serving_model.py::test_committed_artifact_matches_its_manifest_if_present`)."""
    from src.training.evaluation import TRAINING_DATASET_PATH, load_training_dataset

    if not (BASELINE_PATH.exists() and TRAINING_DATASET_PATH.exists()):
        pytest.skip("no committed drift baseline or training dataset on disk")

    committed = json.loads(BASELINE_PATH.read_text())
    recomputed = compute_feature_baseline(load_training_dataset())

    for feature in MONITORED_FEATURES:
        assert committed["baseline"][feature]["mean"] == pytest.approx(
            recomputed[feature]["mean"]
        )
        assert committed["baseline"][feature]["std"] == pytest.approx(
            recomputed[feature]["std"]
        )


def test_committed_baseline_has_the_expected_shape_if_present():
    """Skipped rather than failed where the artifact has not been generated, so the suite
    still runs on a clone that has not computed one."""
    if not BASELINE_PATH.exists():
        pytest.skip("no committed drift baseline on disk")

    manifest = json.loads(BASELINE_PATH.read_text())

    assert manifest["features"] == MONITORED_FEATURES
    assert set(manifest["baseline"]) == set(MONITORED_FEATURES)
    for feature in MONITORED_FEATURES:
        assert manifest["baseline"][feature]["std"] > 0.0, (
            f"{feature}'s std must be positive -- a zero std would make every z-score 0.0"
        )
