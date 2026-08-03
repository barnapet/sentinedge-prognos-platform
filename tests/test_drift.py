"""Tests for the pure drift-check functions (Issue #90, `src/serving/drift.py`).

The persistence rule itself (the 3-of-10 rolling behaviour) lives on `BearingState` and is
tested in `tests/test_serving_state_drift.py`, since it needs per-bearing rolling history.
This file covers the stateless building blocks: the z-score, the single-reading extreme
check, and the constants both the offline baseline script and the online state module
depend on.
"""
from __future__ import annotations

import pytest

from src.serving.drift import (
    DRIFT_BASELINE,
    DRIFT_DRIFTING,
    DRIFT_NOMINAL,
    MONITORED_FEATURES,
    PERSISTENCE_MIN_COUNT,
    Z_SCORE_THRESHOLD,
    compute_z_score,
    is_extreme,
)


# --- compute_z_score ------------------------------------------------------------------


def test_compute_z_score_matches_the_textbook_definition():
    assert compute_z_score(value=10.0, mean=5.0, std=2.0) == pytest.approx(2.5)
    assert compute_z_score(value=5.0, mean=5.0, std=2.0) == pytest.approx(0.0)
    assert compute_z_score(value=0.0, mean=5.0, std=2.0) == pytest.approx(-2.5)


def test_compute_z_score_guards_against_zero_std():
    """A `std == 0.0` feature would otherwise raise ZeroDivisionError -- a monitoring check
    should degrade to "no signal" (z=0.0), not crash the request that triggered it."""
    assert compute_z_score(value=100.0, mean=5.0, std=0.0) == 0.0


# --- is_extreme, and the Chebyshev-bound threshold -------------------------------------


def test_is_extreme_uses_the_documented_threshold():
    assert Z_SCORE_THRESHOLD == 3.0


def test_is_extreme_true_above_threshold():
    assert is_extreme(3.1) is True
    assert is_extreme(-3.1) is True


def test_is_extreme_false_at_or_below_threshold():
    """Strictly greater than, not greater-or-equal -- exactly 3.0 does not count."""
    assert is_extreme(3.0) is False
    assert is_extreme(2.9) is False
    assert is_extreme(0.0) is False


def test_is_extreme_respects_a_custom_threshold():
    assert is_extreme(2.5, threshold=2.0) is True
    assert is_extreme(1.5, threshold=2.0) is False


# --- docs/monitoring_design.md Section 3's persistence constant -----------------------


def test_persistence_min_count_is_the_documented_3_of_10_rule():
    assert PERSISTENCE_MIN_COUNT == 3


def test_drift_status_constants_are_the_documented_strings():
    assert DRIFT_NOMINAL == "nominal"
    assert DRIFT_DRIFTING == "drifting"


# --- MONITORED_FEATURES / DRIFT_BASELINE, re-exported for src.serving.state ------------


def test_monitored_features_excludes_rms_ratio():
    """The critical constraint: rms_ratio must never be a key the drift check reads, since
    that is what keeps it from ever driving `drifting`/`drift_status` (Section 2)."""
    assert "rms_ratio" not in MONITORED_FEATURES
    assert set(MONITORED_FEATURES) == {"rms", "kurtosis", "skewness", "skewness_smoothed"}


def test_drift_baseline_loads_and_covers_every_monitored_feature():
    """Loaded once at import time from the committed `models/drift_baseline.json` -- this
    is the real artifact, not a synthetic stand-in, since it is committed (non-gitignored)
    and therefore always present at checkout, unlike `data/processed/`."""
    assert set(DRIFT_BASELINE) == set(MONITORED_FEATURES)
    for feature in MONITORED_FEATURES:
        mean, std = DRIFT_BASELINE[feature]
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0.0
