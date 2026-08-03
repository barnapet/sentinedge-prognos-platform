"""Tests for `BearingState`'s drift tracking (Issue #90, `docs/monitoring_design.md`
Sections 2-3): the rolling extreme-flag history, the 3-of-10 persistence rule, the
`predicted_class_counts` tally, and -- the critical constraint -- that `rms_ratio` can
never reach any of it.

All tests here use a controlled, synthetic `drift_baseline` (mean=0, std=1 for three
features; kurtosis mean=3, std=1, matching its real baseline being centred near the
Gaussian reference point) rather than the committed `models/drift_baseline.json`, so a
feature's z-score is simply its raw value (or value-3 for kurtosis) -- exact, easy-to-read
numbers instead of the real baseline's non-round mean/std.
"""
from __future__ import annotations

from src.features.extraction import ROLLING_WINDOW
from src.serving.drift import DRIFT_DRIFTING, DRIFT_NOMINAL, PERSISTENCE_MIN_COUNT
from src.serving.state import BearingState, BearingStateStore

# mean=0, std=1 for rms/skewness/skewness_smoothed; kurtosis centred at 3 (its real
# Gaussian reference point) with std=1 -- so z = value for the first three, z = value - 3
# for kurtosis. Chosen so test values can be read directly as z-scores.
UNIT_BASELINE = {
    "rms": (0.0, 1.0),
    "kurtosis": (3.0, 1.0),
    "skewness": (0.0, 1.0),
    "skewness_smoothed": (0.0, 1.0),
}

NOMINAL_READING = {"rms": 0.1, "kurtosis": 3.1, "skewness": 0.05, "skewness_smoothed": 0.05}
EXTREME_RMS_READING = {"rms": 10.0, "kurtosis": 3.1, "skewness": 0.05, "skewness_smoothed": 0.05}


def make_state() -> BearingState:
    return BearingState(drift_baseline=UNIT_BASELINE)


# --- a single extreme reading does not persist -----------------------------------------


def test_one_isolated_extreme_reading_does_not_flip_drifting():
    """docs/monitoring_design.md Section 3: a lone 3-sigma reading is expected occasionally
    by chance (the Chebyshev bound in Section 1), so it alone must not read as drifting."""
    state = make_state()

    state.observe_drift(EXTREME_RMS_READING)
    for _ in range(9):
        state.observe_drift(NOMINAL_READING)

    assert state.feature_drift("rms")["drifting"] is False
    assert state.drift_status == DRIFT_NOMINAL


def test_two_extreme_readings_in_ten_do_not_flip_drifting():
    """One below the 3-of-10 threshold -- the boundary this rule is built around."""
    state = make_state()

    state.observe_drift(EXTREME_RMS_READING)
    state.observe_drift(EXTREME_RMS_READING)
    for _ in range(8):
        state.observe_drift(NOMINAL_READING)

    assert state.feature_drift("rms")["drifting"] is False
    assert state.drift_status == DRIFT_NOMINAL


# --- the exact 3-of-10 boundary, in both directions ------------------------------------


def test_the_third_extreme_reading_in_the_window_flips_drifting():
    """The precise transition: 2 extreme readings -> not drifting; the 3rd -> drifting.
    Matches this project's convention of testing state-machine boundaries at the exact
    transition point, not just "eventually true" (see `test_baseline_locks_on_the_50th_
    file_not_the_49th_or_51st` in `tests/test_serving_features.py` for the precedent)."""
    state = make_state()

    state.observe_drift(EXTREME_RMS_READING)
    assert state.feature_drift("rms")["drifting"] is False

    state.observe_drift(EXTREME_RMS_READING)
    assert state.feature_drift("rms")["drifting"] is False

    state.observe_drift(EXTREME_RMS_READING)
    assert state.feature_drift("rms")["drifting"] is True
    assert state.drift_status == DRIFT_DRIFTING


def test_sustained_extreme_readings_stay_flagged_as_drifting():
    """The "real, sustained shift" case docs/monitoring_design.md Section 3 contrasts with
    a lone excursion: every reading extreme, well past the 3-of-10 threshold."""
    state = make_state()

    for _ in range(ROLLING_WINDOW):
        state.observe_drift(EXTREME_RMS_READING)

    assert state.feature_drift("rms")["drifting"] is True
    assert state.drift_status == DRIFT_DRIFTING


# --- the window is rolling: old extremes age out ----------------------------------------


def test_drifting_reverts_to_nominal_once_extreme_readings_age_out_of_the_window():
    """"Persistently drifting" means the last `rolling_window` readings, not "ever". Once
    three old extreme readings have scrolled past the back of a full 10-window of nominal
    readings, the flag must revert -- a sensor that recovers should stop being flagged."""
    state = make_state()

    for _ in range(3):
        state.observe_drift(EXTREME_RMS_READING)
    assert state.feature_drift("rms")["drifting"] is True

    # Push exactly ROLLING_WINDOW nominal readings through -- enough to evict all three
    # extreme ones from the length-10 deque.
    for _ in range(ROLLING_WINDOW):
        state.observe_drift(NOMINAL_READING)

    assert state.feature_drift("rms")["drifting"] is False
    assert state.drift_status == DRIFT_NOMINAL


# --- drift_history respects ROLLING_WINDOW, the same constant as rms_history ------------


def test_drift_history_is_bounded_to_rolling_window():
    state = make_state()
    for _ in range(ROLLING_WINDOW + 5):
        state.observe_drift(NOMINAL_READING)

    assert len(state.drift_history["rms"]) == ROLLING_WINDOW


# --- latest_z_scores / feature_drift's "z" ----------------------------------------------


def test_feature_drift_reports_the_latest_reading_s_z_score():
    state = make_state()
    state.observe_drift({"rms": 2.0, "kurtosis": 3.0, "skewness": 0.0, "skewness_smoothed": 0.0})

    assert state.feature_drift("rms")["z"] == 2.0
    assert state.feature_drift("kurtosis")["z"] == 0.0  # 3.0 - mean(3.0) = 0.0

    state.observe_drift({"rms": -1.5, "kurtosis": 3.0, "skewness": 0.0, "skewness_smoothed": 0.0})
    assert state.feature_drift("rms")["z"] == -1.5, "z must reflect the LATEST reading"


# --- the critical constraint: rms_ratio can never reach drift_status -------------------


def test_rms_ratio_is_never_part_of_the_drift_computation():
    """`rms_ratio` is not one of `MONITORED_FEATURES`, so `observe_drift` is never even
    called with it -- this replays the real request path (`compute_online_features`) with
    a bearing whose `rms_ratio` is driven extremely high by a collapsed baseline, while
    keeping the four monitored features themselves within the (unit) baseline, and asserts
    `drift_status` stays nominal regardless."""
    import numpy as np

    from src.serving.features import compute_online_features

    state = BearingState(drift_baseline=UNIT_BASELINE)
    # 50 quiet files lock a tiny baseline_rms.
    quiet = np.array([0.001, -0.001, 0.001, -0.001], dtype=np.float32)
    for _ in range(50):
        compute_online_features(quiet, state)

    # A file whose rms sits within the (unit) baseline (z small) but whose *ratio* to the
    # tiny locked baseline_rms is enormous.
    moderate_rms_signal = np.array([0.15, -0.15, 0.2, -0.2], dtype=np.float32)
    features = compute_online_features(moderate_rms_signal, state)

    assert features.rms_ratio > 15.0, (
        "the ratio must actually be extreme for this test to mean anything -- for context, "
        "real bearings' Critical-class rms_ratio tops out around 3.0-7.2 "
        "(docs/eda_findings.md Section 3 / docs/model_training_decision.md Section 3b)"
    )
    assert features.drift_status == DRIFT_NOMINAL, (
        "an extreme rms_ratio must never flip drift_status -- only the four "
        "MONITORED_FEATURES may (docs/monitoring_design.md Section 2)"
    )


def test_rms_ratio_key_would_raise_if_ever_passed_to_observe_drift():
    """Defence in depth: `observe_drift` reads `self.drift_baseline[feature]`, and the
    committed/synthetic baselines here never define an `rms_ratio` entry -- so even a
    future accidental call site passing `rms_ratio` would fail loudly (KeyError) rather
    than silently start influencing `drift_status`."""
    import pytest

    state = make_state()
    assert "rms_ratio" not in state.drift_baseline

    with pytest.raises(KeyError):
        state.observe_drift({"rms_ratio": 999.0})


# --- predicted_class_counts -------------------------------------------------------------


def test_record_prediction_tallies_labels():
    state = BearingState()

    state.record_prediction("Normal")
    state.record_prediction("Normal")
    state.record_prediction("Critical")

    assert state.predicted_class_counts == {"Normal": 2, "Critical": 1}


def test_predicted_class_counts_starts_empty():
    assert BearingState().predicted_class_counts == {}


# --- two bearings stay independent ------------------------------------------------------


def test_two_bearings_have_independent_drift_histories():
    store = BearingStateStore(drift_baseline=UNIT_BASELINE)
    a, b = store.get_or_create("bearing-a"), store.get_or_create("bearing-b")

    for _ in range(3):
        a.observe_drift(EXTREME_RMS_READING)
    for _ in range(3):
        b.observe_drift(NOMINAL_READING)

    assert a.drift_status == DRIFT_DRIFTING
    assert b.drift_status == DRIFT_NOMINAL


def test_two_bearings_have_independent_predicted_class_counts():
    store = BearingStateStore()
    store.get_or_create("bearing-a").record_prediction("Critical")
    store.get_or_create("bearing-b").record_prediction("Normal")

    assert store.get_or_create("bearing-a").predicted_class_counts == {"Critical": 1}
    assert store.get_or_create("bearing-b").predicted_class_counts == {"Normal": 1}


# --- constructor injection matches the ROLLING_WINDOW/BASELINE_N_FILES precedent --------


def test_bearing_state_store_forwards_drift_baseline_to_new_states():
    store = BearingStateStore(drift_baseline=UNIT_BASELINE)

    state = store.get_or_create("bearing-a")

    assert state.drift_baseline == UNIT_BASELINE


def test_default_drift_baseline_is_the_committed_artifact():
    """With no override, a fresh `BearingState` uses the real committed
    `models/drift_baseline.json` (via `src.serving.drift.DRIFT_BASELINE`) -- the production
    default the API layer relies on implicitly."""
    from src.serving.drift import DRIFT_BASELINE

    assert BearingState().drift_baseline == DRIFT_BASELINE
