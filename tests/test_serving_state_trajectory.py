"""Tests for the per-bearing trajectory history and the endpoint that serves it (Issue
#140, `docs/agent_design.md` Section 12's addendum).

This is the half of `find_similar_historical_pattern` that lives in the serving process:
`docs/agent_design.md` Section 12 specified everything about the comparison except where a
live bearing's query comes from, and nothing retained one -- `rms_history` is 10 deep and
holds *raw* rms, not the three derived channels.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.api import create_app
from src.serving.drift import MONITORED_FEATURES
from src.serving.features import compute_online_features
from src.serving.state import (
    TRAJECTORY_CHANNELS,
    TRAJECTORY_HISTORY,
    BearingState,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def client(tmp_path):
    app = create_app(lock_path=tmp_path / "worker.lock")
    with TestClient(app) as test_client:
        yield test_client


def signal(rng, scale: float = 1.0) -> list[float]:
    """A realistic-shaped window: 20,480 samples, non-degenerate so kurtosis and skewness
    are finite."""
    return (rng.standard_normal(20480) * scale).tolist()


# --------------------------------------------------------------------------------------
# BearingState
# --------------------------------------------------------------------------------------


def test_observe_trajectory_records_all_three_channels():
    state = BearingState()
    state.observe_trajectory({"rms_ratio": 1.0, "kurtosis": 3.0, "skewness_smoothed": -0.1})

    assert state.trajectory() == {
        "rms_ratio": [1.0],
        "kurtosis": [3.0],
        "skewness_smoothed": [-0.1],
    }


def test_a_partial_update_is_refused():
    """Every channel is required. A partial update would leave the three deques at
    different lengths and silently misalign the columns of any matrix built from them --
    a wrong answer rather than a missing one."""
    state = BearingState()

    with pytest.raises(KeyError, match="missing"):
        state.observe_trajectory({"rms_ratio": 1.0, "kurtosis": 3.0})

    assert state.trajectory() == {channel: [] for channel in TRAJECTORY_CHANNELS}


def test_the_history_is_bounded_and_keeps_the_most_recent_values():
    state = BearingState()
    for i in range(TRAJECTORY_HISTORY + 25):
        state.observe_trajectory(
            {"rms_ratio": float(i), "kurtosis": float(i), "skewness_smoothed": float(i)}
        )

    trajectory = state.trajectory()
    assert len(trajectory["rms_ratio"]) == TRAJECTORY_HISTORY
    # Oldest first, ending at the most recent observation.
    assert trajectory["rms_ratio"][0] == 25.0
    assert trajectory["rms_ratio"][-1] == float(TRAJECTORY_HISTORY + 24)


def test_a_shorter_window_returns_the_tail():
    state = BearingState()
    for i in range(20):
        state.observe_trajectory(
            {"rms_ratio": float(i), "kurtosis": float(i), "skewness_smoothed": float(i)}
        )

    assert state.trajectory(window=3)["kurtosis"] == [17.0, 18.0, 19.0]


def test_asking_for_more_than_exists_returns_what_exists_rather_than_padding():
    state = BearingState()
    state.observe_trajectory({"rms_ratio": 1.0, "kurtosis": 2.0, "skewness_smoothed": 3.0})

    assert state.trajectory(window=50)["rms_ratio"] == [1.0]


def test_trajectory_history_is_its_own_constant_not_a_reuse_of_the_baseline():
    """Both are 50. They are equal by coincidence, not by derivation -- collapsing them
    would make a later change to either silently move the other."""
    state = BearingState()

    assert TRAJECTORY_HISTORY == 50
    assert state.trajectory_history == TRAJECTORY_HISTORY


# --------------------------------------------------------------------------------------
# The drift/trajectory separation
# --------------------------------------------------------------------------------------


def test_rms_ratio_is_a_trajectory_channel_and_not_a_drift_feature():
    """The one place these two Sections' opposite decisions about `rms_ratio` meet.
    `docs/monitoring_design.md` Section 2 excludes it (no population baseline exists for a
    per-bearing-normalized quantity); `docs/agent_design.md` Section 12 requires it (the
    leakage-safe normalization is what makes bearings comparable at all). Both hold,
    because the two paths share no state."""
    assert "rms_ratio" in TRAJECTORY_CHANNELS
    assert "rms_ratio" not in MONITORED_FEATURES


def test_observe_drift_still_refuses_an_rms_ratio_key():
    """Issue #90's guarantee, re-asserted here because #140 added a *second* per-file
    observer that does take `rms_ratio` -- so the exclusion now has a near neighbour that
    could be confused for it."""
    state = BearingState()

    with pytest.raises(KeyError):
        state.observe_drift({"rms_ratio": 1.0})


def test_compute_online_features_fills_both_histories_from_one_file():
    rng = np.random.default_rng(0)
    state = BearingState()
    features = compute_online_features(np.asarray(signal(rng)), state)

    trajectory = state.trajectory()
    assert trajectory["rms_ratio"] == [features.rms_ratio]
    assert trajectory["kurtosis"] == [features.kurtosis]
    assert trajectory["skewness_smoothed"] == [features.skewness_smoothed]
    # ...and the drift path advanced on the same call, unaffected.
    assert set(state.latest_z_scores) == set(MONITORED_FEATURES)


# --------------------------------------------------------------------------------------
# GET /monitoring/history/{bearing_id}
# --------------------------------------------------------------------------------------


def test_the_endpoint_returns_a_bearings_trajectory(client):
    rng = np.random.default_rng(1)
    for _ in range(12):
        client.post("/predict", json={"bearing_id": "b1", "signal": signal(rng)})

    body = client.get("/monitoring/history/b1").json()

    assert body["found"] is True
    assert body["file_count"] == 12
    assert body["n_points"] == 12
    assert body["baseline_status"] == "warming_up"
    assert set(body["channels"]) == set(TRAJECTORY_CHANNELS)
    assert all(len(values) == 12 for values in body["channels"].values())


def test_an_untracked_bearing_is_a_200_not_a_404(client):
    """A 404 would reach the agent as `ServingRejected` -- "the service refused this
    request" -- when the truthful answer is "nobody is tracking that bearing, and here is
    who is". Section 10 case 1 is precisely the failure of inventing a state for an
    untracked bearing, and the tool layer cannot avoid that if it cannot tell the two
    cases apart."""
    rng = np.random.default_rng(2)
    client.post("/predict", json={"bearing_id": "known", "signal": signal(rng)})

    response = client.get("/monitoring/history/ghost")

    assert response.status_code == 200
    assert response.json() == {
        "bearing_id": "ghost",
        "found": False,
        "tracked_bearings": ["known"],
    }


def test_the_endpoint_never_advances_a_bearing(client):
    rng = np.random.default_rng(3)
    client.post("/predict", json={"bearing_id": "b1", "signal": signal(rng)})

    for _ in range(3):
        client.get("/monitoring/history/b1")

    assert client.get("/monitoring/history/b1").json()["file_count"] == 1


def test_a_non_positive_window_is_rejected(client):
    rng = np.random.default_rng(4)
    client.post("/predict", json={"bearing_id": "b1", "signal": signal(rng)})

    assert client.get("/monitoring/history/b1", params={"window": 0}).status_code == 422


def test_the_window_parameter_limits_what_comes_back(client):
    rng = np.random.default_rng(5)
    for _ in range(15):
        client.post("/predict", json={"bearing_id": "b1", "signal": signal(rng)})

    body = client.get("/monitoring/history/b1", params={"window": 4}).json()

    assert body["n_points"] == 4
    assert body["file_count"] == 15  # the bearing itself is unchanged


def test_a_non_finite_reading_is_served_as_null_rather_than_breaking_the_endpoint(client):
    """A constant window makes kurtosis and skewness 0/0, which leaves `NaN` in the
    history. `NaN` is not valid JSON, so serializing it raised -- leaving that bearing's
    history endpoint permanently 500ing, a worse failure than the one that caused it.

    The `NaN` is put into the state directly rather than by posting a constant signal:
    such a signal makes `/predict` itself raise inside the model (a pre-existing edge
    case, unrelated to this endpoint), which would stop the test before it reached the
    thing under test. The state it *leaves behind* is what this asserts on.
    """
    store = client.app.state.extractor.store
    store.get_or_create("flat").observe_trajectory(
        {"rms_ratio": 1.0, "kurtosis": float("nan"), "skewness_smoothed": float("-inf")}
    )

    response = client.get("/monitoring/history/flat")

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["channels"]["kurtosis"] == [None]
    assert body["channels"]["skewness_smoothed"] == [None]
    assert math.isfinite(body["channels"]["rms_ratio"][0])


def test_each_bearing_keeps_its_own_trajectory(client):
    rng = np.random.default_rng(6)
    for _ in range(5):
        client.post("/predict", json={"bearing_id": "a", "signal": signal(rng, scale=1.0)})
    for _ in range(9):
        client.post("/predict", json={"bearing_id": "b", "signal": signal(rng, scale=4.0)})

    assert client.get("/monitoring/history/a").json()["n_points"] == 5
    assert client.get("/monitoring/history/b").json()["n_points"] == 9


def test_an_ordinary_bearing_id_round_trips_verbatim(client):
    """Ids are echoed back exactly as given, so a caller can tell which bearing a body
    describes without re-deriving it. (Path *quoting* of unusual ids is the agent client's
    job and is asserted in `tests/test_agent_mcp_tools.py`, where the request line actually
    sent is observable.)"""
    body = client.get("/monitoring/history/2nd_test-demo").json()

    assert body["bearing_id"] == "2nd_test-demo"
    assert body["found"] is False
