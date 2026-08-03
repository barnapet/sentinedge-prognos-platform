"""Integration tests for the monitoring endpoints (Issue #90): `GET /monitoring/drift` and
`GET /monitoring`. Exercises the full HTTP path via FastAPI's `TestClient`, the same
convention `tests/test_api.py` (#84) already uses -- deliberately not re-testing what
`tests/test_serving_state_drift.py` already covers at the module level (the persistence
rule itself, the rms_ratio exclusion). What's specific to this layer: the endpoint's
response shape, that it reflects real `/predict` traffic, and that the dashboard page is
actually served.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.api import create_app
from src.serving.drift import MONITORED_FEATURES


def make_signal(n: int = 64, seed: int = 0, amplitude: float = 0.08) -> list[float]:
    return np.random.default_rng(seed).normal(0.0, amplitude, n).tolist()


@pytest.fixture()
def app(tmp_path):
    return create_app(lock_path=tmp_path / "serving.lock")


@pytest.fixture()
def client(app):
    with TestClient(app) as c:
        yield c


# --- GET /monitoring/drift: shape -------------------------------------------------------


def test_drift_endpoint_returns_an_empty_bearings_map_before_any_predict_call(client):
    response = client.get("/monitoring/drift")

    assert response.status_code == 200
    assert response.json() == {"bearings": {}}


def test_drift_endpoint_matches_the_documented_shape_after_one_predict_call(client):
    """docs/monitoring_design.md Section 3's illustrative shape, checked key by key."""
    client.post("/predict", json={"bearing_id": "1st_test-demo", "signal": make_signal()})

    body = client.get("/monitoring/drift").json()

    assert set(body) == {"bearings"}
    bearing = body["bearings"]["1st_test-demo"]
    assert set(bearing) == {
        "file_count",
        "baseline_status",
        "drift_status",
        "features",
        "rms_ratio_latest",
        "predicted_class_counts",
    }
    assert bearing["file_count"] == 1
    assert bearing["baseline_status"] == "warming_up"
    assert bearing["drift_status"] in {"nominal", "drifting"}
    assert set(bearing["features"]) == set(MONITORED_FEATURES)
    for feature in MONITORED_FEATURES:
        assert set(bearing["features"][feature]) == {"z", "drifting"}
        assert isinstance(bearing["features"][feature]["z"], float)
        assert isinstance(bearing["features"][feature]["drifting"], bool)
    assert isinstance(bearing["rms_ratio_latest"], float)


def test_drift_endpoint_never_includes_rms_ratio_among_the_monitored_features(client):
    """The critical constraint, checked at the wire level: `rms_ratio` is surfaced only as
    `rms_ratio_latest`, never as a key inside `features` (where `drifting` lives)."""
    client.post("/predict", json={"bearing_id": "b", "signal": make_signal()})

    features = client.get("/monitoring/drift").json()["bearings"]["b"]["features"]

    assert "rms_ratio" not in features


def test_drift_endpoint_reflects_two_bearings_independently(client):
    client.post("/predict", json={"bearing_id": "bearing-a", "signal": make_signal(seed=1)})
    for _ in range(3):
        client.post("/predict", json={"bearing_id": "bearing-b", "signal": make_signal(seed=2)})

    bearings = client.get("/monitoring/drift").json()["bearings"]

    assert set(bearings) == {"bearing-a", "bearing-b"}
    assert bearings["bearing-a"]["file_count"] == 1
    assert bearings["bearing-b"]["file_count"] == 3


# --- predicted_class_counts accumulate through the real /predict path -----------------


def test_predicted_class_counts_accumulate_across_predict_calls(client):
    n_requests = 12
    labels = []
    for i in range(n_requests):
        response = client.post(
            "/predict", json={"bearing_id": "tally-bearing", "signal": make_signal(seed=i)}
        )
        labels.append(response.json()["label"])

    counts = client.get("/monitoring/drift").json()["bearings"]["tally-bearing"][
        "predicted_class_counts"
    ]

    assert sum(counts.values()) == n_requests
    for label in set(labels):
        assert counts[label] == labels.count(label)


# --- PredictResponse's additive drift_status field --------------------------------------


def test_predict_response_includes_drift_status(client):
    response = client.post("/predict", json={"bearing_id": "b", "signal": make_signal()})

    body = response.json()
    assert "drift_status" in body
    assert body["drift_status"] in {"nominal", "drifting"}


def test_predict_response_s_existing_fields_are_unchanged(client):
    """docs/monitoring_design.md Section 3: drift_status is additive -- label,
    baseline_status, and model_notes must be exactly what Issue #84 already established."""
    from src.serving.model_notes import MODEL_NOTES

    body = client.post("/predict", json={"bearing_id": "b", "signal": make_signal()}).json()

    assert set(body) == {"label", "baseline_status", "model_notes", "drift_status"}
    assert body["model_notes"] == MODEL_NOTES


# --- GET /monitoring: the dashboard page -------------------------------------------------


def test_monitoring_page_is_served(client):
    response = client.get("/monitoring")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_monitoring_page_polls_the_drift_endpoint(client):
    body = client.get("/monitoring").text

    assert "/monitoring/drift" in body
    assert "fetch(" in body


def test_monitoring_page_makes_no_external_requests():
    """docs/monitoring_design.md Section 4: no CDN, no external font/script/stylesheet --
    the page must be fully self-contained so it works inside the container with no
    internet access."""
    from src.serving.api import MONITORING_PAGE_PATH

    html = MONITORING_PAGE_PATH.read_text()

    assert "http://" not in html
    assert "https://" not in html
