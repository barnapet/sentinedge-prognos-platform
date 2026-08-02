"""Integration tests for the `/predict` API layer (Issue #84).

These exercise the full HTTP request path -- JSON in, JSON out, through FastAPI's
`TestClient` -- deliberately not re-testing what `tests/test_serving_features.py` (#82)
and `tests/test_train_serving_model.py` (#80) already cover at the module level. What's
specific to *this* layer: the request/response contract, the unconditional `model_notes`
disclosure, the single-worker enforcement, and measured end-to-end latency.
"""
from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.labeling import LABELS
from src.serving.api import create_app
from src.serving.model_notes import MODEL_NOTES
from src.serving.single_worker import SingleWorkerViolation

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_signal(n: int = 64, seed: int = 0, amplitude: float = 0.08) -> list[float]:
    """A small synthetic single-channel snapshot -- fast for tests that send many of
    these; the latency test below uses the real dataset's full 20,480-point window
    instead, since window size is what the <500ms target is actually about."""
    return np.random.default_rng(seed).normal(0.0, amplitude, n).tolist()


@pytest.fixture()
def app(tmp_path):
    return create_app(lock_path=tmp_path / "serving.lock")


@pytest.fixture()
def client(app):
    with TestClient(app) as c:
        yield c


# --- contract: request/response shape ---------------------------------------------

def test_predict_returns_a_valid_response(client):
    response = client.post(
        "/predict", json={"bearing_id": "bearing-1", "signal": make_signal()}
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "baseline_status", "model_notes"}
    assert body["label"] in LABELS
    assert body["baseline_status"] in {"warming_up", "stable"}


def test_predict_accepts_the_documented_payload_shape(client):
    """docs/serving_design.md §1's exact illustrative payload: `bearing_id` + `signal`,
    an optional `timestamp` that is accepted but never affects the result."""
    response = client.post(
        "/predict",
        json={
            "bearing_id": "1st_test-bearing3",
            "signal": make_signal(),
            "timestamp": "2003.10.22.12.00.00",
        },
    )
    assert response.status_code == 200


def test_predict_rejects_an_empty_signal(client):
    response = client.post("/predict", json={"bearing_id": "b", "signal": []})
    assert response.status_code == 422


def test_predict_rejects_a_blank_bearing_id(client):
    response = client.post("/predict", json={"bearing_id": "", "signal": make_signal()})
    assert response.status_code == 422


# --- the unconditional model_notes disclosure (Section 4) -------------------------

def extract_model_notes_from_design_doc() -> str:
    """The same extraction docs/serving_design.md §4's fenced block undergoes to become
    `src.serving.model_notes.MODEL_NOTES` -- reimplemented here independently (not by
    importing that module's own logic) so this test can't pass by construction."""
    text = (REPO_ROOT / "docs" / "serving_design.md").read_text()
    match = re.search(r'"model_notes":\s*"(.+?)"\n```', text, re.DOTALL)
    assert match, "docs/serving_design.md's model_notes code block was not found"
    return " ".join(line.strip() for line in match.group(1).splitlines())


def test_model_notes_constant_matches_the_design_doc_byte_for_byte():
    assert MODEL_NOTES == extract_model_notes_from_design_doc()


def test_predict_response_carries_the_disclosure_byte_for_byte(client):
    response = client.post("/predict", json={"bearing_id": "b", "signal": make_signal()})
    assert response.json()["model_notes"] == MODEL_NOTES


def test_disclosure_is_identical_regardless_of_health_state(client):
    """§4: the disclosure is *static*, never conditional on what the signal looks like --
    a quiet, low-amplitude signal and a wild, high-amplitude one must get the same text."""
    quiet = client.post(
        "/predict", json={"bearing_id": "quiet", "signal": make_signal(amplitude=0.01)}
    )
    wild = client.post(
        "/predict", json={"bearing_id": "wild", "signal": make_signal(amplitude=5.0)}
    )
    assert quiet.json()["model_notes"] == wild.json()["model_notes"] == MODEL_NOTES


# --- cold-start / stable state, and cross-request persistence (issue task 5) ------

def test_first_request_for_a_new_bearing_is_cold_start(client):
    response = client.post("/predict", json={"bearing_id": "fresh-bearing", "signal": make_signal()})
    assert response.json()["baseline_status"] == "warming_up"


def test_state_persists_across_requests_and_locks_at_the_50th(client):
    """The strongest available proof that state survives between HTTP calls: reaching
    `"stable"` is only possible if 50 separate `/predict` calls accumulated into *one*
    shared, persisting `BearingState` -- if the API created a fresh, empty state on every
    request instead of reusing the store, `baseline_status` would read `"warming_up"`
    forever, since no single request would ever see a 50th file. Checked at every one of
    the 50 requests, not just the last, matching the boundary rigor of #82's own test."""
    statuses = [
        client.post(
            "/predict", json={"bearing_id": "run-to-failure", "signal": make_signal(seed=i)}
        ).json()["baseline_status"]
        for i in range(50)
    ]

    assert statuses[:49] == ["warming_up"] * 49
    assert statuses[49] == "stable"


def test_two_bearings_interleaved_through_the_api_stay_independent(client):
    """The API must key state by `bearing_id`, not share one global history -- proven by
    interleaving two bearings' requests and confirming neither's cold-start status leaks
    into the other's."""
    for i in range(5):
        r_a = client.post("/predict", json={"bearing_id": "bearing-a", "signal": make_signal(seed=i)})
        r_b = client.post("/predict", json={"bearing_id": "bearing-b", "signal": make_signal(seed=i + 100)})
        assert r_a.json()["baseline_status"] == "warming_up"
        assert r_b.json()["baseline_status"] == "warming_up"

    extractor = client.app.state.extractor
    assert extractor.store.get_or_create("bearing-a").file_count == 5
    assert extractor.store.get_or_create("bearing-b").file_count == 5


# --- /health ------------------------------------------------------------------------

def test_health_reports_ok_once_the_model_is_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


# --- single-worker enforcement (issue task 3) --------------------------------------

def test_acquiring_the_same_lock_twice_raises(tmp_path):
    from src.serving.single_worker import acquire_single_worker_lock, release_single_worker_lock

    lock_path = tmp_path / "unit.lock"
    first = acquire_single_worker_lock(lock_path)
    try:
        with pytest.raises(SingleWorkerViolation):
            acquire_single_worker_lock(lock_path)
    finally:
        release_single_worker_lock(first)

    # Once released, a new holder can acquire it -- proves this is a live mutual-exclusion
    # lock, not a one-shot "file already exists" check that would wrongly refuse a clean
    # restart of the same single server.
    second = acquire_single_worker_lock(lock_path)
    release_single_worker_lock(second)


def test_a_second_app_instance_cannot_start_while_the_first_is_running(tmp_path):
    """End-to-end proof that `create_app`'s lifespan actually wires the lock in: two apps
    pointed at the same lock path, the second one's startup must fail loudly."""
    lock_path = tmp_path / "serving.lock"
    app1 = create_app(lock_path=lock_path)
    app2 = create_app(lock_path=lock_path)

    with TestClient(app1):
        with pytest.raises(SingleWorkerViolation):
            with TestClient(app2):
                pytest.fail("second app must not complete startup")


def test_uvicorn_refuses_multiple_workers_for_an_already_built_app_object(tmp_path):
    """The first, cheaper enforcement layer `src/serving/main.py` relies on:
    `uvicorn.run` cannot fork additional worker *processes* from a live Python object --
    it needs an import string to re-import fresh in each child -- so it refuses outright
    rather than silently starting one worker. This is `uvicorn`'s behaviour, not this
    project's code, so it is verified here rather than assumed from a changelog."""
    script = (
        "import uvicorn\n"
        "from pathlib import Path\n"
        "from src.serving.api import create_app\n"
        f"app = create_app(lock_path=Path({str(tmp_path / 'sw.lock')!r}))\n"
        "uvicorn.run(app, host='127.0.0.1', port=0, workers=2)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 3
    assert "import string" in result.stderr
    assert "workers" in result.stderr


# --- measured latency (issue task 6) -----------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_real_server(tmp_path: Path) -> tuple[subprocess.Popen, str]:
    """Launch the actual app under a real `uvicorn` process bound to a real TCP port --
    not `TestClient`'s in-process ASGI transport -- so the latency measured below includes
    real HTTP request/response handling, the same path a client hits in the eventual
    container."""
    port = _free_port()
    lock_path = tmp_path / "latency.lock"
    script = (
        "import uvicorn\n"
        "from pathlib import Path\n"
        "from src.serving.api import create_app\n"
        f"app = create_app(lock_path=Path({str(lock_path)!r}))\n"
        f"uvicorn.run(app, host='127.0.0.1', port={port}, workers=1, log_level='warning')\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=REPO_ROOT)
    base_url = f"http://127.0.0.1:{port}"

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server process exited early with code {proc.returncode}")
        try:
            if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                return proc, base_url
        except httpx.TransportError:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError("server did not become healthy within 15s")


def test_predict_latency_meets_the_500ms_target_over_real_http(tmp_path):
    """`docs/PRD.md` §7/§10's target, measured -- not inferred from
    `docs/serving_design.md`'s complexity analysis. Uses a full 20,480-point signal, the
    real single-window size (`docs/PRD.md` §6), over real HTTP against a real `uvicorn`
    process, so both the feature/model compute cost and FastAPI/Starlette's own request
    handling overhead are included."""
    proc, base_url = _start_real_server(tmp_path)
    try:
        rng = np.random.default_rng(0)
        durations_ms = []
        with httpx.Client(base_url=base_url, timeout=5.0) as http_client:
            for i in range(30):
                signal = rng.normal(0.0, 0.08, 20_480).tolist()
                start = time.perf_counter()
                response = http_client.post(
                    "/predict", json={"bearing_id": "latency-probe", "signal": signal}
                )
                durations_ms.append((time.perf_counter() - start) * 1000)
                assert response.status_code == 200

        durations_ms.sort()
        p50 = durations_ms[len(durations_ms) // 2]
        p95 = durations_ms[int(len(durations_ms) * 0.95)]
        worst = durations_ms[-1]
        print(
            f"\n/predict latency over real HTTP, 20,480-point signal, n={len(durations_ms)}: "
            f"p50={p50:.1f}ms p95={p95:.1f}ms max={worst:.1f}ms"
        )

        assert worst < 500, f"slowest request took {worst:.1f}ms, target is <500ms"
    finally:
        proc.terminate()
        proc.wait(timeout=5)
