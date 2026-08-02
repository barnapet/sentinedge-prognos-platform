"""Tests for the demo sample, the playback client, and the container contract (Issue #86).

What is deliberately *not* here: building the Docker image. That path is covered by its own
workflow (`.github/workflows/docker-demo.yml`), which builds the image and runs a real
playback against a real container -- it can afford to, because the committed sample means
that job needs no dataset at all. These tests stay fast and cover the invariants that a
built image cannot check for itself: that the committed sample really is what its manifest
says, that playback selects the right channel and sends the documented payload, and that
nothing in the container contract quietly reintroduces multiple workers.
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from demo import playback
from demo.sample import (
    MANIFEST_PATH,
    SAMPLE_EXPERIMENT,
    SAMPLE_PATH,
    SAMPLE_STEP,
    load_sample,
)
from src.features.extraction import BASELINE_N_FILES, EXPERIMENTS
from src.labeling import LABELS

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- the committed sample ---------------------------------------------------------

def test_sample_loads_without_pickle():
    """`load_sample` passes `allow_pickle=False`; a committed binary that can only be read
    by unpickling would be an unauditable artifact to ship in a portfolio repo."""
    sample = load_sample()

    assert sample.signals.dtype == np.float32
    assert sample.signals.ndim == 2
    assert len(sample) == len(sample.file_indices) == len(sample.filenames) == len(sample.labels)


def test_sample_snapshots_are_full_length_unmodified_windows():
    """The claim `demo/sample.py` makes: only *which* files are present is reduced, never
    their contents. Every window must still be the full 20,480-point recording
    (docs/PRD.md Section 6), not a truncated or downsampled one."""
    sample = load_sample()
    assert sample.signals.shape[1] == 20_480


def test_sample_files_are_the_documented_decimation_starting_at_file_zero():
    """Starting at 0 and stepping uniformly is required for the replay to be correct:
    docs/serving_design.md Section 1 has the server infer position from arrival order, so
    the first request must really be that bearing's first file."""
    sample = load_sample()

    assert sample.experiment == SAMPLE_EXPERIMENT
    assert sample.channel_idx == EXPERIMENTS[SAMPLE_EXPERIMENT].channel_idx
    assert sample.file_indices[0] == 0
    assert np.all(np.diff(sample.file_indices) == SAMPLE_STEP)


def test_sample_is_long_enough_to_cross_the_cold_start_boundary():
    """A demo that never reaches the 50th file could not show the warming_up -> stable
    transition at all, which is one of Issue #86's acceptance criteria."""
    assert len(load_sample()) > BASELINE_N_FILES


def test_sample_covers_every_health_state():
    """A replay that is all-Normal would not demonstrate anything about the model. This is
    the property that made 2nd_test the right experiment to cut the sample from."""
    labels = set(load_sample().labels.tolist())
    assert labels == set(LABELS)


def test_sample_matches_its_manifest():
    """Same integrity check Issue #80 applies to the committed model artifact: the bytes on
    disk must still hash to what the manifest recorded, so the committed binary is
    verifiable rather than trusted."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    sample = load_sample()

    assert manifest["sha256"] == hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest()
    assert manifest["n_files_sampled"] == len(sample)
    assert manifest["samples_per_file"] == sample.signals.shape[1]
    assert manifest["step"] == SAMPLE_STEP
    assert manifest["channel_idx"] == sample.channel_idx
    counts = {label: int((sample.labels == label).sum()) for label in LABELS}
    assert manifest["label_counts"] == {k: v for k, v in counts.items() if v}


# --- playback: channel selection and payload ---------------------------------------

def write_snapshot(path, columns):
    n_rows = len(columns[0])
    with open(path, "w") as f:
        for row in range(n_rows):
            f.write("\t".join(str(col[row]) for col in columns) + "\n")


@pytest.mark.parametrize("experiment", sorted(EXPERIMENTS))
def test_raw_playback_selects_the_documented_tracked_channel(tmp_path, experiment):
    """docs/serving_design.md Section 1 puts channel selection on the client precisely so
    the server never carries this dataset's per-bearing channel map. If playback read the
    wrong column, the server would have no way to notice -- it would just score a different
    bearing's vibration. Each column carries a distinct constant, so a wrong pick shows up."""
    raw_dir = tmp_path / experiment
    raw_dir.mkdir()
    columns = [[float(j), -float(j), float(j), -float(j)] for j in range(1, 9)]
    write_snapshot(raw_dir / "2003.10.22.12.00.00", columns)

    snapshots = list(playback.iter_raw_snapshots(tmp_path, experiment))

    expected = np.array(columns[EXPERIMENTS[experiment].channel_idx], dtype=np.float32)
    assert len(snapshots) == 1
    assert np.array_equal(snapshots[0].signal, expected)


def test_raw_playback_reads_files_in_chronological_order(tmp_path):
    raw_dir = tmp_path / "2nd_test"
    raw_dir.mkdir()
    for name in ["2004.02.12.11.22.39", "2004.02.12.10.32.39", "2004.02.12.12.12.39"]:
        write_snapshot(raw_dir / name, [[1.0, -1.0, 1.0, -1.0]] * 4)

    names = [s.name for s in playback.iter_raw_snapshots(tmp_path, "2nd_test")]

    assert names == sorted(names)


def test_sample_playback_yields_snapshots_with_ground_truth_for_display():
    snapshots = list(playback.iter_sample_snapshots(limit=3))

    assert len(snapshots) == 3
    assert snapshots[0].file_index == 0
    assert all(s.true_label in LABELS for s in snapshots)


def test_limit_stops_playback_early():
    assert len(list(playback.iter_sample_snapshots(limit=7))) == 7


# --- playback: end-to-end against a real server ------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live_server(tmp_path):
    """A real uvicorn process, as `tests/test_api.py` does -- playback speaks HTTP over a
    socket, so an in-process ASGI transport would not exercise what it actually does."""
    port = _free_port()
    script = (
        "import uvicorn\n"
        "from pathlib import Path\n"
        "from src.serving.api import create_app\n"
        f"app = create_app(lock_path=Path({str(tmp_path / 'pb.lock')!r}))\n"
        f"uvicorn.run(app, host='127.0.0.1', port={port}, workers=1, log_level='warning')\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=REPO_ROOT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        playback.wait_for_server(base_url, timeout_s=30)
        yield base_url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_playback_drives_a_real_server_through_the_cold_start_transition(live_server, capsys):
    """The acceptance criterion, exercised for real: replaying past the 50th file must show
    warming_up before it and stable from it onward, over actual HTTP."""
    playback.main(
        ["--url", live_server, "--interval", "0", "--limit", str(BASELINE_N_FILES + 2)]
    )

    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith("[")]
    assert len(rows) == BASELINE_N_FILES + 2
    assert all("baseline=warming_up" in row for row in rows[: BASELINE_N_FILES - 1])
    assert all("baseline=stable" in row for row in rows[BASELINE_N_FILES - 1 :])
    assert f"stable from request {BASELINE_N_FILES}" in out


def test_playback_surfaces_the_model_notes_disclosure_from_the_server(live_server, capsys):
    """docs/serving_design.md Section 4 requires the disclosure on every response; a demo
    that dropped it would defeat the point of making it unconditional. Compared against the
    served constant, so this cannot pass on a truncated or reworded copy."""
    from src.serving.model_notes import MODEL_NOTES

    playback.main(["--url", live_server, "--interval", "0", "--limit", "2"])

    assert MODEL_NOTES in capsys.readouterr().out


def test_playback_reports_agreement_against_ground_truth(live_server, capsys):
    playback.main(["--url", live_server, "--interval", "0", "--limit", "10"])

    out = capsys.readouterr().out
    assert "Replayed 10 snapshots" in out
    assert re.search(r"Agreement with committed ground-truth labels: \d+/10", out)


# --- the container contract ---------------------------------------------------------

def parse_pins(requirements: str) -> dict[str, str]:
    """`package==version` lines, ignoring comments and blanks."""
    pins = {}
    for line in requirements.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            pins[name.strip()] = version.strip()
    return pins


def test_serving_requirements_are_a_subset_of_requirements_at_identical_pins():
    """requirements.txt stays the single source of truth for versions. Without this check,
    the serving image could silently drift to a different scikit-learn than the one the
    model artifact was fitted under (Issue #80 records those versions in its manifest for
    exactly that reason) -- and the drift would only surface as an unpickling warning, or
    not at all."""
    full = parse_pins((REPO_ROOT / "requirements.txt").read_text())
    serving = parse_pins((REPO_ROOT / "requirements-serving.txt").read_text())

    assert serving, "requirements-serving.txt must pin something"
    for package, version in serving.items():
        assert package in full, f"{package} is in requirements-serving.txt but not requirements.txt"
        assert version == full[package], (
            f"{package} pinned to {version} for serving but {full[package]} in requirements.txt"
        )


def test_serving_requirements_cover_what_the_serving_path_imports():
    """A missing pin here would not fail any test -- it would fail at `docker compose up`,
    on a reviewer's machine. These are the distributions behind the imports in
    src/serving/ and its transitive src/features, src/training reach."""
    serving = parse_pins((REPO_ROOT / "requirements-serving.txt").read_text())

    for package in ["numpy", "scipy", "pandas", "scikit-learn", "fastapi", "uvicorn"]:
        assert package in serving


def strip_comments(text: str) -> str:
    """Drop comment lines and trailing inline comments.

    The prose in these files legitimately *mentions* `--workers` and `gunicorn` while
    explaining why neither is used, so the check below has to look at instructions rather
    than at raw file content.
    """
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        lines.append(line.split(" #", 1)[0])
    return "\n".join(lines)


def test_container_entrypoint_does_not_reintroduce_multiple_workers():
    """docs/serving_design.md Section 2's constraint, at the one layer the code cannot
    enforce for itself: the image's own start command. `python -m src.serving.main` hands
    uvicorn a live app object, which is what makes multi-worker impossible (Issue #84)."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    compose = (REPO_ROOT / "docker-compose.yml").read_text()

    assert 'CMD ["python", "-m", "src.serving.main"]' in dockerfile
    for text, name in [(dockerfile, "Dockerfile"), (compose, "docker-compose.yml")]:
        instructions = strip_comments(text)
        assert "--workers" not in instructions, f"{name} must not pass --workers"
        assert "gunicorn" not in instructions, (
            f"{name} must not use a multi-worker process manager"
        )


def test_dockerignore_excludes_the_raw_dataset():
    """Without this the 6.2 GB dataset would be sent to the daemon as build context on
    every build -- the single biggest avoidable cost in the time-to-demo measurement."""
    assert "data/" in (REPO_ROOT / ".dockerignore").read_text().splitlines()
