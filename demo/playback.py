"""Simulated live-feed playback against a running `/predict` (Issue #86).

The client `docs/serving_design.md` Section 1 was designed for, and explicitly deferred to
this issue ("`demo/playback.py`, out of scope for this issue but the consumer this contract
is designed for"). It replays one bearing's run-to-failure history in chronological order,
one snapshot per request, at an accelerated cadence.

Two things this script owns that the server deliberately does not:

- **Channel selection.** `src/features/extraction.py`'s `EXPERIMENTS` dict maps each
  experiment to its tracked bearing's channel index. Section 1's reasoning: that lookup is
  a fixed property of this three-experiment dataset, and making the server carry it would
  tie a general-purpose endpoint to this one dataset. So the client picks the channel and
  sends that one array; the server never learns which column it came from.
- **Position in the stream.** Nothing in the payload says "this is file 37." The server
  infers each bearing's position from the arrival order of requests carrying that
  `bearing_id`, so this script sends files strictly in order, starting at file 0.

By default it replays the committed sample (`demo/sample.py`) and needs no dataset
download. `--raw-dir` replays real snapshot files instead, for anyone who has fetched the
full 6.2 GB dataset and wants a different experiment or full resolution.

    python -m demo.playback                                    # committed sample
    python -m demo.playback --interval 0.2                     # faster than real cadence
    python -m demo.playback --raw-dir data/raw --experiment 1st_test

Only `numpy` is required beyond the standard library -- HTTP goes through `urllib` rather
than `httpx`/`requests` so a reviewer can run this against the container without installing
a client stack.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from demo.sample import SAMPLE_EXPERIMENT, load_sample
from src.features.extraction import EXPERIMENTS, list_snapshot_files, load_channel

DEFAULT_URL = "http://localhost:8000"
# docs/PRD.md Section 8: "one window every ~2s" -- a compressed demo timescale, not a claim
# about sensor sampling rate. The real recordings are ~10 minutes apart.
DEFAULT_INTERVAL_S = 2.0


@dataclass(frozen=True)
class Snapshot:
    """One window to send: the signal, plus context this script prints but never sends."""

    signal: np.ndarray
    name: str
    file_index: int
    true_label: str | None


def iter_sample_snapshots(limit: int | None = None) -> Iterator[Snapshot]:
    """Snapshots from the committed sample (`demo/sample.py`)."""
    sample = load_sample()
    for i in range(len(sample) if limit is None else min(limit, len(sample))):
        yield Snapshot(
            signal=sample.signals[i],
            name=str(sample.filenames[i]),
            file_index=int(sample.file_indices[i]),
            true_label=str(sample.labels[i]),
        )


def iter_raw_snapshots(
    raw_dir: Path, experiment: str, step: int = 1, limit: int | None = None
) -> Iterator[Snapshot]:
    """Snapshots read from real raw files, selecting this experiment's tracked channel."""
    channel_idx = EXPERIMENTS[experiment].channel_idx
    files = list_snapshot_files(raw_dir / experiment)[::step]
    for i, path in enumerate(files[:limit] if limit else files):
        yield Snapshot(
            signal=load_channel(path, channel_idx),
            name=path.name,
            file_index=i * step,
            true_label=None,
        )


def post_predict(base_url: str, bearing_id: str, signal: np.ndarray, timeout: float = 30.0) -> dict:
    """One `/predict` call -- exactly Section 1's payload, nothing more."""
    payload = json.dumps({"bearing_id": bearing_id, "signal": signal.tolist()}).encode()
    request = urllib.request.Request(
        f"{base_url}/predict", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(base_url: str, timeout_s: float = 60.0) -> None:
    """Block until `/health` reports the model is loaded, so playback does not start
    against a container that is still importing scikit-learn."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if json.load(response).get("model_loaded"):
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.5)
    raise SystemExit(f"server at {base_url} did not become healthy within {timeout_s:.0f}s")


def format_row(position: int, total: int, snapshot: Snapshot, response: dict) -> str:
    """One playback line: what was sent, what came back, and whether it was right."""
    true_label = snapshot.true_label
    if true_label is None:
        verdict = ""
    elif true_label == response["label"]:
        verdict = f"  true={true_label}"
    else:
        verdict = f"  true={true_label}  <- MISMATCH"
    return (
        f"[{position:4d}/{total}] {snapshot.name}  "
        f"file={snapshot.file_index:<5d} "
        f"predicted={response['label']:<10s} "
        f"baseline={response['baseline_status']:<11s}{verdict}"
    )


def print_summary(rows: list[tuple[Snapshot, dict]]) -> None:
    """Closing report: the prediction mix, and agreement where ground truth is known."""
    print("\n" + "=" * 78)
    predictions: dict[str, int] = {}
    for _, response in rows:
        predictions[response["label"]] = predictions.get(response["label"], 0) + 1
    print(f"Replayed {len(rows)} snapshots.  Predicted: {predictions}")

    labelled = [(s, r) for s, r in rows if s.true_label is not None]
    if labelled:
        agreed = sum(1 for s, r in labelled if s.true_label == r["label"])
        print(f"Agreement with committed ground-truth labels: {agreed}/{len(labelled)} "
              f"({agreed / len(labelled):.1%})")

    first_stable = next(
        (i for i, (_, r) in enumerate(rows, start=1) if r["baseline_status"] == "stable"), None
    )
    if first_stable:
        print(f"baseline_status: warming_up for requests 1-{first_stable - 1}, "
              f"stable from request {first_stable} onward "
              "(docs/serving_design.md Section 3: the baseline locks on the 50th file)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="base URL of the serving API")
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_S,
        help="seconds between requests (compressed demo timescale, docs/PRD.md Section 8)",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N snapshots")
    parser.add_argument(
        "--bearing-id", default=None, help="defaults to <experiment>-demo"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=None,
        help="replay real snapshot files from this directory instead of the committed sample",
    )
    parser.add_argument(
        "--experiment", default=SAMPLE_EXPERIMENT, choices=sorted(EXPERIMENTS),
        help="which experiment to replay (only with --raw-dir)",
    )
    parser.add_argument(
        "--step", type=int, default=1, help="replay every Nth raw file (only with --raw-dir)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.raw_dir:
        snapshots = list(iter_raw_snapshots(args.raw_dir, args.experiment, args.step, args.limit))
        source = f"{args.raw_dir}/{args.experiment} (every {args.step} file(s))"
        experiment = args.experiment
    else:
        snapshots = list(iter_sample_snapshots(args.limit))
        source = "committed sample (demo/sample_data/, no dataset download needed)"
        experiment = SAMPLE_EXPERIMENT

    bearing_id = args.bearing_id or f"{experiment}-demo"

    print(f"Replaying {len(snapshots)} snapshots from {source}")
    print(f"  -> {args.url}/predict as bearing_id={bearing_id!r}, "
          f"one every {args.interval}s\n")
    wait_for_server(args.url)

    rows: list[tuple[Snapshot, dict]] = []
    for position, snapshot in enumerate(snapshots, start=1):
        response = post_predict(args.url, bearing_id, snapshot.signal)
        rows.append((snapshot, response))
        print(format_row(position, len(snapshots), snapshot, response), flush=True)

        if position == 1:
            # Printed once, from the server's own response rather than restated here --
            # docs/serving_design.md Section 4 requires this disclosure on every response,
            # and a demo that hid it would be exactly the omission that decision guards against.
            print(f"\n  model_notes: {response['model_notes']}\n", flush=True)

        if position < len(snapshots):
            time.sleep(args.interval)

    print_summary(rows)


if __name__ == "__main__":
    main()
