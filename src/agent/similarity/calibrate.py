"""Leave-one-out calibration of the no-match threshold (Issue #140,
`docs/agent_design.md` Section 12).

Section 12 specified the procedure: "query a window from one, exclude its own experiment,
observe the distance to the other two -- and the implementation issue must publish that
measurement **and** the honest note that a threshold calibrated on *n* = 3 is coarse."

**The rule was fixed before the numbers were looked at**, per this repo's
commit-to-the-metric-first discipline (`docs/evaluation_protocol.md`, written before any
model existed):

    threshold = the largest leave-one-out best-distance observed over all sampled
                windows of all three held-out experiments, rounded up to 2 decimals.

The reasoning is a floor on what must *not* be refused rather than a fit to what should be
accepted. Every sampled window is a real trajectory from a real bearing on this rig, so
every one of them has to be able to match something in the archive; a threshold below the
worst of them would refuse data of exactly the kind the archive is made of. What that buys
is a threshold that is **deliberately permissive** -- it is a guard against a query whose
shape resembles nothing here, not a fine discriminator between the three references.

`--probe` additionally scores a few synthetic shapes that are *not* bearing trajectories,
which is the check that the threshold refuses anything at all. Those are a sanity probe and
are deliberately **not** calibration inputs: fitting a threshold to hand-made negatives
would be choosing the answer, and n = 3 does not support that pretence.

Running it:

    python -m src.agent.similarity.calibrate [--stride N] [--window N] [--probe]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.agent.similarity.archive import (
    DTW_CHANNELS,
    NO_MATCH_THRESHOLD,
    load_archive,
    query_matrix,
    rank_against_archive,
)
from src.agent.similarity.build_archive import ARCHIVE_PATH

# Section 12's "the last 50 requests".
QUERY_WINDOW = 50
DEFAULT_STRIDE = 10


def raw_trajectories(path=ARCHIVE_PATH) -> dict[str, np.ndarray]:
    """Each experiment's `(n, 3)` channel matrix, **un-normalized**.

    Calibration has to slice raw values and z-normalize each 50-point window on its own,
    because that is what happens at request time: the live query is normalized over the 50
    points it contains. Slicing out of the archive's already-whole-sequence-normalized
    matrix would normalize against statistics a live bearing has no access to, and would
    calibrate the threshold against a quantity the tool never computes.
    """
    frame = pd.read_parquet(path)
    return {
        str(experiment): group.sort_values("file_index", kind="stable")[
            list(DTW_CHANNELS)
        ].to_numpy(dtype=np.float64)
        for experiment, group in frame.groupby("experiment", sort=True)
    }


def leave_one_out_distances(
    held_out: str,
    raw: dict[str, np.ndarray],
    window: int = QUERY_WINDOW,
    stride: int = DEFAULT_STRIDE,
) -> list[float]:
    """Best distance to the *other two* archives, for every sampled window of `held_out`."""
    series = raw[held_out]
    starts = range(0, len(series) - window + 1, stride)
    distances = []
    for start in starts:
        chunk = series[start : start + window]
        query = query_matrix({channel: chunk[:, i] for i, channel in enumerate(DTW_CHANNELS)})
        ranked = rank_against_archive(query, exclude=held_out)
        distances.append(ranked[0]["normalized_distance"])
    return distances


def summarize(distances: list[float]) -> dict[str, float]:
    array = np.asarray(distances, dtype=np.float64)
    return {
        "n_windows": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def synthetic_probes(window: int = QUERY_WINDOW) -> dict[str, np.ndarray]:
    """Shapes that are not bearing trajectories, for the does-it-ever-refuse check.

    Deterministic: the noise probe uses a fixed seed, so this reports the same numbers on
    every run rather than a different anecdote each time.
    """
    index = np.arange(window, dtype=np.float64)
    rng = np.random.default_rng(0)
    return {
        "flat": np.zeros((window, 3)),
        "white_noise": rng.standard_normal((window, 3)),
        "sawtooth": np.column_stack([(index % 5) for _ in range(3)]).astype(np.float64),
        "alternating": np.column_stack([(-1.0) ** index for _ in range(3)]),
        "anti_correlated": np.column_stack([index, -index, index * 0 + np.sin(index)]),
    }


def format_report(
    per_experiment: dict[str, dict[str, float]],
    threshold: float,
    probes: dict[str, list[dict]] | None = None,
) -> str:
    lines = [
        f"Leave-one-out calibration, query window = {QUERY_WINDOW}",
        "",
        f"{'held out':<12}{'windows':>9}{'min':>9}{'median':>9}{'p90':>9}{'p95':>9}{'max':>9}",
    ]
    for experiment, stats in per_experiment.items():
        lines.append(
            f"{experiment:<12}{stats['n_windows']:>9}{stats['min']:>9.3f}"
            f"{stats['median']:>9.3f}{stats['p90']:>9.3f}{stats['p95']:>9.3f}{stats['max']:>9.3f}"
        )
    overall = max(stats["max"] for stats in per_experiment.values())
    lines += [
        "",
        f"worst leave-one-out distance over all held-out windows: {overall:.4f}",
        f"threshold (that, rounded up to 2 dp):                   {threshold:.2f}",
        f"NO_MATCH_THRESHOLD currently in archive.py:             {NO_MATCH_THRESHOLD:.2f}",
        "",
        "n = 3 archived trajectories. This is a coarse threshold and is meant to be read",
        "as one: it refuses shapes unlike anything on this rig, not near-misses between",
        "the three references.",
    ]
    if probes:
        lines += ["", "Sanity probe -- synthetic shapes that are not bearing trajectories:"]
        for name, ranked in probes.items():
            best = ranked[0]
            verdict = "REFUSED" if best["normalized_distance"] > threshold else "accepted"
            lines.append(
                f"  {name:<16} best {best['normalized_distance']:.3f} "
                f"vs {best['experiment']:<10} -> {verdict}"
            )
        lines.append("  (a probe, not a calibration input -- see this module's docstring)")
    return "\n".join(lines)


def calibrate(window: int = QUERY_WINDOW, stride: int = DEFAULT_STRIDE, probe: bool = False):
    raw = raw_trajectories()
    per_experiment = {
        experiment: summarize(leave_one_out_distances(experiment, raw, window, stride))
        for experiment in sorted(raw)
    }
    overall = max(stats["max"] for stats in per_experiment.values())
    threshold = float(np.ceil(overall * 100.0) / 100.0)

    probes = None
    if probe:
        probes = {
            name: rank_against_archive(
                query_matrix({channel: shape[:, i] for i, channel in enumerate(DTW_CHANNELS)})
            )
            for name, shape in synthetic_probes(window).items()
        }
    return per_experiment, threshold, probes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, default=QUERY_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--probe", action="store_true", help="also score synthetic non-bearing shapes")
    args = parser.parse_args(argv)

    load_archive()  # fail loudly here if the artifact is missing, before any work
    per_experiment, threshold, probes = calibrate(args.window, args.stride, args.probe)
    print(format_report(per_experiment, threshold, probes))


if __name__ == "__main__":
    main()
