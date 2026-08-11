"""The committed trajectory archive, and ranking a query against it (Issue #140,
`docs/agent_design.md` Section 12).

Loading is cached per path, because `find_similar_historical_pattern` is a per-question
tool call and re-reading 9,464 rows of parquet and re-z-normalizing three references on
every call would be work repeated for a result that cannot change: the archive is a
committed artifact, not runtime state.

The two properties Section 12 calls load-bearing both live here rather than in the tool
body, so they hold for every caller and are testable without an MCP server:

- **`rms` is never a comparison channel.** `DTW_CHANNELS` is the only place the query
  matrix's columns are chosen, and `rms` is not in it. The archive stores the column (it is
  the trajectory record) but nothing here can read it into a distance.
- **A ranking is not a match.** `rank_against_archive` returns all three, ordered; deciding
  whether the best one is close enough is `best_match_or_none`, against a calibrated
  threshold, and it returns `None` rather than the least-bad of three.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.similarity.build_archive import ARCHIVE_MANIFEST_PATH, ARCHIVE_PATH
from src.agent.similarity.dtw import BAND_FRACTION, DTWMatch, banded_subsequence_dtw, z_normalize

# `docs/agent_design.md` Section 12's three channels, in the column order every query
# matrix is built in. Declared here rather than imported from `src.serving.state`, which
# holds the same tuple for the producing side: Section 2 forbids anything under
# `src/agent/mcp/` importing `src/serving/`, and this module sits directly behind that tool.
# `tests/test_agent_similarity_archive.py` asserts the two tuples agree, so the seam is
# pinned by a test instead of by an import that would violate the boundary.
DTW_CHANNELS = ("rms_ratio", "kurtosis", "skewness_smoothed")

# Section 12 says each channel is "z-normalized per sequence", which in a *subsequence*
# search has two readings it does not disambiguate. Issue #140 measured both and recorded
# the resolution in Section 12's addendum; the short version is that this is not a
# stylistic choice:
#
#   True  -- each candidate stretch of the reference is normalized against its own window
#            statistics (the standard z-normalized subsequence search). A window drawn from
#            an experiment matches itself at distance 0.0000, at the correct index range.
#   False -- each reference is normalized once over its whole length. A window drawn from an
#            experiment then fails to find itself (0.998-1.398, at the wrong location on all
#            three), and a *flat* query scores 0.105 -- better than every real bearing window
#            -- because whole-sequence normalization parks the long Normal stretch near zero,
#            so "no shape at all" becomes the best match in the archive.
#
# The rejected arm stays reachable rather than deleted, so `calibrate.py` can reproduce both
# sets of numbers -- the same convention `src/features/candidate_features.py` and
# `src/training/candidate_scalers.py` already follow for evaluated-and-dropped approaches.
NORMALIZE_WINDOWS = True

# Section 12's exact caveat text, which its output contract requires on every result.
CAVEAT = (
    "Ranked among 3 archived experiments from one lab rig at one operating condition; "
    '"most similar" is a rank, not a similarity claim.'
)

# Calibrated by leave-one-out over the three archived trajectories: the largest best-distance
# any of 933 held-out real windows had to the other two experiments was 1.3642, rounded up.
# The rule was fixed before the numbers were read -- `calibrate.py`'s docstring has it, and
# the full per-experiment measurement is in the PR for Issue #140.
#
# The honest reading, which Section 12 requires be stated wherever this number appears: a
# threshold fitted on n = 3 trajectories from one lab rig is coarse. It is set at a floor on
# what must not be refused, so it is deliberately permissive -- a guard against a query whose
# shape resembles nothing on this rig, not a fine discriminator between the three references.
NO_MATCH_THRESHOLD = 1.37

NO_MATCH_REASON = (
    "no reference within threshold: the closest archived trajectory is "
    "{distance:.3f} away and the no-match threshold is {threshold:.2f}"
)


@dataclass(frozen=True)
class ArchivedTrajectory:
    """One experiment's full trajectory, z-normalized once at load time.

    `labels` is positional and parallel to `matrix`'s rows, so a matched index range reads
    straight into it.
    """

    experiment: str
    matrix: np.ndarray
    raw_matrix: np.ndarray
    labels: tuple[str, ...]

    def __len__(self) -> int:
        return self.matrix.shape[0]


def query_matrix(channels: dict[str, list[float]]) -> np.ndarray:
    """Assemble a `(n_points, 3)` query matrix from a channel mapping, z-normalized.

    Every channel in `DTW_CHANNELS` is required and no other key is read. A missing channel
    raises rather than being filled with zeros: a zero column is a *valid-looking* flat
    channel that would quietly shift every distance, which is worse than a loud failure.
    """
    missing = [channel for channel in DTW_CHANNELS if channel not in channels]
    if missing:
        raise KeyError(f"query is missing channel(s) {missing}")
    lengths = {len(channels[channel]) for channel in DTW_CHANNELS}
    if len(lengths) != 1:
        raise ValueError(f"channels have differing lengths: {lengths}")
    columns = []
    for channel in DTW_CHANNELS:
        values = channels[channel]
        if any(value is None for value in values):
            # `null` is how `GET /monitoring/history` reports a non-finite reading, which
            # a degenerate signal can leave in a bearing's history. Refused rather than
            # interpolated or dropped: a hole in a trajectory is missing *shape*, and any
            # filling of it invents the very thing being measured.
            raise ValueError(f"{channel} contains missing readings")
        column = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(column)):
            raise ValueError(f"{channel} contains non-finite readings")
        columns.append(column)
    return z_normalize(np.column_stack(columns))


@lru_cache(maxsize=4)
def load_archive(path: Path = ARCHIVE_PATH) -> tuple[ArchivedTrajectory, ...]:
    """The three archived trajectories, z-normalized, in experiment-name order.

    Raises `FileNotFoundError` when the artifact is absent -- the tool body turns that into
    a plain-language error result, per Section 2's rule that a tool returns failures rather
    than raising them at the model.
    """
    path = Path(path)
    frame = pd.read_parquet(path)
    trajectories = []
    for experiment, group in frame.groupby("experiment", sort=True):
        group = group.sort_values("file_index", kind="stable")
        indices = group["file_index"].to_numpy()
        # Positional row number is used as the reported `matched_index_range`, so it must
        # actually equal `file_index`. If a future archive ever had gaps, the ranges would
        # silently point at the wrong files.
        if indices[0] != 0 or not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError(
                f"{experiment}'s file_index is not a contiguous 0..n-1 range; "
                "matched_index_range would not correspond to file indices"
            )
        raw_matrix = group[list(DTW_CHANNELS)].to_numpy(dtype=np.float64)
        trajectories.append(
            ArchivedTrajectory(
                experiment=str(experiment),
                matrix=z_normalize(raw_matrix),
                raw_matrix=raw_matrix,
                labels=tuple(group["label"].astype(str)),
            )
        )
    return tuple(trajectories)


@lru_cache(maxsize=4)
def archive_source_id(manifest_path: Path = ARCHIVE_MANIFEST_PATH) -> str:
    """Section 12's `trajectory_archive@<hash>`, from the manifest's **content** hash.

    Content rather than file hash on purpose (see `build_archive.py`): a re-encode on a
    different `pyarrow` build changes the parquet's bytes without changing a number, and an
    id that moved for that would invalidate every earlier citation for no real change.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    return f"trajectory_archive@{manifest['content_sha256'][:16]}"


def rank_against_archive(
    query: np.ndarray,
    trajectories: tuple[ArchivedTrajectory, ...] | None = None,
    exclude: str | None = None,
    fraction: float = BAND_FRACTION,
    normalize_windows: bool = NORMALIZE_WINDOWS,
) -> list[dict]:
    """Every reference, scored and ordered best-first. Never filtered by a threshold.

    Args:
        query: A z-normalized `(n_points, 3)` matrix, from `query_matrix`.
        trajectories: Defaults to the committed archive.
        exclude: An experiment to leave out -- the leave-one-out calibration's mechanism,
            and the only caller that passes it. The tool never does.
        fraction: Sakoe-Chiba band as a fraction of query length.

    `label_at_match` is the label at the **end** of the matched range, not the majority over
    it: the question this tool answers is "what did this trajectory turn into", and a range
    that begins Normal and ends Critical is characterized by where it arrived. Named here
    because Section 12's example shows a single label without saying which one it is.
    """
    trajectories = trajectories if trajectories is not None else load_archive()
    ranked = []
    for trajectory in trajectories:
        if exclude is not None and trajectory.experiment == exclude:
            continue
        reference = trajectory.raw_matrix if normalize_windows else trajectory.matrix
        match: DTWMatch = banded_subsequence_dtw(
            query, reference, fraction=fraction, normalize_windows=normalize_windows
        )
        start, end = match.matched_index_range
        ranked.append(
            {
                "experiment": trajectory.experiment,
                "normalized_distance": round(match.normalized_distance, 4),
                "matched_index_range": [start, end],
                "label_at_match": trajectory.labels[end],
            }
        )
    return sorted(ranked, key=lambda row: (row["normalized_distance"], row["experiment"]))


def best_match_or_none(
    ranked: list[dict], threshold: float = NO_MATCH_THRESHOLD
) -> tuple[dict | None, str | None]:
    """`(best_match, no_match_reason)` -- exactly one of the two is `None`.

    Section 12's refusal to always name a winner, in the one place every caller goes
    through: "always returning a winner out of three is how 'most resembles' quietly
    becomes a false claim about an unfamiliar bearing."
    """
    if not ranked:
        return None, "no archived references were available to compare against"
    best = ranked[0]
    if best["normalized_distance"] > threshold:
        return None, NO_MATCH_REASON.format(
            distance=best["normalized_distance"], threshold=threshold
        )
    return best, None
