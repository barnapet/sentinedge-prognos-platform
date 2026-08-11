"""Tier-1 tests for the trajectory archive and the ranking over it (Issue #140,
`docs/agent_design.md` Section 12).

These run against the **committed** `models/trajectory_archive.parquet` rather than a
fixture, which is the point of committing it: no raw dataset, no `data/processed/`, no
network -- so unlike most things that touch this project's data, these run on every PR in
CI exactly as they do locally.
"""
from __future__ import annotations

import json

import pytest

from src.agent.similarity.archive import (
    CAVEAT,
    DTW_CHANNELS,
    NO_MATCH_THRESHOLD,
    NORMALIZE_WINDOWS,
    archive_source_id,
    best_match_or_none,
    load_archive,
    query_matrix,
    rank_against_archive,
)
from src.agent.similarity.build_archive import (
    ARCHIVE_MANIFEST_PATH,
    ARCHIVE_PATH,
    ARCHIVE_COLUMNS,
    build_archive_frame,
    content_hash,
    file_hash,
)
from src.serving.state import TRAJECTORY_CHANNELS

EXPECTED_ROWS = {"1st_test": 2156, "2nd_test": 984, "3rd_test": 6324}


@pytest.fixture(scope="module")
def archive():
    return load_archive()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(ARCHIVE_MANIFEST_PATH.read_text())


# --------------------------------------------------------------------------------------
# The channel selection -- Section 12's most important exclusion
# --------------------------------------------------------------------------------------


def test_raw_rms_is_not_a_comparison_channel():
    """Section 12's sharpest decision: raw RMS amplitude does not transfer between bearings
    (`docs/model_training_decision.md` Section 3a -- `1st_test`'s *minimum* exceeds both
    other experiments' *means*), so a distance dominated by it would report a **scale**
    finding as a **shape** finding, with a plausible number attached."""
    assert "rms" not in DTW_CHANNELS
    assert set(DTW_CHANNELS) == {"rms_ratio", "kurtosis", "skewness_smoothed"}


def test_a_query_cannot_smuggle_rms_in_as_a_channel():
    # Extra keys are ignored rather than read, and the matrix is built in DTW_CHANNELS
    # order regardless of what the caller's dict happened to contain.
    channels = {channel: [1.0, 2.0, 3.0] for channel in DTW_CHANNELS}
    with_rms = {**channels, "rms": [99.0, 99.0, 99.0]}

    assert query_matrix(with_rms) == pytest.approx(query_matrix(channels))


def test_the_agent_and_serving_channel_tuples_agree():
    """`src/agent/` may not import `src/serving/` (Section 2), so the two sides declare this
    tuple separately. That makes drift possible, so it is pinned here -- the same
    doc-vs-code discipline `tests/test_api.py` applies to `model_notes`."""
    assert DTW_CHANNELS == TRAJECTORY_CHANNELS


# --------------------------------------------------------------------------------------
# The committed artifact
# --------------------------------------------------------------------------------------


def test_the_archive_holds_every_row_of_all_three_experiments(archive):
    assert {trajectory.experiment for trajectory in archive} == set(EXPECTED_ROWS)
    assert {t.experiment: len(t) for t in archive} == EXPECTED_ROWS
    assert sum(len(t) for t in archive) == 9464


def test_every_row_carries_a_label(archive):
    for trajectory in archive:
        assert len(trajectory.labels) == len(trajectory)
        assert set(trajectory.labels) <= {"Normal", "Degrading", "Critical"}


def test_the_committed_parquet_matches_its_manifests_file_hash(manifest):
    """What makes a non-human-diffable committed artifact auditable: the manifest's hash is
    checked against the bytes actually in the tree, so an edited archive fails here rather
    than being discovered by a wrong answer later."""
    assert file_hash(ARCHIVE_PATH) == manifest["archive_sha256"]


def test_the_manifest_records_the_row_counts_it_claims(manifest):
    assert manifest["n_rows"] == 9464
    assert manifest["n_rows_per_experiment"] == EXPECTED_ROWS
    assert manifest["columns"] == ARCHIVE_COLUMNS


def test_the_content_hash_survives_a_reencode(manifest, tmp_path):
    """`content_sha256` hashes the *data*, not the file. Re-reading the archive and writing
    it out again changes the parquet's bytes (it embeds a writer version) but must not
    change the content hash -- which is why the citable `source_id` is derived from that one
    and not from `archive_sha256`."""
    import pandas as pd

    reencoded = tmp_path / "again.parquet"
    frame = pd.read_parquet(ARCHIVE_PATH)
    frame.to_parquet(reencoded, index=False)

    assert content_hash(pd.read_parquet(reencoded)) == manifest["content_sha256"]


def test_the_source_id_is_the_content_hash(manifest):
    assert archive_source_id() == f"trajectory_archive@{manifest['content_sha256'][:16]}"


def test_rebuilding_the_frame_from_the_archive_is_stable():
    """`build_archive_frame` is idempotent over its own output -- so a regenerated archive
    differs from the committed one only if the *source data* changed, not because the
    builder reorders or retypes anything on a second pass."""
    import pandas as pd

    once = pd.read_parquet(ARCHIVE_PATH)
    twice = build_archive_frame(once)

    assert content_hash(twice) == content_hash(once)


# --------------------------------------------------------------------------------------
# The metric on real data
# --------------------------------------------------------------------------------------


def _window(trajectory, start: int, length: int = 50) -> dict[str, list[float]]:
    """A raw (un-normalized) slice of an archived trajectory, in the shape a live query
    arrives in -- which is what makes the self-match assertion below meaningful."""
    chunk = trajectory.raw_matrix[start : start + length]
    return {channel: list(chunk[:, i]) for i, channel in enumerate(DTW_CHANNELS)}


@pytest.mark.parametrize("experiment", sorted(EXPECTED_ROWS))
def test_a_window_drawn_from_an_experiment_matches_itself_exactly(archive, experiment):
    """The correctness invariant any subsequence matcher must satisfy, and the measurement
    that settled Section 12's normalization ambiguity (Issue #140): a stretch taken out of a
    reference must be found *in that reference*, at *its own indices*, at distance 0.

    Whole-sequence normalization fails this on all three experiments -- see
    `test_whole_reference_normalization_cannot_find_a_window_in_its_own_source` below, kept
    as the recorded evidence for the rejected arm.
    """
    trajectory = next(t for t in archive if t.experiment == experiment)
    ranked = rank_against_archive(query_matrix(_window(trajectory, 800)), archive)
    own = next(row for row in ranked if row["experiment"] == experiment)

    assert own["normalized_distance"] == pytest.approx(0.0, abs=1e-9)
    assert own["matched_index_range"] == [800, 849]
    assert ranked[0]["experiment"] == experiment


def test_whole_reference_normalization_cannot_find_a_window_in_its_own_source(archive):
    """The rejected arm, kept and tested rather than deleted -- the convention
    `src/features/candidate_features.py` and `src/training/candidate_scalers.py` already
    set. This is the measurement, not an opinion: normalizing each reference once over its
    whole length leaves a window unable to locate itself."""
    trajectory = next(t for t in archive if t.experiment == "2nd_test")
    ranked = rank_against_archive(
        query_matrix(_window(trajectory, 800)), archive, normalize_windows=False
    )
    own = next(row for row in ranked if row["experiment"] == "2nd_test")

    assert own["normalized_distance"] > 0.5
    assert own["matched_index_range"] != [800, 849]


def test_ranking_returns_every_reference_ordered_best_first(archive):
    ranked = rank_against_archive(query_matrix(_window(archive[0], 400)), archive)

    assert len(ranked) == 3
    assert {row["experiment"] for row in ranked} == set(EXPECTED_ROWS)
    distances = [row["normalized_distance"] for row in ranked]
    assert distances == sorted(distances)


def test_exclude_leaves_that_experiment_out(archive):
    """The leave-one-out calibration's mechanism. The tool never passes it."""
    ranked = rank_against_archive(
        query_matrix(_window(archive[0], 400)), archive, exclude="2nd_test"
    )

    assert {row["experiment"] for row in ranked} == set(EXPECTED_ROWS) - {"2nd_test"}


def test_label_at_match_is_the_label_at_the_end_of_the_matched_range(archive):
    """Named because Section 12's example shows a single label without saying which one:
    the range's *end*, since the question is what the trajectory turned into."""
    trajectory = next(t for t in archive if t.experiment == "3rd_test")
    ranked = rank_against_archive(query_matrix(_window(trajectory, 800)), archive)
    row = next(r for r in ranked if r["experiment"] == "3rd_test")

    assert row["label_at_match"] == trajectory.labels[row["matched_index_range"][1]]


# --------------------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------------------


def test_the_calibrated_threshold_is_the_committed_one():
    """Pins the number the leave-one-out calibration produced (worst held-out distance
    1.3642, rounded up). If someone re-runs `calibrate.py` and the archive has changed, this
    fails rather than the threshold quietly drifting away from its published measurement."""
    assert NO_MATCH_THRESHOLD == 1.37
    assert NORMALIZE_WINDOWS is True


def test_a_distant_query_gets_no_best_match_but_keeps_its_ranking():
    """Section 12: "always returning a winner out of three is how 'most resembles' quietly
    becomes a false claim about an unfamiliar bearing." The ranking survives the refusal, so
    the answer can say what it was closest to *and* that it was not close enough."""
    ranked = [
        {"experiment": "2nd_test", "normalized_distance": 1.80, "matched_index_range": [0, 49]},
        {"experiment": "1st_test", "normalized_distance": 1.95, "matched_index_range": [0, 49]},
    ]
    best, reason = best_match_or_none(ranked)

    assert best is None
    assert "no reference within threshold" in reason
    assert "1.80" in reason and "1.37" in reason


def test_a_close_query_gets_the_best_match_and_no_reason():
    ranked = [
        {"experiment": "2nd_test", "normalized_distance": 0.41, "matched_index_range": [0, 49]}
    ]
    best, reason = best_match_or_none(ranked)

    assert best is ranked[0]
    assert reason is None


def test_an_empty_ranking_refuses_rather_than_indexing_into_nothing():
    best, reason = best_match_or_none([])

    assert best is None
    assert "no archived references" in reason


def test_the_threshold_boundary_is_inclusive():
    # Exactly at the threshold is a match; the refusal is "exceeds", not "reaches".
    at = [{"experiment": "2nd_test", "normalized_distance": NO_MATCH_THRESHOLD}]
    just_over = [{"experiment": "2nd_test", "normalized_distance": NO_MATCH_THRESHOLD + 0.01}]

    assert best_match_or_none(at)[0] is not None
    assert best_match_or_none(just_over)[0] is None


# --------------------------------------------------------------------------------------
# Query validation
# --------------------------------------------------------------------------------------


def test_a_missing_channel_raises_rather_than_being_zero_filled():
    """A zero column is a *valid-looking* flat channel that would quietly shift every
    distance -- worse than a loud failure."""
    with pytest.raises(KeyError, match="missing channel"):
        query_matrix({"rms_ratio": [1.0, 2.0], "kurtosis": [1.0, 2.0]})


def test_channels_of_differing_lengths_raise():
    with pytest.raises(ValueError, match="differing lengths"):
        query_matrix(
            {"rms_ratio": [1.0, 2.0], "kurtosis": [1.0], "skewness_smoothed": [1.0, 2.0]}
        )


def test_a_missing_reading_raises_rather_than_being_interpolated():
    """`null` is how `GET /monitoring/history` reports a non-finite reading. A hole in a
    trajectory is missing *shape*, and filling it invents the thing being measured."""
    channels = {channel: [1.0, 2.0, 3.0] for channel in DTW_CHANNELS}
    channels["kurtosis"] = [1.0, None, 3.0]

    with pytest.raises(ValueError, match="missing readings"):
        query_matrix(channels)


def test_a_non_finite_reading_raises():
    channels = {channel: [1.0, 2.0, 3.0] for channel in DTW_CHANNELS}
    channels["rms_ratio"] = [1.0, float("nan"), 3.0]

    with pytest.raises(ValueError, match="non-finite"):
        query_matrix(channels)


def test_the_caveat_carries_the_sample_size():
    """Section 12 requires `n_references: 3` and this text on every result: "a similarity
    claim over three lab bearings that arrives without its sample size is a claim dressed up
    as more than it is"."""
    assert "3 archived experiments" in CAVEAT
    assert "is a rank, not a similarity claim" in CAVEAT
