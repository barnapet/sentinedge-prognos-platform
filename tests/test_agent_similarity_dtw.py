"""Tier-1 tests for the DTW implementation (Issue #140; `docs/agent_design.md` Section 8's
"Section 12's DTW implementation against hand-computed cases", which Issue #138's coverage
audit recorded as the one known tier-1 gap).

**Every expected number in this file is worked out by hand in the comment above the
assertion**, not captured from a previous run. That is the whole reason Section 12 declined
`dtaidistance`/`tslearn` for forty lines of numpy: a library's output can only be compared
against itself, and a characterization test that records whatever the code did on the day it
was written cannot fail when the code is wrong from the start.

No model, no network, no project data -- these run on arrays defined in the file.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.agent.similarity.dtw import (
    BAND_FRACTION,
    band_width,
    banded_subsequence_dtw,
    window_statistics,
    z_normalize,
)


def seq(*values: float) -> np.ndarray:
    """A single-channel `(n, 1)` sequence -- the shape the DTW takes, at the size a person
    can check with a pen."""
    return np.array([[float(value)] for value in values])


# --------------------------------------------------------------------------------------
# The band
# --------------------------------------------------------------------------------------


def test_band_width_is_ten_percent_of_the_query_rounded_up():
    # Section 12: "a Sakoe-Chiba band of 10% of the query length". 10% of 50 is exactly 5;
    # 10% of 51 is 5.1, which rounds up to 6 rather than truncating to 5.
    assert band_width(50) == 5
    assert band_width(51) == 6
    assert BAND_FRACTION == 0.10


def test_band_width_is_never_zero():
    # 10% of 4 is 0.4. A band of 0 forbids warping outright, which would silently turn this
    # into index-aligned Euclidean distance -- the thing Section 12 chose DTW over.
    assert band_width(4) == 1
    assert band_width(1) == 1


# --------------------------------------------------------------------------------------
# Exact alignments: the cases where the answer must be zero
# --------------------------------------------------------------------------------------


def test_identical_sequences_align_at_zero_cost():
    # Each point pairs with its twin: |0-0| + |1-1| + |2-2| = 0, over 3 path cells.
    match = banded_subsequence_dtw(seq(0, 1, 2), seq(0, 1, 2), band=1)

    assert match.distance == 0.0
    assert match.normalized_distance == 0.0
    assert match.path_length == 3
    assert match.matched_index_range == (0, 2)


def test_a_query_is_found_inside_a_longer_reference():
    # Open begin *and* open end: [0,1,2] sits at reference positions 2..4, and the leading
    # [5,5] and trailing [5] cost nothing because the match neither has to start at index 0
    # nor run to the end.
    match = banded_subsequence_dtw(seq(0, 1, 2), seq(5, 5, 0, 1, 2, 5), band=1)

    assert match.distance == 0.0
    assert match.matched_index_range == (2, 4)
    assert match.path_length == 3


def test_a_match_at_the_very_end_of_the_reference_is_reachable():
    # The tail is where every one of these bearings actually fails, so a match that runs to
    # the last reference index must not be cut off. [7,8] is exactly R[3:5].
    match = banded_subsequence_dtw(seq(7, 8), seq(0, 0, 0, 7, 8), band=1)

    assert match.distance == 0.0
    assert match.matched_index_range == (3, 4)


def test_a_repeated_reference_point_is_absorbed_by_one_horizontal_step():
    # R has an extra 1: [0, 1, 1, 2] against query [0, 1, 2]. DTW pairs query point 1 with
    # *both* reference 1s (one horizontal step), so the cost stays 0 -- and the path is one
    # cell longer than the query, which is what path_length records.
    match = banded_subsequence_dtw(seq(0, 1, 2), seq(0, 1, 1, 2), band=1)

    assert match.distance == 0.0
    assert match.path_length == 4
    assert match.matched_index_range == (0, 3)


# --------------------------------------------------------------------------------------
# Non-zero costs, worked out by hand
# --------------------------------------------------------------------------------------


def test_a_two_point_alignment_with_one_mismatch():
    # Q = [0, 2] against R = [0, 1]. Only one alignment exists inside the reference:
    #   |0 - 0| = 0
    #   |2 - 1| = 1
    # distance 1 over 2 path cells -> normalized 1/2.
    match = banded_subsequence_dtw(seq(0, 2), seq(0, 1), band=1)

    assert match.distance == 1.0
    assert match.normalized_distance == 0.5
    assert match.path_length == 2


def test_local_cost_is_the_euclidean_norm_across_channels():
    # One query point at (3, 4), one reference point at (0, 0): sqrt(3**2 + 4**2) = 5.
    # This is the "dependent" multivariate DTW the module docstring names -- channels are
    # combined into one distance, not summed as three independent alignments.
    match = banded_subsequence_dtw(np.array([[3.0, 4.0]]), np.array([[0.0, 0.0]]), band=0)

    assert match.distance == 5.0
    assert match.path_length == 1


def test_normalized_distance_is_the_mean_cost_per_aligned_pair():
    # The identity the threshold is applied to, asserted directly rather than assumed:
    # normalized = distance / path_length, on a case with both values non-trivial.
    match = banded_subsequence_dtw(seq(0, 5), seq(0, 1), band=1)

    # |0-0| + |5-1| = 4, over 2 cells.
    assert match.distance == 4.0
    assert match.path_length == 2
    assert match.normalized_distance == pytest.approx(4.0 / 2.0)


# --------------------------------------------------------------------------------------
# The band actually constrains the warp
# --------------------------------------------------------------------------------------


def test_a_zero_band_forbids_warping_entirely():
    # The same input as the horizontal-step case above, which reaches 0 with band=1. With
    # band=0 no horizontal step is available, so the best it can do is a strict diagonal:
    # starting at 0, |0-0| + |1-1| + |2-1| = 1 over 3 cells.
    match = banded_subsequence_dtw(seq(0, 1, 2), seq(0, 1, 1, 2), band=0)

    assert match.distance == 1.0
    assert match.path_length == 3
    assert match.normalized_distance == pytest.approx(1.0 / 3.0)


def test_a_long_warp_needs_a_wide_enough_band():
    # Q = [1, 5, 9] against R = [1, 5, 5, 5, 5, 9]. The zero-cost alignment pairs query
    # point 1 with all four of R's 5s -- three horizontal steps, which pushes the path 3
    # columns off its own diagonal. So it is reachable only at band >= 3:
    #
    #   band=3: (0,0) (1,1) (1,2) (1,3) (1,4) (2,5)  -> 0 + 0+0+0+0 + 0 = 0, 6 cells
    #   band=2: (0,0) (1,1) (1,2) (1,3) (2,4)        -> |9-5| = 4, 5 cells -> 4/5 = 0.8
    #   band=1: (0,0) (1,1) (1,2) (2,3)              -> |9-5| = 4, 4 cells -> 4/4 = 1.0
    query, reference = seq(1, 5, 9), seq(1, 5, 5, 5, 5, 9)

    wide = banded_subsequence_dtw(query, reference, band=3)
    assert wide.distance == 0.0
    assert wide.path_length == 6
    assert wide.matched_index_range == (0, 5)

    medium = banded_subsequence_dtw(query, reference, band=2)
    assert medium.distance == 4.0
    assert medium.path_length == 5
    assert medium.normalized_distance == pytest.approx(0.8)

    narrow = banded_subsequence_dtw(query, reference, band=1)
    assert narrow.distance == 4.0
    assert narrow.path_length == 4
    assert narrow.normalized_distance == pytest.approx(1.0)


def test_the_band_bounds_how_long_a_matched_stretch_can_be():
    # A consequence worth pinning because it contradicts the illustrative example in
    # Section 12's prose: with a band of w, a length-m query cannot match a stretch longer
    # than m + w. Here m = 4, w = 1, so no match may span more than 5 reference points.
    match = banded_subsequence_dtw(seq(1, 2, 3, 4), seq(1, 2, 2, 2, 2, 2, 3, 4), band=1)

    start, end = match.matched_index_range
    assert end - start + 1 <= 4 + 1


# --------------------------------------------------------------------------------------
# z-normalization
# --------------------------------------------------------------------------------------


def test_z_normalize_centers_and_scales_one_channel():
    # [1, 2, 3]: mean 2, population variance ((1-2)^2 + 0 + (3-2)^2)/3 = 2/3,
    # std = sqrt(2/3) ~= 0.816496580927726. So the values become -1/std, 0, +1/std.
    out = z_normalize(seq(1, 2, 3)).ravel()

    expected = 1.0 / math.sqrt(2.0 / 3.0)
    assert out[0] == pytest.approx(-expected)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(expected)


def test_z_normalize_leaves_a_constant_channel_as_zeros_rather_than_nan():
    # A flat channel has zero variance. Dividing by it would give NaN, and a single NaN
    # poisons every distance the sequence takes part in -- silently, since NaN comparisons
    # are all False. Zeros are the value that contributes nothing instead.
    out = z_normalize(seq(5, 5, 5, 5))

    assert np.all(out == 0.0)
    assert np.all(np.isfinite(out))


def test_z_normalize_scales_channels_independently():
    # Channel 0 spans 1..3, channel 1 spans 100..300. After normalization they are the same
    # shape -- which is the entire point of Section 12's per-sequence normalization: level
    # and spread stop mattering, only shape does.
    out = z_normalize(np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]]))

    assert out[:, 0] == pytest.approx(out[:, 1])


def test_z_normalize_rejects_a_non_2d_input():
    with pytest.raises(ValueError):
        z_normalize(np.array([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------------------
# Per-window reference statistics
# --------------------------------------------------------------------------------------


def test_window_statistics_are_per_start_over_the_query_length():
    # R = [1, 2, 3, 4] with m = 2. Windows: [1,2] [2,3] [3,4] and the truncated tail [4].
    #   means 1.5, 2.5, 3.5, 4.0
    #   population stds 0.5, 0.5, 0.5, 0.0
    mean, std = window_statistics(seq(1, 2, 3, 4), 2)

    assert mean.ravel() == pytest.approx([1.5, 2.5, 3.5, 4.0])
    assert std.ravel() == pytest.approx([0.5, 0.5, 0.5, 0.0])


def test_window_normalization_finds_a_rescaled_copy_of_the_query():
    # The property whole-sequence normalization fails and this exists to provide: the query
    # [0,1,2,1,0] appears in the reference at positions 3..7 as [10,20,30,20,10] -- the same
    # *shape* at a completely different level and scale. Normalizing each candidate window
    # against its own statistics makes that an exact, zero-distance match.
    query = z_normalize(seq(0, 1, 2, 1, 0))
    reference = seq(4, 4, 4, 10, 20, 30, 20, 10, 4, 4)

    match = banded_subsequence_dtw(query, reference, band=1, normalize_windows=True)

    assert match.distance == pytest.approx(0.0, abs=1e-12)
    assert match.matched_index_range == (3, 7)


# --------------------------------------------------------------------------------------
# Failures are raised, not silently scored
# --------------------------------------------------------------------------------------


def test_mismatched_channel_counts_raise():
    with pytest.raises(ValueError, match="channel mismatch"):
        banded_subsequence_dtw(np.zeros((3, 2)), np.zeros((5, 3)))


def test_an_empty_sequence_raises():
    with pytest.raises(ValueError, match="non-empty"):
        banded_subsequence_dtw(np.zeros((0, 1)), np.zeros((5, 1)))


def test_a_query_longer_than_the_reference_raises_rather_than_scoring():
    # Returning a large-but-finite distance here would let a malformed comparison be ranked
    # against real ones; the caller's data is wrong, which is not the same as "dissimilar".
    with pytest.raises(ValueError, match="cannot align"):
        banded_subsequence_dtw(seq(*range(20)), seq(1, 2, 3), band=1)
