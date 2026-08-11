"""Banded subsequence DTW in numpy (Issue #140, `docs/agent_design.md` Section 12).

Section 12 decided the metric and the reasons; this module is that decision in code and
adds none of its own beyond the implementation choices called out below.

**What it computes.** Open-begin/open-end subsequence DTW: the best alignment of a short
`query` against *any* contiguous stretch of a long `reference`, under a Sakoe-Chiba band of
`BAND_FRACTION` of the query length. Multivariate over the three channels Section 12
chose, with the per-pair local cost being the Euclidean norm across channels (the standard
"dependent" multivariate DTW, DTW_D -- one warping path shared by all channels, which is
the right reading here because the three channels are three views of one bearing's single
degradation, not three independently drifting signals).

**No new dependency.** Section 12's reasoning, and this repo's precedent twice over
(`docs/class_imbalance_decision.md` Section 2, `docs/frequency_domain_decision.md`). The
concrete payoff is that the recurrence below is checkable against cases small enough to
work out by hand, which `tests/test_agent_similarity_dtw.py` does -- a library's output can
only be compared against itself.

## How the band and the free start are made to coexist

This is the one part worth reading carefully, because the obvious formulation does not
work. A Sakoe-Chiba band constrains how far an alignment may stray from the diagonal --
but with an *open begin* there is no single diagonal to measure against: each candidate
start column has its own. Carrying a "where did this path start" array alongside the cost
matrix and filtering on it would work and is what a naive implementation does, but the
filter depends on an argmin, so it cannot be vectorized and runs ~1M Python-level steps for
one real query.

Instead the DP is re-parametrized. Write `j = s + i + o`, where `s` is the path's start
column, `i` the query index and `o` the offset from that path's own diagonal. The band is
then exactly `|o| <= w`, a *fixed, small* range independent of `s` -- so `s` becomes a free
axis that numpy vectorizes over, and the recurrence runs as `m x (2w + 1)` vector
operations rather than `m x n` scalar ones. The three DTW steps land at:

    (i-1, j-1)  diagonal   -> same `o`, previous row
    (i-1, j)    vertical   -> `o + 1`, previous row      (query advances, reference does not)
    (i,   j-1)  horizontal -> `o - 1`, same row          (reference advances, query does not)

Only the horizontal step reaches inside the current row, which is why `o` is stepped
in increasing order and everything else is a whole-array operation.

One consequence worth stating rather than discovering: because the band bounds `o`, the
matched stretch of the reference is bounded to `m - w .. m + w` points. A 50-point query
cannot match a 62-point stretch. Section 12's worked example in the design doc is an
illustration with hand-written numbers, not output of this function.

## Normalization

`distance` is the raw accumulated cost, which grows with path length and so is not
comparable between references. `normalized_distance` divides it by the number of aligned
pairs on the path, making it a **mean per-pair Euclidean distance in z-units** -- an
interpretable quantity, and the one the archive ranks and thresholds on.

The DP minimizes raw accumulated cost and the free-end choice then minimizes the
normalized value among the end cells. That is the standard practical approximation:
optimizing a length-normalized DTW objective directly is a different and harder problem,
and this composition can in principle return a normalized value above the true optimum.
Said plainly rather than left implicit, since the number it produces is thresholded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# `docs/agent_design.md` Section 12: "a Sakoe-Chiba band of 10% of the query length".
BAND_FRACTION = 0.10


def band_width(query_length: int, fraction: float = BAND_FRACTION) -> int:
    """The Sakoe-Chiba half-width for a query of this length, at least 1.

    Rounded up: a band of 0 forbids warping entirely, which would make this plain
    index-aligned Euclidean distance -- the thing Section 12 chose DTW over.
    """
    if query_length < 1:
        raise ValueError(f"query_length must be positive, got {query_length}")
    return max(1, math.ceil(fraction * query_length))


def z_normalize(sequence: np.ndarray) -> np.ndarray:
    """Per-channel z-normalization of one `(n_points, n_channels)` sequence.

    Section 12's "each channel is z-normalized per sequence before the comparison":
    without it the distance is dominated by level rather than shape, and for `rms_ratio`
    level encodes per-bearing *severity* (the `critical_multiple` values span 1.932 /
    2.866 / 3.049) rather than a common scale.

    Mean and variance go through `math.fsum` rather than `numpy`'s pairwise summation, for
    the reason this repo has now had to fix twice (`src.serving.state.window_mean`, Issue
    #82/#83; `compute_drift_baseline._compensated_mean_std`, Issue #93): a plain `.mean()`
    is a property of the installed numpy build as well as of the input, and the archived
    references are normalized once here and then compared against a committed threshold.

    A constant channel (zero variance) normalizes to all zeros rather than raising or
    producing `NaN`. A flat channel genuinely carries no shape information, and zeros are
    the value that contributes nothing to the distance -- whereas a `NaN` would silently
    poison every comparison involving that sequence.
    """
    seq = np.asarray(sequence, dtype=np.float64)
    if seq.ndim != 2:
        raise ValueError(f"expected a 2-D (n_points, n_channels) array, got shape {seq.shape}")
    if seq.shape[0] < 1:
        raise ValueError("cannot z-normalize an empty sequence")

    out = np.empty_like(seq)
    n = seq.shape[0]
    for channel in range(seq.shape[1]):
        column = seq[:, channel]
        mean = math.fsum(column) / n
        centered = column - mean
        variance = math.fsum(centered * centered) / n
        out[:, channel] = centered / math.sqrt(variance) if variance > 0.0 else centered
    return out


@dataclass(frozen=True)
class DTWMatch:
    """Where a query best matched inside a reference, and how well.

    `matched_index_range` is inclusive at both ends and indexes the reference as it was
    passed in.
    """

    distance: float
    normalized_distance: float
    path_length: int
    matched_index_range: tuple[int, int]


def _local_cost(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`(m, n)` matrix of per-pair Euclidean distances across channels."""
    diff = query[:, None, :] - reference[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def window_statistics(reference: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-start mean and std of `reference[s : s + m]`, for every start `s`.

    Computed directly over each window (via `sliding_window_view`) rather than from
    cumulative sums of `x` and `x**2`: the `E[x**2] - E[x]**2` shortcut is the textbook
    catastrophic-cancellation formula, and this repo has twice paid for summation-precision
    shortcuts already (Issues #82/#83, #93). `n * m` is 6,324 x 50 here -- small enough that
    the exact route costs nothing worth having.

    Starts within `m` of the end use a truncated window (`reference[s:]`), so the tail --
    which is where every one of these bearings actually fails -- stays reachable instead of
    being cut off by the last full window.
    """
    n = reference.shape[0]
    mean = np.empty_like(reference)
    std = np.empty_like(reference)
    full = min(n - m + 1, n) if n >= m else 0
    if full > 0:
        windows = np.lib.stride_tricks.sliding_window_view(reference, m, axis=0)[:full]
        mean[:full] = windows.mean(axis=-1)
        std[:full] = windows.std(axis=-1)
    for s in range(max(full, 0), n):
        window = reference[s:]
        mean[s] = window.mean(axis=0)
        std[s] = window.std(axis=0)
    return mean, std


def banded_subsequence_dtw(
    query: np.ndarray,
    reference: np.ndarray,
    band: int | None = None,
    fraction: float = BAND_FRACTION,
    normalize_windows: bool = False,
) -> DTWMatch:
    """Best open-begin/open-end alignment of `query` inside `reference`, under the band.

    Both arrays are `(n_points, n_channels)` and are used **as given** -- z-normalization
    is `z_normalize`'s job and is deliberately not done here, so the hand-computed tests
    can exercise the recurrence on values they choose.

    Args:
        query: The short sequence to locate.
        reference: The long sequence to locate it in.
        band: Sakoe-Chiba half-width. Defaults to `band_width(len(query), fraction)`.
        fraction: Used only when `band` is None.

    Raises:
        ValueError: on mismatched channel counts, an empty input, or a query so much
            longer than the reference that no alignment fits inside the band. Raised
            rather than returned as an infinite distance: each is a caller error about the
            shape of the data, not a "these are dissimilar" answer, and returning a number
            for it would let a malformed comparison be ranked against real ones.
    """
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if q.ndim != 2 or r.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got {q.shape} and {r.shape}")
    if q.shape[1] != r.shape[1]:
        raise ValueError(f"channel mismatch: query has {q.shape[1]}, reference {r.shape[1]}")
    m, n = q.shape[0], r.shape[0]
    if m < 1 or n < 1:
        raise ValueError(f"both sequences must be non-empty, got lengths {m} and {n}")

    w = band_width(m, fraction) if band is None else band
    if w < 0:
        raise ValueError(f"band must be non-negative, got {w}")
    if m - w > n:
        raise ValueError(
            f"query of length {m} cannot align inside a reference of length {n} "
            f"under a band of {w}"
        )

    if normalize_windows:
        window_mean, window_std = window_statistics(r, m)
        safe_std = np.where(window_std > 0.0, window_std, 1.0)
        cost = None
    else:
        cost = _local_cost(q, r)
    n_offsets = 2 * w + 1

    def cost_row(i: int, offset: int) -> np.ndarray:
        """Local cost of aligning `query[i]` with `reference[s + i + offset]`, for every
        start `s`; `inf` where that column falls off the reference.

        Under `normalize_windows` the reference point is z-normalized against **its own
        candidate window's** statistics -- which depend on `s` alone, so this stays one
        vector operation per `(i, offset)` rather than becoming a per-start loop.
        """
        row = np.full(n, np.inf)
        shift = i + offset
        # `s` is bounded twice over: it indexes this array (0 <= s < n) and its column
        # `s + shift` must land inside the reference. A negative shift makes the second
        # bound the looser one, so both are needed.
        lo, hi = max(0, -shift), min(n, n - shift)
        if hi > lo:
            if cost is not None:
                row[lo:hi] = cost[i, lo + shift : hi + shift]
            else:
                points = r[lo + shift : hi + shift]
                points = (points - window_mean[lo:hi]) / safe_std[lo:hi]
                diff = points - q[i]
                row[lo:hi] = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        return row

    # Row 0: a path *starts* at (0, s), so its offset is 0 by definition. Every other
    # offset is unreachable, which is what makes the begin open -- the alternative
    # (allowing horizontal steps within row 0) would re-derive the same alignments with an
    # extra leading cost already counted under a later start.
    acc = np.full((n_offsets, n), np.inf)
    length = np.zeros((n_offsets, n), dtype=np.int64)
    acc[w] = cost_row(0, 0)
    length[w] = np.where(np.isfinite(acc[w]), 1, 0)

    unreachable = np.full(n, np.inf)
    no_steps = np.zeros(n, dtype=np.int64)

    for i in range(1, m):
        prev_acc, prev_length = acc, length
        acc = np.full((n_offsets, n), np.inf)
        length = np.zeros((n_offsets, n), dtype=np.int64)
        # Increasing `o`, because the horizontal step reads this row at `o - 1`.
        for oi in range(n_offsets):
            diagonal_acc, diagonal_length = prev_acc[oi], prev_length[oi]
            vertical_acc, vertical_length = (
                (prev_acc[oi + 1], prev_length[oi + 1])
                if oi + 1 < n_offsets
                else (unreachable, no_steps)
            )
            horizontal_acc, horizontal_length = (
                (acc[oi - 1], length[oi - 1]) if oi - 1 >= 0 else (unreachable, no_steps)
            )

            candidates = np.stack([diagonal_acc, vertical_acc, horizontal_acc])
            candidate_lengths = np.stack([diagonal_length, vertical_length, horizontal_length])
            chosen = np.argmin(candidates, axis=0)[None, :]
            best = np.take_along_axis(candidates, chosen, axis=0)[0]
            best_length = np.take_along_axis(candidate_lengths, chosen, axis=0)[0]

            acc[oi] = best + cost_row(i, oi - w)
            length[oi] = np.where(np.isfinite(acc[oi]), best_length + 1, 0)

    # Free end: every (offset, start) cell in the final row is a candidate ending.
    reachable = np.isfinite(acc) & (length > 0)
    if not reachable.any():
        raise ValueError(
            f"no alignment of a length-{m} query fits inside a length-{n} reference "
            f"under a band of {w}"
        )
    normalized = np.where(reachable, acc / np.maximum(length, 1), np.inf)
    # `argmin` on a C-ordered array breaks ties toward the smallest offset and then the
    # earliest start -- arbitrary but fixed, so the same inputs always report the same
    # match rather than one that moves with the numpy version.
    offset_index, start = np.unravel_index(int(np.argmin(normalized)), normalized.shape)
    offset = int(offset_index) - w
    end = int(start) + (m - 1) + offset
    return DTWMatch(
        distance=float(acc[offset_index, start]),
        normalized_distance=float(normalized[offset_index, start]),
        path_length=int(length[offset_index, start]),
        matched_index_range=(int(start), end),
    )
