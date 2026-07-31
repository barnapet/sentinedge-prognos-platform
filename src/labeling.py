"""Health-state labeling logic for bearing vibration data.

Extracted from notebooks/02_health_state_labeling.ipynb during M1.5-Housekeeping
(Issue #19), so the core labeling logic can be unit tested independently of full
notebook execution. See notebooks/02_health_state_labeling.ipynb for the full
derivation narrative behind these constants and the threshold rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["Normal", "Degrading", "Critical"]

# Carried over from Issue #9 (do not change without re-running #9's analysis).
ONSET_MULTIPLE = 1.3

# Derived in notebooks/02_health_state_labeling.ipynb, Section 3
# ("near-zero" = raw RMS below this fraction of the recent Critical level).
COLLAPSE_FRACTION = 0.2

# Investigated in Issue #20: 1st_test's rms_ratio dips to 0.036 below ONSET_MULTIPLE
# during its onset-boundary flapping (files 1906-1999) before resuming its climb --
# this is the deepest reversal-causing dip observed across all three experiments.
# 0.05 clears that dip with headroom and was validated (via the cached #9/#10 RMS
# data) to fully suppress reverts in 1st_test and 2nd_test's own smaller blip,
# without shifting onset/critical detection timing in any of the three experiments,
# across a 0.04-0.10 sweep. Not re-derivable from this module alone -- see
# docs/label_hysteresis_decision.md for the full investigation.
HYSTERESIS_MARGIN = 0.05


def _hysteresis_ranks(
    ratio: np.ndarray,
    onset_multiple: float,
    critical_multiple: float,
    margin: float,
) -> np.ndarray:
    """Sequentially confirm a health-state rank (0=Normal, 1=Degrading, 2=Critical)
    per row, applying hysteresis to downward moves only.

    Upward moves (ratio crosses further above a boundary) are immediate and can skip
    a rank -- a severe, sudden fault can jump straight to Critical-level ratios, and
    this must not be delayed. Downward moves relax only one rank at a time, and only
    once the ratio has dropped `margin` past the boundary it's re-crossing (not just
    back below it) -- this is the hysteresis band that prevents noise straddling a
    boundary from flipping the label back and forth (Issue #20). A single-file drop
    of more than one rank (e.g. a rig-shutdown RMS collapse) is intentionally NOT
    followed rank-by-rank here; that artifact is handled separately by the
    rig-shutdown override below, which operates on raw RMS rather than this ratio.
    """
    raw_rank = np.where(ratio > critical_multiple, 2, np.where(ratio > onset_multiple, 1, 0))
    confirmed = np.empty(len(ratio), dtype=int)
    confirmed[0] = raw_rank[0]

    for i in range(1, len(ratio)):
        prev = confirmed[i - 1]
        if raw_rank[i] > prev:
            confirmed[i] = raw_rank[i]
        elif raw_rank[i] < prev:
            exit_threshold = {2: critical_multiple, 1: onset_multiple}.get(prev)
            if exit_threshold is not None and ratio[i] <= exit_threshold - margin:
                confirmed[i] = prev - 1
            else:
                confirmed[i] = prev
        else:
            confirmed[i] = prev

    return confirmed


def assign_labels(
    df: pd.DataFrame,
    critical_multiple: float,
    collapse_fraction: float = COLLAPSE_FRACTION,
    onset_multiple: float = ONSET_MULTIPLE,
    hysteresis_margin: float = HYSTERESIS_MARGIN,
) -> pd.DataFrame:
    """Label each file Normal / Degrading / Critical, then apply the rig-shutdown override.

    Threshold rule (on `rms_ratio`), with hysteresis on downward moves (Issue #20):
        Normal -> Degrading : rms_ratio > onset_multiple (immediate)
        Degrading -> Normal : rms_ratio <= onset_multiple - hysteresis_margin
        Degrading -> Critical : rms_ratio > critical_multiple (immediate)
        Critical -> Degrading : rms_ratio <= critical_multiple - hysteresis_margin
    See `_hysteresis_ranks` for the full rationale. Requires `df` in row/time order,
    same as the rig-shutdown override below -- both passes are sequential, not
    row-independent.

    Rig-shutdown override (post-processing pass, in row order):
        If the previous row's label is Critical, and this row's raw RMS collapses
        below `collapse_fraction` of the mean raw RMS over the preceding Critical
        run, force this row's label to Critical regardless of the threshold result.
        Collapse is detected on raw RMS (`rms`), not `rms_ratio`, because that is the
        channel the shutdown artifact actually appears in.

    Args:
        df: Must contain `rms_ratio` and `rms` columns.
        critical_multiple: Per-bearing Critical threshold (see notebook Section 2b).
        collapse_fraction: Override sensitivity; defaults to the value derived in
            the notebook.
        onset_multiple: Normal/Degrading boundary; defaults to the value locked in
            Issue #9.
        hysteresis_margin: How far the ratio must fall past a boundary before a
            downward move is confirmed; defaults to the value derived in Issue #20
            (see docs/label_hysteresis_decision.md).

    Returns:
        A copy of `df` with three columns added:
            label               : final label after the override pass
            label_pre_override  : what the threshold-with-hysteresis rule alone said
                                   (lets you tell "override fired" apart from
                                   "override changed the label" -- the override can
                                   fire on a row the rule already called Critical)
            override_applied    : True where the rig-shutdown override fired
    """
    out = df.copy()
    ratio = out["rms_ratio"].to_numpy()

    # --- threshold rule, with hysteresis on downward moves --------------
    ranks = _hysteresis_ranks(ratio, onset_multiple, critical_multiple, hysteresis_margin)
    threshold_label = np.array(LABELS, dtype=object)[ranks]
    label = threshold_label.copy()

    # --- rig-shutdown artifact override ---------------------------------
    # Walk forward: once a row is Critical, a sharp collapse in *raw* RMS is the rig
    # stopping, not the bearing recovering. Reference level = mean raw RMS of the
    # Critical run so far.
    raw = out["rms"].to_numpy()
    override = np.zeros(len(out), dtype=bool)
    critical_run: list[float] = []

    for i in range(len(out)):
        if i > 0 and label[i - 1] == "Critical" and critical_run:
            if raw[i] < collapse_fraction * float(np.mean(critical_run)):
                label[i] = "Critical"
                override[i] = True
        if label[i] == "Critical" and not override[i]:
            critical_run.append(raw[i])

    out["label"] = pd.Categorical(label, categories=LABELS, ordered=True)
    out["label_pre_override"] = pd.Categorical(threshold_label, categories=LABELS, ordered=True)
    out["override_applied"] = override
    return out
