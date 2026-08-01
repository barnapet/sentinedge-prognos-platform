"""Health-state labeling logic for bearing vibration data.

Extracted from notebooks/02_health_state_labeling.ipynb during M1.5-Housekeeping
(Issue #19), so the core labeling logic can be unit tested independently of full
notebook execution. `derive_critical_multiple` followed in Issue #65, for the same
reason and by the same pattern -- it is the one input `assign_labels` cannot supply
itself, and M3 needs it importable rather than notebook-bound (anticipated in
docs/feature_extraction_versioning.md Section 1). See
notebooks/02_health_state_labeling.ipynb for the full derivation narrative behind
these constants and the threshold rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["Normal", "Degrading", "Critical"]

# Carried over from Issue #9 (do not change without re-running #9's analysis).
ONSET_MULTIPLE = 1.3

# Derived in notebooks/02_health_state_labeling.ipynb, Section 2b: the Critical boundary
# sits at the geometric midpoint (f = 0.5) of each bearing's own onset -> peak span.
# Geometric rather than arithmetic because the quantity being split is itself a ratio and
# degradation compounds multiplicatively; the notebook's Section 2b sensitivity sweep shows
# f = 0.5 sits inside a smooth region, not on a cliff edge.
CRITICAL_SPAN_FRACTION = 0.5

# Rounding pinned by Issue #65 -- see derive_critical_multiple's docstring for why these are
# part of the contract rather than presentation detail.
PEAK_RATIO_DECIMALS = 2
CRITICAL_MULTIPLE_DECIMALS = 3

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


def derive_critical_multiple(
    peak_ratio: float,
    span_fraction: float = CRITICAL_SPAN_FRACTION,
) -> float:
    """Derive one experiment's Critical threshold from its own peak rolling RMS ratio.

    Interpolates in log space between `ONSET_MULTIPLE` and the worst rolling RMS ratio
    that bearing actually reached:

        critical_multiple = onset_multiple * (peak_ratio / onset_multiple) ** span_fraction

    At the default `span_fraction = 0.5` this is the geometric midpoint,
    `sqrt(onset_multiple * peak_ratio)`. Extracted from
    notebooks/02_health_state_labeling.ipynb Section 2b (Issue #65), the same way
    `assign_labels` was extracted in Issue #19; that notebook carries the full derivation
    narrative, including why a single global multiplier does not work across three
    experiments whose peak ratios span 2.5x.

    Note this is a **retrospective** quantity: `peak_ratio` is the maximum over a completed
    run-to-failure experiment, so it is not knowable at inference time. That is a documented
    property of the labeling scheme, not a defect -- see docs/eda_findings.md Section 3
    (Issue #62) for the statement of it, and Section 2b of the notebook for the fixed ~2.6x
    fallback recorded for serving.

    **The rounding is part of the contract, not presentation.** The notebook derives these
    values through `summarize()`, which rounds the peak ratio to 2 decimals before the
    formula, and then rounds the result to 3 decimals. Those rounded results -- 1.932 /
    2.866 / 3.049 -- are the values already published in docs/eda_findings.md Section 3 and
    hardcoded in notebooks/04_feature_pipeline_validation.ipynb's `CRITICAL_MULTIPLE`. An
    implementation that skipped the rounding would return 1.931235 / 2.867072 / 3.049334
    instead, silently disagreeing with both. So the rounding is replicated here rather than
    dropped as a tidy-up. It is safe to pin: applying it changes **no labels at all** (both
    paths produce 1906/233/17, 651/310/23, 6158/99/67 -- the closed table in
    docs/eda_findings.md Section 3) and leaves the notebook's own f = 0.40-0.65 sensitivity
    sweep unchanged in every cell. Do not "simplify" it away; `tests/test_labeling.py` guards
    against that.

    Args:
        peak_ratio: The experiment's maximum `rms_ratio` (10-file rolling mean of RMS over
            the first-50-file baseline mean -- the same series `assign_labels` thresholds).
            Passed raw; this function applies the 2-decimal rounding itself, so callers do
            not need to pre-round, and passing an already-rounded value is equivalent.
        span_fraction: How far along the log-space onset -> peak span to place the boundary.
            Defaults to the derived `CRITICAL_SPAN_FRACTION`; exposed as a parameter only so
            the notebook's sensitivity sweep can vary it.

    Returns:
        The per-experiment `critical_multiple`, rounded to 3 decimals, ready to pass to
        `assign_labels`.
    """
    rounded_peak = round(peak_ratio, PEAK_RATIO_DECIMALS)
    critical_multiple = ONSET_MULTIPLE * (rounded_peak / ONSET_MULTIPLE) ** span_fraction
    return float(round(critical_multiple, CRITICAL_MULTIPLE_DECIMALS))


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
