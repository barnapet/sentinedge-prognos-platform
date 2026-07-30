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


def assign_labels(
    df: pd.DataFrame,
    critical_multiple: float,
    collapse_fraction: float = COLLAPSE_FRACTION,
    onset_multiple: float = ONSET_MULTIPLE,
) -> pd.DataFrame:
    """Label each file Normal / Degrading / Critical, then apply the rig-shutdown override.

    Threshold rule (on `rms_ratio`):
        Normal     : rms_ratio <= onset_multiple
        Degrading  : onset_multiple < rms_ratio <= critical_multiple
        Critical   : rms_ratio > critical_multiple

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

    Returns:
        A copy of `df` with three columns added:
            label               : final label after the override pass
            label_pre_override  : what the threshold rule alone said (lets you tell
                                   "override fired" apart from "override changed
                                   the label" -- the override can fire on a row the
                                   threshold already called Critical)
            override_applied    : True where the rig-shutdown override fired
    """
    out = df.copy()
    ratio = out["rms_ratio"].to_numpy()

    # --- threshold rule -------------------------------------------------
    threshold_label = np.where(
        ratio > critical_multiple, "Critical",
        np.where(ratio > onset_multiple, "Degrading", "Normal"),
    ).astype(object)
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
