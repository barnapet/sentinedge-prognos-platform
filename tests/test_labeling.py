import pandas as pd
import pytest

from src.labeling import assign_labels


def make_df(rms_ratio, rms):
    return pd.DataFrame({"rms_ratio": rms_ratio, "rms": rms})


def test_all_normal_no_override():
    """Baseline sanity: ratios well below onset stay Normal, no override anywhere."""
    df = make_df(rms_ratio=[1.0, 1.0, 1.0], rms=[0.1, 0.1, 0.1])

    result = assign_labels(df, critical_multiple=2.0)

    assert result["label"].tolist() == ["Normal", "Normal", "Normal"]
    assert not result["override_applied"].any()


def test_threshold_boundaries_are_strict_greater_than():
    """Boundary values: `>` not `>=`, so a ratio exactly at a threshold does NOT
    cross into the next label."""
    df = make_df(
        rms_ratio=[1.3, 1.30001, 2.0, 2.00001],
        rms=[0.1, 0.1, 0.1, 0.1],
    )

    result = assign_labels(df, critical_multiple=2.0, onset_multiple=1.3)

    assert result["label"].tolist() == ["Normal", "Degrading", "Degrading", "Critical"]
    assert not result["override_applied"].any()


def test_sustained_critical_run_no_override():
    """A run that stays Critical throughout, with no raw-RMS collapse -- the
    1st_test-like case where the rig never shuts down mid-run."""
    df = make_df(rms_ratio=[3.0, 3.0, 3.0], rms=[1.0, 1.0, 1.0])

    result = assign_labels(df, critical_multiple=2.0, collapse_fraction=0.2)

    assert result["label"].tolist() == ["Critical", "Critical", "Critical"]
    assert not result["override_applied"].any()


def test_override_fires_but_does_not_change_label():
    """Override can fire on a row the threshold already called Critical -- the
    flag distinguishes 'fired' from 'changed the label', per label_pre_override."""
    df = make_df(
        rms_ratio=[3.0, 3.0],   # both rows are Critical by threshold alone
        rms=[1.0, 0.1],         # second row collapses: 0.1 < 0.2 * mean([1.0])
    )

    result = assign_labels(df, critical_multiple=2.0, collapse_fraction=0.2)

    assert result["override_applied"].tolist() == [False, True]
    assert result["label"].tolist() == ["Critical", "Critical"]
    # the defining check: override fired, but didn't change anything
    assert (result["label"] == result["label_pre_override"]).all()


def test_override_changes_label():
    """The actual purpose of the override: a raw-RMS collapse after a Critical run
    forces Critical even though the threshold rule alone would say otherwise.

    label_pre_override is "Degrading", not "Normal", here: hysteresis (Issue #20)
    relaxes downward moves one rank at a time, so a single-file plunge from a
    Critical-level ratio doesn't fall straight through to Normal -- it's exactly
    this kind of drastic single-file drop that the rig-shutdown override (not
    hysteresis) is meant to catch, and the override's own behavior is unaffected."""
    df = make_df(
        rms_ratio=[3.0, 0.5],   # second row would be Degrading by threshold+hysteresis alone
        rms=[1.0, 0.05],        # collapses: 0.05 < 0.2 * mean([1.0])
    )

    result = assign_labels(df, critical_multiple=2.0, collapse_fraction=0.2)

    assert result["label_pre_override"].tolist() == ["Critical", "Degrading"]
    assert result["label"].tolist() == ["Critical", "Critical"]
    assert result["override_applied"].tolist() == [False, True]


def test_override_reference_level_uses_critical_run_mean():
    """The collapse reference is the mean raw RMS of the Critical run *so far*,
    not just the immediately preceding value."""
    df = make_df(
        rms_ratio=[3.0, 3.0, 0.5],
        # critical_run mean after two Critical rows = mean([1.0, 3.0]) = 2.0
        # 0.2 * 2.0 = 0.4, so raw=0.3 should collapse
        rms=[1.0, 3.0, 0.3],
    )

    result = assign_labels(df, critical_multiple=2.0, collapse_fraction=0.2)

    assert result["label"].tolist() == ["Critical", "Critical", "Critical"]
    assert result["override_applied"].tolist() == [False, False, True]


# --- Issue #20: onset-boundary hysteresis -------------------------------------

def test_hysteresis_suppresses_1st_test_style_flapping():
    """Regression test for Issue #20: a synthetic ratio series shaped like
    1st_test's real files 1906-1999 (rms_ratio hovers within a few percent of
    ONSET_MULTIPLE=1.3, briefly crosses above, dips ~0.04 below, then climbs back
    and stays above for good). Without hysteresis this reverts to Normal and back;
    with the default hysteresis_margin=0.05 it must transition once, cleanly."""
    ratio = [
        1.290, 1.296,                   # approaching onset, still Normal
        1.301, 1.304, 1.307, 1.305,     # crosses onset -- Degrading (immediate)
        1.299, 1.292,                   # dips just under 1.3 (would revert with no hysteresis)
        1.279, 1.264,                   # deepest dip: 0.036 below onset, matches EDA finding
        1.270, 1.281, 1.290, 1.296,     # climbs back, still below 1.3
        1.300, 1.306, 1.308, 1.309,     # crosses back above and stays -- real onset
    ]
    df = make_df(rms_ratio=ratio, rms=[0.1] * len(ratio))

    no_hysteresis = assign_labels(df, critical_multiple=3.0, hysteresis_margin=0.0)
    with_hysteresis = assign_labels(df, critical_multiple=3.0)  # default margin=0.05

    # Confirms the bug being fixed: with no margin, this shape really does flap.
    assert "Normal" in no_hysteresis["label"].tolist()[2:], (
        "fixture no longer reproduces flapping with margin=0.0 -- fixture is stale"
    )
    no_hysteresis_labels = no_hysteresis["label"].tolist()
    reverts = sum(
        1 for i in range(1, len(no_hysteresis_labels))
        if no_hysteresis_labels[i] == "Normal" and no_hysteresis_labels[i - 1] != "Normal"
    )
    assert reverts >= 1

    # The fix: once Degrading is confirmed, it never reverts to Normal mid-series.
    labels = with_hysteresis["label"].tolist()
    assert "Degrading" in labels
    first_degrading = labels.index("Degrading")
    assert "Normal" not in labels[first_degrading:]


def test_hysteresis_does_not_delay_a_clean_transition():
    """Hysteresis must not introduce lag on a boundary that never flaps (e.g.
    3rd_test's clean single crossing) -- only reverts are gated, not the initial
    upward move."""
    ratio = [1.0, 1.0, 1.0, 1.35, 1.4, 1.45]
    df = make_df(rms_ratio=ratio, rms=[0.1] * len(ratio))

    result = assign_labels(df, critical_multiple=3.0)

    assert result["label"].tolist() == [
        "Normal", "Normal", "Normal", "Degrading", "Degrading", "Degrading",
    ]


def test_hysteresis_requires_dropping_past_margin_not_just_below_threshold():
    """A dip that crosses back below onset_multiple but not past the hysteresis
    margin must NOT revert; a dip that clears the margin must revert."""
    # Dips to 1.26: onset_multiple(1.3) - default margin(0.05) = 1.25, so 1.26
    # is back below 1.3 but doesn't clear the exit threshold -- stays Degrading.
    shallow_dip = make_df(rms_ratio=[1.35, 1.26, 1.35], rms=[0.1, 0.1, 0.1])
    result = assign_labels(shallow_dip, critical_multiple=3.0)
    assert result["label"].tolist() == ["Degrading", "Degrading", "Degrading"]

    # Dips to 1.20, clearing 1.25 -- reverts to Normal.
    deep_dip = make_df(rms_ratio=[1.35, 1.20, 1.35], rms=[0.1, 0.1, 0.1])
    result = assign_labels(deep_dip, critical_multiple=3.0)
    assert result["label"].tolist() == ["Degrading", "Normal", "Degrading"]


def test_hysteresis_does_not_affect_3rd_test_style_clean_run():
    """3rd_test never flapped in the original threshold rule (EDA confirmed its
    Critical/Degrading runs are single contiguous blocks) -- hysteresis must be a
    no-op on a sequence with no near-boundary noise."""
    ratio = [1.0] * 5 + [1.35] * 5 + [3.5] * 5
    df = make_df(rms_ratio=ratio, rms=[0.1] * len(ratio))

    with_hysteresis = assign_labels(df, critical_multiple=3.0)
    no_hysteresis = assign_labels(df, critical_multiple=3.0, hysteresis_margin=0.0)

    assert with_hysteresis["label"].tolist() == no_hysteresis["label"].tolist()
    assert with_hysteresis["label"].tolist() == (
        ["Normal"] * 5 + ["Degrading"] * 5 + ["Critical"] * 5
    )
