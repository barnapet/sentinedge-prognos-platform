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
    forces Critical even though the threshold rule alone would say Normal."""
    df = make_df(
        rms_ratio=[3.0, 0.5],   # second row would be Normal by threshold alone
        rms=[1.0, 0.05],        # collapses: 0.05 < 0.2 * mean([1.0])
    )

    result = assign_labels(df, critical_multiple=2.0, collapse_fraction=0.2)

    assert result["label_pre_override"].tolist() == ["Critical", "Normal"]
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
