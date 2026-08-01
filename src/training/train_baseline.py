"""M3 baseline classifier, `rms_ratio` ablation, and the `1st_test` scale diagnosis (Issue #72).

Fourth and final step of the M3 preparation sequence (#65 -> #67 -> #69 -> #21 -> this
module). It **extends** `src/training/` rather than replacing it: the model family
(`make_baseline_model`), the imbalance treatment (`class_weight='balanced'`, #21's adopted
default), the LOEO split and metrics (`loeo_folds`, `fold_metrics`, `aggregate`), and the
per-fold runner (`run_strategy_on_fold`) are all reused unchanged.

What is new here is the *comparison axis*. #21 held the feature set fixed and varied the
imbalance treatment; this module holds the imbalance treatment fixed at #21's decision and
varies the feature set, across four configurations (`FEATURE_SETS`):

- `full` -- `FEATURE_MATRIX_COLUMNS` unchanged. Issue #72 Task 1's reference point, not a
  tuned final model.
- `no_rms_ratio` -- Task 2's required ablation. `rms_ratio` is both the strongest feature
  and the signal the labels are thresholded from, so its contribution has to be measured
  rather than assumed (Issue #67 Task 3; `docs/class_imbalance_decision.md` Section 6).
- `no_raw_rms` -- Task 3's candidate fix for the scale problem: raw `rms` is the
  *absolute* amplitude channel, `rms_ratio` the per-experiment-normalised one.
- `kurtosis_skewness_only` -- the floor. What is left when neither RMS-derived feature is
  available, which bounds how much of the model's behaviour is RMS-driven.

Two diagnostics are exposed as functions rather than cited as prose numbers, so Issue
#72's "diagnosed with evidence, not just described" criterion is re-verifiable against
whatever dataset is loaded:

- `raw_rms_scale_summary` -- the amplitude-scale mismatch behind `1st_test`'s majority-class
  collapse (`docs/class_imbalance_decision.md` Section 4).
- `critical_band_summary` -- the *threshold-transfer* mismatch, which is a distinct and
  more fundamental problem that no feature scaling addresses. See
  `docs/model_training_decision.md` Sections 3-4.

**No hyperparameter is tuned anywhere**, per-fold or otherwise -- the same leakage-avoidance
choice `src/training/imbalance.py` documents, carried forward unchanged. The only quantity
that varies between runs here is the feature-column list.

Reproducing:

    python -m src.features.build_training_dataset   # Issue #67, if not already built
    python -m src.training.train_baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.labeling import LABELS
# `run_strategy_on_fold` lives in compare_imbalance.py because #21 wrote it there. It is
# the shared fit-and-score-one-fold step, not something specific to that comparison, so it
# is imported rather than duplicated. Left in place instead of moved so this issue's diff
# does not disturb #21's merged module or its tests.
from src.training.compare_imbalance import run_strategy_on_fold
from src.training.evaluation import (
    EXPERIMENTS,
    FEATURE_MATRIX_COLUMNS,
    aggregate,
    load_training_dataset,
    loeo_folds,
)
from src.training.imbalance import STRATEGIES

# #21's adopted default (`docs/class_imbalance_decision.md` Section 5), reused as-is.
# Issue #72 Task 1 asks for "the same model family #21 used" and the imbalance technique
# is #21's settled decision -- not this module's to re-open.
BASELINE_STRATEGY = next(s for s in STRATEGIES if s.name == "class_weight_balanced")

# Derived from FEATURE_MATRIX_COLUMNS rather than hardcoded, so a feature added upstream
# does not leave these lists silently stale.
FEATURE_SETS: dict[str, list[str]] = {
    "full": list(FEATURE_MATRIX_COLUMNS),
    "no_rms_ratio": [c for c in FEATURE_MATRIX_COLUMNS if c != "rms_ratio"],
    "no_raw_rms": [c for c in FEATURE_MATRIX_COLUMNS if c != "rms"],
    "kurtosis_skewness_only": [
        c for c in FEATURE_MATRIX_COLUMNS if c not in ("rms", "rms_ratio")
    ],
}

HEADLINE_METRIC = ("recall", "Critical")


def raw_rms_scale_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-experiment raw-`rms` min/mean/max -- evidence for the `1st_test` scale problem.

    `docs/class_imbalance_decision.md` Section 4 diagnosed that fold's majority-class
    collapse as an amplitude-scale mismatch: `1st_test`'s *minimum* raw RMS exceeds both
    other experiments' *means*, so a `StandardScaler` fitted (correctly) on the two
    training experiments maps every `1st_test` row into the high-RMS region of the
    training distribution. Recomputed here rather than cited, so the claim is checkable.
    """
    summary = df.groupby("experiment")["rms"].agg(["min", "mean", "max"])
    return {name: row.to_dict() for name, row in summary.iterrows()}


def critical_band_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per fold: do the held-out experiment's `Critical` rows fall inside the `rms_ratio`
    band its training fold ever labelled `Critical`?

    This is the diagnostic that separates the two failures stacked on the `1st_test` fold.
    The scale problem above concerns raw amplitude and is fixable by dropping or
    renormalising a column. This one is not: `critical_multiple` is derived per experiment
    from that experiment's own eventual peak (`docs/eda_findings.md` Section 3 -- a
    look-ahead, retrospective quantity), so each bearing's Degrading->Critical boundary
    sits at a different `rms_ratio`. If a held-out experiment's entire `Critical` band lies
    below the lowest `rms_ratio` its training fold ever saw labelled `Critical`, then no
    monotone decision boundary learned from that training fold can reach those rows --
    regardless of scaling, class weighting, or model family.

    `docs/evaluation_protocol.md` Section 2 predicted exactly this as the reason LOEO was
    chosen over a random split; this function measures whether it actually bites.

    Returns, per held-out experiment: the training fold's minimum `Critical` `rms_ratio`,
    the held-out `Critical` band's min and max, and how many of its `Critical` rows fall
    below that training-fold minimum.
    """
    out = {}
    for held_out in [e for e in EXPERIMENTS if e in set(df["experiment"])]:
        train = df[df["experiment"] != held_out]
        test = df[df["experiment"] == held_out]
        train_min = float(train.loc[train["label"] == "Critical", "rms_ratio"].min())
        test_critical = test.loc[test["label"] == "Critical", "rms_ratio"]
        out[held_out] = {
            "train_min_critical_rms_ratio": train_min,
            "test_critical_rms_ratio_min": float(test_critical.min()),
            "test_critical_rms_ratio_max": float(test_critical.max()),
            "test_critical_rows": int(len(test_critical)),
            "test_critical_rows_below_train_min": int((test_critical < train_min).sum()),
        }
    return out


def run_feature_set(feature_columns: list[str], df: pd.DataFrame | None = None) -> dict:
    """Run the `class_weight='balanced'` baseline over all three LOEO folds for one
    feature-column configuration.

    Scored on `docs/evaluation_protocol.md` Section 4's committed metrics, aggregated by
    Section 5's rules (mean and range, never a standard deviation). The caller is expected
    to report `per_fold` alongside `aggregate`, never the summary alone.
    """
    df = load_training_dataset() if df is None else df
    folds = loeo_folds(df, feature_columns=feature_columns)
    per_fold = [run_strategy_on_fold(BASELINE_STRATEGY, fold) for fold in folds]

    metric_kind, metric_class = HEADLINE_METRIC
    return {
        "feature_columns": list(feature_columns),
        "per_fold": per_fold,
        "aggregate": {
            "critical_recall": aggregate([f[metric_kind][metric_class] for f in per_fold]),
            "critical_precision": aggregate([f["precision"]["Critical"] for f in per_fold]),
            "macro_f1": aggregate([f["macro_f1"] for f in per_fold]),
            "normal_recall": aggregate([f["recall"]["Normal"] for f in per_fold]),
        },
    }


def run_all_feature_sets(
    feature_sets: dict[str, list[str]] | None = None,
    df: pd.DataFrame | None = None,
) -> dict:
    """Run every configuration against the same loaded dataset, so results are comparable."""
    feature_sets = feature_sets or FEATURE_SETS
    df = load_training_dataset() if df is None else df
    return {name: run_feature_set(cols, df=df) for name, cols in feature_sets.items()}


def _metric_column(fold_result: dict, metric_key: str) -> float:
    """Pull one reported metric out of a per-fold result dict."""
    if metric_key == "critical_recall":
        return fold_result["recall"]["Critical"]
    if metric_key == "critical_precision":
        return fold_result["precision"]["Critical"]
    if metric_key == "normal_recall":
        return fold_result["recall"]["Normal"]
    return fold_result["macro_f1"]


def format_report(results: dict, df: pd.DataFrame | None = None) -> str:
    """Human-readable report: all three fold values before any summary.

    Ordering follows `docs/evaluation_protocol.md` Section 5's rule that a metric is
    "never collapsed to a single number without also showing the three it came from", and
    Section 4's rule that recall and precision are reported separately so a
    recall-favouring configuration cannot hide a precision collapse behind one number.
    """
    lines: list[str] = []
    folds = [f["held_out"] for f in next(iter(results.values()))["per_fold"]]

    if df is not None:
        lines.append("Diagnostic 1: raw `rms` amplitude scale by experiment")
        for name, stats in raw_rms_scale_summary(df).items():
            lines.append(
                f"  {name}: min={stats['min']:.4f}  mean={stats['mean']:.4f}  max={stats['max']:.4f}"
            )
        lines.append("")
        lines.append("Diagnostic 2: does the held-out Critical band overlap the training fold's?")
        for name, stats in critical_band_summary(df).items():
            lines.append(
                f"  held out {name}: train min Critical rms_ratio="
                f"{stats['train_min_critical_rms_ratio']:.3f}  held-out band="
                f"[{stats['test_critical_rms_ratio_min']:.3f}, "
                f"{stats['test_critical_rms_ratio_max']:.3f}]  unreachable="
                f"{stats['test_critical_rows_below_train_min']}/{stats['test_critical_rows']}"
            )
        lines.append("")

    for heading, metric_key in [
        ("Critical-class recall (headline metric)", "critical_recall"),
        ("Critical-class precision", "critical_precision"),
        ("Normal-class recall (majority class -- the 1st_test collapse shows up here)", "normal_recall"),
        ("Macro-F1", "macro_f1"),
    ]:
        lines.append(heading)
        lines.append(
            f"{'feature set':<24}" + "".join(f"{f:>12}" for f in folds) + f"{'mean':>10}{'range':>10}"
        )
        for name, res in results.items():
            per_fold = [_metric_column(f, metric_key) for f in res["per_fold"]]
            agg = res["aggregate"][metric_key]
            lines.append(
                f"{name:<24}"
                + "".join(f"{v:>12.3f}" for v in per_fold)
                + f"{agg['mean']:>10.3f}{agg['range']:>10.3f}"
            )
        lines.append("")

    lines.append(f"Confusion matrices (rows = true, cols = predicted, order {LABELS})")
    for name, res in results.items():
        lines.append(f"  {name}:")
        for f in res["per_fold"]:
            lines.append(f"    held out {f['held_out']}: {f['confusion_matrix']}")

    return "\n".join(lines)


def main(output_path: Path | None = None) -> None:
    df = load_training_dataset()
    results = run_all_feature_sets(df=df)

    print(format_report(results, df=df))
    if output_path is not None:
        payload = {
            "raw_rms_scale_summary": raw_rms_scale_summary(df),
            "critical_band_summary": critical_band_summary(df),
            "results": results,
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
