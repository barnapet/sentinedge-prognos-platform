"""Compare class-imbalance handling strategies under the LOEO protocol (Issue #21).

CLI entrypoint. Run from the repo root, after `python -m src.features.build_training_dataset`:

    python -m src.training.compare_imbalance

Runs every strategy in `src.training.imbalance.STRATEGIES` through all three folds of
`docs/evaluation_protocol.md`'s leave-one-experiment-out split, scoring each on that
document's committed metrics (per-class recall/precision headlined by `Critical` recall,
macro-F1, full confusion matrices). Results are printed and, optionally, written to JSON.

This produces the evidence behind `docs/class_imbalance_decision.md`. It is a comparison
harness, not the final M3 training script -- the model it uses is a fixed, deliberately
simple baseline whose only job is to hold everything except the imbalance treatment
constant.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.labeling import LABELS
from src.training.evaluation import (
    Fold,
    aggregate,
    fold_metrics,
    load_training_dataset,
    loeo_folds,
)
from src.training.imbalance import STRATEGIES, Strategy, prior_correct

HEADLINE_METRIC = ("recall", "Critical")


def run_strategy_on_fold(strategy: Strategy, fold: Fold) -> dict:
    """Fit one strategy on one fold's training rows and score it on the held-out rows.

    Resampling, where a strategy uses it, is applied to the training rows *only* -- never
    to the held-out experiment, which must stay exactly as observed for the evaluation to
    mean anything. Prior correction likewise reads class frequencies from the training
    rows only (`docs/evaluation_protocol.md` Section 1).
    """
    X_train, y_train = fold.X_train, fold.y_train
    if strategy.resample is not None:
        X_train, y_train = strategy.resample(X_train, y_train)

    model = strategy.build_model()
    model.fit(X_train, y_train)

    if strategy.apply_prior_correction:
        scores = prior_correct(model.predict_proba(fold.X_test), model.classes_, y_train)
        y_pred = model.classes_[np.argmax(scores, axis=1)]
    else:
        y_pred = model.predict(fold.X_test)

    metrics = fold_metrics(fold.y_test, y_pred)
    metrics["held_out"] = fold.held_out
    metrics["n_train_rows"] = int(len(y_train))
    return metrics


def run_comparison(feature_columns: list[str] | None = None) -> dict:
    """Run every strategy across all three LOEO folds and aggregate per Section 5."""
    df = load_training_dataset()
    folds = loeo_folds(df, feature_columns=feature_columns)

    results = {}
    for strategy in STRATEGIES:
        per_fold = [run_strategy_on_fold(strategy, fold) for fold in folds]
        metric_kind, metric_class = HEADLINE_METRIC
        results[strategy.name] = {
            "description": strategy.description,
            "per_fold": per_fold,
            "aggregate": {
                "critical_recall": aggregate([f[metric_kind][metric_class] for f in per_fold]),
                "critical_precision": aggregate([f["precision"]["Critical"] for f in per_fold]),
                "macro_f1": aggregate([f["macro_f1"] for f in per_fold]),
            },
        }
    return results


def format_report(results: dict) -> str:
    """Human-readable report: per-fold values first, then the aggregate.

    Deliberately prints all three fold values before any summary, per
    `docs/evaluation_protocol.md` Section 5's "never collapsed to a single number without
    also showing the three it came from".
    """
    lines = []
    folds = [f["held_out"] for f in next(iter(results.values()))["per_fold"]]

    lines.append("Critical-class recall (headline metric, docs/evaluation_protocol.md Section 4)")
    lines.append(f"{'strategy':<24}" + "".join(f"{f:>12}" for f in folds) + f"{'mean':>10}{'range':>10}")
    for name, res in results.items():
        per_fold = [f["recall"]["Critical"] for f in res["per_fold"]]
        agg = res["aggregate"]["critical_recall"]
        lines.append(
            f"{name:<24}"
            + "".join(f"{v:>12.3f}" for v in per_fold)
            + f"{agg['mean']:>10.3f}{agg['range']:>10.3f}"
        )

    lines.append("")
    lines.append("Critical-class precision")
    lines.append(f"{'strategy':<24}" + "".join(f"{f:>12}" for f in folds) + f"{'mean':>10}{'range':>10}")
    for name, res in results.items():
        per_fold = [f["precision"]["Critical"] for f in res["per_fold"]]
        agg = res["aggregate"]["critical_precision"]
        lines.append(
            f"{name:<24}"
            + "".join(f"{v:>12.3f}" for v in per_fold)
            + f"{agg['mean']:>10.3f}{agg['range']:>10.3f}"
        )

    lines.append("")
    lines.append("Macro-F1")
    lines.append(f"{'strategy':<24}" + "".join(f"{f:>12}" for f in folds) + f"{'mean':>10}{'range':>10}")
    for name, res in results.items():
        per_fold = [f["macro_f1"] for f in res["per_fold"]]
        agg = res["aggregate"]["macro_f1"]
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
    results = run_comparison()
    print(format_report(results))
    if output_path is not None:
        output_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
