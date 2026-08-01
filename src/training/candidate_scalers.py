"""Scaling approaches evaluated for the `1st_test` fold and rejected (Issue #72).

Kept rather than deleted, following the same convention as
`src/features/candidate_features.py` (Issues #22/#23): a rejected approach is more useful
as runnable, tested code than as a number quoted in a document, because it lets the
rejection be re-checked instead of trusted. Neither function here is imported by
`src/training/train_baseline.py` — the adopted path does not depend on this module.

Both address the amplitude-scale problem diagnosed in
`docs/class_imbalance_decision.md` Section 4: `1st_test`'s minimum raw RMS exceeds both
other experiments' means, so a `StandardScaler` fitted on the training experiments maps
every held-out row into the high-RMS region of the training distribution.

See `docs/model_training_decision.md` Section 4 for the full reasoning. In short:

- `evaluate_averaged_per_experiment_scaler` is **leakage-safe but ineffective** — any
  affine transform fitted without seeing the held-out experiment preserves that
  experiment's displacement, so recentring moves train and test together.
- `evaluate_transductive_scaler` is **effective on `1st_test` but forbidden** — it fits on
  the held-out experiment's own rows, which `docs/evaluation_protocol.md` Section 1 rules
  out, and it degrades the other two folds anyway. Implemented only to quantify what is
  being declined; declining a fix is more credible when its size is known.

Run `python -m src.training.candidate_scalers` to reproduce both tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.training.evaluation import (
    EXPERIMENTS,
    FEATURE_MATRIX_COLUMNS,
    fold_metrics,
    load_training_dataset,
)
from src.training.imbalance import BASELINE_MODEL_PARAMS


def _fit_baseline_classifier(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """The bare estimator from `make_baseline_model`, without its `StandardScaler` step.

    These candidates replace the pipeline's scaling stage, so they need the classifier on
    its own. Hyperparameters come from `BASELINE_MODEL_PARAMS` unchanged — nothing is
    tuned here either, so the comparison against the adopted configuration stays
    one-variable.
    """
    return LogisticRegression(class_weight="balanced", **BASELINE_MODEL_PARAMS).fit(X_train, y_train)


def averaged_per_experiment_moments(
    train_df: pd.DataFrame, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std averaged with equal weight per training experiment, not per row.

    The motivating idea: a pooled `StandardScaler` is dominated by whichever experiment
    contributes most rows (`3rd_test` is 67% of the dataset,
    `docs/frequency_domain_decision.md` Section 6a made the same observation), so equal
    per-experiment weighting might produce a centre that transfers better to an unseen
    bearing.

    Leakage-safe by construction: `train_df` contains the fold's training experiments
    only, and the held-out experiment contributes nothing to either moment.
    """
    means = [g[feature_columns].mean().to_numpy() for _, g in train_df.groupby("experiment")]
    stds = [g[feature_columns].std().to_numpy() for _, g in train_df.groupby("experiment")]
    return np.mean(means, axis=0), np.mean(stds, axis=0)


def evaluate_averaged_per_experiment_scaler(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> dict[str, dict]:
    """Candidate A: LOEO with the equal-per-experiment-weighted scaler above.

    **Rejected: ineffective.** Measured at macro-F1 0.152 -> 0.154 on the `1st_test` fold
    (`docs/model_training_decision.md` Section 4). The reason is structural rather than a
    tuning failure: changing the centre and scale moves the training and held-out rows
    together, so the held-out experiment's displacement relative to the training
    distribution is preserved.
    """
    feature_columns = feature_columns or list(FEATURE_MATRIX_COLUMNS)
    results = {}
    for held_out in EXPERIMENTS:
        train = df[df["experiment"] != held_out]
        test = df[df["experiment"] == held_out]
        mean, std = averaged_per_experiment_moments(train, feature_columns)

        X_train = (train[feature_columns].to_numpy() - mean) / std
        X_test = (test[feature_columns].to_numpy() - mean) / std
        model = _fit_baseline_classifier(X_train, train["label"].astype(str).to_numpy())

        metrics = fold_metrics(test["label"].astype(str).to_numpy(), model.predict(X_test))
        metrics["held_out"] = held_out
        results[held_out] = metrics
    return results


def evaluate_transductive_scaler(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> dict[str, dict]:
    """Candidate C: each experiment standardised against its *own* rows, test included.

    **Rejected: violates `docs/evaluation_protocol.md` Section 1**, which requires fitted
    preprocessing to see the fold's training rows only. This function deliberately breaks
    that rule — it is a measurement of a path not taken, never a configuration to adopt,
    and nothing in `src/training/train_baseline.py` imports it.

    Two reasons it is declined, beyond the protocol violation. It is not a free win: it
    repairs `1st_test` (macro-F1 0.152 -> 0.800) while degrading `2nd_test`
    (0.936 -> 0.816) and `3rd_test` (0.945 -> 0.624). And it is unavailable in practice for
    the reason `docs/eda_findings.md` Section 3 already gives about look-ahead quantities:
    standardising a bearing against its own completed run requires the whole run, which a
    bearing still in operation has not finished.
    """
    feature_columns = feature_columns or list(FEATURE_MATRIX_COLUMNS)
    results = {}
    for held_out in EXPERIMENTS:
        train = df[df["experiment"] != held_out]
        test = df[df["experiment"] == held_out]

        X_train = StandardScaler().fit_transform(train[feature_columns].to_numpy())
        # Fitted on the held-out experiment's own rows -- the forbidden step.
        X_test = StandardScaler().fit_transform(test[feature_columns].to_numpy())
        model = _fit_baseline_classifier(X_train, train["label"].astype(str).to_numpy())

        metrics = fold_metrics(test["label"].astype(str).to_numpy(), model.predict(X_test))
        metrics["held_out"] = held_out
        results[held_out] = metrics
    return results


def format_report(candidate_results: dict[str, dict[str, dict]]) -> str:
    """Per-fold `Normal` recall, `Critical` recall and macro-F1 for each rejected candidate."""
    lines = []
    for name, per_fold in candidate_results.items():
        lines.append(f"{name} (REJECTED -- see docs/model_training_decision.md Section 4)")
        lines.append(f"{'held out':<12}{'Normal recall':>16}{'Critical recall':>18}{'macro-F1':>12}")
        for held_out, m in per_fold.items():
            lines.append(
                f"{held_out:<12}{m['recall']['Normal']:>16.3f}"
                f"{m['recall']['Critical']:>18.3f}{m['macro_f1']:>12.3f}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = load_training_dataset()
    print(
        format_report(
            {
                "Candidate A: per-experiment moments averaged over training experiments": (
                    evaluate_averaged_per_experiment_scaler(df)
                ),
                "Candidate C: scaler fitted on the held-out experiment's own rows": (
                    evaluate_transductive_scaler(df)
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
