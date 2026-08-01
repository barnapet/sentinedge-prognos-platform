# MLflow Tracking for the M3 LOEO Runs (Issue #74)

Decision note accompanying `src/training/mlflow_tracking.py`. `docs/PRD.md` Section 11
defines M3 as "trained, tracked in MLflow, evaluated honestly", and Section 10's acceptance
criteria include "at least one MLflow run is visible, showing the trained model, its
metrics, and its parameters — not just a pickled file with no lineage." Neither #21
(`compare_imbalance.py`) nor #72 (`train_baseline.py`) logged to MLflow when they ran; this
issue adds that instrumentation without changing either harness.

**Decision up front:** `mlflow-skinny` + `sqlalchemy` + `alembic` (not the full `mlflow`
package), backed by a local SQLite store (`mlflow.db`), logging one MLflow run per #21 arm
(five) and per #72 configuration restricted to `full`/`no_rms_ratio` (two) — seven runs
total. Both halves of that decision were forced by conflicts discovered while actually
installing and running this, not chosen up front; Sections 1–2 below are the evidence.

## 1. The dependency conflict: full `mlflow` pins `pandas<3`

`pip install mlflow` (3.15.0, the current release) declares `pandas<3` as a hard
constraint. This project pins `pandas==3.0.5` (Issue #27, validated against the M1-EDA
notebooks). Installing full `mlflow` into this project's environment does not fail loudly —
it silently downgrades pandas:

```
$ pip install mlflow
...
Successfully installed ... pandas-2.3.3 ...
  Attempting uninstall: pandas
    Found existing installation: pandas 3.0.5
    Uninstalling pandas-3.0.5
```

Confirmed by inspecting the package metadata directly (`importlib.metadata`), not just
observing the downgrade once:

```
mlflow 3.15.0 Requires-Dist: pandas<3, numpy<3, scikit-learn<2, scipy<2, matplotlib<4, ...
```

Every released `mlflow` version pins `pandas<3` — pandas 3.0 postdates `mlflow`'s current
compatibility testing. This is exactly the installation conflict Issue #74 flagged as a risk
("the first PR that brings in a new, heavier dependency").

**Resolution: `mlflow-skinny` instead of `mlflow`.** `mlflow-skinny` is the tracking-client
subset of the same project — `start_run`, `log_param`, `log_metric`, `log_dict`,
`search_runs` — with none of the modules that carry the pandas/numpy/scikit-learn/scipy
pins (autologging integrations, `mlflow.evaluate`, dataset profiling). Its own
`Requires-Dist` list has no pandas or numpy entry at all. That is everything Issue #74's
Task 2 needs.

Verified end-to-end in a clean virtualenv: installing `requirements.txt` plus
`mlflow-skinny==3.15.0`, `sqlalchemy==2.0.51`, `alembic==1.18.5` together resolves to
exactly the pinned versions of every existing dependency (`pandas 3.0.5`, `numpy 2.4.6`,
`scipy 1.17.1`, `scikit-learn 1.9.0`, `matplotlib 3.11.1`, `pyarrow 25.0.0`) with zero
drift, and the full `pytest` suite (138 tests, pre-#74) passes unchanged against it.

**Trade-off accepted:** the graphical `mlflow ui` server needs `Flask`/`gunicorn`, which
`mlflow-skinny` does not install. Not added as a project dependency — see Section 4 on why
`mlflow.search_runs()` is the primary, CI-compatible verification path and the UI is an
optional convenience a reviewer can add locally (`pip install flask gunicorn`, or the full
`mlflow` package in a disposable environment) without touching this repo's pins.

## 2. The tracking-store choice: SQLite, not the plain file store

`docs/PRD.md` Section 8 says "local/file-backed is fine for MVP" — read here as "no remote
tracking server required", not as a commitment to MLflow's specific `file:./mlruns` store
implementation. That distinction turned out to matter: MLflow 3.x's file store is in
maintenance mode and now refuses to write without an explicit opt-out:

```
mlflow.exceptions.MlflowException: The filesystem tracking backend (e.g., './mlruns') is
in maintenance mode and will not receive further updates... set
`MLFLOW_ALLOW_FILE_STORE=true` to opt out of this exception.
```

Rather than depending on an escape-hatch environment variable for a backend MLflow itself
says is being phased out, this module uses `sqlite:///mlflow.db` — still a single local
file, still no server process, and the backend MLflow recommends going forward. It needs
`sqlalchemy` and `alembic` (MLflow's SQL-backed tracking store dependencies, not included
in `mlflow-skinny`), which is why both are added alongside it. Neither has any
pandas/numpy version constraint, confirmed the same way as `mlflow-skinny` above — adding
them does not reopen the Section 1 conflict.

Artifacts (the confusion-matrix JSON per run) are logged under `mlartifacts/`, the second
path `docs/PRD.md`'s 2026-08-01 audit had already anticipated in `.gitignore`. Both
`mlflow.db` and `mlartifacts/` are gitignored — see Section 4.

## 3. What is logged

Two MLflow experiments, matching Issue #74 Task 2's scope exactly — not a broader sweep:

- **`m3-imbalance-comparison`** — one run per strategy in `imbalance.STRATEGIES` (`none`,
  `class_weight_balanced`, `random_oversample`, `random_undersample`, `prior_correction`),
  all on the full M2 feature set (the configuration `docs/class_imbalance_decision.md`
  Section 3 reports). Params: `strategy`, `description`, `feature_set`, `feature_columns`,
  `n_folds`. Tags: `issue=21`, `component=imbalance_comparison`.
- **`m3-baseline-ablation`** — one run per feature-set configuration, restricted to `full`
  and `no_rms_ratio` — Issue #74 Task 2 names these two explicitly as "the two #72
  configurations (baseline, ablation)". `no_raw_rms` and `kurtosis_skewness_only` are #72's
  own diagnostic side-runs (Section 4 of `docs/model_training_decision.md`), not part of
  what #74 asks to be tracked; they remain reproducible via
  `python -m src.training.train_baseline` directly, unlogged. Params: `feature_set`,
  `feature_columns`, `imbalance_strategy`, `n_folds`. Tags: `issue=72`,
  `component=baseline_ablation`.

Every run logs, per held-out experiment (`1st_test`/`2nd_test`/`3rd_test`): per-class
recall, precision, and support, and macro-F1 — `docs/evaluation_protocol.md` Section 4's
committed metrics, suffixed by the held-out experiment name rather than logged on MLflow's
`step` axis (the three folds are independent held-out experiments, not steps of one
training run, and this way each metric name is directly comparable across runs in the
MLflow UI's table/chart view). Aggregates (mean/min/max/range, Section 5's "never a
standard deviation") are logged for `critical_recall`, `critical_precision`, `macro_f1`,
and — for the `m3-baseline-ablation` runs only — `normal_recall`. The full per-fold
confusion matrices are logged as a `confusion_matrices.json` artifact per run.

## 4. Verification: logged metrics match the published numbers exactly

Ran `python -m src.training.mlflow_tracking` against the real
`data/processed/training_dataset.parquet` and queried the result back with
`mlflow.search_runs()`. No number below was computed for this document — every one is
`mlflow.search_runs()`'s own output, pulled from the SQLite store the run just wrote.

**`m3-imbalance-comparison` — `Critical` recall, vs. `docs/class_imbalance_decision.md` §3:**

| Strategy | `1st_test` | `2nd_test` | `3rd_test` | mean | Published mean |
|---|---|---|---|---|---|
| `none` | 0.1176 | 0.9130 | 1.0000 | 0.6769 | 0.677 |
| `class_weight_balanced` | 0.0588 | 0.9130 | 1.0000 | 0.6573 | 0.657 |
| `random_oversample` | 0.0588 | 0.9130 | 1.0000 | 0.6573 | 0.657 |
| `random_undersample` | 0.1176 | 0.9130 | 1.0000 | 0.6769 | 0.677 |
| `prior_correction` | 0.1765 | 0.9130 | 1.0000 | 0.6965 | 0.697 |

`Critical` precision and macro-F1 means were checked the same way and also match to display
precision (e.g. `class_weight_balanced` precision mean 0.8697 → published 0.870;
`random_undersample` macro-F1 mean 0.6162 → published 0.616).

**`m3-baseline-ablation` — vs. `docs/model_training_decision.md` §1–2:**

| Config | `Critical` recall (1st/2nd/3rd) | mean | Published mean | `Critical` precision mean | `Normal` recall mean | Macro-F1 mean |
|---|---|---|---|---|---|---|
| `full` | 0.0588 / 0.9130 / 1.0000 | 0.6573 | 0.657 | 0.8697 (pub. 0.870) | 0.6910 (pub. 0.691) | 0.6779 (pub. 0.678) |
| `no_rms_ratio` | 0.9412 / 0.9130 / 0.8209 | 0.8917 | 0.892 | 0.6028 (pub. 0.603) | 0.6661 (pub. 0.666) | 0.5990 (pub. 0.599) |

**No drift.** Every logged number matches its published counterpart to the precision the
docs report at — expected, since the module calls `compare_imbalance.run_comparison()` and
`train_baseline.run_all_feature_sets()` unmodified (fixed `random_state`, no tuning, per
those modules' own leakage guards) and simply logs what they return.

### How to re-verify

```
python -m src.features.build_training_dataset     # if data/processed/ is empty
python -m src.training.mlflow_tracking             # writes mlflow.db, mlartifacts/, prints run IDs
mlflow ui --backend-store-uri sqlite:///mlflow.db   # needs `pip install flask gunicorn` first — see Section 1
```

or, without the UI:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.search_runs(experiment_names=["m3-imbalance-comparison", "m3-baseline-ablation"])
```

## 5. What this does not do

- **No model artifact is persisted.** Unchanged from `docs/model_training_decision.md`
  Section 5: LOEO trains three models per configuration, so there is no single "the model"
  to log as an MLflow model artifact. That is the separate, still-open M3 item ("a servable
  model artifact") `CLAUDE.md` tracks.
- **Does not log #72's `no_raw_rms` or `kurtosis_skewness_only` diagnostics**, or #21
  Section 6's `rms_ratio`-ablated robustness check — out of Issue #74's stated scope (the
  five arms and the two named configurations). Both remain reproducible and printed by
  their respective modules' existing `main()` entrypoints, just not logged to MLflow.
- **No remote or shared tracking server.** `mlflow.db` and `mlartifacts/` are local,
  gitignored, and per-clone — consistent with `docs/PRD.md`'s MVP scope (no Kubernetes, no
  shared infrastructure).

## Reproducing

```
python -m src.features.build_training_dataset   # Issue #67, writes training_dataset.parquet
python -m src.training.mlflow_tracking           # this issue: logs both experiments
```

Deterministic, same as the harnesses it wraps: fixed `random_state` throughout, no
hyperparameter tuning anywhere.
