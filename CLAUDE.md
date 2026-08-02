# SentinEdge Prognos Platform — Project Context for Claude Code

## What this project is

A predictive maintenance AI platform engineering portfolio project. Predicts bearing health
state (Normal / Degrading / Critical) from vibration signal data (NASA IMS bearing dataset),
served through a proper serving layer with monitoring — not a notebook demo.

Full scope, decisions, and rationale: see `docs/PRD.md`. Read it before proposing anything that
changes scope, architecture, or model approach — the classification-vs-RUL decision, the
Kubernetes/Machina/CI-CT-CD exclusions, and the latency/monitoring targets are already decided
and documented there. Don't reopen them without being asked.

## Conventions — follow these without being reminded

**Commits:** Conventional Commits format. `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
`test:`, `perf:`. Short, specific subject lines. See `CONTRIBUTING.md` for full detail.

**Branches:** `<type>/<short-description>`, e.g. `feat/predict-endpoint`.

**Workflow:** one Issue per non-trivial task, branch off `main`, small logical commits, PR
referencing the Issue (`Closes #N`), merge only when the relevant PRD Acceptance Criteria
(Section 10) are satisfied.

**Definition of Done:** check work against `docs/PRD.md` Section 10 before marking anything
complete — not "it runs locally."

**Merging and history:** never run `gh pr merge` without explicit approval for that specific
merge — a prior approval for branch deletion or PR opening does not extend to the merge action
itself. Always request explicit approval before any force-push or history rewrite (rebase,
cherry-pick that overwrites a remote branch), even when using `--force-with-lease`.

**Verify before proceeding:** in any multi-step git/gh workflow, confirm the actual output of
each command before moving to the next step, rather than assuming success. Chained shell
commands can partially fail silently (a missing file breaking a `cp`, then a downstream `git
commit` succeeding on stale state) — check real output, don't infer it from the absence of an
error on an unrelated line.

## Scope discipline

This project deliberately excludes (for MVP, see PRD Section 4): Kubernetes, full CI/CT/CD with
automated retraining, the Machina agent layer, multi-model serving, real-time streaming
ingestion, a FinOps dashboard. If a task seems to call for one of these, flag it and ask before
introducing it — don't add infrastructure the current milestone doesn't need.

## Repo structure (target — see PRD Section 10a)

```
docs/           PRD and other documentation
data/           gitignored — raw dataset, not committed
notebooks/      EDA, exploratory work
src/labeling.py health-state labeling logic — shared across features/training, deliberately
                top-level rather than nested (see rationale below)
src/features/   feature extraction pipeline
src/training/   training scripts, MLflow logging
src/serving/    FastAPI app, /predict and /metrics
demo/           simulated live-feed playback script
tests/
```

`src/labeling.py` stays top-level rather than moving under `src/features/`: it produces the
target variable (labels), not a feature — feature extraction and training both consume it, so
it doesn't belong to either. Revisit only if a genuine cross-cutting `src/common/` need emerges
later; don't create that structure speculatively for one file.

## Current milestone

M1-EDA is complete: dataset acquisition, exploratory analysis, and health-state label
threshold definition are done (Issues #8, #9, #10, #11 closed; see `docs/eda_findings.md`).
The M1.5-Housekeeping stretch that followed close-out is also complete (CI pipeline, unit
tests, README, pinned dependencies, notebook-output policy, data-versioning note, and these
conventions — Issues #18, #19, #24, #25, #26, #27, #28, #29 all closed). This was tracked
informally as "M1.5" in CLAUDE.md/Issues only, not a PRD milestone (PRD Section 11 notes it
without renumbering).

**M2-Features is complete** — all eight of its Issues are closed with merged PRs: #40
(rolling-window decision), #20 (onset-boundary hysteresis), #41 (core feature module +
code/data versioning), #43 (`experiment` column), #42 (validation notebook), #23
(skewness/crest factor redundancy), #22 (frequency-domain investigation), #49 (refresh of
figures made stale by #20). What it produced:

- `src/features/` — `extraction.py` (the pipeline), `versioning.py` (code+data hashing),
  `build_dataset.py` (CLI entrypoint), `candidate_features.py` (evaluated-but-unused).
- `data/processed/<name>_features.parquet` + `<name>_features_manifest.json` per experiment.
- `notebooks/04_feature_pipeline_validation.ipynb`, and five decision notes under `docs/`.
- Retained feature set: `rms`/`rms_ratio`, `kurtosis`, `skewness`/`skewness_smoothed`. Crest
  factor (#23) and the BPFO/BPFI/spectral-kurtosis features (#22) were evaluated and dropped —
  kept, tested, in `candidate_features.py` rather than deleted.

**M3-Model (baseline classifier, MLflow tracking) is in progress.** Its preparation sequence
is #65 → #67 → #69 → #21 → #72 (model training + `rms_ratio` ablation); all five are done:

- **#65** — `derive_critical_multiple` extracted into `src/labeling.py`, tested.
- **#67** — `src/features/build_training_dataset.py` joins the feature parquets with labels
  into `data/processed/training_dataset.parquet` (`docs/training_dataset_versioning.md`).
- **#69** — `docs/evaluation_protocol.md` commits to leave-one-experiment-out (LOEO) and to
  `Critical`-class recall as the headline metric. Written before any model existed, on
  purpose; **do not re-decide the metric or the split** when implementing training.
- **#21** — `src/training/` compares five class-imbalance approaches under LOEO;
  `class_weight='balanced'` adopted (`docs/class_imbalance_decision.md`). Added
  `scikit-learn` as the one new dependency.

- **#72** — `src/training/train_baseline.py` trains the `class_weight='balanced'` baseline
  under LOEO and runs the `rms_ratio` ablation as a first-class comparison
  (`docs/model_training_decision.md`). Rejected scaling approaches kept, tested, in
  `candidate_scalers.py` — same convention as `src/features/candidate_features.py`.

**The headline result is that the baseline works on two bearings and fails on the third,
and this is reported rather than averaged away** (`docs/PRD.md` §7 carries the note, not
just the decision doc). `Critical` recall is 0.913 / 1.000 on `2nd_test`/`3rd_test` and
0.059 on `1st_test`; the cross-fold mean (0.657) describes no fold and **should not be
quoted as the project's number**. Three things follow that later work should not re-derive:

- The `1st_test` fold has **two** independent failures, not the one #21 handed forward.
  The raw-`rms` scale problem (§3a) destroys `Normal` recall and is fixable; the
  **threshold-transfer problem (§3b) destroys `Critical` recall and is not** — all 17 of
  its `Critical` rows sit below the lowest `rms_ratio` its training fold ever labelled
  `Critical`, so no boundary learned from the other two can reach them. That is a property
  of the per-experiment `critical_multiple`, so it constrains **any** model trained on
  these labels. The next lever is the label/feature definition, not the estimator.
- **The ablation's headline gain is not a capability gain.** Removing `rms_ratio` raises
  mean `Critical` recall (0.657 → 0.892) while collapsing precision (0.870 → 0.603); on
  `1st_test` it predicts `Normal` for zero of 1,906 truly-`Normal` rows. Don't read that
  0.892 as an improvement.
- **Raw `rms` was kept**: #72 conditioned dropping it on it not earning its place
  elsewhere, and it does (macro-F1 0.945 → 0.747 on `3rd_test` without it).
  `prior_correction` (#21 §6's "arm to beat") was **not** re-tested — it shifts decisions
  further toward rare classes, which compounds the over-alarming above rather than
  isolating it. Still open.

- **#74** — `src/training/mlflow_tracking.py` instruments #21's and #72's LOEO comparisons
  with MLflow (`docs/PRD.md` §10/§11's requirement, deliberately left out of #72's own
  acceptance criteria and deferred here). Seven runs across two experiments
  (`m3-imbalance-comparison`: #21's five arms; `m3-baseline-ablation`: #72's `full`/
  `no_rms_ratio` configurations), logged metrics verified to match
  `docs/class_imbalance_decision.md`/`docs/model_training_decision.md` exactly
  (`docs/mlflow_tracking.md`). Dependency is `mlflow-skinny` + `sqlalchemy` + `alembic`,
  not the full `mlflow` package — full `mlflow` pins `pandas<3`, which silently downgrades
  this project's pinned `pandas==3.0.5` (Issue #27) on install; verified empirically before
  choosing the lighter combination. Tracking store is local SQLite (`mlflow.db`,
  gitignored, per-clone), not MLflow's plain file store, which is now in maintenance mode.

M3 left one item open: a servable model artifact — #72 deliberately persists none, since
LOEO trains three models per configuration and there is no single "the model" to save.

**M4-Serving is in progress**, and it is where that item landed:

- **#78** — `docs/serving_design.md`, decided before any `src/serving/` code: the
  `/predict` contract (client sends a raw single-window signal + `bearing_id`; the
  **server** owns rolling history, so `extraction.py`'s window/baseline logic is never
  duplicated client-side), per-bearing in-memory state (single process, no external
  store — which makes single-worker a hard constraint), a cold-start rule for a bearing's
  first 50 files (expanding baseline + `baseline_status` flag, never refuse), pooled
  training for the served model with a **static** disclosure of the known
  `1st_test`-shaped failure mode on every response, and explicit non-goals.
- **#80** — `src/training/train_serving_model.py` trains that pooled model on all three
  experiments and persists it to `models/serving_model.joblib` (**committed**, ~1.7 KB,
  byte-for-byte reproducible), plus `src/training/serving_model_tracking.py` for its
  MLflow run. Details and the gitignore-exception reasoning:
  `docs/serving_model_artifact.md`. Two things later work should not re-derive: the
  configuration is **imported** from `train_baseline`/`evaluation`, never re-declared, so
  it cannot drift from what #72 measured; and the pooled model's in-sample metrics
  (macro-F1 0.946) are **fit, not capability** — every logged metric is `insample_`-prefixed
  for that reason, and `docs/model_training_decision.md` §6 remains the honest number.
- **#82** — `src/serving/state.py` (`BearingState`, `BearingStateStore`) and
  `src/serving/features.py` (`compute_online_features`, `OnlineFeatureExtractor`) reframe
  M2's batch feature computation as a per-request update: §2's four state fields, §3's
  cold-start rule, no API framework. Two results not to re-derive: **exact bit-equality
  with `pandas`' rolling mean is not attainable incrementally** — `pandas` carries one
  Kahan-compensated running sum across the whole column, so its row `i` depends on every
  earlier row's arithmetic, not just the window. Measured over all 9,464 real files:
  `rms`/`kurtosis`/`skewness`/`skewness_smoothed` bit-identical, `rms_ratio` bit-identical
  on 80.5–81.4% of post-warmup rows and ≤2 ULP on the rest; replicating the `pandas`
  internal was considered and rejected. And the 50th file (index 49) answers
  **`"stable"`**, not `"warming_up"` — a
  label-only choice, since its expanding baseline already equals the locked one.

## When in doubt

Prefer asking over assuming when a decision would affect scope, architecture, or the
Acceptance Criteria. Small, reviewable steps over large speculative changes.
