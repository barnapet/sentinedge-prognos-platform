# The Pooled Serving Model Artifact (Issue #80)

Decision note accompanying `src/training/train_serving_model.py` and
`src/training/serving_model_tracking.py`. First implementation step of M4-Serving, and the
first module in this project to **persist** a model: #21's and #72's harnesses deliberately
train and discard three throwaway models per run (`docs/model_training_decision.md` §5),
which is correct for measuring generalization under LOEO but leaves nothing to serve.

What to train was already decided — `docs/serving_design.md` §4 committed to one model on
all three experiments pooled, with the exact configuration
`docs/model_training_decision.md` adopted. This note decides the three things that decision
left to implementation: **where the artifact lives and whether it is committed**, **whether
re-running reproduces it**, and **how its MLflow run is told apart from the evaluation
runs**.

**Decision up front:** `models/serving_model.joblib` + `models/serving_model_manifest.json`,
**committed to the repo** as an explicit, narrow exception to `.gitignore`'s model-artifact
rule; training is byte-for-byte reproducible (verified across processes, not assumed); and
the MLflow run lives in its own `m4-serving-model` experiment tagged
`run_purpose=serving_artifact`, with every metric prefixed `insample_`.

## 1. What is persisted, and that it matches the adopted configuration

The whole fitted `Pipeline` — `StandardScaler` + `LogisticRegression` — not just the
classifier. The scaler is a *fitted* component (it carries the pooled training rows' means
and scales), so persisting the classifier alone would leave the serving layer feeding
unscaled features to a model fitted on scaled ones: silently wrong, with no error.
`docs/evaluation_protocol.md` §1's reason for keeping the scaler inside the pipeline applies
verbatim at serving time.

The configuration is **imported, not re-declared**. `train_serving_model.py` uses
`train_baseline.BASELINE_STRATEGY` (the same `Strategy` object #72 used) and
`evaluation.FEATURE_MATRIX_COLUMNS` (the same five columns), so there is no second copy of
the configuration that could drift from the one #72's LOEO numbers were measured on:

| | Adopted (`docs/model_training_decision.md`) | Persisted artifact |
|---|---|---|
| Features | `rms`, `rms_ratio`, `kurtosis`, `skewness`, `skewness_smoothed` | same, same order |
| Imbalance treatment | `class_weight='balanced'` | `class_weight='balanced'` |
| Model params | `BASELINE_MODEL_PARAMS` (`max_iter=2000`, `random_state=0`) | identical |
| Preprocessing | `StandardScaler` inside the `Pipeline` | identical |
| Split | LOEO, three folds | **pooled, no held-out fold** (§4 of `docs/serving_design.md`) |

The split is the one intended difference. `tests/test_train_serving_model.py` pins the rest
by *independently* rebuilding a pipeline from `imbalance.make_baseline_model` in-test and
requiring the loaded artifact to produce identical `predict` and `predict_proba` output — a
copy-pasted configuration that drifted would fail that test rather than pass it quietly.

## 2. Decision: `models/`, committed — not gitignored

This is the decision with a genuine second defensible answer, so both are stated.

### The case for gitignoring it (rejected)

Real, and it is the status quo. `.gitignore` already carries `*.joblib`, `models/*` and the
comment *"Model artifacts (large binaries — don't commit trained models directly)"*, and
**every** generated artifact in this repo is currently gitignored without exception: the
three feature parquets and their manifests, `training_dataset.parquet` and its manifest,
`mlflow.db`, `mlartifacts/`. Committing this one breaks a pattern that has held all project.
A pickled sklearn estimator is also version-coupled — it is only guaranteed to load under
the library versions that wrote it — so a committed binary can rot against a future
`requirements.txt` bump in a way a `.parquet` would not.

### The case for committing it (adopted)

Three things, and the first two answer the objections above rather than merely outweighing
them:

- **The rule's stated reason does not apply.** `.gitignore`'s comment says *large*
  binaries. This artifact is **1,745 bytes**, its manifest **1,203 bytes** — about 2.9 KB
  together, smaller than most source files in `src/`. Honouring the rule's letter while its
  purpose is absent would be cargo-culting, so the exception is written into `.gitignore`
  with its reasoning attached rather than left as a silent override.
- **It is auditable, not opaque** — which is the real objection to binaries in version
  control. Training is byte-for-byte deterministic (§3), the manifest beside it records the
  artifact's SHA-256, and `verify_artifact_integrity()` checks the file against that hash in
  one call (`tests/test_train_serving_model.py` covers both the match and the tamper case).
  A committed blob that anyone can regenerate and hash-compare is a different object from
  one that has to be trusted.
- **It is the one artifact serving needs that cannot be regenerated cheaply.**
  `docs/PRD.md` §10 requires *fresh clone → running demo in under 15 minutes*, and G4 the
  same. Regenerating this model requires `training_dataset.parquet` → the three feature
  parquets → the **6.2 GB** raw NASA archive. Nothing at *serve* time needs the parquets;
  the container needs exactly this one file at startup. Gitignoring it would put a 6.2 GB
  download between a fresh clone and a served prediction.

**Honest limit on that third argument, stated rather than glossed:** committing the model
does **not** by itself deliver the <15-minute criterion today. `data/*` is entirely
gitignored, so a fresh clone has no vibration snapshots to play back either — the demo's
input data is a separate, still-open problem for a later M4 issue. What this decision does
is remove the *model* from that critical path, so that the remaining problem is only about
demo input. It is one of two blockers cleared, not the criterion met.

### The version-coupling objection, handled rather than dismissed

`serving_model_manifest.json` records `library_versions` (Python, scikit-learn, numpy,
joblib) alongside the artifact, and `requirements.txt` pins `scikit-learn==1.9.0` and
`numpy==2.4.6`. If a future bump makes the committed pickle stale, the manifest is what
makes that diagnosable instead of mysterious, and regenerating is one command (§5).

### Scope of the exception

Deliberately two filenames, not a directory:

```gitignore
models/*
!models/.gitkeep
!models/serving_model.joblib
!models/serving_model_manifest.json
```

Verified with `git check-ignore`: `models/serving_model.joblib` and its manifest are
trackable, while `models/other_model.joblib` and `models/scratch.pkl` remain ignored. A
future larger model does not inherit this exception by accident — it would need its own
decision, which is the point.

## 3. Reproducibility: byte-for-byte, verified

`docs/training_dataset_versioning.md` verified that re-running the training-dataset join
reproduces the same `combined_hash`. The equivalent claim here is stronger, because it is
what §2 leans on to justify committing a binary, so it was measured rather than argued:

**Re-running the training produces a byte-identical artifact.** SHA-256 of the serialised
pipeline, from four separate Python processes:

| Run | SHA-256 (first 24 hex chars) |
|---|---|
| process 1 | `3b4a3cd275fb9a2e43d839b3` |
| process 2 | `3b4a3cd275fb9a2e43d839b3` |
| process 3, `PYTHONHASHSEED=12345` | `3b4a3cd275fb9a2e43d839b3` |
| process 4, `PYTHONHASHSEED=99` | `3b4a3cd275fb9a2e43d839b3` |

Separate processes and varied hash seeds, because same-process repetition would not rule
out state carried between fits, and Python's per-process hash randomisation is the obvious
candidate for a source of ordering nondeterminism.

Why it holds: there is no randomness left in the adopted path. `random_state=0` is fixed
(`BASELINE_MODEL_PARAMS`), the adopted configuration does **no** resampling (the
resampling arms were #21's, and `class_weight='balanced'` won), no hyperparameter is tuned
anywhere (`docs/model_training_decision.md` §5's leakage guard), the LOEO fold ordering that
could have mattered is gone since nothing is held out, and `joblib.dump` embeds no timestamp.

This matters beyond tidiness: it means the committed binary is checkable rather than
trusted, and `tests/test_train_serving_model.py::test_refitting_produces_a_byte_identical_artifact`
pins it at the byte level rather than at the metric level — two models can score identically
and still differ.

### Provenance chain

`serving_model_manifest.json` extends the same chain
`docs/training_dataset_versioning.md` §2 built, one link further:

- **`serving_code_hash`** — SHA-256 over `imbalance.py`, `evaluation.py`, and
  `train_serving_model.py`: the files whose content determines what this model *is*. Its own
  tuple (`SERVING_CODE_FILES`), deliberately **not** merged into
  `src/features/versioning.py`'s `GENERATING_CODE_FILES`, for the reason #65/#67 already
  established — doing so would change every feature parquet's `combined_hash` and imply a
  stale-cache signal against raw data that did not change.
- **`training_dataset_version`** — Issue #67's `training_dataset_manifest.json`
  `combined_hash`, reused rather than recomputed.
- **`combined_hash`** — `SHA256(serving_code_hash:training_dataset_version)`, via the same
  generic `compute_combined_hash`.
- Plus `model_sha256`, `library_versions`, `split`, `trained_on`, `n_training_rows`,
  `class_support`, and the full adopted configuration.

Answering: *was this artifact produced by this training code, from that dataset version?*

## 4. The MLflow run, and how it is told apart from #21/#72's

Logged to the same local SQLite store `docs/mlflow_tracking.md` chose, in **its own
experiment** `m4-serving-model` (alongside `m3-imbalance-comparison` and
`m3-baseline-ablation`, not inside them). Three mechanisms make it unmistakable, because
Issue #80 asks specifically that "which run produced the actual served artifact" be obvious:

- **Tag `run_purpose=serving_artifact`** (plus `issue=80`, `component=serving_model`).
  `mlflow.search_runs(filter_string="tags.run_purpose = 'serving_artifact'")` returns
  exactly one run out of the eight now in the store.
- **Every metric is prefixed `insample_`**, and `metrics_scope=in_sample_training_fit` is
  logged as a parameter. This is the honesty-critical part: the pooled model has no held-out
  rows, so its metrics measure **fit, not capability**. In-sample macro-F1 is **0.946**, with
  `Critical` recall **0.991** — numbers that look far better than
  `docs/model_training_decision.md` §1's LOEO results and mean something entirely different.
  Sitting unprefixed next to #72's runs in a UI table, they would invite exactly the misread
  this project's reporting standard exists to prevent. The run also logs
  `loeo_evaluation_experiment=m3-baseline-ablation`, pointing at where its actual
  generalization evidence lives.
- **The binary itself is a run artifact.** `serving_model.joblib` and its manifest are
  attached to the run, and `model_sha256`/`combined_hash` are logged as parameters, so the
  run and the file on disk are tied together rather than merely adjacent.

**`mlflow.log_artifact`, not `mlflow.sklearn.log_model`** — verified, not assumed: the
sklearn flavor is unavailable under `mlflow-skinny` and raises
`ModuleNotFoundError: No module named 'skops'`. Installing the full `mlflow` package to get
it would reopen exactly the conflict `docs/mlflow_tracking.md` §1 documented (full `mlflow`
pins `pandas<3` and silently downgrades this project's pinned `pandas==3.0.5`). Logging the
joblib file as a plain artifact gives what this issue needs — the exact served bytes,
attached to the run — with no dependency change at all.

## 5. Reproducing

```
python -m src.features.build_training_dataset    # Issue #67, if data/processed/ is empty
python -m src.training.train_serving_model       # trains, persists, and logs the MLflow run
```

Re-logging a run for an artifact already on disk, without retraining:

```
python -m src.training.serving_model_tracking
```

Checking the committed artifact against its manifest:

```python
from src.training.train_serving_model import verify_artifact_integrity
verify_artifact_integrity()   # True if models/serving_model.joblib still hashes as recorded
```

## 6. What this does not settle

- **The in-sample metrics are not a capability claim, and nothing here re-evaluates the
  model.** `docs/model_training_decision.md` §6 remains the honest statement: `Critical`
  recall 0.913 / 1.000 on `2nd_test`/`3rd_test`, **0.059 on `1st_test`**. Pooled training
  means the artifact has now *seen* `1st_test`'s rows, so §3b's specific unreachability does
  not bind it — but as `docs/serving_design.md` §4 already sets out, that is fitting
  `1st_test`, not evidence of generalizing to an unseen impulsive/inner-race bearing, of
  which this dataset contains exactly one.
- **The service's disclosure of that failure mode is not implemented here.**
  `docs/serving_design.md` §4 specifies a static `model_notes` field on every `/predict`
  response; this issue produces the artifact, not the API.
- **No retraining trigger, schedule, or model registry.** The artifact is static and
  produced offline, per `docs/PRD.md` §4's exclusion of automated retraining and
  `docs/serving_design.md` §5's non-goals.
- **Demo input data.** §2 notes this: the <15-minute fresh-clone criterion also needs
  playback input that the repo does not currently carry. Out of scope here, and named so it
  is not mistaken for solved.
