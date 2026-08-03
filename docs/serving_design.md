# Serving Design (Issue #78)

Decision-only note, no implementation — same before-code discipline as
`docs/evaluation_protocol.md` (#69) and `docs/feature_windowing_decision.md` (#40). First step
of M4-Serving. Written after a read-only readiness audit (2026-08-01) found that a naive
`/predict(window) -> label` endpoint cannot be built without first deciding who owns rolling
history, because three of `src/features/extraction.py`'s five `FEATURE_COLUMNS` are stateful:

| Feature | Serving-ready? | Requires |
|---|---|---|
| `rms`, `kurtosis`, `skewness` | Stateless | Just the current window |
| **`rms_ratio`** | **Stateful** | Previous 9 RMS values + that bearing's first-50-file baseline mean |
| **`skewness_smoothed`** | **Stateful** | Previous 9 skewness values |

This matters more than a generic "add caching" problem because `docs/model_training_decision.md`
§2 found `rms_ratio` carries most of the model's discriminative power — the hardest-to-serve
feature is the most important one. No shortcut around solving statefulness is available.

Every decision below is final for M4's first implementation. Where a decision had a genuine
second defensible path, both sides are stated and one is chosen — nothing is left as "TBD."

## 1. The `/predict` contract

**Decision: the client sends a raw single-window reading — one channel's raw vibration signal
for one snapshot, plus a `bearing_id` — and the server computes every feature, stateless and
stateful alike. The server owns 100% of feature computation; the client owns none of it.**

> **Implemented in Issue #84.** `POST /predict` in `src/serving/api.py` accepts exactly
> this payload (`PredictRequest`: `bearing_id`, `signal`, an optional display-only
> `timestamp`), calls `src.serving.features.OnlineFeatureExtractor.observe` once per
> request in arrival order (Issue #82), and runs the persisted pipeline (Issue #80) on the
> result. The response is `{label, baseline_status, model_notes}` — `model_notes` always
> Section 4's exact disclosure text, verified byte-for-byte in
> `tests/test_api.py::test_model_notes_constant_matches_the_design_doc_byte_for_byte`
> against this document's own Section 4 code block, so the two cannot silently drift apart.

### The two options considered

| | Client sends | Server does | Risk |
|---|---|---|---|
| **A — server owns history** | Raw signal (one channel, ~20,480 floats) + `bearing_id` | Computes `rms`/`kurtosis`/`skewness` per call, maintains rolling state for `rms_ratio`/`skewness_smoothed` | Server must hold per-bearing state (Section 2) |
| **B — client owns history** | Pre-computed 5-column feature vector | Runs the model on whatever it's given | Client must reimplement `add_rolling_rms_ratio`/`add_rolling_skewness`'s exact rolling-window and baseline logic |

Option B was rejected. `src/features/extraction.py`'s rolling logic is not a generic rolling
mean — it has specific, easy-to-get-wrong parameters (`ROLLING_WINDOW = 10`, `min_periods=1`,
a baseline fixed from exactly the *first* 50 files, not a trailing window) that are already
pinned by `docs/feature_windowing_decision.md` and by unit tests. A client-side reimplementation
is a second copy of that logic with no mechanism forcing it to stay in sync if `extraction.py`
ever changes, and a silent divergence there would feed the model a feature vector it was never
trained on — with no error, just quietly wrong predictions. Option A eliminates this by
construction: there is exactly one implementation of the rolling/baseline math, imported and run
inside the serving process itself, not translated into a second client.

Option A's cost — the server must hold state — is real, but it's a cost this project already has
to pay for `rms_ratio`/`skewness_smoothed` no matter which option is chosen (Option B just moves
the same statefulness to the client instead of removing it). Section 2 scopes that cost to what
a local-container demo actually needs.

### Why "raw signal," not "raw file"

The dataset's raw snapshot files are multi-channel (up to 8 columns for `1st_test`, per
`docs/eda_findings.md` §1); only one channel per experiment is the tracked bearing
(`src/features/extraction.py`'s `EXPERIMENTS` dict). Channel selection is not a stateful,
learned, or model-relevant computation — it's a fixed lookup the demo playback script already
has to make to know which physical bearing it's replaying. Requiring the server to also carry
`EXPERIMENTS`-style per-bearing channel maps would tie a general-purpose serving endpoint to
this specific three-experiment dataset. The client (`demo/playback.py`, out of scope for this
issue but the consumer this contract is designed for) selects the channel and sends that one
array; the server never needs to know which column index a bearing's data came from.

> **That client exists as of Issue #86.** `demo/playback.py` holds the `EXPERIMENTS` lookup
> and sends one channel's raw array per request, in order from file 0 — the split this
> section describes, unchanged. `tests/test_demo_playback.py` pins the channel choice per
> experiment, since a wrong column would still produce a perfectly plausible-looking
> prediction: the server has no way to notice it was handed a different bearing's vibration.
> The demo's default source is a committed 6.0 MB signal sample rather than the raw dataset,
> so `docker compose up` works on a fresh clone (`demo/sample.py`, `docs/PRD.md` §10).

### Payload shape (illustrative, not an API spec — no server code in this issue)

```
POST /predict
{
  "bearing_id": "1st_test-bearing3",
  "signal": [<20480 floats, one channel, one snapshot>]
}
```

No client-supplied file index or sequence number. The server infers each bearing's position in
its own history from **arrival order of requests carrying that `bearing_id`**, not from anything
the client asserts — this matches how `demo/playback.py` is expected to work (replay one
bearing's files strictly in chronological order, per `docs/PRD.md` §8's playback framing) and
avoids trusting a client-supplied index that could be wrong, replayed, or out of order. A
`timestamp` field may be included for logging/monitoring display only; it is never read by the
feature-computation path. This constrains the eventual playback implementation: a bearing's
stream must start from file 0 and proceed in order for its rolling state to be correct — resuming
mid-stream after a server restart is explicitly not supported (Section 5).

## 2. State ownership

**Decision: an in-memory, single-process dictionary keyed by `bearing_id`, holding exactly the
values needed to reproduce `add_rolling_rms_ratio`/`add_rolling_skewness`'s math incrementally.
No database, no Redis, no other persistent or distributed store.**

> **Implemented in Issue #82.** `src/serving/state.py` is this dictionary
> (`BearingStateStore`) and the per-bearing container (`BearingState`) with exactly the four
> fields named below; `src/serving/features.py` computes one file's five features against it,
> reusing `src/features/extraction.py`'s `compute_rms`/`compute_kurtosis`/`compute_skewness`
> rather than restating them. The single-worker constraint below is carried as a module-level
> docstring on the state module, since it follows from that module's design rather than from
> the API layer's. `tests/test_serving_features.py` replays a whole experiment file-by-file
> and checks the result against the batch pipeline's output: `rms`/`kurtosis`/`skewness`
> bit-identical, the two rolling features equal to within 2 ULP — see the PR for Issue #82
> for why exact bit-equality on the rolling means is not attainable against `pandas`.

### What's actually needed per bearing

`src/features/extraction.py`'s two stateful features need, per `bearing_id`:

- **`rms_history`**: the last 10 raw `rms` values (a fixed-size deque) — enough to reproduce
  `rolling(10, min_periods=1).mean()` for the next incoming file.
- **`skewness_history`**: the last 10 raw `skewness` values, same reasoning.
- **`baseline_rms_values`**: raw `rms` values from files 0 through 49, until 50 are collected —
  then their mean is fixed forever as that bearing's baseline (matching
  `add_rolling_rms_ratio`'s `out["rms"].head(baseline_n_files).mean()` exactly). After the 50th
  file, only the fixed scalar mean needs to be kept; the 50 raw values can be dropped.
- **`file_count`**: how many files this bearing has been served, used to decide whether the
  baseline is still accumulating (Section 3) and to size `baseline_rms_values`.

This is a few hundred floats per actively-monitored bearing — trivially small, and exactly the
data `src/features/extraction.py` already computes in one batch pass over a whole experiment.
Serving reframes the same computation as an online update per request instead of a `pandas`
column operation; it is not new math.

### Why not a database or distributed cache

The issue explicitly asks this to be scoped to what's needed, not to a production-grade
distributed cache, and `docs/PRD.md` §10/§11 back that framing: `<500ms for single-window
inference, local container, no batch queueing` and no Kubernetes/multi-tenant serving in MVP
scope (`docs/PRD.md` §4). A demo has:

- One process, one container (`docker compose up`, per `docs/PRD.md` §8's proposed architecture).
- A handful of concurrently-monitored bearings (the demo plays back at most the three dataset
  experiments), not a fleet.
- No requirement to survive a restart mid-run — `docs/PRD.md` §8's playback note already frames
  the demo as a compressed, replayable simulation, not a live production stream with an uptime
  guarantee.

A database (SQLite, Postgres) or an external cache (Redis) would add a second moving part to
`docker compose up` — a service to start, a connection to manage, a schema or key format to
design — for state that fits in a few kilobytes of process memory and does not need to outlive
the process. That complexity is the kind `docs/PRD.md` §4 already rules out by name for the
platform's bigger architectural choices (Kubernetes, multi-tenant serving); a persistent store
for per-bearing rolling history would be the same mistake at smaller scale. If a later phase
needs multiple server replicas or restart-durable state, that is a distributed-systems problem
the in-memory dict is explicitly not designed to solve (Section 5) — not a gap in this decision,
a boundary of it.

### Concurrency constraint this decision implies

An in-memory dict is only correct if there is exactly one process holding it. **The server must
run as a single worker process** (e.g. `uvicorn` without `--workers N > 1`) for this design to be
correct — multiple worker processes would each hold their own copy of the dict, and a given
bearing's requests could be routed to different workers with different, diverging rolling
histories. This is stated here as a hard constraint on the eventual implementation, not deferred
to it, because it follows directly from choosing in-memory-single-process state and would
otherwise be an easy mistake to make invisibly (the server would still respond in under 500ms,
just with silently wrong rolling values on some fraction of requests). Multi-worker/horizontal
scaling is a non-goal (Section 5).

> **Enforced, not just documented, in Issue #84** — two independent layers, since neither
> alone covers every way this could be violated. (1) `src/serving/main.py`, the one
> documented run command, passes an already-built `FastAPI` object to `uvicorn.run`; passed
> an object rather than an import string, `uvicorn` cannot fork additional workers at all
> and exits immediately (`SystemExit(3)`) if `workers > 1` is requested, rather than
> silently starting one. Confirmed empirically, not assumed from changelogs — see the PR
> for Issue #84. (2) `src/serving/single_worker.py` takes an exclusive, non-blocking OS
> file lock at app startup, independent of how the process was launched; a second process
> — `uvicorn ... --workers N` invoked directly, a second `python -m src.serving.main`, any
> launcher neither of the above anticipated — fails loudly at startup
> (`SingleWorkerViolation`, process exit code 3) rather than silently serving alongside the
> first. Both confirmed against the real process, not only in test isolation.

## 3. Cold-start behavior

**Decision: never refuse to score. For files 0–49 of a new `bearing_id`, compute `rms_ratio`
against an *expanding* baseline (the mean of whatever RMS values have been seen so far, 1 up to
50) instead of the eventual fixed 50-file baseline, and mark every response made under this
condition with an explicit `baseline_status: "warming_up"` field. Once the 50th file is seen, the
baseline locks to that fixed mean permanently for that bearing, and `baseline_status` switches to
`"stable"` and never reverts.**

> **Implemented in Issue #82**, in `BearingState.baseline_status` /
> `BearingState.effective_baseline_rms`. One ambiguity this section left had to be resolved to
> write it: "files 0–49" and "once the 50th file is seen" disagree about the 50th file (index
> 49) itself. It is answered `"stable"`, following this section's own `file_count < 50`
> formulation with the current file counted. The choice is label-only and changes no number —
> on that file the expanding baseline is the mean of files 0–49, which *is* the locked
> baseline. Reasoning in full in the PR for Issue #82.

### Why not refuse

Refusing to score files 0–49 was the other option the issue named. It was rejected for this
project specifically: `docs/PRD.md` §8 frames the demo as replaying "a bearing's run-to-failure
history" so a dashboard can "visibly animate a bearing's degradation," and every bearing's
history starts at file 0. A rule that refuses the first 50 of roughly 984–6,324 files per
experiment (`docs/eda_findings.md` §1) discards 0.8–5% of a demo run for a period that is, per
that same table, deep in each bearing's `Normal` region regardless — not a period where a
missing prediction would hide a meaningful health-state transition. There is no safety
justification for refusing in a portfolio demo the way there might be in a system making a real
maintenance decision.

### Why an expanding baseline, not a placeholder value

This mirrors a convention the codebase already commits to rather than inventing a new one.
`src/features/extraction.py`'s `add_rolling_rms_ratio` and `add_rolling_skewness` both use
`rolling(10, min_periods=1)` for the 10-file window specifically so that "no row is ever `NaN`,
including the first `rolling_window - 1` rows" (`docs/feature_windowing_decision.md` §3) — the
window shrinks to whatever's available rather than blocking or padding. The 50-file baseline has
the identical shape of problem (a lookback quantity that isn't full-size yet) and gets the
identical fix: an expanding-window mean over whatever's been collected (1 to 50 files), not a
fixed placeholder like `1.0` or a refusal. `docs/feature_windowing_decision.md` §3 already
names the consequence of this pattern — "elevated variance in the first 9 files' ratio estimate,"
not a missing-data problem — and that reasoning applies without modification to the baseline's
own 1-to-50-file convergence: an early estimate, not an absent one.

This is also, concretely, the closest existing precedent the issue points to:
`docs/eda_findings.md` §3's ~2.6x `critical_multiple` fallback exists because a bearing's
eventual peak `rms_ratio` — a retrospective, look-ahead quantity — "cannot be evaluated at
serving time at all," and the documented response was to substitute a reasoned, explicitly-named
stand-in rather than block. The baseline's cold-start problem is not the same quantity (it
becomes exactly computable at file 50, not permanently unknowable like the peak), but the pattern
transfers directly: when a value serving needs isn't available yet, substitute a stated,
inspectable approximation and say so, rather than fail closed or fail silently.

### Why a flag, not a rejection or a silent substitution

Two things are true at once: refusing loses demo value for a period where nothing important is
usually happening (above), and an expanding baseline from 1–2 files really is a much noisier
estimate than the eventual 50-file one — a single early high-vibration transient could set an
unrepresentative baseline for that whole bearing's remaining life. Silently returning a
prediction with no indication of which regime produced it would hide that. `docs/PRD.md` §5's
plant-engineer framing and this project's general "state it, don't average it away" convention
(`docs/evaluation_protocol.md` §5, `docs/model_training_decision.md` §1) both argue for making
the distinction visible rather than deciding it doesn't matter. `baseline_status` is cheap to
compute (it's just `file_count < 50`) and lets a consuming dashboard choose to visually
de-emphasize or annotate warming-up predictions without the server having to decide that policy
on its behalf.

## 4. What "the served model" is

**Decision: for M4, train one model on the pooled union of all three experiments
(`1st_test` + `2nd_test` + `3rd_test`), using the exact fixed configuration
`docs/model_training_decision.md` already adopted — standardized M2 feature set (`rms`,
`rms_ratio`, `kurtosis`, `skewness`, `skewness_smoothed`), `class_weight='balanced'`,
`BASELINE_MODEL_PARAMS` — and persist that single artifact. Every `/predict` response carries a
static, always-present disclosure of the model's known `1st_test`-shaped failure mode. The
disclosure does not attempt to be conditional on the incoming signal.**

> **Implemented in Issue #80.** `src/training/train_serving_model.py` trains exactly this
> model and persists it to `models/serving_model.joblib` (committed, ~1.7 KB, byte-for-byte
> reproducible), with an MLflow run in its own `m4-serving-model` experiment tagged
> `run_purpose=serving_artifact` to keep it distinct from #21/#72's evaluation-only runs.
> The artifact-location, gitignore, reproducibility, and provenance decisions this section
> left to implementation are recorded in `docs/serving_model_artifact.md`. The `model_notes`
> disclosure below was unimplemented at that point — it belongs to the API layer, not the
> artifact — and **is implemented in Issue #84**: `src/serving/model_notes.py` holds the
> text below verbatim, and `src/serving/api.py` attaches it to every `/predict` response,
> unconditionally, exactly as this section requires.

### Why pooled training, not one of the LOEO folds

`docs/model_training_decision.md` §5 is explicit about why LOEO trains and discards three models
without persisting any of them: "LOEO trains three models per configuration and there is no
single 'the model' to save... a more capable model class trained on these same labels would face
the same obstacle on this same fold" if it held out `1st_test`. That reasoning is specific to
*evaluation* — LOEO exists to measure generalization to an unseen bearing, and picking any one
fold's model as "the" servable artifact would arbitrarily privilege whichever two experiments it
trained on. For **serving**, the question is different: there is no reason to withhold any of the
three known experiments from the model that will actually answer requests, since none of it is
being held out to estimate anything anymore — the LOEO numbers already reported in
`docs/model_training_decision.md` and `docs/PRD.md` §7 remain the record of this model class's
measured generalization. Training on all three uses every labeled example this project has for
the model that will run in production, which is the ordinary reason to prefer pooled training
once the evaluation phase that needed folds is done.

### What this changes, and what it doesn't, about the known `1st_test` failure

This distinction matters and is stated explicitly rather than left for a reader to work out:

- **What changes:** `docs/model_training_decision.md` §3b's specific unreachability finding —
  "all 17 of `1st_test`'s `Critical` rows lie below the lowest `rms_ratio` its training fold
  ever labelled `Critical`" — was a property of a fold that excluded `1st_test` from training.
  The pooled model *does* see `1st_test`'s own rows during training, so that literal
  unreachability does not apply to it: the pooled model has been shown `1st_test`'s actual
  Critical-labeled examples and can fit a boundary that includes them.
- **What doesn't change:** this is fitting `1st_test`, not demonstrating the model generalizes
  to a bearing *like* `1st_test` that it hasn't seen. `docs/evaluation_protocol.md` §6 already
  states the limit this runs into: with `n = 1` inner-race experiment in the whole dataset,
  nothing distinguishes "the model generalizes across inner-race/impulsive failures" from "the
  model fits this one bearing's idiosyncrasies." A future bearing with an impulsive,
  inner-race-style degradation signature is, evidentially, in the same position `1st_test` was
  under LOEO: the only prior evidence this project has about that failure mode's generalization
  is the LOEO `1st_test` fold's result — `Critical` recall 0.059, macro-F1 0.152
  (`docs/model_training_decision.md` §1) — and pooling does not add a second inner-race example
  to improve on that. Pooling changes whether the model has memorized `1st_test` specifically;
  it does not change how much evidence exists about the failure mode in general.

### Why the disclosure is static, not conditional

A conditional disclosure — flagging a specific response as "this one resembles the risky case" —
would need a detector for "does this signal look impulsive/inner-race-like," and no such detector
exists or has been evaluated. Building one now would be exactly the kind of unearned, unmeasured
addition `docs/model_training_decision.md` and this project's general conventions argue against
(`docs/frequency_domain_decision.md` already investigated and rejected spectral/impulsive
features for a related purpose — see `docs/eda_findings.md` §4 — so an ad hoc heuristic here
would be weaker evidence than something already tried and found insufficient). The honest
alternative available without inventing new, unvalidated machinery is a disclosure that is always
present, worded plainly, and points to where the full evidence lives:

```
"model_notes": "Trained on all 3 dataset experiments (1st_test/2nd_test/3rd_test) pooled.
LOEO evaluation found this model class does not reliably detect the Critical health state
on impulsive, inner-race degradation signatures resembling the 1st_test bearing (Critical
recall 0.059 when that experiment was held out) — see docs/model_training_decision.md.
Reliable on the two outer-race, amplitude-driven failure modes evaluated (Critical recall
0.913 / 1.000)."
```

This is included on **every** response, not just ones the server suspects are risky, because the
server has no basis for suspecting anything about a given request — it is a property of the
model class and this dataset's coverage, disclosed the same way regardless of input. This also
matches `docs/PRD.md` §7's own convention: the PRD's Success Metrics table states the `1st_test`
failure next to the headline number unconditionally, not as a footnote that only appears in some
circumstances.

## 5. Non-goals

Stated explicitly so M4's implementation has a bounded scope to build against, per the issue's
own request and this project's general scope-discipline convention (`docs/PRD.md` §4):

- **Multi-bearing batch scoring in one request.** Section 1's contract is one window, one
  bearing, one response — no `/predict` variant that accepts a list of windows or bearings.
  Matches `docs/PRD.md` §10's "single-window inference... no batch queueing" framing exactly;
  a batch endpoint is a different contract this document does not define.
- **Multi-worker / horizontally-scaled serving.** Section 2's in-memory state requires exactly
  one server process. Making rolling state correct across multiple workers or replicas needs an
  external, shared store — out of scope, and out of scope for the platform generally per
  `docs/PRD.md` §4 (no Kubernetes, no multi-tenant serving).
- **State durability across process restarts.** A restarted server has no memory of any
  `bearing_id`'s history; a bearing being monitored across a restart would need to resume from
  file 0 (Section 1) to be correct again. No write-ahead log, snapshot, or persistent store is
  in scope for M4.
- **Production-grade auth, rate limiting, or multi-tenant isolation.** This is a local-container
  demo for a single reviewer running `docker compose up`, not a multi-user service.
- **Online or automatic retraining.** The persisted model (Section 4) is a static artifact
  produced offline; nothing in serving updates it. Matches `docs/PRD.md` §4's exclusion of full
  CI/CT/CD with automated retraining.
- **A confidence or reliability score conditioned on the input signal.** Section 4 explains why:
  no such detector exists or has been validated, and inventing one for this issue would be scope
  creep beyond a design-decisions document. The disclosure is static, not per-request-adaptive.
- **Real-time or streaming ingestion protocols (MQTT, OPC-UA).** Already excluded at the platform
  level (`docs/PRD.md` §4); `/predict` is a plain synchronous HTTP endpoint fed by
  `demo/playback.py`'s simulated cadence, not a streaming consumer.
- **Any server, API-framework, or Docker code.** Per the issue's own scope: this document is the
  decision record `src/serving/` implementation follows, not a partial implementation of it.

## Reconciling with `docs/PRD.md` §10/§11

- **`<500ms for single-window inference, local container, no batch queueing` (§10):** every
  decision above is sized to stay well inside this. Feature computation on one ~20,480-point
  signal (`compute_rms`/`compute_kurtosis`/`compute_skewness`, all `O(n)` over one window) and an
  update to a length-10 deque are sub-millisecond operations; the in-memory dict lookup (Section
  2) is `O(1)`; model inference is a single `scikit-learn` `Pipeline.predict` call on a 5-value
  feature vector. Nothing in this design does I/O, network calls, or unbounded work per request.
  The "no batch queueing" framing is honored directly by Section 1's contract (one window in, one
  response out) and Section 5's explicit exclusion of a batch endpoint.
- **§11's Milestone 4 scope ("containerized API serving the model"):** this document defines what
  that API's contract and the model it serves are, ahead of writing it, matching the sequencing
  `docs/evaluation_protocol.md` and `docs/feature_windowing_decision.md` already established for
  M2/M3 (decide before code). `src/serving/` implementation is the next issue in the M4 sequence,
  not this one.
- **§4's non-goals (Kubernetes, multi-tenant serving, streaming ingestion):** Section 5 above
  restates and extends these into serving-specific terms (single-worker process, no external
  state store) rather than re-deciding them.
