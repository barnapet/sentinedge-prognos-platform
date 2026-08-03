# Monitoring Design (Issue #88)

Decision-only note, no implementation — the same before-code discipline as
`docs/evaluation_protocol.md` (#69) and `docs/serving_design.md` (#78). First step of
M5-Monitoring, opened immediately after M4-Serving closed (`v0.4-serving`).

`docs/PRD.md` §10's M5 acceptance criterion is broad on purpose: *"at least one drift/health
signal (e.g., input feature distribution shift) visible on a dashboard, not just logged."*
This document turns that into six concrete decisions, each with the reasoning that produced
it, so `src/serving/` implementation has exactly one design to follow rather than several
plausible ones. No monitoring code, no dashboard code, and no new dependency are introduced
here — everything below is what a later issue implements.

Two pieces of already-decided context this document does not re-open: `docs/serving_design.md`
§2's single-worker/in-memory-state constraint (the server is one process, holding one
in-memory dict, for the lifetime of that process), and §4's static `model_notes` disclosure
(unconditional, on every response). `docs/serving_design.md`'s own M5-readiness note observed
that a drift signal is a natural **second** field on `PredictResponse`, not a breaking change —
that observation is taken up in Section 3.

## 1. What drift signal, computed against what baseline

**Decision: a per-feature z-score against that feature's pooled training-set mean and
standard deviation, flagged when `|z| > 3`, with a short rolling-persistence rule (Section 3)
so a single noisy reading cannot flip the flag alone.**

### Why z-score, not PSI or a KS-test

All three were named as candidates. The deciding constraint is architectural, not
statistical: `docs/serving_design.md` §1 fixes `/predict` as single-window, single-bearing,
one request in and one response out — no batch endpoint, per §5's own non-goals. Every
incoming observation is therefore **one scalar per feature**, not a sample.

- **Population Stability Index** compares two *binned distributions* — a reference histogram
  against a current one. It needs a current-side sample large enough to populate bins
  meaningfully, plus a choice of bin edges and a rule for near-empty bins. None of that is
  available from a single incoming reading; PSI would need its own accumulation window built
  first, before it could compute anything, and the binning scheme is itself a design decision
  this issue would otherwise have to make on top of the drift decision.
- **A KS-test** compares two *samples* directly (`scipy.stats.ks_2samp`) and is available as a
  dependency already (`scipy` is pinned). But it has the same shape problem as PSI: the
  "current" side of the comparison needs to be a batch of recent observations, not the one
  value a single `/predict` call produces. Using it here would mean building the same
  accumulation machinery PSI needs, for a test whose output (a p-value against an
  asymptotic null) is harder to explain on a dashboard than "how many standard deviations
  from what training saw" — more machinery for a less legible result.
- **A z-score against a fixed, precomputed (mean, std) pair per feature** needs nothing an
  incoming reading doesn't already have. It is defined the instant one scalar arrives, is
  O(1) per feature, and is exactly the granularity `/predict`'s architecture actually
  produces. The **rolling persistence** on top of it (Section 3) is what turns single-point
  noise into a distribution-shift signal — but that rolling window is built from the already-
  cheap per-point z-scores, not from re-deriving PSI/KS machinery under a different name.

> **Implemented in Issue #90.** `src/serving/drift.py`'s `compute_z_score`/`is_extreme`
> are exactly this — no PSI/KS-test machinery anywhere in `src/serving/`.
> `tests/test_drift.py` pins the threshold and the zero-std guard.

This is the same trade a fuller observability stack would resolve differently — PSI and
KS-tests are the right tool when a monitoring system has a real batch-scoring or micro-batch
ingestion path to compute a "current window" against. `docs/PRD.md` §10 explicitly frames
this project's serving layer as the opposite of that ("no batch queueing"), so reaching for a
batch-shaped statistic here would be solving a problem this architecture does not have —
exactly the kind of unearned infrastructure `docs/skewness_crestfactor_decision.md`'s Method
note and `docs/frequency_domain_decision.md`'s no-new-dependency framing both argue against
for a demo of this scale.

**Distribution shape is not a blocker.** None of the four monitored features (Section 2) is
Gaussian — baseline kurtosis sits around 3.4–3.5 across experiments
(`docs/feature_windowing_decision.md` §2), and baseline `|skewness|` around 0.03
(`docs/skewness_crestfactor_decision.md` §0) — but a z-score does not require normality to be
informative. Chebyshev's inequality bounds the tail mass at any `k` standard deviations
regardless of the underlying distribution's shape: `P(|Z| ≥ 3) ≤ 1/9 ≈ 11.1%` in the worst
case, for *any* distribution with finite variance. `|z| > 3` is therefore a conservative,
assumption-light threshold, not one that quietly assumes a bell curve it doesn't have. What is
given up, honestly: this is not a calibrated p-value the way a formal hypothesis test would
produce, and it is not meant to be — it is a coarse "does this look like anything training
ever produced" check, matched to a demo's needs, not a statistically calibrated production
alerting threshold.

### The baseline: pooled training distribution, all labels included

**Decision: the reference (mean, std) per feature is computed once, offline, from the full
pooled `data/processed/training_dataset.parquet` (#67) — all three experiments, all three
health-state labels together — not from `Normal`-only rows.**

This is a deliberate choice, because the more intuitive-sounding alternative ("baseline
should be what *healthy* looks like, so drift means moving away from healthy") asks a
question this system already answers a different way. Moving from `Normal` towards
`Critical` **is not drift** — it is exactly what the served model (#80) is trained to detect
and what its `label` field already reports. Conflating "the bearing is degrading" with "this
input doesn't look like anything the model has seen" would make the drift signal redundant
with the classifier's own output rather than complementary to it. The question this document
answers is narrower and different: *is this reading inside the envelope of values this
deployed model was ever fit against, degraded or not* — and that envelope has to include the
`Degrading`/`Critical` rows, because the model was fit on them too and is expected to handle
them. A reading outside the full pooled envelope is a stronger, more specific claim than "this
bearing looks unhealthy": it is "this looks unlike *anything* in training, healthy or not,"
which is the thing a classifier fit on the training distribution has no grounds to be
confident about, whatever label it outputs.

> **Implemented in Issue #90.** `src/training/compute_drift_baseline.py` computes exactly
> this: per-feature `(mean, std)` pooled over all 9,464 real training rows, all three
> experiments, all three labels, committed to `models/drift_baseline.json` (mirroring
> `models/serving_model_manifest.json`'s precedent — a small, human-diffable artifact, not
> gitignored). Measured, not assumed: `rms` mean=0.0963/std=0.0496,
> `kurtosis` mean=3.638/std=2.774, `skewness` mean≈0.0/std=0.104,
> `skewness_smoothed` mean≈0.0/std=0.056 — and the `training_dataset_version` this baseline
> was computed from matches, byte-for-byte, the hash already recorded in the committed
> `models/serving_model_manifest.json` (Issue #80), confirming both artifacts were built
> from the identical dataset version.

## 2. Which features to monitor — and why `rms_ratio` is treated differently

**Decision: `rms`, `kurtosis`, `skewness`, `skewness_smoothed` get the population z-score
drift check from Section 1. `rms_ratio` is excluded from that check and reported on the
dashboard separately, framed as a severity readout tied to the model's own prediction, not as
an independent drift signal.**

### Why the other four are exactly the right shape for a population baseline

`rms`, `kurtosis`, `skewness`, and `skewness_smoothed` are **not** normalized against
anything bearing-specific — they are the raw (or rolling-smoothed) measurement of the current
signal, full stop (`docs/feature_windowing_decision.md` §1–2). A population-level z-score
against pooled training statistics asks exactly the right question for a value like this:
*does this sensor's raw reading look like readings this model has been calibrated against?*

That question is not hypothetical here — it is precisely the question `1st_test`'s raw-RMS
problem shows the model badly needs answered. `docs/model_training_decision.md` §3a measured
this directly:

| Experiment | min raw `rms` | mean raw `rms` | max raw `rms` |
|---|---|---|---|
| `1st_test` | 0.1289 | 0.1618 | 0.5936 |
| `2nd_test` | 0.0015 | 0.1061 | 0.7250 |
| `3rd_test` | 0.0040 | 0.0724 | 0.7588 |

`1st_test`'s *minimum* raw RMS already exceeds both other experiments' *means* — a bearing
whose amplitude scale simply sits in a different range, which §3a shows drives `Normal`
recall on that fold down to 0.074 once a `StandardScaler` fitted on the other two maps every
`1st_test` row into the high-RMS tail. **A population z-score on raw `rms` would have flagged
exactly this bearing as out-of-distribution from its very first file** — file 0's raw RMS is
already 0.1289, inside `1st_test`'s own tight range but far outside `2nd_test`/`3rd_test`'s
typical values. That is the concrete case this monitoring signal exists for: surfacing "this
input doesn't resemble training" *before* it turns into a silently-wrong prediction, which is
a strictly earlier and more general warning than waiting for the label itself to look
suspicious.

### Why `rms_ratio` does not get the same treatment

`rms_ratio` is structurally different from the other four in one specific way:
`docs/feature_windowing_decision.md` §1 and `src/serving/state.py` both define it as a ratio
to *that bearing's own* first-50-file baseline mean — it is already normalized, per bearing,
before it is ever compared to anything else. Layering a second, population-level
normalization on top of an already bearing-relative quantity raises the same problem
`docs/frequency_domain_decision.md` §6a diagnosed for pooling a per-experiment-relative
quantity without removing between-group offsets: pooled `rms_ratio` values are dominated by
each experiment's own `critical_multiple` (1.932 / 2.866 / 3.049, `docs/eda_findings.md` §3),
which varies by 58% between the smallest and largest — not because bearings' sensors read
differently, but because each bearing's own eventual failure severity differs. A population
z-score on `rms_ratio` would therefore mostly measure *which experiment a bearing resembles*,
not *whether its sensor looks anomalous* — the same between-group-variance conflation
`docs/frequency_domain_decision.md` had to explicitly correct for (its pooled-vs.
within-experiment z-scoring comparison, §6a) when a raw pooled F-statistic was found to be
inflated by exactly this effect.

More importantly, an elevated `rms_ratio` is not a surprising, out-of-band reading for this
system — it is the single strongest signal the classifier already thresholds on
(`docs/model_training_decision.md` §2's ablation: removing it collapses macro-F1 on
`2nd_test`/`3rd_test` from 0.936/0.945 to 0.889/0.747). A bearing with `rms_ratio` = 3.5 isn't
behaving like a broken sensor; it is behaving like a bearing whose degradation the model was
built to catch, and the served model already surfaces that as `label = "Critical"`. Flagging
it a *second* time as "drift" would duplicate information the dashboard already shows via the
predicted-class distribution (Section 5) rather than adding an independent signal — and per
`docs/model_training_decision.md` §3b, it would do so unreliably for exactly the bearings that
matter most: `1st_test`'s entire `Critical` band (`rms_ratio` ∈ [1.948, 2.869]) sits *below*
the lowest value its own training fold ever called `Critical`, purely because
`critical_multiple` is derived per-experiment. A population `rms_ratio` baseline would
therefore either miss `1st_test`-shaped Critical readings entirely (if calibrated on
`2nd_test`/`3rd_test`'s higher bands) or over-fire on ordinary `2nd_test`/`3rd_test` Degrading
readings (if calibrated low enough to catch `1st_test`) — a documented, structural mismatch
this issue is not positioned to re-solve, on top of not needing to: that is a label-threshold
question already recorded as open in `docs/model_training_decision.md` §3b, and duplicating a
noisy version of it inside the drift signal only borrows its problems into a new place.

**`rms_ratio` is still computed and still shown** — the dashboard's per-bearing view
(Section 5) includes its current value, because it is genuinely useful context for reading
the predicted-class panel next to it. It simply is not part of the boolean/z-score "is this
input drifting" computation the other four features drive.

> **Implemented in Issue #90, and confirmed against the exact case this section predicts.**
> `MONITORED_FEATURES` (`src/serving/drift.py`) is `["rms", "kurtosis", "skewness",
> "skewness_smoothed"]` — `rms_ratio` is never a key `BearingState.observe_drift` receives,
> enforced by a test that asserts passing it raises `KeyError`
> (`tests/test_serving_state_drift.py::test_rms_ratio_key_would_raise_if_ever_passed_to_observe_drift`).
> Measured over real HTTP, replaying all 2,156 real `1st_test` files at full resolution
> (`python -m demo.playback --raw-dir data/raw --experiment 1st_test --interval 0`):
> `rms`'s z-score reached **10.02** (flagged `drifting: true`) — exactly the raw-amplitude
> scale mismatch this section predicts, measured directly rather than assumed. `rms_ratio`
> itself read 2.87 at the same moment, well within the range the classifier already handles
> via its `label`, confirming it carries no independent drift signal here.

## 3. Where the drift computation lives, and when it runs

**Decision: computed inline, per request, inside the existing `/predict` handler — no
background job, no scheduler, no second process. The result is stored in per-bearing state by
extending the existing `BearingState`/`BearingStateStore` (#82), not a second, parallel store.
It is surfaced two ways: one lightweight additive field on `PredictResponse` for the calling
bearing, and a new `GET /monitoring/drift` endpoint for the cross-bearing dashboard view.**

### Why not a periodic background job

A periodic job (an `asyncio` task on a timer, or a thread) is the natural alternative, and it
was rejected for a reason specific to this codebase: `src/serving/state.py`'s own docstring
already declines to add a lock around `BearingState`/`BearingStateStore`, reasoning that doing
so "would suggest a level of concurrency safety the single-process, in-memory design does not
otherwise provide." A background task reading and writing the same per-bearing dicts a request
handler is concurrently mutating reopens exactly that question — it would need the same
safety guarantee the state module deliberately does not build in, for a computation that costs
nothing extra to run inline. There is also no new information a periodic job would have that
the request path doesn't: every feature value the drift check needs is already computed once
per request by `OnlineFeatureExtractor.observe` (#82); a periodic job would either recompute
it (wasted work) or read a cache of it (the same per-bearing state, accessed from two places
instead of one). Computing it synchronously inside the request that already produced the
inputs is strictly simpler and adds no new failure mode.

### Why the state lives in `BearingState`, not a second store

`docs/serving_design.md` §2's whole argument against Redis/a database — "state that fits in a
few kilobytes of process memory and does not need to outlive the process" — applies with equal
force to a second in-memory drift-tracking dict running alongside the first. Two independent
per-`bearing_id` stores can silently fall out of sync (a bearing present in one after a partial
failure, absent from the other), for no benefit over adding a few fields to the structure that
already exists and is already keyed the same way. `BearingState` therefore gains, per bearing:
a short rolling history of recent per-feature z-score flags (reusing `ROLLING_WINDOW = 10` —
the same constant `rms_history`/`skewness_history` already use, rather than introducing a
second arbitrary window size) and a running tally of predicted-class counts (Section 5). Both
are read and written exactly where `rms_history`/`skewness_history` already are, inside the
same `observe()` call, under the same single-worker/no-lock reasoning already documented
there.

**Persistence rule:** a bearing/feature is reported as `"drifting"` when at least 3 of its
last 10 requests had `|z| > 3` for that feature (an `OR` across the four monitored features
for the bearing-level flag). A lone 3σ reading is expected occasionally by chance — Section
1's Chebyshev bound already allows up to ~11% per feature per reading in the worst case —
so a single excursion should not flip a persistent status. Requiring 3 of the last 10 makes
an isolated false positive require the same rare event to recur multiple times inside one
short window (increasingly unlikely if genuinely independent noise), while a real, sustained
shift — the thing "distribution shift" in `docs/PRD.md` §10's own phrasing actually names —
pushes most or all of the next several readings past the threshold and satisfies the rule
within 3 requests of onset, not after the full window fills.

> **Implemented in Issue #90.** `BearingState` gains `drift_history` (one
> `deque(maxlen=ROLLING_WINDOW)` per monitored feature, imported from the same
> `src.features.extraction.ROLLING_WINDOW` constant `rms_history`/`skewness_history`
> already use — no second window size), `latest_z_scores`, and `predicted_class_counts`,
> all updated inline inside `observe_drift`/`record_prediction`, called from
> `compute_online_features`/the `/predict` handler respectively — no lock, no background
> task, added anywhere. The exact 3-of-10 boundary is pinned in both directions
> (`tests/test_serving_state_drift.py::test_the_third_extreme_reading_in_the_window_flips_drifting`,
> `..._reverts_to_nominal_once_extreme_readings_age_out_of_the_window`), matching this
> project's convention of testing state-machine transitions at the exact point rather than
> "eventually true" (Issue #82's `baseline_status` boundary tests, in
> `tests/test_serving_features.py`, are the precedent).

### Why two surfaces, not one

`docs/serving_design.md`'s M5-readiness note already flagged that a drift field is a natural,
additive extension to `PredictResponse` — and it is, for the one thing a `/predict` caller can
see: *this bearing's own* current status. `PredictResponse` therefore gains one field,
`drift_status: "nominal" | "drifting"`, alongside the existing `label`/`baseline_status`/
`model_notes` — no existing field changes, matching the enum-constant convention
`WARMING_UP`/`STABLE` already establish in `src/serving/state.py`.

But `docs/serving_design.md` §1/§5 fix `/predict` as one bearing, one response, no batch —
which means a single `/predict` call has no way to represent "all bearings currently being
tracked," and a dashboard's natural view (Section 5) is exactly that: every `bearing_id` the
server currently holds state for, at once. Extending `PredictResponse` cannot serve that need
without violating the no-batch contract `docs/serving_design.md` §5 already fixed. A separate
`GET /monitoring/drift` — a plain read of `BearingStateStore`'s current contents, no
computation of its own beyond what `/predict` already wrote — is therefore not a preferred
alternative to extending the schema; it is required by a constraint extending the schema
cannot satisfy. Illustrative shape (not an API spec, matching `docs/serving_design.md` §1's own
convention for unimplemented contracts):

```
GET /monitoring/drift
{
  "bearings": {
    "1st_test-demo": {
      "file_count": 133,
      "baseline_status": "stable",
      "drift_status": "nominal",
      "features": {
        "rms":               {"z": 1.2, "drifting": false},
        "kurtosis":          {"z": 4.1, "drifting": true},
        "skewness":          {"z": 0.3, "drifting": false},
        "skewness_smoothed": {"z": 0.5, "drifting": false}
      },
      "rms_ratio_latest": 2.31,
      "predicted_class_counts": {"Normal": 100, "Degrading": 30, "Critical": 3}
    }
  }
}
```

> **Implemented in Issue #90**, matching this shape exactly. `PredictResponse` gains
> `drift_status: "nominal" | "drifting"` (`src/serving/api.py`); `GET /monitoring/drift`
> returns `{"bearings": {...}}`, one entry per `bearing_id` `BearingStateStore` currently
> holds, built from a plain loop over the store with no computation beyond what `/predict`
> already wrote (`tests/test_monitoring_endpoint.py` pins the shape key-by-key, and
> separately asserts `"rms_ratio"` never appears as a key inside `"features"`).

The per-feature (mean, std) baseline pairs themselves are computed once, offline, from
`training_dataset.parquet` — the implementation issue commits them to a small file alongside
`models/`, following the precedent `docs/serving_model_artifact.md` already set for
`models/serving_model_manifest.json`: eight floats (four features × mean/std) is well under
the "large binary" concern `.gitignore`'s model-artifact rule targets, so it is committed, not
regenerated at container startup.

## 4. Dashboard technology and scope

**Decision: a single static HTML page with inline vanilla JavaScript (no framework, no build
step, no charting library), served by the existing FastAPI app at `GET /monitoring`, polling
`GET /monitoring/drift` on a short interval. No Prometheus, no Grafana, no Streamlit.**

### Reconciling this against `docs/PRD.md` §8/§9's proposed Prometheus/Grafana shape

`docs/PRD.md` §8 names Prometheus + Grafana as the *proposed* MVP monitoring architecture, and
§9's tech stack lists "Prometheus + Grafana (**or a lighter alternative**)." §10's acceptance
criterion is more specific in its literal wording ("`/metrics` endpoint... scraped by a local
Prometheus instance; a Grafana dashboard (**or equivalent**)") but still explicitly hedges the
dashboard half. This document exercises that hedge deliberately, for reasons specific to this
project's stated scale rather than a general dislike of the tools:

- **Moving-part cost.** Prometheus + Grafana is two additional long-running services in
  `docker-compose.yml`, each needing its own image pull, health check, and configuration
  (a scrape-target config for Prometheus, a datasource + dashboard JSON for Grafana) before a
  reviewer sees a single number. `docs/serving_design.md` §2 already rejected Redis/a database
  for per-bearing state on precisely this basis — "adds a second moving part... for state that
  fits in a few kilobytes" — and two full monitoring services for a demo whose actual payload
  is four z-scores and a label histogram is the identical mistake at a different layer. It
  would also work directly against `docs/PRD.md` §10's own measured fresh-clone criterion
  (3s clone + 53s to a healthy API, Issue #86): adding two more images to pull and start is a
  cost this project has already gone out of its way to avoid elsewhere.
- **Dependency cost.** `prometheus_client` would be a new pinned dependency for a text-format
  exposition format that only matters if something is actually scraping it — and nothing
  in this project's demo scope runs a scrape loop unless Prometheus itself is also added.
  Streamlit is heavier still: a full server process and Python web framework, for a page whose
  content is one JSON object polled on a timer. `docs/skewness_crestfactor_decision.md`'s
  Method note rejected a heavier evaluation tool for exactly this shape of reason — an
  existing, already-available mechanism was preferred over new infrastructure "just for this
  one" need — and the same reasoning applies here with more force: FastAPI can already return
  an `HTMLResponse`/serve a static file with zero new packages, so the zero-new-dependency
  option is not a compromise, it is strictly more capable of meeting this issue's own "no new
  dependencies" instruction than either alternative.
- **What is actually needed vs. what those tools are for.** Grafana's value is dashboards
  that outlive a single process, are shared across a team, and compose many services' metrics
  — none of which applies to a single-reviewer, single-container, `docker compose up` demo
  whose entire monitoring surface is "one page showing whether the last few readings looked
  like training data." A static page reading one endpoint is not a scaled-down Grafana; it is
  the right-sized tool for a fundamentally smaller job.

> **Implemented in Issue #90.** `src/serving/static/monitoring.html` -- one file, inline
> CSS/JS, no `<script src=...>` or `<link>` to anything external
> (`tests/test_monitoring_endpoint.py::test_monitoring_page_makes_no_external_requests`
> asserts neither `"http://"` nor `"https://"` appears anywhere in it). Served by
> `GET /monitoring` via a plain `FileResponse` -- no new dependency (`fastapi.responses` is
> already part of the pinned `fastapi` package), no change to `Dockerfile`/
> `docker-compose.yml` (`COPY src/ ./src/` already carries the new file; no new port, no
> new service).

This is a documented, deliberate deviation from §8/§9's *proposed* shape, not from a
*committed* one — matching the project's existing convention of naming and justifying
deviations rather than silently drifting from them (`docs/CONTRIBUTING.md`'s own
`CONTRIBUTING.md`-location deviation note is the precedent for how this project records this
kind of choice). Whoever closes out M5 should update `docs/PRD.md` §10's checkbox wording to
say so explicitly — out of scope for this design-only issue, which touches
`docs/monitoring_design.md` alone, but flagged here so it is not lost.

### Why a single static file rather than a small app of its own

The page's entire job is: poll one JSON endpoint, render one table (one row per tracked
`bearing_id`, one cell per monitored feature's status, plus the predicted-class counts and
`rms_ratio_latest`). That does not need component state, routing, or a build step — a `fetch`
call on a `setInterval`, writing into a plain HTML table, is the entire mechanism. FastAPI
serves it directly (`StaticFiles`/a plain route returning the file's contents), so "the
dashboard" adds no new service to `docker-compose.yml` at all — it is the same container,
same port, same process already serving `/predict`.

## 5. What "visible on a dashboard" concretely means for the demo

**Decision: yes, it updates live during `demo/playback.py`'s replay, because it reads the same
state `/predict` already updates on every call — a reviewer opens one browser tab, no
additional command.**

Concretely, after `docker compose up` (already the documented Quick Start, unchanged by this
issue): a reviewer opens `http://localhost:8000/monitoring` in a browser. That page polls
`GET /monitoring/drift` roughly every second — matched to `docker-compose.yml`'s existing
0.5s playback interval so the page visibly updates between snapshots rather than idling
between infrequent polls, mirroring how the interval choice there was already picked relative
to playback cadence rather than to the underlying dataset's real ~10-minute spacing
(`docs/PRD.md` §8's compressed-timescale framing, restated in `demo/playback.py`'s own
`DEFAULT_INTERVAL_S` comment). As `demo/playback.py` sends snapshots for its one `bearing_id`,
each `/predict` call updates that bearing's row: `file_count` climbs, `baseline_status` flips
`warming_up` → `stable` at file 50 exactly as the existing README walkthrough describes,
`predicted_class_counts` accumulates, and each monitored feature's `z`/`drifting` value
reflects the latest reading. Nothing about `demo/playback.py` itself changes — it is a
`/predict` client and stays one; the dashboard is a second, independent consumer of state the
existing playback already causes to change.

A reviewer who wants to *see* a drift flag fire (rather than take the mechanism on faith) can
do so with the tools this repo already documents: `demo/playback.py --raw-dir` accepts a
different experiment (README's existing "running the demo against the full dataset" section);
feeding a signal whose raw amplitude sits far outside the pooled training range — the
`1st_test` scale problem itself is a real, already-measured example, Section 2 above — would
be expected to flip `rms`'s `drifting` flag within the 3-of-10-request persistence window.
Constructing and documenting that specific walkthrough is implementation-issue work, not a
question this design leaves open — the mechanism that would produce it is already fully
specified above.

> **Implemented and verified in Issue #90 — this walkthrough was actually run, not just
> anticipated.** `python -m src.serving.main` (real process, real port, no `TestClient`) +
> `python -m demo.playback --raw-dir data/raw --experiment 1st_test --interval 0` against
> it: `baseline_status` flipped `warming_up` → `stable` at exactly request 50;
> `predicted_class_counts` accumulated to `{"Normal": 1834, "Degrading": 303, "Critical":
> 19}`, matching the playback client's own tally exactly; and `rms`'s `drifting` flag
> reached `true` (`z ≈ 10.02`) well before the replay finished. `GET /monitoring` was loaded
> in a real headless-Chromium browser session (not curl) and screenshotted, showing three
> tracked bearings with the drifting ones visibly highlighted. `docker compose up` itself
> could not be exercised in the environment this was implemented in — Docker Hub image
> pulls were network-blocked there — so this ran the identical `src/serving`/`demo`
> application code directly instead; see the Issue #90 PR description for the full account.

## 6. Non-goals

Stated explicitly, per this issue's own request and the project's general scope-discipline
convention (`docs/PRD.md` §4, `CLAUDE.md`'s "Scope discipline" section):

- **Alerting or paging.** The dashboard is a passive display a reviewer looks at; nothing here
  emails, pages, or otherwise pushes a notification when `drift_status` flips. `docs/PRD.md`
  §8 already scopes monitoring to "one scrape target, a handful of panels," and alerting rules
  are named there as explicitly out of the MVP surface.
- **Concept drift / accuracy monitoring.** This signal measures **input** distribution shift
  only — whether incoming feature values resemble training data. It cannot measure whether the
  model's *predictions* are still accurate, because that requires ground-truth labels arriving
  after the fact, which a real deployment (and this demo) never receives. This is a real,
  named limitation, not an oversight: an input-drift signal and a model-accuracy signal answer
  different questions, and only the first is possible without a labeling pipeline this project
  does not have and is not building here.
- **Multi-model comparison.** One served model (#80), one drift computation. No champion/
  challenger, no A/B framing — there is exactly one pooled model in production per
  `docs/serving_design.md` §4, unchanged by this issue.
- **Historical trend storage beyond the demo's own runtime.** Per-bearing rolling history
  (Section 3) lives in the same in-memory `BearingStateStore` `docs/serving_design.md` §5
  already declines to make durable across restarts. A restart loses drift history exactly as
  it already loses rolling feature history — consistent, not a new gap this issue introduces.
- **A literal Prometheus/Grafana stack**, per Section 4's decision — recorded again here
  because it is the one non-goal this document actively chooses against the PRD's originally
  *proposed* (not committed) shape, so it should not be mistaken for an omission.
- **Authentication or access control** on `/monitoring` or `/monitoring/drift`. Same posture
  as `/predict` and `/health` (`docs/serving_design.md` §5): a local-container demo for a
  single reviewer, not a multi-user service.
- **Root-cause attribution.** The dashboard reports *that* a feature looks anomalous
  (`drifting: true/false` plus its `z`), not *why* — no per-feature explanation, no comparison
  to which training experiment it resembles least. That is a reasonable follow-up, not part of
  this design.
- **Triggering retraining.** Nothing here feeds back into training. Matches `docs/PRD.md` §4's
  platform-level exclusion of automated retraining and `docs/serving_design.md` §5's identical
  non-goal for the model artifact itself.
- **Multi-worker/horizontally-scaled monitoring.** Inherits `docs/serving_design.md` §2's
  single-worker constraint directly, since drift state lives inside the same
  `BearingStateStore` that constraint already governs. Nothing new to decide here.

## Reconciling with `docs/PRD.md` and `docs/serving_design.md`

- **`docs/PRD.md` §10 — "at least one drift/health signal... visible on a dashboard, not just
  logged":** met by two complementary signals on one page — the four-feature input-drift
  check (Sections 1–2) and the predicted-class distribution `docs/PRD.md` §8 names as its own
  example signal (Section 3's `predicted_class_counts`, surfaced in Section 5) — both updating
  live from state `/predict` already maintains, not from a separate logging pass a human would
  have to go read.
- **§8/§9's proposed Prometheus/Grafana architecture:** deliberately not built, per Section 4's
  reasoning; §10's own "(or equivalent)" wording is the hedge this document exercises, and the
  deviation is named rather than silent, matching this project's established convention for
  documenting scope deviations.
- **`docs/serving_design.md` §2 — single-worker, in-memory state:** unchanged and directly
  reused. Drift tracking adds fields to the existing `BearingState`/`BearingStateStore`, runs
  inline inside the same request handler, and introduces no new process, thread, lock, or
  external store — the same constraints that already govern `rms_history`/`skewness_history`
  now also govern the drift rolling-window fields, for the same stated reasons.
- **`docs/serving_design.md` §5 — no batch endpoint:** respected by design. `PredictResponse`
  gains exactly one additive field (`drift_status`) scoped to the single bearing in that
  request; the cross-bearing view that a batch-shaped response would otherwise need to provide
  is served instead by a separate `GET /monitoring/drift` read, not by bending `/predict`'s
  contract.
- **Response schema:** `PredictResponse` changes additively only — `label`, `baseline_status`,
  and `model_notes` are unchanged; `drift_status` is a new field, matching
  `docs/serving_design.md`'s own M5-readiness observation that this is exactly the kind of
  change `/predict`'s schema was left open to.
