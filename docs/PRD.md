# PRD: Predictive Maintenance AI Platform (MVP)

**Owner:** Barna Peter
**Status:** v1.1 — finalized for MVP build
**Last updated:** 2026-07-24

---

## 1. Summary

A portfolio project demonstrating AI Platform Engineering competency: a predictive
maintenance system that ingests industrial bearing vibration data, serves a failure-prediction
model through a proper serving layer, and monitors that model in production — as opposed to a
one-off notebook or script. The goal is to show the full "model to platform" path on a scope
small enough to actually finish.

This is a **new, standalone repository**, not a refactor of `heating-monitor-iot`. The heating
monitor project remains as a separate portfolio piece demonstrating edge-to-cloud IoT and DevOps
skills; this project is scoped specifically to demonstrate ML/AI platform skills on data that
actually supports prediction.

## 2. Problem Statement

Unplanned failure of rotating machinery (pumps, motors, compressors) is expensive and hard to
predict from simple threshold rules. Vibration signals contain early signatures of bearing
degradation (wear, fatigue cracking) well before failure, but turning raw signal data into a
reliable, monitored, production-style prediction service requires more than a trained model —
it requires a platform: reproducible data pipeline, a served model, and observability into
whether that model is still trustworthy.

## 3. Goals

- G1: Predict bearing health state (Normal / Degrading / Critical) from vibration signal data
  with a documented, reasonable level of accuracy. **Decided (locked, not revisited before
  Milestone 4):** classification, not RUL regression — see Section 6 for rationale. RUL
  estimation is a stretch goal (Section 11), not part of MVP scope.
- G2: Serve the model through a real serving layer (API/container), not a notebook.
- G3: Monitor the deployed model — track prediction distribution / basic drift signals over
  time, not just training-time metrics.
- G4: Package the whole thing so a reviewer can run it locally in under 15 minutes
  (`docker compose up` or equivalent).
- G5: Document engineering decisions (why this model, why this serving approach) so the project
  reads as an engineering artifact, not a Kaggle notebook with a Docker wrapper.

## 4. Non-Goals (explicitly out of scope for MVP)

These are common "AI platform" buzzword items that this MVP deliberately **excludes** to avoid
the trap of bolting on complexity that isn't earned by the problem size:

- ❌ Kubernetes orchestration — a single model, single service does not justify K8s. Note this
  explicitly in the README as a scoping decision, not an omission.
- ❌ Full CI/CT/CD with automated retraining — no continuous training loop in v1.
- ❌ Machina / agent layer integration — evaluated as a **future phase**, not MVP. Reasoning:
  the MVP's job is to prove the ML + serving + monitoring core is real and earned; adding an
  agent framework now would obscure what was actually built vs. configured.
- ❌ Multi-model / multi-tenant serving.
- ❌ Real-time streaming ingestion (OPC-UA/MQTT) — dataset is file-based and historical.
- ❌ Cost/FinOps dashboard.

Anything cut here is a candidate for a documented "Phase 2" roadmap section, not a promise.

## 5. Target Audience / Use Case

Primary "user" of this project is a technical reviewer (recruiter, hiring manager, senior
engineer) evaluating it as evidence of AI platform engineering skill. Secondary framing: a
plant engineer who wants an early warning before a bearing fails.

## 6. Dataset

**NASA IMS (Rexnord) Bearing Dataset** — Center for Intelligent Maintenance Systems, University
of Cincinnati, distributed via the NASA Prognostics Data Repository (also mirrored on Kaggle).

Key characteristics to design around:
- Three independent run-to-failure experiments, four bearings each, constant operating
  condition (2000 RPM, 6000 lbs radial load).
- Each file = a 1-second vibration snapshot, 20,480 points, sampled at 20 kHz, recorded at
  regular intervals (~every 10 minutes) across the bearing's life until failure.
- Set 1 has 8 channels (2 per bearing); Sets 2 and 3 have 4 channels (1 per bearing).
- Ground truth: each experiment ends in a documented failure mode (e.g., outer race failure)
  for a specific bearing — this gives a real "time-to-failure" label, which is what makes RUL
  estimation or degradation-stage classification meaningful (unlike the heating monitor's
  binary on/off signal).

**Decision (locked): health-state classification (Normal / Degrading / Critical), not RUL
regression, for the MVP.**

Rationale:
- The final phase of a bearing's life shows exponential vibration growth, which makes direct
  RUL regression numerically unstable and easy to sink disproportionate time into tuning
  without a corresponding payoff for what the project is trying to demonstrate.
- Three-class health state maps directly to a business-legible signal (green/yellow/red on a
  plant floor), which is easier to justify and explain than a regression number.
- Classification produces cleaner drift metrics for the monitoring layer (e.g., shift in the
  distribution of predicted classes over time is simpler to reason about than shift in a
  continuous RUL estimate).

RUL regression remains a documented stretch goal (Section 11) — not committed scope, and not
revisited before the health-state classifier is working end-to-end.

## 7. Success Metrics

| Metric | Target |
|---|---|
| Model performance | Evaluated via leave-one-experiment-out (LOEO) — see `docs/evaluation_protocol.md` for the full protocol, the label-leakage rationale for why LOEO rather than a random/stratified split, and the committed primary metric (per-class recall/precision, headlined by `Critical`-class recall); report honestly rather than chasing a number. **Measured (M3 baseline, Issue #72): `Critical` recall 0.059 / 0.913 / 1.000 across the three folds — see the note below, which is part of this row, not a caveat on it** |
| Serving latency | <500ms for single-window inference, local container, no batch queueing — not framed as a production SLA |
| Reproducibility | Fresh clone → running demo in <15 min following README |
| Monitoring | At least one drift/health signal (e.g., input feature distribution shift) visible on a dashboard, not just logged |
| Documentation | README explains architecture, scoping decisions, and what was deliberately left out |

> **The M3 baseline works on two of three bearings and fails on the third. Recorded here
> rather than in the decision doc alone, so the headline row above cannot be quoted without
> it** (Issue #72, full evidence in `docs/model_training_decision.md`).
>
> | Fold | `Critical` recall | `Critical` precision | `Normal` recall | Macro-F1 |
> |---|---|---|---|---|
> | `2nd_test` | 0.913 | 0.750 | 1.000 | 0.936 |
> | `3rd_test` | 1.000 | 0.859 | 0.999 | 0.945 |
> | `1st_test` | **0.059** | 1.000 *(one prediction)* | **0.074** | **0.152** |
>
> **The cross-fold mean (`Critical` recall 0.657) describes no fold and is not this
> project's headline number.** `docs/evaluation_protocol.md` §5 requires a sharply
> divergent fold to be stated rather than averaged away; this is that case.
>
> The `1st_test` failure is diagnosed, not mysterious, and has two independent causes:
> raw RMS amplitude does not transfer between bearings (`1st_test`'s *minimum* raw RMS
> exceeds both other experiments' *means*), and all 17 of its `Critical` rows lie below the
> lowest `rms_ratio` its training fold ever labelled `Critical`, making them unreachable by
> any boundary learned from the other two experiments. The second cause is a property of the
> per-experiment `critical_multiple` derivation (§6, `docs/eda_findings.md` §3), so it
> constrains any model trained on these labels — not just this baseline. Three leakage-safe
> and one protocol-violating remedy were measured; none resolves it, and the reasoning is in
> `docs/model_training_decision.md` §4.
>
> Two further honest limits on the numbers above: the `rms_ratio` ablation required by
> Issue #67 Task 3 *raises* mean `Critical` recall (0.657 → 0.892) while collapsing
> precision (0.870 → 0.603) — the gain is over-alarming, not capability, and on `1st_test`
> that configuration predicts `Normal` for zero of 1,906 truly-`Normal` rows. And the two
> folds that work are both outer-race failures while the failing one is the only inner-race
> experiment, which `docs/evaluation_protocol.md` §6 already notes cannot be separated from
> "fails on this particular bearing" at *n* = 1.

## 8. Proposed Architecture (MVP)

> **This section is the original, pre-build proposal — kept as written, not edited to match
> what got built.** For the as-built pipeline, see README.md's "Architecture" section (a
> Mermaid diagram, added in Issue #92/M6-Packaging), which also lists every point where the
> implementation ended up differing from what's proposed below: the monitoring stack (static
> page + JSON endpoint, not Prometheus/Grafana — Section 9 and `docs/monitoring_design.md`
> Section 4 already flagged this hedge), two separate training paths instead of one generic
> "Training pipeline" box, the `/predict` contract's raw-signal-in decision
> (`docs/serving_design.md` Section 1), and MLflow's SQLite tracking store
> (`docs/mlflow_tracking.md`). Reconciling this section by rewriting it would erase the record
> of what was originally proposed versus what the constraints of an actual implementation
> changed — the same "document deviations, don't silently drift" convention this PRD already
> follows for Section 10a's repo-structure target.

```
[Raw vibration files] 
      -> [Preprocessing / feature extraction pipeline] (offline, versioned)
      -> [Training pipeline] -> [Model artifact + experiment tracking]
      -> [Serving layer: FastAPI + Docker, model loaded at startup]
      -> [Client / demo script feeds windows in, simulating a live feed]
      -> [Monitoring: log predictions + input stats -> simple dashboard]
```

- **Feature extraction:** time-domain + frequency-domain features (RMS, kurtosis, FFT peak
  energy, etc.) per window — this is the "domain knowledge" layer that shows ML maturity beyond
  "throw raw signal at a CNN."
- **Experiment tracking:** MLflow (local/file-backed is fine for MVP) — logs runs, metrics,
  model versions.
- **Serving:** FastAPI service in a container, exposing a `/predict` endpoint.
- **Monitoring:** lightweight, concrete MVP shape — the FastAPI service exposes a `/metrics`
  endpoint via `prometheus_client`, tracking at minimum: predicted-class distribution over
  time, and basic input feature stats (e.g., mean/std of the incoming feature vector) as a
  proxy drift signal. A local Prometheus instance scrapes `/metrics`; Grafana visualizes it.
  No full observability stack (tracing, alerting rules, multi-service correlation) — one
  scrape target, a handful of panels, that's the whole MVP monitoring surface.
- **Demo playback note:** the demo script that feeds windows into `/predict` runs on an
  accelerated, simulated cadence (e.g., one window every ~2s) to make the dashboard visibly
  animate a bearing's degradation. This is explicitly a **compressed demo timescale**, not a
  claim about real sensor sampling rate — the underlying dataset records roughly every 10
  minutes. State this plainly in the README/demo script comments to avoid it being read as a
  real-time capability claim.
- **IaC:** keep it simple for MVP — Docker Compose is enough; Terraform/cloud deployment is a
  Phase 2 item if you want to demonstrate cloud-native deployment later.

## 9. Tech Stack (proposed)

- Python 3.11+, scikit-learn / XGBoost for the baseline model (justify before reaching for
  deep learning — CNN/LSTM only if the simpler model demonstrably underperforms)
- MLflow for experiment tracking
- FastAPI for serving
- Docker + Docker Compose for packaging
- Prometheus + Grafana (or a lighter alternative) for monitoring
- pytest for testing the pipeline and API

**As-built vs. this list:** the "lighter alternative" this section already hedged for
monitoring is what got built — a static HTML/vanilla-JS page + JSON endpoint, no Prometheus,
no Grafana (`docs/monitoring_design.md` Section 4, README's "Architecture" section). MLflow's
tracking store is `mlflow-skinny` + SQLite, not the full `mlflow` package (`docs/mlflow_tracking.md`).
Everything else in this list — Python, scikit-learn, FastAPI, Docker/Compose, pytest — matches
what was actually used.

## 10. Acceptance Criteria (Definition of Done for MVP)

The MVP is considered done when all of the following hold, verifiable by a reviewer with no
prior context:

- [x] Fresh `git clone` → running demo (`docker compose up` or documented equivalent) in
      under 15 minutes, following only the README. **Measured (Issue #86), cold Docker cache
      — base image pulled, every wheel downloaded: 3s to clone (16 MB), 53s to a healthy
      containerised API, first predictions ~2s after that. Roughly one minute against a
      fifteen-minute budget.** The blocker this criterion faced was the dataset, not the
      model: `docs/serving_model_artifact.md` (#80) flagged that committing the model
      cleared only one of two obstacles while `data/*` stayed gitignored. Resolved by
      committing a 6.0 MB slice of real signal (`demo/sample_data/`, see `demo/sample.py`
      for what it is and why it is cut that way) rather than requiring the 1.1 GB download —
      which this repo's own CI measures at 216–269s of download and extraction alone, on a
      datacenter connection, before any of `unrar`/`py7zr`/6.2 GB of disk.
- [x] `/predict` endpoint accepts a feature window and returns a health-state prediction
      (Normal / Degrading / Critical) with a response time under 500ms (single-window,
      no batch queueing — see Section 7). **Measured (Issue #84) over real HTTP with a full
      20,480-point signal: p50 22ms, p95 25ms, max 34ms.** The contract takes the raw signal
      rather than a pre-computed feature window, deliberately — `docs/serving_design.md`
      Section 1 gives the server sole ownership of feature computation so the rolling-window
      and baseline logic cannot drift into a second client-side copy.
- [x] At least one MLflow run is visible, showing the trained model, its metrics, and its
      parameters — not just a pickled file with no lineage. Seven runs across two
      experiments (`m3-imbalance-comparison`, `m3-baseline-ablation`), covering #21's five
      imbalance arms and #72's baseline/ablation configurations, logged to a local SQLite
      store (`mlflow.db`). Verify with `mlflow ui --backend-store-uri sqlite:///mlflow.db`
      or `mlflow.search_runs(...)` — see `docs/mlflow_tracking.md` for the full dependency
      rationale and an exact-match check against the already-published metrics.
- [x] A live monitoring view shows at least one real signal (predicted-class distribution
      and per-feature input-drift status) updating as the demo script plays back a bearing's
      run-to-failure history. **Delivered as `GET /monitoring` (a static HTML/vanilla-JS
      page, no build step) polling a new `GET /monitoring/drift` JSON endpoint (Issue #90),
      not the originally proposed Prometheus + Grafana stack** — `docs/monitoring_design.md`
      Section 4 documents this as a deliberate, named deviation, exercising this criterion's
      own "(or equivalent)" wording: two additional long-running services would cost more in
      moving parts and dependencies than this project's local-container demo scope
      justifies. Measured: replaying the full-resolution `1st_test` experiment
      (`docs/model_training_decision.md` Section 3a's raw-RMS scale problem) visibly flips
      `rms`'s `drifting` flag (`z ≈ 10`) within the documented persistence window, over real
      HTTP, in an actual browser tab.
- [x] README documents: the architecture, the classification-vs-RUL decision and why, what was
      deliberately left out of scope (Section 4) and why, and the demo playback timescale
      caveat (Section 8). **Delivered (Issue #92):** README's new "Architecture" section
      carries a Mermaid diagram of the as-built pipeline plus an explicit "Deviations from
      Section 8/9" list (monitoring shape, the two separate training paths, the `/predict`
      raw-signal contract, MLflow's SQLite store) — not redrawing the original proposal and
      hoping no one compares them. The classification-vs-RUL rationale (Section 6), the
      Section 4 non-goals, and the Section 8 compressed-timescale caveat were already present
      from earlier milestones and were re-verified, not re-written, as part of this issue's
      full read-through.
- [x] Repo structure matches Section 10a below (or documents any deviation). **Verified
      (Issue #92)** against `git ls-files`, file by file: every tracked file has an entry (or
      falls under a summarizing line, e.g. `tests/`) in README's "Repo structure" tree, and
      every `docs/*.md` file is named explicitly in README's "Learn more" section (previously
      several M2 decision docs were only reachable via a wildcard bullet naming issue numbers,
      not filenames — fixed). `LICENSE` (Apache 2.0) was missing from the tree entirely; added.
      The `10a` deviations already noted below (CONTRIBUTING.md's location) remain accurate;
      no new undocumented deviation was found.

## 10a. Repo Structure (target)

```
sentinedge-prognos-platform/
├── README.md
├── docs/
│   └── PRD.md
├── data/                    # gitignored — raw dataset not committed
├── notebooks/               # EDA, exploratory work
├── src/
│   ├── features/            # feature extraction pipeline
│   ├── training/             # training script(s), MLflow logging
│   └── serving/              # FastAPI app, /predict and /metrics
├── demo/
│   └── playback.py           # simulated live-feed demo script
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt / pyproject.toml
├── CONTRIBUTING.md
└── .gitignore
```

This is a target, not a constraint set in stone before Milestone 1 — if EDA reveals a better
shape, update this section and note why.

**Deviation note:** `CONTRIBUTING.md` actually lives at `docs/CONTRIBUTING.md` rather than the
repo root shown above — grouping it with the other project docs is a common convention and has
no functional downside (no tooling in this project looks for a root-level `CONTRIBUTING.md`).

## 11. Milestones

1. **EDA & framing** — explore the dataset, decide classification vs. RUL, define labels.
2. **Feature pipeline** — reproducible script/notebook → versioned feature dataset.
3. **Baseline model** — trained, tracked in MLflow, evaluated honestly.
4. **Serving layer** — containerized API serving the model.
5. **Monitoring layer** — basic drift/health dashboard.
6. **Packaging & docs** — README, architecture diagram, one-command demo.
7. *(Stretch)* RUL regression variant, or a documented Machina integration as a technician-facing
   Q&A layer on top of the finished platform.

**Note:** an informal "M1.5-Housekeeping" stretch ran between Milestone 1 and Milestone 2
(CI pipeline, unit tests, dependency pinning, README expansion, notebook-output policy,
data-versioning note — see the M1.5-Housekeeping milestone in GitHub Issues). It's tracked
in `CLAUDE.md` and GitHub Issues only; this numbered list is intentionally left unrenumbered,
since M1.5 was project hygiene rather than a product milestone.

## 12. Risks / Open Questions

- Dataset only covers **one operating condition** — worth stating explicitly as a limitation
  (a real plant would need condition-normalization); this is a good thing to name in the
  README, since acknowledging limitations reads as engineering maturity.
- ~~Deciding classification vs. RUL early avoids scope creep mid-project~~ — **resolved**: locked
  to classification in Section 6, before Milestone 1 begins.
- Keep the "why not Kubernetes / why not Machina yet" reasoning documented, since a reviewer
  familiar with over-engineered portfolio projects will specifically notice restraint.
- ~~**Data versioning for processed feature caches (future concern, not solved now):** from M2
  onward, feature-cache files built under `data/processed/` will need versioning /
  reproducibility guarantees — if the labeling logic, feature-extraction code, or raw dataset
  changes, a stale cached feature file could silently get reused in modeling without anyone
  noticing. Rough direction to revisit in M2/M3: hash the generating code + raw data version
  into the cache filename/directory, or keep a simple manifest recording which commit/config
  produced each cache~~ (Issue #24) — **resolved** in M2, Issue #41: both halves of that rough
  direction were implemented rather than one. `src/features/versioning.py` hashes the
  generating code (`extraction.py` + `versioning.py`) and the raw dataset (filename+size
  fingerprint per experiment) into a `combined_hash`, and every parquet output is written
  alongside a `data/processed/<name>_features_manifest.json` recording that hash, the
  generation timestamp, the feature columns, and the row count. A consumer can therefore detect
  a stale cache by recomputing the hash — which
  `notebooks/04_feature_pipeline_validation.ipynb` Section 2 does as an executable check on
  every CI run. Full rationale (why filename+size rather than full content hashing, why one
  manifest per experiment, why labels are not in this output): `docs/feature_extraction_versioning.md`.

## 13. Out-of-scope, Future Phase Ideas (not commitments)

- Kubernetes + KServe/Seldon model serving
- CI/CT/CD with automated retraining triggers
- Machina agent layer for technician Q&A over CMMS-style work orders, grounded in this
  platform's predictions
- Cloud deployment (AWS) with Terraform/CDK, tying back into the IaC skill shown in
  `heating-monitor-iot`
