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
  estimation is a stretch goal (Section 10), not part of MVP scope.
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

RUL regression remains a documented stretch goal (Section 10) — not committed scope, and not
revisited before the health-state classifier is working end-to-end.

## 7. Success Metrics

| Metric | Target |
|---|---|
| Model performance | Reasonable classification accuracy/F1 on held-out run(s) — exact number TBD once EDA is done; report honestly rather than chasing a number |
| Serving latency | <500ms for single-window inference, local container, no batch queueing — not framed as a production SLA |
| Reproducibility | Fresh clone → running demo in <15 min following README |
| Monitoring | At least one drift/health signal (e.g., input feature distribution shift) visible on a dashboard, not just logged |
| Documentation | README explains architecture, scoping decisions, and what was deliberately left out |

## 8. Proposed Architecture (MVP)

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

## 10. Acceptance Criteria (Definition of Done for MVP)

The MVP is considered done when all of the following hold, verifiable by a reviewer with no
prior context:

- [ ] Fresh `git clone` → running demo (`docker compose up` or documented equivalent) in
      under 15 minutes, following only the README.
- [ ] `/predict` endpoint accepts a feature window and returns a health-state prediction
      (Normal / Degrading / Critical) with a response time under 500ms (single-window,
      no batch queueing — see Section 7).
- [ ] At least one MLflow run is visible, showing the trained model, its metrics, and its
      parameters — not just a pickled file with no lineage.
- [ ] `/metrics` endpoint is live and scraped by a local Prometheus instance; a Grafana
      dashboard (or equivalent) shows at least one real signal (predicted-class distribution
      or input feature stats) updating as the demo script plays back a bearing's run-to-failure
      history.
- [ ] README documents: the architecture, the classification-vs-RUL decision and why, what was
      deliberately left out of scope (Section 4) and why, and the demo playback timescale
      caveat (Section 8).
- [ ] Repo structure matches Section 10a below (or documents any deviation).

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

## 12. Risks / Open Questions

- Dataset only covers **one operating condition** — worth stating explicitly as a limitation
  (a real plant would need condition-normalization); this is a good thing to name in the
  README, since acknowledging limitations reads as engineering maturity.
- ~~Deciding classification vs. RUL early avoids scope creep mid-project~~ — **resolved**: locked
  to classification in Section 6, before Milestone 1 begins.
- Keep the "why not Kubernetes / why not Machina yet" reasoning documented, since a reviewer
  familiar with over-engineered portfolio projects will specifically notice restraint.

## 13. Out-of-scope, Future Phase Ideas (not commitments)

- Kubernetes + KServe/Seldon model serving
- CI/CT/CD with automated retraining triggers
- Machina agent layer for technician Q&A over CMMS-style work orders, grounded in this
  platform's predictions
- Cloud deployment (AWS) with Terraform/CDK, tying back into the IaC skill shown in
  `heating-monitor-iot`
