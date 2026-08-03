# SentinEdge Prognos Platform

Predictive maintenance AI platform for industrial bearing vibration data — a served,
monitored ML system, not a notebook demo.

Bearing failures are expensive and hard to predict from raw sensor noise. This project
turns vibration signal data into an early-warning classifier: given a bearing's recent
vibration readings, predict whether it's Normal, Degrading, or heading toward Critical
failure, in time to act on it.

## Status

This repository implements a predictive maintenance MVP:
- Bearing health-state classification (Normal / Degrading / Critical) from vibration signal
  data — chosen over RUL regression for the MVP; see `docs/PRD.md` Section 6 for the
  rationale.
- A served model (FastAPI) with basic monitoring (Prometheus/Grafana), built out milestone
  by milestone per `docs/PRD.md` Section 11.

Kubernetes, multi-model serving, and the Machina agent layer are **not** planned for this
MVP — they are a documented, deliberate Non-Goal (`docs/PRD.md` Section 4), not an omission
or a future-iteration promise.

Current milestone: see `CLAUDE.md`'s "Current milestone" section for the up-to-date status
(kept there rather than duplicated here, so there's a single source of truth).

## Quick start — watch a bearing fail

**No dataset download, no Python environment.** Docker is the only prerequisite.

```bash
git clone https://github.com/barnapet/sentinedge-prognos-platform.git
cd sentinedge-prognos-platform
docker compose up
```

That starts the serving API and replays a real bearing's run-to-failure history against it,
one snapshot per request. Measured end-to-end on a cold Docker cache (base image pulled,
every wheel downloaded): **3s to clone, 53s to a healthy API, first predictions ~2s later**
— against `docs/PRD.md` Section 10's 15-minute budget.

Real output from that run (abridged — 197 requests in total):

```
[   1/197] 2004.02.12.10.32.39  file=0     predicted=Normal     baseline=warming_up   true=Normal

  model_notes: Trained on all 3 dataset experiments (1st_test/2nd_test/3rd_test) pooled. LOEO
  evaluation found this model class does not reliably detect the Critical health state on
  impulsive, inner-race degradation signatures resembling the 1st_test bearing (Critical recall
  0.059 when that experiment was held out) — see docs/model_training_decision.md. Reliable on
  the two outer-race, amplitude-driven failure modes evaluated (Critical recall 0.913 / 1.000).

[  49/197] 2004.02.14.02.32.39  file=240   predicted=Normal     baseline=warming_up   true=Normal
[  50/197] 2004.02.14.03.22.39  file=245   predicted=Normal     baseline=stable       true=Normal
...
[ 133/197] 2004.02.17.00.32.39  file=660   predicted=Degrading  baseline=stable       true=Degrading
...
[ 197/197] 2004.02.19.05.52.39  file=980   predicted=Critical   baseline=stable       true=Critical

==============================================================================
Replayed 197 snapshots.  Predicted: {'Normal': 132, 'Degrading': 59, 'Critical': 6}
Agreement with committed ground-truth labels: 194/197 (98.5%)
baseline_status: warming_up for requests 1-49, stable from request 50 onward
```

Three things worth watching for, because each is a documented design decision doing its job:

- **`baseline_status` flips from `warming_up` to `stable` at request 50.** Each bearing's
  `rms_ratio` baseline is the mean of its first 50 snapshots, which the server cannot know
  until it has seen them. Rather than refusing to answer, it scores against an expanding
  baseline and says so (`docs/serving_design.md` Section 3).
- **`model_notes` says the model is unreliable on one of the three bearings, on every single
  response.** That is not an error path — the model genuinely fails on `1st_test`-style
  impulsive failures (`Critical` recall 0.059 when that experiment is held out), and
  `docs/serving_design.md` Section 4 decided to disclose it unconditionally rather than
  quietly.
- **`true=` is the committed ground-truth label, shown by the client for comparison.** It is
  never sent to the server — the payload is a raw signal and a `bearing_id`, nothing else.
  The three mismatches in the run above are all one-class-adjacent boundary calls.

> **Playback runs on a compressed, simulated timescale.** The snapshots being replayed were
> recorded roughly 10 minutes apart across a week of real bearing life; the demo sends one
> every 0.5s so the degradation is watchable. This is a demo cadence, not a claim about
> sensor sampling rate or real-time capability (`docs/PRD.md` Section 8).

The API stays up after the replay finishes, so you can talk to it directly:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"bearing_id": "demo", "signal": [0.1, -0.1, 0.2, -0.2]}'
```

Stop it with `Ctrl+C`, then `docker compose down`.

### Why this works without the 6.2 GB dataset

`demo/sample_data/` holds a **committed 6.0 MB slice of real signal**: every 5th snapshot of
the `2nd_test` experiment, tracked channel only, 197 of its 984 files. Each snapshot is the
complete, unmodified 20,480-point recording — only *which* files are present is reduced, never
their contents. That is what makes `docs/PRD.md` Section 10's "fresh clone → running demo"
criterion actually reachable: acquiring the real dataset instead costs **216–269s of download
and extraction alone** (measured, from this repo's own CI runs on a datacenter connection),
plus `unrar`/`py7zr` and 6.2 GB of disk.

`demo/sample.py` documents the full reasoning, including why there is deliberately **no**
`1st_test` sample: decimating that experiment would make the model look like it fails on it for
the wrong reason (measured — a 1-in-10 replay predicts zero `Critical`, while a full-resolution
one predicts them correctly), manufacturing a false demonstration of a real, separately
documented limitation.

### Running the demo against the full dataset

If you have fetched the real dataset (`data/README.md`), `demo/playback.py` replays any
experiment at full resolution — this is how the `1st_test` numbers quoted above were measured:

```bash
docker compose up -d api                       # API only
pip install numpy                              # the client's only dependency beyond stdlib
python -m demo.playback --raw-dir data/raw --experiment 1st_test --interval 0
```

Worth knowing before you run that one: **the served model predicts `1st_test`'s Critical
region correctly** (16 of its 17 `Critical` files, measured over real HTTP), which can look
like it contradicts the `model_notes` disclosure. It does not. The served model is trained on
all three experiments pooled (`docs/serving_design.md` Section 4), so on `1st_test` it is
recalling data it was fitted on — that is in-sample fit, not evidence it would generalize.
The 0.059 figure in the disclosure is what happens when `1st_test` is *held out* and the model
must handle an impulsive failure mode it has never seen, which is the situation a genuinely
new bearing would present. Both numbers are real; they measure different things.

## Setup

Only needed for development — running the notebooks, the test suite, or the training
pipeline. The demo above needs none of it.

```bash
# Clone and enter the repo
git clone https://github.com/barnapet/sentinedge-prognos-platform.git
cd sentinedge-prognos-platform

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Getting the raw dataset

The raw NASA IMS bearing dataset is not committed to the repo — it's gitignored under
`data/raw/`. Budget roughly **1.1 GB to download** (the compressed archive) and **~6.2 GB on
disk** once extracted (9,464 snapshot files across the three experiments). See
`data/README.md` for the download source and step-by-step extraction instructions (the archive
is nested 7z + RAR, with a couple of undocumented-upstream quirks that `data/README.md`
covers).

### Running the notebooks

```bash
jupyter notebook notebooks/
```

Run in order: `01_vibration_signal_evolution.ipynb` → `02_health_state_labeling.ipynb` →
`03_feature_candidate_screening.ipynb` → `04_feature_pipeline_validation.ipynb`. The first
three (M1-EDA) each depend on outputs cached by the previous one under `data/processed/`; the
fourth (M2) validates the `src/features/` pipeline against those M1 findings and builds its own
parquet outputs by calling `src/features/build_dataset.py` directly.

### Running the tests

```bash
pytest tests/ -v
```

Every pull request also runs this automatically, along with a full notebook execution
check — see `.github/workflows/notebook-ci.yml`.

## Repo structure

```
.
├── README.md              this file
├── CLAUDE.md               project context and conventions for AI-assisted work
├── Dockerfile              serving container: single-worker by construction (M4, #86)
├── docker-compose.yml      `docker compose up` -> API + demo playback (M4, #86)
├── requirements.txt        pinned Python dependencies
├── requirements-serving.txt  the serving subset, same pins — what goes in the image
├── data/
│   ├── README.md           dataset source, download, and extraction instructions
│   └── raw/, processed/    gitignored — populated locally, see data/README.md
├── docs/
│   ├── PRD.md              product requirements: goals, non-goals, architecture, milestones
│   ├── eda_findings.md      EDA results and feature-extraction candidates from M1
│   ├── CONTRIBUTING.md      commit conventions, branch/PR workflow
│   ├── *_decision.md        M2/M3 decision notes: windowing (#40), label hysteresis (#20),
│   │                        feature extraction + versioning (#41), skewness/crest factor
│   │                        (#23), frequency domain (#22), class imbalance (#21), M3
│   │                        baseline + rms_ratio ablation (#72)
│   ├── evaluation_protocol.md       M3 LOEO split + committed metrics, fixed before
│   │                                training (#69)
│   ├── training_dataset_versioning.md  feature-parquet + label join, its own hash
│   │                                    chain (#67)
│   ├── uncertainty_quantification.md   intervals/corrections on M1-M2 point estimates
│   │                                    (#63)
│   ├── mlflow_tracking.md           MLflow dependency choice + how to inspect the M3
│   │                                 LOEO runs (#74)
│   ├── serving_design.md            M4 /predict contract, state ownership, cold start,
│   │                                 what "the served model" is, non-goals (#78)
│   └── serving_model_artifact.md    where the served model lives, why it is committed,
│                                     and its reproducibility evidence (#80)
├── demo/                    the playback client and the committed signal sample (#86)
│   ├── playback.py          replays a bearing against /predict; owns channel selection
│   ├── sample.py            what the committed sample is, and why it was cut that way
│   ├── build_sample.py      CLI: regenerate the sample from the full raw dataset
│   └── sample_data/         6.0 MB of real signal + its manifest — committed, so a fresh
│                             clone can run the demo with no dataset download
├── notebooks/               exploratory analysis (M1-EDA) + pipeline validation (M2)
├── src/                     extracted, testable modules imported by the notebooks
│   ├── labeling.py          health-state labeling logic (assign_labels)
│   ├── features/            M2 feature pipeline
│   │   ├── extraction.py            RMS / kurtosis / skewness per snapshot
│   │   ├── versioning.py            code + raw-data hashing, parquet manifests
│   │   ├── build_dataset.py         CLI: build all three experiments' parquet + manifest
│   │   ├── build_training_dataset.py  CLI: join feature parquets + labels into
│   │   │                              training_dataset.parquet (M3, #67)
│   │   └── candidate_features.py    evaluated-but-unused features (crest factor, BPFO/BPFI,
│   │                                spectral kurtosis) — kept, tested, not in the output
│   ├── training/             M3 baseline classifier, LOEO evaluation, MLflow tracking
│   │   ├── evaluation.py            LOEO folds, committed metrics, aggregation (#69)
│   │   ├── imbalance.py             five class-imbalance strategies compared under LOEO (#21)
│   │   ├── compare_imbalance.py     CLI: runs the #21 imbalance-strategy comparison
│   │   ├── train_baseline.py        CLI: M3 baseline + rms_ratio ablation + diagnostics (#72)
│   │   ├── candidate_scalers.py     evaluated-but-rejected scaling fixes — kept, tested
│   │   ├── mlflow_tracking.py       CLI: logs #21/#72's LOEO runs to MLflow (#74)
│   │   ├── train_serving_model.py   CLI: trains + persists the pooled M4 serving model (#80)
│   │   └── serving_model_tracking.py  logs the #80 serving run, tagged apart from #21/#72
│   └── serving/              M4 serving layer
│       ├── state.py          per-bearing rolling state, single-process by design (#82)
│       ├── features.py       online feature computation, batch-equivalent to #41 (#82)
│       ├── api.py            FastAPI /predict + /health (#84)
│       ├── single_worker.py  OS-level lock enforcing the single-worker constraint (#84)
│       ├── model_notes.py    the Section 4 disclosure, one source of truth (#84)
│       └── main.py           `python -m src.serving.main` — the documented run command
├── models/                  the served model artifact + its manifest — committed, ~1.7 KB,
│                             byte-reproducible (M4, #80; docs/serving_model_artifact.md)
├── tests/                   pytest unit tests for src/
└── .github/workflows/       CI: notebook execution + unit tests, and a container build
                              that runs a real playback against a real container (#86)
```

## Learn more

- **`docs/PRD.md`** — the full product definition: problem statement, goals and explicit
  non-goals, target architecture, tech stack, and milestone plan.
- **`docs/eda_findings.md`** — what the M1 exploratory analysis found: dataset
  characteristics, labeling-threshold derivation, and open items carried into later
  milestones.
- **`docs/CONTRIBUTING.md`** — commit message conventions and the branch/PR workflow used
  throughout this repo.
- **`docs/*_decision.md`** — the M2 decision notes (why features are windowed the way they
  are, why label transitions use hysteresis, how feature outputs are versioned, and which
  candidate features were evaluated and dropped) plus the M3 ones: which class-imbalance
  approach was adopted (`class_imbalance_decision.md`, #21) and the M3 baseline model with
  its `rms_ratio` ablation and `1st_test` diagnosis (`model_training_decision.md`, #72).
- **M3 protocol/versioning notes** — `docs/evaluation_protocol.md` fixes the leave-one-
  experiment-out split and committed metrics *before* any model existed (#69);
  `docs/training_dataset_versioning.md` covers how the training dataset is joined and
  versioned (#67); `docs/uncertainty_quantification.md` adds intervals and multiple-
  comparison correction to M1-M2's point estimates (#63); `docs/mlflow_tracking.md` covers
  the MLflow dependency choice and how to inspect the logged LOEO runs (#74).
