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

## Setup

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
├── requirements.txt        pinned Python dependencies
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
│   └── training/             M3 baseline classifier, LOEO evaluation, MLflow tracking
│       ├── evaluation.py            LOEO folds, committed metrics, aggregation (#69)
│       ├── imbalance.py             five class-imbalance strategies compared under LOEO (#21)
│       ├── compare_imbalance.py     CLI: runs the #21 imbalance-strategy comparison
│       ├── train_baseline.py        CLI: M3 baseline + rms_ratio ablation + diagnostics (#72)
│       ├── candidate_scalers.py     evaluated-but-rejected scaling fixes — kept, tested
│       ├── mlflow_tracking.py       CLI: logs #21/#72's LOEO runs to MLflow (#74)
│       ├── train_serving_model.py   CLI: trains + persists the pooled M4 serving model (#80)
│       └── serving_model_tracking.py  logs the #80 serving run, tagged apart from #21/#72
├── models/                  the served model artifact + its manifest — committed, ~1.7 KB,
│                             byte-reproducible (M4, #80; docs/serving_model_artifact.md)
├── tests/                   pytest unit tests for src/
└── .github/workflows/       CI: notebook execution + unit tests on every PR
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
