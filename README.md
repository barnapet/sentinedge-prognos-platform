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
│   └── *_decision.md        M2 decision notes: windowing (#40), label hysteresis (#20),
│                            feature extraction + versioning (#41), skewness/crest factor
│                            (#23), frequency domain (#22)
├── notebooks/               exploratory analysis (M1-EDA) + pipeline validation (M2)
├── src/                     extracted, testable modules imported by the notebooks
│   ├── labeling.py          health-state labeling logic (assign_labels)
│   └── features/            M2 feature pipeline
│       ├── extraction.py        RMS / kurtosis / skewness per snapshot
│       ├── versioning.py        code + raw-data hashing, parquet manifests
│       ├── build_dataset.py     CLI: build all three experiments' parquet + manifest
│       └── candidate_features.py  evaluated-but-unused features (crest factor, BPFO/BPFI,
│                                  spectral kurtosis) — kept, tested, not in the output
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
- **`docs/*_decision.md`** — the M2 decision notes: why features are windowed the way they
  are, why label transitions use hysteresis, how feature outputs are versioned, and which
  candidate features were evaluated and dropped (with the evidence behind each drop).
