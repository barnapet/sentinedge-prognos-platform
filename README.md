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

The raw NASA IMS bearing dataset (~1.1 GB) is not committed to the repo — it's gitignored
under `data/raw/`. See `data/README.md` for the download source and step-by-step extraction
instructions (the archive is nested 7z + RAR, with a couple of undocumented-upstream quirks
that `data/README.md` covers).

### Running the notebooks

```bash
jupyter notebook notebooks/
```

Run in order: `01_vibration_signal_evolution.ipynb` → `02_health_state_labeling.ipynb` →
`03_feature_candidate_screening.ipynb`. Each depends on outputs cached by the previous one
under `data/processed/`.

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
│   └── CONTRIBUTING.md      commit conventions, branch/PR workflow
├── notebooks/               exploratory analysis and labeling logic (M1-EDA)
├── src/                     extracted, testable modules imported by the notebooks
│   └── labeling.py          health-state labeling logic (assign_labels)
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
