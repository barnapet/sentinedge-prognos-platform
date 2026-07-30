# SentinEdge Prognos Platform
Predictive maintenance AI platform for industrial bearing vibration data — a served,
monitored ML system, not a notebook demo.

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
