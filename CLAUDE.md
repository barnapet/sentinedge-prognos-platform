# SentinEdge Prognos Platform — Project Context for Claude Code

## What this project is

A predictive maintenance AI platform engineering portfolio project. Predicts bearing health
state (Normal / Degrading / Critical) from vibration signal data (NASA IMS bearing dataset),
served through a proper serving layer with monitoring — not a notebook demo.

Full scope, decisions, and rationale: see `docs/PRD.md`. Read it before proposing anything that
changes scope, architecture, or model approach — the classification-vs-RUL decision, the
Kubernetes/Machina/CI-CT-CD exclusions, and the latency/monitoring targets are already decided
and documented there. Don't reopen them without being asked.

## Conventions — follow these without being reminded

**Commits:** Conventional Commits format. `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
`test:`, `perf:`. Short, specific subject lines. See `CONTRIBUTING.md` for full detail.

**Branches:** `<type>/<short-description>`, e.g. `feat/predict-endpoint`.

**Workflow:** one Issue per non-trivial task, branch off `main`, small logical commits, PR
referencing the Issue (`Closes #N`), merge only when the relevant PRD Acceptance Criteria
(Section 10) are satisfied.

**Definition of Done:** check work against `docs/PRD.md` Section 10 before marking anything
complete — not "it runs locally."

**Merging and history:** never run `gh pr merge` without explicit approval for that specific
merge — a prior approval for branch deletion or PR opening does not extend to the merge action
itself. Always request explicit approval before any force-push or history rewrite (rebase,
cherry-pick that overwrites a remote branch), even when using `--force-with-lease`.

**Verify before proceeding:** in any multi-step git/gh workflow, confirm the actual output of
each command before moving to the next step, rather than assuming success. Chained shell
commands can partially fail silently (a missing file breaking a `cp`, then a downstream `git
commit` succeeding on stale state) — check real output, don't infer it from the absence of an
error on an unrelated line.

## Scope discipline

This project deliberately excludes (for MVP, see PRD Section 4): Kubernetes, full CI/CT/CD with
automated retraining, the Machina agent layer, multi-model serving, real-time streaming
ingestion, a FinOps dashboard. If a task seems to call for one of these, flag it and ask before
introducing it — don't add infrastructure the current milestone doesn't need.

## Repo structure (target — see PRD Section 10a)

```
docs/           PRD and other documentation
data/           gitignored — raw dataset, not committed
notebooks/      EDA, exploratory work
src/features/   feature extraction pipeline
src/training/   training scripts, MLflow logging
src/serving/    FastAPI app, /predict and /metrics
demo/           simulated live-feed playback script
tests/
```

## Current milestone

M1-EDA is complete: dataset acquisition, exploratory analysis, and health-state label
threshold definition are done (Issues #8, #9, #10, #11 closed; see `docs/eda_findings.md`).
The M1.5-Housekeeping stretch that followed close-out is also complete (CI pipeline, unit
tests, README, pinned dependencies, notebook-output policy, data-versioning note, and these
conventions — Issues #18, #19, #24, #25, #26, #27, #28, #29 all closed). This was tracked
informally as "M1.5" in CLAUDE.md/Issues only, not a PRD milestone (PRD Section 11 notes it
without renumbering). M2 (feature pipeline) is next.

## When in doubt

Prefer asking over assuming when a decision would affect scope, architecture, or the
Acceptance Criteria. Small, reviewable steps over large speculative changes.
