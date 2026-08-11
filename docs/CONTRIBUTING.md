# Contributing / Project Conventions

This is a solo-maintained portfolio project, but it follows the same discipline as a team repo
on purpose — the conventions themselves are part of what the project demonstrates.

## Commit Messages — Conventional Commits

Every commit follows: `<type>: <short description>`

| Type | Use for |
|---|---|
| `feat` | New functionality (a new endpoint, a new pipeline step) |
| `fix` | Bug fix |
| `docs` | Documentation only (README, PRD, comments) |
| `chore` | Tooling, config, `.gitignore`, dependency bumps — no behavior change |
| `refactor` | Code restructuring with no behavior change |
| `test` | Adding or fixing tests |
| `perf` | Performance improvement |

Examples:
```
feat(serving): add /predict endpoint
fix(features): correct FFT window overlap bug
docs: add PRD v1.1
chore: add .gitignore for Python/ML artifacts
```

Keep the description short and specific. If a commit needs more explanation, put a blank line
after the summary and add detail in the commit body — don't cram it into the subject line.

## Branch Naming

`<type>/<short-description>`, matching the commit type prefixes above:

```
feat/predict-endpoint
fix/fft-window-overlap
docs/prd-v1.1
chore/gitignore
```

## Workflow (even solo)

1. Open an Issue before starting non-trivial work — one Issue per Milestone task (see
   `docs/PRD.md` Section 11 for the milestone list). This is the "think before you build" step.
2. Branch off `main` using the naming convention above.
3. Commit in small, logical steps — each commit should be a coherent unit, not "wip" or "misc
   changes."
4. Open a Pull Request against `main`, even solo. Reference the Issue it closes
   (`Closes #<number>`).
5. Merge only after the PR description confirms the relevant Acceptance Criteria item(s) from
   the PRD are met.
6. Delete the branch after merge.

This produces a readable history: anyone reviewing the repo can follow *why* each change
happened, not just *what* changed.

## Issue Template

Issue bodies follow a four-header shape — Context, Task, Constraints, Output:

- **Context** — why the issue exists: the background, the problem being addressed, and
  pointers to related issues or decisions it builds on (or deliberately doesn't re-open).
- **Task** — the concrete, usually numbered, list of what to build.
- **Constraints** — explicit scope limits: what not to touch, what not to re-decide, and any
  process rules (e.g. "prepare the PR and stop; wait for explicit approval").
- **Output** — what the resulting PR must contain and confirm, so "done" is checkable rather
  than a judgment call.

This is a quick reference for the shape that's already been used consistently, not a rigid
template to fill mechanically.

## Notebook Outputs

Committed notebooks (`notebooks/*.ipynb`) **keep their execution outputs** — plots, printed
tables, and cell results are not stripped before commit. This is a deliberate choice, not an
oversight: this is a portfolio project, and outputs let anyone browsing the repo on GitHub see
the actual EDA results (plots, derived thresholds, etc.) without cloning and re-running
anything. `README.md`'s "Repo structure" section relies on this.

The trade-off: diffs on notebook files include output/execution-count noise alongside real
code changes, and merge conflicts can occur on output blobs even when the underlying code
doesn't conflict. This is considered acceptable at the project's current scale (a handful of
notebooks, solo-maintained) — re-evaluate if M2/M3 add enough notebooks or contributors that
this becomes painful in practice. If so, `nbstripout` (strips outputs, cleanest diffs, loses
GitHub-viewable outputs) or `jupytext` (paired `.py`/`.md` representation for clean code diffs,
keeps outputs in the `.ipynb`) are the two options considered and rejected for now — see
Issue #28.

## Milestone Tags

At the end of each milestone (see `docs/PRD.md` Section 11), tag the commit that completes it:

```
v0.1-eda
v0.2-feature-pipeline
v0.3-baseline-model
v0.4-serving
v0.5-monitoring
v1.0-mvp
```

Add a short release note per tag summarizing what was completed and any deviations from the
PRD's original plan — deviations are fine, but should be documented, not silent.

This list covers the original MVP's milestones only — it's not an exhaustive list going
forward. Later phases (M7 and beyond) may introduce their own tags as they complete.

## Definition of Done

Before closing an Issue or merging a milestone PR, check it against the Acceptance Criteria
that apply to its phase — don't mark something done based on "it runs on my machine."

- For issues within the original MVP's milestones (M1–M6, see `docs/PRD.md` Section 11), check
  against the Acceptance Criteria in `docs/PRD.md` Section 10, as before.
- For issues in a later phase (M7 and beyond), `docs/PRD.md` Section 10 doesn't describe them —
  check against that phase's own design doc instead (e.g. M7-Agent-Layer against
  `docs/agent_design.md`).
