# Training Dataset Join & Versioning Decisions (Issue #67)

Decision note accompanying `src/features/build_training_dataset.py`. Covers the two
points Issue #67 asked to be resolved explicitly rather than left implicit: whether
`derive_critical_multiple` needs `data/raw/`, and whether the joined output gets its
own manifest. Written down per the same rigor as `docs/feature_extraction_versioning.md`
(#41), `docs/frequency_domain_decision.md` (#22), and `docs/skewness_crestfactor_decision.md`
(#23) — verified against the actual data and code, not assumed.

## 1. No dependency on `data/raw/`

`derive_critical_multiple` (#65) takes one number: an experiment's peak `rms_ratio`. The
feature parquet (#41) already carries a full `rms_ratio` column — the same 10-file
rolling-mean-over-baseline series `assign_labels` thresholds
(`docs/feature_extraction_versioning.md` §1) — so `df["rms_ratio"].max()` supplies it
without touching raw data.

Verified, not assumed: taking the peak from each experiment's `*_features.parquet` and
calling `derive_critical_multiple` on it reproduces `docs/eda_findings.md` §3's exact
values, and calling `assign_labels` on the result reproduces that section's exact label
counts:

| Experiment | `critical_multiple` | Normal | Degrading | Critical |
|---|---|---|---|---|
| `1st_test` | 1.932 | 1,906 | 233 | 17 |
| `2nd_test` | 2.866 | 651 | 310 | 23 |
| `3rd_test` | 3.049 | 6,158 | 99 | 67 |

This matches the table exactly (`tests/test_build_training_dataset.py`, per-experiment
row counts and label counts). Consequence: `build_training_dataset` reads only the three
`data/processed/<name>_features.parquet` files and their manifests — nothing in this
module opens `data/raw/`, and it never needs to (confirmed by #65's own pre-work check,
which flagged this as the expected outcome before #67 started).

## 2. Decision: the joined dataset gets its own manifest, independent of `GENERATING_CODE_FILES`

**Chosen: a dedicated `training_dataset_manifest.json`, with its own hash chain
entirely separate from `src/features/versioning.py`'s `GENERATING_CODE_FILES`.**

### Why not "no versioning, always regenerate"

The counter-case is real: this join is deterministic and runs in well under a second
against data already on disk, so "just re-run it before every M3 training run" is not an
unreasonable position. But `docs/PRD.md` §12 named the exact failure mode this project
already committed to guarding against, before #41 existed: *"if the labeling logic,
feature-extraction code, or raw dataset changes, a stale cached feature file could
silently get reused in modeling without anyone noticing."* `training_dataset.parquet`
sits one join downstream of both halves of that sentence — the feature parquets *and*
`src/labeling.py` — so leaving it unversioned reopens precisely the gap #41 closed one
layer up. A no-manifest choice would need its own justification for why this artifact is
exempt from a principle the project already applied twice; nothing about this join makes
it exempt.

### Why not reuse/extend `GENERATING_CODE_FILES`

The issue's own phrasing floated "now also hashing `labeling.py`'s content" as if it were
an extension of the existing tuple. It is deliberately **not** implemented that way.
`GENERATING_CODE_FILES` in `src/features/versioning.py` is scoped to what determines the
*feature-parquet* `combined_hash`, which is paired with `raw_dataset_version` — a
fingerprint of the ~6.2 GB raw NASA archive. Issue #65 already established the reason not
to add `labeling.py` there: doing so would change every feature parquet's
`combined_hash` and imply a stale-cache signal that forces re-running extraction against
raw data neither the extraction code nor the raw dataset actually changed. That reasoning
applies here with equal force, so this module leaves `GENERATING_CODE_FILES` untouched
(verified: `compute_code_hash()` returns the same value, `d4d6585e...`, before and after
this issue) and defines its own `LABELING_CODE_FILES` tuple in
`build_training_dataset.py` instead.

### What the new hash chain actually covers

Two independent pieces, combined the same way `docs/feature_extraction_versioning.md` §2
combines code + raw-data version — same principle, different inputs:

- **`labeling_code_hash`** — SHA-256 over the sorted, concatenated bytes of
  `src/labeling.py` and `src/features/build_training_dataset.py`
  (`LABELING_CODE_FILES`). Full file content, not a (filename, size) shortcut: unlike the
  6.2 GB raw dataset that motivated that shortcut in #41, these are two small source
  files, so a full content hash costs nothing extra.
- **`upstream_feature_version`** — SHA-256 over `f"{experiment}:{combined_hash}"` for
  each experiment's *existing* feature-parquet manifest, sorted by experiment name. This
  is the join's actual "data version": it changes if `build_dataset.py` (#41) is
  re-run against changed extraction code or changed raw data, without this module
  re-reading raw data or re-hashing parquet content itself — the three feature manifests
  already did that work, once, upstream.
- **`combined_hash`** — `SHA256(f"{labeling_code_hash}:{upstream_feature_version}")`,
  reusing `src.features.versioning.compute_combined_hash` (generic over two hash
  strings, not tied to `GENERATING_CODE_FILES` or `raw_dataset_version` — no
  modification needed to that function or file).

This answers the question that matters for M3: *was this training dataset produced by
this exact labeling code, joined against this exact set of upstream feature-parquet
versions?* — one level up the pipeline from what the feature-parquet manifests already
answer for extraction against raw data.

### Manifest format

`data/processed/training_dataset_manifest.json`: `generated_at` (UTC ISO-8601),
`labeling_code_hash`, `upstream_feature_version`, `combined_hash`, `critical_multiple`
(per-experiment, for a human-readable check against `docs/eda_findings.md` §3 without
opening the parquet — same rationale as `n_files` in the feature manifest,
`docs/feature_extraction_versioning.md` §4), `labels`, `columns`, `n_files`
(per-experiment) and `n_files_total`. One manifest for the one combined output file, not
one per experiment — unlike the feature parquets, this artifact has a single output and
a single hash chain, so there is nothing to keep separate.

Reproducibility verified directly: re-running `build_training_dataset()` against
unchanged inputs reproduces the same `combined_hash`
(`tests/test_build_training_dataset.py::test_reproducibility_...`).

## 3. `rms_ratio` is included, unmodified — the leakage question is out of scope here

`rms_ratio` is both the signal `assign_labels` thresholds on and the strongest
health-state feature (confirmed by the F-statistics in
`docs/frequency_domain_decision.md` §6a). Whether that circularity should exclude it
from a trained model is a Step 4 (model training + ablation) question, explicitly *not*
resolved, worked around, or hedged here — Issue #67 Task 3, restated in the module
docstring. `training_dataset.parquet` carries `rms_ratio` as a plain column, computed
and joined exactly as the upstream feature parquet produced it.
