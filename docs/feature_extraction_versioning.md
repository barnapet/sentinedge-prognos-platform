# Feature Extraction Module & Versioning Decisions (Issue #41)

Decision note accompanying `src/features/`. Covers the points that were genuinely open
when implementing Issue #41 and had no single obviously-correct answer: the parquet
schema, the hash algorithm, the manifest format, and whether labels belong in this
issue's output. Written down explicitly, per the issue's own instruction, since
Issues #22, #23, and M3 (training) all build on this module and shouldn't have to
reverse-engineer these choices from code.

## 1. Parquet schema — no labels in this output

`data/processed/<name>_features.parquet` contains exactly `FEATURE_COLUMNS`:

| Column | Meaning |
|---|---|
| `experiment` | Which test set the row belongs to (`1st_test`/`2nd_test`/`3rd_test`) — added in Issue #43 so the three parquet files can be `pd.concat`-ed and filtered/grouped by test set without relying on filename |
| `file_index` | 0-based chronological snapshot index within the experiment |
| `timestamp` | Parsed from the snapshot filename |
| `rms` | Raw per-file RMS of the tracked bearing's channel |
| `rms_ratio` | 10-file rolling mean of `rms`, divided by the first-50-file baseline mean — the same ratio `src.labeling.assign_labels` consumes (`docs/feature_windowing_decision.md`) |
| `kurtosis` | Raw per-file standard (Pearson) kurtosis, no rolling window |

**Note (Issue #43):** #41 originally shipped this schema without a column identifying
the test set — only the filename (`<name>_features.parquet`) distinguished the three
outputs. Issue #43's AC 2 required grouping/filtering by test set without relying on
filename, so `experiment` was added as a constant per-row tag, stamped on by
`extract_experiment_features`/`build_experiment` rather than derived by any
test-set-specific branching inside the extraction functions themselves (per #43 AC 3).

**Decision: no `label`/`label_pre_override`/`override_applied` columns here.** Issue #41
is scoped to RMS/kurtosis feature extraction; producing a label requires
`critical_multiple`, a per-experiment value currently derived only inside
`notebooks/02_health_state_labeling.ipynb` (geometric-midpoint sweep over each
bearing's own onset→peak span — see `docs/eda_findings.md` Section 3), and not yet
extracted into an importable function the way `assign_labels` itself was (Issue #19).
Extracting that derivation now would be scope creep beyond what #41 asks for.

This does **not** mean labels are disconnected from hysteresis (Issue #20,
`docs/label_hysteresis_decision.md`): `rms_ratio` here is computed with the exact same
window, `min_periods=1`, and baseline formula that feeds `assign_labels`, so nothing in
this module duplicates or diverges from the current (post-#45, hysteresis-patched)
labeling behavior. Any consumer — M3 or a future issue — can call
`src.labeling.assign_labels(df, critical_multiple)` directly on this module's output
(it already has the required `rms_ratio`/`rms` columns) once it has a
`critical_multiple` to pass in. If a future issue needs the join productionized (e.g.
M3 needing it repeatedly), extracting `derive_critical_multiple` out of the notebook
is a natural, separate follow-up rather than part of #41.

## 2. Hash algorithm

Two independent hashes are combined, per the "combining both approaches" instruction
and `docs/PRD.md` Section 12's rough direction:

- **`code_hash`** — SHA-256 over the sorted, concatenated bytes of
  `src/features/extraction.py` and `src/features/versioning.py` (`GENERATING_CODE_FILES`
  in `versioning.py`). Sorting the file list before hashing makes the result independent
  of argument order.
- **`raw_dataset_version`** — SHA-256 over `f"{filename}:{size_in_bytes}"` for every
  file in the experiment's raw directory, sorted by filename.
- **`combined_hash`** — `SHA256(f"{code_hash}:{raw_dataset_version}")`. This is the
  single value that answers "was this parquet produced by this exact code against this
  exact raw data?" and is what the reproducibility check (re-run → same hash) is
  checking.

**Decision: `raw_dataset_version` fingerprints (filename, size), not full file content.**
The NASA IMS raw dataset is ~6.2 GB across 9,464 files; a full SHA-256 over file
content costs several minutes per run — comparable to the feature extraction itself.
It's a fixed, publicly archived, immutable dataset (see `data/README.md`); the
realistic ways it could actually "change" in this project (wrong file count from a
bad re-extraction, a truncated download, the archive's well-documented layout quirks
being handled differently) all show up as a different file listing or a different file
size. A silent same-size content mutation would not be caught by this fingerprint —
that's a deliberate, documented trade-off given the cost of the alternative, not an
oversight. If this ever needs to be stronger, hashing the specific channel arrays this
pipeline already reads (zero extra I/O, since they're loaded into memory for RMS/
kurtosis anyway) would be the next step up in strength without a second read pass over
raw disk data — noted here as the natural escalation path, not implemented now because
nothing has motivated the extra complexity yet.

## 3. Manifest format — one JSON file per experiment, not one combined file

`data/processed/<name>_features_manifest.json` accompanies each parquet, containing:
`experiment`, `generated_at` (UTC ISO-8601), `code_hash`, `raw_dataset_version`,
`combined_hash`, `feature_columns`, `n_files`.

**Decision: JSON, one per experiment, not a single combined manifest for all three.**
JSON over an alternative (e.g. YAML, a row in a CSV) because Python's stdlib `json`
needs no new dependency and the structure is a flat dict, not the kind of thing
prose-style formats help with. One manifest per experiment (not one manifest covering
all three) because each experiment has its own raw directory, its own
`raw_dataset_version`, and — per the windowing decision doc — could in principle use
different extraction parameters per experiment in the future; keeping them separate
means a change to only `1st_test`'s raw data doesn't touch `2nd_test`/`3rd_test`'s
manifests. This mirrors the existing per-experiment file pattern already used for the
notebook-era cache (`<name>_rms_kurtosis.csv`). Filenames are stable
(`<name>_features.parquet`, not hash-suffixed) — the manifest is the versioning
evidence; a predictable path is more useful to M3 than a hash-encoded one that would
need discovery via manifest lookup anyway to find the current file.

## 4. `n_files` as an integrity cross-check

`manifest["n_files"]` is the row count actually written to the parquet, letting a
consumer sanity-check the parquet without opening it (e.g. compare against
`docs/eda_findings.md`'s documented per-experiment file counts: 2,156 / 984 / 6,324).
It is not part of the hash itself — it's redundant with `raw_dataset_version`'s file
listing but cheap to include and useful as a human-readable check.
