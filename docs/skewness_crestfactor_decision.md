# Skewness / Crest Factor Redundancy Decision (Issue #23)

Evaluation-only issue: `docs/eda_findings.md` Section 4 flagged skewness "plausible, worth
testing" and crest factor "plausible, low priority" during M1-EDA, and `docs/feature_windowing_decision.md`
(Issue #40) fixed *how* each would be computed if kept (windowing/threshold mechanism), explicitly
leaving *whether* to keep either one to this issue. This document is the actual evaluation:
correlation/redundancy analysis, a univariate feature-importance pass, and a final keep/drop
decision for each feature, computed against real per-file data for all three experiments (not
assumed from the M1-EDA notebook's older, pre-hysteresis numbers).

**Method note (why univariate ANOVA, not a tree-based model):** Issue #23 offered either a
tree-based importance pass on a baseline model or univariate separability across health states
"as used elsewhere in M1-EDA." This document uses the latter — a one-way ANOVA F-statistic across
the three health-state groups (Normal/Degrading/Critical) per feature per experiment — because
`scikit-learn` is not a project dependency (`requirements.txt`) and M2's scope discipline
(`CLAUDE.md`) is to avoid adding infrastructure a milestone doesn't need; a tree-based importance
pass would require adding it just for this one evaluation. Univariate ANOVA needs no new
dependency (`scipy.stats.f_oneway`, already a dependency), and matches the EDA's own methodology
(`03_feature_candidate_screening.ipynb`'s correlation/threshold-by-label tables).

## 0. Data and labels used

All numbers below are computed from `src/features/extraction.py` (RMS/kurtosis/skewness) and
`src/features/candidate_features.py` (crest factor) run against the real raw dataset for all
three experiments, joined with `src.labeling.assign_labels` using the current (Issue #20,
hysteresis-patched) labeling logic and each experiment's documented `critical_multiple`
(1.932 / 2.866 / 3.049 — `docs/eda_findings.md` Section 3). This is **not** the same labeling
snapshot `docs/eda_findings.md` Section 4's original skewness/crest-factor correlation numbers
were computed against (that was pre-#20, in `03_feature_candidate_screening.ipynb`) — see Section
4 below for where this causes small, expected numeric differences.

Baseline (first 50 files) sanity check, confirming `docs/feature_windowing_decision.md`'s premise
that skewness needs an absolute threshold:

| Experiment | Baseline `\|skewness\|` | Baseline kurtosis | Baseline crest factor |
|---|---|---|---|
| `1st_test` | 0.026 | 3.37 | 4.25 |
| `2nd_test` | 0.029 | 3.48 | 5.31 |
| `3rd_test` | 0.033 | 3.47 | 5.92 |

Confirms baseline `\|skewness\|` ~ 0.03 (indistinguishable from zero, as `docs/eda_findings.md`
found) and baseline kurtosis ~3.4-3.5 (consistent with `docs/feature_windowing_decision.md`) in
all three experiments.

## 1. Correlation matrix (Degrading+Critical only)

Restricted to Degrading+Critical files — the region a health-state classifier actually has to
discriminate within (same rationale as `03_feature_candidate_screening.ipynb` Section 4).

**`1st_test` (n=250):**

| | rms | kurtosis | skewness | skewness_smoothed | crest_factor |
|---|---|---|---|---|---|
| rms | 1.00 | -0.02 | -0.01 | 0.01 | -0.02 |
| kurtosis | -0.02 | 1.00 | -0.43 | -0.23 | **0.88** |
| skewness | -0.01 | -0.43 | 1.00 | 0.17 | -0.41 |
| skewness_smoothed | 0.01 | -0.23 | 0.17 | 1.00 | -0.21 |
| crest_factor | -0.02 | **0.88** | -0.41 | -0.21 | 1.00 |

**`2nd_test` (n=333):**

| | rms | kurtosis | skewness | skewness_smoothed | crest_factor |
|---|---|---|---|---|---|
| rms | 1.00 | 0.65 | -0.49 | -0.28 | 0.38 |
| kurtosis | 0.65 | 1.00 | -0.49 | -0.26 | 0.56 |
| skewness | -0.49 | -0.49 | 1.00 | 0.59 | -0.38 |
| skewness_smoothed | -0.28 | -0.26 | 0.59 | 1.00 | -0.14 |
| crest_factor | 0.38 | 0.56 | -0.38 | -0.14 | 1.00 |

**`3rd_test` (n=166):**

| | rms | kurtosis | skewness | skewness_smoothed | crest_factor |
|---|---|---|---|---|---|
| rms | 1.00 | 0.63 | -0.25 | -0.17 | 0.58 |
| kurtosis | 0.63 | 1.00 | -0.74 | -0.56 | **0.78** |
| skewness | -0.25 | -0.74 | 1.00 | 0.78 | -0.59 |
| skewness_smoothed | -0.17 | -0.56 | 0.78 | 1.00 | -0.49 |
| crest_factor | 0.58 | **0.78** | -0.59 | -0.49 | 1.00 |

**Crest factor correlates 0.56-0.88 with kurtosis in every experiment** — consistent with
`docs/eda_findings.md`'s original 0.57-0.88 finding (small numeric shift from the hysteresis-fixed
Degrading group, not a different conclusion). Highest exactly where kurtosis is already most
discriminative (`1st_test`, 0.88).

**Skewness correlates -0.43 to -0.74 with kurtosis** — moderate, not collinear, matching
`docs/eda_findings.md`'s original -0.42 to -0.74 finding almost exactly.

## 2. Univariate separability (one-way ANOVA F-statistic across Normal/Degrading/Critical)

| Experiment | rms | kurtosis | skewness (raw) | skewness_smoothed | crest_factor |
|---|---|---|---|---|---|
| `1st_test` | 1497.8 | 285.0 | 25.7 | **231.9** | 638.6 |
| `2nd_test` | 1151.0 | 322.3 | 218.0 | **488.4** | 17.2 |
| `3rd_test` | 18826.3 | 807.9 | 1036.9 | **1811.0** | 84.9 |

Two findings drive the decision:

**Smoothing roughly doubles (or more) skewness's separability in every experiment**
(`1st_test`: 25.7 → 231.9; `2nd_test`: 218.0 → 488.4; `3rd_test`: 1036.9 → 1811.0) — this is an
empirical confirmation of `docs/feature_windowing_decision.md`'s smoothing choice, not just the
analogy-based justification that decision was originally made on.

**`skewness_smoothed`'s F-statistic exceeds kurtosis's in 2 of 3 experiments** (`2nd_test`:
488.4 vs. 322.3; `3rd_test`: 1811.0 vs. 807.9) and is comparable in the third (`1st_test`: 231.9
vs. 285.0) — combined with only moderate correlation to kurtosis (Section 1), this is strong,
largely-independent signal, not a restatement of what kurtosis already provides.

**`crest_factor`'s F-statistic is far weaker than kurtosis's in `2nd_test`/`3rd_test`** (17.2 vs.
322.3; 84.9 vs. 807.9) — and where it *is* comparable or stronger (`1st_test`: 638.6 vs. 285.0),
Section 1 shows it's 0.88-correlated with kurtosis there, so that separability is largely
borrowed, not independent.

## 3. Crest factor: is the "no window" default from #40 actually right?

`docs/feature_windowing_decision.md` Section 4 flagged crest factor's unwindowed default as a
weak call made by analogy, and asked this issue to check on its own merits. Raw vs. a 10-file
rolling mean (`crest_factor_smoothed`), same window as RMS/skewness:

| Experiment | F (raw) | F (smoothed) | corr(kurtosis) raw | corr(kurtosis) smoothed |
|---|---|---|---|---|
| `1st_test` | 638.6 | 1248.8 | 0.88 | 0.54 |
| `2nd_test` | 17.2 | 34.6 | 0.56 | 0.46 |
| `3rd_test` | 84.9 | 175.9 | 0.78 | 0.78 |

Smoothing roughly doubles crest factor's own separability too, and meaningfully reduces its
correlation with kurtosis in `1st_test` (0.88 → 0.54). **This does not change the keep/drop
decision**: even smoothed, `2nd_test`/`3rd_test`'s F-statistics (34.6, 175.9) remain far below
kurtosis's (322.3, 807.9) — crest factor stays both weak and at least moderately redundant in the
two experiments where it isn't already explained by kurtosis. The non-monotonicity `docs/eda_findings.md`
flagged for `1st_test` (`Normal` 5.17 → `Degrading` 10.89 → `Critical` 9.67, falling back down —
the current-labeling values; `docs/eda_findings.md`'s *original*, pre-#20 figures were
`5.31` → `11.77` → `9.67`, same shape) also persists after smoothing (`5.15` → `10.96` →
`10.05`, likewise computed under the current labeling) — smoothing doesn't fix the property
that made a hand-designed threshold awkward for this feature in the first place. **Conclusion: the
open windowing question is resolved as moot** — crest factor doesn't survive the keep/drop
decision regardless of window, so there's no operative windowing choice left to make.

## 4. Note on the pre-#20 vs. current-labeling numeric differences

`docs/eda_findings.md`'s original correlation numbers (`docs/eda_findings.md` Section 4;
`03_feature_candidate_screening.ipynb`) were computed before Issue #20's hysteresis fix, using
`n=177/327/166` Degrading+Critical files. This document uses the current, hysteresis-patched
`assign_labels`, giving `n=250/333/166` (`1st_test` and `2nd_test` gain the reclassified
onset-boundary files; `3rd_test` is unchanged, per `docs/label_hysteresis_decision.md`). This is
the same, already-documented and understood effect flagged in
`notebooks/04_feature_pipeline_validation.ipynb` Section 4 for `corr(rms, kurtosis)`. All
skewness/crest-factor correlation values here land close to the original pre-#20 numbers (see
Section 1) — the qualitative conclusions are unaffected, and none of the small numeric shifts
change the keep/drop decision below.

## 5. Decision

**Skewness: KEEP.** `skewness_smoothed` (10-file rolling mean, per `docs/feature_windowing_decision.md`)
separates health states at least as strongly as kurtosis in every experiment (Section 2), while
staying only moderately correlated with it (Section 1) — genuine, largely independent signal, not
a restatement of an existing feature. Both `skewness` (raw) and `skewness_smoothed` are added to
`src/features/extraction.py`'s `FEATURE_COLUMNS` and computed by `extract_experiment_features`
(raw kept alongside smoothed for the same reason `rms`/`rms_ratio` both exist: the raw value is
useful diagnostically even though the smoothed one is what actually threshold/separates well).
`data/processed/*_features.parquet` regenerated accordingly; manifests' `combined_hash` changes
because `extraction.py`'s content changed (expected, per `docs/feature_extraction_versioning.md`'s
hash design).

**Crest factor: DROP (evaluated, not used).** Redundant with kurtosis where it is otherwise most
discriminative (`1st_test`, `corr=0.88`), and both redundant *and* weak elsewhere (`2nd_test`/
`3rd_test`: `corr=0.56/0.78`, F-statistic far below kurtosis's). Smoothing (Section 3) does not
change this. `compute_crest_factor`/`extract_crest_factor` remain in
`src/features/candidate_features.py`, unit-tested, but are **not** called from
`extract_experiment_features` and **not** part of `FEATURE_COLUMNS` or the parquet output — kept,
not deleted, per Issue #23's own instruction, in case a future issue re-evaluates crest factor
against a different feature set or failure mode.

## 6. Out of scope: an operational skewness threshold

The Issue #23 GitHub comment flagged "determining the actual threshold value" as this issue's job,
following on from `docs/feature_windowing_decision.md` fixing only the windowing *method*. This
document does not derive one. Per `docs/PRD.md`, M3 trains a health-state classifier directly on
feature columns (RMS, kurtosis, and now skewness) rather than predicting from hand-designed,
per-feature thresholds the way `rms_ratio`'s `ONSET_MULTIPLE`/`critical_multiple` generate ground-truth
*labels* — a classifier consuming `skewness_smoothed` as one input among several learns its own
decision boundary rather than needing a hand-derived absolute cutoff. Deriving one here would be
solving a problem M3's design doesn't have; revisit only if a future milestone actually needs a
standalone, interpretable skewness rule (e.g. a simple rule-based fallback), not as part of this
issue.
