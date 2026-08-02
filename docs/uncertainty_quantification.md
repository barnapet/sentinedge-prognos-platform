# Uncertainty Quantification for M1-M2 Statistics (Issue #63)

M1-EDA and M2 report a large number of point estimates — ANOVA F-statistics, correlations,
per-label means, ratios — at two to four significant figures, with no indication of how much of
each figure is signal and how much is sampling noise. Some rest on very small samples:
`1st_test`'s `Critical` class is **17 files**, and `docs/frequency_domain_decision.md` Section 5's
envelope ratios are computed on it. Thirty-nine per-experiment ANOVA tests are reported across
Issues #22 and #23 with no multiple-comparison correction.

This note adds that missing layer. It **re-derives nothing and changes nothing**: every existing
point estimate stands exactly as published, and no keep/drop decision moves. What changes is that
the figures now carry an interval or a corrected p-value, and two of them turn out to deserve a
caveat that was not previously visible.

**No new dependency.** Everything here uses `scipy.stats.f_oneway` (which already returns a
p-value alongside the statistic) and `numpy`. Adding `statsmodels` for multiple-comparison
correction would contradict `docs/skewness_crestfactor_decision.md`'s own Method note, which chose
univariate ANOVA specifically to avoid adding infrastructure this milestone doesn't need. Holm's
procedure is about five lines of `numpy`.

## 1. What is covered, and what deliberately is not

| Covered | Where it comes from |
|---|---|
| 39 per-experiment ANOVA F-statistics (13 features × 3 experiments) | `docs/skewness_crestfactor_decision.md` §2-3, `docs/frequency_domain_decision.md` §4 |
| 14 pooled ANOVA tests (7 features, raw and within-experiment z-scored) | `docs/frequency_domain_decision.md` §6a |
| 6 envelope Critical/Normal ratios | `docs/frequency_domain_decision.md` §5 |
| 3 per-experiment `critical_multiple` values | `docs/eda_findings.md` §3 |

**Not covered, on purpose:**

- **The ~2.6x M4 serving fallback.** It is the geometric mean of three values from three bearings
  run under one operating condition — not three draws from a population. An interval computed from
  `n = 3` would be arithmetically expressible and substantively meaningless, and quoting one would
  imply a generalisation the dataset cannot support. The honest statement is the sample size
  itself, which `docs/eda_findings.md` §3 now carries. See also §7 below.
- **Correlation coefficients** (#22 §4/§6d, #23 §1). They are used qualitatively throughout
  ("moderate, not collinear"; "largely redundant"), and no decision turns on a specific decimal.
  Adding 30+ intervals would add length without changing a single conclusion.
- **Class-imbalance handling.** Adjacent to Issue #21 and deliberately left there — nothing in this
  note preempts M3's evaluation-protocol decisions.

## 2. Method, and why these choices

### 2a. Multiple comparisons: Holm, not Bonferroni

Both control the family-wise error rate under arbitrary dependence, both need no distributional
assumptions beyond what the F-test already makes, and both are a few lines of `numpy`. Holm's
step-down procedure is **uniformly more powerful** — it never rejects less than Bonferroni — so
there is no reason to prefer the weaker one.

On this corpus the difference is not academic: at α = 0.05 over the 39-test family, **Bonferroni
would reject two tests that Holm keeps** (`3rd_test`'s `spectral_kurtosis`, p_bonf = 0.80; and
`1st_test`'s `bpfi_amplitude_norm`, p_bonf = 0.48). Holm-adjusted, both come out at 0.0245. Both
figures are reported in §3 so a reader can apply whichever criterion they prefer.

**Family definition.** The 39 per-experiment tests are treated as one family: they answer a single
question ("which feature separates health states *within* one bearing's life?"), they were reported
across two documents only because the issues were sequential, and correcting per-document would
make the threshold depend on that accident. The 14 pooled tests (#22 §6a) are treated as a
**separate** family, because their unit of analysis is different — a pooled test asks about
separation across the concatenated dataset, not within an experiment. Correcting them jointly would
mix two questions into one threshold.

The subsampled high-pass sweep in #22 §6c is excluded from both families. It was exploratory,
computed on every 8th file, and #22 already treats its F-values as relative rather than absolute.

### 2b. Bootstrap for the envelope ratios: percentile, B = 10,000, and a block variant

The Section 5 quantity is a **ratio of two group means** (Critical mean ÷ Normal mean), which has
no clean closed-form interval, so a percentile bootstrap is the natural choice. `B = 10,000`
resamples: Monte-Carlo error on a 2.5%/97.5% percentile falls as `1/sqrt(B)`, and at 10,000 it is
well below the width of the intervals themselves; the whole computation costs seconds, so there was
no reason to economise. Seed `20260801`, fixed, so the numbers below are reproducible.

**Two variants are reported, and this matters more than the choice of B.** The `Critical` files are
not a random sample — they are a *contiguous run* at the end of the bearing's life (all 17 of
`1st_test`'s are consecutive), and consecutive snapshots of a degrading bearing are correlated. An
i.i.d. bootstrap therefore treats 17 correlated observations as 17 independent ones and **understates**
the interval. A moving-block bootstrap (block length 5) preserves local autocorrelation and gives
the more honest bound. Both appear in §4; where they disagree, the block interval is the one to
believe.

### 2c. `critical_multiple`: sensitivity analysis, *not* a bootstrap

This is the one place where the obvious method is the wrong one, so the reasoning is recorded
rather than the result alone.

`critical_multiple = sqrt(1.3 × peak_ratio_rolling)` is a function of the **maximum** of the rolling
RMS-ratio series. The naive bootstrap is *inconsistent* for the sample maximum — resampling with
replacement cannot produce a value above the observed max, so the bootstrap distribution piles up at
the boundary and its percentiles do not converge to those of the true sampling distribution. This is
a textbook failure case, not a borderline judgement. A second, independent problem: the series being
maximised is a 10-file rolling mean, so it is strongly autocorrelated by construction, and i.i.d.
resampling is invalid for it regardless of the statistic.

Quoting a bootstrap interval here would therefore be worse than quoting nothing: it would look
rigorous and mean nothing. What is reported instead is the question a reader actually has — **how
much does the threshold depend on which single file happened to be the peak?** — answered by
recomputing `critical_multiple` from the k-th largest rolling ratio for k = 1, 2, 3, 5, 10, and
then reporting the decision-relevant consequence: how many files change label.

## 3. Result: multiple-comparison correction changes nothing

**All 39 per-experiment tests survive Holm correction at α = 0.05.** The Bonferroni threshold for
this family is p < 1.282 × 10⁻³. The six weakest tests:

| Experiment | Feature | F | raw p | Holm p | Bonferroni p |
|---|---|---|---|---|---|
| `3rd_test` | `spectral_kurtosis` | 3.9 | 0.0205 | **0.0245** | 0.80 ✗ |
| `1st_test` | `bpfi_amplitude_norm` | 4.4 | 0.0123 | **0.0245** | 0.48 ✗ |
| `1st_test` | `bpfo_amplitude_norm` | 8.9 | 1.46e-4 | 4.38e-4 | 5.69e-3 |
| `2nd_test` | `bpfi_envelope_norm` | 13.9 | 1.12e-6 | 4.47e-6 | 4.36e-5 |
| `2nd_test` | `crest_factor` | 17.2 | 4.42e-8 | 2.21e-7 | 1.72e-6 |
| `1st_test` | `skewness` | 25.7 | 9.74e-12 | 6.28e-11 | 3.80e-10 |

Every other test in the family has a raw p below 10⁻¹¹.

In the pooled family (14 tests), **exactly one test fails to reach significance**:
`spectral_kurtosis`, z-scored within experiment, **F = 0.8, p = 0.445**. Every other pooled test has
p below 10⁻²²³. This is a direct statistical confirmation of #22 §6d's "no separability at all"
verdict — that section reached it from the F-statistic alone, and it now has a p-value behind it.
Correction does not change this: with m = 14 the adjusted p is still 0.445.

**The substantive reading is not "everything is significant, so everything is useful."** It is the
opposite: significance was never the binding constraint. #22 and #23 dropped crest factor, the
band-amplitude features and the envelope features while they were all *statistically significant*,
on effect-size, redundancy and stability grounds. This correction confirms that those drop decisions
were not driven by noise — and equally, that a significance-only criterion would have retained
every feature examined, including ones that are demonstrably useless. That is a point in favour of
how #22/#23 actually reasoned, and it is worth stating explicitly.

## 4. Result: envelope-ratio intervals, and one figure that needs a caveat

`docs/frequency_domain_decision.md` §5, Critical/Normal ratio of the mean. §5's own criterion is
stated as *"A working defect feature should be > 1."*

| Experiment | Feature | n (Critical) | Point | 95% CI (i.i.d.) | 95% CI (block-5) |
|---|---|---|---|---|---|
| `1st_test` | `bpfo_envelope_norm` | 17 | 1.31x | [1.15, 1.51] | [1.06, 1.54] |
| `1st_test` | `bpfi_envelope_norm` | 17 | **1.93x** | [1.75, 2.14] | [1.67, 2.09] |
| `2nd_test` | `bpfo_envelope_norm` | 23 | **1.08x** | [0.93, 1.27] ⚠ | [0.90, 1.12] ⚠ |
| `2nd_test` | `bpfi_envelope_norm` | 23 | 0.85x | [0.79, 0.91] | [0.79, 0.93] |
| `3rd_test` | `bpfo_envelope_norm` | 67 | **1.57x** | [1.49, 1.66] | [1.48, 1.65] |
| `3rd_test` | `bpfi_envelope_norm` | 67 | 1.44x | [1.37, 1.52] | [1.35, 1.56] |

⚠ **`2nd_test`'s `bpfo_envelope_norm` interval includes 1.0, under both bootstrap variants.** Its
1.08x point estimate is not distinguishable from "no response at all". This is the one place in
#22 where a reported figure does not support the strength of its framing, and it is worth being
precise about what it does and does not affect:

- **§5's comparative claim survives.** The claim that carries #22's physics check is that the
  defect frequency *matching the documented fault mode* responds **more than the other one**. For
  `2nd_test` that is BPFO (1.08x) vs BPFI (0.85x), and BPFI's interval sits entirely below 1.0
  while BPFO's straddles it — the ordering BPFO > BPFI holds. The same ordering in `1st_test`
  (1.93x vs 1.31x, both intervals excluding 1.0) and `3rd_test` (1.57x vs 1.44x) is unambiguous.
  So "the fault-matched frequency responds more strongly in all three experiments" stands.
- **§5's absolute criterion does not hold for this one cell.** "A working defect feature should be
  > 1" is not established for `2nd_test`'s `bpfo_envelope_norm` at 95% confidence.
- **No decision changes.** #22 dropped every one of these features anyway, for reasons (§6a's
  within-experiment z-scoring reversal, §6b's inconsistency, §6c's high-pass instability) that this
  interval only reinforces — `2nd_test` was already the experiment where the envelope features were
  weakest (F = 13.9 and 103.2).

The `1st_test` figures, the ones most exposed to the 17-file sample, hold up: both intervals exclude
1.0, and the headline 1.93x for the fault-matched BPFI is comfortably away from it even under the
block bootstrap.

## 5. Result: `critical_multiple` is sensitive in value, robust in labels

Recomputing `sqrt(1.3 × k-th largest rolling ratio)`:

| Experiment | k=1 (published) | k=2 | k=3 | k=5 | k=10 | Spread | Files changing label (k=2…10) |
|---|---|---|---|---|---|---|---|
| `1st_test` | **1.932** | 1.849 | 1.790 | 1.650 | 1.630 | 15.6% | 4 → 12 of 2,156 (≤ 0.56%) |
| `2nd_test` | **2.866** | 2.851 | 2.819 | 2.699 | 2.444 | 14.8% | 0 → 16 of 984 (≤ 1.63%) |
| `3rd_test` | **3.049** | 2.944 | 2.919 | 2.891 | 2.766 | 9.3% | 2 → 6 of 6,324 (≤ 0.09%) |

`k=1`'s values are corrected here (from `1.931`/`2.867`, unchanged for `3rd_test`'s `3.049`) to
match `src.labeling.derive_critical_multiple`'s canonical, rounding-contract output (`docs/eda_findings.md`
§3, `src/labeling.py`'s docstring, Issue #65) rather than the unrounded-peak-ratio figure that
docstring names as the value an implementation skipping that rounding would produce (`1.931235` /
`2.867072` / `3.049334`, which rounds to what this table originally carried).
This corrects only the `k=1` citation; the `k=2…10` sensitivity columns, `Spread`, and
label-churn figures are unaffected — they characterise the sweep, not this one already-published
value.

Two things follow.

**The square root is doing useful work.** Relative error in the peak propagates to
`critical_multiple` at half its size: `d(cm)/cm = ½ · d(peak)/peak`. Measured across the k-sweep the
observed ratio is 0.52–0.54 in all three experiments, matching the analytic ½ almost exactly. A
threshold defined directly on the peak would be twice as sensitive to the same perturbation.

**The labels are far more stable than the threshold.** Even the k=10 case — discarding the ten
largest rolling ratios, a far more aggressive perturbation than any plausible measurement error —
moves at most **1.63%** of files in any experiment, and under 0.6% in two of the three. The
`Critical` class boundary sits in a region where the rolling ratio is changing steeply, so moving
the threshold by 15% moves the crossing point by only a handful of files. This is the reassuring
half of the result, and it is the part that matters for M3: the training labels are not balanced on
a knife edge.

The value sensitivity is nonetheless real, and it compounds with the look-ahead property documented
in `docs/eda_findings.md` §3 — a threshold derived from one extreme observation of a completed run.
Neither is a defect for offline ground-truth labelling; both are constraints on how much weight the
specific numbers 1.932 / 2.866 / 3.049 should carry outside that role.

## 6. What this note does and does not change

**Does not change:** any F-statistic, correlation, ratio, per-label mean, `critical_multiple` value,
`ONSET_MULTIPLE`, the ~2.6x fallback, `FEATURE_COLUMNS`, any parquet output, or any keep/drop
decision from #22 or #23. Nothing here is a correction; the arithmetic in those documents
reproduces exactly.

**Does change:** three published figures now carry an explicit qualification —
`2nd_test`'s `bpfo_envelope_norm` ratio (interval includes 1.0), `spectral_kurtosis`'s pooled
non-separability (now with p = 0.445 behind it), and the three `critical_multiple` values (±15%
under peak perturbation, ≤1.6% label churn).

## 7. Limitations of this analysis itself

- **n = 3 experiments is the binding constraint, and no resampling fixes it.** Every interval above
  is a *within-experiment* interval: it quantifies sampling noise given this bearing's run. It says
  nothing about how the figure would transfer to a fourth bearing. The dataset has three
  run-to-failure experiments under a single operating condition, so between-bearing variability is
  estimated from three points, and this note deliberately declines to put an interval on any
  cross-experiment quantity (§1).
- **Effective sample size is below nominal.** The `Critical` runs are contiguous and the underlying
  series are rolling means, so even the block bootstrap's 17 correlated files carry less information
  than 17 independent ones. The block-5 intervals should be read as a lower bound on the true width,
  not an exact one.
- **The ANOVA assumptions are inherited, not re-examined.** One-way ANOVA assumes independent
  observations within groups, which time-ordered snapshots of a degrading bearing violate. This
  affects the p-values in §3 in the anti-conservative direction. Since all 39 tests clear the
  corrected threshold by many orders of magnitude — and the one test that fails, fails decisively —
  the conclusions are not close enough to the boundary for this to change them. A rank-based or
  block-permutation test would be the way to remove the assumption if a future issue needs a tighter
  claim.
- **Reproducibility.** Following the #22/#23 precedent, the analysis is recorded here rather than
  shipped as a module — the alternative would put new code inside `src/features/`, and anything
  added to `extraction.py` or `versioning.py` changes `code_hash` and invalidates every existing
  parquet manifest (`docs/feature_extraction_versioning.md` §2). The parameters needed to re-derive
  every number are stated in §2: `scipy.stats.f_oneway` p-values, Holm step-down over the families
  defined in §2a, percentile bootstrap with `B = 10,000` and seed `20260801`, moving-block length 5,
  and `critical_multiple` recomputed from the k-th largest value of the `rms_ratio` column in
  `data/processed/<name>_features.parquet`.
