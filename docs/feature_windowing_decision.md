# Feature Windowing Decision (Issue #40)

Decision-only note, no implementation. Answers: does M2's core feature extraction (RMS,
kurtosis, and the tentative skewness/crest factor candidates from `docs/eda_findings.md`
Section 4) use the same 10-file rolling window as `src/labeling.py`, or is a deviation
justified?

**Blocks:** Issue #41 (core feature-extraction module) cannot start until this is closed.
**Precondition for:** Issue #23 (skewness/crest factor redundancy check) — that evaluation
needs to know how those features are computed before it can assess them.
**May affect:** Issue #20 (onset-boundary hysteresis) — unrelated windowing mechanism, but
touches the same rolling-RMS computation; noted, not resolved here.

## 1. Decision

| Feature | Window | Ratio-to-baseline or absolute? |
|---|---|---|
| **RMS** | Same 10-file rolling mean as `src/labeling.py` (`min_periods=1`) | Ratio-to-baseline (unchanged) |
| **Kurtosis** | **Raw per-file — no rolling window** | Absolute (raw kurtosis value; Gaussian ≈ 3 is already a meaningful zero-point, no baseline-ratio needed) |
| **Skewness** *(gated on #23)* | 10-file rolling mean, same window as RMS, computed as its own smoothed series | Absolute threshold (not ratio-to-baseline) |
| **Crest factor** *(gated on #23)* | Raw per-file — no rolling window (default; see Section 4 for the open question) | Absolute / not yet thresholded — #23 decides if it's kept at all |

> **Outcome of the #23 gate (added post-hoc, decision text above left as written).** Issue #23
> ran the evaluation: **skewness was kept** — both `skewness` (raw) and `skewness_smoothed` (the
> 10-file rolling mean specified above) are in `src/features/extraction.py`'s `FEATURE_COLUMNS`,
> and #23 confirmed empirically that smoothing roughly doubles its separability, validating this
> note's analogy-based choice. **Crest factor was dropped**, so its windowing row above never
> became operative. See `docs/skewness_crestfactor_decision.md`, and Section 4 below for how
> that resolves this note's one open question.

So: RMS reuses the labeling window as-is. Kurtosis deliberately does **not** — this is the one
substantive divergence the issue asked about, and it goes the opposite direction from what
might be assumed (RMS and kurtosis diverge from each other, not because kurtosis needs its own
special window, but because it needs *no* window). Skewness, if it survives #23, converges back
onto the RMS window for smoothing but keeps its own (absolute) threshold logic. Crest factor is
tentatively unwindowed pending #23.

## 2. Rationale

**RMS ← same window.** `src/labeling.py`'s `rms_ratio` input already *is* the 10-file rolling
RMS ratio to baseline (mean of the first 50 files) — this isn't a new computation for M2 to
design, it's the existing one. Recomputing it identically means the RMS feature column and the
label are always derived from the same signal, which is the property you want for a feature
that is also the labeling basis. No technical obstacle: `pandas.Series.rolling(10,
min_periods=1)` never produces `NaN` (see Section 3) — the rolling window just shrinks to
whatever's available for files 0–8, which is already the behavior `src/labeling.py`'s inputs
and all three EDA notebooks rely on today (`01_vibration_signal_evolution.ipynb`,
`02_health_state_labeling.ipynb`, `03_feature_candidate_screening.ipynb` — grep confirms all
three use `rolling(ROLLING_WINDOW, min_periods=1)` identically).

**Kurtosis ← no window, deliberately.** This is the substantive decision. The entire evidentiary
case for keeping kurtosis as a feature (`docs/eda_findings.md` Section 4) rests on its **raw,
per-file, spiky** behavior:

- `1st_test`'s peak kurtosis of 74.6 is a per-file maximum, not a rolling-smoothed one
  (`01_vibration_signal_evolution.ipynb` cell 13 takes `kurt.max()` on the raw column).
- The reason kurtosis decouples from RMS specifically for `1st_test` (`corr = -0.02` over the
  `n=250` Degrading+Critical files — this note originally cited `-0.10` over `n=177`, the
  pre-#20 figure, refreshed per `docs/eda_findings.md` Section 4 and Issue #49 — vs.
  `0.63–0.65` for the other two experiments) is that the inner-race failure produces sharp,
  isolated impacts — transient spikes that a 10-file rolling mean would average down toward the
  surrounding quieter files, blunting exactly the signal that makes kurtosis worth having
  alongside RMS in the first place.
- The documented lead/lag behavior (kurtosis leads RMS onset by ~17h for `1st_test`, lags by
  ~24–53h for the other two) was measured on the raw series. Smoothing kurtosis before
  thresholding would change this timing and hasn't been evaluated.

Applying the RMS window to kurtosis "for consistency" would actively work against the reason
kurtosis is in the feature set. Kurtosis also doesn't need a baseline-ratio to be interpretable
the way RMS does — the standard (Pearson) kurtosis used throughout this project has a natural
reference point (Gaussian = 3; baseline kurtosis measured at ~3.4 across all three experiments,
per `03_feature_candidate_screening.ipynb` cell 16), so an absolute value is already meaningful
without dividing by a baseline. To be precise about the convention, since it is easy to invert:
this is `scipy.stats.kurtosis(..., fisher=False)` — *Pearson's* form, where a Gaussian sits at 3.
Fisher's is the *excess* form, which subtracts 3 and puts a Gaussian at 0; it is not what is
computed here (`src/features/extraction.py`'s `compute_kurtosis`, and the unit test
`test_compute_kurtosis_uses_standard_not_fisher_convention` that locks the convention in). The
~3.4 baseline figure is unaffected by this correction — only the name attached to it was wrong.

**Skewness ← RMS window, but absolute threshold.** This is exactly the case the issue flagged.
`03_feature_candidate_screening.ipynb` (Section 5) found baseline `|skewness|` ≈ 0.03 in all
three experiments — indistinguishable from zero — so "N× baseline" is meaningless for this
feature (a tiny absolute value times any multiplier is still tiny, and ordinary noise crosses it
trivially). Two separate problems, two separate fixes:
1. **Thresholding mechanism**: must be absolute, not ratio-to-baseline. This is unavoidable and
   isn't really a "windowing" choice — it follows directly from the baseline being ~0.
2. **Smoothing**: raw per-file skewness is noisy (per the notebook's own recommendation: "M2
   should test it smoothed... rather than raw per-file"). Unlike kurtosis, there's no evidence
   here that skewness's signal is carried by sharp per-file spikes that smoothing would destroy
   — to the contrary, the notebook explicitly proposes reusing "the same 10-file rolling mean
   used for RMS" to denoise it. So skewness reuses the RMS window for smoothing, independently
   of the fact that its threshold is absolute rather than ratio-based. These two properties
   (window, threshold type) are orthogonal — don't conflate "same window as RMS" with "same
   thresholding as RMS."

This is provisional on #23 confirming skewness is worth keeping at all.

**Crest factor ← no window (tentative).** No EDA evidence recommends smoothing crest factor, and
the one qualitative finding about it — non-monotonic with severity in `1st_test` (Normal 5.17 →
Degrading 10.89 → Critical 9.67, then falling; this note originally cited the pre-#20 values
5.31 → 11.77 → 9.67, refreshed per `docs/eda_findings.md` Section 4 and Issue #49, same shape) —
is itself a per-file, non-monotonic pattern. Smoothing it would suppress exactly the shape that
makes it interesting to look at, similar to the kurtosis argument. But this is a weaker case than
kurtosis's: crest factor is already flagged low-priority and largely redundant with kurtosis
(`corr` 0.56–0.88 in the Degrading+Critical window; originally cited as 0.57–0.88 pre-#20), so it
may not survive #23 regardless of windowing. Treated as raw/unwindowed by default here so #23
evaluates it on a comparable, unsmoothed basis to kurtosis; not a strong claim.

## 3. Boundary effects and NaN handling

Checked directly against `src/labeling.py`'s actual behavior and all three notebooks, rather
than assumed:

**There is no NaN-handling problem, because there are no NaNs.** Every rolling computation in
this codebase uses `min_periods=1`:

```python
df["rms"].rolling(ROLLING_WINDOW, min_periods=1).mean()
```

With `min_periods=1`, the rolling mean for file index `i < 9` is computed over whatever's
available (1 file at index 0, up to 10 by index 9), not `NaN`. `src/labeling.py` itself doesn't
compute this rolling column — it consumes a pre-computed `rms_ratio` column — but every producer
of that column upstream (both EDA notebooks and the planned M2 module) uses `min_periods=1`
identically. So the "boundary effect" here is **elevated variance in the first 9 files' ratio
estimate**, not a missing-data problem: file 0's "rolling RMS" is just that one file's RMS. This
is an existing, already-shipped property of the labeling pipeline (it's what produced the labels
in `docs/eda_findings.md` Section 3) — M2 inherits it rather than introducing it. Given the
baseline itself is defined over the first 50 files and is heavily Normal-dominated in every
experiment, this early-window noise has not been observed to cause mislabeling in practice.

**Temporal alignment across differently-windowed features is trivial, not a hard problem,
precisely because nothing produces NaN.** RMS (rolling) and kurtosis (raw) live in the same
output frame, indexed by the same `file_index`/`timestamp` — one column is a lookback aggregate,
the other is a per-row value, but both are defined for every row starting at file 0. There's no
row where one feature is present and another is `NaN` that needs dropping, forward-filling, or
alignment logic. If skewness is added with its own rolling smoothing, it follows the same
pattern — same index, same "no missing rows" property, just a different lookback computation
under the hood. The core module (#41) does not need any special-cased alignment or NaN-handling
logic for this reason; it only needs to compute each feature's column independently against the
shared `file_index` axis.

## 4. Open question — ~~not resolved here~~ resolved in Issue #23

**Crest factor's windowing (Section 2, last paragraph) is a weaker call than kurtosis's**, made
by analogy rather than direct EDA evidence — no notebook cell tested a smoothed vs. raw crest
factor comparison the way skewness's smoothing recommendation was explicit. If #23's
redundancy/importance pass finds crest factor worth keeping, it should re-examine whether raw or
smoothed reads better on its own terms rather than treating this note's default as settled. This
is flagged as genuinely open, not deferred as a formality.

**Resolved (Issue #23, `docs/skewness_crestfactor_decision.md` Section 3):** the comparison this
section asked for was run — smoothing roughly doubles crest factor's own separability
(`1st_test` F 638.6 → 1248.8) and cuts its correlation with kurtosis there (0.88 → 0.54), but
even smoothed it stays far below kurtosis in `2nd_test`/`3rd_test` (F 34.6 / 175.9 vs. 322.3 /
807.9), and the non-monotonicity persists. Crest factor was dropped either way, so the question
resolves as **moot**: there is no operative windowing choice left to make. `add_rolling_crest_factor`
exists in `src/features/candidate_features.py` (tested, unused) if a future issue re-opens this.
