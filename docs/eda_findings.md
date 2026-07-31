# EDA Findings — M1

This document synthesizes the exploratory work behind Milestone 1 (`docs/PRD.md` Section 11)
into one place: what the dataset actually looks like, the health-state labeling rule it produced,
and the feature-extraction candidates M2 should build from. Every number here traces back to a
cached CSV and an executed notebook cell in `notebooks/` — nothing is restated from memory.

Source notebooks:
- `01_vibration_signal_evolution.ipynb` (#9) — raw signal and RMS/kurtosis lifecycle trends.
- `02_health_state_labeling.ipynb` (#10, merged) — the Normal/Degrading/Critical labeling rule.
- `03_feature_candidate_screening.ipynb` (#11) — whether skewness/crest factor/peak-to-peak add
  anything beyond RMS/kurtosis.

## 1. Dataset

Three independent run-to-failure experiments, one tracked bearing per experiment (the one NASA's
documentation records as the one that actually failed):

| Experiment | Files | Span | Bearing / channel | Documented failure mode |
|---|---|---|---|---|
| `1st_test` | 2,156 | 2003-10-22 → 2003-11-25 (~5 weeks) | Bearing 3, Ch 5 | inner race defect |
| `2nd_test` | 984 | 2004-02-12 → 2004-02-19 (~1 week) | Bearing 1, Ch 1 | outer race failure |
| `3rd_test` | 6,324 | 2004-03-04 → 2004-04-18 (~6 weeks) | Bearing 3, Ch 3 | outer race failure |

9,464 files total, each a 1-second, 20,480-point snapshot at 20 kHz, recorded roughly every 10
minutes across the bearing's life. Full acquisition/validation detail (channel counts, archive
quirks, download steps): `data/README.md` (#8).

## 2. Key findings from the raw signal (#9)

**Degradation onset does not generalize as a fixed point in time.** A 10-file rolling mean of RMS
crossing 1.3x the baseline RMS (mean of the first 50 files) — the "onset" marker — lands at very
different points per experiment:

| Experiment | Onset (% of life) | Peak RMS ratio (rolling) | Peak kurtosis |
|---|---|---|---|
| `1st_test` | 88.4% | 2.87x | 74.6 |
| `2nd_test` | 66.2% | 6.32x | 17.1 |
| `3rd_test` | 97.4% | 7.15x | 19.7 |

A threshold expressed as elapsed time or file count would not transfer between experiments —
everything downstream is defined relative to each bearing's own baseline instead.

**The three experiments split into two distinct failure signatures**, not incidental noise:

- `1st_test`'s inner-race failure is **impulsive**: peak kurtosis is 74.6 (vs. 17-20 for the other
  two) but its RMS only reaches 2.87x baseline — the lowest of the three. Failure shows up in
  signal *shape* well before it shows up in *amplitude*.
- `2nd_test` and `3rd_test`'s outer-race failures are **amplitude-driven**: RMS reaches 6.3x-7.2x
  baseline, with comparatively modest kurtosis.

This distinction is not just descriptive — #11's feature screening (Section 4 below) shows it has
direct, measurable consequences for which features actually carry independent information.

**End-of-life rig-shutdown artifact:** the final recorded file(s) of `2nd_test` and `3rd_test`
collapse to near-zero RMS (0.0015g / 0.0040g) — the test rig auto-stopping the shaft after
detecting failure, not a return to health. `1st_test` has no such collapse (ends at its own peak,
0.594g) — its documented failure mode let the scheduled test run to completion rather than
tripping a stop condition.

## 3. Health-state labeling rule (#10)

Per file, on the 10-file rolling RMS ratio to baseline:

| Label | Condition |
|---|---|
| `Normal` | ratio ≤ 1.3 |
| `Degrading` | 1.3 < ratio ≤ `critical_multiple` |
| `Critical` | ratio > `critical_multiple` |

**`critical_multiple` is derived per experiment**, not a fixed constant — a sweep showed why a
constant doesn't work: any value ≥3.0 gives `1st_test` zero Critical files (its peak ratio is only
2.87x), while a value low enough to populate `1st_test` fires far too early on the other two. The
derivation is the geometric midpoint of each bearing's own onset→peak span:

```
critical_multiple = sqrt(1.3 * peak_ratio_rolling)
```

giving **1.93x / 2.87x / 3.05x** for `1st_test` / `2nd_test` / `3rd_test` — three independently
derived values spanning only 1.58x, from inputs spanning 2.49x. Recorded fallback for M4 serving
(where a bearing's eventual peak is unknowable): the geometric mean, **~2.6x baseline**.

**Rig-shutdown override:** after a Critical file, a raw-RMS collapse below 20% of the preceding
Critical window's own RMS forces `Critical` regardless of the raw threshold result. Verified
against the data (not assumed) to be needed only for `2nd_test` (files 982-983) and `3rd_test`
(file 6323); `1st_test` has zero files anywhere in its run below 0.5x baseline. At the current
10-file rolling window the override detects the artifact but changes no labels (the rolling mean
already absorbs a 1-2 file collapse); it becomes label-changing at smaller windows, so it stays in
as a guard.

**Label distribution:**

| Experiment | Normal | Degrading | Critical | Critical lead time |
|---|---|---|---|---|
| `1st_test` | 1,979 (91.8%) | 160 (7.4%) | 17 (0.8%) | 9.5h |
| `2nd_test` | 657 (66.8%) | 304 (30.9%) | 23 (2.3%) | 3.7h |
| `3rd_test` | 6,158 (97.4%) | 99 (1.6%) | 67 (1.1%) | 11.0h |
| **Pooled** | **8,794 (92.9%)** | **563 (6.0%)** | **107 (1.1%)** | — |

**Class imbalance is 82:1 Normal-to-Critical.** M3 needs class weighting or resampling, and should
report per-class recall rather than plain accuracy — the imbalance is inherent to run-to-failure
data, not a labeling defect, and should not be corrected by moving the threshold.

**Resolved in Issue #20** (was: known open issue deferred to M2): 73 files (3.4%) of `1st_test`
were reverting to `Normal` after degradation had already begun, flapping across the 1.3x onset
line over files 1906-1999 — its RMS rises so gently the rolling mean straddled the boundary for a
while. Every one of those files genuinely sits within a few percent of the boundary (ambiguous in
the signal, not miscomputed by the rule). `2nd_test` turned out to have a smaller, previously
unflagged version of the same thing (6 files, 788-793). Fixed with hysteresis on downward label
transitions only — contrary to what this section originally assumed, this did **not** require
changing #9's locked `ONSET_MULTIPLE` constant: the upward (onset-entering) comparison against
1.3 is unchanged, and only a new exit-side margin was added for reverting. Full investigation,
quantification, and the rejected alternative (N-consecutive-file confirmation): see
`docs/label_hysteresis_decision.md`.

## 4. Feature-extraction candidates for M2 (#11)

Checked against the cached/computed data rather than assumed from a generic vibration-analysis
list. Full derivation and correlation tables: `03_feature_candidate_screening.ipynb`.

| Feature | Status | Evidence |
|---|---|---|
| **RMS** | Confirmed useful | Already the basis of the #10 labeling rule; generalizes as a ratio-to-baseline across all three experiments. |
| **Kurtosis** | Confirmed useful | Decouples from RMS specifically in the impulsive failure (`1st_test`: `corr(rms, kurtosis) = -0.10` within the Degrading+Critical window, vs. `0.63-0.65` for the other two) — catches a failure mode RMS amplitude undersells. Lead/lag vs. RMS onset is failure-mode-dependent: leads by ~17h for `1st_test`, lags by ~24-53h for `2nd_test`/`3rd_test` — not a uniform "early warning" feature. |
| **Skewness** | Plausible, worth testing | Real, non-redundant trend (`corr` with kurtosis `-0.42` to `-0.74` in the Degrading+Critical window, not collinear): increasingly negative with severity in `2nd_test`/`3rd_test`. Shows pre-onset transient spikes in `1st_test` (5 files, elevated kurtosis too, all within ~60 files of the RMS-based onset). Caveat: baseline `|skewness|` sits near zero (~0.03) in all three experiments, so #9/#10's ratio-to-baseline threshold pattern does not transfer — M2 needs an absolute threshold and likely a smoothed (rolling) version, not raw per-file skewness. |
| **Crest factor** | Plausible, low priority | Correlates `0.57-0.88` with kurtosis in the Degrading+Critical window across all three experiments — largely redundant where it counts. Non-monotonic with severity in `1st_test` (Normal 5.31 → Degrading 11.77 → Critical 9.67 — it falls back down). Cheap to compute; include in an M2 feature-importance/redundancy check rather than hand-designing a threshold around it. |
| **Peak-to-peak** | Checked, not recommended | Correlates `0.65-0.95` with RMS, whole-life and within Degrading+Critical alike; near-constant multiple of RMS for `2nd_test`/`3rd_test` (coefficient of variation `0.09-0.10`). Adds negligible information beyond RMS. |
| **Frequency-domain** (spectral kurtosis, BPFI/BPFO-aligned energy, dominant-frequency shift) | Untested, flagged for M2 | No FFT/spectral analysis has been performed on this dataset. Plausible given the documented inner-race (`1st_test`) vs. outer-race (`2nd_test`/`3rd_test`) split — classical bearing-fault theory predicts different characteristic defect frequencies for the two — but nothing here confirms a specific feature works. M2 should investigate, grounded in the documented fault type, rather than assume a generic FFT-peak-energy feature is sufficient. |

## 5. Open items carried into M2

- ~~Hysteresis for `1st_test`'s onset-boundary flapping~~ (Section 3) — **resolved**, Issue #20:
  see `docs/label_hysteresis_decision.md`.
- Skewness and crest factor need an explicit feature-importance/redundancy pass once real
  extraction is built, rather than being included by default. Tracked as Issue #23.
- Frequency-domain features are a real investigation candidate, not yet started.
  Tracked as Issue #22.
- Class imbalance (82:1) must be handled explicitly in M3's training approach (class weights or
  resampling) and evaluation (per-class recall). Tracked as Issue #21.

## Validation performed (Issue #11, M1-EDA)

- [x] Onset-generalization and failure-mode findings (#9) restated with figures pulled from the
      executed notebook, not memory.
- [x] Labeling rule, derivation, override, and class balance (#10) restated with figures pulled
      from the executed notebook.
- [x] Skewness, crest factor, and peak-to-peak checked against computed per-file data (not
      assumed) before being included or excluded from the M2 shortlist.
- [x] Frequency-domain features explicitly flagged as untested rather than asserted.

No feature-extraction pipeline code is implemented here — that is M2's scope
(`src/features/`, `docs/PRD.md` Section 11).
