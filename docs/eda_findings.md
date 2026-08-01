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

> **How this threshold is derived — retrospective, not causal (added in Issue #61).**
> `peak_ratio_rolling` is the highest rolling RMS ratio the bearing *actually reached*, which is
> only knowable once its run is over. The threshold that classifies a file early in the run is
> therefore set using information from the end of that same run: this is a **look-ahead
> derivation, not a causal/online one**. For a bearing still in operation there is no peak yet, so
> the formula above cannot be evaluated at serving time at all.
>
> That is precisely why the ~2.6x figure exists, and what it is: the geometric mean *across* the
> three retrospectively derived values, standing in for the per-experiment value a live bearing
> cannot supply. It is M4's fallback, not a fourth derived threshold.
>
> **This is a deliberate choice, not an oversight.** This section's job is to produce **offline
> ground-truth labels** for supervised training, and for that purpose using a completed run in full
> is both legitimate and standard — the labels describe what did happen to a bearing whose life is
> already over. The look-ahead would only be a defect if these thresholds were themselves the
> predictor, which they are not: M3 trains a classifier on features, and the labels are its target.
> What the look-ahead does constrain is the **serving** design, which is where the fallback question
> actually lands — stated here so a reader is told it rather than left to infer it from the
> parenthetical above.
>
> **Sensitivity (Issue #63).** These three values depend on a single extreme observation each — the
> peak rolling ratio. `docs/uncertainty_quantification.md` §5 quantifies that dependence: they move
> by 9-16% if the peak is taken from the 10th-largest file instead of the largest, but at most 1.6%
> of files change label as a result. Value-sensitive, label-robust.

**Rig-shutdown override:** after a Critical file, a raw-RMS collapse below 20% of the preceding
Critical window's own RMS forces `Critical` regardless of the raw threshold result. Verified
against the data (not assumed) to be needed only for `2nd_test` (files 982-983) and `3rd_test`
(file 6323); `1st_test` has zero files anywhere in its run below 0.5x baseline. At the current
10-file rolling window the override detects the artifact but changes no labels (the rolling mean
already absorbs a 1-2 file collapse); it becomes label-changing at smaller windows, so it stays in
as a guard.

**Label distribution** (current, post-#20 hysteresis — see the note below this table):

| Experiment | Normal | Degrading | Critical | Critical lead time |
|---|---|---|---|---|
| `1st_test` | 1,906 (88.4%) | 233 (10.8%) | 17 (0.8%) | 9.5h |
| `2nd_test` | 651 (66.2%) | 310 (31.5%) | 23 (2.3%) | 3.7h |
| `3rd_test` | 6,158 (97.4%) | 99 (1.6%) | 67 (1.1%) | 11.0h |
| **Pooled** | **8,715 (92.1%)** | **642 (6.8%)** | **107 (1.1%)** | — |

> **Updated in Issue #49.** This table originally recorded the pre-#20 counts
> (`1st_test` 1,979/160/17; `2nd_test` 657/304/23; pooled 8,794/563/107). Issue #20's
> hysteresis fix reclassified 73 `1st_test` files and 6 `2nd_test` files from `Normal` to
> `Degrading` — the flapping files described below — which moves those two experiments'
> Normal/Degrading split. **`3rd_test` is unchanged** (it never flapped), and **`Critical`
> counts and lead times are unchanged in all three**: hysteresis only affects files that were
> bouncing back to `Normal` mid-degradation, not the onset or Critical crossing points
> (`docs/label_hysteresis_decision.md` Section 4).

**Class imbalance is 81:1 Normal-to-Critical.** M3 needs class weighting or resampling, and should
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

> **Figures refreshed in Issue #49.** Every correlation and per-label statistic in this section is
> computed over the Degrading+Critical window, so all of them depend on the labeling rule.
> `03_feature_candidate_screening.ipynb` produced them *before* Issue #20's hysteresis fix
> (`n=177/327/166` Degrading+Critical files); the values below are recomputed against the current
> `src.labeling.assign_labels` (`n=250/333/166`), matching
> `notebooks/04_feature_pipeline_validation.ipynb` Section 4. Only `1st_test` moves materially —
> it gained 73 of the 79 reclassified files. Where a figure changed, the original pre-#20 value is
> given in parentheses so the two documents reconcile. Sources:
> `docs/label_hysteresis_decision.md` (the fix) and
> `notebooks/04_feature_pipeline_validation.ipynb` (the recomputation).

> **Uncertainty (Issue #63).** The separability statistics behind this table, and the
> frequency-domain figures in the last two rows, are point estimates on samples that get small at
> the `Critical` end (17 / 23 / 67 files). `docs/uncertainty_quantification.md` adds
> multiple-comparison correction and bootstrap intervals without changing any figure here: all 39
> per-experiment ANOVA tests survive Holm correction, and one #22 ratio turns out to need a caveat
> (see that document's §4).

| Feature | Status | Evidence |
|---|---|---|
| **RMS** | Confirmed useful | Already the basis of the #10 labeling rule; generalizes as a ratio-to-baseline across all three experiments. |
| **Kurtosis** | Confirmed useful | Decouples from RMS specifically in the impulsive failure (`1st_test`: `corr(rms, kurtosis) = -0.02` over `n=250` Degrading+Critical files — was `-0.10` over `n=177` pre-#20 — vs. `0.63-0.65` for the other two, unchanged) — catches a failure mode RMS amplitude undersells. The 73 files #20 reclassified sit right at the onset boundary, so they are near baseline RMS and not yet showing the sharp kurtosis spikes that dominate later; including them pulls the coefficient toward zero from below. Both values say the same thing — decoupled for `1st_test`, correlated for the other two — so the conclusion is unaffected. Lead/lag vs. RMS onset is failure-mode-dependent: leads by ~17h for `1st_test`, lags by ~24-53h for `2nd_test`/`3rd_test` (unchanged by #20 — onset indices did not move) — not a uniform "early warning" feature. |
| **Skewness** | Confirmed useful (Issue #23) | Real, non-redundant trend (`corr` with kurtosis `-0.43` to `-0.74` in the Degrading+Critical window — `1st_test` was `-0.42` pre-#20; the other two unchanged — not collinear): increasingly negative with severity in `2nd_test`/`3rd_test`. Shows pre-onset transient spikes in `1st_test` (5 files, unchanged by #20 since onset did not move; elevated kurtosis too, all within ~60 files of the RMS-based onset). Caveat: baseline `\|skewness\|` sits near zero (~0.03) in all three experiments, so #9/#10's ratio-to-baseline threshold pattern does not transfer — M2 needs an absolute threshold and likely a smoothed (rolling) version, not raw per-file skewness. |
| **Crest factor** | Evaluated and dropped (Issue #23) | Correlates `0.56-0.88` with kurtosis in the Degrading+Critical window across all three experiments (`2nd_test` was `0.57` pre-#20; the other two unchanged) — largely redundant where it counts. Non-monotonic with severity in `1st_test` (Normal 5.17 → Degrading 10.89 → Critical 9.67 — it falls back down; pre-#20: 5.31 → 11.77 → 9.67, same shape). The M2 redundancy check this row asked for was done in Issue #23 and dropped it — see `docs/skewness_crestfactor_decision.md`. |
| **Peak-to-peak** | Checked, not recommended | Correlates `0.65-0.95` with RMS, whole-life and within Degrading+Critical alike; near-constant multiple of RMS for `2nd_test`/`3rd_test` (coefficient of variation `0.09-0.10`). Adds negligible information beyond RMS. |
| **Frequency-domain** (spectral kurtosis, BPFI/BPFO-aligned energy) | **Investigated in Issue #22 — not adopted** | Was: "untested, flagged for M2". Issue #22 computed BPFO (236.40 Hz) / BPFI (296.93 Hz) band amplitudes, envelope-demodulated versions, and spectral kurtosis for all three experiments. The predicted inner-race/outer-race split **was confirmed** — the fault-matched defect frequency responds more strongly in all three experiments — but none of the features beat the retained time-domain set once between-experiment baseline offsets are removed, and the fixed-band formulation proved unstable to its own high-pass constant. Spectral kurtosis showed no separability (pooled F=0.8) and correlates 0.76 with time-domain kurtosis. Full analysis, bearing-geometry sourcing, and follow-up directions: `docs/frequency_domain_decision.md`. |
| **Frequency-domain** (dominant-frequency shift) | **Not evaluated — ruled out on physical grounds (Issue #59)** | Listed here as an M2 candidate alongside the two above, but never actually carried into Issue #22, whose verdict this row originally shared without that being true for it. Examined on its merits in Issue #59 instead of being run: at this rig's constant 2000 RPM the defect frequencies are kinematically pinned, so the shift being hunted is tiny (a contact-angle change far larger than wear produces moves BPFO by 1.9 FFT bins), while a 1% belt-slip speed wander moves it *further* (2.4 bins) — and the dataset has no tachometer channel to order-track that confound away, so any drift measured could not be attributed to wear. Cost was not the reason; the number simply would not have been interpretable. The physically real version of this idea (migration of the excited **resonance** band, not of the defect line) is already `docs/frequency_domain_decision.md` Section 7's top follow-up, kurtogram-based band selection. Full reasoning: that document's Section 8. |

## 5. Open items carried into M2

- ~~Hysteresis for `1st_test`'s onset-boundary flapping~~ (Section 3) — **resolved**, Issue #20:
  see `docs/label_hysteresis_decision.md`.
- ~~Skewness and crest factor need an explicit feature-importance/redundancy pass~~ —
  **resolved**, Issue #23: skewness kept (added to the feature pipeline), crest factor evaluated
  and dropped. See `docs/skewness_crestfactor_decision.md`.
- ~~Frequency-domain features are a real investigation candidate, not yet started~~ —
  **resolved**, Issue #22: BPFI/BPFO and spectral kurtosis investigated and not adopted (the
  documented fault-mode split was confirmed physically, but the features underperform the
  retained time-domain set). See `docs/frequency_domain_decision.md`, which also records the
  three follow-up directions worth trying if frequency-domain work is revisited. The third
  candidate originally listed in Section 4's row, *dominant-frequency shift*, was **not** part of
  that investigation and was ruled out separately in Issue #59 without being run — at constant
  shaft speed the confound (belt-slip speed wander, unmeasurable here for want of a tachometer
  channel) is larger than the effect. See that document's Section 8.
- Class imbalance (81:1) must be handled explicitly in M3's training approach (class weights or
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
