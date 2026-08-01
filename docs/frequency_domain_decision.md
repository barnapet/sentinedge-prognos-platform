# Frequency-Domain Feature Decision (Issue #22)

First frequency-domain analysis performed on this dataset — everything in M1-EDA and M2 so far
(`docs/eda_findings.md`, `docs/skewness_crestfactor_decision.md`) is time-domain. Issue #22 asks
whether BPFI/BPFO defect-band energy and spectral kurtosis add anything beyond the retained
time-domain set (`rms`/`rms_ratio`, `kurtosis`, `skewness`/`skewness_smoothed`).

**Decision up front: all frequency-domain features evaluated here are dropped.** They are kept,
tested, in `src/features/candidate_features.py` as evaluated-but-unused, alongside crest factor
from Issue #23. `FEATURE_COLUMNS` and the `data/processed/` parquet outputs are unchanged by this
issue. Sections 4-6 give the evidence; Section 7 records what a future issue should try instead,
because "dropped" here does **not** mean "frequency domain does not work on this dataset."

Method matches Issue #23 for comparability: one-way ANOVA F-statistic across
Normal/Degrading/Critical plus a correlation matrix restricted to Degrading+Critical, using the
current (Issue #20, hysteresis-patched) `assign_labels` and each experiment's documented
`critical_multiple`. No new dependency was added — `scipy.fft`/`scipy.signal`/`scipy.stats` are
already required.

## 1. Bearing parameters: what is documented, what is not

This mattered more than expected, so it is recorded precisely.

**Documented in this project / the dataset's own README PDF** (`data/raw/Readme Document for IMS
Bearing Data.pdf`, "Test Rig Setup"):

| Parameter | Value | Source |
|---|---|---|
| Shaft speed | 2000 RPM, held constant | dataset PDF; restated in `docs/PRD.md` Section 6 |
| Bearing model | Rexnord ZA-2115, double row | dataset PDF |
| Radial load | 6000 lbs | dataset PDF |
| Sampling | 20,480 points/file @ 20 kHz | dataset PDF; validated in `data/README.md` |

**Not documented anywhere in this repo or in the dataset PDF:** roller count, pitch diameter,
roller diameter, contact angle. The PDF names the bearing model but publishes no geometry table.

This is the blocking-input question Issue #22 asked about, and the honest answer is that the
geometry is **not** available from the project's own materials. It was sourced externally — the
geometry is a property of the ZA-2115 part, not of any one paper, and the values below are
consistent across the published literature on this dataset:

| Parameter | Value used |
|---|---|
| Rollers per row (`n`) | 16 |
| Pitch diameter (`D`) | 2.815 in |
| Roller diameter (`d`) | 0.331 in |
| Contact angle (`θ`) | 15.17° |

**Why these are trusted rather than assumed.** They are cross-validated, not taken on faith:
substituting them and the *documented* 2000 RPM into the standard ball-pass formulas reproduces
the characteristic frequencies independently published for this dataset (~236 Hz BPFO / ~297 Hz
BPFI) to within rounding. Geometry that was wrong would not land on both published values
simultaneously. `test_compute_bearing_frequencies_matches_published_values_for_this_dataset`
locks this in, so a future edit to any geometry constant fails the suite.

**Residual risk, stated rather than buried:** the contact angle is the weakest link — it is the
least consistently published of the four, and it enters only through `cos θ` (0.965 at 15.17°),
so a moderate error moves BPFO/BPFI by well under the ±3 Hz band half-width used below. The
conclusions in this document are therefore not sensitive to it. If a future issue needs
narrower bands or higher harmonics, the geometry should be re-sourced from a manufacturer
datasheet first.

## 2. Computed characteristic frequencies

    BPFO = (n/2) · f_r · (1 − (d/D)·cos θ)
    BPFI = (n/2) · f_r · (1 + (d/D)·cos θ)

| Quantity | Value |
|---|---|
| Shaft rotation `f_r` | 33.333 Hz |
| **BPFO** | **236.40 Hz** |
| **BPFI** | **296.93 Hz** |
| FFT bin width (20 kHz / 20,480) | 0.977 Hz |

These are identical for all three experiments — shaft speed is held constant across the whole
dataset, so the characteristic frequencies do not vary per experiment. What varies is the
documented *fault mode* (`1st_test` inner race → BPFI relevant; `2nd_test`/`3rd_test` outer race
→ BPFO relevant), which Section 5 uses as a physical check.

Both BPFO and BPFI are computed for **every** experiment, not just the one matching its
documented fault. This keeps a single code path with no test-set-specific branching (the same
constraint Issue #43 placed on `extract_experiment_features`) and lets the fault-mode
correspondence be *measured* rather than assumed.

## 3. What was computed, and two implementation choices that mattered

Five features per snapshot, on the same tracked-bearing channel the time-domain pipeline uses
(`EXPERIMENTS[...].channel_idx`, `docs/eda_findings.md` Section 1):

| Feature | Definition |
|---|---|
| `bpfo_amplitude`, `bpfi_amplitude` | Summed amplitude in ±3 Hz bands around the defect frequency and its first 2 harmonics |
| `bpfo_amplitude_norm`, `bpfi_amplitude_norm` | The same, divided by the spectrum's median (noise floor) |
| `bpfo_envelope_norm`, `bpfi_envelope_norm` | Noise-floor-normalised band amplitude of the **envelope** spectrum (1 kHz high-pass → Hilbert → FFT) |
| `spectral_kurtosis` | Max over frequency bins of `E[\|X\|⁴]/E[\|X\|²]² − 2` across STFT frames (Antoni's SK indicator) |

**Choice 1 — a Hann window is applied before the FFT.** Defect frequencies are not bin-centred
(236.40 Hz on a 0.977 Hz grid), so an unwindowed FFT leaks rectangular-window sidelobes across
the spectrum: measured on a synthetic BPFO tone, ~1.7% of its amplitude landed in the BPFI band
60 Hz away. Since BPFO and BPFI are read as separate features, that leakage would make each
partly a measurement of the other. Hann improves the selectivity ratio from 57:1 to ~10,900:1.
This was caught by a unit test failing, not by inspection.

**Choice 2 — envelope analysis is included, and it is what makes a null result credible.** A
rolling-element defect usually does not inject energy *at* 236 Hz; each impact rings the
structure's high-frequency resonances, so the defect rate appears as the *modulation rate* of a
high-frequency carrier. Reading the raw spectrum at BPFO can therefore find nothing while the
defect is plainly present. On a synthetic resonance-modulated defect (4 kHz carrier rung at the
BPFO rate, with no energy at 236 Hz itself), the plain spectrum reads 122 and the envelope
spectrum 2379 — a ~20x difference. Without this, "frequency-domain features don't separate here"
would have been an artifact of using the wrong method, not a finding.

## 4. Separability per experiment (ANOVA F across Normal/Degrading/Critical)

Existing retained features in the first three columns for reference.

| Experiment | `rms` | `kurtosis` | `skew_sm` | `bpfo_amp` | `bpfi_amp` | `bpfo_norm` | `bpfi_norm` | `bpfo_env` | `bpfi_env` | `spec_kurt` |
|---|---|---|---|---|---|---|---|---|---|---|
| `1st_test` | 1497.8 | 285.0 | 231.9 | 333.3 | 496.4 | 8.9 | 4.4 | 133.0 | **508.5** | 55.8 |
| `2nd_test` | 1151.0 | 322.3 | 488.4 | 374.1 | 406.8 | 62.1 | 35.2 | 103.2 | 13.9 | 26.1 |
| `3rd_test` | 18826.3 | 807.9 | 1811.0 | 2111.0 | 2122.4 | 169.5 | 197.7 | 587.8 | 537.2 | 3.9 |

**Raw band amplitudes are redundant with RMS.** Their high F-statistics are borrowed: in the
Degrading+Critical window they correlate **0.78–0.89** with `rms` in every experiment (`1st_test`
0.78/0.88, `2nd_test` 0.87/0.89, `3rd_test` 0.89/0.88). A defect band gets louder mostly because
*everything* gets louder. This is the same trap crest factor fell into in Issue #23, and it is
why the normalised forms exist.

**Plain-spectrum normalised amplitudes are independent but weak.** Correlation with `rms` drops to
−0.29…0.00, but separability collapses to F = 4.4–197.7 — below every retained feature. Worse,
they move the *wrong way*: Critical/Normal mean ratios are 0.70–0.96x, i.e. the defect tone sinks
*relative to* the noise floor as the bearing dies, because broadband energy rises faster than the
discrete tones. Physically sensible, but not a degradation indicator.

**Envelope features are the only ones that behave correctly** — Section 5.

## 5. The envelope features do reproduce the documented physics

Critical/Normal ratio of the mean, per feature. A working defect feature should be > 1.

| Experiment | Documented fault | `bpfo_env` | `bpfi_env` | Fault-matched frequency responds more? |
|---|---|---|---|---|
| `1st_test` | inner race → expect **BPFI** | 1.31x | **1.93x** | ✅ BPFI > BPFO |
| `2nd_test` | outer race → expect **BPFO** | **1.08x** | 0.85x | ✅ BPFO > BPFI |
| `3rd_test` | outer race → expect **BPFO** | **1.57x** | 1.44x | ✅ BPFO > BPFI |

> **Bootstrap intervals (Issue #63).** These ratios are computed on the `Critical` class, which is
> 17 / 23 / 67 files. `docs/uncertainty_quantification.md` §4 puts 95% intervals on all six. The
> comparative claim below — fault-matched frequency responds more strongly, in all three — survives.
> One cell needs a caveat: **`2nd_test`'s `bpfo_env` 1.08x interval includes 1.0** (i.i.d.
> [0.93, 1.27], block [0.90, 1.12]), so for that experiment the *absolute* "> 1" criterion stated
> just above is not established at 95% confidence, even though BPFO > BPFI still holds (BPFI's
> interval lies entirely below 1.0). The `1st_test` figures, most exposed to the 17-file sample,
> both exclude 1.0. No value in the table changes, and the drop decision in §6 is unaffected.

**In all three experiments the defect frequency matching the documented fault mode responds more
strongly than the other one.** That is a genuine, independent confirmation that the bearing
geometry (Section 1), the derived frequencies (Section 2), and the envelope implementation
(Section 3) are all correct — the physics shows up where theory says it should. It is the main
positive result of this issue, and it is why the drop below is a judgement about *strength*, not
a claim that the approach is invalid.

They are also genuinely independent of the existing set — pooled Degrading+Critical correlations
are |r| ≤ 0.22 against `rms`, `kurtosis`, and `skewness_smoothed` alike, lower than skewness's own
correlation with kurtosis (−0.23…−0.56) that Issue #23 accepted as "not collinear".

## 6. Why they are dropped anyway

**6a. The apparently strong pooled result is an artifact.** Pooled across all three experiments,
the envelope features look excellent — beating `kurtosis` by more than 2x:

| Feature | Pooled F (raw) | Pooled F (z-scored within experiment) |
|---|---|---|
| `rms_ratio` | 14702.7 | 15019.6 |
| `rms` | 2982.6 | 11362.6 |
| `skewness_smoothed` | 2396.7 | 1508.5 |
| `kurtosis` | 665.5 | **1122.0** |
| `bpfo_envelope_norm` | 1490.6 | **607.7** |
| `bpfi_envelope_norm` | 1357.9 | **543.0** |
| `spectral_kurtosis` | 73.8 | **0.8** |

The raw pooled column is misleading: `3rd_test` supplies 67% of the rows, and the envelope
features sit at different baseline levels per experiment, so part of that F reflects
*between-experiment* differences rather than health-state separation. Z-scoring within each
experiment before pooling removes those offsets and asks the question a classifier actually
faces — separating states *within a bearing's own life*. Under that fair test the envelope
features fall to roughly **half** of `kurtosis`, a feature already retained. This reversal is the
single most decision-relevant number in this document, and it only appears once the offsets are
removed.

**6b. Per-experiment behaviour is inconsistent.** `bpfi_env` is the best non-RMS feature in
`1st_test` (508.5, beating both `kurtosis` and `skewness_smoothed`) but near-useless in
`2nd_test` (13.9). No single frequency-domain column is reliably strong across all three. A
fault-agnostic `max(bpfo_env, bpfi_env)` was tested as a workaround — necessary because a serving
model cannot know the fault mode in advance — and scored 537.9 / 75.8 / 924.2, still below
`skewness_smoothed` in two of three experiments and below `kurtosis` pooled (1030.2 vs 1122.0).

**6c. They depend heavily on a hand-picked constant.** The 1 kHz envelope high-pass is a
convention, not something derivable from this data. Sweeping it (subsampled every 8th file, so F
values are lower than the full-data figures above):

| Experiment | 500 Hz | 1000 Hz | 2000 Hz | 4000 Hz |
|---|---|---|---|---|
| `1st_test` (`bpfo`/`bpfi`) | 18.4 / 46.0 | 18.7 / 43.9 | 16.5 / 36.2 | 10.6 / 30.9 |
| `2nd_test` (`bpfo`/`bpfi`) | 27.3 / 12.4 | 16.0 / **0.1** | 18.1 / **0.1** | 16.7 / 2.2 |
| `3rd_test` (`bpfo`/`bpfi`) | 46.7 / 29.2 | **67.6** / 50.2 | 38.1 / 35.5 | **0.8** / 4.0 |

`3rd_test`'s `bpfo_env` swings from 67.6 to 0.8 across the sweep, and `2nd_test`'s `bpfi_env`
collapses to 0.1. A feature whose separability moves by ~80x depending on a constant chosen by
convention is not something to hand to M3 as settled — the fixed-band approach, not the physics,
is what is fragile here (see Section 7).

**6d. Spectral kurtosis fails outright and is largely redundant.** Within-experiment pooled
F = **0.8** — no separability at all (Issue #63 attached a p-value to this: **p = 0.445**, the only
test in either family that fails to reach significance — `docs/uncertainty_quantification.md` §3).
Per experiment it reaches only 55.8 / 26.1 / 3.9, and it
declines toward Critical in `2nd_test` (SK 1.35 → 1.00). It also correlates **0.76** with
time-domain `kurtosis` in the pooled Degrading+Critical window (0.71 in `1st_test`), so what
little it carries largely restates a feature already in the set. Both the "adds independent
signal" and the "separates health states" tests fail. This is the clearest drop of the group.

**6e. Compute cost is explicitly *not* the reason.** Measured per 20,480-point snapshot: plain
FFT 0.38 ms, envelope spectrum 3.99 ms, spectral kurtosis 0.34 ms — 4.71 ms on top of the 1.56 ms
the time-domain features cost (~3x). Against `docs/PRD.md`'s 500 ms single-window serving target
that is negligible, so cost is not doing any work in this decision. Recorded so a future reader
does not assume the features were dropped as too expensive; they were dropped on evidence.

## 7. Decision, and what a future issue should try instead

**Dropped: `bpfo_amplitude`, `bpfi_amplitude`, `bpfo_amplitude_norm`, `bpfi_amplitude_norm`,
`bpfo_envelope_norm`, `bpfi_envelope_norm`, `spectral_kurtosis`.** None are added to
`FEATURE_COLUMNS`; `extract_experiment_features` is untouched; `data/processed/` parquet outputs
and their manifests are unchanged. All computation lives in `src/features/candidate_features.py`
with unit tests, kept rather than deleted per Issue #22's and Issue #23's shared instruction, so a
future issue can pick this up without rebuilding it.

Applying Issue #23's standard verbatim — *does it separate health states at least as strongly as
kurtosis, without being a restatement of an existing feature?* — the envelope features pass the
independence half (|r| ≤ 0.22) but fail the strength half (roughly half of `kurtosis` on the fair
within-experiment test), and are additionally unstable under Section 6c's sweep. Spectral
kurtosis fails both halves.

**This is not "frequency-domain analysis doesn't work on this dataset."** Section 5 is a clear
positive: the fault-matched defect frequency responds in all three experiments, exactly as
bearing theory predicts. What underperforms is specifically the *fixed-band, fixed-high-pass*
formulation. Concretely worth trying, in rough order of expected payoff:

1. **Kurtogram-based adaptive band selection** — choose the demodulation band per snapshot by
   maximising spectral kurtosis, instead of the fixed 1 kHz high-pass that Section 6c shows is
   the fragile part. This is the standard fix for exactly the instability observed here, and it
   would also give spectral kurtosis a role (band selector) that suits it better than the direct
   feature role it failed at in 6d.
2. **Defect-band energy as a ratio to its own baseline**, mirroring how `rms_ratio` generalises
   across experiments where raw `rms` does not — the per-experiment baseline offsets that
   confounded the pooled comparison in 6a are exactly what a baseline ratio would remove.
3. **Sidebands around BPFI** — inner-race defects are amplitude-modulated by shaft rotation as the
   defect passes through the load zone, producing BPFI ± `f_r` sidebands that a single band
   centred on BPFI does not capture. Relevant to `1st_test`, already the most promising case.

Not opened as issues now: M2's remaining scope is the feature pipeline itself, and M3 has enough
signal from the retained time-domain set to proceed. Recorded here so the next person starts from
the sweep and the physics check rather than from scratch.

## 8. Addendum (Issue #59): dominant-frequency shift — not evaluated, and why

`docs/eda_findings.md` Section 4's frequency-domain row named **three** M2 candidates: spectral
kurtosis, BPFI/BPFO-aligned energy, and **dominant-frequency shift** — tracking the location of
the dominant FFT peak (near BPFO/BPFI) across the bearing's life and asking whether it moves as
wear progresses. Sections 1-7 above evaluated the first two exhaustively. The third was never
carried into Issue #22 at all: it is absent from the feature table, the results, and the follow-up
list. That omission was not deliberate at the time, and the row's "Investigated in Issue #22 — not
adopted" verdict silently covered it anyway.

Issue #59 resolved this by examining the candidate on its merits rather than by running it. **It
is not evaluated, and should not be, on this dataset** — not because it would be expensive, but
because the measurement would not be interpretable. Three reasons, in increasing order of how
decisive they are:

**8a. The defect frequencies are kinematically pinned, so there is little to shift.** BPFO and
BPFI are fixed by geometry × shaft speed (Section 2), and this rig holds 2000 RPM constant across
the entire dataset. At constant speed the only wear-related mechanism that moves them is a change
in contact angle, entering through `cos θ`. Substituting into `compute_bearing_frequencies()`, a
contact-angle change far larger than bearing wear produces — 15.17° → 25° — moves BPFO by
**1.85 Hz (1.9 FFT bins)**. A more realistic 15.17° → 20° moves it **0.80 Hz (0.8 bins)**. The
effect being hunted is at or below the measurement's own resolution.

**8b. The confound is larger than the signal, and this dataset cannot remove it.** The rig is
belt-driven, so shaft speed wanders around its nominal 2000 RPM — the reason
`DEFECT_BAND_HALFWIDTH_HZ = 3.0` exists in the first place. A **1% speed wander moves BPFO by
2.36 Hz (2.4 bins)** and BPFI by 2.97 Hz (3.0 bins) — *larger than the extreme-wear shift in 8a*.
Separating the two requires a speed reference, and this dataset has none: every channel is an
accelerometer (`1st_test` 8 = 2 per bearing × 4 bearings; `2nd_test`/`3rd_test` 4 = 1 per bearing
× 4), with no tachometer or key-phasor channel to order-track against. So any drift the feature
reported could not be attributed to wear rather than to the belt. This is the decisive objection:
the feature would be *uninterpretable*, not merely weak.

**8c. The search band is too coarse to quantise a shift.** At 0.977 Hz bin width, the ±3 Hz
defect band spans only about 6-7 bins (the figure `DEFECT_BAND_HALFWIDTH_HZ`'s own comment cites),
so an in-band peak location can take roughly half a dozen distinct values — a coarser scale than
the effects in 8a, and it is undefined in early life, where the defect tone
has not yet risen out of the noise floor. That is exactly the region a baseline would have to be
established in.

**Cost is explicitly not the reason** (same framing as 6e). `compute_amplitude_spectrum()` already
exists; adding an in-band `argmax` and running it over all 9,464 files would cost roughly what
Section 4's plain-FFT pass did — a couple of minutes. It is skipped because the number it produced
would not answer the question, not because producing it is hard.

**What *is* worth pursuing is already recorded above.** The physically real version of "the
spectrum's dominant content moves as the defect grows" is not a shift of the defect *line* — it is
migration of the **resonance band** the impacts excite, as a spall widens and changes the impulse
shape. That is Section 7's follow-up item 1 (kurtogram-based adaptive band selection), already
recorded as the highest-expected-payoff direction. Dominant-frequency shift is therefore not a
missing fourth idea; it is a less well-posed statement of one already on the list.

**What would reopen this:** a dataset with a tachometer or key-phasor channel (enabling order
tracking, which normalises speed wander out and would make 8b disappear), or a variable-speed rig
where the frequencies genuinely move and order tracking is mandatory anyway. Neither describes
NASA IMS.
