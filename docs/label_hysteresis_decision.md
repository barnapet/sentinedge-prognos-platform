# Onset-Boundary Hysteresis Decision (Issue #20)

## 1. The problem, quantified

`docs/eda_findings.md` Section 5 flagged that `1st_test` reverts to `Normal` 73 times after
degradation has already begun. Reproduced directly against `src/labeling.py`'s actual output
(cached `rms_ratio` from `01_vibration_signal_evolution.ipynb`/`02_health_state_labeling.ipynb`,
`ONSET_MULTIPLE=1.3`, `critical_multiple=1.931`):

| Experiment | Flapping window (file index) | Files affected | `rms_ratio` range in window | Transitions | Reverts to `Normal` |
|---|---|---|---|---|---|
| `1st_test` | 1906–1999 (94 files) | 73 | 1.264 – 1.309 | 8 | 73 |
| `2nd_test` | 788–793 (6 files) | 6 | ~1.299 – 1.301 | 2 | 6 |
| `3rd_test` | none | 0 | — | 0 | 0 |

**`2nd_test` was not previously known to flap.** It wasn't flagged in `docs/eda_findings.md`
(that document's Section 5 only names `1st_test`), but the same reproduction shows a smaller,
6-file blip at its own onset boundary: `Degrading` from file 651, a brief revert to `Normal` at
788–793, then back to `Degrading` at 794 for good. Same mechanism, much smaller scale — the ratio
only grazes back under `1.3` briefly rather than drifting 90+ files through a shallow dip like
`1st_test`. Included here since the hysteresis fix and its generalization check (Task item 4)
both depend on knowing this existed. `3rd_test` genuinely has zero reverts at either boundary —
both its `Normal→Degrading` and `Degrading→Critical` transitions are single, clean crossings.

The deepest reversal-causing dip in either affected experiment is `1st_test`'s: `rms_ratio` falls
to `1.264`, **0.036 below** `ONSET_MULTIPLE=1.3`, before climbing back and crossing for good. This
number anchors the margin choice below.

## 2. Mechanism chosen: two thresholds (entry / exit), not N-consecutive-file confirmation

Issue #20 offered two directions. Both were implemented and compared against the real data before
picking one.

**N-consecutive-file confirmation (rejected).** Debouncing by requiring `N` consecutive files past
a boundary before confirming a state change was tried first. Problem: `1st_test`'s pre-onset noise
happens to include an 8-file consecutive `Degrading`-by-threshold streak (files 1908–1915) that
isn't the real onset — so `N=9` is needed to reject it, but `N=9` also happens to delay confirmed
onset all the way to file 1996 (vs. the true 1906), because it's the *next* sustained run that
first reaches 9 consecutive files. Worse: `1st_test`'s entire `Critical` region is only **17 files
long** (all consecutive, no gaps) — any `N` at or above that swallows the Critical class for that
experiment entirely (verified: `N=20` produces zero confirmed `Critical` labels for `1st_test`).
Given `1st_test` already has the fewest Critical examples of the three experiments (17 vs. 23 and
67), this is a fragile mechanism whose safe operating range depends on a coincidental run-length
in the noise, not a property of the signal being thresholded. Rejected.

**Two thresholds — entry unchanged, exit requires clearing a margin (adopted).** For each
boundary, moving *up* a state still fires immediately at the existing locked constant (unchanged:
`ONSET_MULTIPLE=1.3` for Normal→Degrading, `critical_multiple` for Degrading→Critical). Moving
*down* a state requires the ratio to fall not just back below that same constant, but past
`constant - hysteresis_margin`. This is a genuine dead-band around the existing, unmodified
threshold, not a new threshold value — Issue #9's locked `ONSET_MULTIPLE` is never read or
compared differently for the upward direction.

`hysteresis_margin = 0.05`, chosen with headroom above `1st_test`'s observed 0.036 worst-case dip.
Swept `0.00` (no hysteresis) through `0.10` against all three experiments' real cached data: `0.04`
is the minimum that fully suppresses `1st_test`'s reverts, `0.02` is enough for `2nd_test`'s
smaller blip, and results are identical for `0.04` through `0.10` in both experiments — the fix
isn't sensitive to the exact value once past the observed dip depth, so `0.05` isn't a knife-edge
choice. See Section 5 for the one caveat on how this number was derived.

Downward moves relax **one rank at a time** (Critical→Degrading, not Critical→Normal in one
step), gated by the margin at whichever boundary is being re-crossed. This has a physical
justification, not just a convenient one: `rms_ratio` is continuous, so a real, gradual recovery
from Critical-range values down to Normal-range values necessarily passes through the
Degrading-range numbers on the way — a single file jumping across both bands in one step doesn't
happen from ordinary signal drift. The one place that *does* happen is the already-documented
rig-shutdown collapse (`docs/eda_findings.md` Section 2's end-of-life artifact), and that's
deliberately left to the existing override mechanism (which inspects raw RMS, not `rms_ratio`) —
confirmed to still fire identically with hysteresis in place (Section 3). Upward moves are not
rank-limited: a sudden severe fault jumping straight from a Normal-range to a Critical-range ratio
in one file is a real possibility this shouldn't delay.

Implementation: `_hysteresis_ranks()` in `src/labeling.py`, called from `assign_labels()` in place
of the old vectorized `np.where` threshold rule. New parameter `hysteresis_margin` (default
`HYSTERESIS_MARGIN = 0.05`) on `assign_labels()`; `ONSET_MULTIPLE` and the per-experiment
`critical_multiple` derivation are untouched.

## 3. Interaction with the rig-shutdown override

The override (forces `Critical` on a raw-RMS collapse after a Critical run) runs on top of the
hysteresis-confirmed label, exactly as it ran on top of the plain threshold label before. Verified
against all three experiments' real cached data: **override firing indices are identical with and
without hysteresis**, in all three experiments. This is expected — the override triggers on raw
RMS collapsing relative to the recent Critical run's own mean, which hysteresis doesn't touch.

## 4. Effect on label counts and timing

Checked directly against `src/labeling.py`'s output for all three experiments (not assumed):

| Experiment | Onset file idx (before/after) | Critical file idx (before/after) | Files relabeled vs. no-hysteresis | Degrading count (before → after) |
|---|---|---|---|---|
| `1st_test` | 1906 → 1906 (unchanged) | 2139 → 2139 (unchanged) | 73 (all 1907–1998, Normal→Degrading) | 160 → 233 |
| `2nd_test` | 651 → 651 (unchanged) | 961 → 961 (unchanged) | 6 (788–793, Normal→Degrading) | 304 → 310 |
| `3rd_test` | 6158 → 6158 (unchanged) | 6257 → 6257 (unchanged) | 0 | 99 → 99 |

**Onset and Critical detection timing are exactly unchanged in all three experiments** — hysteresis
only affects files that were incorrectly bouncing back to `Normal`; it does not delay the initial
crossing. This means the `Critical` lead-time figures in `docs/eda_findings.md` Section 3
(9.5h / 3.7h / 11.0h) are unaffected: the fix only reclassifies files that were already inside the
Degrading/Critical span as such, rather than moving the span's boundaries. `Degrading` counts
increase (only for the two affected experiments) because files previously mislabeled `Normal`
mid-degradation are now correctly counted as `Degrading` — this is the fix working as intended,
not a side effect requiring separate justification.

## 5. Open question — flagged, not resolved here

**The `hysteresis_margin = 0.05` value is calibrated to this dataset's observed noise, the same
way `ONSET_MULTIPLE` and `critical_multiple` are** — it is not a universal constant. It was derived
from the single deepest dip observed across the two affected experiments (`1st_test`'s `0.036`).
If a future bearing/dataset showed a deeper transient dip at its onset boundary, `0.05` might not
be enough and would need re-deriving the same way, against that data. This is flagged rather than
generalized further because there is no evidence in this dataset of what a "worse" dip looks like
— extrapolating a safety factor beyond what's actually been observed would be guessing, not
deriving.

**Not in scope for this issue, but worth flagging separately per Issue #20's own constraint:** the
investigation did not surface any evidence that `ONSET_MULTIPLE=1.3` itself (the value, not the
transition logic around it) needs revisiting — every affected file's `rms_ratio` sits within a few
percent of `1.3`, consistent with `docs/eda_findings.md`'s existing description of the flapping as
"ambiguous in the signal, not miscomputed by the rule." No separate proposal is made here to
reopen Issue #9's locked constant.

## 6. Downstream dependencies

This closes the labeling-side half of the M2 blocker chain from `docs/eda_findings.md` Section 5.
`docs/feature_windowing_decision.md` (Issue #40) already notes that Issue #20 "touches the same
rolling-RMS computation" as the windowing decision — confirmed here to be a non-issue: this fix
touches only the label-assignment step downstream of `rms_ratio`, not how `rms_ratio` itself is
computed (`ROLLING_WINDOW=10`, `min_periods=1`, unchanged). Issue #41 (core feature-extraction
module) and Issue #23 (skewness/crest factor redundancy) both consume labels produced by
`assign_labels()` and are unaffected beyond now getting more accurate `Degrading` labels for
`1st_test`/`2nd_test`.
