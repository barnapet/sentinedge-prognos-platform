# Evaluation Protocol (Issue #69)

Written **before** any M3 model training happens — the third step of the M3 preparation
sequence (#65 → #67 → this document → #21 → model training + `rms_ratio` ablation), per
that sequence's own ordering. This is deliberate, not a formality: an evaluation protocol
chosen after seeing results is much easier to unconsciously bias toward whatever the model
already did well. Nothing below depends on any model having been trained, and nothing here
should change once one has.

**Scope.** This document is a protocol definition only — no model code, no training runs.
It is written to be detailed enough that model training (Step 4) can implement the split it
specifies directly, without further design decisions about how to divide the data. It does
**not** resolve Issue #21 (class-imbalance handling) — Section 4 commits to the metric #21's
own acceptance criteria will be measured against, but the technique (class weighting,
resampling, threshold-moving) remains #21's open decision.

**Input.** The dataset this protocol splits is `data/processed/training_dataset.parquet`,
produced by `src/features/build_training_dataset.py` (Issue #67,
`docs/training_dataset_versioning.md`) — the concatenation of all three experiments' feature
rows with `label`/`label_pre_override`/`override_applied` columns attached, grouped by its
`experiment` column (`1st_test` / `2nd_test` / `3rd_test`).

## 1. The protocol: leave-one-experiment-out (LOEO)

Three folds, one per experiment. In fold *i*, the held-out experiment's rows are the test
set; the other two experiments' rows, in full, are the training set. No row-level splitting
within an experiment — each experiment is wholly train or wholly test in a given fold, never
both.

| Fold | Train on | Test on |
|---|---|---|
| A | `2nd_test` + `3rd_test` | `1st_test` |
| B | `1st_test` + `3rd_test` | `2nd_test` |
| C | `1st_test` + `2nd_test` | `3rd_test` |

Concretely, for fold *i*: `train_df = training_dataset[training_dataset.experiment != held_out]`,
`test_df = training_dataset[training_dataset.experiment == held_out]`. Grouping by the
`experiment` column already present in the dataset (added for exactly this kind of
filtering, per Issue #43) is sufficient — no additional column or index is needed.

**Any fitted preprocessing (scaling, normalization, imputation, etc.) must be fit on that
fold's training rows only, then applied unchanged to the held-out rows.** Fitting on the
pooled dataset before splitting would leak the held-out experiment's own distribution into
its own evaluation — a smaller-scale version of the same leakage Section 2 describes for
labels. This is stated here because it is a property of the *split*, not a model choice, so
it belongs in the protocol rather than being left for Step 4 to reason out independently.

**Hyperparameter selection**, if Step 4's model needs any, is a modeling decision this
document does not resolve — but whatever mechanism is used (a nested split of a fold's two
training experiments, a fixed default, etc.) must not look at the held-out experiment's rows
under any circumstance. Doing so would silently turn a LOEO evaluation into the very
within-experiment leakage Section 2 exists to prevent.

## 2. Why LOEO, not a random or stratified split

`critical_multiple` — the Degrading→Critical boundary each row's label is thresholded
against — is derived **per experiment**, from that same experiment's own peak rolling
`rms_ratio` (`docs/eda_findings.md` §3): `1.932` for `1st_test`, `2.866` for `2nd_test`,
`3.049` for `3rd_test`. `docs/eda_findings.md` §3 itself flags this (added in #61, restated
in #62) as a **look-ahead, retrospective derivation, not a causal/online one**: the peak
used to set a given file's threshold is only knowable once that experiment's run is over,
which is legitimate for producing offline ground-truth labels but has a direct consequence
for evaluation design. Two distinct mechanisms follow from it, and a row-level split misses
both:

**a. Every row in an experiment shares one threshold.** Within a single experiment, `label`
is a deterministic function of `rms_ratio` and a handful of constants — `onset_multiple`
(global), `hysteresis_margin` (global), and `critical_multiple` (fixed per experiment). A
random or stratified split puts some of an experiment's rows in training and others in test,
but *both partitions share the identical threshold*. A model can therefore learn `1st_test`'s
own boundary (`~1.93x`) from its training rows and apply that exact value to `1st_test`'s
test rows — which is not evidence it has learned to locate a Degrading→Critical boundary in
general, only that it can recall one specific bearing's already-observed threshold. The three
experiments' thresholds span `1.93x`–`3.05x` (a `1.58x` spread, `docs/eda_findings.md` §3), so
recalling one experiment's threshold is not the same skill as inferring an unseen one.

**b. The threshold is computed from the whole run, including whatever a "test" split would
hold out.** `critical_multiple` uses `max(rms_ratio)` over an experiment's entire life. A row
placed in a random test partition can be the very file that peak came from, or can otherwise
be labeled using a statistic that summarizes it. This is look-ahead in the label construction
itself, not merely in features — a row-level split does not remove it, because the leaking
information (the aggregate peak) was baked into every row's label before any split is drawn,
train or test alike.

Both point to the same fix: only holding out an **entire experiment** evaluates whether a
model generalizes to a bearing whose own eventual threshold it has never seen or implicitly
learned, directly or in aggregate — which is what `docs/eda_findings.md` §3's own note says
matters for serving ("For a bearing still in operation there is no peak yet, so the formula
... cannot be evaluated at serving time at all"). With three experiments total, LOEO is the
only split that does this and still leaves anything to train on.

This is the same underlying issue — experiment-level heterogeneity — that motivated
`docs/frequency_domain_decision.md` §6a's within-experiment z-scoring: there, pooled
statistics were dominated by `3rd_test`'s 67% share of rows and needed per-experiment
normalization to ask a fair question. Here, the manifestation is a threshold rather than a
baseline offset, and the fix is a split rather than a normalization, but both trace back to
three experiments being genuinely different bearings, not interchangeable draws from one
population.

## 3. What this means for class counts per fold

With only three experiments, each fold's held-out set is exactly one experiment's `Critical`
count, and each fold's training set is the pooled total minus that count. Pooled Critical
total is **107** (`docs/eda_findings.md` §3).

| Held-out (test) | Test rows | Test Critical | Train rows | Train Critical |
|---|---|---|---|---|
| `1st_test` | 2,156 | **17** | 7,308 | 90 |
| `2nd_test` | 984 | **23** | 8,480 | 84 |
| `3rd_test` | 6,324 | **67** | 3,140 | 40 |

Stated plainly, as a known limitation of `n = 3`, not something to work around here: every
fold evaluates Critical-class performance on a double-digit-to-low-triple-digit sample, and
the `1st_test` fold in particular tests on only 17 Critical rows. `docs/uncertainty_quantification.md`
already treats these same three counts (17/23/67) as too small for some statistical claims
(e.g. it declines to interval the M4 serving fallback for the analogous reason — "an interval
computed from `n = 3` would be arithmetically expressible and substantively meaningless").
The same caution applies here: a single fold's Critical-class recall/precision is a noisy
estimate, and Section 5 addresses how that constrains aggregation rather than trying to
manufacture more precision than three bearings can provide.

## 4. Primary metric(s)

**Committed now, before training: per-class recall and precision, computed per fold, with
`Critical`-class recall reported as the single headline number.**

Why recall on `Critical`, specifically:

- `docs/eda_findings.md` §3 already ruled out plain accuracy ("Class imbalance is 81:1
  Normal-to-Critical... should report per-class recall rather than plain accuracy") — at this
  ratio a trivial always-Normal classifier scores >90% accuracy while catching zero failures.
- `docs/PRD.md` §6 frames the three classes as a "green/yellow/red on a plant floor" signal;
  §5 frames the secondary use case as an early warning for a plant engineer. In that framing,
  missing a `Critical` file (a false negative) is operationally worse than a false `Critical`
  alarm on a file that was actually `Degrading` — an early-warning system's primary job is not
  to miss the failure it exists to catch.
- Recall and precision are reported **separately**, not folded into F1 as the headline, so
  that a recall-favoring imbalance technique chosen later in #21 (e.g. aggressive class
  weighting) doesn't get to hide a precision collapse behind a single blended number.

Secondary, supporting metrics reported alongside the headline (not replacing it):

- **Macro-averaged F1** across all three classes, per fold — one comparable number across
  model variants (in particular, the `rms_ratio`-included vs. `rms_ratio`-excluded ablation
  Step 4 is required to run per Issue #67's Task 3).
- **The full 3×3 confusion matrix, per fold**, reported in full rather than only as derived
  summary statistics. With only three folds, the raw matrix carries more honest information
  than any single number derived from it — it shows what the model actually confused
  (`Degrading`↔`Critical` errors, which matter for an early-warning use case, look very
  different from `Normal`↔`Degrading` errors, even if they earn the same F1 penalty).

**Explicitly not committed to now: PR-AUC or any threshold-swept metric.** These presuppose a
calibrated probability output and a specific operating threshold — exactly what #21's
threshold-moving option, if chosen, would decide. If #21 lands on a probability-based
approach, a precision-recall curve becomes a natural *addition* alongside the recall/precision
commitment made here, not a replacement for it; committing to it now would presume a decision
#21 hasn't made.

**Relationship to Issue #21.** #21's own acceptance criteria include "confirms which
evaluation metric will be primary going forward" — this document is that confirmation. #21
still owns the open question this document does not answer: which imbalance-handling
technique (class weighting, resampling, or threshold-moving) to use. Whichever it picks
should be evaluated against the metrics committed to here, not against a metric chosen after
seeing which technique looks best under it.

## 5. Aggregation across folds

Three folds is too small a sample for a standard error or confidence interval to mean what
those tools normally mean — the same reasoning `docs/uncertainty_quantification.md` already
applies to the analogous `n = 3` case (the M4 serving fallback): an interval computed from
three folds would be arithmetically expressible and substantively meaningless, implying a
precision the dataset cannot support.

Reported instead, per metric (Critical-class recall/precision, macro-F1):

- **All three fold values individually**, not only their summary — e.g. "Critical recall:
  fold `1st_test` = _x_, fold `2nd_test` = _y_, fold `3rd_test` = _z_", never collapsed to a
  single number without also showing the three it came from.
- **Mean and range (min–max)** across the three, as the aggregate summary — range, not
  standard deviation, since a spread over three points is more honestly described by its two
  extremes than by a statistic that implies a distribution shape three samples cannot establish.
- **All three per-fold confusion matrices**, in full (Section 4) — for `n = 3`, this is more
  information-dense, and more honest, than any scalar aggregate.

**If one fold's result differs sharply from the other two, that must be stated explicitly,
not averaged away.** `docs/eda_findings.md` §2 already documents that `1st_test`'s inner-race
failure is impulsive (kurtosis-led) while `2nd_test`/`3rd_test`'s outer-race failures are
amplitude-led (RMS-led) — a real, physically-grounded difference between failure modes, not
noise. A mean across three heterogeneous experiments can hide a genuine failure-mode-dependent
weakness (e.g., a model that generalizes well across the two outer-race experiments but fails
on the held-out inner-race one) behind an unremarkable-looking average. The three individual
values (previous bullet) exist precisely so this doesn't happen unnoticed.

## 6. Explicit non-goals

To keep M3's write-up from overreaching what three experiments can support:

- **This protocol answers "does this generalize to an unseen bearing/failure trajectory,"
  not "does this generalize to unseen bearings of this exact failure mode."** Inner-race
  failure appears in exactly one experiment (`1st_test`); outer-race failure appears in two
  (`2nd_test`, `3rd_test`). `n = 1` and `n = 2` per failure mode cannot separate "the model
  generalizes across this failure mode" from "the model happens to fit this one/these two
  bearings' idiosyncrasies" — there is no fourth inner-race or third outer-race experiment to
  check against. A strong LOEO result is evidence the model generalizes across the three
  *specific* trajectories in this dataset; it is not evidence it generalizes across the
  *population* of inner- or outer-race failures in general.
- **This protocol does not re-validate the labeling rule itself.** Whether `ONSET_MULTIPLE`,
  the hysteresis margin, the rig-shutdown override, or the geometric-midpoint
  `critical_multiple` derivation are the *right* rules was already settled in #9/#10/#20, with
  their look-ahead property and sensitivity separately addressed in #61/#62/#63. LOEO
  evaluates whether a downstream classifier generalizes given labels it is told to trust — it
  is silent on whether those labels are themselves correct.
- **This protocol says nothing about generalization across operating conditions.** All three
  experiments ran at one shaft speed and one radial load (`docs/PRD.md` §6); §12 already names
  "dataset only covers one operating condition" as a documented MVP limitation. LOEO varies
  the bearing and failure trajectory, not the operating condition, and cannot speak to the
  latter.
- **The reported recall/precision numbers are not a claim about a deployed fleet's expected
  performance.** Three bearings from one lab rig are not a statistically representative sample
  of bearings in the field; M3's write-up should present LOEO results as evidence about *this
  dataset's three trajectories*, not as a forecast of real-world failure-detection rates.

## Decision points flagged, not silently resolved

Two choices in this document had no single obviously-correct answer and are recorded here
rather than left implicit, per this repo's own convention for genuinely open calls
(`docs/frequency_domain_decision.md`, `docs/skewness_crestfactor_decision.md`):

- **Headline metric = `Critical`-class recall, not a blended score.** F1 or a macro-average
  would compress three classes (and, per Section 5, three folds) into one number faster, but
  would also be the first thing to hide exactly the failure mode (missed `Critical` files)
  this project's early-warning framing cares most about. Recall was chosen over precision as
  the *single* headline (both are still reported) because a missed failure is judged more
  costly than a false alarm in this use case (Section 4); this is a judgment call about
  relative cost, stated explicitly rather than left for a reader to infer from a metric choice.
- **Range, not standard deviation, for cross-fold spread.** A standard deviation is the more
  familiar statistic, but computing one over three points implies a sampling distribution
  three points cannot characterize. Range says the same thing (how much the folds disagree)
  without borrowing statistical machinery that presumes a larger sample.
