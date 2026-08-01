# Class-Imbalance Handling Decision (Issue #21)

Decision note accompanying `src/training/`. The dataset is **81:1 Normal:Critical**
pooled (`docs/eda_findings.md` §3), which Issue #21 flagged during M1-EDA as something
that would bias any M3 classifier toward `Normal` and undermine recall on the class the
project exists to catch.

Method follows `docs/evaluation_protocol.md` (Issue #69) exactly: leave-one-experiment-out
(LOEO), three folds, scored on that document's already-committed metrics — per-class
recall and precision headlined by `Critical` recall, macro-F1, and full confusion
matrices. **The metric was not re-decided here.** #69 was written before any model
existed specifically so a metric could not be chosen to flatter whichever technique won,
and this document is bound by it.

**Decision up front: `class_weight='balanced'`, adopted as the M3 default — but on
parsimony and worst-case grounds, not because it won the headline metric.** It did not.
On the current feature set no strategy, including the untreated control, is
distinguishable from the others on `Critical` recall. That null result is the most
important finding here and Section 3 states it before Section 5 gets to the decision.

## 1. What was compared

Five arms, holding everything except the imbalance treatment fixed. Issue #21 named three
directions (class weighting, resampling, threshold-moving/cost-sensitive); all three are
represented, plus an untreated control — **without a control there is no way to tell
whether any handling helped at all**, and the control turned out to matter more than
expected.

| Arm | What it does |
|---|---|
| `none` | Control: no imbalance handling |
| `class_weight_balanced` | sklearn `class_weight='balanced'` — loss weights inversely proportional to class frequency |
| `random_oversample` | Duplicate minority rows up to the majority count, training fold only |
| `random_undersample` | Subsample majority rows down to the minority count, training fold only |
| `prior_correction` | Untreated model; predicted probabilities divided by training-fold class priors before `argmax` |

**Baseline model: standardised features + multinomial logistic regression**
(`src/training/imbalance.py:make_baseline_model`). This is the *instrument* for the
comparison, not a candidate for the final M3 model — Issue #21's own scope note. Logistic
regression because it responds directly and predictably to both weighting and resampling
(so the comparison measures the treatment, not a model's idiosyncrasies), runs all
5 arms × 3 folds in seconds, and emits probabilities the `prior_correction` arm needs.

## 2. Dependency decisions

**Added: `scikit-learn==1.9.0`.** Genuinely warranted — M3 is the milestone that needs a
classifier, and the estimator, the `Pipeline`/`StandardScaler` machinery that enforces
§1's fit-on-training-rows-only rule, and the metric implementations are not a few lines of
numpy.

**Not added: `imbalanced-learn`.** Issue #21 named "SMOTE or simple duplication" as the
resampling options. Plain duplication was chosen, so the dependency isn't needed, and
that choice has its own reasoning beyond dependency count: SMOTE synthesises new minority
points by interpolating between nearest neighbours, and with **17 `Critical` rows in
`1st_test`** those neighbours are sparse points strung along a degradation trajectory in
time, not samples from a dense cluster. Interpolating between them would invent vibration
states *between distinct moments in a bearing's life* and present them as observations.
Plain duplication makes no such claim about the space between observed points. This
mirrors `docs/uncertainty_quantification.md`'s reasoning for implementing Holm correction
in numpy rather than adding `statsmodels` for one function.

## 3. Results — and the null result that dominates them

Primary feature set (`rms`, `rms_ratio`, `kurtosis`, `skewness`, `skewness_smoothed`).
Per-fold values given in full per `docs/evaluation_protocol.md` §5, which forbids
reporting a mean without the three values behind it. Mean and range, never a standard
deviation — three points cannot characterise a sampling distribution.

**`Critical`-class recall (headline metric):**

| Arm | `1st_test` | `2nd_test` | `3rd_test` | mean | range |
|---|---|---|---|---|---|
| `none` | 0.118 | 0.913 | 1.000 | 0.677 | 0.882 |
| `class_weight_balanced` | 0.059 | 0.913 | 1.000 | 0.657 | 0.941 |
| `random_oversample` | 0.059 | 0.913 | 1.000 | 0.657 | 0.941 |
| `random_undersample` | 0.118 | 0.913 | 1.000 | 0.677 | 0.882 |
| `prior_correction` | 0.176 | 0.913 | 1.000 | 0.697 | 0.824 |

**The headline metric does not discriminate between the arms, and this is the central
finding.** Two observations make that concrete:

- On the `2nd_test` and `3rd_test` folds, **every arm scores identically** — 0.913 and
  1.000 — including the untreated control. Imbalance handling changes `Critical` recall on
  those folds not at all.
- The entire spread in the mean column (0.657–0.697) comes from the `1st_test` fold, where
  it is the difference between catching **1, 2, or 3 Critical rows out of 17**. A
  0.04 swing in a headline mean that traces to ±2 rows on one fold is noise, not evidence.

Reading a winner off that column would be exactly the failure this issue was supposed to
avoid — confirming an expected result rather than measuring one.

**`Critical`-class precision** (reported separately, per #69 §4, so a recall-favouring
technique cannot hide a precision collapse):

| Arm | `1st_test` | `2nd_test` | `3rd_test` | mean |
|---|---|---|---|---|
| `none` | 1.000 | 0.840 | 0.944 | **0.928** |
| `class_weight_balanced` | 1.000 | 0.750 | 0.859 | 0.870 |
| `random_oversample` | 1.000 | 0.750 | 0.859 | 0.870 |
| `random_undersample` | 1.000 | 0.724 | 0.838 | 0.854 |
| `prior_correction` | 1.000 | 0.724 | 0.779 | 0.834 |

Precision falls monotonically with treatment aggressiveness, as expected. The control is
the most precise arm.

**Where the arms *do* differ: the `Degrading` class.** Mean recall across folds:

| Arm | `Degrading` recall (mean) | `2nd_test` specifically |
|---|---|---|
| `class_weight_balanced` | **0.955** | 0.719 → **0.977** |
| `random_oversample` | 0.936 | 0.919 |
| `prior_correction` | 0.919 | 0.948 |
| `none` | 0.893 | 0.719 |
| `random_undersample` | 0.745 | 0.648 |

This is a real, non-noise difference on an informative fold: in `2nd_test` the control
misclassifies 83 `Degrading` rows as `Normal`, and `class_weight='balanced'` misclassifies
none of them, recovering 80 of them as correct `Degrading` predictions (223/310 →
303/310). For an early-warning use case, `Degrading` is the amber
light — the class that provides lead time — so this is not an incidental improvement to a
class nobody cares about.

**Macro-F1** (supporting metric): `none` 0.685, `prior_correction` 0.683,
`class_weight_balanced` 0.678, `random_oversample` 0.668, `random_undersample` 0.616.
Again tightly clustered except for undersampling.

## 4. The `1st_test` fold is unstable, and not because of class imbalance

Issue #21's brief asked for this to be called out if it occurred. It did, more severely
than anticipated, and the cause is worth stating precisely because it is *not* the problem
this issue is about.

Every arm collapses on the `1st_test` fold: macro-F1 0.15–0.22, versus 0.82–0.98 on the
other two. The confusion matrices show why — with `1st_test` held out, the control
predicts `Degrading` for **1,744 of its 1,906 true `Normal` rows** (`Normal` recall
0.085). The model is not failing to find the minority class; it is failing to recognise
the majority one.

**Cause: the raw `rms` column's absolute scale is experiment-specific.** `1st_test`'s
*minimum* raw RMS (0.1289 g) is higher than either training experiment's *mean*
(`2nd_test` 0.1061, `3rd_test` 0.0724). The `StandardScaler`, correctly fitted on the two
training experiments only, therefore maps every single `1st_test` row into the
high-RMS region of the training distribution. A diagnostic run with the raw `rms` column
dropped more than doubles that fold's macro-F1 (0.192 → 0.493), confirming the mechanism.

Two consequences, both deliberately left open here:

- **`1st_test`'s `Critical` recall is not an informative comparison signal.** It is 1–3
  rows out of 17, on a fold where the model's whole decision boundary is displaced. No
  arm's ranking should rest on it, and none in Section 5 does.
- **This is a feature-selection problem for Step 4, not a class-imbalance problem.** No
  weighting or resampling scheme fixes a feature whose absolute scale doesn't transfer
  between bearings. Recorded here as a hand-off, not resolved: `docs/eda_findings.md` §2
  already documents `1st_test`'s inner-race failure as impulsive (kurtosis-led) rather
  than amplitude-led, which is the physical reason its RMS distribution sits where it
  does. This is exactly the failure-mode-dependent weakness `docs/evaluation_protocol.md`
  §5 warned must be stated rather than averaged away.

## 5. Decision: `class_weight='balanced'`

Adopted as the M3 default. The reasoning is deliberately modest, because the evidence is:

1. **It is never the worst arm, and the two arms it loses to lose elsewhere.** `none` has
   better precision but the worst `Degrading` recall of the non-undersampling arms;
   `prior_correction` has nominally better `Critical` recall but the worst precision of
   all five.
2. **It gives the largest real improvement the comparison found** — `Degrading` recall,
   0.893 → 0.955 mean, driven by a 0.719 → 0.977 gain on `2nd_test` — at a
   precision cost (0.928 → 0.870) that is visible and bounded rather than a collapse.
3. **It costs nothing structurally.** One constructor argument. No extra rows, no
   synthetic points, no RNG seed to carry, no post-hoc decision rule, no constraint on
   Step 4's model choice. Every other non-control arm costs at least one of those.
4. **It composes with whatever Step 4 picks.** `class_weight` is supported across
   sklearn's classifier family, so this decision does not have to be revisited if the
   final M3 model is not logistic regression.

**Rejected: `random_undersample`.** Clearly worst — lowest macro-F1 (0.616) and lowest
`Degrading` recall (0.745) of all five arms — and the mechanism is not subtle: with 107
pooled `Critical` rows, it trains each fold on **270 rows out of 7,308**, discarding ~96%
of the data. There is no configuration in which that trade looks good on this dataset.

**Not adopted, but not rejected on the merits: `random_oversample`.** It produced
results numerically near-identical to `class_weight_balanced` in every fold and both
feature configurations — unsurprising, since duplicating a row and upweighting it are
close to the same operation for a linear model's loss. Given equivalent results, the arm
with less machinery wins; this is a parsimony call, not a performance one.

**Deferred, with evidence, to Step 4: `prior_correction`.** It posted the best `Critical`
recall in both feature configurations tested, and the margin becomes substantial rather
than noise when `rms_ratio` is removed (see Section 6). It is not adopted *now* for three
reasons: its advantage on the current feature set is within the ±2-row noise band of
Section 3; it has the worst `Critical` precision of all five arms; and
`docs/evaluation_protocol.md` §4 explicitly declined to commit to threshold-swept or
probability-calibration-dependent evaluation, noting that if #21 landed on a
probability-based approach then precision-recall curves become a required *addition*.
Adopting it would therefore also be adopting a reporting obligation and a constraint that
Step 4's model must emit well-calibrated probabilities. Flagged as the first thing to
re-test once the feature set is settled — not silently dropped.

## 6. Robustness check on the ranking (does *not* resolve the Step 4 ablation)

Because Section 3's null result could have been an artifact of `rms_ratio` — which is both
the strongest feature and the signal the labels are thresholded from — the same comparison
was re-run with `rms_ratio` removed, purely to check whether the *ranking* of arms is
stable. **This does not resolve, pre-empt, or take a position on the `rms_ratio`
label-leakage question, which remains Step 4's (Issue #67 Task 3).** It asks only: is the
decision in Section 5 an artifact of one feature?

| Arm | `Critical` recall (mean) | `Critical` precision (mean) |
|---|---|---|
| `prior_correction` | **0.966** | 0.587 |
| `class_weight_balanced` | 0.892 | 0.603 |
| `random_oversample` | 0.892 | 0.604 |
| `none` | 0.872 | 0.674 |
| `random_undersample` | 0.822 | 0.651 |

What holds across both configurations, and is therefore not an artifact of one feature:

- `random_undersample` is worst or near-worst on the headline in both — the rejection in
  Section 5 is robust.
- `class_weight_balanced` and `random_oversample` are numerically near-identical in both —
  the equivalence claim in Section 5 is robust.
- `prior_correction` leads the headline in both — the deferral in Section 5 is a genuine
  open question, not a dismissal.
- `none` has the best precision in both.

What changes: the metric becomes discriminating. Without `rms_ratio`, the spread in
`Critical` recall is 0.822–0.966 rather than 0.657–0.697, and the differences are no
longer single-row artifacts. **This is itself the observation Issue #21's brief asked to
be flagged if relevant:** the reason imbalance handling looks like it barely matters in
Section 3 is at least partly that `rms_ratio` is doing the work. If Step 4's ablation
removes it, imbalance handling matters considerably more, and `prior_correction` becomes
the arm to beat. Recorded, not acted on.

## 7. What this does not settle

- **The final M3 model.** The logistic-regression baseline here is a measuring instrument.
  A different model class may interact differently with class weighting, and Section 5's
  decision was chosen partly for composing well with that uncertainty.
- **The `rms_ratio` ablation** (Issue #67 Task 3) — Section 6 is a robustness check on
  this document's own decision, not a verdict on that question.
- **The `1st_test` feature-scale problem** (Section 4) — diagnosed, handed to Step 4.
- **Whether `Critical` precision at ~0.87 is operationally acceptable.** That is a
  cost-of-false-alarm question `docs/PRD.md` does not currently quantify, and this
  document does not invent a threshold for it.

## Reproducing

```
python -m src.features.build_training_dataset   # Issue #67, writes training_dataset.parquet
python -m src.training.compare_imbalance        # prints the tables in Section 3
```

Deterministic: every arm uses a fixed `random_state`, and no hyperparameter is tuned
anywhere (see Section 8 on why).

## 8. How tuning leakage was avoided

`docs/evaluation_protocol.md` §1 requires that hyperparameter selection never see the
held-out experiment. This comparison avoids that risk by removing the mechanism entirely
rather than managing it: **no hyperparameters are tuned, per-fold or otherwise.** Every
arm uses the identical fixed model (`BASELINE_MODEL_PARAMS` — default regularisation,
fixed `random_state`, `max_iter` raised only far enough for the solver to converge, which
is a numerical setting rather than a performance-tuned one). The only quantity that
differs between arms is the imbalance treatment, which makes this a clean one-variable
comparison and leaves nothing that *could* be tuned on one fold and tested on another.

Three further leakage guards, each enforced in code and pinned by a test in
`tests/test_training.py`:

- **`StandardScaler` lives inside the `Pipeline`**, so it is fitted on each fold's
  training rows and only transforms the held-out rows. A scaler fitted on the pooled
  dataset would leak the held-out experiment's own distribution into its evaluation.
- **Resampling is applied to training rows only.** The held-out experiment is never
  over- or undersampled; its class support is identical across all five arms.
- **`prior_correction` reads class priors from the training fold only.** Using the
  held-out experiment's class frequencies would leak the very quantity being predicted.
