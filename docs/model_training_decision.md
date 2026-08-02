# M3 Baseline Model, `rms_ratio` Ablation, and the `1st_test` Fold (Issue #72)

Decision note accompanying `src/training/train_baseline.py`. Fourth and final step of the
M3 preparation sequence (#65 → #67 → #69 → #21 → this document).

Method is bound by two prior documents and re-decides neither. `docs/evaluation_protocol.md`
(Issue #69) fixes the split (leave-one-experiment-out) and the metrics (per-class recall and
precision, headlined by `Critical` recall; macro-F1 and full confusion matrices as support;
mean and range across folds, never a standard deviation). `docs/class_imbalance_decision.md`
(Issue #21) fixes the imbalance treatment (`class_weight='balanced'`) and the model family
(standardised features + multinomial logistic regression). Both were settled before this
run, on purpose, so that neither could be chosen to flatter whatever the ablation turned up.

**Decision up front: keep the full M2 feature set as the M3 baseline, and report the
`1st_test` fold as an explicitly unresolved limitation rather than engineering it away.**
The evidence for both halves is below. The more important content of this document is not
the decision but the diagnosis: the `1st_test` fold carries **two independent failures**,
only one of which is the scale problem #21 handed forward, and the second one is not
fixable by any feature transformation.

## 1. Headline result, stated plainly before anything else

Under LOEO, with `class_weight='balanced'` and the unchanged M2 feature set:

| Metric | `1st_test` | `2nd_test` | `3rd_test` | mean | range |
|---|---|---|---|---|---|
| **`Critical` recall (headline)** | **0.059** | **0.913** | **1.000** | 0.657 | 0.941 |
| `Critical` precision | 1.000 | 0.750 | 0.859 | 0.870 | 0.250 |
| `Normal` recall | 0.074 | 1.000 | 0.999 | 0.691 | 0.926 |
| Macro-F1 | 0.152 | 0.936 | 0.945 | 0.678 | 0.793 |

**The mean is not a usable summary of this model, and should not be quoted as one.**
`docs/evaluation_protocol.md` §5 anticipated exactly this case — "if one fold's result
differs sharply from the other two, that must be stated explicitly, not averaged away" —
and it has occurred. A mean `Critical` recall of 0.657 describes no fold: two folds are at
0.913–1.000 and one is at 0.059. The honest one-line statement of this model's capability
is in §6.

Two of these cells are more fragile than they look and are flagged here rather than left to
be misread:

- **`1st_test`'s `Critical` precision of 1.000 is one prediction.** The model predicted
  `Critical` exactly once on that fold and happened to be right. A precision computed from
  a single prediction is not evidence of precision, and it should not be read as the one
  encouraging number on an otherwise failed fold.
- **`3rd_test`'s `Critical` recall of 1.000 is 67 rows** out of 6,324, and `2nd_test`'s
  0.913 is 21 of 23. `docs/evaluation_protocol.md` §3 already recorded that every fold
  tests `Critical` on a double- to low-triple-digit sample.

These reproduce `docs/class_imbalance_decision.md` §3's `class_weight_balanced` row
(0.059 / 0.913 / 1.000) exactly, which is the intended consistency check: this step changed
the comparison axis, not the harness.

## 2. The `rms_ratio` ablation (Issue #67 Task 3, Issue #72 Task 2)

This is the answer to the question `docs/training_dataset_versioning.md` §3 explicitly declined
to resolve when it joined `rms_ratio` into `training_dataset.parquet` unmodified: "whether that
circularity should exclude it from a trained model." Run as a first-class comparison, not a
footnote. Four feature-set configurations, all with the imbalance treatment, model family, and
hyperparameters held fixed — the only thing that varies is which columns the model sees.

| Configuration | Columns |
|---|---|
| `full` | `rms`, `rms_ratio`, `kurtosis`, `skewness`, `skewness_smoothed` |
| `no_rms_ratio` | `full` minus `rms_ratio` — the required ablation |
| `no_raw_rms` | `full` minus `rms` — the candidate scale fix (§4) |
| `kurtosis_skewness_only` | neither RMS-derived feature — the floor |

**`Critical`-class recall (headline metric), per fold and aggregated:**

| Configuration | `1st_test` | `2nd_test` | `3rd_test` | mean | range |
|---|---|---|---|---|---|
| `full` | 0.059 | 0.913 | 1.000 | 0.657 | 0.941 |
| `no_rms_ratio` | 0.941 | 0.913 | 0.821 | **0.892** | 0.120 |
| `no_raw_rms` | 0.000 | 1.000 | 1.000 | 0.667 | 1.000 |
| `kurtosis_skewness_only` | 1.000 | 0.522 | 0.373 | 0.632 | 0.627 |

**Read on the headline metric alone, removing `rms_ratio` improves the model — mean
`Critical` recall rises from 0.657 to 0.892. That reading is wrong, and the rest of this
section is why.** This is precisely the failure mode `docs/evaluation_protocol.md` §4
pre-empted when it insisted recall and precision be reported separately rather than blended
into F1: a configuration that predicts the alarm class more freely will always look better
on recall alone.

**`Critical`-class precision, and `Normal`-class recall, for the same runs:**

| Configuration | `Critical` precision (mean) | `Normal` recall (mean) | Macro-F1 (mean) |
|---|---|---|---|
| `full` | **0.870** | 0.691 | 0.678 |
| `no_rms_ratio` | 0.603 | 0.666 | 0.599 |
| `no_raw_rms` | 0.358 | **0.884** | **0.679** |
| `kurtosis_skewness_only` | 0.139 | 0.823 | 0.368 |

The ablation's recall gain is bought with a precision collapse (0.870 → 0.603) and a
macro-F1 drop (0.678 → 0.599). On the `1st_test` fold the mechanism is visible directly in
the confusion matrix (rows = true, columns = predicted, order `Normal`/`Degrading`/`Critical`):

```
no_rms_ratio, held out 1st_test:  [[   0, 1876,   30],
                                   [   0,  166,   67],
                                   [   0,    1,   16]]
```

**The model predicts `Normal` for zero of 1,906 truly-`Normal` rows.** It catches 16 of 17
`Critical` rows because it has stopped calling anything `Normal` at all — 30 `Normal` rows
and 67 `Degrading` rows are also called `Critical`, giving a `Critical` precision of
16/113 = 0.142. A classifier that never says "healthy" trivially achieves high recall on
the alarm class, and that is what the 0.941 is.

**What the ablation actually shows, stated as the finding Issue #72 asked for:**

`rms_ratio` is not doing *all* the discriminative work — without it, the model still reaches
macro-F1 0.828 and 0.839 on `2nd_test` and `3rd_test`, which is degraded but far from
chance. So the answer to "can the model distinguish health states without `rms_ratio`" is
**yes on two of three folds, at a real cost, and no on the third**. What `rms_ratio`
supplies is not discrimination as such but a *calibrated place to put the boundary*:
removing it does not blind the model, it makes the model over-alarm. The
`kurtosis_skewness_only` floor confirms the direction — with neither RMS-derived feature,
`Critical` precision falls to 0.139 and macro-F1 to 0.368, which is the genuinely
near-chance configuration.

This settles the question `docs/class_imbalance_decision.md` §6 left open in its favour:
§6 observed that removing `rms_ratio` widened the spread between imbalance arms and
inferred that `rms_ratio` was masking their differences. That inference holds, but the
mechanism is now identified — the spread widens because every arm is pushed toward
over-alarming, not because the arms become better distinguished on merit.

## 3. The `1st_test` fold: two stacked failures, not one

`docs/class_imbalance_decision.md` §4 diagnosed one cause (raw-`rms` amplitude scale) and
handed it forward. Decomposing the fold shows the scale problem is real but accounts for
only half of what goes wrong, and the two halves damage *different classes*.

Holding everything else fixed and removing only raw `rms`:

| `1st_test`, held out | `Normal` recall | `Critical` recall | Macro-F1 |
|---|---|---|---|
| `full` | 0.074 | 0.059 | 0.152 |
| `no_raw_rms` | **0.659** | **0.000** | 0.401 |

### 3a. Failure one — amplitude scale (destroys `Normal` recall)

Confirmed, not assumed. `src/training/train_baseline.py:raw_rms_scale_summary` recomputes
it from the loaded dataset:

| Experiment | min raw `rms` | mean raw `rms` | max raw `rms` |
|---|---|---|---|
| `1st_test` | **0.1289** | 0.1618 | 0.5936 |
| `2nd_test` | 0.0015 | **0.1061** | 0.7250 |
| `3rd_test` | 0.0040 | **0.0724** | 0.7588 |

`1st_test`'s *minimum* raw RMS exceeds both training experiments' *means*. A
`StandardScaler` fitted — correctly — on the two training experiments therefore maps every
single `1st_test` row into the high-RMS region of the training distribution: its scaled
`rms` ranges from +1.39 to +13.78 standard deviations, with a mean of +2.26. The model
concludes the bearing is already degraded on its first file. Removing the column recovers
`Normal` recall from 0.074 to 0.659, confirming the mechanism — the same direction and
roughly the same magnitude as #21's own diagnostic (macro-F1 0.192 → 0.493 for the
untreated arm).

The physical reason is documented: `docs/eda_findings.md` §2 records `1st_test`'s inner-race
failure as impulsive (kurtosis-led), while `2nd_test`/`3rd_test`'s outer-race failures are
amplitude-led. Its RMS baseline simply sits higher.

### 3b. Failure two — threshold transfer (destroys `Critical` recall, and is not fixable)

This one is new to this issue and is the more consequential of the two.
`src/training/train_baseline.py:critical_band_summary` asks, per fold: do the held-out
experiment's `Critical` rows fall inside the `rms_ratio` band its *training* fold ever
labelled `Critical`?

| Held out | train-fold min `Critical` `rms_ratio` | held-out `Critical` band | unreachable rows |
|---|---|---|---|
| `1st_test` | 2.870 | [1.948, 2.869] | **17 / 17** |
| `2nd_test` | 1.948 | [2.870, 6.323] | 0 / 23 |
| `3rd_test` | 1.948 | [3.090, 7.153] | 0 / 67 |

**Every one of `1st_test`'s 17 `Critical` rows lies below the lowest `rms_ratio` that its
training fold ever labelled `Critical`.** The two bands do not overlap at all — 2.869 is the
top of the held-out band and 2.870 is the bottom of the training one. No monotone decision
boundary learned from `2nd_test`+`3rd_test` can reach those rows, irrespective of scaling,
class weighting, or model family. That is why `no_raw_rms` — which fixes the scale problem
cleanly — drives `1st_test`'s `Critical` recall to exactly **0.000**: with the amplitude
confound removed, the model relies on `rms_ratio`, and `rms_ratio` places `1st_test`'s
Critical band squarely inside what its training fold calls `Degrading`.

The cause is structural and already documented upstream. `critical_multiple` is derived per
experiment from that experiment's own eventual peak `rms_ratio` (`docs/eda_findings.md` §3,
a look-ahead retrospective quantity): 1.932 / 2.866 / 3.049. `1st_test`'s boundary sits
1.58× lower than `3rd_test`'s. `docs/evaluation_protocol.md` §2 predicted this precisely —
"recalling one experiment's threshold is not the same skill as inferring an unseen one" —
and gave it as the reason LOEO was chosen over a random split. This fold is that prediction
coming true, measured.

Worth noting the asymmetry, because it explains why the other two folds do well:
`1st_test`'s low band (1.948) is *in the training fold* when `2nd_test` or `3rd_test` is
held out, which drags the learned boundary low enough to catch their `Critical` rows. The
experiment that cannot be predicted is the one that makes the others predictable.

## 4. The scale fix: what was tried, measured, and decided

Issue #72 Task 3 named two candidate paths. Both were implemented and measured rather than
argued about, and a third was measured specifically to quantify what is being declined.

### Candidate A — per-experiment normalisation fitted on training-fold data only

(`src/training/candidate_scalers.py:evaluate_averaged_per_experiment_scaler`)

**Rejected: ineffective, and measurement confirms it.** Per-experiment `StandardScaler`
parameters were fitted on each training experiment separately and averaged (equal weight per
experiment rather than per row), then applied to both train and held-out rows. This is
strictly leakage-safe — the held-out experiment's rows are never fitted on.

| Held out | `full` macro-F1 | Candidate A macro-F1 |
|---|---|---|
| `1st_test` | 0.152 | 0.154 |
| `2nd_test` | 0.936 | 0.942 |
| `3rd_test` | 0.945 | 0.945 |

It does essentially nothing (0.152 → 0.154), and the reason is structural rather than a
tuning failure: **any affine transform fitted without seeing the held-out experiment
preserves that experiment's displacement relative to the training distribution.** Changing
the centre and scale moves train and test together. The problem is not that the scaler's
parameters are wrong; it is that `1st_test`'s raw-RMS distribution genuinely does not
overlap the training experiments'.

There is also a framing point worth making explicit, because it changes what the issue's
proposed fix means. **A leakage-safe per-experiment normalisation of RMS already exists in
this feature set — it is `rms_ratio`.** `src/features/extraction.py:add_rolling_rms_ratio`
defines it as the 10-file rolling mean of `rms` divided by the mean `rms` of that
experiment's *first 50 files*, a baseline drawn from early life and therefore causally
available at serving time, not look-ahead. So "add a per-experiment scaler" and "drop raw
`rms` in favour of `rms_ratio`" are not two competing fixes; the second **is** the first,
already implemented upstream in M2 and available without new machinery. Candidate A was
still run, because "the fix already exists" is a claim that should be checked rather than
asserted.

### Candidate B — dropping raw `rms` (`no_raw_rms`)

**Not adopted, on evidence.** Issue #72 conditioned this path on the ablation showing raw
`rms` "isn't pulling its weight elsewhere". Measured, it is:

| Held out | `Critical` precision, `full` → `no_raw_rms` | Macro-F1, `full` → `no_raw_rms` |
|---|---|---|
| `1st_test` | 1.000 → 0.000 | 0.152 → **0.401** |
| `2nd_test` | 0.750 → **0.548** | 0.936 → **0.889** |
| `3rd_test` | 0.859 → **0.528** | 0.945 → **0.747** |

Raw `rms` earns its place on two of the three folds, substantially so on `3rd_test`
(macro-F1 0.945 → 0.747, `Critical` precision 0.859 → 0.528). The stated condition for
dropping it is therefore not met. Dropping it trades a large, real degradation on the two
folds that work for a partial improvement on the fold that does not — and even there it
improves macro-F1 (0.152 → 0.401) only by moving the failure from the majority class to the
headline metric, taking `Critical` recall to 0.000.

### Candidate C — scaler fitted on the held-out experiment's own rows

(`src/training/candidate_scalers.py:evaluate_transductive_scaler`)

**Rejected on protocol grounds; measured only to state honestly what is being given up.**
`docs/evaluation_protocol.md` §1 is unambiguous that fitted preprocessing must see the
fold's training rows only, so this is not an available option. It was run anyway, because
declining a fix is more credible when its size is known:

| Held out | `full` macro-F1 | Candidate C macro-F1 |
|---|---|---|
| `1st_test` | 0.152 | **0.800** |
| `2nd_test` | 0.936 | 0.816 |
| `3rd_test` | 0.945 | 0.624 |

It would largely repair `1st_test` (`Normal` recall 0.074 → 0.968, `Critical` recall
0.059 → 1.000) while **degrading both other folds**, so it is not even a free win on its own
terms. Beyond the protocol violation, it is unavailable in practice for the reason
`docs/eda_findings.md` §3 already gives: normalising a bearing against its own completed run
requires the whole run, which a bearing still in operation has not finished. Recorded here
so that the `1st_test` numbers in §1 are understood as a choice made under a stated rule,
not an oversight.

### Decision

**Keep the full M2 feature set. Leave the `1st_test` fold unresolved and reported.** No
leakage-safe transformation available at this step repairs it: Candidate A does nothing,
Candidate B moves the damage rather than removing it, Candidate C is ruled out by the
protocol. And §3b shows that even a perfect fix to the amplitude-scale problem would leave
`Critical` recall on that fold at 0.000, because the second failure is a property of how the
labels were thresholded, not of how the features were scaled.

What this means concretely for that fold's numbers, stated so it cannot be misread: **the
`1st_test` row of every table in this document should be read as "this model does not work
on this bearing", not as a low-but-meaningful score.** Macro-F1 0.152 with `Normal` recall
0.074 is a model predicting `Degrading` for almost everything.

## 5. Leakage guards

Carried over from `docs/class_imbalance_decision.md` §8 and extended to this step's second
axis of variation (the feature set). Each is enforced in code and pinned by a test in
`tests/test_train_baseline.py`:

- **No hyperparameter is tuned, per fold or per feature set.** Every configuration uses the
  identical fixed model (`BASELINE_MODEL_PARAMS` — default regularisation, fixed
  `random_state`, `max_iter` raised only far enough to converge). The mechanism that could
  leak is removed rather than managed, so there is nothing that *could* be tuned on one fold
  and tested on another.
- **`StandardScaler` stays inside the `Pipeline`**, fitted on each fold's training rows and
  only transforming the held-out rows. Pinned behaviourally rather than structurally: one
  test asserts the fitted scaler's centre equals the training rows' mean and differs from
  the pooled mean; a second asserts that perturbing the held-out experiment's feature values
  by +100 leaves the fitted scaler bit-for-bit unchanged.
- **The held-out class support is identical across all four configurations**, so the
  configurations are comparable and no configuration is quietly scored on a different test
  set.
- **No model artifact is persisted.** LOEO trains three models per configuration and there
  is no single "the model" to save; persisting one would also invite it being scored on rows
  it trained on. Deferred to the M3 step that produces a servable artifact.

## 6. Honest statement of this model's LOEO-validated capability

Written to be quotable without the surrounding tables, since that is how a summary line gets
reused:

> Under leave-one-experiment-out evaluation, the M3 baseline (standardised M2 features +
> multinomial logistic regression, `class_weight='balanced'`) detects the `Critical` health
> state well on two of the three bearings — `Critical` recall 0.913 and 1.000, at precision
> 0.750 and 0.859, macro-F1 0.936 and 0.945 — and **fails on the third** (`1st_test`:
> `Critical` recall 0.059, macro-F1 0.152). The failure is diagnosed, not mysterious, and
> has two independent causes: raw RMS amplitude does not transfer between bearings, and
> `1st_test`'s entire `Critical` band lies below the lowest `rms_ratio` its training fold
> ever labelled `Critical`, making those 17 rows unreachable by any boundary learned from
> the other two experiments. The mean across folds (`Critical` recall 0.657) describes no
> fold and should not be quoted as the model's performance.

Three qualifications on that statement, none of which the numbers above can settle:

- **The two folds that work are both outer-race failures; the fold that fails is the only
  inner-race one.** This is suggestive but not establishable: `docs/evaluation_protocol.md`
  §6 already recorded that with *n* = 1 inner-race and *n* = 2 outer-race experiments, "the
  model fails on inner-race failures" cannot be separated from "the model fails on this
  particular bearing". Both remain live readings of the same evidence.
- **This is a baseline, and its failure mode is informative about the labels, not only the
  model.** §3b's unreachability is a property of per-experiment `critical_multiple`, so a
  more capable model class trained on these same labels would face the same obstacle on this
  same fold. That does not mean nothing can be done — it means the next lever is the label
  or feature definition, not the estimator.
- **These are three bearings from one lab rig at one operating condition.** Per
  `docs/evaluation_protocol.md` §6, the numbers are evidence about these three trajectories,
  not a forecast of field failure-detection rates.

## 7. What this does not settle

- **Whether a different model family closes the gap on `2nd_test`/`3rd_test`.** The logistic
  regression here is #21's measuring instrument, carried forward deliberately so this step
  changed one axis at a time. Trying tree-based models is a reasonable next step and is
  unaffected by anything decided here (`class_weight` is supported across sklearn's
  classifier family).
- **Whether `1st_test` is recoverable at all**, and if so by which lever — a
  causally-available per-experiment normalisation richer than `rms_ratio`, a revised
  `critical_multiple` derivation, or the frequency-domain features `docs/frequency_domain_decision.md`
  evaluated and dropped for a different purpose. §3b narrows *where* to look (the label
  threshold, not the estimator) without resolving it.
- **`prior_correction`**, which `docs/class_imbalance_decision.md` §5 deferred to this step
  as "the arm to beat once `rms_ratio` is removed". It was not re-tested here: §2's finding
  is that the `no_rms_ratio` configuration over-alarms, and `prior_correction` shifts
  decisions further toward rare classes, so re-running it would compound the effect this
  section identifies rather than isolate it. Still open, now with a more specific reason to
  be sceptical of its §6 lead.
- **MLflow tracking.** `docs/PRD.md` §11 lists it under this milestone and §10's acceptance
  criteria require at least one visible run; Issue #72's own acceptance criteria do not
  mention it, and it is not added here. Deferred to a separate issue.
- **Whether `Critical` precision at ~0.87 is operationally acceptable** — unchanged from
  `docs/class_imbalance_decision.md` §7; `docs/PRD.md` does not quantify a false-alarm cost
  and this document does not invent one.

## Reproducing

```
python -m src.features.build_training_dataset   # Issue #67, writes training_dataset.parquet
python -m src.training.train_baseline           # every table in Sections 1-3
python -m src.training.candidate_scalers        # the rejected Candidates A and C in Section 4
```

Deterministic: fixed `random_state` throughout, no tuning anywhere, no resampling in the
adopted configuration.

The rejected scaling candidates are kept as runnable, tested code in
`src/training/candidate_scalers.py` rather than left as numbers quoted here — the same
convention `src/features/candidate_features.py` follows for the features evaluated and
dropped in #22/#23. A rejection is worth more when it can be re-checked than when it has to
be trusted. Neither candidate is imported by `src/training/train_baseline.py`, and a test
pins that, so the protocol-violating Candidate C cannot drift into the adopted path.
