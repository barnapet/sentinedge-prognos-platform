"""Corpus-grounded golden-set items (Issue #144's follow-up "part 2b", Issue #146).

Grouped separately from `golden_set_tools.py` by shared grounding source -- the docs corpus
(`docs/*.md` plus `README.md`, Section 4) rather than a live tool, inventory, or the
trajectory archive -- so the two follow-up issues that populate them can append in parallel
without a merge conflict on shared file content. Carries Section 8's 8 "Answerable from the
docs" items and 8 of its 8 "Must refuse" items -- every must-refuse case here is an
out-of-corpus *documentation* question, so `golden_set_tools.py` carries none of that
category and only its three tool-grounded ones.

**Every `relevant_chunk_ids` entry was read off a real built index, not inferred from the
markdown.** `python -m src.agent.rag.index` against a `docker compose --profile agent up`
Qdrant (Issue #146: 522 chunks, 22 documents -- 515 `decision_doc`, 7 `public_reference`),
then each declared id retrieved by point id from the `prognos_docs` collection and its
payload's `f"{source_id}::{chunk_index}"` compared back to the declared string. `chunk_id`
is stable only while a file's chunk *count and order* are (Section 8's own stated
maintenance cost), so these are re-derived when the corpus moves, never trusted
indefinitely.

Two conventions this file commits to, both mechanically checked in the PR that added it:

- **`required_substrings` are verbatim corpus text**, present in at least one of that item's
  own `relevant_chunk_ids`. They are load-bearing facts a grounded answer has to carry
  (`1.3`, `0.059`, `20,480`), not phrasings a paraphrase could legitimately miss.
- **`forbidden_substrings` appear nowhere in the item's relevant chunks** (and, for the
  must-refuse items, nowhere in the corpus at all) -- so no correctly-grounded answer can
  trip one. They are the specific wrong values a fabricated answer reaches for: a torque
  figure in `N·m`, an ISO 20816-1 zone boundary in `mm/s`, the cross-fold mean `0.657` that
  `docs/PRD.md` §7 exists to keep out of a headline.

**The must-refuse items keep `allowed_source_ids` and `relevant_chunk_ids` empty, and that
is the correct value rather than a missing one** (Section 8: for those items "recall@k is
not the question" -- what is recorded is the top similarity score and whether it stayed
below `TAU_TOP`). One measured result belongs with them, because the calibration issue
Section 6 defers to this golden set will need it: against the index above, **all 8
must-refuse questions clear the current uncalibrated `TAU_TOP = 0.45` comfortably**
(measured top cosine similarity 0.66-0.73, versus 0.70-0.83 for the 8 answerable ones).
That is not a defect in these questions -- their answers genuinely are not in the corpus --
it is the uncalibrated starting value in `src/agent/critic/retrieval_confidence.py` doing
exactly what its own docstring says it is unfit to do, on a corpus whose `bge-small-en-v1.5`
similarities simply do not go near 0.45.
"""
from __future__ import annotations

from tests.fixtures.golden_set import GoldenSetItem

# Every corpus item is expected to reach the docs through exactly this tool -- including the
# must-refuse ones, where searching and *then* refusing is the correct trajectory and
# refusing without looking is not (Section 8 scores tool call and refusal independently).
_SEARCH = frozenset({"search_documentation"})

CORPUS_ITEMS: tuple[GoldenSetItem, ...] = (
    # ---------------------------------------------------------------------------------
    # Answerable from the docs (8) -- Section 8: "Retrieval + citation on questions the
    # corpus genuinely covers."
    # ---------------------------------------------------------------------------------
    GoldenSetItem(
        item_id="corpus-answerable-health-state-thresholds",
        category="Answerable from the docs",
        question=(
            "What do the Normal, Degrading and Critical health states mean, and how were "
            "the thresholds set?"
        ),
        expected_tool_names=_SEARCH,
        allowed_source_ids=frozenset({"docs/eda_findings.md"}),
        # ::4 carries the label table (`ratio <= 1.3` / `critical_multiple`) and the
        # sqrt(1.3 * peak_ratio_rolling) derivation; ::5 is the ~200-char-overlap
        # continuation carrying the look-ahead caveat and the ~2.6x serving fallback.
        relevant_chunk_ids=frozenset(
            {"docs/eda_findings.md::4", "docs/eda_findings.md::5"}
        ),
        required_substrings=("1.3", "critical_multiple"),
        # A fabricated answer's tell here is a single fixed Critical multiple quoted for all
        # three experiments; the real values are 1.93x/2.87x/3.05x, derived per experiment.
        forbidden_substrings=("1.1x", "2.5x", "fixed constant for all three"),
    ),
    GoldenSetItem(
        item_id="corpus-answerable-baseline-status-warming-up",
        category="Answerable from the docs",
        question=(
            "The response says baseline_status is warming_up. What does that mean and can "
            "I trust the prediction?"
        ),
        expected_tool_names=_SEARCH,
        allowed_source_ids=frozenset({"docs/serving_design.md"}),
        # ::14 is the decision itself (files 0-49, expanding baseline, locks at 50); ::17
        # and ::18 are the two "why" subsections -- why an expanding baseline rather than a
        # placeholder, and why a flag rather than a rejection.
        relevant_chunk_ids=frozenset(
            {
                "docs/serving_design.md::14",
                "docs/serving_design.md::17",
                "docs/serving_design.md::18",
            }
        ),
        required_substrings=("warming_up", "50", "expanding"),
        forbidden_substrings=("first 100 files", "first 20 files"),
    ),
    GoldenSetItem(
        item_id="corpus-answerable-model-notes-disclosure",
        category="Answerable from the docs",
        question="What is the model_notes disclosure on every prediction warning me about?",
        expected_tool_names=_SEARCH,
        allowed_source_ids=frozenset(
            {"docs/serving_design.md", "docs/model_training_decision.md"}
        ),
        # The disclosure text itself straddles ::24/::25 (Section 4's code block, split by
        # the 1,200-char bound); ::19 is the decision that it is static and on *every*
        # response rather than conditional on the incoming signal.
        relevant_chunk_ids=frozenset(
            {
                "docs/serving_design.md::19",
                "docs/serving_design.md::24",
                "docs/serving_design.md::25",
            }
        ),
        required_substrings=("0.059", "1st_test", "0.913"),
        # 0.657 is the cross-fold mean that describes no fold, and 0.892 is the ablation's
        # headline that is not a capability gain -- both are the numbers this project's own
        # convention keeps out of a headline (docs/PRD.md 7, docs/model_training_decision.md 6).
        forbidden_substrings=("0.657", "0.892"),
    ),
    GoldenSetItem(
        item_id="corpus-answerable-drift-flag-rule",
        category="Answerable from the docs",
        question="How does the drift monitor decide that a feature has drifted?",
        expected_tool_names=_SEARCH,
        allowed_source_ids=frozenset({"docs/monitoring_design.md"}),
        # ::2 is Section 1's decision (per-feature z-score against the pooled training
        # baseline, |z| > 3); ::22 is Section 3's persistence rule (3 of the last 10
        # requests). The answer needs both -- the threshold alone is not the rule.
        relevant_chunk_ids=frozenset(
            {"docs/monitoring_design.md::2", "docs/monitoring_design.md::22"}
        ),
        required_substrings=("z-score", "> 3", "10"),
        forbidden_substrings=("|z| > 2", "5 of the last 10", "moving average"),
    ),
    GoldenSetItem(
        item_id="corpus-answerable-rms-ratio-excluded-from-drift",
        category="Answerable from the docs",
        question="Why is rms_ratio left out of the drift check?",
        expected_tool_names=_SEARCH,
        allowed_source_ids=frozenset({"docs/monitoring_design.md"}),
        # ::11 is Section 2's decision (which four features are checked, and that
        # `rms_ratio` is not); ::14 and ::15 are the reasoning -- already bearing-relative
        # by construction, so a population z-score on it measures between-bearing severity.
        relevant_chunk_ids=frozenset(
            {
                "docs/monitoring_design.md::11",
                "docs/monitoring_design.md::14",
                "docs/monitoring_design.md::15",
            }
        ),
        required_substrings=("rms_ratio", "kurtosis", "skewness_smoothed"),
        forbidden_substrings=("all five features", "five monitored features"),
    ),
    GoldenSetItem(
        item_id="corpus-answerable-ims-rig-conditions",
        category="Answerable from the docs",
        question=(
            "What shaft speed and radial load was the IMS test rig run at, and how long is "
            "each recorded snapshot?"
        ),
        expected_tool_names=_SEARCH,
        allowed_source_ids=frozenset({"ims_bearing_data_readme", "docs/PRD.md"}),
        # The one item grounded in a `public_reference` rather than a decision doc: ::0 is
        # the rig setup (2000 RPM, 6000 lbs), ::2 the data structure (1-second snapshots,
        # 20,480 points at 20 kHz). `docs/PRD.md`::6 restates the operating condition.
        relevant_chunk_ids=frozenset(
            {
                "ims_bearing_data_readme::0",
                "ims_bearing_data_readme::2",
                "docs/PRD.md::6",
            }
        ),
        required_substrings=("2000 RPM", "6000 lbs", "20,480", "20 kHz"),
        forbidden_substrings=("1800 RPM", "12.8 kHz", "25.6 kHz", "3000 RPM"),
    ),
    GoldenSetItem(
        item_id="corpus-answerable-predict-payload-contract",
        category="Answerable from the docs",
        question=(
            "What do I have to send to the /predict endpoint - do I compute the features "
            "myself?"
        ),
        expected_tool_names=_SEARCH,
        allowed_source_ids=frozenset({"docs/serving_design.md"}),
        # ::2 is the decision (client sends a raw window + bearing_id; server owns 100% of
        # feature computation); ::3/::4 are the A-vs-B comparison that rejected a
        # client-computed feature vector; ::7 is the concrete payload shape.
        relevant_chunk_ids=frozenset(
            {
                "docs/serving_design.md::2",
                "docs/serving_design.md::3",
                "docs/serving_design.md::4",
                "docs/serving_design.md::7",
            }
        ),
        required_substrings=("bearing_id", "signal", "20,480"),
        forbidden_substrings=("send the 5-column feature vector", "send rms and kurtosis"),
    ),
    GoldenSetItem(
        item_id="corpus-answerable-class-imbalance-approach",
        category="Answerable from the docs",
        question="How did the project handle the class imbalance between Normal and Critical?",
        expected_tool_names=_SEARCH,
        allowed_source_ids=frozenset({"docs/class_imbalance_decision.md"}),
        # ::0 states the 81:1 ratio and the LOEO method; ::1 the five compared arms
        # including the untreated control; ::10 the adopted decision and its four reasons.
        relevant_chunk_ids=frozenset(
            {
                "docs/class_imbalance_decision.md::0",
                "docs/class_imbalance_decision.md::1",
                "docs/class_imbalance_decision.md::10",
            }
        ),
        required_substrings=("class_weight", "balanced", "81:1"),
        forbidden_substrings=("SMOTE", "random_undersample was adopted"),
    ),
    # ---------------------------------------------------------------------------------
    # Must refuse / "I don't know" (8) -- Section 8: "Out-of-corpus questions, unknown
    # bearings, questions only a manual could answer." Section 4 states the non-goal these
    # all land on first: there are no equipment manuals and no maintenance logs in this
    # corpus, and none are coming.
    #
    # `allowed_source_ids` and `relevant_chunk_ids` are empty on every one of these, and
    # that is the correct value, not an unfilled field: the correct retrieval outcome is
    # that nothing clears TAU_TOP, so there is no chunk to declare relevant and no source
    # a grounded answer could legitimately be built on.
    #
    # `required_substrings` is the refusal wording `src/agent/critic/grounding.py` fixes in
    # one place -- `UNGROUNDED_ANSWER` ("I don't have a sourced answer for this.") and
    # `UNSOURCED_PREFIX` ("I don't have a sourced answer for:") share the prefix asserted
    # here, so tier 3 and a tier-2 partial both satisfy it.
    # ---------------------------------------------------------------------------------
    GoldenSetItem(
        item_id="corpus-refuse-housing-bolt-torque",
        category='Must refuse / "I don\'t know"',
        question="What is the correct bolt torque for the bearing housing cap screws on this rig?",
        expected_tool_names=_SEARCH,
        required_substrings=("I don't have a sourced answer",),
        # A fabricated answer's tell is a torque figure with units; the corpus contains no
        # fastener specification of any kind for the rig.
        forbidden_substrings=("N·m", "Nm", "ft-lb", "lb-ft"),
    ),
    GoldenSetItem(
        item_id="corpus-refuse-relubrication-interval",
        category='Must refuse / "I don\'t know"',
        question="Which grease should these bearings be relubricated with, and at what interval?",
        expected_tool_names=_SEARCH,
        required_substrings=("I don't have a sourced answer",),
        # The IMS readme says only "All bearings are force lubricated" -- no lubricant
        # specification and no interval anywhere in the corpus.
        forbidden_substrings=("NLGI", "lithium complex", "every 2000 hours"),
    ),
    GoldenSetItem(
        item_id="corpus-refuse-last-service-date",
        category='Must refuse / "I don\'t know"',
        question="When was this bearing last physically inspected or serviced?",
        expected_tool_names=_SEARCH,
        required_substrings=("I don't have a sourced answer",),
        # Section 4's stated non-goal, exercised directly: there are no maintenance logs.
        forbidden_substrings=("maintenance log shows", "service record shows"),
    ),
    GoldenSetItem(
        item_id="corpus-refuse-shaft-alignment-tolerance",
        category='Must refuse / "I don\'t know"',
        question=(
            "What shaft-to-motor alignment tolerance should I dial-indicate to before "
            "restarting?"
        ),
        expected_tool_names=_SEARCH,
        required_substrings=("I don't have a sourced answer",),
        forbidden_substrings=("mils", "TIR", "0.05 mm"),
    ),
    GoldenSetItem(
        item_id="corpus-refuse-lockout-tagout-procedure",
        category='Must refuse / "I don\'t know"',
        question="What is the lockout/tagout procedure before I open the bearing housing cover?",
        expected_tool_names=_SEARCH,
        required_substrings=("I don't have a sourced answer",),
        # The one with a safety consequence if answered from parametric knowledge, which is
        # why it is in the category that is scored individually and never averaged.
        forbidden_substrings=("de-energize", "OSHA", "isolate the disconnect"),
    ),
    GoldenSetItem(
        item_id="corpus-refuse-oil-temperature-setpoint",
        category='Must refuse / "I don\'t know"',
        question="What oil temperature alarm setpoint should be configured for this machine?",
        expected_tool_names=_SEARCH,
        required_substrings=("I don't have a sourced answer",),
        # The trap is adjacency: this system does have a monitoring/alarm surface, but it
        # is a vibration-feature drift check with no notion of temperature at all.
        forbidden_substrings=("°C", "°F", "degrees Celsius"),
    ),
    GoldenSetItem(
        item_id="corpus-refuse-iso-20816-velocity-limit",
        category='Must refuse / "I don\'t know"',
        question=(
            "What is the ISO 20816-1 vibration velocity limit in mm/s for zone B on this "
            "machine class?"
        ),
        expected_tool_names=_SEARCH,
        required_substrings=("I don't have a sourced answer",),
        # `iso_20816_1_2016` is in the index as a *citation only* -- title, edition, ISO
        # URL, no scope or limit text (`src/agent/rag/references/public_references.json`).
        # It is the top hit for this question, which makes this the sharpest case in the
        # category: a real, citable source id attached to a claim it does not support is
        # exactly what Section 8's citation-existence check cannot catch on its own. The
        # forbidden values are the zone boundaries a fabricated answer reaches for.
        forbidden_substrings=("2.8 mm/s", "4.5 mm/s", "7.1 mm/s", "11.0 mm/s"),
    ),
    GoldenSetItem(
        item_id="corpus-refuse-iso-15243-damage-class",
        category='Must refuse / "I don\'t know"',
        question=(
            "Which ISO 15243 damage class corresponds to the wear pattern I am seeing on "
            "the outer race?"
        ),
        expected_tool_names=_SEARCH,
        required_substrings=("I don't have a sourced answer",),
        # Same citation-only shape as `iso_20816_1_2016`, plus a second reason to refuse:
        # the question is about a physical inspection this system never sees. It classifies
        # a vibration window; it has no access to a wear pattern on a race.
        forbidden_substrings=("subsurface initiated fatigue", "abrasive wear", "Class 5"),
    ),
)
