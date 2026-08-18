"""Tier-1 tests for Agent B, the Critic (Issue #114, `docs/agent_design.md` Sections 1, 5
and 6). No API key, no model call, no network — Section 8 makes that a hard requirement for
this tier, and the deterministic layer is the part of the critic that has to hold without a
model at all.

What is asserted here:

- **Each of Section 6's four deterministic checks catches its own failure mode on its own.**
  Every one of those tests feeds a draft that fails exactly one check and passes the other
  three, so a passing suite cannot be explained by one check masking another.
- **The three-tier assembly**, including the two things Section 6 rules out: it never
  answers un-grounded, and it never silently drops a claim.
- **The escalation rule**, on hand-built inputs rather than a live call: clean pass plus a
  recommendation, or clean pass plus overlap below the floor, and nothing otherwise.
- **The concept-domain relevance check** (Issue #171): the registry's ids are the ones the tool
  layer really mints, a claim inside its cited source's domain survives, one outside it is
  demoted, and a claim citing prose alongside is not evaluated here at all.
- **The critic holds no tools**, structurally: nothing in the package can reach the tool
  layer, and the request it builds carries no tool argument.

The one live call lives in `tests/test_agent_critic_live.py` and is API-key-gated.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.agent.answerer import Claim, Draft
from src.agent.critic import escalation as escalation_module
from src.agent.critic import grounding as grounding_module
from src.agent.critic import retrieval_confidence as retrieval_confidence_module
from src.agent.critic.deterministic import (
    CITATION_COVERAGE,
    CITATION_EXISTENCE,
    NUMERIC_FIDELITY,
    EvidenceItem,
    TurnEvidence,
    extract_numbers,
    gate_recommendation,
    unknown_source_ids,
    unsupported_numbers,
    verify,
)
from src.agent.critic.escalation import (
    EFFORT,
    LEXICAL_OVERLAP_FLOOR,
    MAX_TOKENS,
    MODEL,
    PROSE_SOURCE_TYPES,
    SYSTEM_PROMPT,
    THINKING,
    VERDICTS,
    Entailment,
    build_messages,
    escalate_async,
    escalations_needed,
    lexical_overlap,
    parse_verdict,
    verdict_schema,
)
from src.agent.critic.grounding import (
    GROUNDED,
    OFF_DOMAIN_REASON,
    PARTIAL,
    UNGROUNDED,
    UNGROUNDED_ANSWER,
    UNSOURCED_PREFIX,
    assemble,
)
from src.agent.critic.relevance import (
    CONCEPT_DOMAINS,
    DRIFT_SOURCE_ID,
    INVENTORY_ORDERS_SOURCE_ID,
    PREDICT_SOURCE_ID,
    REGISTERED_SOURCE_IDS,
    check_claim_domain,
    domain_for,
    off_domain_demotions,
)
from src.agent.critic.relevance import (
    # Aliased: `src.agent.mcp.tools` exports the same name, and the one test that compares the
    # two needs both of them unambiguously.
    INVENTORY_SOURCE_ID as INVENTORY_DOMAIN_SOURCE_ID,
)
from src.agent.critic.retrieval_confidence import (
    MIN_SUPPORTING_CHUNKS,
    TAU_SUPPORT,
    TAU_TOP,
    assess_retrieval,
)
from src.agent.mcp.results import payload_of
from src.agent.mcp.tools import search_documentation
from src.agent.rag.retrieval import RetrievedChunk
from src.agent.untrusted import TAG, UntrustedEnvelope

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_DOC = REPO_ROOT / "docs" / "agent_design.md"
CRITIC_DIR = REPO_ROOT / "src" / "agent" / "critic"
CRITIC_MODULES = (
    "src.agent.critic.deterministic",
    "src.agent.critic.retrieval_confidence",
    "src.agent.critic.grounding",
    "src.agent.critic.escalation",
    "src.agent.critic.relevance",
)

# One real chunk's worth of text, quoting real numbers from this repo's own decision docs —
# which is the case Section 6's numeric-fidelity check exists for.
CHUNK_ID = "docs/model_training_decision.md::7"
CHUNK_TEXT = (
    "Critical recall is 0.913 / 1.000 on 2nd_test / 3rd_test and 0.059 on 1st_test. "
    "The cross-fold mean of 0.657 describes no fold and should not be quoted as the "
    "project's number."
)
OTHER_CHUNK_ID = "docs/serving_design.md::3"
OTHER_CHUNK_TEXT = (
    "A bearing's first 50 files are scored against an expanding baseline and flagged with "
    "baseline_status warming_up; the server never refuses to score."
)


def evidence(
    *, scores: tuple[float, float] = (TAU_TOP + 0.05, TAU_SUPPORT + 0.02)
) -> TurnEvidence:
    """Two retrieved chunks, both citable, with retrieval that clears the thresholds.

    The default scores are expressed relative to `TAU_TOP`/`TAU_SUPPORT` rather than as
    literals: "clears the thresholds" is the property every caller of this fixture depends
    on, and #163's calibration moved the thresholds far enough (0.45/0.35 → 0.75/0.70) that
    the previous literals silently stopped clearing them.

    Both carry `source_type="decision_doc"`, which is what the tool layer mints for a chunk
    retrieved from a `docs/` file (Section 2's vocabulary) and therefore what
    `from_tool_payloads` puts on a real one. It is not decoration: Issue #119 scopes the
    escalation's overlap check to prose chunks, so an evidence item that omits its type is a
    fixture that no longer represents a retrieved chunk.
    """
    return TurnEvidence(
        (
            EvidenceItem(
                CHUNK_ID,
                CHUNK_TEXT,
                score=scores[0],
                title="model_training_decision",
                source_type="decision_doc",
            ),
            EvidenceItem(
                OTHER_CHUNK_ID,
                OTHER_CHUNK_TEXT,
                score=scores[1],
                title="serving_design",
                source_type="decision_doc",
            ),
        )
    )


def draft(*claims: Claim, recommendation: str | None = None, unanswered=()) -> Draft:
    return Draft(
        claims=list(claims), recommendation=recommendation, unanswered=list(unanswered)
    )


GOOD_CLAIM = Claim(
    text="Critical recall is 0.059 on 1st_test.", source_ids=[CHUNK_ID]
)
OTHER_GOOD_CLAIM = Claim(
    text="The server never refuses to score during the first 50 files.",
    source_ids=[OTHER_CHUNK_ID],
)


# --- Section 1: the critic's request configuration --------------------------------------


def test_the_critics_request_configuration_is_section_1s():
    assert MODEL == "claude-opus-5"
    assert MAX_TOKENS == 16000
    assert EFFORT == "low"
    assert THINKING == {"type": "adaptive"}


def test_the_configuration_matches_the_design_documents_own_text():
    """Anti-drift, the same way #113 pins the answerer's: read Section 1 rather than
    trusting that a constant still says what the decision says."""
    section_1 = DESIGN_DOC.read_text(encoding="utf-8").split("## 2. The MCP tool layer")[0]

    assert f"`{MODEL}`" in section_1
    assert re.search(rf"\*\*`max_tokens`:\s*{MAX_TOKENS}\*\*,\s*non-streaming", section_1)
    assert "`low` for the critic's escalated entailment check" in section_1


def test_the_critic_asks_only_the_three_closed_verdicts():
    assert VERDICTS == ("yes", "no", "unclear")
    assert verdict_schema()["properties"]["verdict"]["enum"] == list(VERDICTS)
    assert verdict_schema()["additionalProperties"] is False


def test_the_critics_system_prompt_is_not_the_answerers():
    """Section 6: different system prompt, and no sight of the question or the draft's
    framing. A critic told it is answering a technician is a second answerer."""
    from src.agent.answerer import SYSTEM_PROMPT as ANSWERER_PROMPT

    assert SYSTEM_PROMPT != ANSWERER_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "technician" not in lowered
    assert "tool" not in lowered
    assert "rewrite" in lowered


# --- Section 6, check 1: citation existence ---------------------------------------------


# Carries no numeric literal, so citation existence is the only check that can fire on it —
# which is what makes the pair of tests below evidence about one check rather than two.
_UNNUMBERED = "Critical recall is far lower on 1st_test than on the other two folds."


def test_citation_existence_catches_a_fabricated_id():
    """The check no prompt instruction can substitute for: an id that never appeared in this
    turn's tool results is not a citation, however well-formed it looks."""
    fabricated = Claim(
        text=_UNNUMBERED, source_ids=["docs/model_training_decision.md::99"]
    )

    assert unknown_source_ids(fabricated, evidence()) == (
        "docs/model_training_decision.md::99",
    )

    report = verify(draft(fabricated), evidence())
    (checked,) = report.claims
    assert checked.failures == (CITATION_EXISTENCE,)
    assert report.citation_existence_failed is True


def test_citation_existence_passes_a_real_id_and_nothing_else_fires():
    """The negative half of the same test: the fabricated id above differs from the real one
    by four characters, and everything else about the claim is identical."""
    report = verify(draft(Claim(text=_UNNUMBERED, source_ids=[CHUNK_ID])), evidence())

    assert report.clean is True
    assert report.citation_existence_failed is False

    assert verify(draft(GOOD_CLAIM), evidence()).clean is True


def test_a_fabricated_id_does_not_stop_the_numbers_being_checked_against_the_real_one():
    """Existence deliberately does not short-circuit numeric fidelity: a claim citing one
    real id and one invented one still has real text to check its numbers against, and both
    facts about it are worth reporting."""
    mixed = Claim(
        text="Critical recall is 0.059 on 1st_test.",
        source_ids=[CHUNK_ID, "docs/invented.md::1"],
    )

    (checked,) = verify(draft(mixed), evidence()).claims

    assert checked.failures == (CITATION_EXISTENCE,)
    assert unsupported_numbers(mixed, evidence()) == ()


# --- Section 6, check 2: citation coverage ----------------------------------------------


def test_citation_coverage_catches_an_uncited_claim():
    """A claim with no source_id is not released. Its numbers are real and its text is
    accurate — coverage is the only thing wrong with it."""
    uncited = Claim(text="Critical recall is 0.059 on 1st_test.", source_ids=[])

    report = verify(draft(uncited), evidence())
    (checked,) = report.claims

    assert checked.failures == (CITATION_COVERAGE,)
    assert report.citation_existence_failed is False, (
        "an uncited claim is not a fabricated citation; conflating them would send every "
        "tier-2 response to tier 3"
    )


def test_a_whitespace_only_source_id_is_not_a_citation():
    report = verify(draft(Claim(text="Something.", source_ids=["  "])), evidence())

    assert report.claims[0].failures == (CITATION_COVERAGE,)


# --- Section 6, check 3: numeric fidelity ------------------------------------------------


def test_numeric_fidelity_catches_a_number_the_cited_chunk_does_not_contain():
    """The most damaging hallucination in this project and the most likely one: a
    plausible-but-wrong metric, cited to a real chunk that really is about that metric."""
    wrong = Claim(text="Critical recall is 0.59 on 1st_test.", source_ids=[CHUNK_ID])

    assert unsupported_numbers(wrong, evidence()) == ("0.59",)

    report = verify(draft(wrong), evidence())
    assert report.claims[0].failures == (NUMERIC_FIDELITY,)
    assert report.citation_existence_failed is False


def test_numeric_fidelity_accepts_a_number_taken_from_any_chunk_the_claim_cites():
    """Section 6: "verbatim in the text of at least one chunk it cites" — per literal, so a
    claim citing two chunks may take one number from each."""
    spanning = Claim(
        text="Critical recall is 0.059 on 1st_test, and the first 50 files are a cold start.",
        source_ids=[CHUNK_ID, OTHER_CHUNK_ID],
    )

    assert unsupported_numbers(spanning, evidence()) == ()


def test_numeric_extraction_does_not_read_digits_out_of_this_repos_identifiers():
    """`2nd_test`, `1st_test` and `ZA-2115` all contain digits. Only the part number's is a
    numeric literal; treating the ordinals as ones would make every claim mentioning an
    experiment fail on a `2` that no chunk needs to contain."""
    assert extract_numbers("2nd_test and 1st_test and 3rd_test") == ()
    assert extract_numbers("order ZA-2115") == ("2115",)
    assert extract_numbers("recall 0.913, 20,480 points, 98.5% agreement") == (
        "0.913",
        "20,480",
        "98.5",
    )


# --- Section 6, check 4: risky-recommendation gating -------------------------------------


def test_risky_recommendation_gating_flags_a_part_and_a_quantity():
    gate = gate_recommendation("Order 1 x ZA-2115 for bearing 2nd_test-demo.")

    assert gate.names_part is True
    assert gate.names_quantity is True
    assert gate.is_risky_shape is True
    assert gate.requires_approval is True


def test_a_part_number_alone_is_not_read_as_its_own_quantity():
    """`ZA-2115` contains digits; reading them as a quantity would make every part mention a
    risky-shaped recommendation and drain the distinction of meaning."""
    gate = gate_recommendation("Inspect the ZA-2115 housing at the next opportunity.")

    assert gate.names_part is True
    assert gate.names_quantity is False
    assert gate.is_risky_shape is False
    assert gate.requires_approval is True, (
        "a recommendation is a suggestion for a human to approve whatever shape it has"
    )


def test_no_recommendation_requires_no_approval():
    gate = gate_recommendation(None)

    assert gate.requires_approval is False
    assert gate.is_risky_shape is False


def test_a_risky_recommendation_is_displayed_and_flagged_rather_than_suppressed():
    """Section 6: "may be displayed, but the response is marked as requiring approval and
    cannot itself be an order"."""
    response = assemble(
        draft(GOOD_CLAIM, recommendation="Order 1 x ZA-2115."), evidence()
    )

    assert response.requires_approval is True
    assert "Order 1 x ZA-2115." in response.text
    assert response.recommendation_gate.is_risky_shape is True


# --- The evidence seam: real tool payloads ----------------------------------------------


def _chunk(**overrides) -> RetrievedChunk:
    fields = {
        "chunk_id": CHUNK_ID,
        "source_type": "decision_doc",
        "source_id": "docs/model_training_decision.md",
        "source_ref": "docs/model_training_decision.md",
        "heading_path": "Model Training Decision > 6. The headline result",
        "chunk_index": 7,
        "text": CHUNK_TEXT,
        "score": 0.62,
    }
    return RetrievedChunk(**{**fields, **overrides})


def test_evidence_is_built_from_the_real_tool_result_shape():
    """Not a hand-written dict: the payload comes from #111's own `search_documentation`, so
    the citable set is assembled from what the tool layer actually mints."""
    payload = payload_of(search_documentation("critical recall", search=lambda *a, **k: [_chunk()]))

    turn = TurnEvidence.from_tool_payloads([payload])

    assert CHUNK_ID in turn.source_ids
    assert turn.texts_for(CHUNK_ID) == (CHUNK_TEXT,)
    assert turn.retrieval_scores == (0.62,)
    assert any("model_training_decision" in title for title in turn.document_pointers())


def test_a_failed_tool_result_still_contributes_its_id_and_message():
    """A failure is an observation a claim may legitimately cite ("the prediction service is
    not reachable"), so its id is citable and its message is the text behind it."""
    payload = payload_of(search_documentation("anything", search=_raises))

    turn = TurnEvidence.from_tool_payloads([payload])

    assert turn.source_ids
    assert any("not reachable" in text for text in turn.texts_for(next(iter(turn.source_ids))))


def _raises(*args, **kwargs):
    raise RuntimeError("qdrant is down")


# --- Section 6, step 4: retrieval confidence ---------------------------------------------


def test_the_thresholds_are_the_calibrated_values_from_the_sweep():
    """Issue #163's sweep (published in PR #164) recommended this pair; #165 applied it.
    "Starting values" no longer describes them — the numbers below are measured."""
    assert TAU_TOP == 0.75
    assert TAU_SUPPORT == 0.70
    assert MIN_SUPPORTING_CHUNKS == 2


def test_the_module_docstring_states_the_thresholds_are_calibrated_and_the_ceiling_is_7_of_8():
    """The successor to the tripwire that pinned "the starting values are 0.45 and 0.35".
    That test existed because the constants were only defensible while the document still
    called them uncalibrated, and its own docstring said the calibrating issue was expected
    to move it deliberately. #165 is that issue, so this is where it moved to: the constants
    are now only defensible while the module says what they were measured against, and says
    that 7/8 — not 8/8 — is the answerable ceiling at this pair, so nobody reads the missing
    item as calibration left undone when it is a retrieval-quality finding.
    """
    docstring = retrieval_confidence_module.__doc__ or ""

    assert "calibrated values, measured, not guessed" in docstring
    assert "#163" in docstring and "#164" in docstring
    assert "answerable ceiling is 7/8, not 8/8" in docstring
    assert "overlap by top score" in docstring, "the ceiling is stated with its reason"
    assert "0.7015" in docstring, "the answerable item that is lost"
    for must_refuse_score in ("0.7131", "0.7194", "0.7323"):
        assert must_refuse_score in docstring, "the must-refuse items it scores below"
    assert "starting values, not decisions" not in docstring


def test_retrieval_passes_only_with_a_strong_top_and_a_second_supporting_chunk():
    """Expressed relative to the thresholds rather than in literals, matching
    `tests/test_agent_golden_set_retrieval.py`: what is asserted is the *shape* of the
    predicate, which #163's calibration does not change, and a re-calibration should not
    have to re-derive these numbers by hand a second time."""
    assert assess_retrieval([TAU_TOP + 0.02, TAU_SUPPORT + 0.01]).passed is True
    assert (
        assess_retrieval([TAU_TOP + 0.02, TAU_SUPPORT - 0.15]).passed is False
    ), "one supporting chunk is not two"
    assert (
        assess_retrieval([TAU_TOP - 0.01, TAU_SUPPORT + 0.03, TAU_SUPPORT + 0.01]).passed
        is False
    ), "top is below TAU_TOP"
    assert (
        assess_retrieval([TAU_TOP, TAU_SUPPORT]).passed is True
    ), "the thresholds are inclusive"


def test_a_turn_that_retrieved_nothing_is_not_thereby_below_threshold():
    """The implementation reading flagged in `retrieval_confidence.py`: a question answered
    from a live tool performs no vector search, and demoting every such answer to `partial`
    would empty the tier of meaning."""
    confidence = assess_retrieval([])

    assert confidence.performed is False
    assert confidence.below_threshold is False
    assert confidence.passed is True


# --- Section 6: the three tiers ----------------------------------------------------------


def test_a_fully_verified_draft_with_strong_retrieval_is_grounded():
    response = assemble(draft(GOOD_CLAIM, OTHER_GOOD_CLAIM), evidence())

    assert response.grounding_tier == GROUNDED
    assert [claim.text for claim in response.claims] == [
        GOOD_CLAIM.text,
        OTHER_GOOD_CLAIM.text,
    ]
    assert response.dropped == ()
    assert CHUNK_ID in response.text


def test_one_failed_claim_degrades_to_partial_and_releases_the_rest():
    bad = Claim(text="Critical recall is 0.59 on 1st_test.", source_ids=[CHUNK_ID])

    response = assemble(draft(OTHER_GOOD_CLAIM, bad), evidence())

    assert response.grounding_tier == PARTIAL
    assert [claim.text for claim in response.claims] == [OTHER_GOOD_CLAIM.text]
    assert [dropped.text for dropped in response.dropped] == [bad.text]


def test_a_tier_2_response_never_silently_drops_a_claim():
    """Section 6's named invariant. Dropping quietly would leave the user believing the
    question was fully answered, which is worse than the ungrounded answer it avoids."""
    bad = Claim(text="Critical recall is 0.59 on 1st_test.", source_ids=[CHUNK_ID])
    uncited = Claim(text="The rig ran at 2000 rpm.", source_ids=[])

    response = assemble(draft(OTHER_GOOD_CLAIM, bad, uncited), evidence())

    assert response.grounding_tier == PARTIAL
    assert UNSOURCED_PREFIX in response.text
    for dropped in response.dropped:
        assert dropped.text in response.text, "a dropped claim that is not named is silent"
        assert dropped.reasons, "a dropped claim with no reason cannot be reviewed"
    assert response.human_pointer in response.text


def test_weak_retrieval_alone_degrades_to_partial():
    """Borderline-but-real retrieval: the top chunk fell short of `TAU_TOP`, but both chunks
    cleared the corroboration floor, so there is a real if partial body of evidence.

    The scores were `(0.30, 0.10)` before Issue #177 and are borderline now, deliberately.
    Under #177's condition a turn where *nothing* reached `TAU_SUPPORT` is refused outright,
    so the old literals no longer describe tier 2 — they describe the new tier-3 case, which
    `test_retrieval_that_reached_nothing_at_all_is_refused_rather_than_released` asserts on
    its own. This test keeps its original job by moving to the regime it was always about.
    """
    response = assemble(
        draft(GOOD_CLAIM), evidence(scores=(TAU_TOP - 0.02, TAU_SUPPORT + 0.01))
    )

    assert response.grounding_tier == PARTIAL
    assert response.retrieval.below_threshold is True
    assert response.retrieval.supporting_count == MIN_SUPPORTING_CHUNKS, (
        "borderline-but-real means the corroboration floor was cleared, only TAU_TOP missed"
    )
    assert response.dropped == (), "nothing failed a check; retrieval is what was weak"
    assert response.claims, "verified claims are still released at tier 2"


def test_a_fabricated_citation_sends_the_whole_response_to_tier_3():
    """Section 6's tier table, taken as written: "no claim survives verification, **or the
    citation-existence check failed**". One invented id is evidence about the draft, not
    only about the claim carrying it."""
    fabricated = Claim(text="Recall is 0.059.", source_ids=["docs/invented.md::1"])

    response = assemble(draft(GOOD_CLAIM, fabricated), evidence())

    assert response.grounding_tier == UNGROUNDED
    assert response.claims == ()


def test_the_ungrounded_tier_is_one_fixed_response_with_pointers_not_answers():
    uncited = Claim(text="Critical recall is 0.059 on 1st_test.", source_ids=[])

    response = assemble(draft(uncited), evidence())

    assert response.grounding_tier == UNGROUNDED
    assert response.text.startswith(UNGROUNDED_ANSWER)
    assert response.claims == ()
    assert response.document_pointers, "tier 3 offers document titles as pointers"
    assert uncited.text not in response.text, (
        "the fixed response does not restate the claims it could not verify"
    )
    assert response.human_pointer in response.text


def test_the_ungrounded_tier_withholds_the_recommendation():
    """Releasing a suggested action under "I don't have a sourced answer for this" would be
    answering un-grounded in the one place it matters most."""
    uncited = Claim(text="The bearing is failing.", source_ids=[])

    response = assemble(draft(uncited, recommendation="Order 1 x ZA-2115."), evidence())

    assert response.grounding_tier == UNGROUNDED
    assert response.recommendation is None
    assert response.requires_approval is False
    assert "ZA-2115" not in response.text
    assert response.recommendation_gate.is_risky_shape is True, (
        "the gate's verdict is still on the response for the trace"
    )


def test_the_critic_never_hard_fails():
    """Section 6 rules out a hard failure as firmly as it rules out an un-grounded answer.
    Every one of these is degenerate in a different way and every one produces a response."""
    cases = [
        (draft(), TurnEvidence()),
        (draft(Claim(text="", source_ids=[])), TurnEvidence()),
        (draft(GOOD_CLAIM), TurnEvidence()),
        (draft(GOOD_CLAIM, recommendation=""), evidence()),
        (draft(Claim(text="0.059", source_ids=[CHUNK_ID, CHUNK_ID])), evidence()),
        (draft(GOOD_CLAIM, unanswered=["what the third bearing did"]), evidence()),
    ]

    for one_draft, one_evidence in cases:
        response = assemble(one_draft, one_evidence)
        assert response.grounding_tier in (GROUNDED, PARTIAL, UNGROUNDED)
        assert isinstance(response.text, str) and response.text


def test_an_empty_draft_is_ungrounded_rather_than_grounded_on_nothing():
    response = assemble(draft(), evidence())

    assert response.grounding_tier == UNGROUNDED


def test_the_draft_s_own_unanswered_parts_are_carried_into_the_response():
    response = assemble(draft(GOOD_CLAIM, unanswered=["the 3rd_test failure mode"]), evidence())

    assert response.unanswered == ("the 3rd_test failure mode",)
    assert "the 3rd_test failure mode" in response.text


# --- Section 6: the escalation rule -------------------------------------------------------


def test_lexical_overlap_is_containment_of_the_claims_own_terms():
    assert lexical_overlap("Critical recall is 0.059 on 1st_test.", CHUNK_TEXT) == 1.0
    assert lexical_overlap("", CHUNK_TEXT) == 0.0
    assert 0.0 < lexical_overlap(
        "Recall on inner-race failures cannot be distinguished from a rig idiosyncrasy.",
        CHUNK_TEXT,
    ) < LEXICAL_OVERLAP_FLOOR


def test_nothing_escalates_when_neither_trigger_fires():
    one = draft(GOOD_CLAIM)
    report = verify(one, evidence())

    assert escalations_needed(one, report, evidence()) == ()


def test_a_recommendation_escalates_every_verified_claim():
    """Section 6's trigger (a): an actionable claim is where the cost of being wrong is
    highest, so the whole released answer is put to the entailment check."""
    one = draft(GOOD_CLAIM, OTHER_GOOD_CLAIM, recommendation="Order 1 x ZA-2115.")
    report = verify(one, evidence())

    requests = escalations_needed(one, report, evidence())

    assert [request.claim_index for request in requests] == [0, 1]
    assert {request.trigger for request in requests} == {"recommendation"}


def test_low_lexical_overlap_escalates_on_its_own():
    """Section 6's trigger (b): the claim cites a real chunk, passes every deterministic
    check, and shares almost no vocabulary with it — the case the deterministic layer cannot
    tell from coincidence."""
    coincidental = Claim(
        text="Inner-race defects propagate faster than outer-race defects.",
        source_ids=[CHUNK_ID],
    )
    one = draft(coincidental)
    report = verify(one, evidence())

    (request,) = escalations_needed(one, report, evidence())

    assert report.clean is True, "the deterministic layer has no complaint about this claim"
    assert request.trigger == "lexical_overlap"
    assert request.overlap < LEXICAL_OVERLAP_FLOOR
    assert request.source_id == CHUNK_ID


def test_a_failing_deterministic_pass_escalates_nothing():
    """Section 6 makes a clean deterministic pass a precondition. Paying for a model call to
    re-examine a draft that is already being degraded adds cost and non-determinism to a
    decision that is made."""
    bad = Claim(text="Critical recall is 0.59 on 1st_test.", source_ids=[CHUNK_ID])
    one = draft(bad, recommendation="Order 1 x ZA-2115.")
    report = verify(one, evidence())

    assert report.clean is False
    assert escalations_needed(one, report, evidence()) == ()


def test_the_escalated_pair_is_the_best_matching_cited_chunk():
    """If the chunk most likely to support the claim does not, none of the others will."""
    spanning = Claim(
        text="The server never refuses to score during the first 50 files.",
        source_ids=[CHUNK_ID, OTHER_CHUNK_ID],
    )
    one = draft(spanning, recommendation="Keep replaying.")
    report = verify(one, evidence())

    (request,) = escalations_needed(one, report, evidence())

    assert request.source_id == OTHER_CHUNK_ID


# --- Issue #119: trigger (b) is scoped to prose chunks ------------------------------------
#
# A live-tool result as `TurnEvidence` actually renders it: the text is the JSON
# serialization of `data`, which is why an English claim scores low against it however well
# the payload supports the claim.
LIVE_ID = "GET /monitoring/drift"
LIVE_TEXT = json.dumps(
    {"bearing_id": "2nd_test-demo", "file_count": 197, "baseline_status": "stable"},
    indent=2,
    sort_keys=True,
)
LIVE_CLAIM = Claim(
    text="Bearing 2nd_test-demo has been scored on 197 windows so far.",
    source_ids=[LIVE_ID],
)


def live_evidence() -> TurnEvidence:
    return TurnEvidence(
        (EvidenceItem(LIVE_ID, LIVE_TEXT, source_type="live_endpoint"),)
    )


def test_prose_source_types_are_section_2s_two_chunk_categories():
    assert PROSE_SOURCE_TYPES == {"decision_doc", "public_reference"}


def test_a_claim_citing_only_live_tool_json_is_not_overlap_checked_at_all():
    """Issue #119's core correction, on #117's own measured shape. The claim is supported by
    the payload it cites — `file_count` really is 197 — and scores far below the floor
    because prose is being compared against JSON keys. Trigger (b) must not fire."""
    one = draft(LIVE_CLAIM)
    report = verify(one, live_evidence())

    assert report.clean is True
    assert lexical_overlap(LIVE_CLAIM.text, LIVE_TEXT) < LEXICAL_OVERLAP_FLOOR, (
        "the overlap really is below the floor — the fix is that it is not consulted"
    )
    assert escalations_needed(one, report, live_evidence()) == ()


def test_trigger_a_still_escalates_a_live_only_claim():
    """The documented consequence of the above: trigger (a) remains available. A live-cited
    claim in a draft carrying a recommendation is still put to the critic, against the JSON
    it cites — scoping applies to the overlap measure, not to what (a) may check."""
    one = draft(LIVE_CLAIM, recommendation="Order 1 x ZA-2115.")
    report = verify(one, live_evidence())

    (request,) = escalations_needed(one, report, live_evidence())

    assert request.trigger == "recommendation"
    assert request.source_id == LIVE_ID


def test_a_mixed_claim_is_measured_against_its_prose_source_only():
    """Rule 2: the JSON source is ignored for this check. Here the prose chunk is below the
    floor and the JSON scores higher — the claim must still escalate, because a high-scoring
    payload may not mask a prose chunk that falls below the floor."""
    mixed = Claim(
        text="Inner-race defects propagate faster than outer-race defects.",
        source_ids=[LIVE_ID, CHUNK_ID],
    )
    combined = TurnEvidence(evidence().items + live_evidence().items)
    one = draft(mixed)
    report = verify(one, combined)

    (request,) = escalations_needed(one, report, combined)

    assert request.trigger == "lexical_overlap"
    assert request.source_id == CHUNK_ID, "measured against the prose chunk, not the JSON"


def test_a_live_source_cannot_trigger_escalation_for_a_well_supported_prose_claim():
    """The same rule in the other direction: a claim whose prose chunk clears the floor is
    not dragged below it by a low-scoring JSON source it also cites. Before #119 the best
    pair was taken across every cited source, so this depended on which scored higher."""
    supported = Claim(
        text="Critical recall is 0.059 on 1st_test.", source_ids=[CHUNK_ID, LIVE_ID]
    )
    combined = TurnEvidence(evidence().items + live_evidence().items)
    one = draft(supported)
    report = verify(one, combined)

    assert lexical_overlap(supported.text, CHUNK_TEXT) >= LEXICAL_OVERLAP_FLOOR
    assert escalations_needed(one, report, combined) == ()


def test_the_floor_itself_is_unchanged_by_this_issue():
    """#119 fixes what is compared, not the number compared against. Section 8's golden set
    owns the value."""
    assert LEXICAL_OVERLAP_FLOOR == 0.6


def test_evidence_without_a_source_type_is_not_treated_as_prose():
    """Pinned because it is a real edge, not because it happens in production:
    `from_tool_payloads` always carries a type, and Section 2 makes `source_type` mandatory
    on every minted result. Hand-built evidence that omits it is not prose by the literal
    reading of the rule — recorded here so the behaviour is a decision rather than a
    surprise."""
    untyped = TurnEvidence((EvidenceItem(CHUNK_ID, CHUNK_TEXT),))
    coincidental = Claim(
        text="Inner-race defects propagate faster than outer-race defects.",
        source_ids=[CHUNK_ID],
    )
    one = draft(coincidental)

    assert escalations_needed(one, verify(one, untyped), untyped) == ()


def test_source_type_survives_from_tool_payloads_at_both_levels():
    """The datum the scoping depends on is carried, not re-derived from the id's shape."""
    payloads = [
        {
            "source": {"source_type": "live_endpoint", "source_id": "SEARCH prognos_docs"},
            "data": {
                "results": [
                    {
                        "source": {
                            "source_type": "decision_doc",
                            "source_id": CHUNK_ID,
                            "source_ref": "docs/model_training_decision.md",
                        },
                        "text": CHUNK_TEXT,
                        "score": 0.79,
                    }
                ]
            },
        }
    ]

    built = TurnEvidence.from_tool_payloads(payloads)

    assert [(item.source_id, item.source_type) for item in built.items] == [
        ("SEARCH prognos_docs", "live_endpoint"),
        (CHUNK_ID, "decision_doc"),
    ]


# --- Issue #171: the concept-domain relevance check ----------------------------------------
#
# The gap #119 left named and open, closed by a check that reads *subject matter* rather than
# containment: a claim citing only live-tool/inventory sources had no relevance check at all.
# Every test below is tier 1 in the strict sense — the module cannot make a call, and these
# assert on set operations over hand-built drafts and one real recorded payload shape.

# The off-domain claim is `golden_set_corpus.py`'s `corpus-refuse-oil-temperature-setpoint`
# case, in the shape the answerer would have to write it to reach the critic at all: no number
# (so numeric fidelity has nothing to fail on) and a real, citable, minted id. Section 8 calls
# this one's trap adjacency — the system does have a monitoring surface, and it has no notion
# of temperature whatsoever.
OFF_DOMAIN_LIVE_CLAIM = Claim(
    text="The oil temperature alarm setpoint for this machine is configured at the rig.",
    source_ids=[LIVE_ID],
)


def test_the_registrys_ids_are_the_ones_the_tool_layer_actually_mints():
    """Anti-drift, and the reason the registry may not simply import them: Section 5 forbids
    the critic from importing the tool layer at all (asserted in a clean interpreter above), so
    the ids are literals there and this test is what keeps them honest. A renamed endpoint or
    source id fails here rather than silently un-registering a source and disabling the check
    for it."""
    from src.agent.mcp.serving_client import DRIFT_ENDPOINT, PREDICT_ENDPOINT
    from src.agent.mcp.tools import (
        DOCS_SOURCE_ID,
        INVENTORY_SOURCE_ID,
        ORDER_SOURCE_ID,
        trajectory_source_id,
    )

    assert REGISTERED_SOURCE_IDS == {
        DRIFT_ENDPOINT,
        PREDICT_ENDPOINT,
        INVENTORY_SOURCE_ID,
        ORDER_SOURCE_ID,
    }
    assert DRIFT_SOURCE_ID == DRIFT_ENDPOINT
    assert PREDICT_SOURCE_ID == PREDICT_ENDPOINT
    assert INVENTORY_DOMAIN_SOURCE_ID == INVENTORY_SOURCE_ID
    assert INVENTORY_ORDERS_SOURCE_ID == ORDER_SOURCE_ID
    # The two deliberate omissions, pinned as omissions rather than left to be inferred from
    # the absence of a line: the docs-search wrapper has no fixed subject matter, and the
    # archive's id carries a content hash that moves whenever the archive is rebuilt.
    assert domain_for(DOCS_SOURCE_ID) is None
    assert domain_for(trajectory_source_id()) is None


def test_a_claim_within_its_cited_tools_domain_is_not_demoted():
    """#117's own recorded claim, which is exactly the shape that must survive: it cites only
    live-tool JSON, `"file_count": 197` supports it, and `scored`/`windows` are what
    `/monitoring/drift` reports on."""
    one = draft(LIVE_CLAIM)
    report = verify(one, live_evidence())
    result = check_claim_domain(LIVE_CLAIM, live_evidence())

    assert report.clean is True
    assert result.checked is True
    assert result.matched >= {"scored", "windows"}
    assert result.off_domain is False
    assert off_domain_demotions(report, live_evidence()) == {}


def test_a_claim_outside_its_cited_tools_domain_is_demoted():
    """The must-refuse mechanism, isolated: real data, a real id, every deterministic check
    passed, and nothing in the claim is about what the source reports."""
    one = draft(OFF_DOMAIN_LIVE_CLAIM)
    report = verify(one, live_evidence())
    result = check_claim_domain(OFF_DOMAIN_LIVE_CLAIM, live_evidence())

    assert report.clean is True, "it must fail this check and no other"
    assert result.checked is True
    assert result.matched == frozenset()
    assert off_domain_demotions(report, live_evidence()) == {
        0: f"{OFF_DOMAIN_REASON} (checked against {LIVE_ID})"
    }


def test_a_demotion_drops_the_claim_and_names_the_source_it_was_checked_against():
    """Routed through `demotions`, so `grounding.py`'s tier logic is untouched by this issue:
    the claim is dropped and named exactly like an LLM-demoted one, and its reason says which
    source the check consulted."""
    one = draft(OFF_DOMAIN_LIVE_CLAIM, OTHER_GOOD_CLAIM)
    combined = TurnEvidence(live_evidence().items + evidence().items)
    report = verify(one, combined)

    response = assemble(
        one, combined, report=report, demotions=off_domain_demotions(report, combined)
    )

    assert response.grounding_tier == PARTIAL
    assert [claim.text for claim in response.claims] == [OTHER_GOOD_CLAIM.text]
    (dropped,) = response.dropped
    assert dropped.text == OFF_DOMAIN_LIVE_CLAIM.text
    assert dropped.reasons == (f"{OFF_DOMAIN_REASON} (checked against {LIVE_ID})",)
    assert UNSOURCED_PREFIX in response.text


def test_a_claim_citing_a_prose_source_alongside_is_not_checked_at_all():
    """The issue's own boundary: this check does not touch the prose path. The same off-domain
    text, with a prose chunk cited beside the live source, is *not evaluated* here — trigger
    (b) and the LLM tier own that case, and evaluating it twice under two different measures is
    how the prose path's own weak-citation signal would get masked."""
    mixed = Claim(
        text=OFF_DOMAIN_LIVE_CLAIM.text, source_ids=[LIVE_ID, CHUNK_ID]
    )
    combined = TurnEvidence(evidence().items + live_evidence().items)
    one = draft(mixed)
    report = verify(one, combined)
    result = check_claim_domain(mixed, combined)

    assert result.checked is False, "not checked-and-passed — not evaluated"
    assert result.off_domain is False
    assert off_domain_demotions(report, combined) == {}


def test_a_claim_citing_an_unregistered_live_source_is_not_checked_either():
    """Fail-open on anything the closed registry has no opinion about, including when it is
    cited *alongside* a registered source: having no opinion about one of a claim's sources
    means having no opinion about the claim."""
    docs_wrapper = EvidenceItem(
        "SEARCH prognos_docs", json.dumps({"result_count": 0}), source_type="live_endpoint"
    )
    combined = TurnEvidence(live_evidence().items + (docs_wrapper,))
    unregistered_only = Claim(
        text=OFF_DOMAIN_LIVE_CLAIM.text, source_ids=["SEARCH prognos_docs"]
    )
    alongside = Claim(
        text=OFF_DOMAIN_LIVE_CLAIM.text, source_ids=[LIVE_ID, "SEARCH prognos_docs"]
    )

    assert check_claim_domain(unregistered_only, combined).checked is False
    assert check_claim_domain(alongside, combined).checked is False
    assert off_domain_demotions(verify(draft(unregistered_only, alongside), combined), combined) == {}


def test_a_claim_citing_two_registered_sources_may_be_about_either_one():
    """The union, not the intersection: a claim citing the monitoring read and the parts table
    is about one of them or the other, and requiring both domains to match would demote every
    multi-source claim."""
    parts_item = EvidenceItem(
        INVENTORY_DOMAIN_SOURCE_ID,
        json.dumps({"parts": [{"part_number": "ZA-2115", "quantity_on_hand": 4}]}),
        source_type="inventory",
    )
    both = Claim(
        text="ZA-2115 is the part number stocked for this housing.",
        source_ids=[LIVE_ID, INVENTORY_DOMAIN_SOURCE_ID],
    )
    combined = TurnEvidence(live_evidence().items + (parts_item,))

    result = check_claim_domain(both, combined)

    assert result.checked is True
    assert result.source_ids == (LIVE_ID, INVENTORY_DOMAIN_SOURCE_ID)
    assert result.matched >= {"part", "number", "stocked"}


def test_a_claim_already_failing_a_deterministic_check_gets_no_second_reason():
    """One problem is reported once. The claim below fails numeric fidelity and is off-domain;
    it is dropped by the check that fired first, and this module adds nothing to it."""
    fabricated = Claim(
        text="The oil temperature alarm setpoint is 82 degrees.", source_ids=[LIVE_ID]
    )
    one = draft(fabricated)
    report = verify(one, live_evidence())

    assert report.rejected and NUMERIC_FIDELITY in report.rejected[0].failures
    assert off_domain_demotions(report, live_evidence()) == {}


def test_the_domains_carry_no_word_that_appears_in_every_claim():
    """A domain containing `bearing` could not discriminate anything in this project — every
    claim this agent makes is about a bearing. `bearing_type` stays, because it is a real
    column of the parts table and a claim quoting it is a claim about inventory."""
    for source_id, domain in CONCEPT_DOMAINS.items():
        assert "bearing" not in domain, source_id
        assert "bearings" not in domain, source_id
        assert domain, source_id
        assert all(term == term.lower() and term.strip() for term in domain), source_id


def test_the_registrys_domains_are_disjoint_from_the_must_refuse_vocabulary():
    """The check's whole purpose, stated over Section 8's own must-refuse questions rather than
    over one hand-picked example: none of the words that make those questions unanswerable is
    in any domain, so a claim built out of them cannot match one."""
    unanswerable = {
        "torque", "bolt", "screws", "grease", "lubricant", "relubricated", "nlgi",
        "inspected", "serviced", "alignment", "dial-indicate", "lockout", "tagout",
        "de-energize", "osha", "oil", "temperature", "setpoint", "alarm", "iso",
        "velocity", "mm/s", "damage", "race", "wear",
    }
    for source_id, domain in CONCEPT_DOMAINS.items():
        assert domain.isdisjoint(unanswerable), f"{source_id}: {domain & unanswerable}"


def test_the_check_cannot_make_a_call_at_all():
    """"No LLM call anywhere in this check" as a property of the source, not of a promise: a
    module with no `async def` and no `await` in it cannot reach `messages.create`, which is
    the only way this package ever calls a model."""
    tree = ast.parse((CRITIC_DIR / "relevance.py").read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        assert not isinstance(node, (ast.AsyncFunctionDef, ast.Await)), ast.dump(node)
    assert "anthropic" not in (CRITIC_DIR / "relevance.py").read_text(encoding="utf-8")


# --- The escalated call, against a recording stub -----------------------------------------


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    def __init__(self, text: str) -> None:
        self.content = [_TextBlock(text)]


class _RecordingMessages:
    def __init__(self, verdicts) -> None:
        self.calls: list[dict] = []
        self._verdicts = list(verdicts)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        verdict = self._verdicts.pop(0) if self._verdicts else "yes"
        return _Message(json.dumps({"verdict": verdict}))


class _RecordingClient:
    """Everything `judge_async` is allowed to touch. A client with no tool surface at all —
    if the critic reached for one, this would raise rather than quietly succeed."""

    def __init__(self, *verdicts: str) -> None:
        self.messages = _RecordingMessages(verdicts)


def test_the_escalated_request_carries_no_tool_argument_of_any_kind():
    """Section 5's "B holds no tools at all", asserted on the request the critic actually
    builds rather than on a comment above it."""
    client = _RecordingClient("yes")
    requests = _one_request()

    asyncio.run(escalate_async(requests, client=client))

    (kwargs,) = client.messages.calls
    for forbidden in ("tools", "tool_choice", "mcp_servers", "betas", "container"):
        assert forbidden not in kwargs, f"the critic passed {forbidden!r}"
    assert kwargs["model"] == MODEL
    assert kwargs["max_tokens"] == MAX_TOKENS
    assert kwargs["thinking"] == THINKING
    assert kwargs["output_config"]["effort"] == EFFORT
    assert kwargs["output_config"]["format"]["schema"] == verdict_schema()
    assert "stream" not in kwargs, "Section 1: non-streaming for every call in the chain"


def test_the_critic_sees_one_claim_and_one_chunk_and_no_question():
    """Section 5's "Sees" row: one claim + its cited chunk, at a time."""
    client = _RecordingClient("yes")

    asyncio.run(escalate_async(_one_request(), client=client))

    (kwargs,) = client.messages.calls
    (message,) = kwargs["messages"]
    assert message["role"] == "user"
    assert CHUNK_TEXT in message["content"]
    assert OTHER_CHUNK_TEXT not in message["content"]
    assert "?" in message["content"].split("Statement:")[-1]


def test_both_spans_reach_the_critic_inside_an_untrusted_data_envelope():
    """Section 5: the critic reads untrusted text — a draft a model wrote and chunks the
    harness does not control — so the same chokepoint #113 built applies here."""
    envelope = UntrustedEnvelope(nonce="0" * 32)

    (message,) = build_messages(
        "a claim", "a chunk", source_id=CHUNK_ID, envelope=envelope
    )

    assert message["content"].count(f"<{TAG} ") == 2
    assert message["content"].count(envelope.closing_tag) == 2
    assert f'source_id="{CHUNK_ID}"' in message["content"]


def test_a_payload_cannot_close_the_critics_envelope_early():
    from tests.fixtures.adversarial_payloads import CASE_9_ENVELOPE_BREAKOUT

    envelope = UntrustedEnvelope(nonce="0" * 32)

    (message,) = build_messages(
        "a claim", CASE_9_ENVELOPE_BREAKOUT, source_id=CHUNK_ID, envelope=envelope
    )

    assert message["content"].count(envelope.closing_tag) == 2, (
        "the payload's own closing delimiter was not neutralised"
    )


@pytest.mark.parametrize("verdict, demoted", [("yes", False), ("no", True), ("unclear", True)])
def test_no_and_unclear_demote_and_yes_does_not(verdict, demoted):
    """Section 6: `no` or `unclear` demotes that claim; it does not rewrite it."""
    client = _RecordingClient(verdict)

    demotions = asyncio.run(escalate_async(_one_request(), client=client))

    assert (0 in demotions) is demoted


def test_a_demotion_drops_the_claim_and_never_edits_it():
    """"Its verdict is a gate, not an edit" (Section 5). The claim leaves the response
    whole, in `dropped`, with the reason attached — no rewritten text anywhere."""
    coincidental = Claim(
        text="Inner-race defects propagate faster than outer-race defects.",
        source_ids=[CHUNK_ID],
    )
    one = draft(coincidental, OTHER_GOOD_CLAIM)

    response = assemble(one, evidence(), demotions={0: "the cited source does not support it"})

    assert response.grounding_tier == PARTIAL
    assert [claim.text for claim in response.claims] == [OTHER_GOOD_CLAIM.text]
    assert [dropped.text for dropped in response.dropped] == [coincidental.text]
    assert response.dropped[0].text == coincidental.text, "the text is unchanged"


def test_a_verdict_outside_the_three_values_is_a_loud_failure_not_a_guess():
    """A gate that invents a verdict when the response is malformed is not a gate."""
    with pytest.raises(Exception):
        parse_verdict([_TextBlock(json.dumps({"verdict": "probably"}))])
    with pytest.raises(ValueError):
        parse_verdict([])

    assert parse_verdict([_TextBlock(json.dumps({"verdict": "no"}))]) == Entailment(verdict="no")


def _one_request():
    coincidental = Claim(
        text="Inner-race defects propagate faster than outer-race defects.",
        source_ids=[CHUNK_ID],
    )
    one = draft(coincidental)
    return escalations_needed(one, verify(one, evidence()), evidence())


# --- Section 5 / Section 10 case 6: the critic holds no tools ------------------------------


def test_the_critic_package_cannot_reach_the_tool_layer_at_all():
    """The structural claim, made the way #111 makes its own: in a real interpreter, not by
    reading the source. Importing every critic module must not pull in `mcp`, `src.agent.mcp`
    or even the Anthropic SDK — a critic that cannot import a client cannot hold a session.
    """
    script = (
        "import json, sys\n"
        + "".join(f"import {module}\n" for module in CRITIC_MODULES)
        + "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'mcp', 'anthropic'} or m.startswith('src.agent.mcp'))))\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [], (
        "the critic pulled a tool-layer or SDK module into sys.modules just by being imported"
    )


def test_no_critic_module_names_a_tool_argument_anywhere():
    """The static half: an argument the source never writes is an argument no call can pass.
    Cheap, and it fails at review time rather than at the moment a critic reaches for
    something it should never have been offered."""
    forbidden = {"tools", "tool_choice", "mcp_servers"}

    for path in sorted(CRITIC_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in forbidden:
                raise AssertionError(f"{path.name} passes {node.arg!r} to a call")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                module = getattr(node, "module", "") or ""
                assert not module.startswith("src.agent.mcp"), path.name
                assert module.split(".")[0] != "mcp", path.name
                assert all(name.split(".")[0] != "mcp" for name in names), path.name


def test_the_critic_never_opens_a_session_or_launches_a_server():
    """The names that would make a critic a second answerer, none of which appear."""
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(CRITIC_DIR.glob("*.py"))
    )

    for name in (
        "stdio_client",
        "ClientSession",
        "readonly_tools",
        "readonly_server_params",
        "tool_runner",
        "place_order",
    ):
        assert name not in source, f"the critic names {name!r}"


# --- Issue #177: retrieval that returned no evidence at all reaches tier 3 ------------------
#
# Section 6's tier table sends every below-threshold turn to `partial`, which makes a
# must-refuse question releasable: it retrieves five tangential chunks, cites one, states
# numbers that really are in it, and is published with a recommendation attached. `_tier`
# now carves out the narrowest sub-case of `below_threshold` — retrieval ran and not one
# chunk cleared even `TAU_SUPPORT` — and refuses it.
#
# The condition is measured, not chosen: `tests/fixtures/measure_no_evidence_floor.py` reads
# it off the golden set over the free path. What the tests below pin is the *boundary*, on
# both sides, and the two carve-outs that keep it from reaching an answer it has no business
# refusing.


def test_retrieval_that_reached_nothing_at_all_is_refused_rather_than_released():
    """The case Issue #177 was filed about, in the shape a must-refuse turn actually takes:
    nothing failed a deterministic check. The claim is cited, the citation exists, and its
    number really is in the chunk it cites — the only thing wrong with the turn is that no
    chunk it retrieved was good enough to corroborate anything."""
    response = assemble(
        draft(GOOD_CLAIM, recommendation="Order 1 x ZA-2115."),
        evidence(scores=(TAU_SUPPORT - 0.01, TAU_SUPPORT - 0.05)),
    )

    assert response.report.clean is True, (
        "no deterministic check failed; retrieval is the issue"
    )
    assert response.retrieval.supporting_count == 0
    assert response.grounding_tier == UNGROUNDED
    assert response.claims == (), "a refused turn releases nothing"
    assert response.recommendation is None, "and withholds the recommendation with it"
    assert response.text.startswith(UNGROUNDED_ANSWER)


def test_borderline_retrieval_one_step_above_the_boundary_is_still_partial():
    """The other side of the same boundary, one supporting chunk away. `TAU_SUPPORT` is a
    floor a *corroborating* chunk has to clear; a turn with one chunk over it has evidence,
    just not enough of it, and Section 6's tier 2 is exactly what that is for."""
    response = assemble(
        draft(GOOD_CLAIM), evidence(scores=(TAU_SUPPORT + 0.01, TAU_SUPPORT - 0.05))
    )

    assert response.retrieval.supporting_count == 1
    assert response.retrieval.below_threshold is True
    assert response.grounding_tier == PARTIAL
    assert response.claims, "one real chunk is still a partial answer, not a refusal"


def test_a_live_tool_only_answer_is_untouched_by_the_new_condition():
    """The non-regression Issue #177 names as a hard constraint. A question answered from
    `get_bearing_status` performs no vector search at all, so `performed` is False and there
    is nothing for a retrieval condition to be weak about. Demoting these was already ruled
    out in `retrieval_confidence.py`; refusing them would be worse."""
    response = assemble(draft(LIVE_CLAIM), live_evidence())

    assert response.retrieval.performed is False
    assert response.retrieval.supporting_count == 0, (
        "the count really is zero — it is `performed` that keeps the condition off this turn"
    )
    assert response.grounding_tier == GROUNDED
    assert [claim.text for claim in response.claims] == [LIVE_CLAIM.text]


def test_a_live_tool_claim_is_not_refused_because_a_co_occurring_search_came_back_empty():
    """`performed=False`'s carve-out closed through the other door. A turn may call a live
    tool *and* search; the claim below rests entirely on the live payload, and a documentation
    search that returned nothing usable says nothing about whether `file_count` is 197."""
    mixed = TurnEvidence(live_evidence().items + evidence(scores=(0.30, 0.10)).items)

    response = assemble(draft(LIVE_CLAIM), mixed)

    assert response.retrieval.performed is True
    assert response.retrieval.supporting_count == 0
    assert response.grounding_tier == PARTIAL, (
        "weak retrieval still demotes the turn — it must not refuse a live-grounded claim"
    )
    assert [claim.text for claim in response.claims] == [LIVE_CLAIM.text]


def test_a_claim_citing_a_live_source_alongside_a_weak_chunk_is_not_refused_either():
    """The mixed-citation reading, stated as a test because it is a choice: a claim carrying
    both a live id and a chunk id is *partly* live, and the conservative reading of partly is
    the one that does not refuse."""
    mixed_claim = Claim(
        text="Bearing 2nd_test-demo has been scored on 197 windows so far.",
        source_ids=[LIVE_ID, CHUNK_ID],
    )
    mixed = TurnEvidence(live_evidence().items + evidence(scores=(0.30, 0.10)).items)

    response = assemble(draft(mixed_claim), mixed)

    assert response.grounding_tier == PARTIAL
    assert response.claims


def test_the_new_condition_can_only_ever_turn_a_partial_into_a_refusal():
    """The containment property that makes this additive rather than a re-calibration: it is
    a strict subset of `below_threshold`, so no `grounded` response can reach it. True by
    arithmetic — `passed` needs `MIN_SUPPORTING_CHUNKS` chunks over the floor and this fires
    only at zero — and asserted over a grid rather than argued, because the day someone lowers
    `MIN_SUPPORTING_CHUNKS` to 1 is the day the argument stops holding silently."""
    steps = [
        TAU_SUPPORT - 0.20, TAU_SUPPORT - 0.01, TAU_SUPPORT, TAU_TOP - 0.01, TAU_TOP + 0.05
    ]

    for first in steps:
        for second in steps:
            confidence = assess_retrieval([first, second])
            if confidence.performed and confidence.supporting_count == 0:
                assert confidence.passed is False, (
                    f"({first}, {second}) would be refused while retrieval passed"
                )


def test_the_condition_records_the_measurement_it_was_chosen_from():
    """Same tripwire `test_the_module_docstring_states_the_thresholds_are_calibrated...`
    applies to `TAU_TOP`: a boundary in this package is only defensible while the code still
    says what it was measured against and what it was measured *instead of*. The numbers below
    are `tests/fixtures/measure_no_evidence_floor.py`'s reading on the committed corpus."""
    # Line-wrapped prose, so the assertions read a whitespace-normalized copy rather than
    # depending on where the docstring happens to break.
    docstring = " ".join((grounding_module._no_evidence_at_all.__doc__ or "").split())

    assert "measure_no_evidence_floor.py" in docstring, "the script that produced the numbers"
    assert "fires on 5 of Section 8's 8 must-refuse items and on 0 of its 8 answerable ones" \
        in docstring
    for overlapping in ("0.7131", "0.7194", "0.7323", "0.7015"):
        assert overlapping in docstring, (
            "#163's overlap, which is why top score is not the axis"
        )
    assert "would catch 6 of 8 rather than 5, and is rejected" in docstring, (
        "the rejected alternative, with its own count"
    )
    assert "+0.0049" in docstring, "the margin, stated rather than smoothed over"


def test_the_module_table_names_the_third_ungrounded_condition():
    """`grounding.py` opens with Section 6's tier table, and the table is what a reader checks
    the code against. A third route to `ungrounded` that the table does not mention would make
    that opening quietly wrong."""
    docstring = grounding_module.__doc__ or ""

    assert "retrieval ran and returned no evidence at all" in docstring
    assert "#177" in docstring
    assert "addition to Section 6's table rather than a reading of" in docstring
