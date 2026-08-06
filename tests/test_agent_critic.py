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
    PARTIAL,
    UNGROUNDED,
    UNGROUNDED_ANSWER,
    UNSOURCED_PREFIX,
    assemble,
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


def evidence(*, scores: tuple[float, float] = (0.62, 0.41)) -> TurnEvidence:
    """Two retrieved chunks, both citable, with retrieval that clears the thresholds."""
    return TurnEvidence(
        (
            EvidenceItem(CHUNK_ID, CHUNK_TEXT, score=scores[0], title="model_training_decision"),
            EvidenceItem(
                OTHER_CHUNK_ID, OTHER_CHUNK_TEXT, score=scores[1], title="serving_design"
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


def test_the_thresholds_are_section_6s_starting_values():
    assert TAU_TOP == 0.45
    assert TAU_SUPPORT == 0.35
    assert MIN_SUPPORTING_CHUNKS == 2


def test_the_design_document_still_calls_the_thresholds_starting_values():
    """The constants above are only defensible while the document still says they are
    uncalibrated. If Section 8's issue calibrates them and rewords this, that issue is
    expected to move this test too — deliberately, not by accident."""
    text = DESIGN_DOC.read_text(encoding="utf-8")

    assert "**The starting values are 0.45 and 0.35, and they are explicitly\nstarting values**" in text


def test_retrieval_passes_only_with_a_strong_top_and_a_second_supporting_chunk():
    assert assess_retrieval([0.62, 0.41]).passed is True
    assert assess_retrieval([0.62, 0.20]).passed is False, "one supporting chunk is not two"
    assert assess_retrieval([0.44, 0.40, 0.38]).passed is False, "top is below TAU_TOP"
    assert assess_retrieval([0.45, 0.35]).passed is True, "the thresholds are inclusive"


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
    response = assemble(draft(GOOD_CLAIM), evidence(scores=(0.30, 0.10)))

    assert response.grounding_tier == PARTIAL
    assert response.retrieval.below_threshold is True
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
