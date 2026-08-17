"""Agent A → Agent B, end to end (Issue #116, `docs/agent_design.md` Sections 5 and 6).

    response = answer_and_verify("what is the current status of 2nd_test-demo?")
    response.grounding_tier   # "grounded" | "partial" | "ungrounded"
    response.text             # what a person may be shown

The first place in this repo where a question goes in and a verified, tiered answer comes
out. It is deliberately thin — the two agents are built, tested and decided elsewhere, and
nothing here re-implements either of them:

1. The answerer drafts, and the harness keeps the tool results it drafted from (#116's
   `answer_turn_async`).
2. Those payloads become `TurnEvidence` — the set of ids that were genuinely available to
   cite this turn, held by the harness rather than taken from the model's own text
   (Section 6, step 1).
3. The deterministic checks run on every response, before any LLM critic (Section 6, step 3).
4. Retrieval confidence is scored from the same evidence (step 4).
5. The concept-domain check (`relevance.py`, Issue #171) demotes any surviving claim whose
   only sources are live-tool/inventory results that do not report on what it is about. No
   model call, so it runs on every turn — including one with `escalate=False`.
6. The LLM critic runs **only** when Section 6's escalation rule fires — a clean
   deterministic pass plus either a recommendation or a claim whose lexical overlap with its
   cited chunk is below the floor. On a typical documentation question it costs nothing.
7. `grounding.assemble` produces the three-tier response.

**The order is the contract, not an implementation detail.** Deterministic before LLM, and
the LLM only on escalation, is Section 6's answer to "why not LLM-only" — the citation check
must not be the weakest link, the gate must stay regression-testable, and a full model call
per turn is not worth paying when nothing is wrong.

**A failed answerer call is not caught here.** Section 6's "never hard-fails" is about the
grounding verdict — a draft that cannot be verified becomes a tier-3 answer rather than an
error — and it is not a promise that infrastructure failures are swallowed. A tool that fails
returns a payload the draft can be checked against (that path is real and exercised); an
answerer that cannot reach the API at all raises, and a caller that turned that into "I don't
have a sourced answer for this" would be reporting a grounding problem it does not have.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.agent.answerer import AnsweredTurn, Draft, answer_turn_async
from src.agent.critic.deterministic import DeterministicReport, TurnEvidence, verify
from src.agent.critic.escalation import escalate_async, escalations_needed
from src.agent.critic.grounding import GroundedResponse, assemble
from src.agent.critic.relevance import off_domain_demotions
from src.agent.critic.retrieval_confidence import assess_evidence
from src.agent.untrusted import UntrustedEnvelope


async def verify_turn_async(
    turn: AnsweredTurn,
    *,
    critic_client: Any | None = None,
    escalate: bool = True,
) -> GroundedResponse:
    """Run one already-drafted turn through Agent B and return the tiered response.

    Separated from the answerer call so the whole critic half can be exercised on a recorded
    turn with no model call of its own — which is what `tests/test_agent_pipeline.py` does.

    `escalate=False` runs the deterministic tiers only. It is not a way to skip the LLM
    critic in production; it is how a caller with no credentials still gets Section 6's other
    steps — the concept-domain check among them, since it needs no model — and the response
    says which it got through `report.clean` and the demotions it carries.

    **Two demotion sources, merged into the one mapping `assemble` takes.** The deterministic
    concept-domain check runs first and unconditionally; the LLM tier's demotions are merged
    on top. A claim both of them demote keeps **both** reasons, joined, rather than one
    overwriting the other: `grounding.py` reports one reason per drop from this mapping, and a
    silently discarded reason is the same "the user cannot see what happened" failure a
    silently dropped claim would be. The escalation set is deliberately computed from the
    deterministic report alone, unfiltered by the demotions above it — which claims are put to
    the critic stays a property of Section 6's rule, so a recorded turn's model calls do not
    change with this module's registry.
    """
    evidence = TurnEvidence.from_tool_payloads(turn.tool_payloads)
    report: DeterministicReport = verify(turn.draft, evidence)
    retrieval = assess_evidence(evidence)

    demotions: dict[int, str] = off_domain_demotions(report, evidence)
    if escalate:
        requests = escalations_needed(turn.draft, report, evidence)
        if requests:
            for index, reason in (
                await escalate_async(requests, client=critic_client)
            ).items():
                already = demotions.get(index)
                demotions[index] = f"{already}; {reason}" if already else reason

    return assemble(
        turn.draft,
        evidence,
        report=report,
        retrieval=retrieval,
        demotions=demotions,
    )


async def answer_and_verify_async(
    question: str,
    *,
    client: Any | None = None,
    critic_client: Any | None = None,
    serving_url: str | None = None,
    db_path: Path | None = None,
    envelope: UntrustedEnvelope | None = None,
    escalate: bool = True,
) -> GroundedResponse:
    """Answer one question and return only what survived verification.

    `client` and `critic_client` are separate arguments because A and B are separate agents
    with separate request configurations (Section 1: effort `high` and `low`). They default
    to independent SDK clients; passing one object for both is fine and changes nothing about
    the boundary, which is the tool surface, not the transport.
    """
    turn = await answer_turn_async(
        question,
        client=client,
        serving_url=serving_url,
        db_path=db_path,
        envelope=envelope,
    )
    return await verify_turn_async(turn, critic_client=critic_client, escalate=escalate)


def answer_and_verify(question: str, **kwargs: Any) -> GroundedResponse:
    """Synchronous entry point. The MCP stdio transport is async-native, so the async path is
    the real one and this is the thin wrapper around it."""
    import asyncio

    return asyncio.run(answer_and_verify_async(question, **kwargs))


def evidence_for(turn: AnsweredTurn) -> TurnEvidence:
    """The citable set for one turn, for a caller that wants to inspect it directly."""
    return TurnEvidence.from_tool_payloads(turn.tool_payloads)


def turn_from_payloads(draft: Draft, payloads: Sequence[dict[str, Any]]) -> AnsweredTurn:
    """Rebuild an `AnsweredTurn` from a recorded draft and recorded payloads.

    The replay seam: `tests/fixtures/record_answerer_turn.py` writes both halves, and this
    puts them back together in the shape the critic half consumes, so a tier-1 test drives
    the same `verify_turn_async` a live call does.
    """
    return AnsweredTurn(draft, tuple(dict(payload) for payload in payloads))
