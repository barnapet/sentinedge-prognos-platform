"""The one live call through Agent B's escalated check (Issue #114,
`docs/agent_design.md` Sections 1, 5 and 6).

**This is not Section 8's golden set**, which is a separate, later issue, and it is not
evidence that the critic judges entailment well. It exists to show that the escalated call
is wired correctly against the real API: it returns one of Section 6's three verdicts, and
it attempts no tool call. Everything else about the critic is asserted without a model in
`tests/test_agent_critic.py`, where it cannot flake.

The pair put to it is deliberately one **the deterministic layer cannot settle**: the claim
cites a real chunk, every number in it appears in that chunk, and the chunk still does not
support it — the shape Section 6 names when it explains why deterministic-only is not
enough ("a claim can cite a real chunk, pass every check above, and still not be supported
by it"). The chunk is `docs/model_training_decision.md`'s own finding that *n* = 1 cannot
separate "fails on inner-race failures" from "fails on this bearing", and the claim is that
distinction collapsed.

**The verdict is printed, not asserted beyond the three values.** A test that required a
specific verdict from a model would be asserting the model's judgement, which is exactly the
non-determinism Section 6 keeps out of the deterministic layer; what is asserted is that the
gate got a usable answer and reached for nothing while getting it.

**Skips cleanly without an API key**, following #113's pattern: the reason string says what
is missing and how to supply it, so a skip in a CI log is self-explaining. A skip is not a
pass, and a run that skipped this should say so.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from src.agent.answerer import Claim, Draft
from src.agent.critic.deterministic import EvidenceItem, TurnEvidence, verify
from src.agent.critic.escalation import (
    LEXICAL_OVERLAP_FLOOR,
    VERDICTS,
    escalations_needed,
    judge_async,
    lexical_overlap,
)
from src.agent.critic.grounding import PARTIAL, assemble

CHUNK_ID = "docs/model_training_decision.md::6"

# Verbatim from `docs/model_training_decision.md` §6, which is in the real corpus.
CHUNK_TEXT = (
    "The 1st_test fold has two independent failures. The threshold-transfer problem "
    "destroys Critical recall and is not fixable by the estimator: all 17 of its Critical "
    "rows sit below the lowest rms_ratio its training fold ever labelled Critical, so no "
    "boundary learned from the other two folds can reach them. n = 1 cannot separate "
    "'fails on inner-race failures' from 'fails on this particular bearing'."
)

# Cites that chunk, quotes only numbers that chunk contains, and states as settled exactly
# what the chunk says cannot be settled. Every deterministic check passes on it.
UNSUPPORTED_CLAIM = Claim(
    text=(
        "The model fails on inner-race failures, and all 17 of the Critical rows in "
        "1st_test show this."
    ),
    source_ids=[CHUNK_ID],
)


def _has_anthropic_credentials() -> bool:
    """Whether the SDK has *something* to authenticate with.

    Same check as `tests/test_agent_answerer_live.py`: an unset `ANTHROPIC_API_KEY` does not
    by itself mean there are no credentials, and skipping on a machine where the test would
    have run is the opposite of what a gate is for.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    config_dir = Path(
        os.environ.get("ANTHROPIC_CONFIG_DIR", Path.home() / ".config" / "anthropic")
    )
    return (config_dir / "credentials").is_dir() and any(
        (config_dir / "credentials").glob("*.json")
    )


requires_api_key = pytest.mark.skipif(
    not _has_anthropic_credentials(),
    reason=(
        "requires Anthropic credentials (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an "
        "`ant auth login` profile): this is the single live model call for Issue #114, and "
        "it is deliberately the only test in the critic's suite that needs one. Every other "
        "assertion about the critic runs without credentials in tests/test_agent_critic.py"
    ),
)


class _NoToolClient:
    """The real SDK client, wrapped so the request it sends can be inspected afterwards.

    It records rather than intercepts — the call that goes out is the one the critic built,
    unmodified — so "no tool argument was sent" is asserted about a request the API actually
    answered, not about a stub.
    """

    def __init__(self) -> None:
        from anthropic import AsyncAnthropic

        self._inner = AsyncAnthropic()
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return await self._inner.messages.create(**kwargs)


def _evidence() -> TurnEvidence:
    return TurnEvidence((EvidenceItem(CHUNK_ID, CHUNK_TEXT, score=0.58, title="§6"),))


def test_the_pair_is_one_the_deterministic_layer_genuinely_cannot_settle():
    """Runs without credentials, on purpose: if this ever stops holding, the live test below
    is no longer testing what it claims to test, and that should be visible in every run
    rather than only in the ones that have a key."""
    one = Draft(claims=[UNSUPPORTED_CLAIM], recommendation=None, unanswered=[])
    report = verify(one, _evidence())

    assert report.clean is True, "the deterministic checks have no complaint about this claim"
    assert lexical_overlap(UNSUPPORTED_CLAIM.text, CHUNK_TEXT) >= LEXICAL_OVERLAP_FLOOR, (
        "the claim reuses the chunk's own vocabulary — string overlap cannot tell this "
        "from support, which is why it needs the escalated check"
    )


@requires_api_key
def test_one_real_escalated_check_returns_a_verdict_and_calls_no_tool(capsys):
    client = _NoToolClient()

    entailment = asyncio.run(
        judge_async(
            UNSUPPORTED_CLAIM.text,
            CHUNK_TEXT,
            source_id=CHUNK_ID,
            client=client,
        )
    )

    assert entailment.verdict in VERDICTS

    (kwargs,) = client.calls
    for forbidden in ("tools", "tool_choice", "mcp_servers"):
        assert forbidden not in kwargs, f"the critic sent {forbidden!r} to the API"

    with capsys.disabled():
        print("\n--- live escalated check ---")
        print(f"claim:   {UNSUPPORTED_CLAIM.text}")
        print(f"chunk:   {CHUNK_ID}")
        print(f"verdict: {entailment.verdict}")


@requires_api_key
def test_a_live_demotion_degrades_the_response_rather_than_rewriting_it(capsys):
    """The whole escalated path, end to end: trigger, call, demotion, assembled tier. The
    assertion is on the *shape* of the outcome under each verdict, not on which verdict the
    model returns — a gate whose test depends on the model agreeing is not regression-
    testable, which is Section 6's own argument against an LLM-only critic."""
    supported = Claim(
        text="All 17 of 1st_test's Critical rows sit below the lowest rms_ratio its "
        "training fold ever labelled Critical.",
        source_ids=[CHUNK_ID],
    )
    one = Draft(
        claims=[supported, UNSUPPORTED_CLAIM],
        recommendation="Report the 1st_test fold separately rather than averaging it.",
        unanswered=[],
    )
    evidence = _evidence()
    report = verify(one, evidence)
    requests = escalations_needed(one, report, evidence)
    assert requests, "a draft carrying a recommendation escalates its claims"

    client = _NoToolClient()
    from src.agent.critic.escalation import escalate_async

    demotions = asyncio.run(escalate_async(requests, client=client))
    response = assemble(one, evidence, report=report, demotions=demotions)

    for claim in response.claims:
        assert claim.text in (supported.text, UNSUPPORTED_CLAIM.text), (
            "a released claim was rewritten; the critic is a gate, not an editor"
        )
    for dropped in response.dropped:
        assert dropped.text in (supported.text, UNSUPPORTED_CLAIM.text)
        assert dropped.text in response.text, "a dropped claim was not named"
    if demotions:
        assert response.grounding_tier == PARTIAL

    with capsys.disabled():
        print("\n--- live escalation over a two-claim draft ---")
        print(f"escalated: {[request.claim_index for request in requests]}")
        print(f"demotions: {demotions}")
        print(f"tier:      {response.grounding_tier}")
        print(response.text)
