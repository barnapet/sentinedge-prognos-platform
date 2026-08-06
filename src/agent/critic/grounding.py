"""The three-tier degraded response (Issue #114, `docs/agent_design.md` Section 6).

| Tier | Condition | Behaviour |
|---|---|---|
| `grounded` | All claims cited, all citations verified, retrieval above threshold | Release the full answer with its citations |
| `partial` | Some claims uncited or retrieval below threshold | Release **only the verified claims**, plus an explicit "I don't have a sourced answer for: ..." naming what was dropped, plus a pointer to a human |
| `ungrounded` | No claim survives verification, or the citation-existence check failed | One fixed response, plus the titles of the top retrieved documents **as pointers, not as answers**, and a pointer to a human |

Three properties this module exists to hold, all of them Section 6's:

- **It never answers un-grounded.** A claim that failed a check is not in `claims` and its
  text is not in the released prose except as something named as missing.
- **It never hard-fails.** Every input produces a response; there is no path that raises.
  That shape is inherited from `docs/serving_design.md` §3's cold-start decision -- never
  refuse to score, compute against what you have and flag the regime -- and `grounding_tier`
  is the agent's `baseline_status`.
- **A tier-2 response never silently drops a claim.** Dropping one quietly would leave the
  user believing the question was fully answered, which is worse than the ungrounded answer
  it was meant to avoid. So every dropped claim is named in `dropped`, with its reason, and
  its text is in the assembled prose.

**A tier-3 response withholds the recommendation too.** Releasing a suggested action under a
sentence that says there is no sourced answer would be answering un-grounded in the one place
it matters most. The gate's own verdict is still on the response object for the trace; it is
the released text that carries nothing.

**A tier-3 response does not list the dropped claims**, and that is not the same omission.
Tier 3 is Section 6's *one fixed response*: it says plainly that there is no sourced answer,
so nothing is passing silently. Restating each failed claim there would also be the one
place where echoing text is actively harmful -- a claim rejected for a fabricated number
would have that number re-printed under a heading the reader is already primed to skim.
Flagged in the PR rather than silently decided.

The verdict is a gate, not an edit: nothing here rewrites a claim. `demotions` is how the
LLM tier (`escalation.py`) removes one, and removal is the only thing it can do.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from src.agent.critic.deterministic import (
    DeterministicReport,
    RecommendationGate,
    TurnEvidence,
    verify,
)
from src.agent.critic.retrieval_confidence import RetrievalConfidence, assess_evidence

if TYPE_CHECKING:  # pragma: no cover - typing only, see `deterministic.py`'s docstring
    from src.agent.answerer import Draft

GROUNDED = "grounded"
PARTIAL = "partial"
UNGROUNDED = "ungrounded"
TIERS = (GROUNDED, PARTIAL, UNGROUNDED)

# Section 6's tier-3 response, fixed and singular: one sentence, no hedging, no attempt to
# answer anyway.
UNGROUNDED_ANSWER = "I don't have a sourced answer for this."

# Tier 2's naming of what was dropped. The colon is load-bearing -- what follows it is the
# list, and the test that this module never drops a claim silently reads it.
UNSOURCED_PREFIX = "I don't have a sourced answer for:"

POINTERS_PREFIX = "Related documents, as pointers rather than answers:"

HUMAN_POINTER = (
    "Please confirm with a maintenance engineer before acting on this."
)

APPROVAL_NOTE = (
    "This is a recommendation for a human to approve. It is not an order and it places "
    "none."
)

# Retrieval that ran and came back weak is a demotion reason in its own right (Section 6's
# tier table), so it is named in the response like any other.
WEAK_RETRIEVAL_REASON = "retrieval for this question was below the confidence threshold"

# Why a claim was demoted by the LLM tier. `escalation.py` supplies the verdict; the reason
# text lives here so every drop reason is written in one place.
DEMOTED_REASON = "the cited source was judged not to support it"


@dataclass(frozen=True)
class ReleasedClaim:
    """A claim that survived every check, with the citations it survived on."""

    text: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class DroppedClaim:
    """A claim that did not survive, and why. Never released; always named."""

    text: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GroundedResponse:
    """What the critic releases: a tier, the surviving claims, and everything it withheld.

    `grounding_tier` is a field on every response, surfaced to the user and logged in the
    trace -- the same "state it, don't hide it" convention as `baseline_status` (#82) and
    `drift_status` (#90).
    """

    grounding_tier: str
    claims: tuple[ReleasedClaim, ...]
    dropped: tuple[DroppedClaim, ...]
    unanswered: tuple[str, ...]
    recommendation: str | None
    requires_approval: bool
    document_pointers: tuple[str, ...]
    human_pointer: str
    retrieval: RetrievalConfidence
    report: DeterministicReport
    text: str

    @property
    def recommendation_gate(self) -> RecommendationGate:
        return self.report.recommendation


def _dropped(report: DeterministicReport, demotions: Mapping[int, str]) -> tuple[DroppedClaim, ...]:
    dropped: list[DroppedClaim] = []
    for checked in report.claims:
        reasons = list(checked.reasons)
        if checked.index in demotions:
            reasons.append(demotions[checked.index])
        if reasons:
            dropped.append(DroppedClaim(text=checked.claim.text, reasons=tuple(reasons)))
    return tuple(dropped)


def _released(
    report: DeterministicReport, demotions: Mapping[int, str]
) -> tuple[ReleasedClaim, ...]:
    return tuple(
        ReleasedClaim(
            text=checked.claim.text,
            source_ids=tuple(checked.claim.source_ids),
        )
        for checked in report.verified
        if checked.index not in demotions
    )


def _tier(
    report: DeterministicReport,
    released: tuple[ReleasedClaim, ...],
    dropped: tuple[DroppedClaim, ...],
    retrieval: RetrievalConfidence,
) -> str:
    if report.citation_existence_failed or not released:
        return UNGROUNDED
    if dropped or retrieval.below_threshold:
        return PARTIAL
    return GROUNDED


def _assemble_text(
    tier: str,
    released: tuple[ReleasedClaim, ...],
    dropped: tuple[DroppedClaim, ...],
    unanswered: tuple[str, ...],
    gate: RecommendationGate,
    pointers: tuple[str, ...],
    retrieval: RetrievalConfidence,
) -> str:
    lines: list[str] = []

    if tier == UNGROUNDED:
        lines.append(UNGROUNDED_ANSWER)
        if pointers:
            lines.append(POINTERS_PREFIX)
            lines.extend(f"- {pointer}" for pointer in pointers)
        lines.append(HUMAN_POINTER)
        return "\n".join(lines)

    for claim in released:
        citations = ", ".join(claim.source_ids)
        lines.append(f"- {claim.text} [{citations}]")

    missing = [claim.text for claim in dropped] + list(unanswered)
    if retrieval.below_threshold:
        missing.append(WEAK_RETRIEVAL_REASON)
    if missing:
        lines.append(UNSOURCED_PREFIX)
        lines.extend(f"- {item}" for item in missing)
        lines.append(HUMAN_POINTER)

    if gate.recommendation is not None:
        lines.append(f"Recommendation: {gate.recommendation}")
        lines.append(APPROVAL_NOTE)

    return "\n".join(lines)


def assemble(
    draft: "Draft",
    evidence: TurnEvidence,
    *,
    report: DeterministicReport | None = None,
    retrieval: RetrievalConfidence | None = None,
    demotions: Mapping[int, str] | None = None,
) -> GroundedResponse:
    """Assemble one response from a draft and this turn's evidence.

    `report` and `retrieval` are computed here unless supplied; an orchestrator that has
    already run the deterministic pass (to decide whether to escalate) passes its own so the
    checks run once per turn rather than twice.

    `demotions` maps a claim's index in the draft to the reason it was demoted --
    `escalation.py`'s output. A demoted claim is dropped and named exactly like one that
    failed a deterministic check; there is no third state and nothing is rewritten.
    """
    report = report if report is not None else verify(draft, evidence)
    retrieval = retrieval if retrieval is not None else assess_evidence(evidence)
    demotions = dict(demotions or {})

    released = _released(report, demotions)
    dropped = _dropped(report, demotions)
    tier = _tier(report, released, dropped, retrieval)
    pointers = evidence.document_pointers() if tier == UNGROUNDED else ()
    unanswered = tuple(draft.unanswered)

    return GroundedResponse(
        grounding_tier=tier,
        claims=released if tier != UNGROUNDED else (),
        dropped=dropped,
        unanswered=unanswered,
        recommendation=report.recommendation.recommendation if tier != UNGROUNDED else None,
        requires_approval=report.recommendation.requires_approval and tier != UNGROUNDED,
        document_pointers=pointers,
        human_pointer=HUMAN_POINTER,
        retrieval=retrieval,
        report=report,
        text=_assemble_text(
            tier,
            released if tier != UNGROUNDED else (),
            dropped,
            unanswered,
            report.recommendation if tier != UNGROUNDED else RecommendationGate(None),
            pointers,
            retrieval,
        ),
    )
