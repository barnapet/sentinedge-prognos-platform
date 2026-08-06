"""Retrieval confidence (Issue #114, `docs/agent_design.md` Section 6, step 4).

Retrieval passes when the top chunk's cosine similarity is at or above `TAU_TOP` **and** at
least `MIN_SUPPORTING_CHUNKS` chunks are at or above the lower `TAU_SUPPORT`.

**The two thresholds below are starting values, not decisions, and this issue does not
calibrate them.** Section 6 is explicit that what is decided now is the *procedure* --
calibrate against Section 8's golden set, choose the pair that keeps every must-refuse item
refusing while maximizing pass rate on the answerable ones, and publish the measured values
and the sweep -- and that the numbers are measured later, by the issue that builds that
golden set. Hand-picking a threshold and never checking it is the failure
`docs/evaluation_protocol.md` §4 pre-empts in a different context; guessing one *here*, from
no data, would be the same mistake wearing a calibration's clothes. So they are named
constants in one place, with this note attached, and nothing in this issue tunes them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.agent.critic.deterministic import TurnEvidence

# --- Section 6's starting values, uncalibrated. Section 8's golden-set issue changes these,
# --- and publishes the sweep that justified the change. Nothing else in this package
# --- hard-codes a similarity threshold.
TAU_TOP = 0.45
TAU_SUPPORT = 0.35

# Section 6: "at least two chunks are at or above a lower TAU_SUPPORT". The top chunk counts
# toward it, since TAU_TOP is above TAU_SUPPORT by construction.
MIN_SUPPORTING_CHUNKS = 2


@dataclass(frozen=True)
class RetrievalConfidence:
    """Whether this turn's retrieval was confident enough to support a `grounded` tier."""

    performed: bool
    top_score: float | None
    supporting_count: int
    passed: bool

    @property
    def below_threshold(self) -> bool:
        """Retrieval ran and did not clear the bar -- the condition Section 6's tier table
        names. A turn that never retrieved is not "below threshold"; see `assess_retrieval`.
        """
        return self.performed and not self.passed


def assess_retrieval(
    scores: Sequence[float],
    *,
    tau_top: float = TAU_TOP,
    tau_support: float = TAU_SUPPORT,
    min_supporting: int = MIN_SUPPORTING_CHUNKS,
) -> RetrievalConfidence:
    """Score this turn's retrieval against the thresholds.

    **A turn that retrieved nothing passes, with `performed=False`**, and that is an
    implementation reading worth stating rather than burying. Section 6's tier table makes
    "retrieval below threshold" a demotion condition; it does not say that a turn which
    never searched is thereby ungrounded. It plainly cannot mean that -- a question answered
    from `get_bearing_status` alone performs no vector search, and demoting every live-tool
    answer to `partial` would make the tier meaningless. Those answers are not unchecked
    either: their citations are live-tool ids, and citation existence and numeric fidelity
    apply to them exactly as they do to a chunk. Flagged in the PR rather than silently
    decided.
    """
    ordered = sorted((float(score) for score in scores), reverse=True)
    if not ordered:
        return RetrievalConfidence(
            performed=False, top_score=None, supporting_count=0, passed=True
        )

    top_score = ordered[0]
    supporting = sum(1 for score in ordered if score >= tau_support)
    return RetrievalConfidence(
        performed=True,
        top_score=top_score,
        supporting_count=supporting,
        passed=top_score >= tau_top and supporting >= min_supporting,
    )


def assess_evidence(
    evidence: TurnEvidence,
    *,
    tau_top: float = TAU_TOP,
    tau_support: float = TAU_SUPPORT,
    min_supporting: int = MIN_SUPPORTING_CHUNKS,
) -> RetrievalConfidence:
    """`assess_retrieval` against the scores a turn's evidence carries."""
    return assess_retrieval(
        evidence.retrieval_scores,
        tau_top=tau_top,
        tau_support=tau_support,
        min_supporting=min_supporting,
    )
