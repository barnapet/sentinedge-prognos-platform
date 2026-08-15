"""Retrieval confidence (Issue #114, `docs/agent_design.md` Section 6, step 4).

Retrieval passes when the top chunk's cosine similarity is at or above `TAU_TOP` **and** at
least `MIN_SUPPORTING_CHUNKS` chunks are at or above the lower `TAU_SUPPORT`.

**The two thresholds below are calibrated values, measured, not guessed** (Issues #163/#164,
applied by #165). Section 6 fixed the *procedure* -- sweep against Section 8's golden set,
choose the pair that keeps every must-refuse item refusing while maximizing pass rate on the
answerable ones, and publish the measured values and the sweep -- and
`tests/fixtures/calibrate_retrieval.py` is that procedure, run against real retrieval scores
from the 522-chunk `prognos_docs` index. It swept 320 candidate pairs through *this* module's
own `assess_retrieval`, and `TAU_TOP = 0.75` / `TAU_SUPPORT = 0.70` is what it recommended.
Re-run that script -- it needs no API key -- whenever the corpus moves.

**At this pair the golden set's answerable ceiling is 7/8, not 8/8, and no threshold reaches
8/8.** The two classes overlap by top score: `corpus-answerable-health-state-thresholds`
tops out at 0.7015, *below* three must-refuse items (0.7131, 0.7194, 0.7323). Any `tau_top`
low enough to admit that answerable item therefore also admits those three refusals, and
keeping every must-refuse item refusing is Section 8's zero-tolerance constraint, not a term
to trade against pass rate. So the missing item is a retrieval/chunking finding -- the
corpus, the chunking, or `k` -- and moving these constants cannot fix it. A reader who sees
7/8 on the golden set should not read it as calibration left undone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.agent.critic.deterministic import TurnEvidence

# --- Calibrated against Section 8's golden set by Issue #163 (sweep published in PR #164),
# --- applied here by #165. `tau_top` sits 0.0177 above the highest must-refuse top score and
# --- 0.0099 below the lowest answerable one it still admits; `tau_support` is the strongest
# --- corroboration floor that costs no further item. Nothing else in this package hard-codes
# --- a similarity threshold.
TAU_TOP = 0.75
TAU_SUPPORT = 0.70

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
