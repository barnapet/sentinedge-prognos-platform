"""Agent B's deterministic layer (Issue #114, `docs/agent_design.md` Section 6, step 3).

Four checks, all of them pure string/set operations over a draft and this turn's tool
results. No API key, no model call, no network -- Section 8's tier 1 is a hard requirement
here, not a preference, because this is the layer the LLM critic is explicitly *not* a
substitute for:

- **Citation existence.** Every `source_id` a claim cites must literally appear among the
  ids this turn's tool results carried. A model cannot verify an id exists in its own
  transcript more reliably than a set membership test can.
- **Citation coverage.** A claim with no `source_id` is not released; it is dropped and
  named (`grounding.py` assembles the naming).
- **Numeric fidelity.** Every numeric literal in a claim's text must appear verbatim in the
  text of at least one chunk the claim cites. Section 6 calls this the strongest cheap check
  available in *this* project specifically: the answers quote numbers, and a
  plausible-but-wrong metric is both the most damaging hallucination here and the most
  likely one.
- **Risky-recommendation gating.** A recommendation naming a part and a quantity may be
  displayed, but it is marked as requiring approval and it is never itself an order.

**The draft's schema is not re-declared here.** `Claim` and `Draft` are #113's own models
(`src/agent/answerer.py`), imported under `TYPE_CHECKING` only: this module reads a draft,
it never constructs one, so the runtime dependency is unnecessary -- and dropping it is what
keeps the whole critic package free of the answerer's tool wiring. That is Section 5's "B
holds no tools at all" as an import-graph property rather than a promise;
`tests/test_agent_critic.py` asserts it by importing every critic module in a clean
interpreter and checking `mcp` never appears in `sys.modules`. The schema is still pinned
against the real thing: every tier-1 test builds real `Draft` and `Claim` objects.

**Evidence comes in as parsed tool-result payloads**, in `src/agent/mcp/results.py`'s two
shapes. Section 6's step 1 is that the tool layer mints the ids and the model never does, so
the set this module tests membership against is assembled from the payloads the harness
received -- not from anything the model wrote. Wiring the answerer to hand its tool results
forward is the orchestrator's job (a later issue); `TurnEvidence.from_tool_payloads` is the
seam.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, see the module docstring
    from src.agent.answerer import Claim, Draft

# The four checks, named once. Reported per claim rather than as booleans on a report, so a
# reason string and the check that produced it cannot drift apart.
CITATION_EXISTENCE = "citation_existence"
CITATION_COVERAGE = "citation_coverage"
NUMERIC_FIDELITY = "numeric_fidelity"
RISKY_RECOMMENDATION = "risky_recommendation"

# One numeric literal: digits, optional thousands separators, optional decimal part.
#
# The boundaries are what make this usable on this repo's own vocabulary. `2nd_test`,
# `1st_test` and `M1-EDA` must not contribute a bare `2`/`1` -- a digit followed by a word
# character is not a numeric literal -- while `0.913`, `20,480`, `98.5%` and the `2115` in
# `ZA-2115` are. The leading `.` in the lookbehind keeps `1.2` from also yielding `2`.
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")

# A part number as this project's inventory actually spells them (`src/agent/inventory/
# seed/parts.csv`): two or more capitals, then one or more `-`-joined uppercase-alphanumeric
# segments -- ZA-2115, BRG-6205-2RS, LUBE-NLGI2-400G, COLLAR-SHAFT-LOCK-25MM.
#
# It over-matches other SHOUTED-HYPHENATED text, and that is the safe direction: a false
# positive marks a recommendation as requiring approval, which is where every recommendation
# ends up anyway (see `gate_recommendation`).
_PART_NUMBER = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)+\b")

_NUMBER_WORDS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
)
_QUANTITY = re.compile(
    r"\b(?:\d+|" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE
)


# --- Evidence: this turn's tool results, as the checks need to see them -----------------


@dataclass(frozen=True)
class EvidenceItem:
    """One citable thing this turn produced: an id, the text behind it, and -- for a
    retrieved chunk -- the similarity that retrieved it and a title to point at."""

    source_id: str
    text: str
    score: float | None = None
    title: str = ""


@dataclass(frozen=True)
class TurnEvidence:
    """Everything this turn's tool results made citable.

    Order is preserved as received. Duplicate ids are kept rather than collapsed: two
    searches can legitimately return the same chunk, and a check that reads "at least one
    cited chunk contains this" wants every copy.
    """

    items: tuple[EvidenceItem, ...] = ()

    @property
    def source_ids(self) -> frozenset[str]:
        """Exactly the ids that may be cited this turn (Section 6, step 1)."""
        return frozenset(item.source_id for item in self.items)

    def texts_for(self, source_id: str) -> tuple[str, ...]:
        """The text behind one id. Empty for an id this turn never produced."""
        return tuple(item.text for item in self.items if item.source_id == source_id)

    @property
    def retrieval_scores(self) -> tuple[float, ...]:
        """The similarities of the retrieved chunks, most similar first. Empty when this
        turn performed no retrieval at all -- which is a different thing from retrieving
        badly, and `retrieval_confidence.py` treats it as one."""
        return tuple(
            sorted(
                (item.score for item in self.items if item.score is not None),
                reverse=True,
            )
        )

    def document_pointers(self, limit: int = 3) -> tuple[str, ...]:
        """Titles of the best-scoring retrieved documents, for a tier-3 response to offer
        **as pointers, not as answers** (Section 6's degraded path)."""
        ranked = sorted(
            (item for item in self.items if item.score is not None and item.title),
            key=lambda item: item.score or 0.0,
            reverse=True,
        )
        seen: list[str] = []
        for item in ranked:
            if item.title not in seen:
                seen.append(item.title)
        return tuple(seen[:limit])

    @classmethod
    def from_tool_payloads(
        cls, payloads: Iterable[Mapping[str, Any]]
    ) -> "TurnEvidence":
        """Build the citable set from `src/agent/mcp/results.py`'s payloads.

        Two levels, because the tool layer mints ids at two levels. Every result carries a
        top-level `source.source_id` (`GET /monitoring/drift`, `data/agent/inventory.db`,
        ...), and a `search_documentation` result additionally carries one `source.source_id`
        per retrieved chunk -- the chunk ids Section 6's existence check is really about.
        Both are citable, so both are here.

        A failed result contributes its `error` message as text rather than being skipped:
        the id was still produced this turn, and a claim citing it is citing a real
        observation ("the prediction service is not reachable").
        """
        items: list[EvidenceItem] = []
        for payload in payloads:
            source = payload.get("source") or {}
            source_id = source.get("source_id")
            data = payload.get("data")
            if source_id:
                text = (
                    payload["error"]
                    if "error" in payload
                    else json.dumps(data, indent=2, default=str, sort_keys=True)
                )
                items.append(EvidenceItem(source_id=str(source_id), text=text))
            for result in (data or {}).get("results", []) if isinstance(data, dict) else []:
                chunk_source = result.get("source") or {}
                chunk_id = chunk_source.get("source_id")
                if not chunk_id:
                    continue
                ref = str(chunk_source.get("source_ref", ""))
                heading = str(result.get("heading_path", ""))
                items.append(
                    EvidenceItem(
                        source_id=str(chunk_id),
                        text=str(result.get("text", "")),
                        score=(
                            float(result["score"]) if result.get("score") is not None else None
                        ),
                        title=" — ".join(part for part in (ref, heading) if part),
                    )
                )
        return cls(tuple(items))


# --- The four checks -------------------------------------------------------------------


def extract_numbers(text: str) -> tuple[str, ...]:
    """Every numeric literal in `text`, in order, without repeats.

    Unsigned on purpose: a leading `-` is as often a hyphen or a range as a minus, and
    treating it as part of the literal would fail a claim whose source writes the same
    number without one. Stated as a limitation rather than fixed by guessing -- the check
    is a substring test, and a sign error is one of the things it does not catch.
    """
    seen: list[str] = []
    for match in _NUMBER.finditer(text):
        if match.group() not in seen:
            seen.append(match.group())
    return tuple(seen)


def cited_source_ids(claim: "Claim") -> tuple[str, ...]:
    """A claim's citations, ignoring blanks. A whitespace-only id is not a citation."""
    return tuple(source_id for source_id in claim.source_ids if source_id.strip())


def unknown_source_ids(claim: "Claim", evidence: TurnEvidence) -> tuple[str, ...]:
    """Citation existence: the cited ids this turn never produced."""
    known = evidence.source_ids
    return tuple(
        source_id for source_id in cited_source_ids(claim) if source_id not in known
    )


def is_uncited(claim: "Claim") -> bool:
    """Citation coverage: whether this claim carries no usable citation at all."""
    return not cited_source_ids(claim)


def unsupported_numbers(claim: "Claim", evidence: TurnEvidence) -> tuple[str, ...]:
    """Numeric fidelity: the literals in the claim that no cited chunk contains verbatim.

    Per literal, across the chunks the claim cites -- Section 6's wording is "must appear
    verbatim in the text of at least one chunk it cites", so a claim citing two chunks may
    take one number from each.
    """
    texts = [text for sid in cited_source_ids(claim) for text in evidence.texts_for(sid)]
    return tuple(
        number
        for number in extract_numbers(claim.text)
        if not any(number in text for text in texts)
    )


@dataclass(frozen=True)
class RecommendationGate:
    """What the risky-recommendation check decided about a draft's `recommendation`."""

    recommendation: str | None
    names_part: bool = False
    names_quantity: bool = False

    @property
    def is_risky_shape(self) -> bool:
        """Section 6's shape: a recommendation naming *both* a part and a quantity."""
        return self.names_part and self.names_quantity

    @property
    def requires_approval(self) -> bool:
        """True for **any** recommendation present.

        Deliberately wider than `is_risky_shape`, and not a re-decision of Section 6: the
        section says a part-and-quantity recommendation is marked as requiring approval, and
        says nothing that makes a vaguer one executable. `src/agent/answerer.py` already
        defines a recommendation as "a suggestion for a human to approve", and the executor
        (Agent C, a later issue) cannot act without an out-of-band token regardless. So the
        narrow shape is reported separately, and the flag a response carries is the wide
        one.
        """
        return self.recommendation is not None


def gate_recommendation(recommendation: str | None) -> RecommendationGate:
    """Risky-recommendation gating. Displays, flags, and never converts to an order.

    The part number is removed from the text before the quantity is looked for, so the
    `2115` in `ZA-2115` is not read as "order 2115 of them".
    """
    if recommendation is None:
        return RecommendationGate(None)
    without_parts = _PART_NUMBER.sub(" ", recommendation)
    return RecommendationGate(
        recommendation=recommendation,
        names_part=bool(_PART_NUMBER.search(recommendation)),
        names_quantity=bool(_QUANTITY.search(without_parts)),
    )


# --- The report ------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckedClaim:
    """One claim after the deterministic pass, with its position in the draft kept.

    The index is what lets the LLM tier (`escalation.py`) demote a specific claim without
    matching on its text, and what lets `grounding.py` report drops in draft order.
    """

    index: int
    claim: "Claim"
    failures: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class DeterministicReport:
    """The whole deterministic pass over one draft."""

    claims: tuple[CheckedClaim, ...] = ()
    recommendation: RecommendationGate = field(
        default_factory=lambda: RecommendationGate(None)
    )

    @property
    def verified(self) -> tuple[CheckedClaim, ...]:
        return tuple(checked for checked in self.claims if checked.passed)

    @property
    def rejected(self) -> tuple[CheckedClaim, ...]:
        return tuple(checked for checked in self.claims if not checked.passed)

    @property
    def citation_existence_failed(self) -> bool:
        """Whether any claim cited an id this turn never produced.

        A response-level fact, not a per-claim one, because Section 6's tier table makes it
        one: "no claim survives verification, **or the citation-existence check failed**"
        sends the whole response to tier 3. A fabricated citation anywhere is evidence about
        the whole draft.
        """
        return any(CITATION_EXISTENCE in checked.failures for checked in self.claims)

    @property
    def clean(self) -> bool:
        """Whether every claim passed every check -- the precondition Section 6 puts on
        escalating to the LLM critic at all."""
        return all(checked.passed for checked in self.claims)


def check_claim(index: int, claim: "Claim", evidence: TurnEvidence) -> CheckedClaim:
    """Run the three claim-level checks on one claim.

    Coverage short-circuits the other two: with no citations there is nothing to test
    existence or numbers against, and reporting "cites nothing" plus "its numbers are in
    none of the nothing it cites" is one failure told twice. Existence does **not**
    short-circuit numeric fidelity -- a claim citing one real id and one invented one still
    has real text to check its numbers against.
    """
    if is_uncited(claim):
        return CheckedClaim(
            index=index,
            claim=claim,
            failures=(CITATION_COVERAGE,),
            reasons=("the claim carries no source_id",),
        )

    failures: list[str] = []
    reasons: list[str] = []

    unknown = unknown_source_ids(claim, evidence)
    if unknown:
        failures.append(CITATION_EXISTENCE)
        reasons.append(
            "cites "
            + ", ".join(repr(source_id) for source_id in unknown)
            + ", which this turn's tool results never produced"
        )

    unsupported = unsupported_numbers(claim, evidence)
    if unsupported:
        failures.append(NUMERIC_FIDELITY)
        reasons.append(
            "states "
            + ", ".join(unsupported)
            + ", which appears verbatim in none of the sources it cites"
        )

    return CheckedClaim(index=index, claim=claim, failures=tuple(failures), reasons=tuple(reasons))


def verify(draft: "Draft", evidence: TurnEvidence) -> DeterministicReport:
    """The deterministic pass, run on every response before any LLM critic (Section 6)."""
    return DeterministicReport(
        claims=tuple(
            check_claim(index, claim, evidence) for index, claim in enumerate(draft.claims)
        ),
        recommendation=gate_recommendation(draft.recommendation),
    )


def evidence_from_payloads(payloads: Sequence[Mapping[str, Any]]) -> TurnEvidence:
    """Alias kept for readability at call sites that have a list of payloads in hand."""
    return TurnEvidence.from_tool_payloads(payloads)
