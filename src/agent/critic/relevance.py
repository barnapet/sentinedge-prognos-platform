"""The concept-domain relevance check (Issue #171, `docs/agent_design.md` Section 6).

One question, asked with no model at all: **is this claim about the kind of thing the live
source it cites reports?** A claim citing nothing but live-tool or inventory results whose own
content words do not touch that source's concept domain is demoted -- not rewritten, not
re-scored, and not turned into a third state.

**The gap this closes.** Section 6's four deterministic checks (`deterministic.py`) do not
discriminate by `source_type` at all: numeric fidelity passes as long as a claimed number
appears verbatim somewhere in something the claim cites, whatever that source is about.
`escalation.py`'s `lexical_overlap` *is* a relevance measure, and Issue #119 deliberately
scoped it to prose chunks, because applying a prose-containment metric to a serialized JSON
payload scored a perfectly-supported claim 0.333 against the very payload that supported it.
The documented consequence -- "a claim citing only `live_endpoint`/`inventory` sources has no
overlap check run against it at all" -- is exactly where a speculative tool call turns into a
grounded-looking answer: the tool returns real data about something else, and a claim built on
it passes every check this package had. This module is the check for that case, and it is
deliberately *not* the prose measure applied to JSON: it compares the claim's vocabulary
against a **declared domain per source**, never against the payload's text.

**Why a module of its own, rather than a fifth function in `deterministic.py`.** Section 6
names four checks and `deterministic.py`'s docstring describes exactly those four; this is not
one of them. It also does not reach a response the way they do -- a failed deterministic check
lands in a `CheckedClaim.failures`, while this lands in `assemble`'s `demotions`, the same
mechanism the LLM tier uses ("no third state, nothing rewritten"). Keeping it here leaves both
of those module descriptions true, and leaves the four checks countable.

**No model call, no network, no key** -- set membership and string tokenization only, so this
is tier 1 by `docs/agent_design.md` Section 8's rule and runs on every turn at no cost.

**The registry is a starting point, not a final answer.** The domains below were written from
each source's real payload keys and values (`src/serving/api.py`'s `/monitoring/drift` and
`/predict` bodies, `src/agent/inventory/build_db.py`'s two tables) plus the vocabulary a
truthful answer about them actually uses. What they have not been is *measured*: Section 8's
golden set is where a domain that is too narrow (a demoted true claim) or too wide (a
must-refuse item that still passes) shows up, and iterating them against it is expected work,
not a defect discovered later.

Four properties worth stating, because each is a decision rather than an implementation
detail:

- **It fails open, in three separate ways.** A claim citing *any* prose source alongside is
  not checked here (that is the prose path's job, and #119 owns it). A claim citing any source
  the registry has no entry for is not checked either -- an unregistered source is one this
  module has no opinion about, and having no opinion about one of a claim's sources means
  having no opinion about the claim. And only claims that passed the deterministic checks are
  considered at all; a claim already being dropped does not need a second reason.
- **Two sources are deliberately unregistered.** `search_documentation`'s own top-level id has
  no fixed subject matter -- its payload is whatever was queried, and the citable prose inside
  it carries its own ids. The trajectory archive's id is content-hash-suffixed
  (`trajectory_archive@<hash>`), so a closed registry of literal ids cannot key it without
  either hard-coding a hash that moves whenever the archive is rebuilt or introducing prefix
  matching; a claim citing it is therefore not checked here.
- **The rule is "any content word intersects", not a rate.** Deliberately lenient in this
  first version: a false demotion silently deletes a true claim, which is worse than a miss
  that the LLM tier and the numeric check may still catch. Project-wide vocabulary is left out
  of the domains for the same reason in reverse -- `bearing` appears in nearly every claim this
  agent will ever make, so a domain containing it could not discriminate at all.
- **No stemming.** Both forms are listed where both occur (`window`/`windows`,
  `part`/`parts`). A stemmer is a tuning knob, and this module does not tune; the underscored
  payload keys (`file_count`, `quantity_on_hand`) are listed literally, because
  `content_tokens` keeps them as single tokens and a claim quoting a key should match it.

**The source ids are literals here, not imports.** `src/agent/mcp/tools.py` owns them, and the
critic package may not import the tool layer -- Section 5's boundary, asserted in a clean
interpreter by `tests/test_agent_critic.py`. So they are written out, and a test that may
import both sides compares them to the tool layer's own constants; drift between the two fails
there rather than silently un-registering a source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from src.agent.critic.deterministic import (
    DeterministicReport,
    TurnEvidence,
    cited_source_ids,
)
from src.agent.critic.escalation import PROSE_SOURCE_TYPES, content_tokens
from src.agent.critic.grounding import OFF_DOMAIN_REASON

if TYPE_CHECKING:  # pragma: no cover - typing only, see `deterministic.py`'s docstring
    from src.agent.answerer import Claim

# The ids the tool layer mints for its live and inventory sources, as literal strings. See the
# module docstring: the critic cannot import `src/agent/mcp/tools.py`, and a test compares
# these against it so they cannot drift apart unnoticed.
DRIFT_SOURCE_ID = "GET /monitoring/drift"
PREDICT_SOURCE_ID = "POST /predict"
INVENTORY_SOURCE_ID = "data/agent/inventory.db"
INVENTORY_ORDERS_SOURCE_ID = "data/agent/inventory.db::orders"

# The monitoring read: this bearing's (or every tracked bearing's) current state, as
# `/monitoring/drift` reports it -- counts, statuses, the four monitored features and their
# z-scores, and the tally of what has been predicted for it.
_DRIFT_DOMAIN = frozenset(
    """
    drift drifts drifted drifting drift_status nominal
    baseline baselines baseline_status warming_up warming stable
    status state health monitor monitored monitoring
    tracked tracking untracked tracked_bearings
    z z-score z-scores zscore threshold extreme persistent persistently
    feature features rms rms_ratio rms_ratio_latest ratio kurtosis skewness
    skewness_smoothed smoothed vibration signal sensor
    file files file_count count counts window windows score scored scores
    predicted predicted_class_counts prediction predictions class classes label labels
    normal degrading critical
    """.split()
)

# The scoring endpoint: one raw window in, one health-state label out, with the same
# baseline/drift flags the monitoring read carries.
_PREDICT_DOMAIN = frozenset(
    """
    predict predicts predicted prediction predictions score scored scores scoring
    label labels class classes state health
    normal degrading critical
    baseline baseline_status warming_up warming stable
    drift drifting drift_status nominal
    model model_notes notes classifier
    signal window windows vibration
    feature features rms rms_ratio ratio kurtosis skewness skewness_smoothed smoothed
    """.split()
)

# The parts table: what is stocked, at what price, with what lead time, where.
_INVENTORY_DOMAIN = frozenset(
    """
    part parts part_number number numbers sku
    inventory stock stocked stocks catalog item items spare spares
    quantity quantities quantity_on_hand on_hand hand unit units
    price prices unit_price_usd usd cost costs dollars
    lead lead_time_days lead-time days delivery
    location locations warehouse shelf bin store stored
    description descriptions bearing_type match matches match_count
    available availability in-stock out-of-stock
    """.split()
)

# The orders table: the write side's rows. Unreachable from an `AnsweredTurn` today -- it lives
# behind the Executor's separate write-only server -- and registered anyway, so the registry
# describes the source rather than the current wiring.
_ORDERS_DOMAIN = frozenset(
    """
    order orders ordered ordering order_id placed placement
    part parts part_number quantity quantities unit units
    request requested requested_by requester
    approve approved approval approved_by approved_at approver
    created created_at status inventory stock
    """.split()
)

# The closed registry. Keyed by the minted `source_id`, because that is what a claim cites and
# what `TurnEvidence` carries -- not by tool name, which the critic never sees.
CONCEPT_DOMAINS: Mapping[str, frozenset[str]] = {
    DRIFT_SOURCE_ID: _DRIFT_DOMAIN,
    PREDICT_SOURCE_ID: _PREDICT_DOMAIN,
    INVENTORY_SOURCE_ID: _INVENTORY_DOMAIN,
    INVENTORY_ORDERS_SOURCE_ID: _ORDERS_DOMAIN,
}

REGISTERED_SOURCE_IDS = frozenset(CONCEPT_DOMAINS)


def domain_for(source_id: str) -> frozenset[str] | None:
    """This source's concept domain, or `None` for one the registry does not cover.

    `None` rather than an empty set on purpose: "no declared domain" and "a domain nothing can
    match" would demote in opposite directions, and only the first is what an unregistered
    source means.
    """
    return CONCEPT_DOMAINS.get(source_id)


@dataclass(frozen=True)
class DomainCheck:
    """What this check decided about one claim.

    `checked` is the honest three-way distinction the boolean alone would hide: a claim can be
    on-domain, off-domain, or **not evaluated** -- and the third is the common case, since every
    claim citing a prose chunk lands there.
    """

    checked: bool
    source_ids: tuple[str, ...] = ()
    matched: frozenset[str] = frozenset()

    @property
    def off_domain(self) -> bool:
        """Whether this claim was evaluated and shares nothing with its sources' domains."""
        return self.checked and not self.matched


def check_claim_domain(claim: "Claim", evidence: TurnEvidence) -> DomainCheck:
    """Evaluate one claim against the concept domains of the sources it cites.

    Not evaluated (`checked=False`) when the claim cites a prose source alongside, or cites any
    source the registry has no domain for, or cites nothing this turn produced. Otherwise the
    claim's content words are intersected with the **union** of its cited sources' domains: a
    claim citing two live sources is about one of them or the other, and requiring it to match
    both would demote every claim that cites more than one.
    """
    cited = cited_source_ids(claim)
    types = {item.source_type for item in evidence.items if item.source_id in set(cited)}
    if not types.isdisjoint(PROSE_SOURCE_TYPES):
        return DomainCheck(checked=False)

    registered: list[str] = []
    for source_id in cited:
        if source_id not in CONCEPT_DOMAINS:
            return DomainCheck(checked=False)
        if source_id not in registered:
            registered.append(source_id)
    if not registered:
        return DomainCheck(checked=False)

    domain: frozenset[str] = frozenset().union(
        *(CONCEPT_DOMAINS[source_id] for source_id in registered)
    )
    return DomainCheck(
        checked=True,
        source_ids=tuple(registered),
        matched=frozenset(content_tokens(claim.text)) & domain,
    )


def off_domain_demotions(
    report: DeterministicReport, evidence: TurnEvidence
) -> dict[int, str]:
    """The demotions this check produces: claim index to reason, `assemble`'s `demotions` shape.

    Runs over `report.verified` only -- a claim already failing a deterministic check is already
    dropped and named, and a second reason for it would report one problem twice. Unlike
    `escalations_needed` there is **no clean-pass precondition**: that precondition exists to
    avoid paying for a model call on a draft that is already degraded, and this check costs
    nothing, so each surviving claim is judged on its own regardless of what its neighbours did.
    """
    demotions: dict[int, str] = {}
    for checked in report.verified:
        result = check_claim_domain(checked.claim, evidence)
        if result.off_domain:
            demotions[checked.index] = (
                f"{OFF_DOMAIN_REASON} (checked against "
                f"{', '.join(result.source_ids)})"
            )
    return demotions
