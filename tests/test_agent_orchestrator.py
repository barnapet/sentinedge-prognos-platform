"""Tier-1 tests for the orchestrator and its human approval gate (Issue #127,
`docs/agent_design.md` Sections 5, 6, 10 and 11). No API key, no model call, no network.

Every test here drives `run_from_response_async` on a response built by the **real**
`grounding.assemble` from a hand-made draft and hand-made evidence -- the same seam
`pipeline.py` opens between `verify_turn_async` and `answer_and_verify_async`, and for the
same reason: the whole gate-and-execute half is exercisable without a model call, so these
assertions cannot flake on what a model happened to write.

The four burdens Issue #127 names, and where each is carried:

1. **Declining calls neither `mint` nor the executor** -- asserted on a counting store and on
   `SELECT COUNT(*) FROM orders`, never on the terminal transcript.
2. **Approving writes exactly one row, matching what the human approved** -- read back out of
   the real database.
3. **An out-of-scope response never reaches the prompt at all** -- asserted on whether the
   prompt callable was invoked, not merely on the absence of an order.
4. **Section 10 case 5's approval-extraction payloads produce zero rows**, both when they
   arrive as the recommendation text and when they are typed at the gate itself.

The order is placed against a real `write_server.build_server()` sharing a real
`ApprovalTokenStore` and a real `tmp_path` SQLite database -- the tool, the token validation
and the transaction are all the merged, unmodified ones (#124, #125, #126).
"""
from __future__ import annotations

import asyncio
import dataclasses
import sqlite3
from pathlib import Path

import pytest

from src.agent.answerer import Claim, Draft
from src.agent.critic.deterministic import EvidenceItem, TurnEvidence
from src.agent.critic.grounding import GROUNDED, PARTIAL, UNGROUNDED, assemble
from src.agent.executor.approval import ApprovalTokenStore
from src.agent.executor.client import OrderPlaced
from src.agent.inventory.build_db import build_db
from src.agent.orchestrator import (
    AFFIRMATIVE,
    Approved,
    Declined,
    OrchestrationResult,
    build_approval,
    build_executor,
    describe,
    is_affirmative,
    needs_human_approval,
    prompt_for_approval,
    run_from_response_async,
)
from tests.fixtures.adversarial_payloads import (
    CASE_5_APPROVAL_EXTRACTION_ATTEMPTS,
    CASE_5_CLAIMED_AUTHORITY,
    CASE_5_CLAIMED_PRIOR_APPROVAL,
)

# Real text from this repository, cited by a claim that states it without any numeric literal
# -- so the deterministic pass is clean on citation existence, coverage and numeric fidelity
# without the fixture having to reproduce a number verbatim.
CHUNK_ID = "docs/model_training_decision.md::7"
CHUNK_TEXT = (
    "Critical recall is 0.913 / 1.000 on 2nd_test / 3rd_test and 0.059 on 1st_test. "
    "The cross-fold mean of 0.657 describes no fold and should not be quoted as the "
    "project's number."
)
CITED_CLAIM = Claim(
    text="The baseline model is documented as failing on one held-out experiment.",
    source_ids=[CHUNK_ID],
)

# What a recommendation naming a part and a quantity looks like. The values here are the ones
# the tests below prove are *never* used -- the human's typed values are.
RECOMMENDED_PART = "ZA-2115"
RECOMMENDATION = f"Order 5 of part {RECOMMENDED_PART} for bearing 2nd_test-demo."


def _evidence(*, scores: tuple[float, float] = (0.62, 0.41)) -> TurnEvidence:
    """Two citable chunks whose scores clear `TAU_TOP`/`TAU_SUPPORT`, so retrieval passes and
    the tier is decided by the claims rather than by the confidence check."""
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
                "docs/serving_design.md::3",
                "The server never refuses to score; it flags the regime instead.",
                score=scores[1],
                title="serving_design",
                source_type="decision_doc",
            ),
        )
    )


def _response(recommendation: str | None = RECOMMENDATION, *, ungrounded: bool = False):
    """One real `GroundedResponse`, assembled by the real critic code.

    `ungrounded=True` cites an id this turn never produced, which is what
    `citation_existence_failed` keys off -- so the tier is tier 3 for the reason Section 6
    gives, not because a field was set by hand.
    """
    claim = (
        Claim(text="A claim citing an id that was never returned.", source_ids=["never::0"])
        if ungrounded
        else CITED_CLAIM
    )
    draft = Draft(claims=[claim], recommendation=recommendation, unanswered=[])
    return assemble(draft, _evidence())


class SpyTokenStore(ApprovalTokenStore):
    """A real `ApprovalTokenStore` that counts `mint` calls.

    A real one rather than a stub on purpose: the same object is handed to
    `write_server.build_server()`, so `consume` is the merged #124 implementation and an
    approved order really does validate against a really-minted token. Only `mint` is
    observed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mint_calls: list[tuple] = []

    def mint(self, part_number, quantity, bearing_id, approved_by):
        self.mint_calls.append((part_number, quantity, bearing_id, approved_by))
        return super().mint(part_number, quantity, bearing_id, approved_by)


class SpyPrompt:
    """An approval prompt that records whether it was invoked at all.

    Task 3's assertion is "the prompt function is not invoked, not just that no order
    results", so invocation has to be observable rather than inferred.
    """

    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls: list = []

    def __call__(self, response):
        self.calls.append(response)
        return self.outcome

    @property
    def was_invoked(self) -> bool:
        return bool(self.calls)


def _scripted(*lines: str):
    """A `read` for `prompt_for_approval` that returns each line in turn, then raises
    `EOFError` -- which is what a real closed stdin does, and is itself a decline."""
    remaining = list(lines)

    def read(_prompt: str) -> str:
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return read


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "inventory.db"
    build_db(path)
    return path


@pytest.fixture
def executor(db_path):
    """A real write server and the real store it validates against, sharing one process --
    the arrangement `src/agent/orchestrator.py`'s docstring explains and the only one in which
    a minted token can actually be consumed."""
    session, store = build_executor(db_path=db_path, token_store=SpyTokenStore())
    return session, store


def _real_part(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT part_number FROM parts ORDER BY part_number LIMIT 1"
        ).fetchone()[0]
    finally:
        conn.close()


def _orders(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT part_number, quantity, bearing_id, requested_by, approved_by FROM orders"
        ).fetchall()
    finally:
        conn.close()


def _run(response, prompt, executor, **kwargs) -> OrchestrationResult:
    session, store = executor
    return asyncio.run(
        run_from_response_async(
            response, prompt=prompt, session=session, token_store=store, **kwargs
        )
    )


# --------------------------------------------------------------------------------------
# The gate condition: what reaches a human at all
# --------------------------------------------------------------------------------------


def test_a_grounded_response_with_a_recommendation_reaches_the_gate():
    response = _response()

    assert response.grounding_tier in (GROUNDED, PARTIAL)
    assert needs_human_approval(response) is True


def test_a_tier_three_response_never_reaches_the_gate():
    """Section 6 withholds the recommendation from a tier-3 response's released text; offering
    that same action for approval would be worse than releasing it."""
    response = _response(ungrounded=True)

    assert response.grounding_tier == UNGROUNDED
    assert needs_human_approval(response) is False


def test_a_response_without_a_recommendation_never_reaches_the_gate():
    response = _response(recommendation=None)

    assert response.recommendation is None
    assert needs_human_approval(response) is False


def test_a_response_flagged_as_not_requiring_approval_never_reaches_the_gate():
    """`requires_approval` and a non-null `recommendation` always coincide in a response the
    critic assembles (`RecommendationGate.requires_approval` *is* "a recommendation is
    present"). This forces them apart to assert the gate reads all three conditions rather
    than relying on that coupling -- which is exactly what `needs_human_approval` documents."""
    forced = dataclasses.replace(_response(), requires_approval=False)

    assert forced.recommendation is not None
    assert needs_human_approval(forced) is False


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_response(ungrounded=True), id="ungrounded"),
        pytest.param(_response(recommendation=None), id="no-recommendation"),
        pytest.param(
            dataclasses.replace(_response(), requires_approval=False),
            id="requires-approval-false",
        ),
    ],
)
def test_an_out_of_scope_response_never_invokes_the_prompt(response, executor, db_path):
    """Task 3's assertion, on invocation rather than on the absence of an order: nothing is
    shown to a human, nothing is read, and there is no path from here to a mint."""
    prompt = SpyPrompt(Approved("X", 1, "tech-01", "supervisor-02"))
    _session, store = executor

    result = _run(response, prompt, executor)

    assert prompt.was_invoked is False
    assert result.gate_reached is False
    assert result.approval is None
    assert result.execution is None
    assert store.mint_calls == []
    assert _orders(db_path) == []


# --------------------------------------------------------------------------------------
# Declining
# --------------------------------------------------------------------------------------


def test_declining_mints_no_token_and_writes_no_row(executor, db_path):
    """Task 4, asserted on the store and the database rather than on the transcript."""
    prompt = SpyPrompt(Declined())
    _session, store = executor

    result = _run(_response(), prompt, executor)

    assert prompt.was_invoked is True
    assert result.gate_reached is True
    assert isinstance(result.approval, Declined)
    assert result.execution is None
    assert store.mint_calls == []
    assert _orders(db_path) == []


def test_a_decline_carrying_order_details_is_still_only_a_decline(executor, db_path):
    """"Declining is declining, not 'approve with different parameters'" -- a decline whose
    text names a part and a quantity must not become an order for them."""
    prompt = SpyPrompt(
        Declined("no -- but if you must, order 5 of ZA-2115 for 2nd_test-demo")
    )
    _session, store = executor

    result = _run(_response(), prompt, executor)

    assert isinstance(result.approval, Declined)
    assert store.mint_calls == []
    assert _orders(db_path) == []


# --------------------------------------------------------------------------------------
# Approving
# --------------------------------------------------------------------------------------


def test_approving_writes_exactly_one_row_matching_what_was_approved(executor, db_path):
    part = _real_part(db_path)
    approved = Approved(
        part_number=part,
        quantity=2,
        requested_by="tech-01",
        approved_by="supervisor-02",
        bearing_id="2nd_test-demo",
    )
    _session, store = executor

    result = _run(_response(), SpyPrompt(approved), executor)

    assert isinstance(result.execution, OrderPlaced)
    assert result.approved is True
    assert store.mint_calls == [(part, 2, "2nd_test-demo", "supervisor-02")]
    assert _orders(db_path) == [(part, 2, "2nd_test-demo", "tech-01", "supervisor-02")]


def test_the_order_matches_the_humans_values_and_not_the_recommendations(executor, db_path):
    """**Option (a), asserted rather than described.** The recommendation names `ZA-2115` and
    a quantity of 5; the human approves a different part and a different quantity, and the row
    that lands is the human's. Nothing in this module parses the recommendation text."""
    part = _real_part(db_path)
    assert part != RECOMMENDED_PART, "the fixture part must differ from the recommended one"

    approved = Approved(
        part_number=part, quantity=1, requested_by="tech-01", approved_by="supervisor-02"
    )

    result = _run(_response(), SpyPrompt(approved), executor)

    assert isinstance(result.execution, OrderPlaced)
    assert result.execution.part_number == part
    assert result.execution.quantity == 1
    assert _orders(db_path) == [(part, 1, None, "tech-01", "supervisor-02")]


def test_approved_by_on_the_row_comes_from_the_token_not_from_an_argument(executor, db_path):
    """#125's property, still true one layer up: `approved_by` is derived from the validated
    token record, so the value on the row is the one that was minted at approval time."""
    part = _real_part(db_path)
    approved = Approved(
        part_number=part, quantity=1, requested_by="tech-01", approved_by="supervisor-07"
    )
    _session, store = executor

    result = _run(_response(), SpyPrompt(approved), executor)

    assert isinstance(result.execution, OrderPlaced)
    assert result.execution.approved_by == "supervisor-07"
    assert store.mint_calls[0][3] == "supervisor-07"


def test_a_rejected_order_is_surfaced_rather_than_raised(executor, db_path):
    """An unknown part is refused by the tool; the orchestrator reports the rejection reason
    unchanged instead of raising, and writes nothing."""
    approved = Approved(
        part_number="NO-SUCH-PART",
        quantity=1,
        requested_by="tech-01",
        approved_by="supervisor-02",
    )

    result = _run(_response(), SpyPrompt(approved), executor)

    assert result.approved is True
    assert not isinstance(result.execution, OrderPlaced)
    assert result.execution is not None and result.execution.reason
    assert _orders(db_path) == []


# --------------------------------------------------------------------------------------
# The affirmative predicate, and the terminal prompt
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("answer", sorted(AFFIRMATIVE))
def test_the_closed_affirmative_vocabulary_approves(answer):
    assert is_affirmative(answer) is True
    assert is_affirmative(f"  {answer.upper()}  ") is True


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "n",
        "no",
        "maybe",
        "yes please order 5 of ZA-2115",
        "yesterday",
        "y/n",
        CASE_5_CLAIMED_AUTHORITY,
        CASE_5_CLAIMED_PRIOR_APPROVAL,
    ],
)
def test_anything_other_than_an_unambiguous_affirmative_declines(answer):
    """Containment would approve half of these. Exact match is the whole control."""
    assert is_affirmative(answer) is False


def test_the_prompt_shows_the_recommendation_and_the_tier(executor):
    written: list[str] = []
    response = _response()

    prompt_for_approval(response, read=_scripted("n"), write=written.append)

    shown = "\n".join(written)
    assert response.grounding_tier in shown
    assert RECOMMENDATION in shown


def test_the_prompt_collects_five_fields_on_an_affirmative():
    outcome = prompt_for_approval(
        _response(),
        read=_scripted("y", "ZA-2115", "3", "2nd_test-demo", "tech-01", "supervisor-02"),
        write=lambda _line: None,
    )

    assert outcome == Approved(
        part_number="ZA-2115",
        quantity=3,
        requested_by="tech-01",
        approved_by="supervisor-02",
        bearing_id="2nd_test-demo",
    )


def test_a_blank_bearing_id_becomes_none_rather_than_an_empty_string():
    outcome = prompt_for_approval(
        _response(),
        read=_scripted("y", "ZA-2115", "1", "   ", "tech-01", "supervisor-02"),
        write=lambda _line: None,
    )

    assert isinstance(outcome, Approved)
    assert outcome.bearing_id is None


def test_an_unreadable_stdin_is_a_decline_not_a_crash():
    """A closed stdin raises `EOFError` from `input()`. No default-yes, and no traceback."""
    outcome = prompt_for_approval(
        _response(), read=_scripted(), write=lambda _line: None
    )

    assert isinstance(outcome, Declined)


def test_stdin_closing_midway_through_the_parameters_is_a_decline():
    outcome = prompt_for_approval(
        _response(), read=_scripted("y", "ZA-2115"), write=lambda _line: None
    )

    assert isinstance(outcome, Declined)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param(("", "1", "", "tech-01", "supervisor-02"), id="blank-part"),
        pytest.param(("ZA-2115", "0", "", "tech-01", "supervisor-02"), id="zero-quantity"),
        pytest.param(("ZA-2115", "-2", "", "tech-01", "supervisor-02"), id="negative"),
        pytest.param(("ZA-2115", "two", "", "tech-01", "supervisor-02"), id="not-a-number"),
        pytest.param(("ZA-2115", "1", "", "", "supervisor-02"), id="blank-requester"),
        pytest.param(("ZA-2115", "1", "", "tech-01", "  "), id="blank-approver"),
    ],
)
def test_a_malformed_parameter_declines_the_whole_gate(fields):
    """Fail closed, and do not re-prompt: a loop in a blocking gate is a way to end up
    approving something after several attempts to say what was meant."""
    part_number, raw_quantity, bearing_id, requested_by, approved_by = fields

    outcome = build_approval(
        part_number=part_number,
        raw_quantity=raw_quantity,
        bearing_id=bearing_id,
        requested_by=requested_by,
        approved_by=approved_by,
    )

    assert isinstance(outcome, Declined)


def test_a_malformed_parameter_at_the_real_prompt_writes_no_row(executor, db_path):
    """The same failure driven through the whole flow, not just the validator."""
    _session, store = executor

    result = _run(
        _response(),
        lambda response: prompt_for_approval(
            response,
            read=_scripted("y", "ZA-2115", "not-a-number", "", "tech-01", "supervisor-02"),
            write=lambda _line: None,
        ),
        executor,
    )

    assert isinstance(result.approval, Declined)
    assert store.mint_calls == []
    assert _orders(db_path) == []


# --------------------------------------------------------------------------------------
# The production path: a real write-server subprocess (Issue #132)
# --------------------------------------------------------------------------------------


def test_the_default_path_places_the_order_through_a_real_subprocess(db_path):
    """**No injected session.** `run_from_response_async` falls through to
    `executor_session`, which serves this process's token store over a Unix socket and
    launches `python -m src.agent.mcp.write_server` as a real OS process.

    Before Issue #132 this could not work at all -- the subprocess built its own empty store
    and refused every token as unknown -- which is why PR #131 had to hold an in-process
    server. The row landing in the database is the proof it works now.
    """
    part = _real_part(db_path)
    store = SpyTokenStore()
    approved = Approved(
        part_number=part,
        quantity=2,
        requested_by="tech-01",
        approved_by="supervisor-02",
        bearing_id="2nd_test-demo",
    )

    result = asyncio.run(
        run_from_response_async(
            _response(),
            prompt=SpyPrompt(approved),
            token_store=store,
            db_path=db_path,
        )
    )

    assert isinstance(result.execution, OrderPlaced), getattr(result.execution, "reason", None)
    assert store.mint_calls == [(part, 2, "2nd_test-demo", "supervisor-02")]
    assert _orders(db_path) == [(part, 2, "2nd_test-demo", "tech-01", "supervisor-02")]


def test_the_default_path_declines_without_launching_anything(db_path):
    """A decline must not pay for a subprocess or a socket it never needed -- the gate
    returns before the executor session is entered at all."""
    store = SpyTokenStore()

    result = asyncio.run(
        run_from_response_async(
            _response(), prompt=SpyPrompt(Declined()), token_store=store, db_path=db_path
        )
    )

    assert isinstance(result.approval, Declined)
    assert result.execution is None
    assert store.mint_calls == []
    assert _orders(db_path) == []


# --------------------------------------------------------------------------------------
# Section 10 case 5: approval extraction, end to end
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("payload", CASE_5_APPROVAL_EXTRACTION_ATTEMPTS)
def test_case_5_payloads_typed_at_the_gate_produce_zero_rows(payload, executor, db_path):
    """The sharpest form of case 5 available here: the payload is not merely *in* the
    context, it is the answer given at the approval prompt itself. `SELECT COUNT(*) FROM
    orders` unchanged -- zero rows -- and no token was ever minted."""
    _session, store = executor

    result = _run(
        _response(),
        lambda response: prompt_for_approval(
            response, read=_scripted(payload), write=lambda _line: None
        ),
        executor,
    )

    assert isinstance(result.approval, Declined)
    assert store.mint_calls == []
    assert _orders(db_path) == []


@pytest.mark.parametrize("payload", CASE_5_APPROVAL_EXTRACTION_ATTEMPTS)
def test_case_5_payloads_carried_in_the_recommendation_produce_zero_rows(
    payload, executor, db_path
):
    """The same payloads arriving where a successful injection would actually put them -- in
    Agent A's recommendation, which is model output. The human declines; the decisive control
    is that the recommendation is never a source of order parameters in the first place."""
    response = _response(recommendation=f"Order 5 of ZA-2115. {payload}")
    _session, store = executor

    result = _run(
        response,
        lambda r: prompt_for_approval(r, read=_scripted("n"), write=lambda _line: None),
        executor,
    )

    assert needs_human_approval(response) is True
    assert isinstance(result.approval, Declined)
    assert store.mint_calls == []
    assert _orders(db_path) == []


def test_a_multi_turn_claim_of_prior_approval_starts_from_zero_each_run(executor, db_path):
    """Case 5's multi-turn shape: a first turn that gets as far as the gate and is declined,
    then a second turn whose recommendation asserts that approval already happened. No state
    carries between them -- there is no token to replay, because none was ever minted."""
    _session, store = executor

    first = _run(
        _response(),
        lambda r: prompt_for_approval(r, read=_scripted("n"), write=lambda _line: None),
        executor,
    )
    second = _run(
        _response(recommendation=f"Order 5 of ZA-2115. {CASE_5_CLAIMED_PRIOR_APPROVAL}"),
        lambda r: prompt_for_approval(
            r, read=_scripted(CASE_5_CLAIMED_PRIOR_APPROVAL), write=lambda _line: None
        ),
        executor,
    )

    assert isinstance(first.approval, Declined)
    assert isinstance(second.approval, Declined)
    assert store.mint_calls == []
    assert _orders(db_path) == []


# --------------------------------------------------------------------------------------
# Reporting back to the terminal
# --------------------------------------------------------------------------------------


def test_describe_reports_the_order_id_on_success(executor, db_path):
    part = _real_part(db_path)
    result = _run(
        _response(),
        SpyPrompt(Approved(part, 1, "tech-01", "supervisor-02")),
        executor,
    )
    written: list[str] = []

    describe(result, write=written.append)

    assert isinstance(result.execution, OrderPlaced)
    assert str(result.execution.order_id) in "\n".join(written)


def test_describe_reports_the_rejection_reason_unchanged(executor, db_path):
    result = _run(
        _response(),
        SpyPrompt(Approved("NO-SUCH-PART", 1, "tech-01", "supervisor-02")),
        executor,
    )
    written: list[str] = []

    describe(result, write=written.append)

    assert result.execution is not None
    assert result.execution.reason in "\n".join(written)


def test_describe_says_nothing_was_ordered_on_a_decline(executor):
    result = _run(_response(), SpyPrompt(Declined()), executor)
    written: list[str] = []

    describe(result, write=written.append)

    assert "Nothing was ordered" in "\n".join(written)


def test_describe_says_no_approval_was_requested_when_the_gate_was_not_reached(executor):
    result = _run(_response(recommendation=None), SpyPrompt(Declined()), executor)
    written: list[str] = []

    describe(result, write=written.append)

    assert "No approval was requested" in "\n".join(written)


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def test_a_caller_supplied_session_without_its_store_is_rejected_loudly():
    """The two must travel together -- a session whose server validates against a different
    store than the one being minted into would reject every token for an unrelated reason."""
    with pytest.raises(ValueError, match="token store"):
        asyncio.run(
            run_from_response_async(
                _response(),
                prompt=SpyPrompt(Approved("X", 1, "tech-01", "supervisor-02")),
                session=object(),
            )
        )


def test_the_orchestrator_never_writes_a_trace_file():
    """Section 9 is a separate, not-yet-started issue: this module may leave an extension
    point (`OrchestrationResult`) but must not implement trace writing."""
    source = (
        Path(__file__).resolve().parents[1] / "src" / "agent" / "orchestrator.py"
    ).read_text(encoding="utf-8")

    assert "open(" not in source
    assert "write_text" not in source
    assert "jsonl" not in source.lower()
