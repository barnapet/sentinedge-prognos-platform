"""Tier-1 tests for the approval-token mechanism (Issue #124, `docs/agent_design.md`
Section 5, Section 8's "approval token's scoping/single-use/expiry logic").

Pure `pytest`: no API key, no network, no filesystem. Expiry is asserted by advancing an
injected clock (`ApprovalTokenStore(clock=...)`), never by sleeping in the test.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agent.executor.approval import (
    ALREADY_USED,
    EXPIRED,
    SCOPE_MISMATCH,
    TOKEN_LIFETIME,
    UNKNOWN,
    ApprovalTokenStore,
    ApprovedOrder,
    TokenError,
)

ORDER = {"part_number": "BRG-6205-2RS", "quantity": 1, "bearing_id": "2nd_test-demo"}
APPROVED_BY = "supervisor-02"


class _ManualClock:
    """A settable clock for expiry tests -- advanced explicitly, never slept past."""

    def __init__(self, now: datetime) -> None:
        self._now = now

    def __call__(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


@pytest.fixture
def clock() -> _ManualClock:
    return _ManualClock(datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc))


@pytest.fixture
def store(clock: _ManualClock) -> ApprovalTokenStore:
    return ApprovalTokenStore(clock=clock)


def test_mint_then_consume_with_a_matching_tuple_succeeds_exactly_once(store):
    token = store.mint(**ORDER, approved_by=APPROVED_BY)

    result = store.consume(token.token, **ORDER)

    assert isinstance(result, ApprovedOrder)
    assert result.part_number == ORDER["part_number"]
    assert result.quantity == ORDER["quantity"]
    assert result.bearing_id == ORDER["bearing_id"]
    assert result.approved_by == APPROVED_BY
    assert result.approved_at == token.approved_at


def test_a_second_consume_of_the_same_token_returns_already_used(store):
    token = store.mint(**ORDER, approved_by=APPROVED_BY)
    first = store.consume(token.token, **ORDER)
    assert isinstance(first, ApprovedOrder)

    second = store.consume(token.token, **ORDER)

    assert second == TokenError(ALREADY_USED)


@pytest.mark.parametrize(
    "changed_field, changed_value",
    [
        ("part_number", "ZA-2115"),
        ("quantity", ORDER["quantity"] + 1),
        ("bearing_id", "1st_test-b3"),
    ],
)
def test_a_tuple_differing_in_one_field_returns_scope_mismatch(
    store, changed_field, changed_value
):
    token = store.mint(**ORDER, approved_by=APPROVED_BY)
    presented = {**ORDER, changed_field: changed_value}

    result = store.consume(token.token, **presented)

    assert result == TokenError(SCOPE_MISMATCH)
    # The token itself is untouched by a mismatched attempt -- it can still be consumed
    # against the tuple it was actually minted for.
    assert store.consume(token.token, **ORDER) == ApprovedOrder(
        approved_by=APPROVED_BY, approved_at=token.approved_at, **ORDER
    )


def test_a_token_consumed_after_its_expiry_window_returns_expired(store, clock):
    token = store.mint(**ORDER, approved_by=APPROVED_BY)

    clock.advance(TOKEN_LIFETIME + timedelta(seconds=1))
    result = store.consume(token.token, **ORDER)

    assert result == TokenError(EXPIRED)


def test_an_unknown_token_string_returns_unknown(store):
    result = store.consume("not-a-real-token", **ORDER)

    assert result == TokenError(UNKNOWN)


def test_two_tokens_minted_for_the_same_tuple_consume_independently(store):
    first_token = store.mint(**ORDER, approved_by="supervisor-01")
    second_token = store.mint(**ORDER, approved_by="supervisor-02")
    assert first_token.token != second_token.token

    first_result = store.consume(first_token.token, **ORDER)
    assert isinstance(first_result, ApprovedOrder)
    assert first_result.approved_by == "supervisor-01"

    # The second token is untouched by the first's consumption.
    second_result = store.consume(second_token.token, **ORDER)
    assert isinstance(second_result, ApprovedOrder)
    assert second_result.approved_by == "supervisor-02"

    # Both are now independently spent.
    assert store.consume(first_token.token, **ORDER) == TokenError(ALREADY_USED)
    assert store.consume(second_token.token, **ORDER) == TokenError(ALREADY_USED)


def test_an_unknown_reason_string_is_rejected_at_construction():
    with pytest.raises(ValueError):
        TokenError("not_a_real_reason")


def test_mint_records_the_expiry_a_few_minutes_out(store, clock):
    token = store.mint(**ORDER, approved_by=APPROVED_BY)

    assert token.expires_at == clock() + TOKEN_LIFETIME
    assert not token.consumed
