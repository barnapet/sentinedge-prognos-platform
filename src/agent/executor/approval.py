"""The approval-token mechanism (Issue #124, `docs/agent_design.md` Section 5).

> `place_order` requires an `approval_token` argument. The token is: minted out-of-band by
> the harness, at the moment a human approves, from a cryptographic random source; scoped to
> one order -- bound to `(part_number, quantity, bearing_id)`, so an approval for one part
> cannot authorize another; single-use, consumed on first successful order; short-lived,
> expiring after a few minutes; never present in any model's context until the human has
> approved, and never reconstructible from anything the model reads.

This module builds the token mechanism only: minting a token when a human approves, and
consuming it when an order is placed. It does not build the tool that will require the
token (`place_order`'s schema, `src/agent/mcp/write_server.py` -- a separate issue) or the
orchestrator that presents a recommendation to a human and calls both (a later issue). It
also does not build the "never present in any model's context" property -- that is a
property of *who calls `mint`*, not of this module, since nothing here ever runs inside a
model's tool loop.

**In-process store, no durable state** -- the same posture `src/agent/mcp/budget.py` takes
for the tool-call cap, for the same reason (`docs/agent_design.md` Section 0: per-run state
does not outlive the process). `ApprovalTokenStore` is instantiable fresh per test or per
process, not a module-level global, and there is nothing here that writes to disk or a
database: a token that does not survive a restart is the correct behaviour for a credential
that is supposed to expire in a few minutes anyway.

**Failures are a typed result, not a raised exception** -- the same convention
`src/agent/mcp/results.py` uses for tool failures, and for the same reason: Section 5's gate
must fail on a value the caller cannot skip past by forgetting a `try`/`except`. `consume()`
returns either an `ApprovedOrder` or a `TokenError` naming exactly one of four closed
reasons (`UNKNOWN`, `EXPIRED`, `ALREADY_USED`, `SCOPE_MISMATCH`) -- plain string constants,
the same style `results.py`'s `SOURCE_TYPES` and `deterministic.py`'s check names already
use in this codebase, rather than an `Enum`.

**Why scope is checked before consumed-state.** `consume()` validates, in order: the token
exists, it has not expired, the caller's `(part_number, quantity, bearing_id)` matches the
scope it was minted for exactly, and only then whether it has already been used. A token
that is presented against the wrong order is a scope error regardless of whether it happens
to have been spent already -- an already-used token being replayed against a *different*
order should still be reported as a mismatch, not as "already used", because "already used"
implies the presented order was in fact the one it authorized.

**Quantity validity is deliberately not this module's job.** `mint`/`consume` accept and
compare whatever `quantity` they are given -- this module authorizes a scope, it does not
judge whether that scope is a sensible order -- and a non-positive quantity is already
rejected independently at both places an order actually happens: the `place_order` MCP tool
(`tests/test_agent_mcp_tools.py::test_a_rejected_order_is_a_tool_result_and_writes_nothing`,
`zero-quantity`/`negative-quantity`) and `src/agent/inventory/orders.py`'s `place_order`
(`tests/test_agent_inventory.py::test_place_order_rejects_non_positive_quantity`).
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

# Section 5: "short-lived, expiring after a few minutes".
TOKEN_LIFETIME = timedelta(minutes=5)

# consume() failure reasons -- a closed vocabulary of plain strings, not an Enum. See the
# module docstring for why this mirrors results.py's SOURCE_TYPES.
UNKNOWN = "unknown"
EXPIRED = "expired"
ALREADY_USED = "already_used"
SCOPE_MISMATCH = "scope_mismatch"

TOKEN_ERROR_REASONS = frozenset({UNKNOWN, EXPIRED, ALREADY_USED, SCOPE_MISMATCH})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ApprovalToken:
    """One human approval, scoped to exactly one order.

    Mutable, unlike this package's other result dataclasses: `consumed` is flipped in place
    by `ApprovalTokenStore.consume` on success, and the store is the sole owner of that
    transition -- a caller cannot mark its own token used by holding a reference to it.
    """

    token: str
    part_number: str
    quantity: int
    bearing_id: str
    approved_by: str
    approved_at: datetime
    expires_at: datetime
    consumed: bool = False

    @property
    def scope(self) -> tuple[str, int, str]:
        return (self.part_number, self.quantity, self.bearing_id)


@dataclass(frozen=True)
class ApprovedOrder:
    """What `consume()` returns on success -- everything `place_order`'s later caller needs,
    with `approved_by`/`approved_at` carried from the moment of human approval (`mint`
    time), not from the moment the order was actually placed."""

    part_number: str
    quantity: int
    bearing_id: str
    approved_by: str
    approved_at: datetime


@dataclass(frozen=True)
class TokenError:
    """Why `consume()` refused a token. `reason` is always one of `TOKEN_ERROR_REASONS`."""

    reason: str

    def __post_init__(self) -> None:
        if self.reason not in TOKEN_ERROR_REASONS:
            raise ValueError(
                f"unknown token error reason {self.reason!r}; "
                f"expected one of {sorted(TOKEN_ERROR_REASONS)}"
            )


@dataclass
class ApprovalTokenStore:
    """An in-process, per-run store of minted approval tokens. No durable storage: this is
    a plain dict that lives and dies with the process holding it, same posture as
    `src/agent/mcp/budget.py`'s `ToolCallBudget`.

    `clock` defaults to the real wall clock and exists to be overridden in tests -- expiry
    is asserted by advancing an injected clock, never by sleeping.
    """

    clock: Callable[[], datetime] = _utcnow
    _tokens: dict[str, ApprovalToken] = field(default_factory=dict, init=False)

    def mint(
        self, part_number: str, quantity: int, bearing_id: str, approved_by: str
    ) -> ApprovalToken:
        """Mint one token, at the moment a human approves. Records the scope tuple, who
        approved it, and an expiry `TOKEN_LIFETIME` out from now."""
        now = self.clock()
        record = ApprovalToken(
            token=secrets.token_urlsafe(32),
            part_number=part_number,
            quantity=quantity,
            bearing_id=bearing_id,
            approved_by=approved_by,
            approved_at=now,
            expires_at=now + TOKEN_LIFETIME,
        )
        self._tokens[record.token] = record
        return record

    def consume(
        self, token: str, part_number: str, quantity: int, bearing_id: str
    ) -> ApprovedOrder | TokenError:
        """Validate and, on success, spend one token. See the module docstring for the
        checked order (existence, expiry, scope match, then consumed state) and why."""
        record = self._tokens.get(token)
        if record is None:
            return TokenError(UNKNOWN)
        if self.clock() >= record.expires_at:
            return TokenError(EXPIRED)
        if (part_number, quantity, bearing_id) != record.scope:
            return TokenError(SCOPE_MISMATCH)
        if record.consumed:
            return TokenError(ALREADY_USED)

        record.consumed = True
        return ApprovedOrder(
            part_number=record.part_number,
            quantity=record.quantity,
            bearing_id=record.bearing_id,
            approved_by=record.approved_by,
            approved_at=record.approved_at,
        )
