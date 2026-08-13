"""One retry on `ServingUnreachable` at the three live-serving call sites (Issue #154,
`docs/agent_design.md` §8 tier 3: "Tool times out -> Exactly one retry, then degrade; not
an infinite loop.").

Distinct from `tests/test_agent_mcp_tools.py`, which already covers a *single*-attempt
unreachable failure against a real closed port -- that test is unchanged by this issue and
still passes, because a real closed port fails the same way on both attempts. What's new
here is entirely about **call count**, which a real socket cannot assert on directly: the
`serving_client` functions each site calls (`get_drift`, `post_predict`, `get_history`) are
monkeypatched in place, at the exact name `src/agent/mcp/tools.py` imports and calls, so a
stub can count invocations and hand back a controlled sequence of failure/success.

Three sites, two behaviours each, per the issue's own "Output" section:

- one transient `ServingUnreachable` then success -> the result, not a degrade, and exactly
  two calls were made;
- two consecutive `ServingUnreachable` -> the same degrade `failed(...)` envelope a single
  unreachable attempt already produced, and **exactly two** calls were made, never three.

Plus, per site, that a `ServingRejected` on the first attempt is never retried at all --
the constraint `_call_retrying_once_on_unreachable`'s own docstring states.
"""
from __future__ import annotations

from src.agent.mcp import tools
from src.agent.mcp.results import SERVICE_UNREACHABLE, payload_of
from src.agent.mcp.serving_client import ServingRejected, ServingUnreachable

DRIFT_BODY = {"bearings": {}}
PREDICT_BODY = {
    "label": "Normal",
    "baseline_status": "stable",
    "drift_status": "stable",
    "model_notes": "…",
}
# `found: False` is a real, fully-formed `get_history` success body (the "untracked
# bearing" shape) -- chosen so this file needs no real HTTP and no archived-trajectory
# fixture data to prove the retry succeeded, only that `find_similar_historical_pattern`
# turned a stub's return value into a non-degraded result.
UNTRACKED_HISTORY_BODY = {"found": False, "tracked_bearings": []}


def _fails_then(n_failures: int, result):
    """A stub matching `get_drift`/`post_predict`/`get_history`'s calling shape: raises
    `ServingUnreachable` on each of its first `n_failures` calls, then on every call after
    either returns `result` or, if `result` is an exception instance, raises it instead --
    which is how the "does not retry a rejection" tests hand back `ServingRejected` as the
    very first attempt's outcome.

    The call count lives on the stub itself (`stub.calls`) rather than in a `nonlocal` a
    test would have no way to read back.
    """
    calls = {"n": 0}

    def _call(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= n_failures:
            raise ServingUnreachable("simulated: nothing answered")
        if isinstance(result, Exception):
            raise result
        return result

    _call.calls = calls
    return _call


# --------------------------------------------------------------------------------------
# get_bearing_status -> get_drift
# --------------------------------------------------------------------------------------


def test_get_bearing_status_retries_once_then_succeeds(monkeypatch):
    stub = _fails_then(1, DRIFT_BODY)
    monkeypatch.setattr(tools, "get_drift", stub)

    result = tools.get_bearing_status()

    assert result.is_error is False
    assert stub.calls["n"] == 2


def test_get_bearing_status_degrades_after_two_failures_with_no_third_attempt(monkeypatch):
    stub = _fails_then(2, DRIFT_BODY)
    monkeypatch.setattr(tools, "get_drift", stub)

    result = tools.get_bearing_status()

    assert result.is_error is True
    assert payload_of(result)["error"] == SERVICE_UNREACHABLE
    assert stub.calls["n"] == 2  # one retry, not a loop -- a third call would be a bug


def test_get_bearing_status_does_not_retry_a_rejection(monkeypatch):
    stub = _fails_then(0, ServingRejected("rejected", 500))
    monkeypatch.setattr(tools, "get_drift", stub)

    result = tools.get_bearing_status()

    assert result.is_error is True
    assert "500" in payload_of(result)["error"]
    assert stub.calls["n"] == 1


# --------------------------------------------------------------------------------------
# predict_health_state -> post_predict
# --------------------------------------------------------------------------------------


def test_predict_health_state_retries_once_then_succeeds(monkeypatch):
    stub = _fails_then(1, PREDICT_BODY)
    monkeypatch.setattr(tools, "post_predict", stub)

    result = tools.predict_health_state("2nd_test-demo", [1.0, 2.0])

    assert result.is_error is False
    assert payload_of(result)["data"]["label"] == "Normal"
    assert stub.calls["n"] == 2


def test_predict_health_state_degrades_after_two_failures_with_no_third_attempt(monkeypatch):
    stub = _fails_then(2, PREDICT_BODY)
    monkeypatch.setattr(tools, "post_predict", stub)

    result = tools.predict_health_state("2nd_test-demo", [1.0, 2.0])

    assert result.is_error is True
    assert payload_of(result)["error"] == SERVICE_UNREACHABLE
    assert stub.calls["n"] == 2


def test_predict_health_state_does_not_retry_a_rejection(monkeypatch):
    stub = _fails_then(0, ServingRejected("rejected", 422))
    monkeypatch.setattr(tools, "post_predict", stub)

    result = tools.predict_health_state("2nd_test-demo", [1.0, 2.0])

    assert result.is_error is True
    assert "422" in payload_of(result)["error"]
    assert stub.calls["n"] == 1


# --------------------------------------------------------------------------------------
# find_similar_historical_pattern -> get_history
# --------------------------------------------------------------------------------------


def test_find_similar_historical_pattern_retries_once_then_succeeds(monkeypatch):
    stub = _fails_then(1, UNTRACKED_HISTORY_BODY)
    monkeypatch.setattr(tools, "get_history", stub)

    result = tools.find_similar_historical_pattern("untracked-bearing")

    assert result.is_error is False
    assert payload_of(result)["data"]["found"] is False
    assert stub.calls["n"] == 2


def test_find_similar_historical_pattern_degrades_after_two_failures_no_third_attempt(
    monkeypatch,
):
    stub = _fails_then(2, UNTRACKED_HISTORY_BODY)
    monkeypatch.setattr(tools, "get_history", stub)

    result = tools.find_similar_historical_pattern("untracked-bearing")

    assert result.is_error is True
    assert payload_of(result)["error"] == SERVICE_UNREACHABLE
    assert stub.calls["n"] == 2


def test_find_similar_historical_pattern_does_not_retry_a_rejection(monkeypatch):
    stub = _fails_then(0, ServingRejected("rejected", 500))
    monkeypatch.setattr(tools, "get_history", stub)

    result = tools.find_similar_historical_pattern("untracked-bearing")

    assert result.is_error is True
    assert "500" in payload_of(result)["error"]
    assert stub.calls["n"] == 1


# --------------------------------------------------------------------------------------
# The shared helper itself, in isolation
# --------------------------------------------------------------------------------------


def test_the_retry_helper_calls_once_on_immediate_success():
    stub = _fails_then(0, "ok")
    assert tools._call_retrying_once_on_unreachable(stub) == "ok"
    assert stub.calls["n"] == 1


def test_the_retry_helper_calls_twice_on_one_failure_then_success():
    stub = _fails_then(1, "ok")
    assert tools._call_retrying_once_on_unreachable(stub) == "ok"
    assert stub.calls["n"] == 2


def test_the_retry_helper_propagates_the_second_failure_without_a_third_call():
    stub = _fails_then(2, "ok")
    try:
        tools._call_retrying_once_on_unreachable(stub)
        assert False, "expected ServingUnreachable to propagate"
    except ServingUnreachable:
        pass
    assert stub.calls["n"] == 2


def test_the_retry_helper_does_not_catch_serving_rejected():
    stub = _fails_then(0, ServingRejected("rejected", 500))
    try:
        tools._call_retrying_once_on_unreachable(stub)
        assert False, "expected ServingRejected to propagate"
    except ServingRejected:
        pass
    assert stub.calls["n"] == 1
