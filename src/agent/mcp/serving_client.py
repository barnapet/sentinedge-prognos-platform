"""HTTP client for the running serving API (Issue #110, `docs/agent_design.md` Section 2).

**This module exists so that nothing in `src/agent/mcp/` ever imports `src/serving/`.**
That is the sharpest constraint in Section 2, and the reason is mechanical rather than
stylistic:

- `src/serving/single_worker.py` takes an **exclusive OS file lock at app startup** and
  exits with `SingleWorkerViolation` if a second process tries to hold it
  (`docs/serving_design.md` Section 2, enforced and verified in Issue #84). An MCP tool
  that called `create_app()` in-process would therefore **fail outright** against a
  running server...
- ...or, worse, **succeed when no server is running** and then serve predictions from its
  own empty `BearingStateStore` -- a second, divergent copy of the rolling history that
  `docs/serving_design.md` Section 1 rejected Option B specifically to prevent. That case
  produces confident, plausible, wrong `rms_ratio` values with no error at all.

So the tools are HTTP clients, exactly as `demo/playback.py` is: they talk to an
already-running instance over the wire and own none of its state. `demo/playback.py` uses
`urllib` because it is meant to run against the container without a client stack installed;
this package is imported inside an environment that already has `httpx` pinned
(`requirements.txt`, added in #84), and Section 2 names `httpx` for exactly this, so it is
used here.

Failures split into exactly two cases, because they mean different things to the person
eventually reading the answer. **`ServingUnreachable`** is "nothing answered" -- no server
listening, DNS failure, timeout -- and is the case Issue #110 names literally: the tool
result says *the prediction service is not reachable*. **`ServingRejected`** is "something
answered, with an error status", which is a different situation entirely (the service is up;
this request was bad) and would be actively misleading if reported as unreachability. Both
derive from `ServingError` so a caller that genuinely does not care can catch one type.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = os.environ.get("PROGNOS_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT_S = 30.0

DRIFT_ENDPOINT = "GET /monitoring/drift"
PREDICT_ENDPOINT = "POST /predict"


class ServingError(Exception):
    """Base for every way a call to the serving API can fail."""


class ServingUnreachable(ServingError):
    """Nothing answered: connection refused, DNS failure, or timeout."""


class ServingRejected(ServingError):
    """The service answered with a non-2xx status."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _request(method: str, path: str, base_url: str, timeout: float, **kwargs: Any) -> Any:
    url = f"{base_url}{path}"
    try:
        response = httpx.request(method, url, timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        raise ServingUnreachable(f"{method} {url} failed: {exc}") from exc

    if response.is_error:
        raise ServingRejected(
            f"{method} {url} returned {response.status_code}", response.status_code
        )
    try:
        return response.json()
    except ValueError as exc:
        # A 2xx that is not JSON means something other than this API answered on that
        # port -- a proxy error page, a different service. "Unreachable" is the honest
        # reading: the serving API did not answer.
        raise ServingUnreachable(f"{method} {url} returned a non-JSON body: {exc}") from exc


def get_drift(
    base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT_S
) -> dict[str, Any]:
    """`GET /monitoring/drift` -- every bearing this serving process has seen, with its
    `file_count`, `baseline_status`, `drift_status`, per-feature z-scores, `rms_ratio_latest`
    and `predicted_class_counts` (`docs/monitoring_design.md` Section 3)."""
    return _request("GET", "/monitoring/drift", base_url, timeout)


def post_predict(
    bearing_id: str,
    signal: list[float],
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """`POST /predict` -- `docs/serving_design.md` Section 1's payload exactly: one
    bearing id and one raw single-window signal, nothing else. No file index, no
    pre-computed features: the server owns 100% of feature computation, and this client
    reimplements none of it."""
    return _request(
        "POST",
        "/predict",
        base_url,
        timeout,
        json={"bearing_id": bearing_id, "signal": signal},
    )
