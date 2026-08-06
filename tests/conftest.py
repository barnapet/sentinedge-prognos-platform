"""Command-line control over the model-call cassettes (Issue #122).

    pytest tests/test_agent_pipeline_live.py                     # replay: free, no key, no network
    pytest tests/test_agent_pipeline_live.py --record            # re-record: real, billed calls
    pytest tests/test_agent_pipeline_live.py --cassette-mode live  # real calls, writes nothing

`--record` is `--cassette-mode record` spelled shortly, because it is the one a person types
by hand. The `AGENT_CASSETTE_MODE` environment variable does the same job for a caller that is
not pytest, and the flag wins over it when both are given.

The default is `replay` and nothing here changes that: a plain `pytest tests/` -- which is what
`.github/workflows/notebook-ci.yml` runs -- makes no API call and needs no credentials.
"""
from __future__ import annotations

import os
from typing import Any, Callable, ContextManager

import httpx
import pytest

from anthropic import AsyncAnthropic

from tests.fixtures.cassette import (
    LIVE,
    MODE_ENV_VAR,
    MODES,
    RECORD,
    cassette,
    has_anthropic_credentials,
    resolve_mode,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cassette-mode",
        choices=list(MODES),
        default=None,
        help=(
            "how the live tests source their model calls: replay (default, free and offline), "
            "record (real billed calls, overwrites the committed cassette), or live (real "
            "billed calls, writes nothing)"
        ),
    )
    parser.addoption(
        "--record",
        action="store_true",
        default=False,
        help="shorthand for --cassette-mode record. Makes real, billed Anthropic API calls.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Publish the chosen mode through the same environment variable everything else reads.

    One resolution point rather than two: `tests/fixtures/cassette.py` is also used from
    outside pytest, so making the flag set the variable keeps a single answer to "what mode is
    this?" instead of a flag path and an environment path that can disagree.
    """
    mode = config.getoption("--cassette-mode")
    if config.getoption("--record"):
        if mode is not None and mode != RECORD:
            raise pytest.UsageError(
                f"--record conflicts with --cassette-mode {mode}; pass only one"
            )
        mode = RECORD
    if mode is not None:
        os.environ[MODE_ENV_VAR] = mode


def _documentation_index_state() -> str:
    """Whether Qdrant is serving the documentation collection, right now.

    A real probe rather than a hand-written claim, because it is written into a recording as
    fact. It is a plain REST call against Qdrant rather than a `search_documentation` round
    trip on purpose: the answer wanted here is "was the index up", and going through retrieval
    would load the embedding model to find that out.
    """
    from src.agent.rag.index import COLLECTION_NAME, DEFAULT_QDRANT_URL

    url = f"{DEFAULT_QDRANT_URL.rstrip('/')}/collections/{COLLECTION_NAME}"
    try:
        response = httpx.get(url, timeout=2.0)
    except httpx.HTTPError as exc:
        return f"unreachable ({type(exc).__name__})"
    if response.status_code == 200:
        return "reachable"
    return f"unreachable (HTTP {response.status_code})"


@pytest.fixture()
def model_cassette() -> Callable[..., ContextManager[AsyncAnthropic]]:
    """Open a cassette for the current test, with the environment note filled in for free.

    Two things this adds over calling `cassette()` directly, both of which every live test
    wants and none of which belongs in `tests/fixtures/cassette.py`:

    - `notes` carries a *probed* documentation-index state, so a recording says which
      infrastructure was actually up rather than which was meant to be.
    - `live` mode with no credentials skips rather than fails. `record` mode still fails --
      asking for a recording and silently not getting one is how a stale cassette survives a
      run that was supposed to refresh it.
    """

    def _open(name: str, **kwargs: Any) -> ContextManager[AsyncAnthropic]:
        mode = resolve_mode()
        if mode == LIVE and not has_anthropic_credentials():
            pytest.skip(
                "--cassette-mode live makes real API calls and no Anthropic credentials were "
                "found. The default mode is replay, which needs none."
            )
        notes = kwargs.pop("notes", {})
        if mode == RECORD:
            notes = {"documentation_index": _documentation_index_state(), **notes}
        return cassette(name, notes=notes, **kwargs)

    return _open
