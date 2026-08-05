"""The one live call through Agent A (Issue #112, `docs/agent_design.md` Sections 1 and 6).

**This is not Section 8's golden set**, which is a separate, later issue. It exists to prove
the wiring produces a schema-valid, non-empty draft at all, once, against real infrastructure:
a real serving process with real per-bearing state, a real read-only MCP server subprocess,
the real inventory database, and a real model call. Everything else about Agent A is asserted
without a model in `tests/test_agent_answerer.py`, where it cannot flake.

**Skips cleanly without an API key**, following the same pattern as this repo's existing
`data/processed`-dependent skips (`tests/test_build_training_dataset.py`): the reason string
says what is missing and how to supply it, so a skip in a CI log is self-explaining rather
than something to go and look up. A skip is not a pass, and a run that skipped this should say
so.

The bearing is populated the way the rest of the repo populates one -- `demo/playback.py`
against a real uvicorn process, exactly as `tests/test_demo_playback.py` does -- because
`get_bearing_status` reads state that only arrives by replaying windows through `/predict`.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from demo import playback
from src.agent.answerer import Draft, answer
from src.agent.inventory.build_db import build_db
from src.agent.untrusted import UntrustedEnvelope

REPO_ROOT = Path(__file__).resolve().parents[1]

# Enough windows to cross the 50-file baseline lock, so `baseline_status` is "stable" and the
# bearing has a real history to report rather than a cold start.
REPLAY_WINDOWS = 60
BEARING_ID = "2nd_test-demo"
QUESTION = f"What is the current status of bearing {BEARING_ID}?"

def _has_anthropic_credentials() -> bool:
    """Whether the SDK has *something* to authenticate with.

    An unset `ANTHROPIC_API_KEY` does not by itself mean there are no credentials: the SDK
    also reads `ANTHROPIC_AUTH_TOKEN` and a stored `ant auth login` profile. Checking only
    the one env var would skip this test on a machine where it would have run, which is the
    opposite of what a gate is for.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    config_dir = Path(
        os.environ.get("ANTHROPIC_CONFIG_DIR", Path.home() / ".config" / "anthropic")
    )
    return (config_dir / "credentials").is_dir() and any(
        (config_dir / "credentials").glob("*.json")
    )


requires_api_key = pytest.mark.skipif(
    not _has_anthropic_credentials(),
    reason=(
        "requires Anthropic credentials (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an "
        "`ant auth login` profile): this is the single live model call for Issue #112, and "
        "it is deliberately the only test in the suite that needs one. Every other assertion "
        "about the answerer runs without credentials in tests/test_agent_answerer.py"
    ),
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def populated_serving_api(tmp_path):
    """A real serving process with one bearing replayed past its cold-start boundary.

    A uvicorn subprocess rather than an in-process app, as `tests/test_demo_playback.py`
    does: the MCP tools speak HTTP over a socket, so an in-process transport would not
    exercise what they actually do.
    """
    port = _free_port()
    script = (
        "import uvicorn\n"
        "from pathlib import Path\n"
        "from src.serving.api import create_app\n"
        f"app = create_app(lock_path=Path({str(tmp_path / 'answerer.lock')!r}))\n"
        f"uvicorn.run(app, host='127.0.0.1', port={port}, workers=1, log_level='warning')\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=REPO_ROOT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        playback.wait_for_server(base_url, timeout_s=60)
        playback.main(
            ["--url", base_url, "--interval", "0", "--limit", str(REPLAY_WINDOWS),
             "--bearing-id", BEARING_ID]
        )
        yield base_url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@requires_api_key
def test_one_real_question_produces_a_schema_valid_non_empty_draft(
    populated_serving_api, tmp_path, capsys
):
    db_path = tmp_path / "inventory.db"
    build_db(db_path)

    draft = answer(QUESTION, serving_url=populated_serving_api, db_path=db_path)

    # Schema-valid by construction (`answer` parses through the model), and non-empty in the
    # sense that matters: it actually said something, and said where it came from.
    assert isinstance(draft, Draft)
    assert draft.claims, "the answerer produced a draft with no claims at all"
    assert all(claim.text.strip() for claim in draft.claims)
    assert any(claim.source_ids for claim in draft.claims), (
        "no claim carried a citation; Section 6's whole contract is that claims are sourced"
    )

    with capsys.disabled():
        print("\n--- live draft ---")
        print(draft.model_dump_json(indent=2))


@requires_api_key
def test_the_live_call_wraps_its_question_in_this_requests_envelope(populated_serving_api,
                                                                   tmp_path):
    """The envelope is not a test-only wrapper: the same call path that answers a real
    question is the one that builds the envelope, so a live run and a tier-1 run exercise the
    same chokepoint."""
    db_path = tmp_path / "inventory.db"
    build_db(db_path)
    envelope = UntrustedEnvelope()

    draft = answer(
        QUESTION, serving_url=populated_serving_api, db_path=db_path, envelope=envelope
    )

    assert isinstance(draft, Draft)
    assert len(envelope.nonce) == 32
