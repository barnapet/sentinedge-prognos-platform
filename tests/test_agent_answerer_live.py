"""The one recorded call through Agent A (Issue #112, `docs/agent_design.md` Sections 1 and 6).

**This is not Section 8's golden set**, which is a separate, later issue. It exists to prove
the wiring produces a schema-valid, non-empty draft at all, once, against real infrastructure:
a real serving process with real per-bearing state, a real read-only MCP server subprocess,
the real inventory database, and a real model call. Everything else about Agent A is asserted
without a model in `tests/test_agent_answerer.py`, where it cannot flake.

**The model call is replayed from a committed cassette by default (Issue #122)**, so this test
now runs on every PR, in ordinary CI, with no API key, no network access and no cost -- where
before it skipped for lack of a key and therefore protected nothing on the branches that
mattered. Everything else here is still real: the uvicorn process, the 60 replayed windows,
the MCP server subprocess, the inventory database, and the tool calls, which `tool_runner`
really makes in response to the recorded `tool_use` blocks.

**What a replay stops proving**, stated plainly because a green test that has quietly narrowed
is worse than a skipped one: that the model would make these tool choices and write this draft
*today*. It still proves the harness drives the loop, the enveloping chokepoint holds, the
tools answer, and the response parses into Section 6's shape.

**When to re-record.** Deliberately, not routinely -- after a change to the system prompt, the
draft schema, the tool definitions, or the model/request configuration:

    pytest tests/test_agent_answerer_live.py --record      # real, billed API calls

You should rarely have to remember: `tests/fixtures/cassette.py` hashes the prompts and request
configuration into each cassette and a replay fails with the list of what changed, which is
`docs/agent_design.md` Section 8's "no prompt change merges without a recorded run" rule made
mechanical. To hit the real API without touching the committed fixture, use
`--cassette-mode live`.

**The cassettes are deliberately recorded with Qdrant down**, matching CI, where the
documentation index is behind an opt-in compose profile that does not run. Recording with it up
would bake chunk ids into the draft that CI's evidence set cannot contain, and the test would
fail on the honest machine rather than the misconfigured one. Each cassette's `notes` records
the probed state, so this is a fact in the file rather than a claim in a docstring.

The bearing is populated the way the rest of the repo populates one -- `demo/playback.py`
against a real uvicorn process, exactly as `tests/test_demo_playback.py` does -- because
`get_bearing_status` reads state that only arrives by replaying windows through `/predict`.
"""
from __future__ import annotations

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

# One cassette per test rather than one per file: each test is an independent conversation with
# the API, and a shared recording would make the second test's result depend on the first
# having run, in order, in the same process.
DRAFT_CASSETTE = "answerer_live__schema_valid_draft"
ENVELOPE_CASSETTE = "answerer_live__request_envelope"


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


def test_one_real_question_produces_a_schema_valid_non_empty_draft(
    populated_serving_api, tmp_path, capsys, model_cassette
):
    db_path = tmp_path / "inventory.db"
    build_db(db_path)

    with model_cassette(DRAFT_CASSETTE) as client:
        draft = answer(
            QUESTION, client=client, serving_url=populated_serving_api, db_path=db_path
        )

    # Schema-valid by construction (`answer` parses through the model), and non-empty in the
    # sense that matters: it actually said something, and said where it came from.
    assert isinstance(draft, Draft)
    assert draft.claims, "the answerer produced a draft with no claims at all"
    assert all(claim.text.strip() for claim in draft.claims)
    assert any(claim.source_ids for claim in draft.claims), (
        "no claim carried a citation; Section 6's whole contract is that claims are sourced"
    )

    with capsys.disabled():
        print("\n--- draft ---")
        print(draft.model_dump_json(indent=2))


def test_the_live_call_wraps_its_question_in_this_requests_envelope(
    populated_serving_api, tmp_path, model_cassette
):
    """The envelope is not a test-only wrapper: the same call path that answers a real
    question is the one that builds the envelope, so this run and a tier-1 run exercise the
    same chokepoint.

    Replay does not weaken this one at all: the envelope is built by the harness on the way
    *out*, before anything reaches the transport, so what is asserted here never depended on
    the response being fresh."""
    db_path = tmp_path / "inventory.db"
    build_db(db_path)
    envelope = UntrustedEnvelope()

    with model_cassette(ENVELOPE_CASSETTE) as client:
        draft = answer(
            QUESTION,
            client=client,
            serving_url=populated_serving_api,
            db_path=db_path,
            envelope=envelope,
        )

    assert isinstance(draft, Draft)
    assert len(envelope.nonce) == 32
