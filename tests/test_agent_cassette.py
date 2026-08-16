"""The cassette mechanism itself (Issue #122), with no model and no network.

Tier 1 by `docs/agent_design.md` Section 8's rule: pure functions and a transport, no API key,
no network access. The live tests exercise the cassette against the real answerer; these pin
the parts whose failure mode is silent — a secret written into a committed fixture, a stale
recording that replays anyway, a diverged conversation that passes because the transport shrugged.

**The committed cassettes are checked here too**, and that check is the cheap one: a full live
replay needs a uvicorn process, a 60-window playback and an MCP subprocess, so if a prompt
change stales the fixtures it is worth finding out in milliseconds rather than a minute later.
"""
from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from anthropic import AsyncAnthropic

from tests.fixtures import cassette as cassette_module
from tests.fixtures.cassette import (
    CASSETTE_DIR,
    MODE_ENV_VAR,
    RECORD,
    REDACTED,
    REPLAY,
    CassetteMismatch,
    CassetteMissing,
    CassetteStale,
    cassette,
    check_fresh,
    current_fingerprint,
    resolve_mode,
    usage_totals,
)

A_MESSAGE = {
    "id": "msg_cassette_unit_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "replayed"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 11, "output_tokens": 3},
}


def _interaction(body: dict | None = None, *, path: str = "/v1/messages") -> dict:
    return {
        "request": {
            "method": "POST",
            "url": f"https://api.anthropic.com{path}",
            "url_path": path,
            "headers": [["x-api-key", REDACTED], ["content-type", "application/json"]],
            "body_encoding": "json",
            "body": {"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
        },
        "response": {
            "status_code": 200,
            "headers": [["content-type", "application/json"]],
            "body_encoding": "json",
            "body": body if body is not None else A_MESSAGE,
        },
    }


@pytest.fixture()
def written(tmp_path, monkeypatch):
    """Write a cassette into a temporary directory and point the module at it."""
    monkeypatch.setattr(cassette_module, "CASSETTE_DIR", tmp_path)

    def _write(name: str, interactions: list[dict], *, fingerprint=None) -> None:
        document = {
            "cassette_version": 1,
            "name": name,
            "response_source": "live_model_call",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": current_fingerprint() if fingerprint is None else fingerprint,
            "notes": {},
            "usage_totals": usage_totals(interactions),
            "interactions": interactions,
        }
        (tmp_path / f"{name}.json").write_text(json.dumps(document, indent=2), encoding="utf-8")

    return _write


def _create(client: AsyncAnthropic, messages=None):
    return asyncio.run(
        client.messages.create(
            model="claude-opus-5",
            max_tokens=16,
            messages=messages or [{"role": "user", "content": "hi"}],
        )
    )


# --- The default ------------------------------------------------------------------------


def test_the_default_mode_is_replay_so_a_plain_run_costs_nothing(monkeypatch):
    """The whole issue in one assertion: a run that says nothing about cassettes must not
    reach the API. CI says nothing about cassettes."""
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    assert resolve_mode() == REPLAY


def test_an_unknown_mode_is_rejected_rather_than_silently_treated_as_replay(monkeypatch):
    monkeypatch.setenv(MODE_ENV_VAR, "recrod")
    with pytest.raises(ValueError, match="unknown cassette mode"):
        resolve_mode()


def test_the_environment_variable_selects_record(monkeypatch):
    monkeypatch.setenv(MODE_ENV_VAR, "record")
    assert resolve_mode() == RECORD


# --- Replay really replays, and really does not call out ----------------------------------


def test_replay_returns_the_recorded_response_with_no_api_key_and_no_network(
    written, monkeypatch
):
    """End to end through the real SDK: a real `AsyncAnthropic`, a real `messages.create`, and
    a response that came out of a file. Credentials are removed from the environment first, so
    a passing run cannot be one that quietly authenticated."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    written("unit", [_interaction()])

    with cassette("unit") as client:
        message = _create(client)

    assert message.id == "msg_cassette_unit_test"
    assert message.content[0].text == "replayed"
    assert message.usage.input_tokens == 11


def test_replay_opens_no_socket_at_all(written, monkeypatch):
    """Stronger than "it did not need a key": the real transport is replaced with one that
    fails if it is ever reached, so any attempt to leave the process is an error rather than a
    slow test."""
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    written("unit", [_interaction()])

    def _forbidden(*args, **kwargs):
        raise AssertionError("replay mode opened a real HTTP transport")

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", _forbidden)

    with cassette("unit") as client:
        assert _create(client).content[0].text == "replayed"


def test_a_missing_cassette_says_how_to_record_one(tmp_path, monkeypatch):
    monkeypatch.setattr(cassette_module, "CASSETTE_DIR", tmp_path)
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    with pytest.raises(CassetteMissing, match="--record"):
        with cassette("never-recorded"):
            pass


# --- Staleness is loud ---------------------------------------------------------------------


def test_a_prompt_change_stales_the_cassette_and_names_what_changed(written, monkeypatch):
    """`docs/agent_design.md` Section 8's procedural rule, mechanised: a cassette recorded
    against a prompt that no longer exists is not evidence about the code in front of you."""
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    stale = current_fingerprint()
    stale["answerer"]["system_prompt_sha256"] = "0" * 64
    written("unit", [_interaction()], fingerprint=stale)

    with pytest.raises(CassetteStale) as excinfo:
        with cassette("unit"):
            pass

    assert "answerer.system_prompt_sha256" in str(excinfo.value)


def test_a_model_change_stales_the_cassette(written, monkeypatch):
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    stale = current_fingerprint()
    stale["critic"]["model"] = "some-older-model"
    written("unit", [_interaction()], fingerprint=stale)

    with pytest.raises(CassetteStale, match="critic.model"):
        with cassette("unit"):
            pass


def test_a_schema_change_stales_the_cassette(written, monkeypatch):
    """The draft schema is generated from `Draft`, so this is what catches a field added to
    the answerer's output shape without a fresh recording."""
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    stale = current_fingerprint()
    stale["answerer"]["draft_schema_sha256"] = "f" * 64
    written("unit", [_interaction()], fingerprint=stale)

    with pytest.raises(CassetteStale, match="answerer.draft_schema_sha256"):
        with cassette("unit"):
            pass


def test_a_tool_docstring_edit_stales_the_cassette(written, monkeypatch):
    """Issue #169: Section 8 names three things a cassette must go stale on -- "a prompt, a
    tool description, or the model" -- and until now `current_fingerprint()` only checked the
    first and third. This proves the gap is closed by exercising the real mechanism, not by
    hand-editing a fingerprint dict as the tests above do: a cassette recorded against today's
    real tool descriptions, then replayed after one read-only tool's actual `.description`
    changes, must go stale and must name the field that changed.

    The edit is applied to the already-built `Tool` registration rather than to
    `readonly_server.py`'s source (the issue's own constraint is not to touch the tool
    docstrings themselves) -- `MCPServer` has no public "edit an existing tool" call, only
    `remove_tool` and `add_tool`, so this reads the real tool's function and description
    through `_tool_manager` and re-registers it under an appended description, which is
    functionally identical to a technician editing the docstring and restarting the server.
    """
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    written("unit", [_interaction()])  # recorded against today's real tool descriptions

    from src.agent.mcp import readonly_server

    real_build_server = readonly_server.build_server

    def build_server_with_an_edited_tool_docstring(*args, **kwargs):
        server, budget = real_build_server(*args, **kwargs)
        original = server._tool_manager.get_tool("get_bearing_status")
        server.remove_tool("get_bearing_status")
        server.add_tool(
            original.fn,
            name=original.name,
            description=original.description + " Edited for this test only.",
        )
        return server, budget

    monkeypatch.setattr(readonly_server, "build_server", build_server_with_an_edited_tool_docstring)

    with pytest.raises(CassetteStale, match="answerer.tool_descriptions_sha256"):
        with cassette("unit"):
            pass


# --- Divergence is loud too -----------------------------------------------------------------


def test_one_call_too_many_fails_rather_than_returning_something_plausible(written, monkeypatch):
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    written("unit", [_interaction()])

    with pytest.raises(CassetteMismatch, match="no remaining recorded"):
        with cassette("unit") as client:
            _create(client)
            _create(client)


def test_a_recording_that_was_not_fully_replayed_fails(written, monkeypatch):
    """Fewer model calls than were recorded means the harness stopped driving the loop the way
    it did — which a transport that only answers questions it is asked would never notice."""
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    written("unit", [_interaction(), _interaction()])

    with pytest.raises(CassetteMismatch, match="never replayed"):
        with cassette("unit") as client:
            _create(client)


def test_a_turn_that_arrives_with_a_different_number_of_messages_fails(written, monkeypatch):
    """The conversation shape is checked even though the request *body* is not matched on —
    see `_ReplayTransport`'s docstring for why matching on the body would match nothing."""
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    written("unit", [_interaction()])

    with pytest.raises(CassetteMismatch, match="message"):
        with cassette("unit") as client:
            _create(
                client,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "again"},
                ],
            )


# --- Recording, against a real socket ----------------------------------------------------------


class _StubAPI(BaseHTTPRequestHandler):
    """The smallest thing that answers like the Messages API. Loopback only."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = json.dumps(A_MESSAGE).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("anthropic-organization-id", "org-not-a-real-one")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture()
def stub_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubAPI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_recording_writes_a_cassette_that_then_replays(tmp_path, monkeypatch, stub_api):
    """The one path a unit test can otherwise not reach without spending money: the real
    `httpx` transport, the write, and a replay of what was written.

    It runs against a loopback stub rather than the API — which is the point. What is being
    checked is the cassette machinery, not the model: that a recorded exchange survives being
    written to JSON and rebuilt into a response the SDK can parse, and that the API key the SDK
    really put on the wire is not in the file afterwards. A bug in any of that would otherwise
    surface for the first time on a billed `--record` run.
    """
    monkeypatch.setattr(cassette_module, "CASSETTE_DIR", tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-stub-key-for-the-recording-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", stub_api)
    monkeypatch.setenv(MODE_ENV_VAR, "record")

    with cassette("round-trip", notes={"documentation_index": "not probed"}) as client:
        recorded = _create(client)
    assert recorded.id == "msg_cassette_unit_test"

    document = json.loads((tmp_path / "round-trip.json").read_text(encoding="utf-8"))

    assert document["response_source"] == "live_model_call"
    assert document["fingerprint"] == current_fingerprint()
    assert document["notes"] == {"documentation_index": "not probed"}
    assert document["usage_totals"] == {"input_tokens": 11, "output_tokens": 3}

    sent_headers = dict(tuple(pair) for pair in document["interactions"][0]["request"]["headers"])
    assert sent_headers["x-api-key"] == REDACTED
    assert "sk-ant-" not in json.dumps(document)
    # The account identifier the stub returned is redacted on the way in, not filtered out of
    # the committed file afterwards.
    got_headers = dict(tuple(pair) for pair in document["interactions"][0]["response"]["headers"])
    assert got_headers["anthropic-organization-id"] == REDACTED

    # And the file that was just written is a file that replays.
    monkeypatch.setenv(MODE_ENV_VAR, "replay")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with cassette("round-trip") as client:
        assert _create(client).content[0].text == "replayed"


def test_recording_refuses_to_run_without_credentials(tmp_path, monkeypatch):
    """`record` fails rather than silently producing nothing: a run that was meant to refresh a
    stale cassette and quietly did not is how the stale one survives."""
    monkeypatch.setattr(cassette_module, "CASSETTE_DIR", tmp_path)
    monkeypatch.setenv(MODE_ENV_VAR, "record")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-credentials-here"))

    with pytest.raises(cassette_module.CassetteError, match="billed"):
        with cassette("never-written"):
            pass
    assert not list(tmp_path.glob("*.json"))


def test_a_failed_recording_writes_nothing(tmp_path, monkeypatch, stub_api):
    """A cassette written from a turn that raised would record a conversation that never
    completed, and replaying it would pass for the wrong reason."""
    monkeypatch.setattr(cassette_module, "CASSETTE_DIR", tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-stub-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", stub_api)
    monkeypatch.setenv(MODE_ENV_VAR, "record")

    with pytest.raises(RuntimeError, match="the turn fell over"):
        with cassette("half-a-turn") as client:
            _create(client)
            raise RuntimeError("the turn fell over")

    assert not list(tmp_path.glob("*.json"))


# --- Redaction --------------------------------------------------------------------------------


def test_every_secret_header_is_redacted_before_anything_is_written():
    headers = httpx.Headers(
        {
            "x-api-key": "sk-ant-not-a-real-key",
            "authorization": "Bearer not-a-real-token",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    )
    redacted = dict(tuple(pair) for pair in cassette_module._redacted_headers(headers))

    assert redacted["x-api-key"] == REDACTED
    assert redacted["authorization"] == REDACTED
    # Everything else is kept: Issue #122 says redact the key, not summarise the request.
    assert redacted["anthropic-version"] == "2023-06-01"
    assert redacted["content-type"] == "application/json"


@pytest.mark.parametrize("path", sorted(CASSETTE_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_no_committed_cassette_carries_a_secret(path):
    """Redaction asserted against the artefacts themselves, not only against the function that
    produces them. A committed fixture is a published file.

    Two independent checks, because they fail differently: every header on the redaction list
    must actually be redacted wherever it appears, and no key-shaped string may appear anywhere
    in the file at all — which catches a secret that arrived somewhere other than a header."""
    document = json.loads(path.read_text(encoding="utf-8"))

    for interaction in document["interactions"]:
        for half in ("request", "response"):
            for key, value in interaction[half]["headers"]:
                if key.lower() in cassette_module.SECRET_HEADERS:
                    assert value == REDACTED, f"{half} header {key} was written unredacted"

    text = path.read_text(encoding="utf-8")
    assert "sk-ant-" not in text
    assert "Bearer " not in text


@pytest.mark.parametrize("path", sorted(CASSETTE_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_every_committed_cassette_is_still_fresh_and_says_when_it_was_recorded(path):
    """The fast half of what a live replay would find out slowly, and the check that keeps the
    committed fixtures honest about which prompt version they reflect."""
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["response_source"] == "live_model_call"
    datetime.fromisoformat(document["recorded_at"])
    assert document["interactions"], "a cassette with no interactions replays nothing"
    check_fresh(document, path=path)


def test_there_is_a_committed_cassette_for_every_live_test():
    """Guards the way this could rot quietly: a live test added without a recording would
    otherwise fail only for the person who next ran the whole suite."""
    expected = {
        "answerer_live__schema_valid_draft",
        "answerer_live__request_envelope",
        "pipeline_live__verified_tiered_answer",
        "pipeline_live__answer_and_verify",
    }
    assert {path.stem for path in CASSETTE_DIR.glob("*.json")} == expected
