"""Record-once / replay-many for this repo's live model calls (Issue #122).

    with cassette("answerer_live__schema_valid_draft") as client:
        draft = answer(QUESTION, client=client, serving_url=..., db_path=...)

Same shape as `tests/fixtures/record_answerer_turn.py` (#116), extended from tool payloads to
the model call itself: one real recording is committed, every later run replays it, and **the
fixture states how real it is** so a reader never has to guess. `record_answerer_turn.py`
labels its file with `draft_source`; a cassette labels itself with `recorded_at`, the model and
request configuration it was recorded against, and a hash of every prompt and schema that went
into it (`fingerprint`). See "Staleness is loud" below.

**What is replayed, and what stays real.** Only the HTTP conversation with the Anthropic API.
Everything else in a live test still runs for real: the uvicorn serving process, the replayed
bearing state, the read-only MCP server subprocess, the inventory database, and the tool calls
themselves. `tool_runner` drives its loop exactly as it does live -- it receives a recorded
`tool_use` block, *actually calls the tool*, and sends the real result back -- so the wiring
these tests exist to prove is still exercised end to end. What a replay stops proving is that
the model would make those choices again today; that is what `--record` and the live mode are
for, and it is why the fingerprint check below is a failure rather than a warning.

**Where the seam is.** At the httpx transport, not in `src/agent/`. `answerer.py` and
`escalation.py` already accept a `client`, so a cassette is an `AsyncAnthropic` built over a
custom transport and handed in -- no production code knows this module exists, which is what
Issue #122's "test infrastructure only" constraint requires. It also means the fixture holds
the genuine raw request and response, not an SDK object re-serialized.

**Three modes**, resolved from `--cassette-mode` / `--record` (see `tests/conftest.py`) or the
`AGENT_CASSETTE_MODE` environment variable:

- `replay` (**the default**) -- no network call, no API key, free. This is what CI runs.
- `record` -- makes the real, billed calls and overwrites the cassette. **Deliberate use
  only**, after a prompt, schema, or model change; see the note in each live test's docstring.
- `live` -- makes the real, billed calls and writes nothing. The opt-in path for hitting the
  real API on purpose without touching a committed fixture.

**Staleness is loud.** A replay first compares the cassette's `fingerprint` against the prompts
and request configuration in `src/agent/` *right now*, and fails with the list of what changed
if they differ. That is deliberate, and it implements `docs/agent_design.md` Section 8's
procedural rule ("no change to a prompt, a tool description, or the model may merge without a
recorded golden-set run in its PR") as something a test enforces rather than something a
reviewer has to remember. A cassette recorded against a prompt that no longer exists is not
evidence about the code in front of you, and silently passing would be worse than failing.

A replay also fails if the conversation diverges from the recording -- a request that matches no
remaining interaction, a turn that arrived with a different number of messages than was
recorded, or a recording that was not fully consumed. All three mean the same thing: the shape
of the turn changed, and the recording no longer describes it.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
from anthropic import AnthropicError, AsyncAnthropic

CASSETTE_DIR = Path(__file__).resolve().parent / "cassettes"
REPO_ROOT = Path(__file__).resolve().parents[2]

MODE_ENV_VAR = "AGENT_CASSETTE_MODE"
REPLAY = "replay"
RECORD = "record"
LIVE = "live"
MODES = (REPLAY, RECORD, LIVE)

CASSETTE_VERSION = 1

# Replay constructs a real `AsyncAnthropic`, which refuses to be built with no credentials at
# all. Nothing authenticates with this: the transport under the client never opens a socket.
REPLAY_API_KEY = "cassette-replay-not-a-real-key"

REDACTED = "<redacted>"

# Redacted before anything is written. `x-api-key` and `authorization` are the two the SDK
# actually sets; the rest are here because a header that carries a secret must never depend on
# someone having remembered to add it to this list at the moment they added it to a request.
# `anthropic-organization-id` comes back on the *response* and is not a credential, but a
# cassette is a committed file in a public repository and an account identifier does not need
# to be in one.
SECRET_HEADERS = frozenset(
    {
        "x-api-key",
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "anthropic-auth-token",
        "anthropic-organization-id",
    }
)

# Dropped when a recorded response is rebuilt: the body is stored decoded, so carrying the
# encoding and length of the *encoded* body forward would make httpx try to gunzip plain JSON.
# httpx recomputes `content-length` from the content it is given.
_REBUILD_DROPPED_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


def _display(path: Path) -> str:
    """A path as a reader of a failure message wants to see it: repo-relative when it is in
    the repo, absolute when a test has pointed the module at a temporary directory."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class CassetteError(AnthropicError, RuntimeError):
    """Base for every way a cassette can fail to describe the run in front of it.

    **`AnthropicError` is in the bases on purpose, and it is load-bearing.** The SDK's request
    path wraps anything a transport raises in `APIConnectionError` -- and then retries it twice
    -- except for errors that are already its own (`_base_client`: "SDK-originated errors
    already carry their own type; don't wrap"). Without this base, "the cassette has no
    recording for this call" would reach a reader as `Connection error.`, three attempts later,
    with the actual reason discarded. That is the exact failure this module exists to make
    legible.
    """


class CassetteMissing(CassetteError):
    """Replay was asked for a cassette that has never been recorded."""


class CassetteStale(CassetteError):
    """The prompts or request configuration changed since the cassette was recorded."""


class CassetteMismatch(CassetteError):
    """The conversation diverged from the recorded one."""


# --- Credentials ----------------------------------------------------------------------


def has_anthropic_credentials() -> bool:
    """Whether the SDK has *something* to authenticate with.

    Lifted unchanged from #113/#115's live tests, which each carried their own copy, and kept
    for the same reason they wrote it: an unset `ANTHROPIC_API_KEY` does not by itself mean
    there are no credentials -- the SDK also reads `ANTHROPIC_AUTH_TOKEN` and a stored
    `ant auth login` profile. Checking only the one env var would refuse to record on a
    machine where recording would have worked.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    config_dir = Path(
        os.environ.get("ANTHROPIC_CONFIG_DIR", Path.home() / ".config" / "anthropic")
    )
    credentials = config_dir / "credentials"
    return credentials.is_dir() and any(credentials.glob("*.json"))


# --- Mode ---------------------------------------------------------------------------


def resolve_mode(override: str | None = None) -> str:
    """`replay` unless something explicitly asked for otherwise.

    The default is the whole point of the issue: a run that says nothing about cassettes must
    cost nothing and need no key.
    """
    mode = (override or os.environ.get(MODE_ENV_VAR) or REPLAY).strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown cassette mode {mode!r}; expected one of {list(MODES)}")
    return mode


def _require_credentials(mode: str) -> None:
    if not has_anthropic_credentials():
        raise CassetteError(
            f"cassette mode {mode!r} makes real, billed Anthropic API calls and no credentials "
            "were found (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` "
            "profile). The default mode is 'replay', which needs neither."
        )


# --- Fingerprint: what a cassette was recorded against --------------------------------


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _tool_descriptions_sha256() -> str:
    """A hash of the read-only server's tools, name and description, sorted by name.

    These are the answerer's tools (Section 5's agent boundaries), and their descriptions
    reach the model exactly as the system prompt does -- in the request the cassette
    records. `build_server()` needs no arguments for this: nothing here calls a tool, so the
    base URL, database path and search function it would otherwise bind are irrelevant to
    what gets registered and listed. Sorted by name before canonicalizing, the same
    determinism `draft_schema_sha256` gets for free from a schema that is already one object.
    """
    import asyncio

    from src.agent.mcp.readonly_server import build_server as build_readonly_server

    server, _budget = build_readonly_server()
    tools = asyncio.run(server.list_tools())
    described = sorted(
        ({"name": tool.name, "description": tool.description} for tool in tools),
        key=lambda entry: entry["name"],
    )
    return _sha256(_canonical(described))


def current_fingerprint() -> dict[str, Any]:
    """The prompts and request configuration a cassette recorded now would be recording.

    Hashes rather than copies of the prompts: the point is to detect that one changed, and a
    committed fixture carrying two full system prompts would turn every prompt edit into a
    large, unreadable diff in a file nobody reads. The model, effort, thinking and token
    settings are stored as values, because those are the ones a reader of a stale-cassette
    failure wants to see named.

    Imported from `src/agent/` rather than restated, for the same reason `train_serving_model`
    imports its configuration instead of re-declaring it: a fingerprint that could drift from
    what the code actually sends would report freshness it has not checked.

    `tool_descriptions_sha256` lives in the `answerer` section, not a section of its own: the
    read-only tools it hashes are the answerer's only tool source (`readonly_tools()`), so a
    tool-description edit is a change to what the answerer sends, the same as its system
    prompt. Section 8 names three things a cassette must go stale on -- "a prompt, a tool
    description, or the model" -- and this is the middle one, previously unchecked.
    """
    from src.agent import answerer
    from src.agent.critic import escalation

    return {
        "answerer": {
            "model": answerer.MODEL,
            "max_tokens": answerer.MAX_TOKENS,
            "effort": answerer.EFFORT,
            "thinking": answerer.THINKING,
            "system_prompt_sha256": _sha256(answerer.SYSTEM_PROMPT),
            "draft_schema_sha256": _sha256(_canonical(answerer.draft_schema())),
            "tool_descriptions_sha256": _tool_descriptions_sha256(),
        },
        "critic": {
            "model": escalation.MODEL,
            "max_tokens": escalation.MAX_TOKENS,
            "effort": escalation.EFFORT,
            "thinking": escalation.THINKING,
            "system_prompt_sha256": _sha256(escalation.SYSTEM_PROMPT),
            "verdict_schema_sha256": _sha256(_canonical(escalation.verdict_schema())),
        },
    }


def _fingerprint_differences(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Every field that changed, named in the terms the code uses, in order."""
    differences: list[str] = []
    for agent in sorted(set(recorded) | set(current)):
        recorded_agent = recorded.get(agent, {})
        current_agent = current.get(agent, {})
        for field in sorted(set(recorded_agent) | set(current_agent)):
            was = recorded_agent.get(field)
            now = current_agent.get(field)
            if was != now:
                differences.append(f"{agent}.{field}: recorded {was!r}, now {now!r}")
    return differences


# --- The fixture file ------------------------------------------------------------------


def cassette_path(name: str) -> Path:
    return CASSETTE_DIR / f"{name}.json"


def load_cassette(name: str) -> dict[str, Any]:
    path = cassette_path(name)
    if not path.is_file():
        raise CassetteMissing(
            f"no cassette at {_display(path)}. Record one with "
            f"`pytest --record` (this makes real, billed API calls and needs credentials)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def usage_totals(interactions: list[dict[str, Any]]) -> dict[str, int]:
    """Token usage summed over a cassette's responses, as the API itself reported it.

    Every field the responses carried is summed, rather than a fixed list, so a usage field
    added by the API later is reported instead of silently dropped.
    """
    totals: dict[str, int] = {}
    for interaction in interactions:
        usage = interaction.get("response", {}).get("body", {})
        usage = usage.get("usage") if isinstance(usage, dict) else None
        if not isinstance(usage, dict):
            continue
        for field, value in usage.items():
            if isinstance(value, int):
                totals[field] = totals.get(field, 0) + value
    return dict(sorted(totals.items()))


# --- Transports -------------------------------------------------------------------------


def _redacted_headers(headers: httpx.Headers) -> list[list[str]]:
    return [
        [key, REDACTED if key.lower() in SECRET_HEADERS else value]
        for key, value in headers.multi_items()
    ]


def _encode_body(raw: bytes) -> tuple[str, Any]:
    """Store JSON as JSON. A committed fixture that a reviewer can read is worth more than a
    byte-exact blob, and the SDK parses this body as JSON on the way back in either case."""
    try:
        return "json", json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return "text", raw.decode("utf-8", errors="replace")


def _decode_body(encoding: str, body: Any) -> bytes:
    if encoding == "json":
        return json.dumps(body).encode("utf-8")
    return str(body).encode("utf-8")


def _rebuild_response(recorded: dict[str, Any], request: httpx.Request) -> httpx.Response:
    """One recorded response, back as the thing httpx will hand to the SDK.

    Used by both modes on purpose: recording returns the rebuilt response rather than the
    original one, so a cassette that cannot be turned back into a usable response fails at the
    moment it is recorded rather than the first time someone replays it.
    """
    headers = [
        (key, value)
        for key, value in recorded["headers"]
        if key.lower() not in _REBUILD_DROPPED_HEADERS
    ]
    return httpx.Response(
        status_code=recorded["status_code"],
        headers=headers,
        content=_decode_body(recorded.get("body_encoding", "json"), recorded["body"]),
        request=request,
    )


def _record_request(request: httpx.Request, raw: bytes) -> dict[str, Any]:
    encoding, body = _encode_body(raw)
    return {
        "method": request.method,
        "url": str(request.url),
        "url_path": request.url.path,
        "headers": _redacted_headers(request.headers),
        "body_encoding": encoding,
        "body": body,
    }


class _RecordingTransport(httpx.AsyncBaseTransport):
    """The real transport, with every exchange kept."""

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.interactions: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raw_request = await request.aread()
        response = await self._inner.handle_async_request(request)
        raw_response = await response.aread()
        await response.aclose()

        encoding, body = _encode_body(raw_response)
        recorded_response = {
            "status_code": response.status_code,
            "headers": _redacted_headers(response.headers),
            "body_encoding": encoding,
            "body": body,
        }
        self.interactions.append(
            {
                "request": _record_request(request, raw_request),
                "response": recorded_response,
            }
        )
        return _rebuild_response(recorded_response, request)


class _ReplayTransport(httpx.AsyncBaseTransport):
    """The recorded exchanges, in place of a network.

    **Matching is by method and URL path, then in recorded order** -- not by request body.
    Two things in a request body legitimately differ on every run and neither says anything
    about whether the recording still applies: `src/agent/untrusted.py` mints a fresh nonce per
    request, and every tool result carries a `retrieved_at` stamped when the tool ran. Matching
    on the body would therefore never match anything.

    What *is* checked is the shape of the conversation: a turn must arrive carrying the same
    number of messages the recorded turn did, and the recording must be fully consumed. Both
    hold trivially when nothing changed -- replay returns the recorded assistant turns, so the
    tool calls and therefore the message sequence are the recorded ones -- and both break
    loudly when the harness stops driving the loop the way it did.
    """

    def __init__(self, interactions: list[dict[str, Any]], *, path: Path) -> None:
        self._remaining = list(interactions)
        self._path = path
        self.replayed = 0

    def _where(self) -> str:
        return f"cassette {_display(self._path)}"

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raw = await request.aread()

        for index, interaction in enumerate(self._remaining):
            recorded = interaction["request"]
            if recorded["method"] == request.method and recorded["url_path"] == request.url.path:
                self._remaining.pop(index)
                self._check_shape(recorded, raw)
                self.replayed += 1
                return _rebuild_response(interaction["response"], request)

        raise CassetteMismatch(
            f"{self._where()} has no remaining recorded {request.method} {request.url.path}; "
            f"{self.replayed} interaction(s) already replayed. The call sequence changed since "
            "the recording -- re-record with `pytest --record`."
        )

    def _check_shape(self, recorded: dict[str, Any], raw: bytes) -> None:
        if recorded.get("body_encoding") != "json":
            return
        try:
            sent = json.loads(raw)
        except ValueError:
            return
        was = recorded["body"]
        if not isinstance(sent, dict) or not isinstance(was, dict):
            return
        sent_messages = sent.get("messages")
        recorded_messages = was.get("messages")
        if not isinstance(sent_messages, list) or not isinstance(recorded_messages, list):
            return
        if len(sent_messages) != len(recorded_messages):
            raise CassetteMismatch(
                f"{self._where()}: interaction {self.replayed} was recorded with "
                f"{len(recorded_messages)} message(s) and this run sent {len(sent_messages)}. "
                "The conversation diverged from the recording -- re-record with "
                "`pytest --record`."
            )

    def assert_fully_consumed(self) -> None:
        if self._remaining:
            paths = ", ".join(
                f"{item['request']['method']} {item['request']['url_path']}"
                for item in self._remaining
            )
            raise CassetteMismatch(
                f"{self._where()}: {len(self._remaining)} recorded interaction(s) were never "
                f"replayed ({paths}). The run made fewer model calls than the recording -- "
                "re-record with `pytest --record`."
            )


# --- The client ---------------------------------------------------------------------------


def _write_cassette(
    path: Path,
    *,
    name: str,
    interactions: list[dict[str, Any]],
    notes: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "cassette_version": CASSETTE_VERSION,
        "name": name,
        "response_source": "live_model_call",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": current_fingerprint(),
        "notes": notes or {},
        "usage_totals": usage_totals(interactions),
        "interactions": interactions,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def check_fresh(document: dict[str, Any], *, path: Path) -> None:
    """Raise `CassetteStale` if the code moved on since this cassette was recorded."""
    differences = _fingerprint_differences(document.get("fingerprint", {}), current_fingerprint())
    if not differences:
        return
    listed = "\n  ".join(differences)
    raise CassetteStale(
        f"{_display(path)} was recorded on "
        f"{document.get('recorded_at', 'an unknown date')} against a different prompt or "
        f"request configuration:\n  {listed}\n"
        "A replay of it is not evidence about the code in front of you. Re-record with "
        "`pytest --record` (real, billed API calls) and commit the result with the PR that "
        "changed the prompt -- `docs/agent_design.md` Section 8 requires exactly that."
    )


@contextmanager
def cassette(
    name: str,
    *,
    mode: str | None = None,
    notes: dict[str, Any] | None = None,
) -> Iterator[AsyncAnthropic]:
    """An `AsyncAnthropic` that replays, records, or really calls, per the resolved mode.

    `notes` is written into a recording as-is: free-form context about the environment the
    recording was made in, which for these tests is mainly whether the documentation index was
    reachable. It is recorded rather than asserted -- see each live test's docstring for why
    the recordings are deliberately made without Qdrant.

    The httpx client is deliberately not closed. In replay its transport owns no resources at
    all, and in record mode the pool is bound to the event loop the test already finished with,
    so closing it from here would mean opening a second loop to tear down a connection in a
    process that is about to exit anyway.
    """
    resolved = resolve_mode(mode)

    if resolved == LIVE:
        _require_credentials(LIVE)
        yield AsyncAnthropic()
        return

    if resolved == RECORD:
        _require_credentials(RECORD)
        transport = _RecordingTransport()
        yield AsyncAnthropic(http_client=httpx.AsyncClient(transport=transport))
        # Only on a clean exit: a cassette written from a half-failed turn would record a
        # conversation that never completed, and replaying it would pass for the wrong reason.
        _write_cassette(
            cassette_path(name), name=name, interactions=transport.interactions, notes=notes
        )
        return

    path = cassette_path(name)
    document = load_cassette(name)
    check_fresh(document, path=path)
    transport = _ReplayTransport(document["interactions"], path=path)
    yield AsyncAnthropic(
        api_key=REPLAY_API_KEY, http_client=httpx.AsyncClient(transport=transport)
    )
    transport.assert_fully_consumed()
