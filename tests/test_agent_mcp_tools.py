"""Tier-1 tests for the MCP tool bodies (Issue #110, `docs/agent_design.md` Sections 2 and
8). Each tool is called directly -- no model, no agent loop -- and checked for three things
Section 2 fixes: that it wraps the right backing source, that it mints the mandatory
`source` block itself, and that a broken backing source produces an `is_error` tool result
rather than an exception.

**No API key, and no network beyond localhost.** The serving-API tests run against a real
`http.server` bound to `127.0.0.1` (so the HTTP path is genuinely exercised, not stubbed
out), the inventory tests run against a `tmp_path` SQLite database, and the retrieval tests
inject a stub `search` -- there is no Qdrant, no embedding model, and no download anywhere
in this file.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.agent.executor.approval import TOKEN_LIFETIME, ApprovalTokenStore
from src.agent.inventory.build_db import build_db
from src.agent.mcp.results import (
    DOCS_INDEX_UNREACHABLE,
    INVENTORY_UNAVAILABLE,
    SERVICE_UNREACHABLE,
    SOURCE_TYPES,
    TRAJECTORY_UNUSABLE,
    payload_of,
    source_block,
)
from src.agent.mcp.tools import (
    check_inventory,
    find_similar_historical_pattern,
    get_bearing_status,
    place_order,
    predict_health_state,
    search_documentation,
)
from src.agent.similarity.archive import CAVEAT, DTW_CHANNELS, NO_MATCH_THRESHOLD
from src.agent.rag.retrieval import RetrievedChunk

# A port nothing is listening on. 9 is the standard "discard" port and is not bound in CI;
# a connection attempt fails immediately rather than hanging.
CLOSED_PORT_URL = "http://127.0.0.1:9"

DRIFT_BODY = {
    "bearings": {
        "2nd_test-demo": {
            "file_count": 120,
            "baseline_status": "stable",
            "drift_status": "stable",
            "features": {"rms": {"z": 0.4, "drifting": False}},
            "rms_ratio_latest": 1.02,
            "predicted_class_counts": {"Normal": 118, "Degrading": 2},
        }
    }
}

PREDICT_BODY = {
    "label": "Normal",
    "baseline_status": "stable",
    "drift_status": "stable",
    "model_notes": "…",
}


def history_body(channels: dict[str, list[float]], **overrides) -> dict:
    """A `GET /monitoring/history/{bearing_id}` body, in the endpoint's exact shape."""
    n_points = len(next(iter(channels.values()))) if channels else 0
    return {
        "bearing_id": "2nd_test-demo",
        "found": True,
        "file_count": max(n_points, 120),
        "baseline_status": "stable",
        "retained": 50,
        "channels": channels,
        "n_points": n_points,
        **overrides,
    }


def archived_window(experiment: str = "2nd_test", start: int = 800, length: int = 50) -> dict:
    """A real 50-point stretch of a real archived trajectory, in channel form.

    Taken from the committed archive rather than invented, so the distance the tool
    computes from it is a real distance -- these tests then assert on the *envelope and
    decision*, with `tests/test_agent_similarity_archive.py` owning the metric itself.
    """
    from src.agent.similarity.archive import DTW_CHANNELS, load_archive

    trajectory = next(t for t in load_archive() if t.experiment == experiment)
    chunk = trajectory.raw_matrix[start : start + length]
    return {channel: list(chunk[:, i]) for i, channel in enumerate(DTW_CHANNELS)}


def default_history_body() -> dict:
    return history_body(archived_window())


# --------------------------------------------------------------------------------------
# A real localhost stand-in for the serving API
# --------------------------------------------------------------------------------------


class _FakeServingAPI:
    """A real HTTP server answering the two endpoints the tools call.

    Deliberately a socket rather than a monkeypatched function: Section 2's whole point is
    that these tools reach the serving process **over HTTP**, and a test that patched the
    client out would pass just as happily against an in-process `create_app()` import --
    the one thing that must never work.
    """

    def __init__(self, status: int = 200, history_body: dict | None = None):
        self.status = status
        self.requests: list[tuple[str, dict | None]] = []
        self.history_body = history_body if history_body is not None else default_history_body()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _respond(self, body: dict) -> None:
                payload = json.dumps(body).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming
                outer.requests.append((self.path, None))
                # Routed rather than answered uniformly, since #140 added a second GET the
                # tools call. `self.path` is the raw request line, so a test can assert on
                # exactly what went over the wire, percent-encoding included.
                if self.path.startswith("/monitoring/history/"):
                    self._respond(outer.history_body)
                else:
                    self._respond(DRIFT_BODY)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append((self.path, body))
                self._respond(PREDICT_BODY)

            def log_message(self, *args) -> None:  # keep pytest output clean
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "_FakeServingAPI":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def serving_api():
    with _FakeServingAPI() as api:
        yield api


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "inventory.db"
    build_db(path)
    return path


def _some_part(db_path) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT part_number, bearing_type, quantity_on_hand FROM parts "
            "WHERE bearing_type IS NOT NULL ORDER BY part_number LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return {"part_number": row[0], "bearing_type": row[1], "quantity_on_hand": row[2]}


# --------------------------------------------------------------------------------------
# The source block: minted by the tool layer, never by the model
# --------------------------------------------------------------------------------------


def test_source_block_rejects_a_source_type_outside_section_2s_vocabulary():
    with pytest.raises(ValueError, match="unknown source_type"):
        source_block("documentation_index", "whatever")


def test_source_block_stamps_a_timezone_aware_iso_timestamp():
    block = source_block("inventory", "data/agent/inventory.db")
    stamped = datetime.fromisoformat(block["retrieved_at"])
    assert stamped.tzinfo is not None, "retrieved_at must be unambiguous, not naive local time"


def test_every_tool_result_carries_the_three_mandatory_source_fields(serving_api, db_path):
    results = [
        get_bearing_status(base_url=serving_api.url),
        predict_health_state("b", [1.0, 2.0], base_url=serving_api.url),
        check_inventory(db_path=db_path),
        search_documentation("x", search=lambda *a, **k: []),
    ]
    for result in results:
        source = payload_of(result)["source"]
        assert set(source) == {"source_type", "source_id", "retrieved_at"}
        assert source["source_type"] in SOURCE_TYPES


def test_a_failed_result_still_carries_a_source_block_but_no_data_key():
    payload = payload_of(get_bearing_status(base_url=CLOSED_PORT_URL))
    assert payload["source"]["source_id"] == "GET /monitoring/drift"
    assert "data" not in payload, "an error result must not look like an empty answer"
    assert payload["error"]


# --------------------------------------------------------------------------------------
# get_bearing_status -- wraps GET /monitoring/drift
# --------------------------------------------------------------------------------------


def test_get_bearing_status_returns_every_tracked_bearing_when_none_is_named(serving_api):
    result = get_bearing_status(base_url=serving_api.url)

    assert result.is_error is False
    assert payload_of(result)["data"]["bearings"] == DRIFT_BODY["bearings"]
    assert serving_api.requests == [("/monitoring/drift", None)]


def test_get_bearing_status_returns_one_bearings_state_when_named(serving_api):
    data = payload_of(get_bearing_status("2nd_test-demo", base_url=serving_api.url))["data"]

    assert data["found"] is True
    assert data["status"]["baseline_status"] == "stable"
    assert data["status"]["predicted_class_counts"] == {"Normal": 118, "Degrading": 2}


def test_an_unknown_bearing_is_a_structured_not_found_not_an_error(serving_api):
    """`docs/agent_design.md` Section 10 case 1: inventing a state for an unknown bearing
    is the failure being tested for, so the result must be readable as "no such bearing"
    and must contain no health-state label to invent one from."""
    result = get_bearing_status("no-such-bearing", base_url=serving_api.url)

    assert result.is_error is False
    data = payload_of(result)["data"]
    assert data["found"] is False
    assert data["tracked_bearings"] == ["2nd_test-demo"]
    assert "status" not in data and "label" not in data


def test_get_bearing_status_reports_an_unreachable_service_as_a_tool_result():
    result = get_bearing_status(base_url=CLOSED_PORT_URL)

    assert result.is_error is True
    assert payload_of(result)["error"] == SERVICE_UNREACHABLE


def test_an_error_status_is_reported_as_a_rejection_not_as_unreachability():
    """A service that answered 500 is up; calling that "not reachable" would send a
    technician to check the wrong thing."""
    with _FakeServingAPI(status=500) as api:
        payload = payload_of(get_bearing_status(base_url=api.url))

    assert payload["error"] != SERVICE_UNREACHABLE
    assert "500" in payload["error"]


# --------------------------------------------------------------------------------------
# predict_health_state -- wraps POST /predict
# --------------------------------------------------------------------------------------


def test_predict_health_state_sends_exactly_section_1s_payload(serving_api):
    signal = [0.1, -0.2, 0.3]
    result = predict_health_state("1st_test-demo", signal, base_url=serving_api.url)

    path, body = serving_api.requests[-1]
    assert path == "/predict"
    # `docs/serving_design.md` Section 1: a raw signal and a bearing id, and nothing else.
    # No file index, no sequence number, and above all no pre-computed features -- the
    # server owns 100% of feature computation and this client must not send any.
    assert set(body) == {"bearing_id", "signal"}
    assert body["signal"] == signal
    assert payload_of(result)["data"]["label"] == "Normal"


def test_predict_health_state_refuses_an_empty_signal_without_calling_the_service(serving_api):
    result = predict_health_state("b", [], base_url=serving_api.url)

    assert result.is_error is True
    assert serving_api.requests == []


def test_predict_health_state_reports_an_unreachable_service_as_a_tool_result():
    result = predict_health_state("b", [1.0], base_url=CLOSED_PORT_URL)

    assert result.is_error is True
    assert payload_of(result)["error"] == SERVICE_UNREACHABLE


# --------------------------------------------------------------------------------------
# check_inventory -- wraps the Issue #101 query module
# --------------------------------------------------------------------------------------


def test_check_inventory_returns_the_whole_catalogue_when_unfiltered(db_path):
    data = payload_of(check_inventory(db_path=db_path))["data"]

    conn = sqlite3.connect(db_path)
    try:
        expected = conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
    finally:
        conn.close()
    assert data["match_count"] == expected == len(data["parts"])
    assert set(data["parts"][0]) >= {
        "part_number",
        "description",
        "quantity_on_hand",
        "unit_price_usd",
        "lead_time_days",
        "location",
    }


def test_check_inventory_filters_by_part_number_and_by_bearing_type(db_path):
    part = _some_part(db_path)

    by_number = payload_of(check_inventory(part_number=part["part_number"], db_path=db_path))
    assert [p["part_number"] for p in by_number["data"]["parts"]] == [part["part_number"]]

    by_type = payload_of(check_inventory(bearing_type=part["bearing_type"], db_path=db_path))
    assert by_type["data"]["parts"]
    assert all(p["bearing_type"] == part["bearing_type"] for p in by_type["data"]["parts"])


def test_no_matching_part_is_a_successful_empty_result_not_an_error(db_path):
    """"We do not stock that part" is an answer. Returning it as a failure would push a
    perfectly good answer onto Section 6's degraded path."""
    result = check_inventory(part_number="NO-SUCH-PART", db_path=db_path)

    assert result.is_error is False
    assert payload_of(result)["data"]["match_count"] == 0


def test_check_inventory_never_creates_a_database_it_cannot_find(tmp_path):
    missing = tmp_path / "absent.db"
    result = check_inventory(db_path=missing)

    assert result.is_error is True
    assert payload_of(result)["error"] == INVENTORY_UNAVAILABLE
    assert not missing.exists(), "a read-only tool must not bring a database into existence"


def test_check_inventory_holds_a_read_only_connection(db_path, monkeypatch):
    """The read tool on the read-only server opens SQLite in `mode=ro`, so a write cannot
    escape through it even if something in the query path went wrong."""
    calls: list[tuple[tuple, dict]] = []
    real_connect = sqlite3.connect

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    check_inventory(db_path=db_path)

    assert len(calls) == 1
    (args, kwargs) = calls[0]
    # Re-open on the tool's own connect arguments (the tool closes its connection) and show
    # the resulting handle genuinely refuses a write, rather than trusting the string.
    monkeypatch.undo()
    conn = sqlite3.connect(*args, **kwargs)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("UPDATE parts SET quantity_on_hand = 0")
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# search_documentation -- wraps the Issue #99 index
# --------------------------------------------------------------------------------------


def _chunk(**overrides) -> RetrievedChunk:
    fields = {
        "chunk_id": "docs/model_training_decision.md::7",
        "source_type": "decision_doc",
        "source_id": "docs/model_training_decision.md",
        "source_ref": "docs/model_training_decision.md",
        "heading_path": "Model Training Decision > 6. The headline result",
        "chunk_index": 7,
        "text": "Critical recall is 0.913 / 1.000 on 2nd_test/3rd_test and 0.059 on 1st_test.",
        "score": 0.62,
    }
    return RetrievedChunk(**{**fields, **overrides})


def test_each_retrieved_chunk_carries_its_own_citable_source_block():
    """Section 6 verifies citations by set membership over the ids that appeared in this
    turn's tool results, so each chunk's stable `chunk_id` has to arrive as an id, not be
    reconstructable-in-principle from the payload."""
    result = search_documentation("critical recall", search=lambda *a, **k: [_chunk()])

    (hit,) = payload_of(result)["data"]["results"]
    assert hit["source"] == {
        "source_type": "decision_doc",
        "source_id": "docs/model_training_decision.md::7",
        "source_ref": "docs/model_training_decision.md",
    }
    assert hit["source"]["source_type"] in SOURCE_TYPES


def test_chunk_text_is_returned_verbatim():
    """The numeric-fidelity check (Section 6) substring-matches a claim's numbers against
    chunk text. Any reformatting here would break it silently."""
    chunk = _chunk()
    result = search_documentation("q", search=lambda *a, **k: [chunk])

    assert payload_of(result)["data"]["results"][0]["text"] == chunk.text


def test_search_arguments_are_passed_through_to_the_retrieval_layer():
    seen = {}

    def spy(query, limit, source_type):
        seen.update(query=query, limit=limit, source_type=source_type)
        return []

    search_documentation("bearing baseline", 3, "decision_doc", search=spy)

    assert seen == {"query": "bearing baseline", "limit": 3, "source_type": "decision_doc"}


def test_an_unreachable_index_is_a_tool_result_not_an_exception():
    def explode(*args, **kwargs):
        raise ConnectionError("[Errno 111] Connection refused")

    result = search_documentation("q", search=explode)

    assert result.is_error is True
    assert payload_of(result)["error"] == DOCS_INDEX_UNREACHABLE


def test_an_unknown_source_type_filter_is_refused_before_any_retrieval():
    def explode(*args, **kwargs):
        raise AssertionError("retrieval must not run on an invalid filter")

    result = search_documentation("q", source_type="maintenance_log", search=explode)

    assert result.is_error is True


# --------------------------------------------------------------------------------------
# place_order -- wraps the Issue #101 transactional path; the approval-token gate is #125,
# validated against `src/agent/executor/approval.py` (#124, not modified here).
# --------------------------------------------------------------------------------------


def _order_count(db_path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    finally:
        conn.close()


def _stock(db_path, part_number: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT quantity_on_hand FROM parts WHERE part_number = ?", (part_number,)
        ).fetchone()[0]
    finally:
        conn.close()


def _minted(
    store: ApprovalTokenStore,
    part_number: str,
    quantity: int,
    bearing_id: str | None = None,
    approved_by: str = "supervisor-02",
):
    """Mint a token scoped to exactly the order a test is about to place."""
    return store.mint(part_number, quantity, bearing_id, approved_by)


def test_place_order_writes_a_real_row_and_decrements_real_stock(db_path):
    part = _some_part(db_path)
    before = _stock(db_path, part["part_number"])
    store = ApprovalTokenStore()
    token = _minted(store, part["part_number"], 2, bearing_id="2nd_test-demo")

    result = place_order(
        part["part_number"],
        2,
        requested_by="tech-01",
        approval_token=token.token,
        bearing_id="2nd_test-demo",
        token_store=store,
        db_path=db_path,
    )

    data = payload_of(result)["data"]
    assert result.is_error is False
    assert _stock(db_path, part["part_number"]) == before - 2
    assert _order_count(db_path) == 1
    # approved_by/approved_at on the response are what the tool derived from the token,
    # not anything a caller passed -- there is no longer an argument to pass.
    assert data["approved_by"] == "supervisor-02"
    assert data["approved_at"] == token.approved_at.isoformat()

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT approved_by, approved_at, quantity FROM orders WHERE order_id = ?",
            (data["order_id"],),
        ).fetchone()
    finally:
        conn.close()
    # The written row matches the *token record*, not anything the test could have supplied
    # as a tool argument.
    assert row == ("supervisor-02", token.approved_at.isoformat(), 2)


def test_an_oversell_leaves_no_partial_write_behind(db_path):
    part = _some_part(db_path)
    before = _stock(db_path, part["part_number"])
    store = ApprovalTokenStore()
    token = _minted(store, part["part_number"], before + 1)

    result = place_order(
        part["part_number"],
        before + 1,
        requested_by="tech-01",
        approval_token=token.token,
        token_store=store,
        db_path=db_path,
    )

    assert result.is_error is True
    assert _stock(db_path, part["part_number"]) == before
    assert _order_count(db_path) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"part_number": "NO-SUCH-PART", "quantity": 1}, id="unknown-part"),
        pytest.param({"quantity": 0}, id="zero-quantity"),
        pytest.param({"quantity": -3}, id="negative-quantity"),
    ],
)
def test_a_rejected_order_is_a_tool_result_and_writes_nothing(db_path, kwargs):
    part = _some_part(db_path)
    part_number = kwargs.get("part_number", part["part_number"])
    quantity = kwargs["quantity"]
    store = ApprovalTokenStore()
    # zero/negative quantity is rejected before the token is even checked (see tools.py);
    # minting a token scoped to it anyway keeps this test agnostic to that ordering.
    token = _minted(store, part_number, quantity)

    result = place_order(
        part_number,
        quantity,
        requested_by="tech-01",
        approval_token=token.token,
        token_store=store,
        db_path=db_path,
    )

    assert result.is_error is True
    assert payload_of(result)["error"]
    assert _order_count(db_path) == 0


def test_place_order_reports_a_missing_database_rather_than_raising(tmp_path):
    store = ApprovalTokenStore()
    token = _minted(store, "BRG-6205-2RS", 1)

    result = place_order(
        "BRG-6205-2RS",
        1,
        requested_by="tech-01",
        approval_token=token.token,
        token_store=store,
        db_path=tmp_path / "absent.db",
    )

    assert result.is_error is True
    assert payload_of(result)["error"] == INVENTORY_UNAVAILABLE


# --- Issue #125: the four token-rejection paths, and that each leaves `orders` alone -----


def test_an_unknown_token_is_rejected_and_orders_is_unchanged(db_path):
    part = _some_part(db_path)
    store = ApprovalTokenStore()

    result = place_order(
        part["part_number"],
        1,
        requested_by="tech-01",
        approval_token="not-a-real-token",
        token_store=store,
        db_path=db_path,
    )

    assert result.is_error is True
    assert payload_of(result)["error"]
    assert _order_count(db_path) == 0


def test_an_expired_token_is_rejected_and_orders_is_unchanged(db_path):
    part = _some_part(db_path)
    now = datetime(2026, 8, 5, 9, 0, 0)
    clock = [now]
    store = ApprovalTokenStore(clock=lambda: clock[0])
    token = _minted(store, part["part_number"], 1)

    clock[0] = now + TOKEN_LIFETIME + timedelta(seconds=1)
    result = place_order(
        part["part_number"],
        1,
        requested_by="tech-01",
        approval_token=token.token,
        token_store=store,
        db_path=db_path,
    )

    assert result.is_error is True
    assert payload_of(result)["error"]
    assert _order_count(db_path) == 0


def test_an_already_used_token_is_rejected_and_orders_is_unchanged(db_path):
    part = _some_part(db_path)
    store = ApprovalTokenStore()
    token = _minted(store, part["part_number"], 1)
    first = place_order(
        part["part_number"],
        1,
        requested_by="tech-01",
        approval_token=token.token,
        token_store=store,
        db_path=db_path,
    )
    assert first.is_error is False
    assert _order_count(db_path) == 1

    second = place_order(
        part["part_number"],
        1,
        requested_by="tech-01",
        approval_token=token.token,
        token_store=store,
        db_path=db_path,
    )

    assert second.is_error is True
    assert payload_of(second)["error"]
    # No second row: the rejection didn't just fail to charge stock, it wrote nothing.
    assert _order_count(db_path) == 1


def test_a_scope_mismatched_token_is_rejected_and_orders_is_unchanged(db_path):
    """A token minted for a different quantity than the call presents -- otherwise
    identical and otherwise valid -- must still be refused."""
    part = _some_part(db_path)
    store = ApprovalTokenStore()
    token = _minted(store, part["part_number"], 1)

    result = place_order(
        part["part_number"],
        2,  # minted for 1
        requested_by="tech-01",
        approval_token=token.token,
        token_store=store,
        db_path=db_path,
    )

    assert result.is_error is True
    assert payload_of(result)["error"]
    assert _order_count(db_path) == 0

    # The token itself is untouched by the mismatched attempt -- it still consumes
    # correctly against the order it was actually minted for.
    retried = place_order(
        part["part_number"],
        1,
        requested_by="tech-01",
        approval_token=token.token,
        token_store=store,
        db_path=db_path,
    )
    assert retried.is_error is False
    assert _order_count(db_path) == 1


# --------------------------------------------------------------------------------------
# find_similar_historical_pattern (Issue #140, `docs/agent_design.md` Section 12)
# --------------------------------------------------------------------------------------


def test_find_similar_historical_pattern_returns_section_12s_output_shape(serving_api):
    result = find_similar_historical_pattern("2nd_test-demo", base_url=serving_api.url)
    payload = payload_of(result)

    assert result.is_error is False
    assert payload["source"]["source_type"] == "trajectory_match"
    assert payload["source"]["source_id"].startswith("trajectory_archive@")

    data = payload["data"]
    # Section 12's contract: these two are present on every successful result.
    assert data["n_references"] == 3
    assert data["caveat"] == CAVEAT
    assert data["query_window"] == 50
    assert len(data["ranked"]) == 3
    for row in data["ranked"]:
        assert set(row) == {
            "experiment",
            "normalized_distance",
            "matched_index_range",
            "label_at_match",
        }


def test_a_window_taken_from_an_archived_experiment_matches_that_experiment(serving_api):
    """End to end through the HTTP boundary: the query is a real stretch of `2nd_test`, so
    the tool must name `2nd_test`, at its own indices, at distance ~0."""
    result = find_similar_historical_pattern("2nd_test-demo", base_url=serving_api.url)
    best = payload_of(result)["data"]["best_match"]

    assert best["experiment"] == "2nd_test"
    assert best["matched_index_range"] == [800, 849]
    assert best["normalized_distance"] == pytest.approx(0.0, abs=1e-6)


def test_a_shape_unlike_anything_archived_gets_no_best_match():
    """Section 12's refusal to always name a winner. The ranking is still returned, so the
    answer can say what it was closest to *and* that it was not close enough."""
    alternating = [float((-1) ** i) for i in range(50)]
    body = history_body({channel: alternating for channel in DTW_CHANNELS})

    with _FakeServingAPI(history_body=body) as api:
        data = payload_of(find_similar_historical_pattern("odd", base_url=api.url))["data"]

    assert data["best_match"] is None
    assert "no reference within threshold" in data["no_match_reason"]
    assert len(data["ranked"]) == 3
    assert data["ranked"][0]["normalized_distance"] > NO_MATCH_THRESHOLD


def test_an_untracked_bearing_is_a_structured_not_found_not_an_error():
    """Section 10 case 1: an untracked bearing must arrive as a fact about which bearings
    exist, never as a comparison against a trajectory nobody recorded."""
    body = {"bearing_id": "ghost", "found": False, "tracked_bearings": ["a", "b"]}

    with _FakeServingAPI(history_body=body) as api:
        result = find_similar_historical_pattern("ghost", base_url=api.url)

    data = payload_of(result)["data"]
    assert result.is_error is False
    assert data["found"] is False
    assert data["tracked_bearings"] == ["a", "b"]
    assert "best_match" not in data
    assert data["caveat"] == CAVEAT


def test_a_bearing_with_too_little_history_refuses_rather_than_comparing():
    """Not an error -- the honest answer is "ask again later". Inventing a comparison from
    six points would be worse than saying so."""
    body = history_body({channel: [1.0, 2.0, 3.0] for channel in DTW_CHANNELS})

    with _FakeServingAPI(history_body=body) as api:
        result = find_similar_historical_pattern("new-bearing", base_url=api.url)

    data = payload_of(result)["data"]
    assert result.is_error is False
    assert data["best_match"] is None
    assert data["ranked"] == []
    assert "at least 10" in data["no_match_reason"]


def test_a_trajectory_with_a_missing_reading_is_refused_not_interpolated():
    channels = archived_window()
    channels["kurtosis"] = [None] + channels["kurtosis"][1:]
    body = history_body(channels)

    with _FakeServingAPI(history_body=body) as api:
        result = find_similar_historical_pattern("holey", base_url=api.url)

    assert result.is_error is True
    assert TRAJECTORY_UNUSABLE in payload_of(result)["error"]


def test_an_unreachable_serving_api_is_a_readable_failure():
    result = find_similar_historical_pattern("2nd_test-demo", base_url=CLOSED_PORT_URL)

    assert result.is_error is True
    payload = payload_of(result)
    assert payload["error"] == SERVICE_UNREACHABLE
    # The envelope still names which source failed (`results.py`'s one-shape rule).
    assert payload["source"]["source_type"] == "trajectory_match"


def test_a_rejecting_serving_api_is_reported_as_a_rejection_not_as_unreachable():
    """A service that answered is up. Calling that unreachable would send a technician to
    check the wrong thing -- the distinction #110 drew for the other tools."""
    with _FakeServingAPI(status=500) as api:
        result = find_similar_historical_pattern("2nd_test-demo", base_url=api.url)

    assert result.is_error is True
    assert "HTTP 500" in payload_of(result)["error"]


def test_a_window_below_the_minimum_is_refused_before_any_request_is_made(serving_api):
    result = find_similar_historical_pattern("2nd_test-demo", window=2, base_url=serving_api.url)

    assert result.is_error is True
    assert "at least 10" in payload_of(result)["error"]
    assert serving_api.requests == []


def test_the_bearing_id_is_percent_encoded_into_the_request_path(serving_api):
    """`bearing_id` is the one tool argument that reaches a URL *path*, and it can come
    from a model. A slash in it must not re-point the request at another route."""
    find_similar_historical_pattern("odd/name", base_url=serving_api.url)

    (path, _), = serving_api.requests
    assert path.startswith("/monitoring/history/odd%2Fname")
    assert "/monitoring/history/odd/name" not in path


def test_the_requested_window_is_passed_through_to_the_endpoint(serving_api):
    find_similar_historical_pattern("2nd_test-demo", window=30, base_url=serving_api.url)

    (path, _), = serving_api.requests
    assert "window=30" in path
