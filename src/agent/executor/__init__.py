"""Agent C -- the executor (Issue #96, `docs/agent_design.md` Section 5).

`approval.py` is the first piece: the approval-token mechanism behind Section 5's gate,
built independently of the tool that will consume it (`place_order`, `src/agent/mcp/
write_server.py`) and the orchestrator that will call both.

`client.py` (Issue #126) is the second: the write-only MCP client and its fixed-schema
input, built against `place_order` and `approval.py` as they exist, without modifying
either. The orchestrator that decides when to call it and mints the token it consumes is
still a later issue.
"""
