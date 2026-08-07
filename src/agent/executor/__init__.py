"""Agent C -- the executor (Issue #96, `docs/agent_design.md` Section 5).

`approval.py` is the first piece: the approval-token mechanism behind Section 5's gate,
built independently of the tool that will consume it (`place_order`, `src/agent/mcp/
write_server.py`) and the orchestrator that will call both. Those are later issues.
"""
