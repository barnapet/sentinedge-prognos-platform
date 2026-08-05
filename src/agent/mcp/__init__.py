"""The MCP tool layer (Issue #110, `docs/agent_design.md` Section 2).

Two stdio servers, deliberately separate processes:

- `readonly_server` -- `get_bearing_status`, `predict_health_state`, `check_inventory`,
  `search_documentation`. This is the server the answerer connects to.
- `write_server` -- `place_order`, and nothing else. Only the executor connects to it.

The split is the point: Section 2 chose MCP over plain Python callables so least privilege
is a *process boundary* rather than a convention about which functions end up in which
list. A client holding only the read-only server's transport cannot call `place_order` --
not because it is told not to, but because the tool does not exist on the connection it
holds.

Neither server imports anything from `src/serving/`. See `serving_client` for why.
"""
