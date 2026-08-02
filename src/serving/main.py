"""The one documented way to run the serving API (Issue #84).

    python -m src.serving.main

`docs/serving_design.md` Section 2's single-worker constraint is enforced here at the
`uvicorn` call itself, not just stated in a comment: `app` below is an already-constructed
`FastAPI` instance, not an "import string" (`"src.serving.api:app"`). Passing a live object
is exactly what makes `uvicorn.run(..., workers=N)` for `N > 1` refuse to start --
`uvicorn` can only fork additional workers by re-importing the app fresh in each child
process, which it cannot do with an object it was merely handed. Confirmed empirically
(see the PR for Issue #84): `uvicorn.run(app, workers=2)` exits immediately with
`SystemExit(3)` and "You must pass the application as an import string to enable 'reload'
or 'workers'." -- it does not silently fall back to one worker.

That protects this exact entrypoint. It does not protect `uvicorn src.serving.api:app
--workers N` invoked directly (an import string, which *is* what multi-worker mode needs)
or any other launcher this module knows nothing about -- `src/serving/single_worker.py`'s
OS-level lock, taken inside `create_app`'s own startup, is the second, launcher-independent
layer that catches those. Together the two mean there is no run path -- this one or any
other -- that ends in two processes silently holding two copies of one bearing's state.
"""
from __future__ import annotations

import uvicorn

from src.serving.api import create_app

app = create_app()


def main() -> None:
    # workers is intentionally hardcoded, not read from an environment variable or CLI
    # flag -- see the module docstring for why any value other than 1 here would fail
    # loudly at startup rather than silently do the wrong thing.
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()
