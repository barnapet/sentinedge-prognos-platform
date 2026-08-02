"""Enforces `docs/serving_design.md` Section 2's single-process constraint (Issue #84).

`src/serving/state.py`'s `BearingStateStore` is a plain Python object living in one
process's memory. A worker process has no way to ask "how many siblings did my supervisor
spawn" -- that information lives in the supervisor (`uvicorn --workers N`, `gunicorn -w N`,
a second `docker run`, or just two terminals running `python -m src.serving.main`), not in
the worker itself. So this is enforced the same way a second `flock` on the same file always
is, regardless of who spawned the second process or how: an exclusive, non-blocking OS file
lock on one fixed path. Exactly one process can hold it; every other attempt fails
immediately with a clear, actionable error, at `/predict`-serving startup rather than as a
silent divergence discovered later in production traffic.

This is the second, launcher-independent layer. `src/serving/main.py` is the first and
cheaper one: it passes an already-constructed `FastAPI` app object to `uvicorn.run`, which
`uvicorn` itself refuses to run with `workers > 1` (`uvicorn` requires an import string,
not a live object, to fork workers -- confirmed empirically, see the PR for Issue #84) and
exits immediately rather than starting one worker and pretending the rest were honoured.
That protects the *documented* run command specifically. This module protects every other
way the constraint could be violated -- `uvicorn src.serving.api:app --workers N` invoked
directly, two separate `main.py` processes, a supervisor neither of those two developers
anticipated -- by making the violation fail at the one place all of them pass through:
this app's own startup.
"""
from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path
from typing import IO

DEFAULT_LOCK_PATH = Path(tempfile.gettempdir()) / "sentinedge-prognos-serving.lock"


class SingleWorkerViolation(RuntimeError):
    """Raised when a second process tries to hold the serving lock concurrently."""


def acquire_single_worker_lock(lock_path: Path = DEFAULT_LOCK_PATH) -> IO[str]:
    """Take the exclusive lock, or raise `SingleWorkerViolation` if it is already held.

    The returned file object is the lock -- keep it open (and pass it to
    `release_single_worker_lock` at shutdown) for as long as this process serves
    requests; closing or losing it releases the lock early and would let a second
    process start silently overlapping this one's in-memory state.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_file.close()
        raise SingleWorkerViolation(
            f"Another serving process already holds {lock_path}. "
            "docs/serving_design.md Section 2 requires exactly one worker process: "
            "its in-memory per-bearing state is process-local, so a second worker "
            "would hold its own diverging copy rather than sharing this one's history. "
            "Run a single process, e.g. `python -m src.serving.main` -- do not add "
            "`--workers`/`-w` to a raw `uvicorn`/`gunicorn` invocation of this app."
        ) from exc
    # Best-effort diagnostic for whoever finds the lock file while debugging a
    # SingleWorkerViolation -- not load-bearing for the lock itself, which is the
    # flock, not the file's contents.
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def release_single_worker_lock(lock_file: IO[str]) -> None:
    """Release a lock acquired by `acquire_single_worker_lock` and close its file."""
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()
