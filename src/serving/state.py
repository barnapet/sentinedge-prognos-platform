"""Per-bearing rolling state for online feature computation (Issue #82).

`src/features/extraction.py` computes `rms_ratio` and `skewness_smoothed` as `pandas`
column operations over a whole experiment at once. Serving sees one file at a time, so the
same two quantities have to be maintained incrementally. `docs/serving_design.md` Section 2
decided exactly what has to be carried per `bearing_id` to do that, and this module is that
decision in code -- nothing more:

- `rms_history` / `skewness_history`: the last `ROLLING_WINDOW` raw values, as fixed-size
  deques, which is all `rolling(10, min_periods=1).mean()` needs for the next file.
- `baseline_rms_values` -> `baseline_rms`: the first `BASELINE_N_FILES` raw RMS values,
  accumulated until 50 are in hand and then collapsed to their fixed mean, which is exactly
  `add_rolling_rms_ratio`'s `out["rms"].head(50).mean()`. The raw values are dropped once
  the mean is locked (`docs/serving_design.md` Section 2 says they can be).
- `file_count`: how many files this bearing has been served, which is both the cold-start
  test (Section 3) and the size of the accumulating baseline.

`ROLLING_WINDOW` and `BASELINE_N_FILES` are **imported** from `src.features.extraction`
rather than re-declared, for the same anti-drift reason `src/training/train_serving_model.py`
imports its model configuration: if the batch pipeline's window or baseline size ever
changes, serving cannot silently keep using the old one.

## Concurrency (`docs/serving_design.md` Section 2)

**Neither `BearingState` nor `BearingStateStore` is thread-safe, and both are
process-local by construction.** `observe()` is a read-modify-write over four mutable
fields, and `BearingStateStore.get_or_create` is a check-then-insert; neither is atomic.
Two concurrent requests for the same `bearing_id` can therefore interleave into a rolling
window that never existed. Two consequences the eventual API layer inherits, stated here
because they follow from *this* module's design rather than from that one's:

1. The server must run as a **single worker process** (e.g. `uvicorn` without
   `--workers N > 1`). Separate workers hold separate dicts, so a bearing's requests
   landing on different workers would each see a partial history -- and the responses
   would still look entirely normal (`docs/serving_design.md` Section 2).
2. Within that one process, a given `bearing_id`'s requests must be handled one at a
   time. `demo/playback.py` replaying one file after another satisfies this without any
   locking; nothing here enforces it. Adding a lock is deliberately left out rather than
   half-done -- it would suggest a level of concurrency safety the single-process,
   in-memory design does not otherwise provide (state durability across restarts and
   horizontal scaling are both explicit non-goals, Section 5).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from src.features.extraction import BASELINE_N_FILES, ROLLING_WINDOW

# The two values `baseline_status` can take (`docs/serving_design.md` Section 3). Named
# constants rather than bare strings so the API layer and the tests agree on the spelling.
WARMING_UP = "warming_up"
STABLE = "stable"


def window_mean(values: Iterable[float]) -> float:
    """Mean of the values currently in a rolling window (or of the accumulating baseline).

    Deliberately a plain left-to-right `sum(...) / n`, which is what `pandas`' own rolling
    mean accumulates in as well. It is *not* bit-for-bit identical to
    `Series.rolling(10, min_periods=1).mean()`, and cannot be made so without
    reimplementing a `pandas` internal: `pandas` carries one Kahan-compensated running sum
    across the whole column, adding the incoming value and removing the outgoing one, so
    its result at row `i` depends on the arithmetic history of every earlier row -- not
    just on the ten values currently in the window. Any window-local computation therefore
    differs from it in the last bit or two.

    Measured, not assumed: replaying all three experiments' raw files one at a time
    (9,464 files) through this module reproduces the batch pipeline's `skewness_smoothed`
    bit-for-bit on every row, and its `rms_ratio` bit-for-bit on 80.5-81.4% of post-warmup
    rows, **never differing by more than 2 ULP** on the rest.
    `tests/test_serving_features.py` pins that bound.
    Reimplementing `pandas`' compensated accumulator was considered and rejected: it would
    pin serving's hot path to an undocumented `pandas` internal that a version bump could
    change, in exchange for a difference far below anything `StandardScaler` and a
    three-class decision boundary can resolve.
    """
    values = list(values)
    return sum(values) / len(values)


@dataclass
class BearingState:
    """One bearing's rolling history -- everything `rms_ratio`/`skewness_smoothed` need.

    Call `observe()` once per incoming file, **before** reading any of the derived
    properties: the batch pipeline's rolling window at row `i` includes row `i` itself, so
    the current file must already be in the history when its own features are computed.
    """

    rolling_window: int = ROLLING_WINDOW
    baseline_n_files: int = BASELINE_N_FILES

    rms_history: deque[float] = field(init=False, repr=False)
    skewness_history: deque[float] = field(init=False, repr=False)
    baseline_rms_values: list[float] = field(init=False, default_factory=list, repr=False)
    baseline_rms: float | None = field(init=False, default=None)
    file_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.rms_history = deque(maxlen=self.rolling_window)
        self.skewness_history = deque(maxlen=self.rolling_window)

    def observe(self, rms: float, skewness: float) -> None:
        """Record one file's raw (stateless) RMS and skewness, advancing this bearing.

        The baseline locks on the file that completes `baseline_n_files` observations --
        the 50th file, index 49 -- at which point its value is the mean of files 0..49,
        the same 50 values `add_rolling_rms_ratio`'s `head(50).mean()` averages. Once
        locked it is never recomputed, so nothing a bearing does later can move it.
        """
        self.rms_history.append(rms)
        self.skewness_history.append(skewness)
        self.file_count += 1

        if self.baseline_rms is None:
            self.baseline_rms_values.append(rms)
            if len(self.baseline_rms_values) == self.baseline_n_files:
                self.baseline_rms = window_mean(self.baseline_rms_values)
                # Kept only until the mean exists (`docs/serving_design.md` Section 2).
                self.baseline_rms_values = []

    @property
    def baseline_status(self) -> str:
        """`"warming_up"` while the baseline is still expanding, `"stable"` afterwards.

        `docs/serving_design.md` Section 3's own formulation -- "it's just
        `file_count < 50`" -- with `file_count` including the file being answered. The
        50th file (index 49) is therefore the first `"stable"` response, matching that
        section's "once the 50th file is seen... `baseline_status` switches to `stable`".

        The alternative reading (the 50th file is still `"warming_up"`, because it is
        within "files 0-49") changes only the label, never a number: on that file the
        expanding baseline is the mean of files 0..49, which *is* the locked baseline.
        See the PR for Issue #82 for why this reading was chosen.
        """
        return WARMING_UP if self.file_count < self.baseline_n_files else STABLE

    @property
    def effective_baseline_rms(self) -> float:
        """The baseline `rms_ratio` is currently divided by.

        The locked 50-file mean once it exists; before that, the expanding mean of however
        many RMS values have been seen (1 to 49). Same "shrink the lookback rather than
        return nothing" convention `min_periods=1` already applies to the 10-file window
        (`docs/feature_windowing_decision.md` Section 3, `docs/serving_design.md`
        Section 3).
        """
        if self.baseline_rms is not None:
            return self.baseline_rms
        return window_mean(self.baseline_rms_values)

    @property
    def rolling_rms(self) -> float:
        """`rolling(rolling_window, min_periods=1).mean()` of `rms`, for the latest file.

        The deque holds at most `rolling_window` values and fewer than that until the
        window fills, so averaging its contents reproduces `min_periods=1`'s shrinking
        window over files 0..8 without a special case.
        """
        return window_mean(self.rms_history)

    @property
    def rolling_skewness(self) -> float:
        """`rolling(rolling_window, min_periods=1).mean()` of `skewness`, latest file."""
        return window_mean(self.skewness_history)


class BearingStateStore:
    """`bearing_id` -> `BearingState`, in memory, for one process.

    `docs/serving_design.md` Section 2's dictionary, with the creation-on-first-sight
    behaviour Section 1's contract implies: a bearing's first request carries no flag
    saying it is the first, so an unknown `bearing_id` simply starts a new history at file
    0. See this module's docstring for the concurrency constraints that follow.
    """

    def __init__(
        self,
        rolling_window: int = ROLLING_WINDOW,
        baseline_n_files: int = BASELINE_N_FILES,
    ) -> None:
        self.rolling_window = rolling_window
        self.baseline_n_files = baseline_n_files
        self._states: dict[str, BearingState] = {}

    def get_or_create(self, bearing_id: str) -> BearingState:
        """This bearing's state, starting a fresh one the first time it is seen."""
        if bearing_id not in self._states:
            self._states[bearing_id] = BearingState(
                rolling_window=self.rolling_window,
                baseline_n_files=self.baseline_n_files,
            )
        return self._states[bearing_id]

    def reset(self, bearing_id: str | None = None) -> None:
        """Forget one bearing's history, or all of them.

        A bearing that is reset restarts at file 0, which is the only correct way to
        replay it: `docs/serving_design.md` Section 1 has the server infer position from
        arrival order, so resuming mid-stream is not supported.
        """
        if bearing_id is None:
            self._states.clear()
        else:
            self._states.pop(bearing_id, None)

    def __contains__(self, bearing_id: object) -> bool:
        return bearing_id in self._states

    def __len__(self) -> int:
        return len(self._states)

    def __iter__(self) -> Iterator[str]:
        return iter(self._states)
