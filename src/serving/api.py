"""The `/predict` API layer (Issue #84) -- FastAPI, per `docs/PRD.md` Section 8/9's
proposed architecture ("Serving: FastAPI service in a container"). Monitoring endpoints
added in Issue #90.

This module assembles pieces that already exist and are individually tested, and adds no
new computation of its own:

- `models/serving_model.joblib` (Issue #80), loaded once at startup via
  `train_serving_model.load_serving_model` -- the same loader the training module itself
  exposes, not a second `joblib.load` call.
- `src.serving.features.OnlineFeatureExtractor` (Issue #82) for the state update and the
  5-feature vector, numerically proven equivalent to the batch pipeline.
- `src.serving.model_notes.MODEL_NOTES`, `docs/serving_design.md` Section 4's disclosure
  text, attached to every response unconditionally -- Section 4 already rejected making it
  conditional on the incoming signal, so there is no branch here that could skip it.
- `src.serving.single_worker`, Section 2's single-process constraint (see that module's
  docstring and `src/serving/main.py` for the two independent layers enforcing it).
- `src.serving.drift.MONITORED_FEATURES` (Issue #90), for `GET /monitoring/drift`'s
  per-feature breakdown -- see `docs/monitoring_design.md` Section 3 for the shape and why
  a cross-bearing read needs its own endpoint rather than a `/predict` response field.

`create_app()` is a factory rather than a bare module-level `app`, so tests can point each
app instance at its own single-worker lock path (a shared real path would make tests
interfere with each other, and with a developer's already-running server) without needing
to touch this module's internals.
"""
from __future__ import annotations

import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.serving.drift import MONITORED_FEATURES
from src.serving.features import OnlineFeatureExtractor
from src.serving.state import TRAJECTORY_CHANNELS, TRAJECTORY_HISTORY
from src.serving.model_notes import MODEL_NOTES
from src.serving.single_worker import (
    DEFAULT_LOCK_PATH,
    acquire_single_worker_lock,
    release_single_worker_lock,
)
from src.training.train_serving_model import MODEL_PATH, load_serving_model

# docs/monitoring_design.md Section 4: one static HTML file, served directly by this app --
# no template engine, no build step, no second service in docker-compose.yml.
STATIC_DIR = Path(__file__).parent / "static"
MONITORING_PAGE_PATH = STATIC_DIR / "monitoring.html"


class PredictRequest(BaseModel):
    """`docs/serving_design.md` Section 1's payload: a raw signal plus which bearing it's
    from. No file index or sequence number -- the server infers position from arrival
    order (Section 1), which is exactly what calling `OnlineFeatureExtractor.observe` once
    per request, in request order, does.
    """

    bearing_id: Annotated[str, Field(min_length=1)]
    signal: Annotated[list[float], Field(min_length=1)]
    # Logging/display only, per Section 1 -- never read by the feature-computation path.
    timestamp: str | None = None


class PredictResponse(BaseModel):
    """`{label, baseline_status, model_notes}` plus `drift_status` (Issue #90) -- an
    additive field, per `docs/monitoring_design.md` Section 3: the three original fields
    are unchanged.
    """

    label: str
    baseline_status: str
    model_notes: str
    drift_status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def create_app(
    model_path: Path = MODEL_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> FastAPI:
    """Build one `/predict` + `/health` + monitoring app, holding its own model, state
    store, and lock.

    Args:
        model_path: Which persisted pipeline to load (Issue #80's artifact by default).
        lock_path: Where the single-worker guard takes its exclusive lock. Overridable so
            tests (and, in principle, multiple *independent* deployments on one host) don't
            collide on the same real path.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        lock_file = acquire_single_worker_lock(lock_path)
        try:
            app.state.model = load_serving_model(model_path)
            app.state.extractor = OnlineFeatureExtractor()
            yield
        finally:
            release_single_worker_lock(lock_file)

    app = FastAPI(title="SentinEdge Prognos Serving API", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Minimal liveness/readiness signal -- not `/monitoring/drift`. Useful as-is for
        a future container `HEALTHCHECK` (Issue #84's Task 4): a plain `GET /health` that
        200s only once the model is actually loaded and ready to score.
        """
        model_loaded = getattr(request.app.state, "model", None) is not None
        return HealthResponse(status="ok" if model_loaded else "starting", model_loaded=model_loaded)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        extractor: OnlineFeatureExtractor = request.app.state.extractor
        model = request.app.state.model

        signal = np.asarray(payload.signal, dtype=np.float64)
        if not np.all(np.isfinite(signal)):
            raise HTTPException(422, "signal must contain only finite values")

        features = extractor.observe(payload.bearing_id, signal)
        label = str(model.predict([features.feature_vector()])[0])
        extractor.record_prediction(payload.bearing_id, label)

        return PredictResponse(
            label=label,
            baseline_status=features.baseline_status,
            model_notes=MODEL_NOTES,
            drift_status=features.drift_status,
        )

    @app.get("/monitoring/drift")
    def monitoring_drift(request: Request) -> dict:
        """`docs/monitoring_design.md` Section 3's cross-bearing read: a plain view of
        `BearingStateStore`'s current contents, for every `bearing_id` this process has
        seen at least one request for. No computation of its own beyond what `/predict`
        already wrote into that state -- this endpoint never advances a bearing's history.
        """
        extractor: OnlineFeatureExtractor = request.app.state.extractor
        bearings = {}
        for bearing_id in extractor.store:
            state = extractor.store.get_or_create(bearing_id)
            bearings[bearing_id] = {
                "file_count": state.file_count,
                "baseline_status": state.baseline_status,
                "drift_status": state.drift_status,
                "features": {
                    feature: state.feature_drift(feature) for feature in MONITORED_FEATURES
                },
                "rms_ratio_latest": state.rolling_rms / state.effective_baseline_rms,
                "predicted_class_counts": state.predicted_class_counts,
            }
        return {"bearings": bearings}

    @app.get("/monitoring/history/{bearing_id}")
    def monitoring_history(bearing_id: str, request: Request, window: int = TRAJECTORY_HISTORY):
        """One bearing's recent `TRAJECTORY_CHANNELS` trajectory (Issue #140).

        `docs/agent_design.md` Section 12's `find_similar_historical_pattern` compares "the
        last 50 requests" of a live bearing against the committed trajectory archive, and
        this is where those 50 come from -- read over HTTP by
        `src/agent/mcp/serving_client.py`, exactly as every other agent-side read is, so
        nothing in `src/agent/` imports this process's state (Section 2's constraint).

        Read-only in the same sense `/monitoring/drift` is: it returns what `/predict`
        already wrote and never advances a bearing.

        An unknown `bearing_id` is a **200 with `found: false`**, not a 404, matching
        `tools.get_bearing_status`'s structured not-found. A 404 would reach the agent as
        `ServingRejected` -- "the service refused this request" -- when the truthful answer
        is "nobody is tracking that bearing, and here is who is". Section 10 case 1 is
        precisely the failure of inventing a state for an untracked bearing, and a caller
        cannot avoid inventing one if the tool layer cannot tell those two cases apart.
        """
        if window <= 0:
            raise HTTPException(422, "window must be a positive integer")
        extractor: OnlineFeatureExtractor = request.app.state.extractor
        if bearing_id not in extractor.store:
            return {
                "bearing_id": bearing_id,
                "found": False,
                "tracked_bearings": sorted(extractor.store),
            }

        state = extractor.store.get_or_create(bearing_id)
        # Non-finite values are emitted as `null`, not as bare `NaN`. A degenerate signal
        # (a constant window makes kurtosis and skewness 0/0) leaves `NaN` in this
        # bearing's history, and `NaN` is not valid JSON -- serializing it raised here and
        # left that bearing's history endpoint permanently 500ing, which is a worse failure
        # than the one that caused it. `null` is readable, and the agent-side tool refuses
        # to build a query out of one rather than computing a distance from a hole.
        channels = {
            channel: [value if math.isfinite(value) else None for value in values]
            for channel, values in state.trajectory(window=window).items()
        }
        return {
            "bearing_id": bearing_id,
            "found": True,
            "file_count": state.file_count,
            "baseline_status": state.baseline_status,
            "retained": TRAJECTORY_HISTORY,
            "channels": channels,
            "n_points": len(channels[TRAJECTORY_CHANNELS[0]]),
        }

    @app.get("/monitoring", response_class=FileResponse)
    def monitoring_page() -> FileResponse:
        """`docs/monitoring_design.md` Section 4's static dashboard -- one HTML file, no
        framework, served by this same app and port so `docker compose up` needs no
        second service."""
        return FileResponse(MONITORING_PAGE_PATH)

    return app
