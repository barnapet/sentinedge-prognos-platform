"""The `/predict` API layer (Issue #84) -- FastAPI, per `docs/PRD.md` Section 8/9's
proposed architecture ("Serving: FastAPI service in a container").

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

`create_app()` is a factory rather than a bare module-level `app`, so tests can point each
app instance at its own single-worker lock path (a shared real path would make tests
interfere with each other, and with a developer's already-running server) without needing
to touch this module's internals.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.serving.features import OnlineFeatureExtractor
from src.serving.model_notes import MODEL_NOTES
from src.serving.single_worker import (
    DEFAULT_LOCK_PATH,
    acquire_single_worker_lock,
    release_single_worker_lock,
)
from src.training.train_serving_model import MODEL_PATH, load_serving_model


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
    """`{label, baseline_status, model_notes}`, exactly the three fields the issue's
    context section names -- nothing computed here beyond what `OnlineFeatures` and the
    model already produce.
    """

    label: str
    baseline_status: str
    model_notes: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def create_app(
    model_path: Path = MODEL_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> FastAPI:
    """Build one `/predict` + `/health` app, holding its own model, state store, and lock.

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
        """Minimal liveness/readiness signal -- not a monitoring endpoint (that's
        `/metrics`, out of scope for this issue). Useful as-is for a future container
        `HEALTHCHECK` (Issue #84's Task 4): a plain `GET /health` that 200s only once the
        model is actually loaded and ready to score.
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

        return PredictResponse(
            label=label,
            baseline_status=features.baseline_status,
            model_notes=MODEL_NOTES,
        )

    return app
