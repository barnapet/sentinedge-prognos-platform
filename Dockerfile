# Serving container for the /predict API (Issue #86).
#
# Runs the API exactly the way Issue #84 documented it: `python -m src.serving.main`, which
# hands uvicorn an already-built FastAPI object. That is not a stylistic choice -- passed an
# object rather than an import string, uvicorn cannot fork worker processes at all and exits
# rather than starting them, which is the first of the two layers enforcing
# docs/serving_design.md Section 2's single-worker constraint. The second layer, the OS lock
# in src/serving/single_worker.py, runs inside this container too. No gunicorn, no process
# manager, no --workers flag: any of those would violate the constraint the state module is
# built on (see docker-compose.yml for the scaling caveat that survives both layers).

FROM python:3.11-slim

# Match CI's interpreter (.github/workflows/notebook-ci.yml uses 3.11) so the container runs
# the same Python the test suite is verified against -- including the builtin-sum behaviour
# that differs between 3.11 and 3.12 and was pinned down in Issue #83.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first, as their own layer: they change far less often than the source, so
# rebuilds after a code edit reuse this layer instead of re-resolving scikit-learn.
COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

# Only what the request path actually needs. Notably absent: data/ (serving reads no
# dataset), notebooks/, tests/, and demo/ -- the playback client talks to this container
# over HTTP and is not part of it.
COPY src/ ./src/
COPY models/serving_model.joblib models/serving_model_manifest.json models/drift_baseline.json ./models/

EXPOSE 8000

# A plain healthcheck against the endpoint Issue #84 added for exactly this purpose; it
# reports model_loaded, so "ready" means "can actually score", not just "process is up".
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request,json,sys; \
sys.exit(0 if json.load(urllib.request.urlopen('http://localhost:8000/health'))['model_loaded'] else 1)"

CMD ["python", "-m", "src.serving.main"]
