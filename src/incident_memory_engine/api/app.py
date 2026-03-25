from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from incident_memory_engine import __version__
from incident_memory_engine.api.deps import (
    extract_api_key,
    max_ingest_rows,
    max_upload_bytes,
    rate_limit_per_minute,
    require_api_key_if_configured,
)
from incident_memory_engine.api.responses import envelope_error, envelope_ok, new_request_id
from incident_memory_engine.api.schemas import (
    AssistExplainRequest,
    AssistExplainResponse,
    CloseEraRequest,
    CloseEraResponse,
    DriftResponse,
    ExperimentGitHubReplayRequest,
    GitHubDownloadRequest,
    GitHubIngestRequest,
    HealthResponse,
    IncidentVector,
    IngestBatchResponse,
    IngestTextRequest,
    IngestTextResponse,
    PredictInsightRequest,
    PredictInsightResponse,
    PredictRequest,
    PredictResponse,
    SimilarMatch,
    SimilarRequest,
    SimilarResponse,
    SimulationRequest,
    SyntheticBatchResponse,
    TrainRequest,
    TrainResponse,
)
from incident_memory_engine.core.engine import IncidentMemoryEngine
from incident_memory_engine.core.llm_assist import explain_incident, load_llm_config


def _rate_limit_key(request: Request) -> str:
    k = extract_api_key(request)
    if k:
        return f"key:{k}"
    return f"ip:{get_remote_address(request)}"


_rpm = rate_limit_per_minute()
limiter = Limiter(key_func=_rate_limit_key)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or new_request_id()
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "") or ""


def _server_path(relative: str | None, default: str) -> Path:
    """Resolve paths relative to the API process working directory."""
    raw = relative or default
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _incident_to_vector(
    engine: IncidentMemoryEngine,
    inc: IncidentVector,
) -> list[float]:
    if inc.text is not None and inc.text.strip():
        return engine.encode_texts([inc.text.strip()])[0]
    assert inc.features is not None
    return list(inc.features)


def _train_batch_vectors_and_meta(
    engine: IncidentMemoryEngine,
    body: TrainRequest,
) -> tuple[list[list[float]], list[int], list[str], list[str]]:
    incs = body.incidents
    uses_text = bool(incs[0].text and incs[0].text.strip())
    if any(bool(i.text and i.text.strip()) != uses_text for i in incs):
        raise HTTPException(
            status_code=400,
            detail="Mixed text and feature vectors in one batch is not supported",
        )
    if uses_text:
        feats = engine.encode_texts([i.text.strip() for i in incs])
    else:
        feats = [list(i.features) for i in incs if i.features is not None]
    labels = [int(i.label) for i in incs]
    itypes = [i.incident_type or f"class_{i.label}" for i in incs]
    fixes = [i.fix or "" for i in incs]
    return feats, labels, itypes, fixes


def get_engine() -> IncidentMemoryEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = IncidentMemoryEngine()
    return _ENGINE_INSTANCE


_ENGINE_INSTANCE: IncidentMemoryEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    yield


_hide_docs = __import__("os").environ.get("IME_HIDE_DOCS", "").lower() in (
    "1",
    "true",
    "yes",
)

app = FastAPI(
    title="Incident Memory Engine",
    description="Bounded-memory continual learning for streaming incidents",
    version=__version__,
    lifespan=lifespan,
    openapi_url=None if _hide_docs else "/openapi.json",
    docs_url=None if _hide_docs else "/docs",
    redoc_url=None if _hide_docs else "/redoc",
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_envelope(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content=envelope_error(
            "rate_limited",
            str(exc.detail),
            request_id=_rid(request),
            version=__version__,
        ),
    )


app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EngineDep = Annotated[IncidentMemoryEngine, Depends(get_engine)]


@app.get("/demo/synthetic-batch", response_model=SyntheticBatchResponse)
@limiter.limit(f"{_rpm}/minute")
def demo_synthetic_batch(
    request: Request,
    engine: EngineDep,
    era: int = Query(0, ge=0, description="Synthetic stream era"),
    n: int = Query(32, ge=1, le=512, description="Batch size"),
) -> SyntheticBatchResponse:
    require_api_key_if_configured(request)
    try:
        feats, labs = engine.sample_synthetic_batch(era, n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    incidents = [
        IncidentVector(features=feats[i], label=int(labs[i])) for i in range(len(labs))
    ]
    return SyntheticBatchResponse(era=era, incidents=incidents)


@app.get("/health", response_model=HealthResponse)
def health(engine: EngineDep) -> HealthResponse:
    cfg = engine.cfg
    return HealthResponse(
        status="ok",
        version=__version__,
        device=str(engine.device),
        buffer_size=len(engine.buffer) if engine.buffer else 0,
        feature_dim=cfg.feature_dim,
        num_classes=cfg.num_classes,
        highest_closed_era=engine.highest_closed_era,
        encoder_kind=cfg.encoder_kind,
        model_in_dim=engine.model_in_dim,
        state_path=cfg.state_path,
        loaded_from_disk=engine.loaded_from_disk,
        ewc_lambda=cfg.ewc_lambda,
        ewc_active=engine.ewc_is_active,
    )


@app.post("/train", response_model=TrainResponse)
@limiter.limit(f"{_rpm}/minute")
def train_endpoint(
    request: Request, body: TrainRequest, engine: EngineDep
) -> TrainResponse:
    require_api_key_if_configured(request)
    try:
        feats, labels, itypes, fixes = _train_batch_vectors_and_meta(engine, body)
        out = engine.train_batch(
            body.era,
            feats,
            labels,
            incident_types=itypes,
            fixes=fixes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return TrainResponse(
        loss=out["loss"],
        batch_size=out["batch_size"],
        era=body.era,
    )


@app.post("/era/close", response_model=CloseEraResponse)
@limiter.limit(f"{_rpm}/minute")
def close_era_endpoint(
    request: Request, body: CloseEraRequest, engine: EngineDep
) -> CloseEraResponse:
    require_api_key_if_configured(request)
    evals = engine.close_era(body.era)
    return CloseEraResponse(
        evaluations={str(k): float(v) for k, v in evals.items()},
    )


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(f"{_rpm}/minute")
def predict_endpoint(
    request: Request, body: PredictRequest, engine: EngineDep
) -> PredictResponse:
    require_api_key_if_configured(request)
    try:
        vec = _incident_to_vector(engine, body.incident)
        out = engine.predict_one(vec)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return PredictResponse(
        predicted_class=int(out["predicted_class"]),
        confidence=float(out["confidence"]),
    )


@app.post("/predict/insight")
@limiter.limit(f"{_rpm}/minute")
def predict_insight_endpoint(
    request: Request, body: PredictInsightRequest, engine: EngineDep
) -> dict:
    require_api_key_if_configured(request)
    try:
        vec = _incident_to_vector(engine, body.incident)
        filt = set(body.memory_tiers) if body.memory_tiers else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    text_hint = body.incident.text.strip() if body.incident.text else None
    raw = engine.predict_insight(
        vec,
        k_neighbors=body.k_neighbors,
        include_forgetting=body.include_forgetting,
        include_llm=body.include_llm,
        incident_text=text_hint,
        memory_tier_filter=filt,
    )
    parsed = PredictInsightResponse.model_validate(raw)
    return envelope_ok(
        parsed.model_dump(),
        request_id=_rid(request),
        version=__version__,
    )


@app.get("/metrics")
@limiter.limit(f"{_rpm}/minute")
def metrics_endpoint(request: Request, engine: EngineDep) -> dict:
    require_api_key_if_configured(request)
    return engine.metrics_payload()


@app.get("/forgetting-alert")
@limiter.limit(f"{_rpm}/minute")
def forgetting_alert_endpoint(request: Request, engine: EngineDep) -> dict:
    require_api_key_if_configured(request)
    return engine.forgetting_alert_payload()


@app.post("/similar", response_model=SimilarResponse)
@limiter.limit(f"{_rpm}/minute")
def similar_endpoint(
    request: Request, body: SimilarRequest, engine: EngineDep
) -> SimilarResponse:
    require_api_key_if_configured(request)
    try:
        vec = _incident_to_vector(engine, body.incident)
        tier_f = set(body.memory_tiers) if body.memory_tiers else None
        raw = engine.similar_incidents(
            vec, k=body.k, memory_tier_filter=tier_f
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    matches = [
        SimilarMatch(
            label=int(m["label"]),
            era=int(m["era"]),
            distance=float(m["distance"]),
            similarity_score=float(m["similarity_score"]),
            rank_score=float(m["rank_score"]),
            incident_id=m.get("incident_id") if m.get("incident_id") else None,
            incident_type=str(m.get("incident_type") or ""),
            fix=str(m.get("fix") or ""),
            timestamp=float(m.get("timestamp") or 0.0),
            features_preview=[float(x) for x in (m.get("features_preview") or [])],
            memory_tier=str(m.get("memory_tier") or "short_term"),
        )
        for m in raw
    ]
    return SimilarResponse(matches=matches)


@app.post("/simulation/run")
@limiter.limit(f"{_rpm}/minute")
def simulation_run(
    request: Request, body: SimulationRequest, engine: EngineDep
) -> dict:
    require_api_key_if_configured(request)
    payload = engine.run_synthetic_era_simulation(
        steps_per_era=body.steps_per_era,
        num_eras=body.num_eras,
    )
    if body.persist_path:
        path = Path(body.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@app.post("/engine/reset")
@limiter.limit(f"{_rpm}/minute")
def reset_engine(request: Request, engine: EngineDep) -> dict:
    require_api_key_if_configured(request)
    engine.reset()
    return {"status": "reset"}


@app.post("/ingest/text")
@limiter.limit(f"{_rpm}/minute")
def ingest_text(request: Request, body: IngestTextRequest, engine: EngineDep) -> dict:
    require_api_key_if_configured(request)
    texts = [t.strip() for t in body.texts if t and t.strip()]
    if not texts:
        raise HTTPException(
            status_code=400,
            detail=envelope_error(
                "validation_error",
                "No non-empty texts",
                request_id=_rid(request),
                version=__version__,
            ),
        )
    embeddings = engine.encode_texts(texts)
    loss = None
    trained = False
    n = len(texts)
    tier = (body.memory_tier or "short_term").strip()
    if tier not in ("short_term", "long_term", "critical"):
        tier = "short_term"
    if body.train:
        assert body.labels is not None
        labels = [int(x) for x in body.labels]
        ditype = body.default_incident_type or ""
        dfix = body.default_fix or ""
        ptypes = body.per_incident_types or []
        pfixes = body.per_fixes or []
        itypes = [
            (ptypes[i] if i < len(ptypes) and ptypes[i] else ditype)
            or f"class_{labels[i]}"
            for i in range(n)
        ]
        fixes = [
            (pfixes[i] if i < len(pfixes) and pfixes[i] is not None else dfix)
            for i in range(n)
        ]
        out = engine.train_batch(
            body.era,
            embeddings,
            labels,
            incident_types=itypes,
            fixes=fixes,
            memory_tiers=[tier] * n,
        )
        loss = float(out["loss"])
        trained = True
    data = IngestTextResponse(
        count=n,
        era=body.era,
        trained=trained,
        loss=loss,
        embeddings=embeddings if body.return_embeddings else None,
    )
    return envelope_ok(
        data.model_dump(),
        request_id=_rid(request),
        version=__version__,
    )


@app.post("/ingest/batch")
@limiter.limit(f"{_rpm}/minute")
async def ingest_batch(
    request: Request,
    engine: EngineDep,
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    label_column: str = Form("label"),
    era: int = Form(0, ge=0),
    default_incident_type: str = Form(""),
    default_fix: str = Form(""),
    memory_tier: str = Form("short_term"),
    chunk_size: int = Form(128, ge=8, le=2048),
    close_era_at_end: str = Form("false"),
) -> dict:
    require_api_key_if_configured(request)
    raw = await file.read()
    if len(raw) > max_upload_bytes():
        raise HTTPException(
            status_code=413,
            detail=envelope_error(
                "payload_too_large",
                "File exceeds IME_MAX_UPLOAD_BYTES",
                request_id=_rid(request),
                version=__version__,
            ),
        )
    name = (file.filename or "").lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
        elif name.endswith(".json"):
            obj = json.loads(raw.decode("utf-8"))
            if isinstance(obj, list):
                df = pd.DataFrame(obj)
            elif isinstance(obj, dict) and "rows" in obj:
                df = pd.DataFrame(obj["rows"])
            else:
                df = pd.DataFrame(obj)
        else:
            raise ValueError("Upload a .csv or .json file")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=envelope_error(
                "parse_error",
                str(e),
                request_id=_rid(request),
                version=__version__,
            ),
        ) from e
    if text_column not in df.columns or label_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=envelope_error(
                "validation_error",
                f"Columns missing: need {text_column!r} and {label_column!r}",
                request_id=_rid(request),
                version=__version__,
            ),
        )
    df = df[[text_column, label_column]].dropna()
    max_rows = max_ingest_rows()
    if len(df) > max_rows:
        df = df.iloc[:max_rows]
    tier = memory_tier.strip()
    if tier not in ("short_term", "long_term", "critical"):
        tier = "short_term"
    loss_last: float | None = None
    chunks = 0
    for start in range(0, len(df), chunk_size):
        sub = df.iloc[start : start + chunk_size]
        texts = [str(x).strip() for x in sub[text_column].tolist()]
        labels = [int(float(x)) for x in sub[label_column].tolist()]
        feats = engine.encode_texts(texts)
        n = len(labels)
        itypes = [
            (default_incident_type or f"class_{labels[i]}") for i in range(n)
        ]
        fixes = [default_fix for _ in range(n)]
        out = engine.train_batch(
            era,
            feats,
            labels,
            incident_types=itypes,
            fixes=fixes,
            memory_tiers=[tier] * n,
        )
        loss_last = float(out["loss"])
        chunks += 1
    close_e: int | None = None
    if str(close_era_at_end).lower() in ("1", "true", "yes"):
        engine.close_era(era)
        close_e = era
    resp = IngestBatchResponse(
        rows_processed=int(len(df)),
        chunks_trained=chunks,
        era=era,
        close_era=close_e,
        loss_last=loss_last,
    )
    return envelope_ok(
        resp.model_dump(),
        request_id=_rid(request),
        version=__version__,
    )


@app.post("/assist/explain")
@limiter.limit(f"{_rpm}/minute")
def assist_explain(request: Request, body: AssistExplainRequest) -> dict:
    require_api_key_if_configured(request)
    summ, hyp, sug, prov = explain_incident(
        incident_text=body.text,
        predicted_class=body.predicted_class,
        confidence=body.confidence,
        similar_lines=body.similar_summaries,
        cfg=load_llm_config(),
    )
    out = AssistExplainResponse(
        summary=summ,
        hypothesis=hyp,
        suggested_fix=sug,
        provider=prov,
    )
    return envelope_ok(
        out.model_dump(),
        request_id=_rid(request),
        version=__version__,
    )


@app.post("/ingest/github")
@limiter.limit(f"{_rpm}/minute")
def ingest_github_endpoint(
    request: Request, body: GitHubIngestRequest, engine: EngineDep
) -> dict:
    require_api_key_if_configured(request)
    try:
        out = engine.ingest_github_issues(
            body.repos,
            body.era,
            body.per_repo,
            token=os.environ.get("GITHUB_TOKEN"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"GitHub ingest failed: {e}",
        ) from e
    return envelope_ok(
        out,
        request_id=_rid(request),
        version=__version__,
    )


@app.post("/data/github/download")
@limiter.limit(f"{_rpm}/minute")
def github_download_endpoint(
    request: Request, body: GitHubDownloadRequest
) -> dict:
    """Download k8s / prometheus / grafana issues JSON to the server disk."""
    require_api_key_if_configured(request)
    from incident_memory_engine.data.github_ingest import (
        DEFAULT_ERA_REPOS,
        ingest_eras,
        save_artifacts,
    )

    out_path = _server_path(body.output_path, "artifacts/github_issues.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows, meta = ingest_eras(
            DEFAULT_ERA_REPOS,
            body.per_era,
            token=os.environ.get("GITHUB_TOKEN"),
        )
        save_artifacts(rows, meta, out_path)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=envelope_error(
                "github_download_failed",
                str(e),
                request_id=_rid(request),
                version=__version__,
            ),
        ) from e
    return envelope_ok(
        {
            "output_path": str(out_path),
            "sample_count": len(rows),
            "repos_by_era": meta.get("era_repos", {}),
        },
        request_id=_rid(request),
        version=__version__,
    )


@app.post("/experiment/github-replay")
@limiter.limit(f"{_rpm}/minute")
def experiment_github_replay_endpoint(
    request: Request,
    body: ExperimentGitHubReplayRequest,
    engine: EngineDep,
) -> dict:
    """
    Full train + era-close pass over ``artifacts/github_issues.json`` (or given path).

    Mutates the live engine so ``/metrics`` reflects the experiment immediately.
    """
    require_api_key_if_configured(request)
    data_p = _server_path(body.data_path, "artifacts/github_issues.json")
    try:
        result = engine.run_github_file_experiment(
            data_p,
            chunk_size=body.chunk_size,
            reset_first=body.reset_engine_first,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=envelope_error(
                "file_not_found",
                f"No GitHub JSON at {data_p}",
                request_id=_rid(request),
                version=__version__,
            ),
        ) from None
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=envelope_error(
                "experiment_failed",
                str(e),
                request_id=_rid(request),
                version=__version__,
            ),
        ) from e
    return envelope_ok(
        result,
        request_id=_rid(request),
        version=__version__,
    )


@app.get("/data/status")
@limiter.limit(f"{_rpm}/minute")
def data_status_endpoint(request: Request, engine: EngineDep) -> dict:
    require_api_key_if_configured(request)
    return engine.data_status_payload()


@app.get("/drift")
@limiter.limit(f"{_rpm}/minute")
def drift_endpoint(request: Request, engine: EngineDep) -> dict:
    require_api_key_if_configured(request)
    snap = engine.drift_snapshot()
    auto = engine.maybe_auto_close_era_on_drift(snap)
    body = DriftResponse(
        score=float(snap["score"]),
        window_samples=int(snap["window_samples"]),
        reference_mean_norm=snap.get("reference_mean_norm"),
        current_mean_norm=snap.get("current_mean_norm"),
        recommendation=str(snap["recommendation"]),
        auto_close_triggered=auto,
    )
    return envelope_ok(
        body.model_dump(),
        request_id=_rid(request),
        version=__version__,
    )
