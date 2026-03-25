from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class IncidentVector(BaseModel):
    """Incident as precomputed features **or** raw text (exactly one required)."""

    features: list[float] | None = None
    text: str | None = Field(None, description="Raw log / incident text to embed")
    label: int | None = Field(
        default=None,
        description="Training label; omit for inference-only payloads",
    )
    incident_type: str | None = Field(
        None,
        description="Human-readable incident family (stored in replay metadata)",
    )
    fix: str | None = Field(
        None,
        description="Associated remediation / runbook snippet for similarity results",
    )

    @model_validator(mode="after")
    def exactly_one_input(self) -> IncidentVector:
        has_f = self.features is not None and len(self.features) > 0
        has_t = self.text is not None and len(self.text.strip()) > 0
        if has_f == has_t:
            raise ValueError(
                "Provide exactly one of: non-empty features, or non-empty text"
            )
        return self


class TrainRequest(BaseModel):
    """Incremental training step on a batch of labeled incidents."""

    era: int = Field(..., ge=0, description="Stream era id")
    incidents: list[IncidentVector] = Field(
        ...,
        min_length=1,
        description="Batch of labeled incidents",
    )

    @model_validator(mode="after")
    def labels_required(self) -> TrainRequest:
        for inc in self.incidents:
            if inc.label is None:
                raise ValueError("Each training incident must include label")
        return self


class TrainResponse(BaseModel):
    loss: float
    batch_size: int
    era: int


class CloseEraRequest(BaseModel):
    """Signal end of an era to run holdout evaluation."""

    era: int = Field(..., ge=0)


class CloseEraResponse(BaseModel):
    evaluations: dict[str, float]


class PredictRequest(BaseModel):
    incident: IncidentVector


class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float


class SimilarRequest(BaseModel):
    incident: IncidentVector
    k: int = Field(5, ge=1, le=100)
    memory_tiers: list[str] | None = Field(
        None,
        description="If set, only neighbors with these memory_tier values",
    )


class SimilarMatch(BaseModel):
    label: int
    era: int
    distance: float
    similarity_score: float
    rank_score: float
    incident_id: str | None = None
    incident_type: str = ""
    fix: str = ""
    timestamp: float = 0.0
    features_preview: list[float] = Field(default_factory=list)
    memory_tier: str = "short_term"


class SimilarResponse(BaseModel):
    matches: list[SimilarMatch]


class SimulationRequest(BaseModel):
    steps_per_era: int = Field(80, ge=1, le=2000)
    num_eras: int | None = Field(
        None,
        ge=1,
        le=50,
        description="Defaults to engine config when omitted",
    )
    persist_path: str | None = Field(
        None,
        description="Optional path to write metrics JSON after run",
    )


class SyntheticBatchResponse(BaseModel):
    """Labeled batch drawn from the server-side synthetic era stream."""

    era: int
    incidents: list[IncidentVector]


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    buffer_size: int
    feature_dim: int
    num_classes: int
    highest_closed_era: int
    encoder_kind: str
    model_in_dim: int
    state_path: str
    loaded_from_disk: bool
    ewc_lambda: float = Field(
        0.0,
        description="EWC regularization strength (0 = disabled).",
    )
    ewc_active: bool = Field(
        False,
        description="True after at least one era close while EWC is enabled.",
    )


class IngestTextRequest(BaseModel):
    """Batch text ingest for embeddings and optional training."""

    texts: list[str] = Field(..., min_length=1)
    era: int = Field(0, ge=0)
    labels: list[int] | None = Field(
        None,
        description="If set, must align with texts; required when train=true",
    )
    default_incident_type: str | None = None
    default_fix: str | None = None
    per_incident_types: list[str] | None = None
    per_fixes: list[str] | None = None
    train: bool = Field(False, description="Run train_batch on encoded vectors")
    return_embeddings: bool = Field(True)
    memory_tier: str | None = Field(
        None,
        description="short_term | long_term | critical (applied to all rows)",
    )

    @model_validator(mode="after")
    def labels_when_train(self) -> IngestTextRequest:
        if self.train:
            if not self.labels or len(self.labels) != len(self.texts):
                raise ValueError("train=true requires labels aligned with texts")
        return self


class IngestTextResponse(BaseModel):
    count: int
    era: int
    trained: bool
    loss: float | None = None
    embeddings: list[list[float]] | None = None


class IngestBatchResponse(BaseModel):
    rows_processed: int
    chunks_trained: int
    era: int
    close_era: int | None = None
    loss_last: float | None = None


class AssistExplainRequest(BaseModel):
    text: str = Field(..., min_length=1)
    predicted_class: int | None = None
    confidence: float | None = None
    similar_summaries: list[str] = Field(default_factory=list)


class AssistExplainResponse(BaseModel):
    summary: str | None = None
    hypothesis: str | None = None
    suggested_fix: str | None = None
    provider: str = "none"


class DriftResponse(BaseModel):
    score: float
    window_samples: int
    reference_mean_norm: float | None
    current_mean_norm: float | None
    recommendation: str
    auto_close_triggered: bool = False


class PredictionInsight(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class NeighborAttribution(BaseModel):
    incident_id: str | None = None
    label: int
    era: int
    distance: float
    similarity_score: float
    incident_type: str = ""
    fix: str = ""
    weight: float = 0.0
    memory_tier: str = "short_term"


class PredictInsightResponse(BaseModel):
    prediction: PredictionInsight
    similar_incidents: list[NeighborAttribution]
    neighbor_vote: dict[str, int] = Field(default_factory=dict)
    suggested_fix: str = ""
    forgetting_warning: dict | None = None
    explainability: dict = Field(
        default_factory=dict,
        description="Attribution: neighbor weights sum to ~1 for top-k",
    )
    llm: AssistExplainResponse | None = None


class PredictInsightRequest(BaseModel):
    incident: IncidentVector
    k_neighbors: int = Field(5, ge=1, le=50)
    include_forgetting: bool = True
    include_llm: bool = False
    memory_tiers: list[str] | None = Field(
        None,
        description="If set, restrict neighbor search to these memory tiers",
    )


class GitHubIngestRequest(BaseModel):
    """Trigger live GitHub issue fetch + encode + train for the given era."""

    repos: list[str] = Field(..., min_length=1, description="owner/name repos")
    era: int = Field(..., ge=0)
    per_repo: int = Field(300, ge=1, le=500)


class GitHubDownloadRequest(BaseModel):
    """Fetch default era repos from GitHub Search API and save JSON on the server."""

    per_era: int = Field(300, ge=50, le=500)
    output_path: str | None = Field(
        None,
        description="Relative to server working directory; default artifacts/github_issues.json",
    )


class ExperimentGitHubReplayRequest(BaseModel):
    """Replay a saved GitHub JSON through train + era-close on the live engine."""

    reset_engine_first: bool = True
    chunk_size: int = Field(64, ge=8, le=256)
    data_path: str | None = Field(
        None,
        description="Relative to server CWD; default artifacts/github_issues.json",
    )
