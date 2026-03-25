from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """Hyperparameters for the continual-learning engine and synthetic stream."""

    feature_dim: int = 64
    num_classes: int = 8
    num_eras: int = 5
    test_samples_per_era: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    hidden_dim: int = 128
    replay_capacity: int = 800
    replay_batch_ratio: float = 0.55
    seed: int = 42
    # Elastic Weight Consolidation (Kirkpatrick et al.): penalty toward anchor params
    # weighted by diagonal Fisher. Use with replay for a standard CL stack. 0 = off.
    ewc_lambda: float = 0.0
    ewc_fisher_batches: int = 8
    ewc_fisher_ema: float = 0.85
    forgetting_alert_low: float = 0.85
    forgetting_alert_medium: float = 0.70
    # Persistence
    state_path: str = "artifacts/engine_state.pkl"
    persistence_enabled: bool = True
    # Text / embeddings: identity | hashing | tfidf | sentence
    encoder_kind: str = "hashing"
    # Similarity ranking: recency weight (half-life seconds for exponential decay)
    similarity_recency_half_life_s: float = 86_400.0
    # Class-level legacy drop to flag an "affected" incident type
    forgetting_class_drop_ratio: float = 0.90
    # Era column drop vs historical peak to mark affected era
    forgetting_era_drop_ratio: float = 0.85
    # Human-readable class names (index = class id); padded with class_{id} if short
    incident_class_names: tuple[str, ...] = (
        "latency",
        "availability",
        "database",
        "security",
        "config",
        "capacity",
        "third_party",
        "unknown",
    )
    # Weighted replay: short_term, long_term, critical
    replay_tier_weight_short: float = 0.5
    replay_tier_weight_long: float = 0.3
    replay_tier_weight_critical: float = 0.2
    # FAISS: candidate pool size vs k for /similar rerank
    faiss_similarity_pool_multiplier: int = 12
