from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from incident_memory_engine.config import EngineConfig


def class_means(cfg: EngineConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample class centroids in R^{feature_dim}."""
    raw = rng.standard_normal((cfg.num_classes, cfg.feature_dim)).astype(np.float32)
    return raw * 2.0


def era_drifts(cfg: EngineConfig, rng: np.random.Generator) -> np.ndarray:
    """Per-era mean shift (concept drift)."""
    drifts = [
        rng.standard_normal(cfg.feature_dim).astype(np.float32) * 0.35
        for _ in range(cfg.num_eras)
    ]
    return np.stack(drifts, axis=0)


def era_label_weights(era: int, cfg: EngineConfig) -> np.ndarray:
    """Dominant classes rotate with era; others remain possible (rare legacy)."""
    w = np.ones(cfg.num_classes, dtype=np.float64)
    focus = [era % cfg.num_classes, (era + 1) % cfg.num_classes]
    w[focus] = 8.0
    w /= w.sum()
    return w


def sample_era_batch(
    era: int,
    n: int,
    centers: np.ndarray,
    drifts: np.ndarray,
    cfg: EngineConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw n labeled incidents for the given era (synthetic stream)."""
    weights = era_label_weights(era, cfg)
    y = rng.choice(cfg.num_classes, size=n, p=weights).astype(np.int64)
    drift = drifts[era]
    x_list = []
    for label in y:
        mu = centers[label] + drift
        x_list.append(rng.standard_normal(cfg.feature_dim).astype(np.float32) + mu)
    x = np.stack(x_list, axis=0)
    return torch.from_numpy(x), torch.from_numpy(y)


def build_per_era_test_sets(
    cfg: EngineConfig,
    rng: np.random.Generator,
) -> tuple[list[TensorDataset], np.ndarray, np.ndarray]:
    """Fixed holdout TensorDatasets per era for evaluation."""
    centers = class_means(cfg, rng)
    drifts = era_drifts(cfg, rng)
    tests: list[TensorDataset] = []
    for e in range(cfg.num_eras):
        x, y = sample_era_batch(
            e, cfg.test_samples_per_era, centers, drifts, cfg, rng
        )
        tests.append(TensorDataset(x, y))
    return tests, centers, drifts


def canonical_incident_text(label: int, era: int, rng: np.random.Generator) -> str:
    """
    Deterministic-enough incident strings whose n-grams differ by class/era.

    Used when the model is trained on **encoded text** (hashing / tf-idf / sentence)
    so holdout evaluation lives in the same feature distribution as training.
    """

    salt = int(rng.integers(0, 1_000_000))
    return (
        f"incident ticket class_{label:02d} training_era_{era} rid_{salt:x} "
        f"severity_{label} component_alpha_{label} bravo_{era} zone_{label ^ era} "
        f"latency_tag_{label} avail_tag_{era} kv_{label * 1000 + era}"
    )


def canonical_text_labels_for_era(
    era: int,
    n: int,
    cfg: EngineConfig,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray]:
    weights = era_label_weights(era, cfg)
    y = rng.choice(cfg.num_classes, size=n, p=weights).astype(np.int64)
    texts = [canonical_incident_text(int(lab), era, rng) for lab in y]
    return texts, y


def build_per_era_text_eval_datasets(
    cfg: EngineConfig,
    rng: np.random.Generator,
    pipeline: object,
) -> list[TensorDataset]:
    """
    Per-era holdouts in **encoder output space** (matches text-trained models).

    TF-IDF is primed once with all holdout strings so the vectorizer is fitted
    before per-era transforms.
    """
    from incident_memory_engine.core.feature_pipeline import TfidfTextEncoder

    if isinstance(pipeline, TfidfTextEncoder):
        all_texts: list[str] = []
        for e in range(cfg.num_eras):
            texts, _ = canonical_text_labels_for_era(
                e, cfg.test_samples_per_era, cfg, rng
            )
            all_texts.extend(texts)
        pipeline.observe_training_texts(all_texts)

    tests: list[TensorDataset] = []
    for e in range(cfg.num_eras):
        texts, y = canonical_text_labels_for_era(
            e, cfg.test_samples_per_era, cfg, rng
        )
        x = pipeline.transform_batch(texts)
        tests.append(
            TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(y.astype(np.int64)),
            )
        )
    return tests


def split_github_samples_train_holdout(
    samples: list[dict],
    holdout_frac: float = 0.12,
    seed: int = 42,
) -> tuple[list[dict], dict[int, list[dict]]]:
    """
    Stratify by era: hold out a fraction for evaluation, rest for training.

    Ensures at least one training row per era when possible.
    """
    rng = np.random.default_rng(seed)
    by_era: dict[int, list[dict]] = {}
    for s in samples:
        by_era.setdefault(int(s["era"]), []).append(s)
    train: list[dict] = []
    holdout: dict[int, list[dict]] = {}
    for era in sorted(by_era.keys()):
        rows = list(by_era[era])
        rng.shuffle(rows)
        n = len(rows)
        if n <= 1:
            holdout[era] = []
            train.extend(rows)
            continue
        nh = max(1, min(int(round(n * holdout_frac)), n - 1))
        holdout[era] = rows[:nh]
        train.extend(rows[nh:])
    return train, holdout
