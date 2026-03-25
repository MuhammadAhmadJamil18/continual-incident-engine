from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .config import ExperimentConfig


def _class_means(cfg: ExperimentConfig, rng: np.random.Generator) -> np.ndarray:
    """Centers in R^{feature_dim} per class; separated enough to be learnable."""
    raw = rng.standard_normal((cfg.num_classes, cfg.feature_dim)).astype(np.float32)
    # Spread classes apart
    return raw * 2.0


def _era_drift(cfg: ExperimentConfig, rng: np.random.Generator) -> np.ndarray:
    """Small shift applied to all classes in era e (non-stationarity)."""
    drifts = []
    for _ in range(cfg.num_eras):
        drifts.append(rng.standard_normal(cfg.feature_dim).astype(np.float32) * 0.35)
    return np.stack(drifts, axis=0)


def era_label_weights(era: int, cfg: ExperimentConfig) -> np.ndarray:
    """
    Dominant incident families change by era; older types still appear rarely
    (realistic ops stream — rare legacy incidents).
    """
    w = np.ones(cfg.num_classes, dtype=np.float64)
    focus = [era % cfg.num_classes, (era + 1) % cfg.num_classes]
    w[focus] = 8.0
    w /= w.sum()
    return w


def sample_era_batch(
    era: int,
    n: int,
    class_means: np.ndarray,
    era_drifts: np.ndarray,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = era_label_weights(era, cfg)
    y = rng.choice(cfg.num_classes, size=n, p=weights).astype(np.int64)
    drift = era_drifts[era]
    x_list = []
    for label in y:
        mu = class_means[label] + drift
        x_list.append(rng.standard_normal(cfg.feature_dim).astype(np.float32) + mu)
    x = np.stack(x_list, axis=0)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    return x_t, y_t


def build_per_era_test_sets(
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[list[TensorDataset], np.ndarray, np.ndarray]:
    class_means = _class_means(cfg, rng)
    era_drifts = _era_drift(cfg, rng)
    tests: list[TensorDataset] = []
    for e in range(cfg.num_eras):
        x, y = sample_era_batch(
            e, cfg.test_samples_per_era, class_means, era_drifts, cfg, rng
        )
        tests.append(TensorDataset(x, y))
    return tests, class_means, era_drifts
