from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    feature_dim: int = 64
    num_classes: int = 8
    num_eras: int = 5
    train_samples_per_era: int = 800
    test_samples_per_era: int = 200
    batch_size: int = 64
    steps_per_era: int = 120
    lr: float = 1e-3
    hidden_dim: int = 128
    replay_capacity: int = 500
    replay_batch_ratio: float = 0.5
    seed: int = 42
