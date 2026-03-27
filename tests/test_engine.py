from __future__ import annotations

from incident_memory_engine.config import EngineConfig
from incident_memory_engine.core.engine import IncidentMemoryEngine


def test_sample_synthetic_batch_shape() -> None:
    eng = IncidentMemoryEngine(
        EngineConfig(seed=1, num_eras=3, persistence_enabled=False)
    )
    feats, labs = eng.sample_synthetic_batch(era=1, n=8)
    assert len(feats) == 8
    assert len(labs) == 8
    assert len(feats[0]) == eng.cfg.feature_dim


def test_run_synthetic_simulation_produces_matrix() -> None:
    eng = IncidentMemoryEngine(
        EngineConfig(
            seed=2,
            num_eras=3,
            batch_size=16,
            test_samples_per_era=32,
            persistence_enabled=False,
        )
    )
    out = eng.run_synthetic_era_simulation(steps_per_era=2, num_eras=3)
    assert len(out["metrics"]["accuracy_matrix"]) == 3


def test_replay_disabled_leaves_buffer_empty() -> None:
    eng = IncidentMemoryEngine(
        EngineConfig(
            seed=0,
            num_eras=3,
            persistence_enabled=False,
            replay_enabled=False,
            batch_size=8,
        )
    )
    f, y = eng.sample_synthetic_batch(0, 4)
    eng.train_batch(0, f, y)
    assert len(eng.buffer) == 0


def test_ewc_synthetic_simulation_runs() -> None:
    """EWC + replay: Fisher/anchor update after era close; training stays stable."""
    eng = IncidentMemoryEngine(
        EngineConfig(
            seed=3,
            num_eras=3,
            batch_size=16,
            test_samples_per_era=24,
            persistence_enabled=False,
            ewc_lambda=25.0,
            ewc_fisher_batches=4,
        )
    )
    out = eng.run_synthetic_era_simulation(steps_per_era=2, num_eras=3)
    assert len(out["metrics"]["accuracy_matrix"]) == 3
    m = out["metrics"]
    assert m["continual_learning"]["ewc_lambda"] == 25.0
    assert m["continual_learning"]["ewc_consolidation_ready"] is True
