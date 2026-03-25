"""Tests for split JSON + torch + buffer checkpoint layout."""

from __future__ import annotations

from pathlib import Path

from incident_memory_engine.config import EngineConfig
from incident_memory_engine.core.engine import IncidentMemoryEngine
from incident_memory_engine.core.persistence import (
    delete_checkpoint,
    resolve_artifact_paths,
)


def test_split_checkpoint_roundtrip(tmp_path: Path) -> None:
    state = tmp_path / "engine_state.pkl"
    cfg = EngineConfig(
        persistence_enabled=True,
        state_path=str(state),
        encoder_kind="hashing",
        replay_capacity=30,
        batch_size=8,
        test_samples_per_era=24,
    )
    eng = IncidentMemoryEngine(cfg)
    feats, labs = eng.sample_synthetic_batch(0, 6)
    eng.train_batch(0, feats, labs)
    del eng

    meta_p, w_p, b_p = resolve_artifact_paths(state)
    assert meta_p.is_file()
    assert w_p.is_file()
    assert b_p.is_file()

    eng2 = IncidentMemoryEngine(cfg)
    assert eng2.loaded_from_disk
    assert len(eng2.buffer) >= 1

    delete_checkpoint(state)
    assert not meta_p.is_file()
