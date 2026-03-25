from __future__ import annotations

import os

os.environ.setdefault("IME_RATE_LIMIT_PER_MINUTE", "100000")

import pytest
from fastapi.testclient import TestClient

from incident_memory_engine.api.app import app, get_engine
from incident_memory_engine.config import EngineConfig
from incident_memory_engine.core.engine import IncidentMemoryEngine


@pytest.fixture
def engine(tmp_path) -> IncidentMemoryEngine:
    """Fresh engine per test (small config for speed)."""
    return IncidentMemoryEngine(
        EngineConfig(
            seed=42,
            num_eras=4,
            replay_capacity=120,
            batch_size=32,
            test_samples_per_era=64,
            persistence_enabled=False,
            state_path=str(tmp_path / "engine_state.pkl"),
            encoder_kind="hashing",
        )
    )


@pytest.fixture
def client(engine: IncidentMemoryEngine) -> TestClient:
    """HTTP client with engine dependency overridden."""
    app.dependency_overrides[get_engine] = lambda: engine
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()
