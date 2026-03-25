from __future__ import annotations

import torch

from incident_memory_engine.buffer import ReplayBuffer


def test_add_sample_and_sample_batch() -> None:
    """Buffer stores samples and returns a torch batch on the requested device."""
    buf = ReplayBuffer(capacity=100, feature_dim=4)
    buf.add_sample([0.0, 1.0, 2.0, 3.0], label=2, era=0, incident_id="a")
    buf.add_sample([1.0, 1.0, 1.0, 1.0], label=1, era=1, incident_id="b")
    out = buf.sample_batch(2, device=torch.device("cpu"))
    assert out is not None
    x, y = out
    assert x.shape == (2, 4)
    assert y.shape == (2,)


def test_reservoir_capacity() -> None:
    """At capacity, reservoir replaces random slots; size stays capped."""
    buf = ReplayBuffer(capacity=5, feature_dim=2)
    for i in range(20):
        buf.add_sample([float(i), float(i)], label=i % 3, era=i % 2)
    assert len(buf) == 5


def test_find_similar_orders_by_distance() -> None:
    """Nearest neighbor in feature space is returned first."""
    buf = ReplayBuffer(capacity=10, feature_dim=3)
    buf.add_sample([0.0, 0.0, 0.0], label=0, era=0)
    buf.add_sample([10.0, 0.0, 0.0], label=1, era=0)
    buf.add_sample([0.2, 0.0, 0.0], label=2, era=1)
    hits = buf.find_similar([0.05, 0.0, 0.0], k=2, recency_half_life_s=1e9)
    assert len(hits) == 2
    assert hits[0][0].label == 0
    assert hits[1][0].label == 2
