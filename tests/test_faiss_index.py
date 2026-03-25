from __future__ import annotations

import numpy as np

from incident_memory_engine.buffer.replay_buffer import ReplayBuffer
from incident_memory_engine.core.vector_index import FaissVectorIndex


def test_faiss_search_matches_bruteforce_order_small() -> None:
    dim = 8
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((24, dim)).astype(np.float32)
    ids = [f"id-{i}" for i in range(24)]
    q = rng.standard_normal(dim).astype(np.float32)

    brute = sorted(
        range(24),
        key=lambda i: float(np.linalg.norm(mat[i] - q)),
    )[:5]
    idx = FaissVectorIndex(dim)
    idx.rebuild_from_vectors(mat, ids)
    got = idx.search(q, 5)
    got_rows = [ids.index(g[0]) for g in got if g[0] is not None]
    assert got_rows == brute
