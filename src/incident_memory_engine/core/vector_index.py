from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorIndex(Protocol):
    """Pluggable ANN index (FAISS local, Pinecone later)."""

    def rebuild_from_vectors(
        self,
        vectors: np.ndarray,
        row_index_to_incident_id: list[str | None],
    ) -> None:
        """Replace index contents; row i matches ``row_index_to_incident_id[i]``."""

    def search(
        self,
        query: np.ndarray,
        k: int,
    ) -> list[tuple[str | None, float]]:
        """Return up to k (incident_id, l2_distance) sorted by distance ascending."""


class FaissVectorIndex:
    """L2 flat index; small/medium buffers (rebuilt each batch)."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = None
        self._row_to_id: list[str | None] = []

    def rebuild_from_vectors(
        self,
        vectors: np.ndarray,
        row_index_to_incident_id: list[str | None],
    ) -> None:
        import faiss

        if vectors.size == 0:
            self._index = None
            self._row_to_id = []
            return
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._row_to_id = list(row_index_to_incident_id)
        self._index = faiss.IndexFlatL2(int(x.shape[1]))
        self._index.add(x)

    def search(
        self,
        query: np.ndarray,
        k: int,
    ) -> list[tuple[str | None, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import faiss

        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        k = min(k, self._index.ntotal)
        dists, idxs = self._index.search(q, k)
        out: list[tuple[str | None, float]] = []
        for j in range(k):
            row = int(idxs[0, j])
            if row < 0 or row >= len(self._row_to_id):
                continue
            out.append((self._row_to_id[row], float(dists[0, j])))
        return out
