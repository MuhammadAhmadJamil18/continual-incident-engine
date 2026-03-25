from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BufferEntry:
    """One stored incident for replay and similarity search."""

    features: np.ndarray  # shape (feature_dim,), float32
    label: int
    era: int
    incident_id: str | None = None
    timestamp: float = 0.0
    incident_type: str = ""
    fix_text: str = ""
    memory_tier: str = "short_term"  # short_term | long_term | critical


class ReplayBuffer:
    """
    Fixed-capacity reservoir replay buffer.

    Each stored item supports rehearsal (sample_batch) and retrieval (/similar).
    """

    def __init__(self, capacity: int, feature_dim: int) -> None:
        self.capacity = max(0, capacity)
        self.feature_dim = feature_dim
        self._entries: list[BufferEntry] = []
        self._by_id: dict[str, BufferEntry] = {}

    def _sync_id_map(self) -> None:
        self._by_id.clear()
        for e in self._entries:
            if e.incident_id:
                self._by_id[e.incident_id] = e

    def get_by_incident_id(self, incident_id: str) -> BufferEntry | None:
        return self._by_id.get(incident_id)

    def iter_entries(self) -> list[BufferEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def add_sample(
        self,
        features: list[float] | np.ndarray,
        label: int,
        era: int,
        incident_id: str | None = None,
        *,
        timestamp: float | None = None,
        incident_type: str = "",
        fix_text: str = "",
        memory_tier: str = "short_term",
    ) -> None:
        vec = np.asarray(features, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.feature_dim:
            raise ValueError(
                f"features length {vec.shape[0]} != feature_dim {self.feature_dim}"
            )
        ts = float(timestamp if timestamp is not None else time.time())
        tier = memory_tier if memory_tier in (
            "short_term",
            "long_term",
            "critical",
        ) else "short_term"
        entry = BufferEntry(
            features=vec.copy(),
            label=int(label),
            era=int(era),
            incident_id=incident_id,
            timestamp=ts,
            incident_type=incident_type or f"class_{int(label)}",
            fix_text=fix_text or "",
            memory_tier=tier,
        )
        if self.capacity <= 0:
            return
        if len(self._entries) < self.capacity:
            self._entries.append(entry)
        else:
            j = random.randint(0, self.capacity - 1)
            old = self._entries[j]
            if old.incident_id and old.incident_id in self._by_id:
                del self._by_id[old.incident_id]
            self._entries[j] = entry
        if entry.incident_id:
            self._by_id[entry.incident_id] = entry

    def sample_batch(
        self,
        k: int,
        device: torch.device,
        *,
        tier_weights: tuple[float, float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if len(self._entries) == 0 or k <= 0:
            return None
        k = min(k, len(self._entries))
        if tier_weights is None:
            idx = random.sample(range(len(self._entries)), k)
        else:
            w_short, w_long, w_crit = tier_weights
            buckets: dict[str, list[int]] = {
                "short_term": [],
                "long_term": [],
                "critical": [],
            }
            for i, e in enumerate(self._entries):
                tier = (
                    e.memory_tier
                    if e.memory_tier in buckets
                    else "short_term"
                )
                buckets[tier].append(i)
            keys = ("short_term", "long_term", "critical")
            weights = [w_short, w_long, w_crit]
            pools: dict[str, list[int]] = {
                t: buckets[t][:] for t in keys if buckets[t]
            }
            wmap = dict(zip(keys, weights))
            idx = []
            for _ in range(k):
                active_tiers = [t for t in keys if pools.get(t)]
                if not active_tiers:
                    break
                tw = sum(wmap[t] for t in active_tiers)
                r = random.random() * tw
                acc = 0.0
                chosen = active_tiers[-1]
                for t in active_tiers:
                    acc += wmap[t]
                    if r <= acc:
                        chosen = t
                        break
                pl = pools[chosen]
                j = random.randrange(len(pl))
                idx.append(pl.pop(j))
            seen = set(idx)
            rest = [i for i in range(len(self._entries)) if i not in seen]
            random.shuffle(rest)
            for i in rest:
                if len(idx) >= k:
                    break
                idx.append(i)
            random.shuffle(idx)
            idx = idx[:k]
        xs = np.stack([self._entries[i].features for i in idx], axis=0)
        ys = [self._entries[i].label for i in idx]
        x_t = torch.from_numpy(xs).to(device)
        y_t = torch.tensor(ys, dtype=torch.long, device=device)
        return x_t, y_t

    def era_histogram(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for e in self._entries:
            out[e.era] = out.get(e.era, 0) + 1
        return dict(sorted(out.items()))

    def find_similar(
        self,
        query_features: list[float] | np.ndarray,
        k: int = 5,
        *,
        recency_half_life_s: float = 86_400.0,
    ) -> list[tuple[BufferEntry, float, float, float]]:
        """
        Rank buffer entries by combined similarity and recency.

        Args:
            query_features: Query embedding.
            k: Max results.
            recency_half_life_s: Larger = slower decay of recency bonus.

        Returns:
            List of tuples ``(entry, euclidean_distance, similarity_score, rank_score)``
            sorted by descending ``rank_score``.
        """
        q = np.asarray(query_features, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.feature_dim:
            raise ValueError(
                f"query length {q.shape[0]} != feature_dim {self.feature_dim}"
            )
        if not self._entries or k <= 0:
            return []
        now = time.time()
        scored: list[tuple[BufferEntry, float, float, float]] = []
        for ent in self._entries:
            dist = float(np.linalg.norm(ent.features - q))
            sim = 1.0 / (1.0 + dist)
            age_s = max(0.0, now - ent.timestamp)
            hl = max(recency_half_life_s, 1.0)
            recency = float(np.exp(-age_s / hl))
            rank = sim + 0.35 * recency
            scored.append((ent, dist, sim, rank))
        scored.sort(key=lambda t: t[3], reverse=True)
        return scored[:k]
