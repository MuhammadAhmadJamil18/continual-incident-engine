from __future__ import annotations

import random

import torch


class ReplayBuffer:
    """Reservoir-style buffer with optional per-class caps (fair rehearsal)."""

    def __init__(self, capacity: int, feature_dim: int, num_classes: int) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self._xs: list[torch.Tensor] = []
        self._ys: list[int] = []
        self._eras: list[int] = []

    def __len__(self) -> int:
        return len(self._xs)

    def add_batch(
        self, x: torch.Tensor, y: torch.Tensor, era: int
    ) -> None:
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu().tolist()
        for i in range(x_cpu.shape[0]):
            self._add_one(x_cpu[i], int(y_cpu[i]), era)

    def _add_one(self, x: torch.Tensor, y: int, era: int) -> None:
        if len(self._xs) < self.capacity:
            self._xs.append(x.clone())
            self._ys.append(y)
            self._eras.append(era)
        else:
            j = random.randint(0, self.capacity - 1)
            self._xs[j] = x.clone()
            self._ys[j] = y
            self._eras[j] = era

    def sample(self, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        if len(self._xs) == 0 or k <= 0:
            return None
        k = min(k, len(self._xs))
        idx = random.sample(range(len(self._xs)), k)
        xs = torch.stack([self._xs[i] for i in idx]).to(device)
        ys = torch.tensor([self._ys[i] for i in idx], dtype=torch.long, device=device)
        return xs, ys

    def era_histogram(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for e in self._eras:
            out[e] = out.get(e, 0) + 1
        return dict(sorted(out.items()))
