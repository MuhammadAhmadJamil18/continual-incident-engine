from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriftTracker:
    """Lightweight streaming stats on embedding L2 norms."""

    window: int = 256
    _norms: deque[float] = field(default_factory=deque)
    reference_mean: float | None = None
    reference_var: float | None = None

    def observe(self, vectors: list[list[float]]) -> None:
        for v in vectors:
            n = float(np.linalg.norm(np.asarray(v, dtype=np.float32)))
            self._norms.append(n)
            while len(self._norms) > self.window:
                self._norms.popleft()
        if self.reference_mean is None and len(self._norms) >= min(32, self.window // 2):
            arr = np.array(self._norms, dtype=np.float64)
            self.reference_mean = float(arr.mean())
            self.reference_var = float(arr.var() + 1e-8)

    def snapshot(self) -> dict:
        if not self._norms:
            return {
                "score": 0.0,
                "window_samples": 0,
                "reference_mean_norm": self.reference_mean,
                "current_mean_norm": None,
                "recommendation": "none",
            }
        arr = np.array(self._norms, dtype=np.float64)
        cur_m = float(arr.mean())
        ref = self.reference_mean
        if ref is None or ref <= 0:
            return {
                "score": 0.0,
                "window_samples": len(self._norms),
                "reference_mean_norm": ref,
                "current_mean_norm": cur_m,
                "recommendation": "collect_more",
            }
        z = abs(cur_m - ref) / (np.sqrt(self.reference_var or 1e-8))
        score = float(min(1.0, z / 4.0))
        if score < 0.15:
            rec = "none"
        elif score < 0.35:
            rec = "monitor"
        elif score < 0.6:
            rec = "run_evaluation"
        else:
            rec = "manual_review"
        return {
            "score": score,
            "window_samples": len(self._norms),
            "reference_mean_norm": ref,
            "current_mean_norm": cur_m,
            "recommendation": rec,
        }
