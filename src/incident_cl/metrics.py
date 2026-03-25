from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class CLMetricSummary:
    """After training through all eras (final row)."""

    avg_acc_all_seen: float
    bwt: float
    forgetting_mean: float


def accuracy_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def evaluate_all_seen_eras(
    model: nn.Module,
    test_loaders: list[DataLoader],
    current_era: int,
    device: torch.device,
) -> dict[int, float]:
    """Evaluate on test sets for eras 0..current_era inclusive."""
    out: dict[int, float] = {}
    for j in range(current_era + 1):
        out[j] = accuracy_on_loader(model, test_loaders[j], device)
    return out


def compute_bwt_and_forgetting(
    acc_matrix: list[dict[int, float]],
) -> tuple[float, float]:
    """
    acc_matrix[k] = accuracies on each test era after training era k (keys 0..k).

    BWT (backward transfer): 1/(T-1) * sum_{i=0}^{T-2} (a_{T-1,i} - a_{i,i}).
    Forgetting (mean max forgetting): for each task i, F_i = max_k (a_{i,i} - a_{k,i})
    for k>=i, then average F_i. Uses diagonal a_{i,i} as peak after learning task i.
    """
    t = len(acc_matrix)
    if t < 2:
        return 0.0, 0.0

    final = acc_matrix[-1]
    bwt_sum = 0.0
    for i in range(t - 1):
        a_ii = acc_matrix[i][i]
        a_Ti = final[i]
        bwt_sum += a_Ti - a_ii
    bwt = bwt_sum / (t - 1)

    forgettings: list[float] = []
    for i in range(t):
        a_ii = acc_matrix[i][i]
        worst = a_ii
        for k in range(i, t):
            a_ki = acc_matrix[k][i]
            worst = min(worst, a_ki)
        forgettings.append(a_ii - worst)
    forgetting_mean = sum(forgettings) / len(forgettings)

    return bwt, forgetting_mean


def summary_from_matrix(acc_matrix: list[dict[int, float]]) -> CLMetricSummary:
    final = acc_matrix[-1]
    keys = sorted(final.keys())
    avg_acc = sum(final[j] for j in keys) / len(keys)
    bwt, f_mean = compute_bwt_and_forgetting(acc_matrix)
    return CLMetricSummary(
        avg_acc_all_seen=avg_acc,
        bwt=bwt,
        forgetting_mean=f_mean,
    )
