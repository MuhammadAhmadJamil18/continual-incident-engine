from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class CLMetricSummary:
    """Aggregate CL statistics after training through the final era."""

    avg_acc_all_seen: float
    bwt: float
    forgetting_mean: float


def per_class_accuracy_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[int, float]:
    """
    Per-class accuracy on a single dataloader (e.g. legacy era-0 holdout).

    Classes with zero support are omitted from the returned map.
    """
    model.eval()
    correct: dict[int, int] = defaultdict(int)
    total: dict[int, int] = defaultdict(int)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            for c in range(num_classes):
                mask = y == c
                if mask.any():
                    total[c] += int(mask.sum().item())
                    correct[c] += int((pred[mask] == y[mask]).sum().item())
    out: dict[int, float] = {}
    for c in total:
        if total[c] > 0:
            out[c] = correct[c] / total[c]
    return out


def accuracy_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute classification accuracy on a dataloader.

    Args:
        model: Classifier in eval mode internally.
        loader: Batches of (x, y).
        device: Torch device.

    Returns:
        Fraction correct in [0, 1].
    """
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
    """
    Evaluate the model on held-out test sets for eras 0..current_era.

    Args:
        model: Trained classifier.
        test_loaders: One loader per era (fixed holdouts).
        current_era: Highest era index included in evaluation.
        device: Torch device.

    Returns:
        Map era_index -> accuracy.
    """
    out: dict[int, float] = {}
    for j in range(current_era + 1):
        out[j] = accuracy_on_loader(model, test_loaders[j], device)
    return out


def compute_bwt_and_forgetting(
    acc_matrix: list[dict[int, float]],
) -> tuple[float, float]:
    """
    Compute backward transfer (BWT) and mean forgetting from an accuracy matrix.

    acc_matrix[k] holds accuracies on each test era after training through era k.

    Args:
        acc_matrix: List of per-era test accuracies keyed by test era index.

    Returns:
        (bwt, mean_forgetting).
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
    """Build summary object from a full accuracy matrix."""
    final = acc_matrix[-1]
    keys = sorted(final.keys())
    avg_acc = sum(final[j] for j in keys) / len(keys)
    bwt, f_mean = compute_bwt_and_forgetting(acc_matrix)
    return CLMetricSummary(
        avg_acc_all_seen=avg_acc,
        bwt=bwt,
        forgetting_mean=f_mean,
    )
