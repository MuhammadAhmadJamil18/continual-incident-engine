from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .buffer import ReplayBuffer
from .config import ExperimentConfig
from .data import build_per_era_test_sets, sample_era_batch
from .metrics import evaluate_all_seen_eras, summary_from_matrix
from .model import IncidentMLP


def _train_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    model.train()
    opt.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    return float(loss.detach().cpu())


def run_era_training(
    cfg: ExperimentConfig,
    use_replay: bool,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    test_datasets, class_means, era_drifts = build_per_era_test_sets(cfg, rng)
    test_loaders = [
        DataLoader(ds, batch_size=256, shuffle=False) for ds in test_datasets
    ]

    model = IncidentMLP(cfg.feature_dim, cfg.hidden_dim, cfg.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    buffer: ReplayBuffer | None = None
    if use_replay:
        buffer = ReplayBuffer(cfg.replay_capacity, cfg.feature_dim, cfg.num_classes)

    acc_matrix: list[dict[int, float]] = []
    buffer_hist: list[dict[int, int]] = []

    for era in range(cfg.num_eras):
        for _ in range(cfg.steps_per_era):
            x_new, y_new = sample_era_batch(
                era,
                cfg.batch_size,
                class_means,
                era_drifts,
                cfg,
                rng,
            )
            x_new = x_new.to(device)
            y_new = y_new.to(device)

            if use_replay and buffer is not None and len(buffer) > 0:
                replay_n = min(
                    int(cfg.batch_size * cfg.replay_batch_ratio),
                    len(buffer),
                )
            else:
                replay_n = 0
            new_n = cfg.batch_size - replay_n

            parts_x: list[torch.Tensor] = []
            parts_y: list[torch.Tensor] = []
            if new_n > 0:
                take = min(new_n, x_new.shape[0])
                idx = torch.randperm(x_new.shape[0], device=device)[:take]
                parts_x.append(x_new[idx])
                parts_y.append(y_new[idx])
            if replay_n > 0 and buffer is not None:
                sampled = buffer.sample(replay_n, device)
                if sampled is not None:
                    xr, yr = sampled
                    parts_x.append(xr)
                    parts_y.append(yr)

            if parts_x:
                x_cat = torch.cat(parts_x, dim=0)
                y_cat = torch.cat(parts_y, dim=0)
            else:
                x_cat, y_cat = x_new, y_new

            _train_step(model, opt, loss_fn, x_cat, y_cat)
            if use_replay and buffer is not None:
                buffer.add_batch(x_new.detach(), y_new.detach(), era)

        evals = evaluate_all_seen_eras(model, test_loaders, era, device)
        acc_matrix.append({int(k): float(v) for k, v in evals.items()})
        if buffer is not None:
            buffer_hist.append(buffer.era_histogram())

    summary = summary_from_matrix(acc_matrix)

    return {
        "mode": "replay" if use_replay else "naive",
        "config": asdict(cfg),
        "device": str(device),
        "acc_matrix": acc_matrix,
        "buffer_history": buffer_hist,
        "summary": {
            "avg_acc_all_seen": summary.avg_acc_all_seen,
            "bwt": summary.bwt,
            "forgetting_mean": summary.forgetting_mean,
        },
    }


def run_comparison(cfg: ExperimentConfig | None = None) -> dict:
    cfg = cfg or ExperimentConfig()
    naive = run_era_training(cfg, use_replay=False)
    replay = run_era_training(cfg, use_replay=True)
    return {"naive": naive, "replay": replay, "config": asdict(cfg)}


def save_results(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
