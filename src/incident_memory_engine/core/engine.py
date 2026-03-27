from __future__ import annotations

import dataclasses
import json
import os
from typing import Any
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from incident_memory_engine.buffer import ReplayBuffer
from incident_memory_engine.config import EngineConfig
from incident_memory_engine.core.data_stream import (
    build_per_era_test_sets,
    build_per_era_text_eval_datasets,
    canonical_text_labels_for_era,
    sample_era_batch,
    split_github_samples_train_holdout,
)
from incident_memory_engine.core.feature_pipeline import (
    FeatureEncoder,
    build_encoder,
    encoder_from_state,
)
from incident_memory_engine.core.model import IncidentMLP
from incident_memory_engine.core.persistence import (
    config_from_payload,
    delete_checkpoint,
    entry_from_dict,
    load_engine_state,
    save_engine_state,
)
from incident_memory_engine.metrics import (
    compute_bwt_and_forgetting,
    evaluate_all_seen_eras,
    per_class_accuracy_on_loader,
    summary_from_matrix,
)
from incident_memory_engine.metrics.forgetting_alert import (
    alert_to_dict,
    compute_forgetting_alert,
)
from incident_memory_engine.core.drift_tracker import DriftTracker
from incident_memory_engine.core.llm_assist import explain_incident, load_llm_config
from incident_memory_engine.core.vector_index import FaissVectorIndex


def _train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    ewc_lambda: float = 0.0,
    ewc_fisher: dict[str, torch.Tensor] | None = None,
    ewc_anchor: dict[str, torch.Tensor] | None = None,
) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    if (
        ewc_lambda > 0
        and ewc_fisher
        and ewc_anchor
        and len(ewc_anchor) > 0
    ):
        pen = torch.zeros((), device=x.device, dtype=loss.dtype)
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in ewc_fisher or name not in ewc_anchor:
                continue
            f = ewc_fisher[name]
            a = ewc_anchor[name]
            pen = pen + (f * (p - a).pow(2)).sum()
        loss = loss + (ewc_lambda / 2.0) * pen
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


class IncidentMemoryEngine:
    """
    Continual-learning engine: MLP + reservoir replay + optional EWC + era evaluations.

    Replay mixes past incidents into each training step; optional Elastic Weight
    Consolidation (``ewc_lambda``) penalizes deviation from consolidated anchors
    after each ``close_era``. Persists model, buffer, metrics, RNG, encoder state,
    and EWC tensors when enabled.
    """

    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        requested = cfg or EngineConfig()
        ewc_env = os.environ.get("IME_EWC_LAMBDA", "").strip()
        if ewc_env:
            requested = dataclasses.replace(requested, ewc_lambda=float(ewc_env))
        self.cfg = requested
        self._rng: np.random.Generator | None = None
        self._class_means: np.ndarray | None = None
        self._era_drifts: np.ndarray | None = None
        self._test_loaders: list[DataLoader] = []
        self._pipeline: FeatureEncoder | None = None
        self._model_in_dim: int = requested.feature_dim
        self.model: IncidentMLP | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._loss_fn = nn.CrossEntropyLoss()
        self.buffer: ReplayBuffer | None = None
        self._current_era: int = 0
        self._highest_closed_era: int = -1
        self._acc_matrix: list[dict[int, float]] = []
        self._legacy_accuracy_history: list[float] = []
        self._peak_legacy_accuracy: float | None = None
        self._buffer_history_snapshots: list[dict[int, int]] = []
        self._last_legacy_per_class: dict[int, float] = {}
        self._peak_legacy_per_class: dict[int, float] = {}
        self._loaded_from_disk: bool = False
        self._faiss_index: FaissVectorIndex | None = None
        self._drift_tracker = DriftTracker()
        self._drift_auto_era_closed: bool = False
        self._github_label_map: dict | None = None
        self._ewc_fisher: dict[str, torch.Tensor] = {}
        self._ewc_anchor: dict[str, torch.Tensor] = {}

        if self.cfg.persistence_enabled and self._try_load_checkpoint():
            self._loaded_from_disk = True
        else:
            self.reset()

    def _build_pipeline(self) -> FeatureEncoder:
        return build_encoder(self.cfg.encoder_kind, self.cfg.feature_dim)

    def _rebuild_model_head(self) -> None:
        assert self._pipeline is not None
        self._model_in_dim = int(self._pipeline.output_dim)
        self.model = IncidentMLP(
            self._model_in_dim,
            self.cfg.hidden_dim,
            self.cfg.num_classes,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def reset(self) -> None:
        """Reinitialize engine; delete on-disk checkpoint when persistence is on."""
        with self._lock:
            if self.cfg.persistence_enabled:
                delete_checkpoint(Path(self.cfg.state_path))
            cfg = self.cfg
            self._rng = np.random.default_rng(cfg.seed)
            torch.manual_seed(cfg.seed)
            tests, self._class_means, self._era_drifts = build_per_era_test_sets(
                cfg, self._rng
            )
            self._pipeline = self._build_pipeline()
            if cfg.encoder_kind.lower() == "identity":
                self._test_loaders = [
                    DataLoader(ds, batch_size=256, shuffle=False) for ds in tests
                ]
            else:
                tds = build_per_era_text_eval_datasets(cfg, self._rng, self._pipeline)
                self._test_loaders = [
                    DataLoader(ds, batch_size=256, shuffle=False) for ds in tds
                ]
            self._rebuild_model_head()
            self.buffer = ReplayBuffer(cfg.replay_capacity, self._model_in_dim)
            self._current_era = 0
            self._highest_closed_era = -1
            self._acc_matrix = []
            self._legacy_accuracy_history = []
            self._peak_legacy_accuracy = None
            self._buffer_history_snapshots = []
            self._last_legacy_per_class = {}
            self._peak_legacy_per_class = {}
            self._loaded_from_disk = False
            self._drift_tracker = DriftTracker()
            self._drift_auto_era_closed = False
            self._github_label_map = None
            self._ewc_fisher = {}
            self._ewc_anchor = {}
            self._rebuild_faiss_index()

    def _try_load_checkpoint(self) -> bool:
        path = Path(self.cfg.state_path)
        payload = load_engine_state(path)
        if payload is None or payload.get("version") not in (2, 3):
            return False
        try:
            new_cfg = config_from_payload(payload["cfg"])
            pipeline = encoder_from_state(
                payload["pipeline_state"],
                new_cfg.feature_dim,
            )
            model_in_dim = int(payload.get("model_in_dim", new_cfg.feature_dim))
            class_means = np.asarray(payload["class_means"], dtype=np.float32)
            era_drifts = np.asarray(payload["era_drifts"], dtype=np.float32)
            rng = np.random.default_rng()
            rng.bit_generator.state = payload["rng_state"]
            if new_cfg.encoder_kind.lower() == "identity":
                tests: list[TensorDataset] = []
                for e in range(new_cfg.num_eras):
                    x_t, y_t = sample_era_batch(
                        e,
                        new_cfg.test_samples_per_era,
                        class_means,
                        era_drifts,
                        new_cfg,
                        rng,
                    )
                    tests.append(TensorDataset(x_t, y_t))
                test_loaders = [
                    DataLoader(ds, batch_size=256, shuffle=False) for ds in tests
                ]
            else:
                tds = build_per_era_text_eval_datasets(new_cfg, rng, pipeline)
                test_loaders = [
                    DataLoader(ds, batch_size=256, shuffle=False) for ds in tds
                ]
            model = IncidentMLP(
                model_in_dim,
                new_cfg.hidden_dim,
                new_cfg.num_classes,
            ).to(self._device)
            sd = payload["model_state"]
            model.load_state_dict({k: v.to(self._device) for k, v in sd.items()})
            optimizer = torch.optim.Adam(model.parameters(), lr=new_cfg.lr)
            optimizer.load_state_dict(payload["optimizer_state"])
            for st in optimizer.state.values():
                for k, v in list(st.items()):
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to(self._device)
            buffer = ReplayBuffer(new_cfg.replay_capacity, model_in_dim)
            for ed in payload["buffer_entries"]:
                buffer._entries.append(entry_from_dict(ed))
            acc_matrix = [
                {int(k): float(v) for k, v in row.items()}
                for row in payload["acc_matrix"]
            ]
            legacy_accuracy_history = list(
                payload.get("legacy_accuracy_history", [])
            )
            peak_legacy_accuracy = payload.get("peak_legacy_accuracy")
            highest_closed_era = int(payload.get("highest_closed_era", -1))
            current_era = int(payload.get("current_era", 0))
            buffer_history_snapshots = list(
                payload.get("buffer_history_snapshots", [])
            )
            last_legacy_per_class = {
                int(k): float(v)
                for k, v in payload.get("last_legacy_per_class", {}).items()
            }
            peak_legacy_per_class = {
                int(k): float(v)
                for k, v in payload.get("peak_legacy_per_class", {}).items()
            }

            self.cfg = new_cfg
            self._pipeline = pipeline
            self._model_in_dim = model_in_dim
            self._class_means = class_means
            self._era_drifts = era_drifts
            self._rng = rng
            self._test_loaders = test_loaders
            self.model = model
            self._optimizer = optimizer
            self.buffer = buffer
            self._acc_matrix = acc_matrix
            self._legacy_accuracy_history = legacy_accuracy_history
            self._peak_legacy_accuracy = peak_legacy_accuracy
            self._highest_closed_era = highest_closed_era
            self._current_era = current_era
            self._buffer_history_snapshots = buffer_history_snapshots
            self._last_legacy_per_class = last_legacy_per_class
            self._peak_legacy_per_class = peak_legacy_per_class
            self._github_label_map = payload.get("github_label_map")
            buffer._sync_id_map()
            self._rebuild_faiss_index()
            self._restore_ewc_from_meta(
                payload.get("ewc_fisher"),
                payload.get("ewc_anchor"),
            )
            return True
        except Exception:
            return False

    def _persist_checkpoint(self) -> None:
        if not self.cfg.persistence_enabled:
            return
        assert (
            self.model is not None
            and self._optimizer is not None
            and self.buffer is not None
            and self._pipeline is not None
            and self._class_means is not None
            and self._era_drifts is not None
            and self._rng is not None
        )
        fm, am = self._ewc_meta_for_save()
        save_engine_state(
            Path(self.cfg.state_path),
            cfg=self.cfg,
            model=self.model,
            optimizer=self._optimizer,
            buffer=self.buffer,
            pipeline=self._pipeline,
            acc_matrix=self._acc_matrix,
            legacy_accuracy_history=self._legacy_accuracy_history,
            peak_legacy_accuracy=self._peak_legacy_accuracy,
            highest_closed_era=self._highest_closed_era,
            current_era=self._current_era,
            buffer_history_snapshots=self._buffer_history_snapshots,
            last_legacy_per_class=self._last_legacy_per_class,
            peak_legacy_per_class=self._peak_legacy_per_class,
            rng_bit_generator=self._rng.bit_generator,
            model_in_dim=self._model_in_dim,
            class_means=self._class_means,
            era_drifts=self._era_drifts,
            github_label_map=self._github_label_map,
            ewc_fisher_meta=fm,
            ewc_anchor_meta=am,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def highest_closed_era(self) -> int:
        return self._highest_closed_era

    @property
    def loaded_from_disk(self) -> bool:
        return self._loaded_from_disk

    @property
    def model_in_dim(self) -> int:
        return self._model_in_dim

    def class_name_for(self, class_id: int) -> str:
        names = self.cfg.incident_class_names
        if 0 <= class_id < len(names):
            return names[class_id]
        return f"class_{class_id}"

    def _rebuild_faiss_index(self) -> None:
        if self.buffer is None:
            return
        if self._faiss_index is None or self._faiss_index._dim != self._model_in_dim:
            self._faiss_index = FaissVectorIndex(self._model_in_dim)
        entries = self.buffer.iter_entries()
        if not entries:
            self._faiss_index.rebuild_from_vectors(
                np.zeros((0, self._model_in_dim), dtype=np.float32),
                [],
            )
            return
        mat = np.stack([e.features for e in entries], axis=0).astype(np.float32)
        ids = [e.incident_id for e in entries]
        self._faiss_index.rebuild_from_vectors(mat, ids)

    def drift_snapshot(self) -> dict:
        with self._lock:
            return self._drift_tracker.snapshot()

    def maybe_auto_close_era_on_drift(self, snap: dict) -> bool:
        flag = os.environ.get("IME_AUTO_ERA_CLOSE_ON_DRIFT", "").lower()
        if flag not in ("1", "true", "yes"):
            return False
        if self._drift_auto_era_closed:
            return False
        if snap.get("recommendation") != "manual_review":
            return False
        if float(snap.get("score", 0.0)) < 0.55:
            return False
        era = self._current_era
        self.close_era(era)
        self._drift_auto_era_closed = True
        return True

    def ingest_github_issues(
        self,
        repos: list[str],
        era: int,
        per_repo: int,
        *,
        token: str | None = None,
        chunk_size: int = 64,
    ) -> dict[str, float | int | None]:
        """
        Fetch issues from GitHub, embed text, and run incremental ``train_batch`` chunks.

        Sets ``_github_label_map`` for persistence. Network I/O runs outside the engine lock;
        training uses existing locked paths.
        """
        from incident_memory_engine.data.github_ingest import (
            exposed_label_map,
            fetch_repo_issues,
            issues_to_samples,
        )

        nc = self.cfg.num_classes
        samples: list[dict[str, Any]] = []
        for repo in repos:
            issues = fetch_repo_issues(repo, per_repo, token=token)
            samples.extend(
                issues_to_samples(issues, era=era, repo=repo, num_classes=nc)
            )
        if not samples:
            raise ValueError(
                "No GitHub issues collected (rate limit, filters, or empty repos)"
            )
        meta = exposed_label_map()
        meta["num_classes"] = nc
        self._github_label_map = meta
        loss_last: float | None = None
        for start in range(0, len(samples), chunk_size):
            sub = samples[start : start + chunk_size]
            texts = [s["text"] for s in sub]
            labels = [int(s["label"]) for s in sub]
            feats = self.encode_texts(texts)
            itypes = [str(s["source"]) for s in sub]
            fixes = [str(s.get("title", ""))[:500] for s in sub]
            iids = [f"gh-{s.get('github_id')}" for s in sub]
            out = self.train_batch(
                era,
                feats,
                labels,
                incident_types=itypes,
                fixes=fixes,
                incident_ids=iids,
            )
            loss_last = float(out["loss"])
        return {
            "trained_rows": len(samples),
            "era": era,
            "loss_last": loss_last,
        }

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed raw log lines / incident text to vectors (matches model input dim)."""
        with self._lock:
            assert self._pipeline is not None
            self._pipeline.observe_training_texts(texts)
            out = [self._pipeline.transform(t).tolist() for t in texts]
            self._drift_tracker.observe(out)
            return out

    def train_batch(
        self,
        era: int,
        features_list: list[list[float]],
        labels: list[int],
        *,
        incident_types: list[str] | None = None,
        fixes: list[str] | None = None,
        timestamps: list[float] | None = None,
        incident_ids: list[str | None] | None = None,
        memory_tiers: list[str] | None = None,
    ) -> dict[str, float]:
        if len(features_list) != len(labels) or not features_list:
            raise ValueError("features_list and labels must be non-empty and aligned")
        n = len(labels)
        itypes = incident_types or [f"class_{labels[i]}" for i in range(n)]
        fixs = fixes or [""] * n
        tstamps = timestamps or [time.time()] * n
        iids = incident_ids or [None] * n
        mtiers = memory_tiers or (["short_term"] * n)
        cfg = self.cfg
        with self._lock:
            assert self.model is not None and self._optimizer is not None
            assert self.buffer is not None
            x_new = torch.tensor(features_list, dtype=torch.float32, device=self._device)
            y_new = torch.tensor(labels, dtype=torch.long, device=self._device)
            self._current_era = era

            if cfg.replay_enabled and len(self.buffer) > 0:
                replay_n = min(
                    int(x_new.shape[0] * cfg.replay_batch_ratio),
                    len(self.buffer),
                )
            else:
                replay_n = 0
            new_n = x_new.shape[0] - replay_n

            parts_x: list[torch.Tensor] = []
            parts_y: list[torch.Tensor] = []
            if new_n > 0:
                take = min(new_n, x_new.shape[0])
                idx = torch.randperm(x_new.shape[0], device=self._device)[:take]
                parts_x.append(x_new[idx])
                parts_y.append(y_new[idx])
            if replay_n > 0:
                sampled = self.buffer.sample_batch(
                    replay_n,
                    self._device,
                    tier_weights=(
                        cfg.replay_tier_weight_short,
                        cfg.replay_tier_weight_long,
                        cfg.replay_tier_weight_critical,
                    ),
                )
                if sampled is not None:
                    xr, yr = sampled
                    parts_x.append(xr)
                    parts_y.append(yr)

            if parts_x:
                x_cat = torch.cat(parts_x, dim=0)
                y_cat = torch.cat(parts_y, dim=0)
            else:
                x_cat, y_cat = x_new, y_new

            loss = _train_step(
                self.model,
                self._optimizer,
                self._loss_fn,
                x_cat,
                y_cat,
                ewc_lambda=self.cfg.ewc_lambda,
                ewc_fisher=self._ewc_fisher or None,
                ewc_anchor=self._ewc_anchor or None,
            )

            if cfg.replay_enabled:
                for i in range(x_new.shape[0]):
                    tier = mtiers[i] if i < len(mtiers) else "short_term"
                    self.buffer.add_sample(
                        features=features_list[i],
                        label=int(labels[i]),
                        era=era,
                        incident_id=iids[i] or str(uuid.uuid4()),
                        timestamp=tstamps[i],
                        incident_type=itypes[i],
                        fix_text=fixs[i],
                        memory_tier=tier,
                    )

            self._rebuild_faiss_index()
            self._persist_checkpoint()
            return {"loss": loss, "batch_size": int(x_new.shape[0])}

    def close_era(self, era: int) -> dict[int, float]:
        with self._lock:
            assert self.model is not None
            evals = evaluate_all_seen_eras(
                self.model, self._test_loaders, era, self._device
            )
            row = {int(k): float(v) for k, v in evals.items()}
            self._acc_matrix.append(row)
            self._highest_closed_era = max(self._highest_closed_era, era)
            legacy = row.get(0)
            if legacy is not None:
                self._legacy_accuracy_history.append(legacy)
                if self._peak_legacy_accuracy is None:
                    self._peak_legacy_accuracy = legacy
                else:
                    self._peak_legacy_accuracy = max(
                        self._peak_legacy_accuracy, legacy
                    )
                pc = per_class_accuracy_on_loader(
                    self.model,
                    self._test_loaders[0],
                    self._device,
                    self.cfg.num_classes,
                )
                self._last_legacy_per_class = pc
                for c, acc in pc.items():
                    prev = self._peak_legacy_per_class.get(c, 0.0)
                    self._peak_legacy_per_class[c] = max(prev, acc)
            if self.buffer is not None:
                self._buffer_history_snapshots.append(self.buffer.era_histogram())
            self._ewc_consolidate_after_close(era)
            self._persist_checkpoint()
            return row

    def _ewc_consolidate_after_close(self, closed_era: int) -> None:
        """Diagonal Fisher + anchor params after an era boundary (EWC)."""
        cfg = self.cfg
        if cfg.ewc_lambda <= 0 or self.model is None:
            return
        self.model.eval()
        fisher_acc: dict[str, torch.Tensor] = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher_acc[n] = torch.zeros_like(p, device=self._device)
        batches_done = 0
        max_b = max(1, cfg.ewc_fisher_batches)
        for e in range(closed_era + 1):
            for xb, yb in self._test_loaders[e]:
                if batches_done >= max_b:
                    break
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                self.model.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self._loss_fn(logits, yb)
                loss.backward()
                for n, p in self.model.named_parameters():
                    if p.grad is None:
                        continue
                    fisher_acc[n] = fisher_acc[n] + p.grad.detach().pow(2)
                batches_done += 1
            if batches_done >= max_b:
                break
        if batches_done == 0:
            return
        scale = 1.0 / float(batches_done)
        ema = cfg.ewc_fisher_ema
        for n, p in self.model.named_parameters():
            if n not in fisher_acc:
                continue
            new_f = fisher_acc[n] * scale
            if n in self._ewc_fisher:
                self._ewc_fisher[n] = ema * self._ewc_fisher[n] + (1.0 - ema) * new_f
            else:
                self._ewc_fisher[n] = new_f.clone()
        self._ewc_anchor = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def _ewc_meta_for_save(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not self._ewc_fisher or not self._ewc_anchor:
            return None, None
        return (
            {k: v.detach().cpu().tolist() for k, v in self._ewc_fisher.items()},
            {k: v.detach().cpu().tolist() for k, v in self._ewc_anchor.items()},
        )

    def _restore_ewc_from_meta(
        self,
        fisher_meta: dict[str, Any] | None,
        anchor_meta: dict[str, Any] | None,
    ) -> None:
        self._ewc_fisher.clear()
        self._ewc_anchor.clear()
        if (
            not fisher_meta
            or not anchor_meta
            or self.model is None
        ):
            return
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in fisher_meta or name not in anchor_meta:
                continue
            try:
                ft = torch.tensor(
                    fisher_meta[name], dtype=p.dtype, device=self._device
                ).reshape(p.shape)
                at = torch.tensor(
                    anchor_meta[name], dtype=p.dtype, device=self._device
                ).reshape(p.shape)
            except (RuntimeError, ValueError, TypeError):
                continue
            self._ewc_fisher[name] = ft
            self._ewc_anchor[name] = at

    @property
    def ewc_is_active(self) -> bool:
        return self.cfg.ewc_lambda > 0 and bool(self._ewc_anchor)

    def predict_one(self, features: list[float]) -> dict[str, float | int]:
        with self._lock:
            assert self.model is not None
            if len(features) != self._model_in_dim:
                raise ValueError(
                    f"Expected {self._model_in_dim} features, got {len(features)}"
                )
            x = torch.tensor([features], dtype=torch.float32, device=self._device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)[0]
                conf, pred = torch.max(probs, dim=-1)
            return {
                "predicted_class": int(pred.item()),
                "confidence": float(conf.item()),
            }

    def metrics_payload(self) -> dict:
        with self._lock:
            bwt, f_mean = compute_bwt_and_forgetting(self._acc_matrix)
            summary = (
                summary_from_matrix(self._acc_matrix)
                if self._acc_matrix
                else None
            )
            matrix_list = [
                {str(k): v for k, v in row.items()} for row in self._acc_matrix
            ]
            current_legacy = None
            if self._legacy_accuracy_history:
                current_legacy = self._legacy_accuracy_history[-1]
            return {
                "accuracy_matrix": matrix_list,
                "bwt": bwt,
                "mean_forgetting": f_mean,
                "avg_acc_all_seen": summary.avg_acc_all_seen if summary else None,
                "legacy_accuracy_history": list(self._legacy_accuracy_history),
                "current_legacy_accuracy": current_legacy,
                "peak_legacy_accuracy": self._peak_legacy_accuracy,
                "highest_closed_era": self._highest_closed_era,
                "buffer_era_histogram": (
                    self.buffer.era_histogram() if self.buffer else {}
                ),
                "last_legacy_per_class": {
                    str(k): v for k, v in self._last_legacy_per_class.items()
                },
                "peak_legacy_per_class": {
                    str(k): v for k, v in self._peak_legacy_per_class.items()
                },
                "encoder_kind": self.cfg.encoder_kind,
                "model_in_dim": self._model_in_dim,
                "loaded_from_disk": self._loaded_from_disk,
                "continual_learning": {
                    "replay_buffer": True,
                    "replay_enabled": self.cfg.replay_enabled,
                    "replay_capacity": self.cfg.replay_capacity,
                    "replay_batch_ratio": self.cfg.replay_batch_ratio,
                    "ewc_lambda": self.cfg.ewc_lambda,
                    "ewc_consolidation_ready": bool(self._ewc_anchor),
                },
            }

    def data_status_payload(self) -> dict:
        """Samples per era in the replay buffer vs on-disk GitHub export (if present)."""
        with self._lock:
            hist = self.buffer.era_histogram() if self.buffer else {}
            file_counts: dict[str, int] = {}
            fp = Path("artifacts/github_issues.json")
            if fp.is_file():
                try:
                    obj = json.loads(fp.read_text(encoding="utf-8"))
                    rows = obj.get("samples", obj if isinstance(obj, list) else [])
                    for row in rows:
                        e = int(row.get("era", 0))
                        sk = str(e)
                        file_counts[sk] = file_counts.get(sk, 0) + 1
                except Exception:
                    pass
            return {
                "buffer_samples_per_era": hist,
                "github_file_samples_per_era": file_counts,
                "github_label_map_in_checkpoint": self._github_label_map is not None,
            }

    def run_github_file_experiment(
        self,
        data_path: str | Path,
        *,
        chunk_size: int = 64,
        reset_first: bool = True,
    ) -> dict[str, Any]:
        """
        Train the **live** engine on a ``github_issues.json``-style file, then close each era.

        Used by the API and dashboard so results appear in ``/metrics`` without a subprocess.
        """
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        raw = json.loads(path.read_text(encoding="utf-8"))
        samples = raw.get("samples", raw if isinstance(raw, list) else [])
        if not samples:
            raise ValueError("No samples in JSON")
        eras = sorted({int(s["era"]) for s in samples})
        max_e = max(eras)
        if max_e >= self.cfg.num_eras:
            raise ValueError(
                f"Data has era {max_e} but engine num_eras={self.cfg.num_eras}; "
                "raise num_eras or use a smaller export."
            )
        train_rows, holdout_by_era = split_github_samples_train_holdout(samples)
        if reset_first:
            self.reset()
        if isinstance(raw.get("label_map"), dict):
            self._github_label_map = raw["label_map"]
        if self.cfg.encoder_kind.lower() != "identity":
            self._install_github_holdout_eval_loaders(holdout_by_era)
        for era in eras:
            era_samples = [s for s in train_rows if int(s["era"]) == era]
            for start in range(0, len(era_samples), chunk_size):
                sub = era_samples[start : start + chunk_size]
                texts = [str(s["text"]) for s in sub]
                labels = [int(s["label"]) for s in sub]
                feats = self.encode_texts(texts)
                itypes = [str(s.get("source", "github")) for s in sub]
                fixes = [str(s.get("title", ""))[:500] for s in sub]
                self.train_batch(
                    era,
                    feats,
                    labels,
                    incident_types=itypes,
                    fixes=fixes,
                )
            self.close_era(era)
        return {
            "samples_trained": len(train_rows),
            "eras_closed": eras,
            "metrics": self.metrics_payload(),
            "forgetting_alert": self.forgetting_alert_payload(),
        }

    def _install_github_holdout_eval_loaders(
        self, holdout_by_era: dict[int, list[dict]]
    ) -> None:
        """Replace per-era test loaders with real held-out rows where available."""
        with self._lock:
            assert self._pipeline is not None
            assert self._rng is not None
            cfg = self.cfg
            loaders: list[DataLoader] = []
            for e in range(cfg.num_eras):
                rows = holdout_by_era.get(e) or []
                if rows:
                    texts = [str(r["text"]) for r in rows]
                    labels = [int(r["label"]) for r in rows]
                    self._pipeline.observe_training_texts(texts)
                    x = self._pipeline.transform_batch(texts)
                    ds = TensorDataset(
                        torch.from_numpy(x),
                        torch.tensor(labels, dtype=torch.long),
                    )
                else:
                    texts, y = canonical_text_labels_for_era(
                        e, cfg.test_samples_per_era, cfg, self._rng
                    )
                    self._pipeline.observe_training_texts(texts)
                    x = self._pipeline.transform_batch(texts)
                    ds = TensorDataset(
                        torch.from_numpy(x),
                        torch.from_numpy(y.astype(np.int64)),
                    )
                loaders.append(DataLoader(ds, batch_size=256, shuffle=False))
            self._test_loaders = loaders

    def forgetting_alert_payload(self) -> dict:
        with self._lock:
            current = (
                self._legacy_accuracy_history[-1]
                if self._legacy_accuracy_history
                else None
            )
            res = compute_forgetting_alert(
                current_legacy_accuracy=current,
                peak_legacy_accuracy=self._peak_legacy_accuracy,
                low_threshold=self.cfg.forgetting_alert_low,
                medium_threshold=self.cfg.forgetting_alert_medium,
                acc_matrix=self._acc_matrix,
                last_legacy_per_class=self._last_legacy_per_class,
                peak_legacy_per_class=self._peak_legacy_per_class,
                era_drop_ratio=self.cfg.forgetting_era_drop_ratio,
                class_drop_ratio=self.cfg.forgetting_class_drop_ratio,
            )
            return alert_to_dict(res)

    def sample_synthetic_batch(self, era: int, n: int) -> tuple[list[list[float]], list[int]]:
        if era < 0 or era >= self.cfg.num_eras:
            raise ValueError(f"era must be in [0, {self.cfg.num_eras - 1}], got {era}")
        if n < 1:
            raise ValueError("n must be >= 1")
        with self._lock:
            assert self._rng is not None
            assert self._class_means is not None and self._era_drifts is not None
            assert self._pipeline is not None
            if self.cfg.encoder_kind.lower() == "identity":
                x_t, y_t = sample_era_batch(
                    era,
                    n,
                    self._class_means,
                    self._era_drifts,
                    self.cfg,
                    self._rng,
                )
                feats = x_t.cpu().numpy().tolist()
            else:
                texts, y_np = canonical_text_labels_for_era(
                    era, n, self.cfg, self._rng
                )
                self._pipeline.observe_training_texts(texts)
                mat = self._pipeline.transform_batch(texts)
                feats = mat.tolist()
                y_t = torch.from_numpy(y_np.astype(np.int64))
            labs = y_t.cpu().numpy().astype(int).tolist()
            return feats, labs

    def similar_incidents(
        self,
        features: list[float],
        k: int = 5,
        *,
        memory_tier_filter: set[str] | None = None,
    ) -> list[dict[str, float | int | str | None]]:
        with self._lock:
            assert self.buffer is not None
            q = np.asarray(features, dtype=np.float32).reshape(-1)
            if q.shape[0] != self._model_in_dim:
                raise ValueError(
                    f"query length {q.shape[0]} != model_in_dim {self._model_in_dim}"
                )
            nbuf = len(self.buffer)
            pool = min(
                max(k * max(1, self.cfg.faiss_similarity_pool_multiplier), k),
                max(nbuf, 1),
            )
            hl = max(self.cfg.similarity_recency_half_life_s, 1.0)
            now = time.time()
            scored: list[tuple[object, float, float, float]] = []
            if (
                self._faiss_index is not None
                and nbuf > 0
                and self._faiss_index._index is not None
            ):
                raw = self._faiss_index.search(q, pool)
                for iid, dist in raw:
                    ent = (
                        self.buffer.get_by_incident_id(iid)
                        if iid
                        else None
                    )
                    if ent is None:
                        continue
                    if (
                        memory_tier_filter
                        and ent.memory_tier not in memory_tier_filter
                    ):
                        continue
                    sim = 1.0 / (1.0 + dist)
                    age_s = max(0.0, now - ent.timestamp)
                    recency = float(np.exp(-age_s / hl))
                    rank = sim + 0.35 * recency
                    scored.append((ent, dist, sim, rank))
                scored.sort(key=lambda t: t[3], reverse=True)
            if len(scored) < k:
                brute = self.buffer.find_similar(
                    features,
                    k=max(k * 4, k),
                    recency_half_life_s=self.cfg.similarity_recency_half_life_s,
                )
                seen_ids = {
                    e.incident_id
                    for e, _, _, _ in scored
                    if getattr(e, "incident_id", None)
                }
                for ent, dist, sim, rank in brute:
                    if memory_tier_filter and ent.memory_tier not in memory_tier_filter:
                        continue
                    if ent.incident_id and ent.incident_id in seen_ids:
                        continue
                    scored.append((ent, dist, sim, rank))
                scored.sort(key=lambda t: t[3], reverse=True)
            ranked = scored[:k]
            out: list[dict[str, float | int | str | None]] = []
            for ent, dist, sim, rank in ranked:
                ent = ent  # BufferEntry
                fix = ent.fix_text or f"label_{ent.label}"
                out.append(
                    {
                        "label": ent.label,
                        "era": ent.era,
                        "distance": dist,
                        "similarity_score": sim,
                        "rank_score": rank,
                        "incident_id": ent.incident_id,
                        "incident_type": ent.incident_type,
                        "fix": fix,
                        "timestamp": ent.timestamp,
                        "features_preview": ent.features[:8].tolist(),
                        "memory_tier": ent.memory_tier,
                    }
                )
            return out

    def predict_insight(
        self,
        features: list[float],
        *,
        k_neighbors: int = 5,
        include_forgetting: bool = True,
        include_llm: bool = False,
        incident_text: str | None = None,
        memory_tier_filter: set[str] | None = None,
    ) -> dict:
        pred = self.predict_one(features)
        cid = int(pred["predicted_class"])
        conf = float(pred["confidence"])
        neigh_raw = self.similar_incidents(
            features,
            k=k_neighbors,
            memory_tier_filter=memory_tier_filter,
        )
        sims = np.array(
            [float(m["similarity_score"]) for m in neigh_raw],
            dtype=np.float64,
        )
        if sims.size == 0:
            weights = sims
        else:
            expw = np.exp(sims - sims.max())
            weights = expw / (expw.sum() + 1e-12)
        neighbor_vote: dict[str, int] = {}
        for m in neigh_raw:
            key = str(int(m["label"]))
            neighbor_vote[key] = neighbor_vote.get(key, 0) + 1
        fix_scores: dict[str, float] = {}
        for i, m in enumerate(neigh_raw):
            fx = str(m.get("fix") or "").strip() or f"label_{m['label']}"
            w = float(weights[i]) if len(weights) > i else 1.0 / max(len(neigh_raw), 1)
            fix_scores[fx] = fix_scores.get(fx, 0.0) + w
        suggested_fix = ""
        if fix_scores:
            suggested_fix = max(
                fix_scores.keys(), key=lambda fk: fix_scores[fk]
            )
        neighbors = []
        for i, m in enumerate(neigh_raw):
            neighbors.append(
                {
                    "incident_id": m.get("incident_id"),
                    "label": int(m["label"]),
                    "era": int(m["era"]),
                    "distance": float(m["distance"]),
                    "similarity_score": float(m["similarity_score"]),
                    "incident_type": str(m.get("incident_type") or ""),
                    "fix": str(m.get("fix") or ""),
                    "weight": float(weights[i]) if len(weights) > i else 0.0,
                    "memory_tier": str(m.get("memory_tier") or "short_term"),
                }
            )
        forgetting_warning = None
        if include_forgetting:
            forgetting_warning = self.forgetting_alert_payload()
        llm_block = None
        if include_llm:
            sim_lines = [
                f"{n['incident_type']}: {n['fix']}"[:500] for n in neighbors
            ]
            summ, hyp, sug, prov = explain_incident(
                incident_text=incident_text or "",
                predicted_class=cid,
                confidence=conf,
                similar_lines=sim_lines,
                cfg=load_llm_config(),
            )
            llm_block = {
                "summary": summ,
                "hypothesis": hyp,
                "suggested_fix": sug,
                "provider": prov,
            }
        return {
            "prediction": {
                "class_id": cid,
                "class_name": self.class_name_for(cid),
                "confidence": conf,
            },
            "similar_incidents": neighbors,
            "neighbor_vote": neighbor_vote,
            "suggested_fix": suggested_fix,
            "forgetting_warning": forgetting_warning,
            "explainability": {
                "neighbor_weights": [float(w) for w in weights.tolist()]
                if weights.size
                else [],
                "mlp_saliency": None,
            },
            "llm": llm_block,
        }

    def run_synthetic_era_simulation(
        self,
        steps_per_era: int = 80,
        num_eras: int | None = None,
    ) -> dict:
        n_eras = num_eras if num_eras is not None else self.cfg.num_eras
        self.reset()
        cfg = self.cfg
        assert self._rng is not None
        assert self._class_means is not None and self._era_drifts is not None

        for era in range(n_eras):
            for _ in range(steps_per_era):
                feats, labs = self.sample_synthetic_batch(era, cfg.batch_size)
                self.train_batch(era, feats, labs)
            self.close_era(era)

        return {
            "simulation": {
                "num_eras": n_eras,
                "steps_per_era": steps_per_era,
            },
            "metrics": self.metrics_payload(),
        }
