"""
Checkpoint I/O for IncidentMemoryEngine.

Uses a versioned split layout under the parent directory of ``state_path``:
``engine_meta.json`` (config, metrics, RNG, pipeline state, optional label maps),
``engine_weights.pt`` (Torch model + optimizer state only),
``buffer.json`` (replay buffer rows).

Legacy single-file ``*.pkl`` checkpoints are still loaded when present so existing
deployments do not break; the next save migrates to the new layout.
"""

from __future__ import annotations

import base64
import json
import logging
import pickle
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from incident_memory_engine.buffer.replay_buffer import BufferEntry, ReplayBuffer
from incident_memory_engine.config import EngineConfig

logger = logging.getLogger(__name__)

# Legacy monolithic pickle
STATE_VERSION_PICKLE = 2
# Split layout (meta JSON + weights .pt + buffer JSON)
ENGINE_FORMAT_VERSION = 3
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def resolve_artifact_paths(state_path: str | Path) -> tuple[Path, Path, Path]:
    """
    Map ``EngineConfig.state_path`` to the three artifact files.

    All live in the same directory as ``state_path`` (typically ``artifacts/``).
    """
    p = Path(state_path)
    parent = p.parent
    return (
        parent / "engine_meta.json",
        parent / "engine_weights.pt",
        parent / "buffer.json",
    )


def _entry_to_dict(e: BufferEntry) -> dict[str, Any]:
    return {
        "features": e.features.astype(float).tolist(),
        "label": e.label,
        "era": e.era,
        "incident_id": e.incident_id,
        "timestamp": e.timestamp,
        "incident_type": e.incident_type,
        "fix_text": e.fix_text,
        "memory_tier": e.memory_tier,
    }


def entry_from_dict(d: dict[str, Any]) -> BufferEntry:
    """Restore a buffer entry from a checkpoint dict."""
    return BufferEntry(
        features=np.asarray(d["features"], dtype=np.float32),
        label=int(d["label"]),
        era=int(d["era"]),
        incident_id=d.get("incident_id"),
        timestamp=float(d.get("timestamp", time.time())),
        incident_type=str(d.get("incident_type", "")),
        fix_text=str(d.get("fix_text", "")),
        memory_tier=str(d.get("memory_tier", "short_term")),
    )


def _jsonify_pipeline_value(v: Any) -> Any:
    if isinstance(v, bytes):
        return {"__b64__": base64.standard_b64encode(v).decode("ascii")}
    if isinstance(v, dict):
        return {k: _jsonify_pipeline_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jsonify_pipeline_value(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _dejsonify_pipeline_value(v: Any) -> Any:
    if isinstance(v, dict) and "__b64__" in v and len(v) == 1:
        return base64.standard_b64decode(v["__b64__"].encode("ascii"))
    if isinstance(v, dict):
        return {k: _dejsonify_pipeline_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_dejsonify_pipeline_value(x) for x in v]
    return v


def _cfg_payload(cfg: EngineConfig) -> dict[str, Any]:
    return {
        "feature_dim": cfg.feature_dim,
        "num_classes": cfg.num_classes,
        "num_eras": cfg.num_eras,
        "test_samples_per_era": cfg.test_samples_per_era,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "hidden_dim": cfg.hidden_dim,
        "replay_capacity": cfg.replay_capacity,
        "replay_batch_ratio": cfg.replay_batch_ratio,
        "seed": cfg.seed,
        "forgetting_alert_low": cfg.forgetting_alert_low,
        "forgetting_alert_medium": cfg.forgetting_alert_medium,
        "state_path": cfg.state_path,
        "encoder_kind": cfg.encoder_kind,
        "persistence_enabled": cfg.persistence_enabled,
        "similarity_recency_half_life_s": cfg.similarity_recency_half_life_s,
        "forgetting_class_drop_ratio": cfg.forgetting_class_drop_ratio,
        "forgetting_era_drop_ratio": cfg.forgetting_era_drop_ratio,
        "incident_class_names": list(cfg.incident_class_names),
        "replay_tier_weight_short": cfg.replay_tier_weight_short,
        "replay_tier_weight_long": cfg.replay_tier_weight_long,
        "replay_tier_weight_critical": cfg.replay_tier_weight_critical,
        "faiss_similarity_pool_multiplier": cfg.faiss_similarity_pool_multiplier,
        "ewc_lambda": cfg.ewc_lambda,
        "ewc_fisher_batches": cfg.ewc_fisher_batches,
        "ewc_fisher_ema": cfg.ewc_fisher_ema,
    }


def save_engine_state(
    path: Path,
    *,
    cfg: EngineConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    pipeline: Any,
    acc_matrix: list[dict[int, float]],
    legacy_accuracy_history: list[float],
    peak_legacy_accuracy: float | None,
    highest_closed_era: int,
    current_era: int,
    buffer_history_snapshots: list[dict[int, int]],
    last_legacy_per_class: dict[int, float],
    peak_legacy_per_class: dict[int, float],
    rng_bit_generator: np.random.BitGenerator,
    model_in_dim: int,
    class_means: np.ndarray,
    era_drifts: np.ndarray,
    github_label_map: dict[str, Any] | None = None,
    ewc_fisher_meta: dict[str, Any] | None = None,
    ewc_anchor_meta: dict[str, Any] | None = None,
) -> None:
    """Write split checkpoint (JSON + torch + buffer JSON) next to ``state_path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_p, weights_p, buffer_p = resolve_artifact_paths(path)

    ps = pipeline.state_dict()
    pipeline_json = _jsonify_pipeline_value(ps)

    meta: dict[str, Any] = {
        "format_version": ENGINE_FORMAT_VERSION,
        "saved_at": time.time(),
        "cfg": _cfg_payload(cfg),
        "model_in_dim": model_in_dim,
        "class_means": np.asarray(class_means, dtype=np.float32).tolist(),
        "era_drifts": np.asarray(era_drifts, dtype=np.float32).tolist(),
        "pipeline_state": pipeline_json,
        "acc_matrix": acc_matrix,
        "legacy_accuracy_history": legacy_accuracy_history,
        "peak_legacy_accuracy": peak_legacy_accuracy,
        "highest_closed_era": highest_closed_era,
        "current_era": current_era,
        "buffer_history_snapshots": buffer_history_snapshots,
        "last_legacy_per_class": {str(k): float(v) for k, v in last_legacy_per_class.items()},
        "peak_legacy_per_class": {str(k): float(v) for k, v in peak_legacy_per_class.items()},
        "rng_state_b64": base64.standard_b64encode(
            pickle.dumps(rng_bit_generator.state, protocol=PICKLE_PROTOCOL)
        ).decode("ascii"),
    }
    if github_label_map is not None:
        meta["github_label_map"] = github_label_map
    if ewc_fisher_meta:
        meta["ewc_fisher"] = ewc_fisher_meta
    if ewc_anchor_meta:
        meta["ewc_anchor"] = ewc_anchor_meta

    tmp_meta = meta_p.with_suffix(".json.tmp")
    with tmp_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    tmp_meta.replace(meta_p)

    tmp_w = weights_p.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
        },
        tmp_w,
    )
    tmp_w.replace(weights_p)

    buf_entries = [_entry_to_dict(e) for e in buffer._entries]
    tmp_b = buffer_p.with_suffix(".json.tmp")
    with tmp_b.open("w", encoding="utf-8") as f:
        json.dump(buf_entries, f)
    tmp_b.replace(buffer_p)

    # Remove legacy pickle if it existed (migration)
    legacy = path if path.suffix == ".pkl" else path.with_suffix(".pkl")
    if legacy.is_file():
        try:
            legacy.unlink()
        except OSError:
            pass


def _load_modern(meta_p: Path, weights_p: Path, buffer_p: Path) -> dict[str, Any] | None:
    try:
        with meta_p.open(encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    fv = int(meta.get("format_version", 0))
    if fv != ENGINE_FORMAT_VERSION:
        logger.warning(
            "Checkpoint format_version %s != %s; skipping load (will reset).",
            fv,
            ENGINE_FORMAT_VERSION,
        )
        return None
    if not weights_p.is_file() or not buffer_p.is_file():
        return None
    try:
        try:
            bundle = torch.load(weights_p, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(weights_p, map_location="cpu")
        model_sd = bundle["model"]
        opt_sd = bundle["optimizer"]
    except Exception:
        return None
    try:
        with buffer_p.open(encoding="utf-8") as f:
            raw_entries = json.load(f)
    except Exception:
        return None

    ps = _dejsonify_pipeline_value(meta["pipeline_state"])
    rng_blob = base64.standard_b64decode(meta["rng_state_b64"].encode("ascii"))
    rng_state = pickle.loads(rng_blob)

    cfg = config_from_payload(meta["cfg"])
    class_means = np.asarray(meta["class_means"], dtype=np.float32)
    era_drifts = np.asarray(meta["era_drifts"], dtype=np.float32)

    return {
        "version": 3,
        "cfg": meta["cfg"],
        "model_in_dim": int(meta.get("model_in_dim", cfg.feature_dim)),
        "class_means": class_means,
        "era_drifts": era_drifts,
        "model_state": model_sd,
        "optimizer_state": opt_sd,
        "buffer_entries": raw_entries,
        "pipeline_state": ps,
        "acc_matrix": meta["acc_matrix"],
        "legacy_accuracy_history": list(meta.get("legacy_accuracy_history", [])),
        "peak_legacy_accuracy": meta.get("peak_legacy_accuracy"),
        "highest_closed_era": int(meta.get("highest_closed_era", -1)),
        "current_era": int(meta.get("current_era", 0)),
        "buffer_history_snapshots": list(meta.get("buffer_history_snapshots", [])),
        "last_legacy_per_class": {
            int(k): float(v) for k, v in meta.get("last_legacy_per_class", {}).items()
        },
        "peak_legacy_per_class": {
            int(k): float(v) for k, v in meta.get("peak_legacy_per_class", {}).items()
        },
        "rng_state": rng_state,
        "github_label_map": meta.get("github_label_map"),
        "ewc_fisher": meta.get("ewc_fisher"),
        "ewc_anchor": meta.get("ewc_anchor"),
    }


def _load_legacy_pickle(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def load_engine_state(path: Path) -> dict[str, Any] | None:
    """
    Load checkpoint: prefer split layout; fall back to legacy ``.pkl``.

    Returns a dict compatible with ``IncidentMemoryEngine._try_load_checkpoint``.
    """
    path = Path(path)
    meta_p, weights_p, buffer_p = resolve_artifact_paths(path)
    if meta_p.is_file():
        loaded = _load_modern(meta_p, weights_p, buffer_p)
        if loaded is not None:
            return loaded
    pkl_path = path if path.suffix == ".pkl" else path.with_suffix(".pkl")
    legacy = _load_legacy_pickle(pkl_path)
    if legacy is not None and legacy.get("version") == STATE_VERSION_PICKLE:
        return legacy
    return None


def config_from_payload(cfg_dict: dict[str, Any]) -> EngineConfig:
    """Rebuild ``EngineConfig`` with defaults for missing keys (backward compatible)."""
    base = EngineConfig()
    merged: dict[str, Any] = {}
    for f in fields(EngineConfig):
        merged[f.name] = getattr(base, f.name)
    for k, v in cfg_dict.items():
        if k not in merged:
            continue
        if k == "incident_class_names" and isinstance(v, list):
            merged[k] = tuple(v)
        else:
            merged[k] = v
    return EngineConfig(**merged)


def delete_checkpoint(path: Path) -> None:
    """Remove split artifacts and legacy pickle under the same directory."""
    path = Path(path)
    meta_p, weights_p, buffer_p = resolve_artifact_paths(path)
    for p in (meta_p, weights_p, buffer_p):
        if p.is_file():
            p.unlink()
    for legacy in (path, path.with_suffix(".pkl")):
        if legacy.is_file():
            legacy.unlink()
