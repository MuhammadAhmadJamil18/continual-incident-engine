#!/usr/bin/env python3
"""
Fetch **1500** public GitHub issues (500 × 3 eras / repos) and compare **replay vs no replay**
on the same file with identical holdout splits.

Requires network. Set ``GITHUB_TOKEN`` for reliable rate limits.

Usage (repo root, package on PYTHONPATH or ``pip install -e .``)::

    python scripts/run_github_1500_ablation.py

Skip re-download if JSON already exists::

    python scripts/run_github_1500_ablation.py --skip-fetch

Outputs:

- ``artifacts/github_issues_1500.json`` — full dataset (gitignored by default).
- ``artifacts/github_replay_ablation_1500.json`` — metrics + caption fields (commit this).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Repo root on path for ``python scripts/...`` without install
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from incident_memory_engine.config import EngineConfig
from incident_memory_engine.core.engine import IncidentMemoryEngine
from incident_memory_engine.data.github_ingest import (
    DEFAULT_ERA_REPOS,
    ingest_eras,
    save_artifacts,
)


def _pct_drop_from_peak(legacy_hist: list[float], peak: float | None) -> float | None:
    if not legacy_hist or peak is None or peak <= 0:
        return None
    last = float(legacy_hist[-1])
    return round((float(peak) - last) / float(peak) * 100.0, 2)


def _run_github_file(
    *,
    data_path: Path,
    replay_enabled: bool,
    chunk_size: int,
) -> dict[str, Any]:
    cfg = EngineConfig(
        persistence_enabled=False,
        ewc_lambda=0.0,
        replay_enabled=replay_enabled,
        num_eras=5,
        encoder_kind="hashing",
    )
    eng = IncidentMemoryEngine(cfg)
    out = eng.run_github_file_experiment(
        data_path,
        chunk_size=chunk_size,
        reset_first=True,
    )
    m = out["metrics"]
    return {
        "samples_trained": out.get("samples_trained"),
        "eras_closed": out.get("eras_closed"),
        "legacy_accuracy_history": list(m.get("legacy_accuracy_history") or []),
        "peak_legacy_accuracy": m.get("peak_legacy_accuracy"),
        "current_legacy_accuracy": m.get("current_legacy_accuracy"),
        "mean_forgetting": m.get("mean_forgetting"),
        "bwt": m.get("bwt"),
        "avg_acc_all_seen": m.get("avg_acc_all_seen"),
        "accuracy_matrix": m.get("accuracy_matrix"),
        "forgetting_alert": out.get("forgetting_alert"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="1500-issue GitHub replay ablation")
    p.add_argument("--per-era", type=int, default=500, help="Issues per era (3 eras → 1500 total)")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument(
        "--data-path",
        type=Path,
        default=_ROOT / "artifacts" / "github_issues_1500.json",
    )
    p.add_argument(
        "--results-path",
        type=Path,
        default=_ROOT / "artifacts" / "github_replay_ablation_1500.json",
    )
    p.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Use existing --data-path JSON (must already exist)",
    )
    args = p.parse_args()

    data_path: Path = args.data_path
    token = os.environ.get("GITHUB_TOKEN")

    if not args.skip_fetch:
        era_map = {0: DEFAULT_ERA_REPOS[0], 1: DEFAULT_ERA_REPOS[1], 2: DEFAULT_ERA_REPOS[2]}
        print(
            f"Ingesting up to {args.per_era} issues × 3 repos (eras 0–2)… "
            f"token={'set' if token else 'not set (anonymous rate limit)'}"
        )
        rows, meta = ingest_eras(era_map, args.per_era, token=token)
        save_artifacts(rows, meta, data_path)
        print(f"Saved {len(rows)} samples to {data_path}")
    elif not data_path.is_file():
        raise SystemExit(f"--skip-fetch but missing file: {data_path}")

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    n_samples = len(raw.get("samples", []))
    print(f"Running paired experiments on {n_samples} samples…")

    with_r = _run_github_file(
        data_path=data_path,
        replay_enabled=True,
        chunk_size=args.chunk_size,
    )
    no_r = _run_github_file(
        data_path=data_path,
        replay_enabled=False,
        chunk_size=args.chunk_size,
    )

    leg_w = with_r["legacy_accuracy_history"]
    leg_n = no_r["legacy_accuracy_history"]
    pk_w = with_r.get("peak_legacy_accuracy")
    pk_n = no_r.get("peak_legacy_accuracy")

    idx_final = min(len(leg_w), len(leg_n)) - 1
    legacy_final_w = leg_w[idx_final] if idx_final >= 0 else None
    legacy_final_n = leg_n[idx_final] if idx_final >= 0 else None
    legacy_after_era1_w = leg_w[1] if len(leg_w) > 1 else None
    legacy_after_era1_n = leg_n[1] if len(leg_n) > 1 else None

    pp_gap_final = None
    if legacy_final_w is not None and legacy_final_n is not None:
        pp_gap_final = round((legacy_final_w - legacy_final_n) * 100.0, 2)

    filled = None
    if (
        legacy_after_era1_w is not None
        and legacy_after_era1_n is not None
        and legacy_final_w is not None
        and legacy_final_n is not None
        and pp_gap_final is not None
    ):
        filled = (
            f"GitHub {n_samples} issues (500/era × K8s, Prometheus, Grafana): "
            f"after era 1, legacy E0 holdout {legacy_after_era1_w * 100:.1f}% with replay "
            f"vs {legacy_after_era1_n * 100:.1f}% naive finetuning; "
            f"after era 2, {legacy_final_w * 100:.1f}% vs {legacy_final_n * 100:.1f}% "
            f"({pp_gap_final:+.2f} pp on E0 with replay)."
        )

    out = {
        "experiment": {
            "source": "github_public_issues",
            "per_era_requested": args.per_era,
            "samples_in_file": n_samples,
            "data_path": str(data_path.as_posix()),
            "chunk_size": args.chunk_size,
            "ewc_lambda": 0,
            "encoder_kind": "hashing",
            "github_token_used": bool(token),
        },
        "with_replay": with_r,
        "no_replay": no_r,
        "caption_metrics": {
            "legacy_e0_after_each_era_close_with_replay": leg_w,
            "legacy_e0_after_each_era_close_no_replay": leg_n,
            "drop_from_peak_legacy_pct_with_replay": _pct_drop_from_peak(leg_w, pk_w),
            "drop_from_peak_legacy_pct_no_replay": _pct_drop_from_peak(leg_n, pk_n),
            "mean_forgetting_with_replay": with_r.get("mean_forgetting"),
            "mean_forgetting_no_replay": no_r.get("mean_forgetting"),
            "bwt_with_replay": with_r.get("bwt"),
            "bwt_no_replay": no_r.get("bwt"),
            "legacy_e0_after_era1_close": {
                "with_replay": legacy_after_era1_w,
                "no_replay": legacy_after_era1_n,
            },
            "legacy_e0_final_both_runs": {
                "with_replay": legacy_final_w,
                "no_replay": legacy_final_n,
            },
            "final_legacy_e0_gap_percentage_points": pp_gap_final,
            "suggested_one_liner_filled": filled,
        },
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.results_path}")


if __name__ == "__main__":
    main()
