#!/usr/bin/env python3
"""
Run a continual-learning experiment on real GitHub Issues JSON (see ``github_ingest``).

Loads ``artifacts/github_issues.json``, trains era-by-era with the same encoder as the API,
closes each era for the accuracy matrix, then writes metrics + forgetting alert to JSON and
plots the forgetting / accuracy-matrix curves to PNG. Intended for portfolio demos (not CI).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repo root on path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from incident_memory_engine.config import EngineConfig
from incident_memory_engine.core.engine import IncidentMemoryEngine
from incident_memory_engine.data.github_ingest import (
    DEFAULT_ERA_REPOS,
    ingest_eras,
    save_artifacts,
)


def _plot_forgetting_curve(metrics: dict, out_png: Path) -> None:
    mat = metrics.get("accuracy_matrix") or []
    if not mat:
        return
    test_eras = sorted(
        {int(k) for row in mat for k in row.keys()},
        key=int,
    )
    plt.figure(figsize=(10, 6))
    for te in test_eras:
        xs: list[int] = []
        ys: list[float] = []
        for ti, row in enumerate(mat):
            sk = str(te)
            if sk in row:
                xs.append(ti)
                ys.append(float(row[sk]))
        if xs:
            plt.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=f"Holdout era {te}",
            )
    plt.xlabel("Train-through era (row index)")
    plt.ylabel("Accuracy")
    plt.title("Forgetting Curve — Real GitHub Issues")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def run_experiment(
    data_path: Path,
    out_json: Path,
    out_png: Path,
    *,
    chunk_size: int = 64,
    fetch_if_missing: bool = False,
    per_era_fetch: int = 300,
) -> dict:
    if not data_path.is_file():
        if not fetch_if_missing:
            raise FileNotFoundError(
                f"Missing {data_path}. Run: python -m incident_memory_engine.data.github_ingest "
                "or pass --fetch"
            )
        token = os.environ.get("GITHUB_TOKEN")
        rows, meta = ingest_eras(DEFAULT_ERA_REPOS, per_era_fetch, token=token)
        save_artifacts(rows, meta, data_path)

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    samples = raw.get("samples", raw if isinstance(raw, list) else [])
    if not samples:
        raise ValueError("No samples in data file")

    eras = sorted({int(s["era"]) for s in samples})
    max_era = max(eras)
    cfg = EngineConfig(
        persistence_enabled=False,
        encoder_kind="hashing",
        num_eras=max(max_era + 2, 5),
        batch_size=min(64, chunk_size),
        replay_capacity=400,
    )
    eng = IncidentMemoryEngine(cfg)
    if isinstance(raw.get("label_map"), dict):
        eng._github_label_map = raw["label_map"]

    for era in eras:
        era_samples = [s for s in samples if int(s["era"]) == era]
        for start in range(0, len(era_samples), chunk_size):
            sub = era_samples[start : start + chunk_size]
            texts = [str(s["text"]) for s in sub]
            labels = [int(s["label"]) for s in sub]
            feats = eng.encode_texts(texts)
            itypes = [str(s.get("source", "github")) for s in sub]
            fixes = [str(s.get("title", ""))[:500] for s in sub]
            eng.train_batch(era, feats, labels, incident_types=itypes, fixes=fixes)
        eng.close_era(era)

    metrics = eng.metrics_payload()
    alert = eng.forgetting_alert_payload()
    bwt = metrics.get("bwt")
    out = {
        "metrics": metrics,
        "forgetting_alert": alert,
        "bwt": bwt,
        "eras_trained": eras,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _plot_forgetting_curve(metrics, out_png)
    return out


def _print_summary_table(payload: dict) -> None:
    m = payload["metrics"]
    alert = payload["forgetting_alert"]
    mat = m.get("accuracy_matrix") or []
    risk = alert.get("risk_level", "n/a")
    bwt = m.get("bwt")
    print(f"Global BWT: {bwt}")
    print(f"Mean forgetting: {m.get('mean_forgetting')}")
    print(f"Forgetting alert risk_level: {risk}")
    print("era | matrix diagonal acc (train i, test i)")
    print("----|----------------------------------------")
    for i, row in enumerate(mat):
        v = row.get(str(i), row.get(i))
        if v is not None:
            print(f" {i}  | {float(v):.4f}")
        else:
            print(f" {i}  | n/a")
    if not mat:
        print("(empty matrix)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Real GitHub issues CL experiment")
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/github_issues.json"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/real_experiment_results.json"),
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=Path("artifacts/forgetting_curve.png"),
    )
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument(
        "--fetch",
        action="store_true",
        help="If data file missing, run GitHub ingest (slow; needs network)",
    )
    ap.add_argument("--per-era", type=int, default=300)
    args = ap.parse_args()
    os.chdir(_ROOT)
    payload = run_experiment(
        args.data,
        args.out_json,
        args.out_png,
        chunk_size=args.chunk_size,
        fetch_if_missing=args.fetch,
        per_era_fetch=args.per_era,
    )
    _print_summary_table(payload)
    print(f"Wrote {args.out_json} and {args.out_png}")


if __name__ == "__main__":
    main()
