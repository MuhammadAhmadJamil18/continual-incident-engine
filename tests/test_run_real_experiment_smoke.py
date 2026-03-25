"""Smoke test for scripts/run_real_experiment.py (no GitHub network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_run_experiment_produces_json_and_png(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "ime_run_exp",
        root / "scripts" / "run_real_experiment.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = {
        "samples": [
            {
                "text": "pod crash loop backoff " * 4,
                "label": 0,
                "era": 0,
                "source": "test",
            },
            {
                "text": "scrape interval too high " * 4,
                "label": 4,
                "era": 1,
                "source": "test",
            },
        ],
        "label_map": {},
    }
    jp = tmp_path / "gh.json"
    jp.write_text(json.dumps(data), encoding="utf-8")
    out_j = tmp_path / "res.json"
    out_p = tmp_path / "curve.png"
    mod.run_experiment(jp, out_j, out_p, chunk_size=8, fetch_if_missing=False)
    assert out_j.is_file()
    assert out_p.is_file()
    payload = json.loads(out_j.read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert "forgetting_alert" in payload
