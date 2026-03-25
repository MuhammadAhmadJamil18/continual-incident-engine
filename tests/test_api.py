from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def test_demo_synthetic_batch(client: TestClient) -> None:
    r = client.get("/demo/synthetic-batch", params={"era": 0, "n": 4})
    assert r.status_code == 200
    data = r.json()
    assert data["era"] == 0
    assert len(data["incidents"]) == 4
    assert all("label" in x for x in data["incidents"])


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["feature_dim"] > 0


def test_synthetic_batch_train_close_metrics(client: TestClient) -> None:
    """Incremental path: synthetic batch -> train -> close era -> metrics populated."""
    r0 = client.get("/demo/synthetic-batch", params={"era": 0, "n": 16})
    assert r0.status_code == 200
    batch = r0.json()["incidents"]
    tr = client.post("/train", json={"era": 0, "incidents": batch})
    assert tr.status_code == 200
    assert "loss" in tr.json()
    cl = client.post("/era/close", json={"era": 0})
    assert cl.status_code == 200
    m = client.get("/metrics").json()
    assert len(m["accuracy_matrix"]) == 1
    a = client.get("/forgetting-alert").json()
    assert a["risk_level"] in ("low", "medium", "high")


def test_predict_bad_feature_dim(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={"incident": {"features": [0.0, 1.0]}},
    )
    assert r.status_code == 400


def test_simulation_run(client: TestClient) -> None:
    r = client.post(
        "/simulation/run",
        json={"steps_per_era": 3, "num_eras": 2},
    )
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert len(metrics["accuracy_matrix"]) == 2


def test_engine_reset(client: TestClient) -> None:
    client.post("/simulation/run", json={"steps_per_era": 2, "num_eras": 2})
    r = client.post("/engine/reset")
    assert r.status_code == 200
    m = client.get("/metrics").json()
    assert m["accuracy_matrix"] == []


def test_health_unauthenticated_without_api_keys(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200


def test_data_status(client: TestClient) -> None:
    r = client.get("/data/status")
    assert r.status_code == 200
    body = r.json()
    assert "buffer_samples_per_era" in body
    assert "github_file_samples_per_era" in body


def test_experiment_github_replay(client: TestClient, tmp_path: Path, engine) -> None:
    """Full era replay mutates engine; uses tiny JSON (no GitHub)."""
    p = tmp_path / "issues.json"
    samples = []
    for era in (0, 1):
        for i in range(6):
            samples.append(
                {
                    "text": f"incident wordbucket {era} {i} " * 6,
                    "label": i % 4,
                    "era": era,
                    "source": "test",
                }
            )
    p.write_text(json.dumps({"samples": samples, "label_map": {}}), encoding="utf-8")
    r = client.post(
        "/experiment/github-replay",
        json={
            "data_path": str(p.resolve()),
            "chunk_size": 8,
            "reset_engine_first": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    data = body["data"]
    # Holdout split reserves ~12% per era for eval (not counted as trained rows).
    assert data["samples_trained"] == 10
    assert data["metrics"]["accuracy_matrix"]


def test_api_key_enforcement(monkeypatch, client: TestClient) -> None:
    monkeypatch.setenv("IME_API_KEYS", "unit-test-key")
    r = client.get("/metrics")
    assert r.status_code == 401
    r2 = client.get("/metrics", headers={"X-API-Key": "unit-test-key"})
    assert r2.status_code == 200


def test_ingest_text_envelope(client: TestClient) -> None:
    body = {
        "texts": ["incident alpha", "incident beta"],
        "era": 0,
        "train": False,
        "return_embeddings": True,
    }
    r = client.post("/ingest/text", json=body)
    assert r.status_code == 200
    env = r.json()
    assert env["ok"] is True
    assert env["data"]["count"] == 2
    assert env["data"]["embeddings"] is not None


def test_ingest_text_train(client: TestClient) -> None:
    body = {
        "texts": ["log line one", "log line two"],
        "labels": [0, 1],
        "era": 0,
        "train": True,
        "return_embeddings": False,
        "memory_tier": "long_term",
    }
    r = client.post("/ingest/text", json=body)
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["trained"] is True
    assert d["loss"] is not None


def test_drift_envelope(client: TestClient) -> None:
    r = client.get("/drift")
    assert r.status_code == 200
    env = r.json()
    assert env["ok"] is True
    assert "score" in env["data"]


def test_predict_insight_envelope(client: TestClient) -> None:
    client.post("/engine/reset")
    r0 = client.get("/demo/synthetic-batch", params={"era": 0, "n": 8})
    assert r0.status_code == 200
    batch = r0.json()["incidents"]
    client.post("/train", json={"era": 0, "incidents": batch})
    feats = batch[0]["features"]
    r = client.post(
        "/predict/insight",
        json={
            "incident": {"features": feats, "label": 0},
            "k_neighbors": 3,
            "include_forgetting": True,
            "include_llm": False,
        },
    )
    assert r.status_code == 200
    env = r.json()
    assert env["ok"] is True
    assert "prediction" in env["data"]
    assert "similar_incidents" in env["data"]
