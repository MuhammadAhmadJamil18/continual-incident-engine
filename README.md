# Incident Memory Engine

Production-style **continual learning** service for streaming **incident** classification: **bounded replay**, optional **Elastic Weight Consolidation (EWC)**, **era-based drift**, **CL metrics** (accuracy matrix, BWT, mean forgetting, legacy era-0 tracking), and a **forgetting alert** derived from legacy holdout accuracy. See [`docs/CONTINUAL_LEARNING.md`](docs/CONTINUAL_LEARNING.md) for the CL stack and era semantics.

## Architecture (strict layers)

| Layer | Path | Role |
|--------|------|------|
| API | [`src/incident_memory_engine/api/`](src/incident_memory_engine/api/) | FastAPI + Pydantic validation |
| Core | [`src/incident_memory_engine/core/`](src/incident_memory_engine/core/) | `IncidentMemoryEngine`, MLP, synthetic era stream |
| Buffer | [`src/incident_memory_engine/buffer/`](src/incident_memory_engine/buffer/) | Reservoir replay + similarity search |
| Metrics | [`src/incident_memory_engine/metrics/`](src/incident_memory_engine/metrics/) | BWT / forgetting / alert logic |
| UI | [`ui/dashboard.py`](ui/dashboard.py) | Streamlit **client only** (HTTP; no torch imports) |
| Tests | [`tests/`](tests/) | Pytest (buffer, metrics, engine, API) |
| CL reference | [`docs/CONTINUAL_LEARNING.md`](docs/CONTINUAL_LEARNING.md) | Replay, EWC, eras E0/E1/…, metrics |

Legacy offline benchmark (naive vs replay): [`src/incident_cl/`](src/incident_cl/) + [`streamlit_app.py`](streamlit_app.py).

---

## Completion guide (session by session)

| Session | What to do | Done when |
|--------|------------|-----------|
| **1 — Environment** | `python -m venv .venv`, `pip install -e ".[dev]"` | `pytest` passes |
| **2 — API** | `uvicorn incident_memory_engine.api.app:app --port 8000` or `ime-api` | `GET /health` returns `ok` |
| **3 — Dashboard** | `streamlit run ui/dashboard.py` | Overview simulation + forgetting banner work |
| **4 — Incremental CL** | Tab *Incremental lab*: synthetic batch → train → `era/close` | Matrix grows row-by-row |
| **5 — Ship** | Optional: `docker compose up --build` | API on port 8000 from container |

---

## Requirements

- Python **3.11+**
- PyTorch (CPU or CUDA)

## Setup

```bash
cd continual
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

## Replay vs naive finetuning (metrics for demos)

```bash
python scripts/run_replay_ablation.py
```

Writes **`artifacts/baseline_comparison.json`**: same synthetic run **with** reservoir replay vs **`replay_enabled=false`** (gradient on current batch only). Use `caption_metrics` / `suggested_one_liner_filled` for posts; re-run if you change `--steps`, `--eras`, or `--seed`.

## Run the API (Uvicorn)

```powershell
.\scripts\run_api.ps1
```

Or:

```bash
ime-api
# or
uvicorn incident_memory_engine.api.app:app --reload --host 127.0.0.1 --port 8000
```

## Run the dashboard (Streamlit)

Second terminal (API must be running):

```powershell
.\scripts\run_ui.ps1
```

Or:

```bash
streamlit run ui/dashboard.py
```

## Docker (API + interactive UI)

```bash
docker compose up --build
```

- **API / OpenAPI:** `http://127.0.0.1:8000` — docs at `/docs`
- **Streamlit (portfolio / LinkedIn demo):** `http://127.0.0.1:8501` — start on the **Why & share** tab for purpose + copy-ready blurb; **Home** runs a one-click simulation; **Predict & similar** is the interactive proof. The UI container sets **`IME_API_BASE=http://api:8000`** so it talks to the API inside Compose.

Local UI only: `streamlit run ui/dashboard.py` (API on `8000` first). Override the API URL with env **`IME_API_BASE`** when the browser’s Streamlit server must reach a different host.

## Persistence

- `EngineConfig.state_path` still names the logical checkpoint (often `artifacts/engine_state.pkl`); on disk the engine writes **three** files next to it in the same directory:
  - **`engine_meta.json`** — format version, full `EngineConfig` fields, accuracy matrix, legacy history, stream centroids, RNG blob, pipeline/encoder state, optional GitHub label map.
  - **`engine_weights.pt`** — `torch.save` of **model** + **optimizer** state dicts only.
  - **`buffer.json`** — replay buffer rows (features as lists, labels, era, metadata).
- If `format_version` in the meta file does not match the runtime, load is **skipped** with a warning and the engine starts fresh (no crash).
- A legacy **single `.pkl`** checkpoint is still **loaded** if present and no split layout exists; the next save migrates to the split layout and removes the pickle.
- `POST /engine/reset` deletes the split files **and** any legacy `.pkl` when persistence is on.

## Text / embeddings (`core/feature_pipeline.py`)

- `EngineConfig.encoder_kind`: **`hashing`** (default for text-friendly demos), **`identity`** (precomputed vectors only), **`tfidf`**, or **`sentence`** (requires `pip install -e ".[sentence]"` and a `feature_dim` that matches the sentence model output size).
- API: send either **`features`** or **`text`** on each incident (`IncidentVector`), not both. Batches must be all-text or all-vector.

### Real log / webhook-style ingest (recommended defaults)

- For **raw log lines** without hand-built vectors, use **`encoder_kind=hashing`** with default **`feature_dim=64`** (good default for noisy text + small MLP).
- For **semantic similarity**, use **`encoder_kind=sentence`** with `feature_dim` equal to the embedding size of your chosen model (install optional extra **`sentence`**).
- Text-first routes: **`POST /ingest/text`** (JSON body) and **`POST /ingest/batch`** (multipart CSV/JSON).
- **GitHub Issues (public Search API):** CLI `python -m incident_memory_engine.data.github_ingest …` → `artifacts/github_issues.json`, or **`POST /data/github/download`** (same ingest, writes JSON). Optional **`GITHUB_TOKEN`** for rate limits. **`POST /ingest/github`** fetches and trains in one step; **`POST /experiment/github-replay`** replays a saved JSON through train + era-close on the **live** engine (dashboard uses both). **`GET /data/status`** shows buffer vs file counts per era.
- Offline CL run: `python scripts/run_real_experiment.py` (writes `artifacts/real_experiment_results.json` and `artifacts/forgetting_curve.png`).

## Environment variables (API / ops)

| Variable | Purpose |
|----------|---------|
| `IME_API_BASE` | **Streamlit only:** default API base URL (e.g. `http://api:8000` in Docker). Lets a deployed UI find the backend without manual sidebar edits. |
| `IME_API_KEYS` | Comma-separated keys; if set, protected routes require `X-API-Key` or `Authorization: Bearer …`. **`/health` stays public.** |
| `IME_RATE_LIMIT_PER_MINUTE` | Per-key (or per-IP if no key) request cap for protected routes (default `120`). |
| `IME_EWC_LAMBDA` | If set, enables/overrides **EWC** strength (`EngineConfig.ewc_lambda`) at API startup for continual regularization alongside replay. |
| `IME_MAX_INGEST_ROWS` | Row cap for `POST /ingest/batch` (default `20000`). |
| `IME_MAX_UPLOAD_BYTES` | Upload size cap for batch ingest (default 20 MiB). |
| `IME_HIDE_DOCS` | Set to `1` / `true` to disable `/docs`, `/redoc`, OpenAPI JSON. |
| `IME_AUTO_ERA_CLOSE_ON_DRIFT` | If `1` / `true`, **`GET /drift`** may call **`POST /era/close`** once when drift score crosses a high threshold (off by default). |
| `OPENAI_API_KEY` | Optional; enables LLM copy in **`POST /assist/explain`** and **`POST /predict/insight`** when `include_llm: true` (OpenAI-compatible API). |
| `OPENAI_BASE_URL` | Optional override (default OpenAI). |
| `IME_LLM_MODEL` | Optional model id (default `gpt-4o-mini`). |

## HTTP endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/demo/synthetic-batch` | Labeled batch from synthetic stream (`era`, `n`) for `POST /train` |
| GET | `/health` | Status + config summary (**no API key**) |
| POST | `/ingest/text` | Encode `texts[]`; optional `train` + `labels[]`; envelope JSON |
| POST | `/ingest/batch` | Multipart `.csv` / `.json`; chunked train; optional `close_era_at_end` |
| POST | `/ingest/github` | Body: `repos`, `era`, `per_repo` — fetch issues, encode, train (envelope) |
| POST | `/data/github/download` | Body: `per_era`, optional `output_path` — ingest to JSON only (envelope) |
| POST | `/experiment/github-replay` | Body: optional `data_path`, `chunk_size`, `reset_engine_first` — full replay → `/metrics` |
| GET | `/data/status` | Buffer + optional `github_issues.json` sample counts per era |
| POST | `/train` | Incremental step: labeled batch + replay mix + buffer insert |
| POST | `/era/close` | Holdout eval for eras `0..era`; extends accuracy matrix |
| POST | `/predict` | Class + softmax confidence |
| POST | `/predict/insight` | Product-shaped bundle: `prediction`, `class_name`, neighbors, `suggested_fix`, optional forgetting + LLM |
| GET | `/metrics` | Accuracy matrix, BWT, mean forgetting, legacy history |
| GET | `/forgetting-alert` | Structured JSON: `risk_level`, `message`, `drop_percentage`, … |
| GET | `/drift` | Embedding norm drift snapshot + recommendation; optional auto era-close |
| POST | `/assist/explain` | Optional OpenAI-compatible explanation (no-key → structured fallback) |
| POST | `/similar` | FAISS-accelerated retrieval + recency rerank; optional `memory_tiers` filter |
| POST | `/simulation/run` | Reset + full synthetic era run (demo orchestration) |
| POST | `/engine/reset` | Clear model, buffer, metrics |

New JSON routes return an envelope: `{ "ok", "data", "error", "meta": { "request_id", "version" } }`.

Optional: `POST /simulation/run` body `{"persist_path": "artifacts/run.json"}` writes metrics JSON (ignored by git via `.gitignore`).

**FAISS:** the replay buffer remains the source of truth; a **flat L2 FAISS index** is rebuilt after training loads so `/similar` avoids scanning huge buffers. Optional future: Pinecone via the same `VectorIndex` port.

## Incremental training flow (custom stream)

1. Obtain batches (your pipeline or `GET /demo/synthetic-batch`).
2. `POST /train` with `{ "era": 0, "incidents": [ { "features": [...], "label": 3 }, ... ] }` (repeat for steps).
3. `POST /era/close` with `{ "era": 0 }`.
4. Advance era and repeat.

Feature vector length must match `EngineConfig.feature_dim` (default **64**).

