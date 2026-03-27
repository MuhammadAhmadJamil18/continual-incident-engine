# Continual learning in Incident Memory Engine

This service implements a **standard rehearsal + regularization** stack suitable for demos and production-style APIs, with explicit **era boundaries** and CL metrics.

## What is implemented

### 1. Experience replay (core CL)

- Fixed-capacity **reservoir buffer** stores past incidents (features, label, era, metadata).
- Each training step mixes **new batch** with **replayed** samples (`replay_batch_ratio`, tier-weighted sampling).
- **Location:** `buffer/replay_buffer.py`, `core/engine.py` (`train_batch`).

### 2. Elastic Weight Consolidation (EWC)

- After each **`close_era`**, the engine estimates a **diagonal Fisher** (squared gradients) on a few minibatches from **seen-era holdouts**, merges it with an **EMA** across era boundaries, and stores **anchor weights**.
- On subsequent training, the loss includes  
  \(\frac{\lambda}{2} \sum_i F_i\, (\theta_i - \theta^\star_i)^2\)  
  so updates that would harm previously important weights are penalized.
- **Papers:** Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (PNAS 2017).
- **Config:** `EngineConfig.ewc_lambda` (default `0` = off), `ewc_fisher_batches`, `ewc_fisher_ema`.
- **Environment:** `IME_EWC_LAMBDA` overrides `ewc_lambda` at process start (no code change).
- **Location:** `core/engine.py` (`_ewc_consolidate_after_close`, `_train_step`).

EWC state (Fisher + anchors) is **persisted** in checkpoint meta when present.

### 3. Era semantics (E0, E1, E2, …)

- An **era** is an integer **time slice** you assign to data (API `era` field or JSON `"era"`).
- Training updates the **same shared classifier** for all eras (single MLP, `num_classes` labels).
- **`close_era(k)`** evaluates on holdouts for eras `0 … k`, appends one **row** to the accuracy matrix, updates legacy / forgetting signals, and (if EWC enabled) consolidates Fisher/anchor.
- **E0 legacy** metrics track how well the model still does on **era-0’s holdout** as later eras arrive.

### 4. Metrics and alerts

- **Accuracy matrix, BWT, mean forgetting:** `metrics/cl_metrics.py`.
- **Forgetting alert:** `metrics/forgetting_alert.py`, `GET /forgetting-alert`.
- **`GET /metrics`** includes a `continual_learning` object describing replay settings, `ewc_lambda`, and whether EWC consolidation is active.

### 5. Evaluation alignment (text encoders)

- For **hashing / tf-idf / sentence** encoders, holdouts are built in **encoder output space** so metrics match **text-trained** models. See `core/data_stream.py` and `build_per_era_text_eval_datasets`.
- **GitHub JSON replay** uses a **train/holdout split per era** so test rows come from the **same distribution** as training text.

## What is not implemented

- No separate **per-task heads**, **GEM** gradient projection, **LwF**, **progressive networks**, or full **Bayesian** CL.
- Those can be added behind the same `EngineConfig` / API patterns if needed.

## Suggested tuning

| Goal | Knob |
|------|------|
| Less forgetting | Raise `replay_capacity`, `replay_batch_ratio`, or enable EWC (`ewc_lambda` ≈ 10–50, tune on your stream). |
| Stronger EWC | Increase `ewc_lambda` (watch underfitting on new eras). |
| Cheaper Fisher | Lower `ewc_fisher_batches`. |

Use **replay + EWC together** for the most “textbook” continual-learning story in portfolios and stakeholder reviews.

## Replay vs naive finetuning (caption metrics)

Run a paired synthetic experiment (same seed, same steps per era):

```bash
python scripts/run_replay_ablation.py
```

This writes **`artifacts/baseline_comparison.json`** with:

- Legacy **E0** holdout accuracy after each `close_era`, **with** vs **without** replay (`replay_enabled=False` = no buffer in loss, buffer not filled).
- **Mean forgetting** and **BWT** for both runs.
- **`caption_metrics.suggested_one_liner_filled`** — copy-paste line for posts (re-run after changing `--steps` / `--eras` / `--seed`).

Numbers are **synthetic-stream** baselines; GitHub replay can behave differently. Always cite the JSON path and experiment block when sharing stats.
