from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMConfig:
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_s: float = 30.0
    max_tokens: int = 400


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("IME_LLM_MODEL", "gpt-4o-mini"),
    )


def explain_incident(
    *,
    incident_text: str,
    predicted_class: int | None,
    confidence: float | None,
    similar_lines: list[str],
    cfg: LLMConfig | None = None,
) -> tuple[str | None, str | None, str | None, str]:
    """
    Returns (summary, hypothesis, suggested_fix, provider).

    If no API key, returns (None, None, None, "none").
    """
    cfg = cfg or load_llm_config()
    if not cfg.api_key:
        return None, None, None, "none"

    sys_prompt = (
        "You are an SRE assistant. Given an incident and similar past cases, "
        "output concise JSON with keys: summary (string), hypothesis (string), "
        "suggested_fix (string). No markdown."
    )
    user_content = {
        "incident": incident_text,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "similar_past": similar_lines[:8],
    }
    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_content)},
        ],
        "max_tokens": cfg.max_tokens,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        f"{cfg.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        parsed = json.loads(text)
        return (
            parsed.get("summary"),
            parsed.get("hypothesis"),
            parsed.get("suggested_fix"),
            "openai_compatible",
        )
    except Exception:
        return (
            "LLM call failed; review similar incidents manually.",
            None,
            None,
            "error",
        )
