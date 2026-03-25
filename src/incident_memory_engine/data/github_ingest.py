"""
Download labeled GitHub issues from public repositories for continual-learning demos.

Uses the GitHub REST API (no token required for anonymous access; set ``GITHUB_TOKEN``
for higher rate limits). Produces a JSON file of cleaned text + integer labels + era ids
so the engine can train on real distribution shift (infra → monitoring → dashboards).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default repo per era (concept drift across OSS domains)
DEFAULT_ERA_REPOS: dict[int, str] = {
    0: "kubernetes/kubernetes",
    1: "prometheus/prometheus",
    2: "grafana/grafana",
}

CLASS_NAMES: list[str] = [
    "bug_defect",
    "feature_request",
    "documentation",
    "infra_platform",
    "observability",
    "ui_dashboard",
    "testing_ci",
    "other",
]

# (class_id, keyword substrings matched against concatenated label names)
LABEL_RULES: list[tuple[int, tuple[str, ...]]] = [
    (0, ("bug", "kind/bug", "defect", "regression", "fix")),
    (1, ("feature", "enhancement", "kind/feature", "proposal", "rfc")),
    (2, ("doc", "documentation", "kind/documentation")),
    (3, ("infra", "network", "node", "cluster", "deployment", "storage", "api-machinery")),
    (4, ("metric", "monitor", "alert", "scrape", "prometheus", "tsdb")),
    (5, ("dashboard", "panel", "grafana", "visualization", "ui", "frontend")),
    (6, ("test", "flaky", "ci", "e2e", "testing")),
]


def exposed_label_map() -> dict[str, Any]:
    """Serializable description of how GitHub labels map to ``CLASS_NAMES`` indices."""
    return {
        "class_names": list(CLASS_NAMES),
        "rules": [
            {"class_id": cid, "class_name": CLASS_NAMES[cid], "keywords": list(kws)}
            for cid, kws in LABEL_RULES
        ],
        "fallback_class_id": len(CLASS_NAMES) - 1,
    }


def clean_text(raw: str | None) -> str:
    """Strip markdown fences, URLs, and HTML-ish tags for simpler text encoders."""
    if not raw:
        return ""
    s = raw
    s = re.sub(r"```.*?```", " ", s, flags=re.DOTALL)
    s = re.sub(r"`[^`]+`", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:8000]


def labels_to_class(label_names: list[str], num_classes: int = 8) -> int:
    """Map GitHub label names to a fixed incident class in ``[0, num_classes)``."""
    blob = " ".join(label_names).lower()
    for cid, keys in LABEL_RULES:
        if any(k in blob for k in keys):
            return min(int(cid), max(0, num_classes - 1))
    return max(0, num_classes - 1)


def _session(token: str | None) -> requests.Session:
    s = requests.Session()
    s.headers["Accept"] = "application/vnd.github+json"
    s.headers["X-GitHub-Api-Version"] = "2022-11-28"
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def fetch_repo_issues(
    repo: str,
    max_items: int,
    *,
    token: str | None = None,
    sleep_s: float = 0.35,
) -> list[dict[str, Any]]:
    """
    Fetch up to ``max_items`` **issues only** (no PRs) from ``owner/name``.

    Uses the Search API (`is:issue`) so results are not mixed with pull requests.
    The legacy ``/repos/.../issues`` list forces very deep pagination on large
    repos (many pages are mostly PRs) and GitHub returns **422** past ~10 pages.
    """
    owner, _, name = repo.partition("/")
    if not name:
        raise ValueError(f"Invalid repo {repo!r}, expected owner/name")
    sess = _session(token)
    out: list[dict[str, Any]] = []
    page = 1
    search_url = "https://api.github.com/search/issues"
    # +type:issue is redundant with is:issue but keeps queries explicit
    q = f"repo:{owner}/{name} is:issue"
    while len(out) < max_items and page <= 10:
        need = min(100, max_items - len(out))
        if need <= 0:
            break
        r = sess.get(
            search_url,
            params={
                "q": q,
                "per_page": need,
                "page": page,
                "sort": "updated",
                "order": "desc",
            },
            timeout=60,
        )
        if r.status_code == 403:
            logger.warning("GitHub API 403 (rate limit?); sleeping 65s once")
            time.sleep(65)
            r = sess.get(
                search_url,
                params={
                    "q": q,
                    "per_page": need,
                    "page": page,
                    "sort": "updated",
                    "order": "desc",
                },
                timeout=60,
            )
        r.raise_for_status()
        data = r.json()
        batch = data.get("items") or []
        if data.get("incomplete_results"):
            logger.warning("GitHub search returned incomplete_results; stopping early")
            break
        for it in batch:
            if len(out) >= max_items:
                break
            out.append(it)
        if len(batch) < need:
            break
        page += 1
        time.sleep(sleep_s)
    return out


def issues_to_samples(
    issues: list[dict[str, Any]],
    *,
    era: int,
    repo: str,
    num_classes: int = 8,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for it in issues:
        labels = [lb["name"] for lb in it.get("labels", []) if isinstance(lb, dict)]
        title = it.get("title") or ""
        body = it.get("body") or ""
        text = clean_text(f"{title}\n{body}")
        if len(text) < 20:
            continue
        lab = labels_to_class(labels, num_classes=num_classes)
        rows.append(
            {
                "text": text,
                "label": lab,
                "era": era,
                "source": repo,
                "github_id": it.get("id"),
                "title": title[:200],
            }
        )
    return rows


def ingest_eras(
    era_repo_map: dict[int, str],
    per_era: int,
    *,
    token: str | None = None,
    num_classes: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch all configured eras and return samples + label map metadata."""
    all_rows: list[dict[str, Any]] = []
    for era in sorted(era_repo_map.keys()):
        repo = era_repo_map[era]
        logger.info("Fetching era %s from %s (up to %s issues)", era, repo, per_era)
        issues = fetch_repo_issues(repo, per_era, token=token)
        all_rows.extend(
            issues_to_samples(issues, era=era, repo=repo, num_classes=num_classes)
        )
    meta = exposed_label_map()
    meta["era_repos"] = {str(k): v for k, v in era_repo_map.items()}
    meta["num_classes"] = num_classes
    return all_rows, meta


def save_artifacts(
    rows: list[dict[str, Any]],
    label_meta: dict[str, Any],
    out_path: Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"samples": rows, "label_map": label_meta}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s samples to %s", len(rows), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Ingest GitHub issues for Incident Memory Engine")
    p.add_argument("--eras", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--per-era", type=int, default=300)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/github_issues.json"),
    )
    args = p.parse_args()
    token = __import__("os").environ.get("GITHUB_TOKEN")
    era_map = {e: DEFAULT_ERA_REPOS[e] for e in args.eras if e in DEFAULT_ERA_REPOS}
    if len(era_map) != len(args.eras):
        missing = set(args.eras) - set(era_map)
        raise SystemExit(f"Unknown era ids (no default repo): {missing}")
    rows, meta = ingest_eras(era_map, args.per_era, token=token)
    save_artifacts(rows, meta, args.output)


if __name__ == "__main__":
    main()
