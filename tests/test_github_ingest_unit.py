"""Unit tests for GitHub ingest helpers (no network)."""

from __future__ import annotations

from incident_memory_engine.data.github_ingest import (
    clean_text,
    exposed_label_map,
    labels_to_class,
)


def test_clean_text_strips_urls_and_fences() -> None:
    raw = "See https://x.com/a ```x``` and <b>tag</b>"
    out = clean_text(raw)
    assert "https://" not in out
    assert "```" not in out
    assert "<b>" not in out


def test_labels_to_class_bug() -> None:
    assert labels_to_class(["kind/bug", "sig/network"], num_classes=8) == 0


def test_labels_to_class_fallback() -> None:
    assert labels_to_class(["random-unknown"], num_classes=8) == 7


def test_exposed_label_map_has_class_names() -> None:
    m = exposed_label_map()
    assert "class_names" in m
    assert len(m["class_names"]) == 8
