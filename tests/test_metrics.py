from __future__ import annotations

from incident_memory_engine.metrics import compute_bwt_and_forgetting, summary_from_matrix


def test_bwt_and_forgetting_matrix() -> None:
    """Sanity check on a toy accuracy matrix."""
    matrix = [
        {0: 0.9},
        {0: 0.7, 1: 0.85},
        {0: 0.5, 1: 0.8, 2: 0.88},
    ]
    bwt, f_mean = compute_bwt_and_forgetting(matrix)
    assert isinstance(bwt, float)
    assert isinstance(f_mean, float)
    assert f_mean >= 0.0
    summary = summary_from_matrix(matrix)
    assert 0.0 <= summary.avg_acc_all_seen <= 1.0
