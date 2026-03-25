from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ForgettingAlertResult:
    """Structured forgetting / legacy-risk assessment for API consumers."""

    risk_level: str
    message: str
    drop_percentage: float
    affected_eras: list[int]
    affected_incident_types: list[str]
    confidence: float
    legacy_accuracy: float | None
    peak_legacy_accuracy: float | None
    explanation: str
    thresholds: dict[str, float]


def _peak_accuracy_per_test_era(acc_matrix: list[dict[int, float]]) -> dict[int, float]:
    peaks: dict[int, float] = {}
    for row_i, row in enumerate(acc_matrix):
        for test_era, acc in row.items():
            te = int(test_era)
            peaks[te] = max(peaks.get(te, 0.0), float(acc))
    return peaks


def compute_forgetting_alert(
    current_legacy_accuracy: float | None,
    peak_legacy_accuracy: float | None,
    low_threshold: float,
    medium_threshold: float,
    *,
    acc_matrix: list[dict[int, float]] | None = None,
    last_legacy_per_class: dict[int, float] | None = None,
    peak_legacy_per_class: dict[int, float] | None = None,
    era_drop_ratio: float = 0.85,
    class_drop_ratio: float = 0.90,
) -> ForgettingAlertResult:
    """
    Build a structured alert including drop %, affected eras/types, and confidence.

    Args:
        current_legacy_accuracy: Latest era-0 holdout accuracy.
        peak_legacy_accuracy: Best era-0 accuracy seen.
        low_threshold: Ratio of peak considered healthy.
        medium_threshold: Absolute floor before escalation.
        acc_matrix: Full accuracy matrix for cross-era regression.
        last_legacy_per_class: Latest per-class accuracies on era-0 holdout.
        peak_legacy_per_class: Historical per-class peaks on era-0.
        era_drop_ratio: Flag era ``j`` if final acc[j] < peak[j] * ratio.
        class_drop_ratio: Flag class ``c`` if current < peak * ratio.
    """
    thresholds = {
        "legacy_ratio_low": low_threshold,
        "legacy_ratio_medium": medium_threshold,
        "era_drop_ratio": era_drop_ratio,
        "class_drop_ratio": class_drop_ratio,
    }

    if current_legacy_accuracy is None or peak_legacy_accuracy is None:
        return ForgettingAlertResult(
            risk_level="low",
            message="Not enough evaluations to assess legacy forgetting.",
            drop_percentage=0.0,
            affected_eras=[],
            affected_incident_types=[],
            confidence=0.0,
            legacy_accuracy=current_legacy_accuracy,
            peak_legacy_accuracy=peak_legacy_accuracy,
            explanation=(
                "Run training and close at least one era to establish legacy metrics."
            ),
            thresholds=thresholds,
        )

    drop_pct = 0.0
    if peak_legacy_accuracy > 0:
        drop_pct = max(
            0.0,
            (peak_legacy_accuracy - current_legacy_accuracy) / peak_legacy_accuracy * 100.0,
        )

    ratio = (
        current_legacy_accuracy / peak_legacy_accuracy
        if peak_legacy_accuracy > 0
        else 0.0
    )

    affected_eras: list[int] = []
    if acc_matrix and len(acc_matrix) >= 1:
        peaks = _peak_accuracy_per_test_era(acc_matrix)
        final = acc_matrix[-1]
        for te_str, acc_now in final.items():
            te = int(te_str)
            peak = peaks.get(te, acc_now)
            if peak > 0 and float(acc_now) < peak * era_drop_ratio:
                affected_eras.append(te)

    affected_types: list[str] = []
    if last_legacy_per_class and peak_legacy_per_class:
        for cls, acc_now in last_legacy_per_class.items():
            peak_c = peak_legacy_per_class.get(cls)
            if peak_c is None or peak_c <= 0:
                continue
            if acc_now < peak_c * class_drop_ratio:
                affected_types.append(f"class_{cls}")

    if current_legacy_accuracy >= medium_threshold and ratio >= low_threshold:
        level = "low"
        msg = "Legacy holdout stable relative to peak; no strong forgetting signal."
    elif current_legacy_accuracy >= medium_threshold or ratio >= low_threshold * 0.9:
        level = "medium"
        msg = "Legacy accuracy is slipping; monitor replay and drift."
    else:
        level = "high"
        msg = "System is forgetting legacy incidents — raise replay budget or review recent shifts."

    if affected_eras and level != "low":
        msg += f" Affected eras (vs historical peak): {affected_eras}."
    if affected_types and level != "low":
        msg += f" Worst legacy classes: {', '.join(affected_types)}."

    expl = (
        f"Legacy acc {current_legacy_accuracy:.3f} vs peak {peak_legacy_accuracy:.3f} "
        f"({ratio:.1%} of peak), drop {drop_pct:.1f}%."
    )

    confidence = min(
        1.0,
        0.25 * (drop_pct / 25.0)
        + 0.25 * (len(affected_eras) / max(len(acc_matrix[-1]) if acc_matrix else 1, 1))
        + 0.25 * (len(affected_types) / max(len(last_legacy_per_class or {}), 1))
        + 0.25 * (1.0 if level == "high" else 0.5 if level == "medium" else 0.2),
    )

    return ForgettingAlertResult(
        risk_level=level,
        message=msg,
        drop_percentage=round(drop_pct, 3),
        affected_eras=sorted(set(affected_eras)),
        affected_incident_types=sorted(set(affected_types)),
        confidence=round(confidence, 4),
        legacy_accuracy=current_legacy_accuracy,
        peak_legacy_accuracy=peak_legacy_accuracy,
        explanation=expl,
        thresholds=thresholds,
    )


def alert_to_dict(result: ForgettingAlertResult) -> dict:
    """Flatten dataclass for JSON (includes legacy ``explanation`` field)."""
    return {**asdict(result)}
