"""
Traffic Light Readiness Indicator.

Implements Spec 010: FR-001 (Traffic Light Readiness Indicator)

Provides instant go/no-go decisions for data modeling readiness:
- Green (≥80): Ready for modeling
- Yellow (60-79): Fixable with suggestions
- Red (<60): Significant issues
"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Traffic light thresholds
GREEN_THRESHOLD = 80
YELLOW_THRESHOLD = 60


@dataclass
class ReadinessStatus:
    """Traffic light readiness status."""

    status: Literal["green", "yellow", "red"]
    message: str
    action_message: str
    usability_score: float
    can_export: bool = True


def get_readiness_status(
    usability_score: float,
    n_suggestions: int = 0,
    estimated_improvement: float = 0.0,
) -> ReadinessStatus:
    """
    Determine traffic light readiness status from usability score.

    Implements Spec 010: FR-001

    Args:
        usability_score: Overall usability score (0-100).
        n_suggestions: Number of available suggestions.
        estimated_improvement: Estimated score improvement if suggestions applied.

    Returns:
        ReadinessStatus with color, message, and action guidance.
    """
    if usability_score >= GREEN_THRESHOLD:
        # Green: Ready to go
        return ReadinessStatus(
            status="green",
            message="✅ READY FOR MODELING",
            action_message="Export and start training! This dataset is modeling-ready.",
            usability_score=usability_score,
            can_export=True
        )

    elif usability_score >= YELLOW_THRESHOLD:
        # Yellow: Fixable
        if n_suggestions > 0:
            potential_score = min(100, usability_score + estimated_improvement)
            return ReadinessStatus(
                status="yellow",
                message="⚠️ FIXABLE",
                action_message=(
                    f"{n_suggestions} automated fix{'es' if n_suggestions != 1 else ''} available. "
                    f"Applying them could improve score to ~{potential_score:.0f}."
                ),
                usability_score=usability_score,
                can_export=True
            )
        else:
            return ReadinessStatus(
                status="yellow",
                message="⚠️ FIXABLE",
                action_message="Dataset needs some improvements. See recommendations below.",
                usability_score=usability_score,
                can_export=True
            )

    else:
        # Red: Needs work
        return ReadinessStatus(
            status="red",
            message="🔴 NEEDS WORK",
            action_message=(
                "Significant data quality issues detected. "
                "Review recommendations below and consider data collection improvements."
            ),
            usability_score=usability_score,
            can_export=False  # Don't encourage exporting poor quality data
        )


def estimate_score_improvement(
    suggestions: list,
    current_score: float
) -> float:
    """
    Estimate score improvement from applying suggestions.

    Simple heuristic:
    - Each suggestion contributes 3-8 points based on type
    - Diminishing returns as score approaches 100

    Args:
        suggestions: List of feature suggestions.
        current_score: Current usability score.

    Returns:
        Estimated improvement in points.
    """
    if not suggestions:
        return 0.0

    # Base improvement per suggestion type
    type_improvements = {
        "remove": 3,  # Removing low-value features
        "transform": 5,  # Log/sqrt transforms
        "combine": 8,  # Feature interactions
        "normalize": 3,  # Scaling
    }

    total_improvement = 0.0
    for suggestion in suggestions:
        suggestion_type = getattr(suggestion, "suggestion_type", "transform")
        base_improvement = type_improvements.get(suggestion_type, 5)

        # Diminishing returns as score approaches 100
        remaining_gap = 100 - current_score
        improvement = base_improvement * (remaining_gap / 100)

        total_improvement += improvement

    # Cap at reasonable maximum
    return min(total_improvement, 20.0)
