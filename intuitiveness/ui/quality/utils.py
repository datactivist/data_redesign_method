"""
Quality Dashboard - Shared Utilities

Phase 1.3 - Code Simplification (011-code-simplification)
Extracted from quality_dashboard.py

Spec Traceability:
------------------
- 009-quality-data-platform: Quality assessment display
- 010-quality-ds-workflow: DS Co-Pilot utilities

Contains:
- Score coloring and labeling utilities
- Session state key aliases
"""

import streamlit as st
from typing import Optional

# Phase 0 utilities (011-code-simplification)
from intuitiveness.utils import SessionStateKeys, score_to_color


# Backward compatibility aliases - use SessionStateKeys directly in new code
SESSION_KEY_QUALITY_REPORT = SessionStateKeys.QUALITY_REPORT
SESSION_KEY_QUALITY_DF = SessionStateKeys.QUALITY_DF
SESSION_KEY_QUALITY_FILE_NAME = SessionStateKeys.QUALITY_FILE_NAME
SESSION_KEY_ASSESSMENT_PROGRESS = SessionStateKeys.ASSESSMENT_PROGRESS
SESSION_KEY_APPLIED_SUGGESTIONS = SessionStateKeys.APPLIED_SUGGESTIONS
SESSION_KEY_TRANSFORMED_DF = SessionStateKeys.TRANSFORMED_DF
SESSION_KEY_TRANSFORMATION_LOG = SessionStateKeys.TRANSFORMATION_LOG
SESSION_KEY_BENCHMARK_REPORT = SessionStateKeys.BENCHMARK_REPORT
SESSION_KEY_EXPORT_FORMAT = SessionStateKeys.EXPORT_FORMAT
SESSION_KEY_QUALITY_REPORTS_HISTORY = SessionStateKeys.QUALITY_REPORTS_HISTORY
SESSION_KEY_CURRENT_REPORT_INDEX = SessionStateKeys.CURRENT_REPORT_INDEX


def get_score_color(score: float) -> str:
    """
    Get color based on score value.

    Uses 4-color mode for traffic light indicator:
    - Red (<40): Poor
    - Orange (40-59): Fair
    - Yellow (60-79): Good
    - Green (80+): Excellent

    Args:
        score: Quality score (0-100)

    Returns:
        Hex color string
    """
    return score_to_color(
        score,
        thresholds=(40, 60, 80),
        colors=("#ef4444", "#f97316", "#eab308", "#22c55e")
    )


def get_score_label(score: float) -> str:
    """
    Get label based on score value.

    Args:
        score: Quality score (0-100)

    Returns:
        Human-readable label (Excellent/Good/Fair/Poor)
    """
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    else:
        return "Poor"


# Legacy aliases for backward compatibility
_score_color = get_score_color
_score_label = get_score_label
