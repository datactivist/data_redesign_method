"""
Metric Card Component for L0 Result Display

Renders styled metric cards for displaying L0 datum results.
Based on css-contracts.md render_metric_card specification (007-streamlit-design-makeup).
Following Gael Penessot's DataGyver principles for clean, professional UI.
"""

import streamlit as st
from typing import Optional

# Import colors from centralized palette
from intuitiveness.styles.palette import COLORS


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    description: Optional[str] = None,
    suffix: Optional[str] = None,
    color: Optional[str] = None,
) -> None:
    """Render a styled metric card.

    Args:
        label: Caption above the value (e.g., "Total Revenue")
        value: Primary display value (e.g., "$1,234")
        delta: Optional change indicator (e.g., "+5%", "-2.3%")
        description: Optional additional context below the value
        suffix: Optional suffix for the value (e.g., "/100", "%")
        color: Optional custom color for the value

    Example:
        render_metric_card(
            label="Average Score",
            value="85.3",
            delta="+2.5%",
            description="Compared to last period",
            suffix="/100",
            color="#22c55e"
        )
    """
    # Determine value color
    value_color = color if color else COLORS["text_primary"]

    # Build delta HTML if provided
    delta_html = ""
    if delta:
        if delta.startswith("+"):
            delta_color = COLORS["success"]
        elif delta.startswith("-"):
            delta_color = COLORS["error"]
        else:
            delta_color = COLORS["text_secondary"]
        delta_html = f'<div style="color: {delta_color}; font-size: 0.875rem; margin-top: 0.25rem;">{delta}</div>'

    # Build description HTML if provided
    desc_html = ""
    if description:
        desc_html = f'<div style="color: {COLORS["text_secondary"]}; font-size: 0.875rem; margin-top: 0.5rem;">{description}</div>'

    # Build value with optional suffix
    value_display = value
    if suffix:
        value_display = f'{value}<span style="font-size: 1rem; font-weight: 400; color: {COLORS["text_muted"]};">{suffix}</span>'

    # Render the card - compact HTML without newlines in attributes
    html = f'''<div style="background: {COLORS["bg_elevated"]}; border-radius: 0.5rem; padding: 1.25rem; border: 1px solid {COLORS["border"]}; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
<div style="color: {COLORS["text_secondary"]}; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
<div style="font-size: 1.75rem; font-weight: 600; color: {value_color}; margin-top: 0.25rem;">{value_display}</div>
{delta_html}
{desc_html}
</div>'''

    st.markdown(html, unsafe_allow_html=True)


def render_metric_card_row(
    metrics: list[dict],
    columns: Optional[int] = None,
) -> None:
    """Render multiple metric cards in a row.

    Args:
        metrics: List of dicts with keys: label, value, delta, description, suffix, color
        columns: Number of columns (default: same as number of metrics, max 4)

    Example:
        render_metric_card_row([
            {"label": "Total", "value": "1,234"},
            {"label": "Average", "value": "56.7", "delta": "+3.2%"},
            {"label": "Count", "value": "42", "suffix": "/100", "color": "#22c55e"},
        ])
    """
    # Auto-determine columns based on number of metrics
    num_metrics = len(metrics)
    if columns is None:
        columns = min(num_metrics, 4)

    cols = st.columns(columns)
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            render_metric_card(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta"),
                description=metric.get("description"),
                suffix=metric.get("suffix"),
                color=metric.get("color"),
            )
