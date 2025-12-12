"""
Metric Card Component for L0 Result Display

Renders styled metric cards for displaying L0 datum results.
Based on css-contracts.md render_metric_card specification (007-streamlit-design-makeup).
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
) -> None:
    """Render a styled metric card.

    Args:
        label: Caption above the value (e.g., "Total Revenue")
        value: Primary display value (e.g., "$1,234")
        delta: Optional change indicator (e.g., "+5%", "-2.3%")
        description: Optional additional context below the value

    Example:
        render_metric_card(
            label="Average Score",
            value="85.3",
            delta="+2.5%",
            description="Compared to last period"
        )
    """
    # Build delta HTML if provided
    delta_html = ""
    if delta:
        # Determine color based on sign
        if delta.startswith("+"):
            delta_color = COLORS["success"]
        elif delta.startswith("-"):
            delta_color = COLORS["error"]
        else:
            delta_color = COLORS["text_secondary"]

        delta_html = f'''
        <div style="
            color: {delta_color};
            font-size: 0.875rem;
            margin-top: 0.25rem;
        ">{delta}</div>
        '''

    # Build description HTML if provided
    desc_html = ""
    if description:
        desc_html = f'''
        <div style="
            color: {COLORS["text_secondary"]};
            font-size: 0.875rem;
            margin-top: 0.5rem;
        ">{description}</div>
        '''

    # Render the card
    st.markdown(f"""
    <div style="
        background: {COLORS["bg_elevated"]};
        border-radius: 0.5rem;
        padding: 1.25rem;
        border: 1px solid {COLORS["border"]};
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    ">
        <div style="
            color: {COLORS["text_secondary"]};
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        ">{label}</div>
        <div style="
            font-size: 1.75rem;
            font-weight: 600;
            color: {COLORS["text_primary"]};
            margin-top: 0.25rem;
        ">{value}</div>
        {delta_html}
        {desc_html}
    </div>
    """, unsafe_allow_html=True)


def render_metric_card_row(
    metrics: list[dict],
    columns: int = 3,
) -> None:
    """Render multiple metric cards in a row.

    Args:
        metrics: List of dicts with keys: label, value, delta (optional), description (optional)
        columns: Number of columns (default 3)

    Example:
        render_metric_card_row([
            {"label": "Total", "value": "1,234"},
            {"label": "Average", "value": "56.7", "delta": "+3.2%"},
            {"label": "Count", "value": "42"},
        ])
    """
    cols = st.columns(columns)
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            render_metric_card(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta"),
                description=metric.get("description"),
            )
