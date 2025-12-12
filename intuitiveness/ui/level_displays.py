"""
Level-Specific Display Components

Feature: 003-level-dataviz-display

This module provides reusable display functions for each abstraction level.
These functions are used by both Guided Mode and Free Navigation Mode
to ensure visual consistency (FR-014).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import streamlit as st
import pandas as pd


class NavigationDirection(Enum):
    """Direction of navigation between levels."""
    DESCEND = "descend"  # L4→L3→L2→L1→L0
    ASCEND = "ascend"    # L0→L1→L2→L3


class DisplayType(Enum):
    """Type of visualization for each level."""
    FILE_LIST = "file_list"           # L4: Raw data files
    GRAPH_WITH_TABS = "graph_with_tabs"  # L3: Graph + entity/relationship tabs
    DOMAIN_TABLE = "domain_table"     # L2: Domain-categorized table
    VECTOR = "vector"                 # L1: Series/list of values
    DATUM = "datum"                   # L0: Single value


@dataclass
class LevelDisplayConfig:
    """
    Configuration for level-specific visualization.

    SC-001: Users can identify which abstraction level within 3 seconds
    """
    level: int  # 0-4
    display_type: DisplayType
    title: str
    show_counts: bool = True
    max_preview_rows: int = 50


# Level to display type mapping (FR-001 through FR-011)
LEVEL_DISPLAY_MAPPING: Dict[int, DisplayType] = {
    4: DisplayType.FILE_LIST,
    3: DisplayType.GRAPH_WITH_TABS,
    2: DisplayType.DOMAIN_TABLE,
    1: DisplayType.VECTOR,
    0: DisplayType.DATUM,
}


def get_display_level(
    source_level: int,
    target_level: int,
    direction: NavigationDirection
) -> int:
    """
    Determine which level's visualization to show.

    FR-012: During ascent, show visualization from LOWER level (source)
    SC-002: 100% of descent transitions show higher level's data
    SC-003: 100% of ascent transitions show lower level's data

    Args:
        source_level: Level user is coming from
        target_level: Level user is going to
        direction: Navigation direction

    Returns:
        Level whose visualization should be displayed
    """
    if direction == NavigationDirection.DESCEND:
        # Descent: show source level (what user is leaving/transforming)
        return source_level
    else:
        # Ascent: show source level (what user is enriching FROM)
        return source_level


def render_l4_file_list(
    files_data: List[Dict[str, Any]],
    show_preview: bool = True,
    max_preview_rows: int = 5
) -> None:
    """
    Render L4 (Raw Data) file list display.

    FR-001: Display uploaded raw dataset files as a list showing
            file name, row count, and column count
    FR-002: Allow users to preview each raw file's first few rows

    Args:
        files_data: List of dicts with 'name', 'dataframe', 'rows', 'columns' keys
        show_preview: Whether to show file preview
        max_preview_rows: Number of rows to show in preview
    """
    st.markdown("### 📁 Your Uploaded Files")

    if not files_data:
        st.info("No files uploaded yet.")
        return

    # Summary table - using domain-friendly labels (Constitution v1.2.0)
    summary_df = pd.DataFrame([
        {
            "File Name": f["name"],
            "Items": f"{f['rows']:,}",
            "Categories": f["columns"]
        }
        for f in files_data
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Preview section
    if show_preview:
        st.markdown("#### Preview")
        for file_info in files_data:
            with st.expander(f"📄 {file_info['name']} - First {max_preview_rows} items"):
                if 'dataframe' in file_info and file_info['dataframe'] is not None:
                    st.dataframe(
                        file_info['dataframe'].head(max_preview_rows),
                        use_container_width=True
                    )
                else:
                    st.info("No preview available")


def render_l2_domain_table(
    domain_data: Dict[str, pd.DataFrame],
    config: Optional[LevelDisplayConfig] = None
) -> None:
    """
    Render L2 (Domain Table) display.

    FR-008: Display the domain-categorized table showing which domain
            each item was classified into
    FR-009: Show domain labels clearly for each item in the table

    Args:
        domain_data: Dict mapping domain names to DataFrames of items
        config: Optional display configuration
    """
    max_rows = config.max_preview_rows if config else 50

    # Constitution v1.2.0: Use domain-friendly labels
    st.markdown("### 📊 Items by Category")

    if not domain_data:
        st.info("No categorized data available.")
        return

    total_items = sum(len(df) for df in domain_data.values())
    st.markdown(f"**{total_items:,} items across {len(domain_data)} categories**")

    for domain_name, df in domain_data.items():
        with st.expander(f"📁 {domain_name} ({len(df):,} items)", expanded=True):
            if df.empty:
                # T036: Handle empty category state
                st.info(f"No items matched the '{domain_name}' category.")
            else:
                display_df = df.head(max_rows) if len(df) > max_rows else df
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                if len(df) > max_rows:
                    st.caption(f"Showing first {max_rows} of {len(df):,} items")


def render_l1_vector(
    vector_data: Union[pd.Series, List[Any]],
    column_name: str,
    config: Optional[LevelDisplayConfig] = None
) -> None:
    """
    Render L1 (Vector) display.

    FR-010: Display the vector as a list or series of values
    FR-011: Show the column name from which the vector was extracted

    Args:
        vector_data: Series or list of values
        column_name: Name of the source column
        config: Optional display configuration
    """
    max_rows = config.max_preview_rows if config else 50

    # Constitution v1.2.0: Use domain-friendly labels
    st.markdown("### 📊 Your Selected Values")
    st.markdown(f"**From:** `{column_name}`")

    if isinstance(vector_data, pd.Series):
        total_count = len(vector_data)
        values_to_show = vector_data.head(max_rows).tolist()
    else:
        total_count = len(vector_data)
        values_to_show = vector_data[:max_rows]

    st.markdown(f"**Values** (showing first {min(max_rows, total_count)} of {total_count:,}):")

    # Display as a simple dataframe for better formatting
    display_df = pd.DataFrame({
        "#": range(1, len(values_to_show) + 1),
        "Value": values_to_show
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)


def render_l0_datum(
    value: Any,
    aggregation_method: str = "computed",
    source_info: Optional[str] = None
) -> None:
    """
    Render L0 (Datum) display.

    Displays a single atomic metric value prominently.

    Args:
        value: The scalar value to display
        aggregation_method: How the value was computed (e.g., "average", "sum")
        source_info: Optional info about source (e.g., "Taux de réussite G from Revenue domain")
    """
    # Constitution v1.2.0: Use domain-friendly labels
    st.markdown("### 📊 Your Computed Result")

    # Display the value prominently
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div style="
                text-align: center;
                padding: 30px;
                background-color: #f0f2f6;
                border-radius: 10px;
                margin: 20px 0;
            ">
                <div style="font-size: 48px; font-weight: bold; color: #1f77b4;">
                    {value}
                </div>
                <div style="font-size: 14px; color: #666; margin-top: 10px;">
                    Calculated using: {aggregation_method}
                </div>
                {f'<div style="font-size: 12px; color: #888; margin-top: 5px;">{source_info}</div>' if source_info else ''}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_navigation_direction_indicator(
    direction: NavigationDirection,
    source_level: int,
    target_level: int
) -> None:
    """
    Render navigation direction indicator.

    FR-013: System MUST clearly indicate the direction of navigation

    Args:
        direction: Ascending or descending
        source_level: Starting level
        target_level: Destination level
    """
    # Constitution v1.2.0: Use domain-friendly labels
    if direction == NavigationDirection.DESCEND:
        icon = "🔍"
        direction_text = "Exploring deeper"
        color = "#2196F3"  # Blue
    else:
        icon = "🔨"
        direction_text = "Building up"
        color = "#4CAF50"  # Green

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 5px 15px;
            background-color: {color}20;
            border-left: 3px solid {color};
            border-radius: 0 5px 5px 0;
            margin-bottom: 10px;
        ">
            <span style="font-size: 16px;">{icon}</span>
            <span style="color: {color}; font-weight: bold;">{direction_text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
