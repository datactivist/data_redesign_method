"""
Quality Data Platform - Catalog Browser UI

Streamlit UI component for browsing, filtering, and managing
the dataset catalog.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List
from uuid import UUID

from intuitiveness.ui.layout import card, spacer
from intuitiveness.ui.header import render_page_header, render_section_header
from intuitiveness.ui.metric_card import render_metric_card_row
from intuitiveness.ui.alert import info, success, warning, error

from intuitiveness.catalog.storage import get_storage, CatalogStorage
from intuitiveness.catalog.search import filter_datasets, search_datasets, get_all_domains, get_score_distribution
from intuitiveness.catalog.models import Dataset, DatasetSummary

# Import consolidated utilities (Phase 2 - 011-code-simplification)
from intuitiveness.utils import SessionStateKeys, score_to_color

# Session state keys - use centralized keys with backward compatibility aliases
SESSION_KEY_CATALOG_FILTER_SCORE = SessionStateKeys.CATALOG_FILTER_SCORE
SESSION_KEY_CATALOG_FILTER_DOMAINS = SessionStateKeys.CATALOG_FILTER_DOMAINS
SESSION_KEY_CATALOG_SEARCH_QUERY = SessionStateKeys.CATALOG_SEARCH_QUERY
SESSION_KEY_CATALOG_SELECTED_DATASET = SessionStateKeys.CATALOG_SELECTED_DATASET


def _score_color(score: Optional[float]) -> str:
    """Get color based on score value. Delegates to utils.score_to_color."""
    if score is None:
        return "#94a3b8"
    # Use consolidated utility with catalog-specific thresholds (4-tier vs 3-tier)
    # Catalog uses 40/60/80 thresholds for 4 colors
    if score >= 80:
        return "#22c55e"  # green
    elif score >= 60:
        return "#eab308"  # yellow
    elif score >= 40:
        return "#f97316"  # orange
    else:
        return "#ef4444"  # red


def _score_badge(score: Optional[float]) -> str:
    """Get badge HTML for score."""
    if score is None:
        return '<span style="color: #94a3b8;">—</span>'

    color = _score_color(score)
    return f'<span style="color: {color}; font-weight: 600;">{score:.0f}</span>'


def render_catalog_stats() -> None:
    """Render catalog statistics cards."""
    storage = get_storage()
    distribution = get_score_distribution(storage)
    total = storage.count()

    render_metric_card_row([
        {
            "label": "Total Datasets",
            "value": str(total),
            "description": "In catalog",
            "color": "#3b82f6",
        },
        {
            "label": "Excellent",
            "value": str(distribution["excellent"]),
            "suffix": " (80+)",
            "description": "High quality",
            "color": "#22c55e",
        },
        {
            "label": "Good",
            "value": str(distribution["good"]),
            "suffix": " (60-79)",
            "description": "Ready for ML",
            "color": "#eab308",
        },
        {
            "label": "Needs Work",
            "value": str(distribution["fair"] + distribution["poor"]),
            "suffix": " (<60)",
            "description": "Requires prep",
            "color": "#f97316",
        },
    ])


def render_catalog_filters() -> tuple:
    """
    Render catalog filter controls.

    Returns:
        Tuple of (min_score, domains, search_query)
    """
    storage = get_storage()
    all_domains = get_all_domains(storage)

    col1, col2, col3 = st.columns([2, 3, 2])

    with col1:
        min_score = st.slider(
            "Minimum Score",
            min_value=0,
            max_value=100,
            value=st.session_state.get(SESSION_KEY_CATALOG_FILTER_SCORE, 0),
            step=10,
            key="catalog_min_score_slider",
        )
        st.session_state[SESSION_KEY_CATALOG_FILTER_SCORE] = min_score

    with col2:
        domains = st.multiselect(
            "Filter by Domain",
            options=all_domains,
            default=st.session_state.get(SESSION_KEY_CATALOG_FILTER_DOMAINS, []),
            key="catalog_domains_select",
        )
        st.session_state[SESSION_KEY_CATALOG_FILTER_DOMAINS] = domains

    with col3:
        search_query = st.text_input(
            "Search",
            value=st.session_state.get(SESSION_KEY_CATALOG_SEARCH_QUERY, ""),
            placeholder="Search datasets...",
            key="catalog_search_input",
        )
        st.session_state[SESSION_KEY_CATALOG_SEARCH_QUERY] = search_query

    return min_score, domains if domains else None, search_query if search_query else None


def render_dataset_card(dataset: Dataset) -> bool:
    """
    Render a dataset card in the catalog list.

    Args:
        dataset: Dataset to display.

    Returns:
        True if user clicked to view details.
    """
    with st.container():
        col1, col2, col3 = st.columns([1, 5, 1])

        with col1:
            # Score circle
            score = dataset.usability_score
            color = _score_color(score)
            score_text = f"{score:.0f}" if score is not None else "—"
            st.markdown(
                f"""
                <div style="
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    background: {color}20;
                    border: 2px solid {color};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                    font-weight: 700;
                    color: {color};
                ">
                    {score_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            # Dataset info
            tags_html = " ".join(
                f'<span style="background: #e2e8f0; color: #475569; padding: 2px 8px; '
                f'border-radius: 4px; font-size: 11px; margin-right: 4px;">{tag}</span>'
                for tag in dataset.domain_tags[:3]
            )

            st.markdown(
                f"""
                <div style="margin-bottom: 4px;">
                    <strong style="font-size: 16px;">{dataset.name}</strong>
                </div>
                <div style="color: #64748b; font-size: 13px; margin-bottom: 6px;">
                    {dataset.description[:100] + "..." if len(dataset.description) > 100 else dataset.description or "No description"}
                </div>
                <div style="display: flex; align-items: center; gap: 12px; font-size: 12px; color: #94a3b8;">
                    <span>📊 {dataset.row_count:,} rows</span>
                    <span>📋 {dataset.feature_count} features</span>
                    {tags_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            view_clicked = st.button(
                "View",
                key=f"view_dataset_{dataset.id}",
                use_container_width=True,
            )
            if view_clicked:
                st.session_state[SESSION_KEY_CATALOG_SELECTED_DATASET] = str(dataset.id)
                return True

        st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    return False


def render_dataset_list(datasets: List[Dataset]) -> None:
    """Render the list of datasets."""
    if not datasets:
        info("No datasets match your filters. Try adjusting the criteria or add new datasets.")
        return

    for dataset in datasets:
        if render_dataset_card(dataset):
            st.rerun()


def render_dataset_detail(dataset_id: str) -> None:
    """
    Render detailed view of a single dataset.

    Args:
        dataset_id: UUID string of the dataset.
    """
    storage = get_storage()
    detail = storage.get_dataset_detail(UUID(dataset_id))

    if detail is None:
        error("Dataset not found.")
        return

    dataset = detail.dataset
    report = detail.latest_report

    # Back button
    if st.button("← Back to Catalog"):
        st.session_state.pop(SESSION_KEY_CATALOG_SELECTED_DATASET, None)
        st.rerun()

    spacer(16)

    # Header
    render_page_header(dataset.name, dataset.description or "No description")

    spacer(16)

    # Metadata card
    with card():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Usability Score", f"{dataset.usability_score:.0f}" if dataset.usability_score else "—")
        with col2:
            st.metric("Rows", f"{dataset.row_count:,}")
        with col3:
            st.metric("Features", dataset.feature_count)
        with col4:
            st.metric("Target", dataset.target_column or "—")

        # Tags
        if dataset.domain_tags:
            st.markdown(
                "**Tags:** " + " ".join(
                    f'`{tag}`' for tag in dataset.domain_tags
                )
            )

    spacer(16)

    # Quality report
    if report:
        render_section_header("Quality Report", f"Assessed on {report.get('created_at', 'Unknown')[:10]}")

        with card():
            render_metric_card_row([
                {"label": "Prediction Quality", "value": f"{report.get('prediction_quality', 0):.0f}", "suffix": "/100"},
                {"label": "Data Completeness", "value": f"{report.get('data_completeness', 0):.0f}", "suffix": "/100"},
                {"label": "Feature Diversity", "value": f"{report.get('feature_diversity', 0):.0f}", "suffix": "/100"},
                {"label": "Size Score", "value": f"{report.get('size_appropriateness', 0):.0f}", "suffix": "/100"},
            ])

            # Feature profiles
            if report.get("feature_profiles"):
                spacer(16)
                st.subheader("Feature Profiles")

                feature_data = []
                for fp in report["feature_profiles"]:
                    feature_data.append({
                        "Feature": fp["feature_name"],
                        "Type": fp["feature_type"].title(),
                        "Importance": f"{fp['importance_score']:.3f}",
                        "Missing": f"{fp['missing_ratio']:.1%}",
                    })

                st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)
    else:
        info("No quality report available. Run an assessment to generate one.")

    spacer(16)

    # Actions
    render_section_header("Actions", "Manage this dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download Data", use_container_width=True):
            # TODO: Implement data download
            info("Download functionality coming soon.")

    with col2:
        if st.button("🔄 Re-assess", use_container_width=True):
            # TODO: Trigger re-assessment
            info("Re-assessment functionality coming soon.")

    with col3:
        if st.button("🗑️ Remove from Catalog", use_container_width=True, type="secondary"):
            if storage.delete_dataset(UUID(dataset_id)):
                success("Dataset removed from catalog.")
                st.session_state.pop(SESSION_KEY_CATALOG_SELECTED_DATASET, None)
                st.rerun()
            else:
                error("Failed to remove dataset.")


def render_catalog_browser() -> None:
    """Render the complete catalog browser."""
    # Check if viewing a specific dataset
    selected_id = st.session_state.get(SESSION_KEY_CATALOG_SELECTED_DATASET)
    if selected_id:
        render_dataset_detail(selected_id)
        return

    render_page_header(
        "Dataset Catalog",
        "Browse and discover high-quality datasets for machine learning"
    )

    spacer(16)

    # Stats
    render_catalog_stats()

    spacer(24)

    # Filters
    with card():
        render_section_header("Filter Datasets", "Find datasets matching your criteria")
        min_score, domains, search_query = render_catalog_filters()

    spacer(16)

    # Results
    storage = get_storage()

    if search_query:
        datasets = search_datasets(search_query, limit=50, storage=storage)
    else:
        datasets = filter_datasets(
            min_score=min_score if min_score > 0 else None,
            domains=domains,
            sort_by="usability_score",
            sort_desc=True,
            limit=50,
            storage=storage,
        )

    render_section_header(
        f"Results ({len(datasets)})",
        "Sorted by usability score"
    )

    render_dataset_list(datasets)
