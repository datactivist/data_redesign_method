"""
L0→L1 Unfold Form - Deterministic Source Vector Expansion

Implements Spec 004: FR-001-004 (L0→L1 Deterministic Unfolding)
Extracted from ui/ascent_forms.py (lines 268-331)

Features:
- Display the source vector from which L0 datum was aggregated
- Show the aggregation method used (mean, sum, count, etc.)
- Block ascent when no parent vector exists
- Preserve column name from original vector

Usage:
    from intuitiveness.ui.ascent import render_l0_to_l1_unfold_form

    result = render_l0_to_l1_unfold_form(l0_dataset)
    if result:
        # User confirmed unfold operation
        enrichment_func = result['enrichment_func']
"""

from typing import Any, Dict, Optional
import streamlit as st
import pandas as pd

from intuitiveness.ui.i18n import t


def render_l0_to_l1_unfold_form(
    dataset: Any,
    key_prefix: str = "l0_to_l1"
) -> Optional[Dict[str, Any]]:
    """
    Render L0→L1 unfold confirmation form.

    FR-001: Display the source vector from which the L0 datum was aggregated
    FR-002: Show the aggregation method that was used
    FR-003: Block ascent when no parent vector exists
    FR-004: Preserve the column name from the original vector

    Args:
        dataset: Level0Dataset with potential parent_data
        key_prefix: Unique prefix for session state keys

    Returns:
        Dict with 'enrichment_func': 'source_expansion' if confirmed, None if blocked/cancelled
    """
    # Info tooltip explaining unfold operation
    st.info(
        f"**{t('expand_result_title')}**: {t('expand_result_info')}"
    )

    # Check if parent data exists (FR-003)
    has_parent = getattr(dataset, 'has_parent', False)

    if not has_parent:
        st.warning(f"**{t('cannot_expand')}**: {t('cannot_expand_reason')}")
        st.info(t("expansion_unavailable_info"))
        return None

    # Get parent data for preview
    parent_data = dataset.get_parent_data()
    aggregation_type = getattr(dataset, 'aggregation_type', None) or \
                       getattr(dataset, 'description', 'aggregation') or 'aggregation'

    # Display calculation method (FR-002)
    st.markdown(f"**{t('calculation_method')}**: `{aggregation_type}`")

    # Show source values preview
    st.markdown(f"**{t('source_values_preview')}** ({t('first_n_values', n=10)}):")
    if parent_data is not None:
        preview = parent_data.head(10) if hasattr(parent_data, 'head') else parent_data[:10]
        if isinstance(preview, pd.Series):
            st.dataframe(preview.to_frame(), use_container_width=True)
        else:
            st.write(preview)

        # Show total count
        total = len(parent_data) if hasattr(parent_data, '__len__') else "unknown"
        st.caption(t("total_values", count=total))

    st.divider()

    # Confirmation button
    if st.button(t("expand_to_source_values"), key=f"{key_prefix}_unfold_btn", type="primary"):
        return {'enrichment_func': 'source_expansion'}

    return None
