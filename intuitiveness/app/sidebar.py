"""
Sidebar Module

Centralizes all sidebar rendering components.

Phase 1 - Code Simplification (011-code-simplification)
Created: 2026-01-09

Spec Traceability:
------------------
- 005-session-persistence: Save/clear buttons
- 006-playwright-mcp-e2e: Language toggle
- 007-streamlit-design-makeup: Branding, styling
- 008-datagouv-search: Dataset basket
- 009-010: Quality tools selector
"""

import streamlit as st
from typing import Callable, Any

from intuitiveness.utils import SessionStateKeys
from intuitiveness.persistence import SessionStore


def render_sidebar(
    store: SessionStore,
    t: Callable[[str], str],
    reset_workflow: Callable,
    Level4Dataset: Any,
):
    """
    Render the complete left sidebar.

    Args:
        store: SessionStore for persistence
        t: Translation function (i18n)
        reset_workflow: Function to reset workflow
        Level4Dataset: Level4Dataset class for data loading
    """
    # Import UI components
    from intuitiveness.streamlit_app import (
        _get_sidebar_branding_html,
        render_free_navigation_sidebar,
    )
    from intuitiveness.ui import (
        render_basket_sidebar,
        render_language_toggle_compact,
        render_tutorial_replay_button,
        is_tutorial_completed,
        reset_tutorial,
        _set_wizard_step,
    )

    with st.sidebar:
        # Branding (007)
        _render_branding(_get_sidebar_branding_html())

        # Language toggle (006)
        _render_language_section(render_language_toggle_compact)

        # Dataset basket (008)
        _render_basket_section(
            render_basket_sidebar=render_basket_sidebar,
            reset_tutorial=reset_tutorial,
            _set_wizard_step=_set_wizard_step,
            Level4Dataset=Level4Dataset,
        )

        # Mode toggle (002)
        _render_mode_section(t)

        # Quality tools (009-010)
        _render_quality_tools_section()

        # Free navigation (002)
        _render_free_nav_section(render_free_navigation_sidebar)

        # Reset button
        _render_reset_button(t, reset_workflow)

        # Tutorial replay (007)
        _render_tutorial_section(
            is_tutorial_completed=is_tutorial_completed,
            render_tutorial_replay_button=render_tutorial_replay_button,
        )

        # Session persistence (005)
        _render_session_section(t, store, reset_workflow)


def _render_branding(branding_html: str):
    """Render branding section."""
    st.markdown(branding_html, unsafe_allow_html=True)
    st.markdown("---")


def _render_language_section(render_language_toggle_compact: Callable):
    """Render language toggle section (006)."""
    render_language_toggle_compact()
    st.divider()


def _render_basket_section(
    render_basket_sidebar: Callable,
    reset_tutorial: Callable,
    _set_wizard_step: Callable,
    Level4Dataset: Any,
):
    """Render dataset basket section (008)."""
    if render_basket_sidebar():
        # User clicked "Continue" - proceed with loaded datasets
        raw_data = st.session_state.datagouv_loaded_datasets.copy()
        st.session_state[SessionStateKeys.RAW_DATA] = raw_data
        st.session_state.datasets['l4'] = Level4Dataset(raw_data)
        st.session_state.datagouv_loaded_datasets = {}
        st.session_state[SessionStateKeys.CURRENT_STEP] = 0
        _set_wizard_step(1)
        reset_tutorial()
        st.rerun()


def _render_mode_section(t: Callable[[str], str]):
    """Render mode toggle section (002)."""
    st.markdown(f"### {t('exploration_mode')}")

    current_mode = st.session_state.get(SessionStateKeys.NAV_MODE, 'guided')
    mode = st.radio(
        t('select_mode'),
        options=['guided', 'free'],
        format_func=lambda x: t('step_by_step') if x == 'guided' else t('free_exploration'),
        index=0 if current_mode == 'guided' else 1,
        key='mode_selector',
        help=t('step_by_step_help')
    )

    # Sync radio with nav_mode (skip if in ascent mode)
    has_loaded_graph = st.session_state.get(SessionStateKeys.LOADED_SESSION_GRAPH)
    if mode != current_mode and not has_loaded_graph:
        st.session_state[SessionStateKeys.NAV_MODE] = mode
        st.rerun()

    st.divider()


def _render_quality_tools_section():
    """Render quality tools section (009-010)."""
    st.markdown("### Data modeling Tools")

    current_tool = st.session_state.get('active_quality_tool', 'none')
    quality_tool = st.radio(
        "Select tool",
        options=['none', 'quality', 'catalog'],
        format_func=lambda x: {
            'none': 'None',
            'quality': '📊 Quality Assessment',
            'catalog': '📁 Dataset Catalog'
        }.get(x, x),
        index=['none', 'quality', 'catalog'].index(current_tool) if current_tool in ['none', 'quality', 'catalog'] else 0,
        key='quality_tool_selector',
        label_visibility='collapsed',
    )

    if quality_tool != current_tool:
        st.session_state.active_quality_tool = quality_tool
        st.rerun()

    st.divider()


def _render_free_nav_section(render_free_navigation_sidebar: Callable):
    """Render free navigation section (002)."""
    is_free_mode = st.session_state.get(SessionStateKeys.NAV_MODE) == 'free'
    has_session = st.session_state.get(SessionStateKeys.NAV_SESSION) is not None

    if is_free_mode and has_session:
        render_free_navigation_sidebar()
        st.divider()


def _render_reset_button(t: Callable[[str], str], reset_workflow: Callable):
    """Render reset workflow button."""
    if st.button(f"🔄 {t('reset_workflow')}"):
        reset_workflow()
        st.rerun()


def _render_tutorial_section(
    is_tutorial_completed: Callable,
    render_tutorial_replay_button: Callable,
):
    """Render tutorial replay section (007)."""
    has_data = st.session_state.get(SessionStateKeys.RAW_DATA) is not None
    if is_tutorial_completed() and has_data:
        render_tutorial_replay_button()


def _render_session_section(
    t: Callable[[str], str],
    store: SessionStore,
    reset_workflow: Callable,
):
    """Render session persistence section (005)."""
    st.markdown(f"### {t('sidebar_session')}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"💾 {t('save_button')}", help=t('save_help')):
            try:
                result = store.save(force=True)
                if result.success:
                    st.success(t('saved_success'))
                else:
                    st.warning(t('save_too_large'))
            except Exception as e:
                st.error(t('save_failed', error=str(e)))

    with col2:
        if st.button(f"🗑️ {t('clear_button')}", help=t('clear_help')):
            store.clear()
            reset_workflow()
            st.session_state.session_recovery_handled = True
            st.rerun()
