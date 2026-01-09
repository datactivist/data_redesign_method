"""
Free Navigation Mode Module

Implements free exploration with decision tree navigation (specs 002-003).

Phase 1 - Code Simplification (011-code-simplification)
Created: 2026-01-09

Spec Traceability:
------------------
- 002-ascent-functionality: US-5 (Navigate Dataset Hierarchy)
- 003-level-dataviz-display: Level-specific visualizations

Features:
---------
- Decision tree sidebar for visual navigation
- Time-travel support (navigate history)
- Session graph loading/saving
- Export functionality
"""

import streamlit as st
from typing import Optional, Dict, Any

from intuitiveness.utils import SessionStateKeys


# =============================================================================
# STATE HELPERS
# =============================================================================

def get_nav_session():
    """Get current NavigationSession instance."""
    return st.session_state.get(SessionStateKeys.NAV_SESSION)


def set_nav_session(session) -> None:
    """Set NavigationSession instance."""
    st.session_state[SessionStateKeys.NAV_SESSION] = session


def has_nav_session() -> bool:
    """Check if navigation session exists."""
    return get_nav_session() is not None


def get_nav_action() -> Optional[str]:
    """Get current navigation action."""
    return st.session_state.get(SessionStateKeys.NAV_ACTION)


def set_nav_action(action: Optional[str]) -> None:
    """Set navigation action."""
    st.session_state[SessionStateKeys.NAV_ACTION] = action


def clear_nav_action() -> None:
    """Clear navigation action."""
    st.session_state.pop(SessionStateKeys.NAV_ACTION, None)


def get_loaded_session_graph() -> Optional[Dict]:
    """Get loaded session graph data."""
    return st.session_state.get(SessionStateKeys.LOADED_SESSION_GRAPH)


def has_loaded_session_graph() -> bool:
    """Check if session graph is loaded."""
    return get_loaded_session_graph() is not None


def is_export_view_active() -> bool:
    """Check if export view is currently active."""
    return st.session_state.get(SessionStateKeys.NAV_EXPORT) is not None


# =============================================================================
# RENDERING FACADE
# =============================================================================

def render_free_content():
    """
    Render free navigation mode main content.

    Handles:
    - Export view
    - Session graph loading
    - Main navigation interface

    Implements specs 002-003.
    """
    from intuitiveness.streamlit_app import (
        render_session_graph_loader,
        render_free_navigation_main,
        render_export_view,
    )

    # Check for export view
    if is_export_view_active():
        render_export_view()
        return

    # Check if data is available
    has_data = st.session_state.get(SessionStateKeys.RAW_DATA) is not None
    has_graph = has_loaded_session_graph()

    if not has_data and not has_graph:
        _render_no_data_message()
        render_session_graph_loader()
    else:
        render_free_navigation_main()


def _render_no_data_message():
    """Render message when no data is available."""
    st.warning(
        "Please upload data first in **Step-by-Step** mode, "
        "or load a saved session graph below."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Switch to Step-by-Step"):
            st.session_state[SessionStateKeys.NAV_MODE] = 'guided'
            st.rerun()
    with col2:
        st.markdown("**OR**")


def render_free_sidebar():
    """
    Render free navigation sidebar with decision tree.

    Only renders when in free mode with active session.
    """
    from intuitiveness.streamlit_app import render_free_navigation_sidebar

    if is_free_mode() and has_nav_session():
        render_free_navigation_sidebar()


# =============================================================================
# MODE HELPERS
# =============================================================================

def is_free_mode() -> bool:
    """Check if currently in free navigation mode."""
    return st.session_state.get(SessionStateKeys.NAV_MODE) == 'free'


def switch_to_free_mode() -> None:
    """Switch to free navigation mode."""
    st.session_state[SessionStateKeys.NAV_MODE] = 'free'


def switch_to_guided_mode() -> None:
    """Switch to guided mode."""
    st.session_state[SessionStateKeys.NAV_MODE] = 'guided'


# =============================================================================
# NAVIGATION ACTIONS
# =============================================================================

def request_descend(target: str) -> None:
    """
    Request descent to target level/node.

    Args:
        target: Target identifier for descent
    """
    set_nav_action('descend')
    st.session_state[SessionStateKeys.NAV_TARGET] = target


def request_ascend(target: str) -> None:
    """
    Request ascent to target level/node.

    Args:
        target: Target identifier for ascent
    """
    set_nav_action('ascend')
    st.session_state[SessionStateKeys.NAV_TARGET] = target


def request_export() -> None:
    """Request export view."""
    st.session_state[SessionStateKeys.NAV_EXPORT] = True


def clear_export() -> None:
    """Clear export view."""
    st.session_state.pop(SessionStateKeys.NAV_EXPORT, None)


# =============================================================================
# SESSION GRAPH OPERATIONS
# =============================================================================

def load_session_graph(graph_data: Dict) -> None:
    """
    Load session graph from imported data.

    Args:
        graph_data: Session graph dictionary
    """
    st.session_state[SessionStateKeys.LOADED_SESSION_GRAPH] = graph_data
    # Switch to free mode when loading graph
    switch_to_free_mode()


def clear_session_graph() -> None:
    """Clear loaded session graph."""
    st.session_state.pop(SessionStateKeys.LOADED_SESSION_GRAPH, None)
    st.session_state.pop(SessionStateKeys.LOADED_GRAPH_DECISIONS, None)


def get_current_level_from_session() -> Optional[int]:
    """
    Get current abstraction level from navigation session.

    Returns:
        Level number (0-4) or None if no session
    """
    session = get_nav_session()
    if session is None:
        return None
    return session.current_level
