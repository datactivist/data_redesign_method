"""
App Initialization Module

Handles session state initialization and recovery (005-session-persistence).

Phase 1 - Code Simplification (011-code-simplification)
Created: 2026-01-09

Spec Traceability:
------------------
- 005-session-persistence: Session recovery, auto-save
- 007-streamlit-design-makeup: Style injection
"""

import streamlit as st
from typing import Optional

from intuitiveness.utils import (
    init_session_state as utils_init_session_state,
    SessionStateKeys,
)
from intuitiveness.persistence import (
    SessionStore,
    SessionCorrupted,
    VersionMismatch,
)
from intuitiveness.ui import (
    RecoveryAction,
    render_recovery_banner,
)
from intuitiveness.styles import inject_all_styles


def init_app_config():
    """
    Initialize Streamlit page configuration.

    Must be called first before any other Streamlit commands.
    """
    st.set_page_config(
        page_title="Data Redesign Method",
        page_icon="🔄",
        layout="wide"
    )


def init_styles():
    """
    Inject all CSS styles for the application.

    Implements 007-streamlit-design-makeup.
    """
    inject_all_styles()


def init_session_state():
    """
    Initialize all session state with defaults.

    Delegates to centralized session manager (Phase 0).
    Session keys defined in utils/session_manager.py with spec traceability.
    """
    utils_init_session_state()


def handle_mode_switching():
    """
    Handle mode switching logic before sidebar widget renders.

    Ensures nav_mode state is consistent with user actions:
    - Switches to free mode when ascent workflow starts
    - Keeps free mode during ascent (loaded_session_graph)
    - Cleans up widget keys to avoid stale state
    """
    # Handle ascent mode switch (before sidebar widget renders)
    if st.session_state.get(SessionStateKeys.SWITCH_TO_ASCENT):
        del st.session_state[SessionStateKeys.SWITCH_TO_ASCENT]
        st.session_state[SessionStateKeys.NAV_MODE] = 'free'
        # Delete widget key so it reinitializes with new nav_mode value
        if SessionStateKeys.MODE_SELECTOR in st.session_state:
            del st.session_state[SessionStateKeys.MODE_SELECTOR]

    # Keep free mode when in ascent workflow (has loaded_session_graph)
    # This prevents the radio widget from resetting nav_mode during ascent
    if (st.session_state.get(SessionStateKeys.LOADED_SESSION_GRAPH) and
        st.session_state.get(SessionStateKeys.NAV_MODE) != 'free'):
        st.session_state[SessionStateKeys.NAV_MODE] = 'free'


def handle_session_recovery(store: SessionStore) -> bool:
    """
    Handle session recovery on first app load.

    Implements 005-session-persistence recovery flow:
    - Checks for saved session
    - Shows recovery banner with options
    - Loads or clears session based on user choice

    Args:
        store: SessionStore instance for persistence operations

    Returns:
        True if recovery handled and app should stop/rerun,
        False if no recovery needed and app should continue
    """
    # Only handle recovery on first load
    if 'session_recovery_handled' in st.session_state:
        return False

    st.session_state.session_recovery_handled = True

    if not store.has_saved_session():
        return False

    info = store.get_session_info()
    if not info:
        return False

    action = render_recovery_banner(info)

    if action == RecoveryAction.CONTINUE:
        try:
            result = store.load()
            if result.warnings:
                for w in result.warnings:
                    st.warning(w)
            st.success(f"Session restored! Resuming from Step {result.wizard_step + 1}")
        except (SessionCorrupted, VersionMismatch) as e:
            st.error(f"Could not restore session: {e}")
            store.clear()
        st.rerun()
        return True

    elif action == RecoveryAction.START_FRESH:
        store.clear()
        st.rerun()
        return True

    elif action == RecoveryAction.PENDING:
        # User hasn't clicked yet - stop here and wait
        st.stop()
        return True

    return False


def is_pure_landing_page() -> bool:
    """
    Check if we're on the pure landing page (no data, no search).

    In this state, sidebar is hidden for minimal SaaS design.

    Returns:
        True if on pure landing page, False otherwise
    """
    return (
        st.session_state.get(SessionStateKeys.NAV_MODE) == 'guided' and
        st.session_state.get(SessionStateKeys.CURRENT_STEP) == 0 and
        st.session_state.get(SessionStateKeys.RAW_DATA) is None and
        not st.session_state.get('datagouv_loaded_datasets') and
        not st.session_state.get('datagouv_results')
    )


def hide_sidebar_on_landing():
    """
    Hide sidebar completely on pure landing page.

    Implements minimal SaaS design (007-streamlit-design-makeup).
    """
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    .stApp > header { display: none !important; }
    </style>
    """, unsafe_allow_html=True)


def run_initialization() -> SessionStore:
    """
    Run all initialization steps in correct order.

    Returns:
        SessionStore instance for use by main app
    """
    init_app_config()
    init_styles()
    init_session_state()
    handle_mode_switching()

    store = SessionStore()
    handle_session_recovery(store)

    return store
