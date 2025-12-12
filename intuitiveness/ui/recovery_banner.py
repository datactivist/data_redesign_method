"""
Recovery banner UI component.

Shows session recovery notification with continue/fresh options.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

import streamlit as st

from intuitiveness.persistence import SessionInfo


class RecoveryAction(Enum):
    """User action on recovery banner."""
    CONTINUE = "continue"
    START_FRESH = "start_fresh"
    DISMISS = "dismiss"
    PENDING = "pending"


def format_time_ago(timestamp: datetime) -> str:
    """
    Format a timestamp as a human-readable "time ago" string.

    Args:
        timestamp: The datetime to format

    Returns:
        Human-readable string like "2 hours ago" or "yesterday"
    """
    now = datetime.now()
    diff = now - timestamp

    seconds = diff.total_seconds()

    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif seconds < 172800:
        return "yesterday"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"


def render_recovery_banner(session_info: SessionInfo) -> RecoveryAction:
    """
    Render a session recovery banner at the top of the app.

    Shows:
    - "Welcome back! Your previous session was saved [time ago]"
    - "You were at Step X with Y files"
    - [Continue] [Start Fresh] buttons

    Args:
        session_info: Metadata about saved session

    Returns:
        RecoveryAction indicating user's choice
    """
    # Create a container for the banner
    with st.container():
        # Use info style for the banner
        st.markdown("""
        <style>
        .recovery-banner {
            background-color: #f0f7ff;
            border: 1px solid #3b82f6;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .recovery-title {
            color: #1e40af;
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 8px;
        }
        .recovery-info {
            color: #374151;
            margin-bottom: 12px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Banner content
        time_ago = format_time_ago(session_info.timestamp)

        st.info(f"""
        **Welcome back!** Your previous session was saved {time_ago}.

        You were at **Step {session_info.wizard_step + 1}** with **{session_info.file_count} file(s)** uploaded.
        """)

        # Action buttons
        col1, col2, col3 = st.columns([2, 2, 4])

        with col1:
            if st.button("Continue where I left off", type="primary", key="recovery_continue"):
                return RecoveryAction.CONTINUE

        with col2:
            if st.button("Start fresh", key="recovery_fresh"):
                return RecoveryAction.START_FRESH

        with col3:
            st.caption(f"Session version: {session_info.version} | Size: {session_info.total_size_bytes / 1024:.1f} KB")

    return RecoveryAction.PENDING


def render_start_fresh_button() -> bool:
    """
    Render a "Start Fresh" button in the sidebar.

    Returns:
        True if button was clicked
    """
    return st.sidebar.button(
        "Start Fresh",
        help="Clear all uploaded files and restart from the beginning",
        key="sidebar_start_fresh"
    )


def render_start_fresh_confirmation() -> bool:
    """
    Render a confirmation dialog for starting fresh.

    Returns:
        True if user confirmed, False otherwise
    """
    st.warning("This will clear all your uploaded files and progress. Are you sure?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Yes, start fresh", type="primary", key="confirm_fresh"):
            return True

    with col2:
        if st.button("Cancel", key="cancel_fresh"):
            return False

    return False
