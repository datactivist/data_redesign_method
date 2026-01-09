"""
Alert Component - Gael Penessot Style.

Uses NATIVE Streamlit alerts (st.info/warning/success/error) which are then
styled via CSS injection. This approach is more reliable than custom HTML divs.

Following Gael Penessot's DataGyver philosophy: style the framework, don't fight it.
"""

from typing import Optional
import streamlit as st


def info(message: str, title: Optional[str] = None, icon: str = "ℹ️") -> None:
    """
    Render an info alert using native Streamlit styling.

    Args:
        message: The alert message content
        title: Optional title (prepended to message)
        icon: Custom icon (default: info icon)
    """
    display_msg = f"**{title}**\n\n{message}" if title else message
    st.info(display_msg, icon=icon)


def success(message: str, title: Optional[str] = None, icon: str = "✅") -> None:
    """
    Render a success alert using native Streamlit styling.

    Args:
        message: The alert message content
        title: Optional title (prepended to message)
        icon: Custom icon (default: check mark)
    """
    display_msg = f"**{title}**\n\n{message}" if title else message
    st.success(display_msg, icon=icon)


def warning(message: str, title: Optional[str] = None, icon: str = "⚠️") -> None:
    """
    Render a warning alert using native Streamlit styling.

    Args:
        message: The alert message content
        title: Optional title (prepended to message)
        icon: Custom icon (default: warning sign)
    """
    display_msg = f"**{title}**\n\n{message}" if title else message
    st.warning(display_msg, icon=icon)


def error(message: str, title: Optional[str] = None, icon: str = "❌") -> None:
    """
    Render an error alert using native Streamlit styling.

    Args:
        message: The alert message content
        title: Optional title (prepended to message)
        icon: Custom icon (default: cross mark)
    """
    display_msg = f"**{title}**\n\n{message}" if title else message
    st.error(display_msg, icon=icon)


def tip(message: str, title: Optional[str] = None, icon: str = "💡") -> None:
    """
    Render a tip alert (uses info styling with bulb icon).

    Args:
        message: The alert message content
        title: Optional title (prepended to message)
        icon: Custom icon (default: light bulb)
    """
    display_msg = f"**{title}**\n\n{message}" if title else message
    st.info(display_msg, icon=icon)


# Legacy function for backwards compatibility
def render_alert(
    message: str,
    alert_type: str = "info",
    title: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """Legacy render_alert function - redirects to native functions."""
    funcs = {
        "info": info,
        "success": success,
        "warning": warning,
        "error": error,
        "tip": tip,
    }
    func = funcs.get(alert_type, info)
    func(message, title, icon)
