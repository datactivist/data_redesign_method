"""
Custom Alert Component.

Replaces st.info/warning/success/error with design-system-aligned alerts.
Part of the deep Streamlit design makeover following Gael Penessot's philosophy.
"""

from typing import Optional
import streamlit as st


# Icons for each alert type - using Unicode for simplicity
ALERT_ICONS = {
    "info": "\u2139\ufe0f",      # Information icon
    "success": "\u2705",         # Check mark
    "warning": "\u26a0\ufe0f",   # Warning sign
    "error": "\u274c",           # Cross mark
    "tip": "\ud83d\udca1",       # Light bulb
}


def render_alert(
    message: str,
    alert_type: str = "info",
    title: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """
    Render a custom-styled alert box.

    Replaces Streamlit's native st.info/warning/success/error with
    subtle, design-system-aligned alerts.

    Args:
        message: The alert message content
        alert_type: One of "info", "success", "warning", "error", "tip"
        title: Optional title displayed above the message
        icon: Optional custom icon (uses default for type if not provided)

    Example:
        render_alert("Operation completed successfully!", "success")
        render_alert("Please review before continuing", "warning", title="Attention")
    """
    # Validate and default alert type
    valid_types = {"info", "success", "warning", "error", "tip"}
    if alert_type not in valid_types:
        alert_type = "info"

    # Get icon
    display_icon = icon or ALERT_ICONS.get(alert_type, ALERT_ICONS["info"])

    # Build HTML
    title_html = f'<div class="custom-alert-title">{title}</div>' if title else ""

    html = f"""
    <div class="custom-alert custom-alert-{alert_type}">
        <div class="custom-alert-icon">{display_icon}</div>
        <div class="custom-alert-content">
            {title_html}
            <div class="custom-alert-message">{message}</div>
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


# Convenience functions matching Streamlit's API
def info(message: str, title: Optional[str] = None, icon: Optional[str] = None) -> None:
    """Render an info alert. Drop-in replacement for st.info()."""
    render_alert(message, "info", title, icon)


def success(message: str, title: Optional[str] = None, icon: Optional[str] = None) -> None:
    """Render a success alert. Drop-in replacement for st.success()."""
    render_alert(message, "success", title, icon)


def warning(message: str, title: Optional[str] = None, icon: Optional[str] = None) -> None:
    """Render a warning alert. Drop-in replacement for st.warning()."""
    render_alert(message, "warning", title, icon)


def error(message: str, title: Optional[str] = None, icon: Optional[str] = None) -> None:
    """Render an error alert. Drop-in replacement for st.error()."""
    render_alert(message, "error", title, icon)


def tip(message: str, title: Optional[str] = None, icon: Optional[str] = None) -> None:
    """Render a tip alert. Bonus type not in Streamlit."""
    render_alert(message, "tip", title, icon)
