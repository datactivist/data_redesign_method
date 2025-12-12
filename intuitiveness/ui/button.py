"""
Custom Button Components.

Provides semantic button variants with modern styling.
Part of the deep Streamlit design makeover following Gael Penessot's philosophy.
"""

from typing import Optional, Callable, Any
import streamlit as st


def render_button(
    label: str,
    variant: str = "primary",
    icon: Optional[str] = None,
    key: Optional[str] = None,
    disabled: bool = False,
    full_width: bool = False,
    on_click: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
) -> bool:
    """
    Render a styled button with semantic variants.

    This wraps st.button with custom styling classes. The actual button
    functionality (callbacks, state) comes from Streamlit.

    Args:
        label: Button text
        variant: One of "primary", "secondary", "ghost", "danger"
        icon: Optional icon to prepend (emoji or unicode)
        key: Streamlit widget key
        disabled: Whether button is disabled
        full_width: Whether button spans full container width
        on_click: Optional callback function
        args: Tuple of args for callback
        kwargs: Dict of kwargs for callback

    Returns:
        bool: True if button was clicked

    Example:
        if render_button("Save Changes", variant="primary", icon="💾"):
            save_data()

        render_button("Cancel", variant="ghost")
        render_button("Delete", variant="danger", icon="🗑️")
    """
    # Build button label with optional icon
    display_label = f"{icon} {label}" if icon else label

    # Map variant to Streamlit button type
    # Streamlit supports: "primary" and "secondary" (default)
    if variant == "primary":
        button_type = "primary"
    else:
        button_type = "secondary"

    # Add custom class via key suffix for CSS targeting
    # We'll apply extra styling via CSS selectors
    widget_key = key or f"btn_{label.lower().replace(' ', '_')}"

    # Render the button
    clicked = st.button(
        display_label,
        key=widget_key,
        disabled=disabled,
        use_container_width=full_width,
        type=button_type,
        on_click=on_click,
        args=args or (),
        kwargs=kwargs or {},
    )

    return clicked


def render_button_row(
    buttons: list[dict],
    spacing: str = "normal",
) -> dict[str, bool]:
    """
    Render a horizontal row of buttons.

    Args:
        buttons: List of button configs, each with keys:
            - label (required): Button text
            - variant: "primary", "secondary", "ghost", "danger"
            - icon: Optional icon
            - key: Widget key
            - disabled: Whether disabled
        spacing: "compact", "normal", or "wide"

    Returns:
        dict: Mapping of button labels to clicked state

    Example:
        results = render_button_row([
            {"label": "Back", "variant": "secondary", "icon": "←"},
            {"label": "Next", "variant": "primary", "icon": "→"},
        ])
        if results["Next"]:
            go_next()
    """
    # Determine column ratios based on spacing
    num_buttons = len(buttons)
    if spacing == "compact":
        cols = st.columns([1] * num_buttons)
    elif spacing == "wide":
        cols = st.columns([1] + [2] * (num_buttons - 1) + [1])
        cols = cols[1:-1]  # Remove padding columns
    else:  # normal
        cols = st.columns(num_buttons)

    results = {}
    for col, btn_config in zip(cols, buttons):
        with col:
            label = btn_config["label"]
            results[label] = render_button(
                label=label,
                variant=btn_config.get("variant", "secondary"),
                icon=btn_config.get("icon"),
                key=btn_config.get("key"),
                disabled=btn_config.get("disabled", False),
                full_width=True,
            )

    return results


# Convenience functions for common button patterns
def primary_button(label: str, icon: Optional[str] = None, **kwargs) -> bool:
    """Render a primary action button."""
    return render_button(label, variant="primary", icon=icon, **kwargs)


def secondary_button(label: str, icon: Optional[str] = None, **kwargs) -> bool:
    """Render a secondary action button."""
    return render_button(label, variant="secondary", icon=icon, **kwargs)


def ghost_button(label: str, icon: Optional[str] = None, **kwargs) -> bool:
    """Render a ghost/text-only button."""
    return render_button(label, variant="ghost", icon=icon, **kwargs)


def danger_button(label: str, icon: Optional[str] = None, **kwargs) -> bool:
    """Render a destructive action button."""
    return render_button(label, variant="danger", icon=icon, **kwargs)
