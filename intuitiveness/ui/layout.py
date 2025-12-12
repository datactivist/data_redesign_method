"""
Layout helper components for SaaS-ready UI.

Provides card containers, section wrappers, and layout utilities
that integrate with Streamlit's declarative model.

Following Gael Penessot's DataGyver philosophy.
"""

import streamlit as st
from typing import Optional
from contextlib import contextmanager


@contextmanager
def card(
    title: str = None,
    variant: str = "default",
    padding: str = "24px",
):
    """
    Context manager to wrap content in a styled card container.

    Usage:
        with card(title="Upload Your Data"):
            st.file_uploader("Choose a file")
            st.button("Process")

    Args:
        title: Optional card header title
        variant: Card style - "default", "compact", "accent", "interactive"
        padding: CSS padding value
    """
    # Map variant to CSS class
    variant_classes = {
        "default": "content-card",
        "compact": "content-card-compact",
        "accent": "content-card-accent",
        "interactive": "content-card-interactive",
    }
    css_class = variant_classes.get(variant, "content-card")

    # Open card container
    st.markdown(
        f'<div class="{css_class}" style="padding: {padding};">',
        unsafe_allow_html=True,
    )

    # Render title if provided
    if title:
        st.markdown(
            f"""<div style="
                font-size: 1rem;
                font-weight: 600;
                color: var(--color-text-primary, #0f172a);
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 1px solid var(--color-border, #e2e8f0);
            ">{title}</div>""",
            unsafe_allow_html=True,
        )

    # Yield control to the with block content
    yield

    # Close card container
    st.markdown("</div>", unsafe_allow_html=True)


def render_card(
    title: str = None,
    content: str = None,
    variant: str = "default",
) -> None:
    """
    Render a simple card with optional title and text content.

    For complex content, use the card() context manager instead.

    Args:
        title: Optional card header
        content: Optional text content (supports markdown)
        variant: Card style - "default", "compact", "accent"
    """
    variant_classes = {
        "default": "content-card",
        "compact": "content-card-compact",
        "accent": "content-card-accent",
    }
    css_class = variant_classes.get(variant, "content-card")

    html_parts = [f'<div class="{css_class}">']

    if title:
        html_parts.append(
            f"""<div style="
                font-size: 1rem;
                font-weight: 600;
                color: var(--color-text-primary, #0f172a);
                margin-bottom: 12px;
            ">{title}</div>"""
        )

    if content:
        html_parts.append(
            f"""<div style="
                color: var(--color-text-secondary, #475569);
                line-height: 1.6;
            ">{content}</div>"""
        )

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def separator(style: str = "gradient") -> None:
    """
    Render a custom section separator.

    Args:
        style: Separator style - "gradient" (default), "solid", "dotted"
    """
    if style == "gradient":
        st.markdown(
            '<div class="section-separator"></div>',
            unsafe_allow_html=True,
        )
    elif style == "solid":
        st.markdown(
            """<div style="
                height: 1px;
                background: var(--color-border, #e2e8f0);
                margin: 24px 0;
            "></div>""",
            unsafe_allow_html=True,
        )
    elif style == "dotted":
        st.markdown(
            """<div style="
                height: 1px;
                border-bottom: 1px dotted var(--color-border, #e2e8f0);
                margin: 24px 0;
            "></div>""",
            unsafe_allow_html=True,
        )


def spacer(size: str = "md") -> None:
    """
    Add vertical spacing.

    Args:
        size: Space size - "xs" (8px), "sm" (16px), "md" (24px), "lg" (32px), "xl" (48px)
    """
    sizes = {
        "xs": "8px",
        "sm": "16px",
        "md": "24px",
        "lg": "32px",
        "xl": "48px",
    }
    height = sizes.get(size, "24px")
    st.markdown(f'<div style="height: {height};"></div>', unsafe_allow_html=True)


def two_column_layout(left_content, right_content, ratio: tuple = (1, 1)):
    """
    Create a two-column layout with specified ratio.

    Args:
        left_content: Function to render left column content
        right_content: Function to render right column content
        ratio: Column width ratio as tuple, e.g., (2, 1) for 2:1 ratio
    """
    col1, col2 = st.columns(ratio)
    with col1:
        left_content()
    with col2:
        right_content()
