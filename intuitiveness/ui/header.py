"""
SaaS-style page header component.

Replaces the default st.title() with a professional header that includes:
- Clean typography without emoji clutter
- Subtitle for context
- Optional breadcrumb navigation
- Optional action buttons area

Following Gael Penessot's DataGyver philosophy.
"""

import streamlit as st
from typing import List, Optional, Tuple


def render_page_header(
    title: str,
    subtitle: str = None,
    breadcrumbs: List[Tuple[str, str]] = None,
    show_accent: bool = True,
) -> None:
    """
    Render a professional SaaS-style page header.

    Instead of: st.title("🔄 Interactive Data Redesign Method")
    Use: render_page_header("Data Redesign", "Transform chaos to clarity")

    Args:
        title: Main page title (no emoji recommended)
        subtitle: Optional descriptive subtitle
        breadcrumbs: Optional list of (label, url) tuples for navigation trail
        show_accent: Whether to show the Klein Blue gradient accent line
    """
    header_class = "page-header-accent" if show_accent else "page-header"

    html_parts = [f'<div class="{header_class}">']

    # Breadcrumb trail
    if breadcrumbs:
        html_parts.append('<nav class="breadcrumb">')
        for i, (label, url) in enumerate(breadcrumbs):
            if i > 0:
                html_parts.append('<span class="breadcrumb-separator">/</span>')
            if url:
                html_parts.append(
                    f'<a href="{url}" class="breadcrumb-item">{label}</a>'
                )
            else:
                html_parts.append(
                    f'<span class="breadcrumb-item-active">{label}</span>'
                )
        html_parts.append("</nav>")

    # Title
    html_parts.append(f'<h1 class="page-header-title">{title}</h1>')

    # Subtitle
    if subtitle:
        html_parts.append(f'<p class="page-header-subtitle">{subtitle}</p>')

    html_parts.append("</div>")

    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    with_dot: bool = True,
) -> None:
    """
    Render a section header to replace st.header().

    Args:
        title: Section title text
        subtitle: Optional descriptive text below the title
        with_dot: Whether to show the Klein Blue accent dot
    """
    header_class = "section-header-dot" if with_dot else "section-header"
    subtitle_html = f'<p style="color: #64748b; font-size: 0.9rem; margin-top: 4px; margin-bottom: 16px;">{subtitle}</p>' if subtitle else ''

    st.markdown(
        f'<h2 class="{header_class}">{title}</h2>{subtitle_html}',
        unsafe_allow_html=True,
    )


def render_card_header(title: str) -> None:
    """
    Render a header for use inside cards.

    Args:
        title: Header text
    """
    st.markdown(
        f'<div style="font-size: 1rem; font-weight: 600; color: var(--color-text-primary, #0f172a); margin-bottom: 12px;">{title}</div>',
        unsafe_allow_html=True,
    )
