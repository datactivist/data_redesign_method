"""
Custom Accordion Component.

Provides a fully custom HTML/CSS accordion as an alternative to st.expander.
Part of the deep Streamlit design makeover following Gael Penessot's philosophy.
"""

from typing import Optional
import streamlit as st
import uuid


def render_accordion(
    title: str,
    content: str,
    expanded: bool = False,
    icon: Optional[str] = None,
    badge: Optional[str] = None,
) -> None:
    """
    Render a custom HTML accordion section.

    Unlike st.expander, this is pure HTML/CSS and doesn't maintain
    Streamlit widget state. Use for static content display.

    Args:
        title: Accordion header text
        content: HTML content to display when expanded
        expanded: Whether to start expanded
        icon: Optional icon to show before title
        badge: Optional badge text (e.g., count) to show on right

    Example:
        render_accordion(
            "View Details",
            "<p>Some detailed content here...</p>",
            expanded=True,
            icon="📋",
            badge="3 items"
        )
    """
    accordion_id = f"accordion_{uuid.uuid4().hex[:8]}"
    checked = "checked" if expanded else ""

    icon_html = f'<span class="accordion-icon">{icon}</span>' if icon else ""
    badge_html = f'<span class="accordion-badge">{badge}</span>' if badge else ""

    html = f"""
    <div class="custom-accordion">
        <input type="checkbox" id="{accordion_id}" class="accordion-toggle" {checked}>
        <label for="{accordion_id}" class="accordion-header">
            <span class="accordion-title">
                {icon_html}
                {title}
            </span>
            {badge_html}
            <span class="accordion-chevron">›</span>
        </label>
        <div class="accordion-content">
            <div class="accordion-inner">
                {content}
            </div>
        </div>
    </div>

    <style>
    .custom-accordion {{
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: 0.5rem;
        background: var(--color-bg-elevated, #ffffff);
        margin: 0.5rem 0;
        overflow: hidden;
    }}

    .accordion-toggle {{
        display: none;
    }}

    .accordion-header {{
        display: flex;
        align-items: center;
        padding: 0.875rem 1rem;
        cursor: pointer;
        font-weight: 500;
        color: var(--color-text-primary, #0f172a);
        transition: background 0.15s ease;
    }}

    .accordion-header:hover {{
        background: var(--color-bg-secondary, #f1f5f9);
    }}

    .accordion-title {{
        flex: 1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .accordion-icon {{
        font-size: 1.125rem;
    }}

    .accordion-badge {{
        font-size: 0.75rem;
        padding: 0.125rem 0.5rem;
        background: var(--color-accent-light, #eff6ff);
        color: var(--color-accent, #3b82f6);
        border-radius: 9999px;
        margin-right: 0.5rem;
    }}

    .accordion-chevron {{
        font-size: 1.25rem;
        color: var(--color-text-muted, #94a3b8);
        transition: transform 0.2s ease;
        transform: rotate(0deg);
    }}

    .accordion-toggle:checked + .accordion-header .accordion-chevron {{
        transform: rotate(90deg);
    }}

    .accordion-content {{
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease;
    }}

    .accordion-toggle:checked ~ .accordion-content {{
        max-height: 2000px;  /* Large enough for most content */
    }}

    .accordion-inner {{
        padding: 0 1rem 1rem 1rem;
        border-top: 1px solid var(--color-border, #e2e8f0);
    }}
    </style>
    """

    st.markdown(html, unsafe_allow_html=True)


def render_accordion_group(
    items: list[dict],
    allow_multiple: bool = True,
) -> None:
    """
    Render a group of accordion sections.

    Args:
        items: List of dicts with keys: title, content, expanded (optional),
               icon (optional), badge (optional)
        allow_multiple: If False, only one accordion can be open at a time
                       (Note: pure CSS limitation means this uses JS)

    Example:
        render_accordion_group([
            {"title": "Section 1", "content": "<p>Content 1</p>"},
            {"title": "Section 2", "content": "<p>Content 2</p>", "expanded": True},
        ])
    """
    for item in items:
        render_accordion(
            title=item["title"],
            content=item["content"],
            expanded=item.get("expanded", False),
            icon=item.get("icon"),
            badge=item.get("badge"),
        )
