# CSS Contracts: Style Module Specifications

**Branch**: `007-streamlit-design-makeup` | **Date**: 2025-12-12

## Overview

This document specifies the CSS contracts for each style module. Each module MUST export a single string constant containing valid CSS.

---

## Contract: HIDE_CHROME_CSS

**Module**: `intuitiveness/styles/chrome.py`
**Export**: `HIDE_CHROME_CSS: str`

**Purpose**: Hide default Streamlit UI elements

**Required Selectors**:
```css
/* MUST hide these elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.viewerBadge_container__r5tak { display: none; }

/* MUST adjust layout after hiding header */
.block-container { padding-top: 2rem !important; }
```

**Test**: Application loads without visible hamburger menu, footer, or "Made with Streamlit" badge.

---

## Contract: TYPOGRAPHY_CSS

**Module**: `intuitiveness/styles/typography.py`
**Export**: `TYPOGRAPHY_CSS: str`

**Purpose**: Load and apply custom typography

**Required Rules**:
```css
/* MUST import font */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

/* MUST set base font */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* MUST style headings */
h1 { font-weight: 600; letter-spacing: -0.02em; }
h2, h3 { font-weight: 500; }

/* MUST style captions */
.stCaption { color: #57534e; font-size: 0.875rem; }
```

**Test**: All text renders in IBM Plex Sans; fallback to system fonts if CDN fails.

---

## Contract: PALETTE_CSS

**Module**: `intuitiveness/styles/palette.py`
**Export**: `PALETTE_CSS: str`

**Purpose**: Define CSS custom properties for color palette

**Required Variables**:
```css
:root {
    --color-bg-primary: #fafaf9;
    --color-bg-secondary: #f5f5f4;
    --color-bg-elevated: #ffffff;
    --color-text-primary: #1c1917;
    --color-text-secondary: #57534e;
    --color-text-muted: #a8a29e;
    --color-accent: #2563eb;
    --color-accent-hover: #1d4ed8;
    --color-border: #e7e5e4;
    --color-success: #22c55e;
    --color-error: #ef4444;
}
```

**Test**: CSS inspector shows custom properties defined on :root.

---

## Contract: COMPONENT_CSS

**Module**: `intuitiveness/styles/components.py`
**Export**: `COMPONENT_CSS: str`

**Purpose**: Style interactive Streamlit components

**Required Rules**:
```css
/* Primary buttons */
.stButton > button[kind="primary"] {
    background: var(--color-accent) !important;
    border: none !important;
    font-weight: 500 !important;
    border-radius: 0.5rem !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--color-accent-hover) !important;
}

/* Secondary buttons */
.stButton > button:not([kind="primary"]) {
    background: transparent !important;
    border: 1px solid var(--color-border) !important;
}

/* Expanders */
.stExpander {
    border: 1px solid var(--color-border) !important;
    border-radius: 0.5rem !important;
    box-shadow: none !important;
}

/* Inputs */
.stTextInput input, .stSelectbox select {
    border-radius: 0.5rem !important;
    border-color: var(--color-border) !important;
}
```

**Test**: Buttons, expanders, and inputs display consistent styling across all screens.

---

## Contract: PROGRESS_CSS

**Module**: `intuitiveness/styles/progress.py`
**Export**: `PROGRESS_CSS: str`

**Purpose**: Style the simplified progress indicator

**Required Rules**:
```css
/* Progress indicator container */
.progress-indicator {
    font-family: var(--font-family);
    padding: 1rem;
}

/* Level markers */
.progress-level {
    display: flex;
    align-items: center;
    padding: 0.5rem 0;
    color: var(--color-text-muted);
}

.progress-level.completed {
    color: var(--color-success);
}

.progress-level.current {
    color: var(--color-accent);
    font-weight: 500;
}
```

**Test**: Progress indicator is compact, readable, and updates correctly on level transitions.

---

## Contract: render_metric_card()

**Module**: `intuitiveness/ui/metric_card.py`
**Export**: `render_metric_card(label: str, value: str, delta: str = None, description: str = None) -> None`

**Purpose**: Render a styled metric card using st.html()

**Required Output HTML Structure**:
```html
<div class="metric-card" style="
    background: var(--color-bg-elevated);
    border-radius: 0.5rem;
    padding: 1.25rem;
    border: 1px solid var(--color-border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
">
    <div class="metric-label" style="
        color: var(--color-text-secondary);
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">{label}</div>
    <div class="metric-value" style="
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--color-text-primary);
        margin-top: 0.25rem;
    ">{value}</div>
    <!-- Optional delta -->
    <div class="metric-delta" style="
        font-size: 0.875rem;
        color: var(--color-success);
        margin-top: 0.25rem;
    ">{delta}</div>
    <!-- Optional description -->
    <div class="metric-description" style="
        font-size: 0.875rem;
        color: var(--color-text-secondary);
        margin-top: 0.5rem;
    ">{description}</div>
</div>
```

**Test**: Metric cards render with consistent styling; values are clearly readable.

---

## Integration Contract

**Module**: `intuitiveness/styles/__init__.py`

**Required Exports**:
```python
from .chrome import HIDE_CHROME_CSS
from .typography import TYPOGRAPHY_CSS
from .palette import PALETTE_CSS
from .components import COMPONENT_CSS
from .progress import PROGRESS_CSS

# Combined CSS for single injection
ALL_STYLES = f"""
<style>
{PALETTE_CSS}
{TYPOGRAPHY_CSS}
{HIDE_CHROME_CSS}
{COMPONENT_CSS}
{PROGRESS_CSS}
</style>
"""

def inject_all_styles():
    """Inject all custom styles into the Streamlit app."""
    import streamlit as st
    st.markdown(ALL_STYLES, unsafe_allow_html=True)
```

**Usage in streamlit_app.py**:
```python
from intuitiveness.styles import inject_all_styles

def main():
    st.set_page_config(...)
    inject_all_styles()  # MUST be called early in main()
    # ... rest of app
```

**Test**: Single function call applies all styling; no duplicate style injections.
