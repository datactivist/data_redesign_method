# Data Model: Design Tokens & Style Entities

**Branch**: `007-streamlit-design-makeup` | **Date**: 2025-12-12

## Overview

This document defines the design tokens and style entities used throughout the Streamlit minimalist design system. These tokens ensure consistency and maintainability.

## Design Tokens

### Color Palette

```python
# intuitiveness/styles/palette.py

COLORS = {
    # Backgrounds
    "bg_primary": "#fafaf9",       # Main background (warm off-white)
    "bg_secondary": "#f5f5f4",     # Sidebar, cards (stone-50)
    "bg_elevated": "#ffffff",      # Cards, modals (white)

    # Text
    "text_primary": "#1c1917",     # Main text (stone-900)
    "text_secondary": "#57534e",   # Captions, hints (stone-600)
    "text_muted": "#a8a29e",       # Disabled, placeholders (stone-400)

    # Accent
    "accent": "#2563eb",           # Primary actions, links (blue-600)
    "accent_hover": "#1d4ed8",     # Hover state (blue-700)
    "accent_subtle": "#dbeafe",    # Accent background (blue-100)

    # Borders
    "border": "#e7e5e4",           # Default borders (stone-200)
    "border_subtle": "#f5f5f4",    # Subtle dividers (stone-100)

    # States
    "success": "#22c55e",          # Success (green-500)
    "success_bg": "#f0fdf4",       # Success background (green-50)
    "warning": "#f59e0b",          # Warning (amber-500)
    "warning_bg": "#fffbeb",       # Warning background (amber-50)
    "error": "#ef4444",            # Error (red-500)
    "error_bg": "#fef2f2",         # Error background (red-50)
}
```

### Typography

```python
# intuitiveness/styles/typography.py

TYPOGRAPHY = {
    "font_family": "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "font_family_mono": "'IBM Plex Mono', 'Menlo', 'Monaco', monospace",

    # Font sizes (rem)
    "text_xs": "0.75rem",    # 12px
    "text_sm": "0.875rem",   # 14px
    "text_base": "1rem",     # 16px
    "text_lg": "1.125rem",   # 18px
    "text_xl": "1.25rem",    # 20px
    "text_2xl": "1.5rem",    # 24px
    "text_3xl": "1.875rem",  # 30px

    # Font weights
    "font_normal": "400",
    "font_medium": "500",
    "font_semibold": "600",

    # Line heights
    "leading_tight": "1.25",
    "leading_normal": "1.5",
    "leading_relaxed": "1.625",

    # Letter spacing
    "tracking_tight": "-0.02em",
    "tracking_normal": "0",
    "tracking_wide": "0.05em",
}
```

### Spacing

```python
# intuitiveness/styles/spacing.py

SPACING = {
    "space_0": "0",
    "space_1": "0.25rem",   # 4px
    "space_2": "0.5rem",    # 8px
    "space_3": "0.75rem",   # 12px
    "space_4": "1rem",      # 16px
    "space_5": "1.25rem",   # 20px
    "space_6": "1.5rem",    # 24px
    "space_8": "2rem",      # 32px
    "space_10": "2.5rem",   # 40px
    "space_12": "3rem",     # 48px
}
```

### Border Radius

```python
# intuitiveness/styles/borders.py

RADIUS = {
    "radius_none": "0",
    "radius_sm": "0.25rem",   # 4px
    "radius_md": "0.5rem",    # 8px - default for cards
    "radius_lg": "0.75rem",   # 12px
    "radius_xl": "1rem",      # 16px
    "radius_full": "9999px",  # Pills
}
```

### Shadows

```python
# intuitiveness/styles/shadows.py

SHADOWS = {
    "shadow_none": "none",
    "shadow_sm": "0 1px 2px rgba(0, 0, 0, 0.04)",
    "shadow_md": "0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04)",
    "shadow_lg": "0 4px 6px rgba(0, 0, 0, 0.05), 0 2px 4px rgba(0, 0, 0, 0.03)",
}
```

## Style Entities

### Theme Configuration Entity

**Purpose**: Centralized Streamlit config.toml settings

```toml
# .streamlit/config.toml

[theme]
primaryColor = "#2563eb"
backgroundColor = "#fafaf9"
secondaryBackgroundColor = "#f5f5f4"
textColor = "#1c1917"
font = "sans serif"
```

### Style Module Entity

**Purpose**: Collection of CSS injection strings

| Module | Responsibility | Exported Constant |
|--------|---------------|-------------------|
| `chrome.py` | Hide Streamlit default UI | `HIDE_CHROME_CSS` |
| `typography.py` | Font loading, text styles | `TYPOGRAPHY_CSS` |
| `palette.py` | Color definitions, CSS variables | `PALETTE_CSS` |
| `components.py` | Button, input, expander styles | `COMPONENT_CSS` |
| `progress.py` | Progress indicator styles | `PROGRESS_CSS` |

### Metric Card Entity

**Purpose**: Reusable data display component

**Attributes**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `label` | str | Yes | Caption above the value |
| `value` | str | Yes | Primary display value |
| `delta` | str | No | Change indicator (e.g., "+5%") |
| `delta_positive` | bool | No | Whether delta is positive (affects color) |
| `description` | str | No | Additional context below value |

**Validation Rules**:
- `label` must be non-empty, max 50 characters
- `value` must be non-empty
- `delta` should include sign prefix if numeric

### Progress State Entity

**Purpose**: Workflow level indicator state

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| `current_level` | int | Current abstraction level (0-4) |
| `direction` | str | "descent" or "ascent" |
| `completed_levels` | list[int] | Levels already visited |

**State Transitions**:
- Descent: L4 → L3 → L2 → L1 → L0
- Ascent: L0 → L1 → L2 → L3

## CSS Variable Contract

All CSS customizations MUST use these CSS custom properties:

```css
:root {
    /* Colors */
    --color-bg-primary: #fafaf9;
    --color-bg-secondary: #f5f5f4;
    --color-text-primary: #1c1917;
    --color-text-secondary: #57534e;
    --color-accent: #2563eb;
    --color-border: #e7e5e4;

    /* Typography */
    --font-family: 'IBM Plex Sans', sans-serif;
    --font-size-base: 1rem;

    /* Spacing */
    --space-unit: 0.25rem;

    /* Radius */
    --radius-default: 0.5rem;
}
```
