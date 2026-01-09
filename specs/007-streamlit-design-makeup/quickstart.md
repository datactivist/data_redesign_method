# Quickstart: Streamlit Minimalist Design Makeup

**Branch**: `007-streamlit-design-makeup` | **Date**: 2025-12-12

## Prerequisites

- Python 3.11 with `myenv311` virtual environment
- Existing `intuitiveness` package installed
- Streamlit >=1.28.0

## Implementation Order

Follow this sequence for minimal disruption:

### Step 1: Create config.toml (5 min)

Create `.streamlit/config.toml` with theme settings:

```toml
[theme]
primaryColor = "#2563eb"
backgroundColor = "#fafaf9"
secondaryBackgroundColor = "#f5f5f4"
textColor = "#1c1917"
font = "sans serif"
```

**Verify**: Run `streamlit run intuitiveness/streamlit_app.py` and confirm colors change.

### Step 2: Create styles module (30 min)

Create directory structure:
```bash
mkdir -p intuitiveness/styles
touch intuitiveness/styles/__init__.py
touch intuitiveness/styles/chrome.py
touch intuitiveness/styles/typography.py
touch intuitiveness/styles/palette.py
touch intuitiveness/styles/components.py
touch intuitiveness/styles/progress.py
```

Implement each module following `contracts/css-contracts.md`.

### Step 3: Integrate styles into main app (15 min)

In `intuitiveness/streamlit_app.py`, add after `st.set_page_config()`:

```python
from intuitiveness.styles import inject_all_styles

def main():
    st.set_page_config(
        page_title="Data Redesign Method",
        page_icon="...",
        layout="wide"
    )
    inject_all_styles()  # Add this line
    # ... rest of app
```

**Verify**: App loads with custom fonts, hidden chrome, consistent colors.

### Step 4: Remove inline CSS (45 min)

Search for and remove these patterns from `streamlit_app.py`:

1. `inject_right_sidebar_css()` function (~80 lines)
2. `render_progress_bar()` HTML injection
3. `render_ascent_progress_bar()` HTML injection
4. Any `st.markdown("""<style>...""", unsafe_allow_html=True)` blocks

Replace with calls to the new styles module.

### Step 5: Create metric card component (20 min)

Create `intuitiveness/ui/metric_card.py`:

```python
import streamlit as st
from intuitiveness.styles.palette import COLORS

def render_metric_card(
    label: str,
    value: str,
    delta: str = None,
    description: str = None
) -> None:
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        color = COLORS["success"] if delta.startswith("+") else COLORS["error"]
        delta_html = f'<div style="color:{color};font-size:0.875rem;margin-top:0.25rem">{delta}</div>'

    desc_html = ""
    if description:
        desc_html = f'<div style="color:{COLORS["text_secondary"]};font-size:0.875rem;margin-top:0.5rem">{description}</div>'

    st.markdown(f"""
    <div style="
        background: {COLORS["bg_elevated"]};
        border-radius: 0.5rem;
        padding: 1.25rem;
        border: 1px solid {COLORS["border"]};
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    ">
        <div style="
            color: {COLORS["text_secondary"]};
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        ">{label}</div>
        <div style="
            font-size: 1.75rem;
            font-weight: 600;
            color: {COLORS["text_primary"]};
            margin-top: 0.25rem;
        ">{value}</div>
        {delta_html}
        {desc_html}
    </div>
    """, unsafe_allow_html=True)
```

### Step 6: Simplify progress indicator (30 min)

Replace the complex progress sidebar with text-based version:

```python
def render_minimal_progress():
    """Render minimal text-based progress indicator."""
    current = st.session_state.get('current_step', 0)
    is_ascent = st.session_state.get('nav_mode') == 'free'

    st.sidebar.markdown("### Progress")

    if is_ascent:
        levels = [("L0", 0), ("L1", 1), ("L2", 2), ("L3", 3)]
        ascent_level = st.session_state.get('ascent_level', 0)
        for name, idx in levels:
            if idx < ascent_level:
                st.sidebar.markdown(f"~~{name}~~ ✓")
            elif idx == ascent_level:
                st.sidebar.markdown(f"**→ {name}**")
            else:
                st.sidebar.markdown(f"<span style='color:#a8a29e'>{name}</span>",
                                  unsafe_allow_html=True)
    else:
        levels = [("L4", 0), ("L3", 2), ("L2", 3), ("L1", 4), ("L0", 5)]
        for name, step_threshold in levels:
            if current > step_threshold:
                st.sidebar.markdown(f"~~{name}~~ ✓")
            elif current >= step_threshold - 1:
                st.sidebar.markdown(f"**→ {name}**")
            else:
                st.sidebar.markdown(f"<span style='color:#a8a29e'>{name}</span>",
                                  unsafe_allow_html=True)
```

### Step 7: Test & Validate (30 min)

Run visual tests with Playwright MCP:

```bash
# Start app
streamlit run intuitiveness/streamlit_app.py &

# Run visual validation (manual with Playwright MCP)
# - Check chrome elements are hidden
# - Verify font rendering
# - Test progress indicator transitions
# - Validate metric card display at L0
```

## Testing Checklist

- [ ] No hamburger menu visible
- [ ] No footer visible
- [ ] No "Made with Streamlit" badge
- [ ] IBM Plex Sans font loads
- [ ] Warm neutral colors applied
- [ ] Buttons have consistent styling
- [ ] Progress indicator is compact
- [ ] Metric cards display correctly
- [ ] Responsive on 768px viewport

## Rollback

If issues occur, revert to previous commit:
```bash
git checkout 006-playwright-mcp-e2e -- intuitiveness/streamlit_app.py
git checkout 006-playwright-mcp-e2e -- .streamlit/
```

## Files Changed

| File | Action |
|------|--------|
| `.streamlit/config.toml` | CREATE |
| `intuitiveness/styles/__init__.py` | CREATE |
| `intuitiveness/styles/chrome.py` | CREATE |
| `intuitiveness/styles/typography.py` | CREATE |
| `intuitiveness/styles/palette.py` | CREATE |
| `intuitiveness/styles/components.py` | CREATE |
| `intuitiveness/styles/progress.py` | CREATE |
| `intuitiveness/ui/metric_card.py` | CREATE |
| `intuitiveness/streamlit_app.py` | MODIFY |
