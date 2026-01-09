# Research: Streamlit Minimalist Design Makeup

**Branch**: `007-streamlit-design-makeup` | **Date**: 2025-12-12

## Research Tasks

### 1. Streamlit config.toml Theming Capabilities

**Question**: What theming options are available in Streamlit's native config.toml?

**Decision**: Use Streamlit 1.28+ native theming via `.streamlit/config.toml`

**Rationale**: Streamlit provides extensive theming options including:
- `primaryColor`: Main accent color for interactive elements
- `backgroundColor`: Main background color
- `secondaryBackgroundColor`: Sidebar and widget backgrounds
- `textColor`: Primary text color
- `font`: Base font family (supports "sans serif", "serif", "monospace", or custom)
- `baseRadius`: Border radius for widgets (new in 1.28+)

**Alternatives Considered**:
- CSS-only approach without config.toml → Rejected because config.toml provides native integration with widgets
- Custom theme package (streamlit-themes) → Rejected because native solution is sufficient

---

### 2. CSS Injection Method for Hiding Chrome

**Question**: What is the most reliable method to hide Streamlit's default UI elements?

**Decision**: Use `st.markdown()` with `unsafe_allow_html=True` for CSS injection

**Rationale**:
- `st.html()` is newer but `st.markdown()` with unsafe_allow_html has broader compatibility
- CSS selectors for hiding Streamlit chrome are well-documented:
  - `#MainMenu {visibility: hidden;}` - Hamburger menu
  - `footer {visibility: hidden;}` - Footer
  - `header {visibility: hidden;}` - Header
  - `.viewerBadge_container__* {display: none;}` - "Made with Streamlit" badge

**Alternatives Considered**:
- `st.html()` only → Works but less compatible with older Streamlit versions
- JavaScript injection → Overly complex for CSS-only changes

---

### 3. Font Loading Strategy

**Question**: How to reliably load IBM Plex Sans from Google Fonts?

**Decision**: Use CSS `@import` within injected styles, with system font fallback

**Rationale**:
- Google Fonts CDN is reliable and fast
- `@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap')` loads efficiently
- Fallback stack: `'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`

**Alternatives Considered**:
- Self-hosted fonts → Adds complexity without significant benefit
- Multiple font families → Rejected in favor of single family with weights

---

### 4. Color Palette Selection

**Question**: What warm neutral colors align with the minimalist philosophy?

**Decision**: Implement the following palette:

| Token | Value | Usage |
|-------|-------|-------|
| `--bg-primary` | `#fafaf9` | Main background (warm off-white) |
| `--bg-secondary` | `#f5f5f4` | Sidebar, cards (stone-50) |
| `--text-primary` | `#1c1917` | Main text (stone-900) |
| `--text-secondary` | `#57534e` | Captions, hints (stone-600) |
| `--text-muted` | `#a8a29e` | Disabled, placeholders (stone-400) |
| `--accent` | `#2563eb` | Primary actions, links (blue-600) |
| `--accent-hover` | `#1d4ed8` | Hover state (blue-700) |
| `--border` | `#e7e5e4` | Borders, dividers (stone-200) |
| `--success` | `#22c55e` | Success states (green-500) |
| `--error` | `#ef4444` | Error states (red-500) |

**Rationale**:
- Warm stone tones (not cool grays) create approachable, professional feel
- Single accent color (blue) provides clear affordance without visual noise
- Aligns with Gael Penessot's recommendation for "warm, earthy tones for premium appearance"

**Alternatives Considered**:
- Cool gray palette → Rejected as too clinical
- Multiple accent colors → Rejected as visually noisy

---

### 5. Progress Indicator Simplification

**Question**: How to simplify the current 120+ line CSS progress indicator?

**Decision**: Replace with text-based indicator using Streamlit native markdown

**Rationale**:
- Current implementation uses complex HTML injection with animations
- Text-based approach using `st.sidebar.markdown()` with simple status symbols (✓, →, ○) achieves same clarity with ~20 lines
- Removes animation dependencies that may conflict with minimalist aesthetic

**Alternatives Considered**:
- Keep animations but simplify → Still adds visual noise
- Use Streamlit progress bar → Doesn't convey discrete levels well

---

### 6. Metric Card Component Pattern

**Question**: What pattern should be used for reusable metric cards?

**Decision**: Create a Python function that returns styled HTML via `st.html()`

**Rationale**:
- Function signature: `render_metric_card(label: str, value: str, delta: str = None, description: str = None)`
- Returns pre-styled HTML card
- Follows Streamlit's component pattern

**Alternatives Considered**:
- Streamlit custom component → Overkill for styled HTML
- st.metric with CSS override → Limited customization of layout

---

## Key Findings Summary

1. **Native theming is sufficient** - No external packages needed
2. **CSS injection is standard practice** - Well-documented selectors for hiding chrome
3. **Google Fonts CDN recommended** - Fast, reliable, with graceful fallback
4. **Warm neutrals + single accent** - Professional without visual noise
5. **Simplify, don't animate** - Text-based progress fits minimalist philosophy
6. **Function-based components** - Clean, testable, reusable

## References

- [Gael Penessot - DataGyver Substack](https://datagyver.substack.com/)
- [Streamlit Theming Documentation](https://docs.streamlit.io/develop/concepts/configuration/theming)
- [Max Braglia - Streamlit Theming Method](https://maxbraglia.substack.com/p/the-streamlit-theming-method-that)
