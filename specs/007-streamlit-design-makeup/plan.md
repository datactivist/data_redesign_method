# Implementation Plan: Streamlit Minimalist Design Makeup

**Branch**: `007-streamlit-design-makeup` | **Date**: 2025-12-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/007-streamlit-design-makeup/spec.md`

## Summary

Transform the Intuitiveness Streamlit application from a default-looking prototype into a polished, professional interface by implementing Gael Penessot's DataGyver design philosophy. This involves creating a centralized theming system via `config.toml`, injecting custom CSS via `st.html()` to hide Streamlit chrome, applying IBM Plex Sans typography, establishing a warm neutral color palette, and refactoring UI components for visual consistency.

## Technical Context

**Language/Version**: Python 3.11 (existing `myenv311` virtual environment)
**Primary Dependencies**: Streamlit >=1.28.0 (already installed), Google Fonts CDN (external)
**Storage**: N/A (UI-only changes, no data layer modifications)
**Testing**: Playwright MCP for E2E visual testing (existing infrastructure)
**Target Platform**: Web browser (desktop-first, responsive to 768px)
**Project Type**: Single project with Streamlit frontend
**Performance Goals**: Page load under 3 seconds including font loading
**Constraints**: Must not break existing workflow functionality; CSS-only visual changes
**Scale/Scope**: ~4500 lines in `streamlit_app.py`, ~12 UI module files to potentially modify

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|-----------|-------|--------|
| **I. Intuitiveness Through Abstraction Levels** | Design changes do not alter L0-L4 navigation structure | PASS |
| **II. Descent-Ascent Cycle** | Visual updates preserve workflow logic | PASS |
| **III. Complexity Quantification** | N/A - UI styling has no impact on complexity metrics | N/A |
| **IV. Human-Data Interaction Granularity** | N/A - No data transformation changes | N/A |
| **V. Design for Diverse Data Publics** | Improved visual design makes the tool MORE accessible to non-technical domain experts by hiding technical framework elements | PASS |
| **Target User Assumption** | Hiding Streamlit chrome and using professional design specifically serves domain curious minds who should not see "Made with Streamlit" | PASS |
| **Quality Gates - Domain Terminology** | Existing UI labels remain unchanged; visual styling only | PASS |

**Constitution Verdict**: All applicable gates pass. The design makeup feature specifically aligns with Principle V (Design for Diverse Data Publics) by making the interface feel like a professional domain tool rather than a technical framework.

## Project Structure

### Documentation (this feature)

```text
specs/007-streamlit-design-makeup/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (design tokens)
├── quickstart.md        # Phase 1 output (implementation guide)
├── contracts/           # Phase 1 output (CSS contracts)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
intuitiveness/
├── styles/              # NEW: Centralized CSS module
│   ├── __init__.py      # Exports all style constants
│   ├── chrome.py        # Hide Streamlit default elements
│   ├── typography.py    # Font loading and text styles
│   ├── palette.py       # Color definitions
│   ├── components.py    # Button, input, card styles
│   └── progress.py      # Simplified progress indicator styles
├── ui/
│   ├── metric_card.py   # NEW: Reusable metric card component
│   └── [existing files] # Minimal modifications for style integration
├── streamlit_app.py     # Refactor: Remove inline CSS, import from styles/
└── [existing files]     # No changes

.streamlit/
├── config.toml          # NEW: Theme configuration
└── secrets.toml.example # Existing
```

**Structure Decision**: Single project structure maintained. New `styles/` module added under `intuitiveness/` to centralize all CSS injection code. This keeps the codebase organized and allows incremental migration of existing inline CSS.

## Complexity Tracking

> No violations - design adheres to constitution principles.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
