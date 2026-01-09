# Implementation Plan: Session Persistence

**Branch**: `005-session-persistence` | **Date**: 2025-12-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-session-persistence/spec.md`

## Summary

Implement browser-based session persistence for the Data Redesign wizard to preserve uploaded files, wizard progress, form selections, and transformation results across browser refreshes and tab closures. Uses Streamlit's `st.cache_data` and browser localStorage via custom JavaScript component for true persistence beyond Streamlit's in-memory session state.

## Technical Context

**Language/Version**: Python 3.11+ (existing Streamlit app)
**Primary Dependencies**: Streamlit >=1.28.0, streamlit-javascript (for localStorage), pandas, networkx
**Storage**: Browser localStorage for persistence, Streamlit session_state for in-memory
**Testing**: pytest, manual browser testing
**Target Platform**: Web browser (Chrome, Firefox, Safari, Edge)
**Project Type**: Single Streamlit web application
**Performance Goals**: Session restore < 2 seconds, data serialization < 1 second for 50MB
**Constraints**: localStorage ~10MB limit per domain, graceful fallback for larger datasets
**Scale/Scope**: Single user per browser session, files up to 50MB total

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Intuitiveness Through Abstraction Levels | PASS | Feature is infrastructure - does not affect L0-L4 navigation |
| II. Descent-Ascent Cycle | PASS | Preserves user's position in descent/ascent without altering the cycle |
| III. Complexity Quantification | N/A | No new complexity introduced |
| IV. Human-Data Interaction Granularity | PASS | Data integrity preserved through serialization/deserialization |
| V. Design for Diverse Data Publics | PASS | Non-technical users benefit most - no re-learning required on return |

**Target User Assumption Check**: Session persistence SHIELDS users from technical complexity by automatically preserving their work. Users never need to understand storage, serialization, or browser mechanics.

## Project Structure

### Documentation (this feature)

```text
specs/005-session-persistence/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
intuitiveness/
├── streamlit_app.py          # Main app - add persistence hooks
├── persistence/              # NEW: Persistence module
│   ├── __init__.py
│   ├── session_store.py      # Core save/load logic
│   ├── serializers.py        # DataFrame, Graph serialization
│   └── storage_backend.py    # localStorage abstraction
├── ui/
│   └── recovery_banner.py    # NEW: Session recovery UI component
└── ...existing files...

tests/
├── unit/
│   └── test_persistence.py   # Unit tests for serialization
└── integration/
    └── test_session_restore.py  # E2E browser tests
```

**Structure Decision**: Single project structure extended with new `persistence/` module. Follows existing `intuitiveness/` package pattern.

## Complexity Tracking

No constitution violations requiring justification.

---

## Phase 0: Research

### Research Questions

1. **Streamlit persistence options**: What mechanisms exist for persisting data beyond session_state?
2. **localStorage limitations**: Size limits, browser support, data format requirements
3. **Serialization strategy**: How to serialize pandas DataFrames and networkx Graphs efficiently
4. **Streamlit-JavaScript integration**: How to bridge Streamlit Python with browser localStorage

### Findings

See [research.md](./research.md) for detailed findings.

---

## Phase 1: Design Artifacts

- [data-model.md](./data-model.md) - Session state schema
- [contracts/](./contracts/) - Persistence API contracts
- [quickstart.md](./quickstart.md) - Integration guide
