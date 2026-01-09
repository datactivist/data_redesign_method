# Implementation Plan: Ascent Phase Precision

**Branch**: `004-ascent-precision` | **Date**: 2025-12-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-ascent-precision/spec.md`

## Summary

This feature refines the three ascent operations (L0→L1, L1→L2, L2→L3) with precise behaviors:
- **L0→L1**: Unfold datum to its source vector (deterministic - uses stored parent_data)
- **L1→L2**: Add domain columns via categorization (reuses L3→L2 pattern)
- **L2→L3**: Build graph by defining extra-dimension entity and relationships

The existing codebase already has substantial infrastructure (Redesigner, EnrichmentRegistry, DimensionRegistry) - this feature enhances the UI/UX and ensures consistency with the L3→L2 domain categorization pattern.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: streamlit>=1.28.0, pandas, networkx, sentence-transformers (for semantic matching)
**Storage**: Session state (st.session_state), pickle for session persistence
**Testing**: Manual verification via Streamlit UI (consistent with existing features)
**Target Platform**: Web application (Streamlit)
**Project Type**: Single Python package with Streamlit frontend
**Performance Goals**: Ascent operations complete within 2 seconds (SC-001)
**Constraints**: No orphan nodes in resulting graphs (Design Principle #1)
**Scale/Scope**: Single-user Streamlit application

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Intuitiveness Through Abstraction Levels | ✅ PASS | Feature enhances L0↔L1↔L2↔L3 navigation |
| I. Step-by-Step Traversal | ✅ PASS | All ascent operations move exactly one level |
| I. No Return to L4 | ✅ PASS | L3 is the highest ascent target |
| II. Descent-Ascent Cycle | ✅ PASS | Feature completes the ascent portion of the cycle |
| III. Complexity Quantification | ✅ PASS | Each ascent adds controlled complexity |
| IV. Human-Data Interaction Granularity | ✅ PASS | L0→L1 unfold maintains traceability to ground truth |
| V. Design for Diverse Data Publics | ✅ PASS | Enables analysts (L2-3) and creators (L3) workflows |

**Post-Phase 1 Re-Check**: ✅ PASS - All constitution principles respected.

## Project Structure

### Documentation (this feature)

```text
specs/004-ascent-precision/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── ascent_api.py    # Ascent operation contracts
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
intuitiveness/
├── complexity.py              # L0-L4 Dataset classes (Level0Dataset stores parent_data) ✓ EXISTS
├── redesign.py                # Transitions between levels ✓ EXISTS
│   ├── _increase_0_to_1()     # L0→L1 unfold ✓ EXISTS
│   ├── _increase_1_to_2()     # L1→L2 dimensions ✓ EXISTS
│   └── _increase_2_to_3()     # L2→L3 relationships ✓ EXISTS
├── navigation.py              # NavigationSession API ✓ EXISTS
├── ascent/
│   ├── enrichment.py          # L0→L1 enrichment functions ✓ EXISTS
│   ├── dimensions.py          # L1→L2 & L2→L3 dimensions ✓ EXISTS
│   └── operations.py          # Ascent operation tracking ✓ EXISTS
├── streamlit_app.py           # UI - render_ascend_options() NEEDS ENHANCEMENT
└── ui/
    ├── entity_tabs.py         # Entity extraction ✓ EXISTS
    ├── level_displays.py      # Level visualizations ✓ EXISTS
    └── ascent_forms.py        # NEW: Ascent UI forms for each transition
```

**Structure Decision**: Extend existing `intuitiveness/` package. Create new `ui/ascent_forms.py` module for ascent-specific UI components that can be reused across Guided and Free Navigation modes.

## Complexity Tracking

> No constitution violations detected - this section is empty.
