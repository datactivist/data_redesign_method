# Implementation Plan: Level-Specific Data Visualization Display

**Branch**: `003-level-dataviz-display` | **Date**: 2025-12-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-level-dataviz-display/spec.md`
**Constitution**: v1.2.0 (Target User Assumption added)

## Summary

This feature defines data visualization displays at each navigation level (L0-L4) for both Guided and Free Navigation modes. Per constitution v1.2.0, visualizations must be designed for **domain curious minds with NO data structure familiarity**—using domain terminology rather than technical terms like "tables," "graphs," "vectors," or "schemas."

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Streamlit >=1.28.0, pandas, networkx
**Storage**: Session state (st.session_state)
**Testing**: Manual UI verification per quickstart.md
**Target Platform**: Web browser (Streamlit)
**Project Type**: Single (Python package with Streamlit UI)
**Performance Goals**: Entity/relationship tabs load <2 seconds for 5,000 nodes (SC-004)
**Constraints**: Visualizations must use domain language, not technical data terms
**Scale/Scope**: Up to 10,000 nodes in typical use cases

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Intuitiveness Through Abstraction Levels
- ✅ **Compliance**: Feature displays all 5 levels (L0-L4) with appropriate visualizations
- ✅ **Navigation Rules**: Step-by-step traversal supported with clear direction indicators

### Principle II: Descent-Ascent Cycle
- ✅ **Compliance**: Both descent (L4→L0) and ascent (L0→L3) visualizations defined
- ✅ **Ascent Rule**: Ascent shows source level (lower) visualization for context

### Principle III: Complexity Quantification
- ✅ **Compliance**: Item counts displayed at each level for user awareness

### Principle IV: Human-Data Interaction Granularity
- ✅ **Compliance**: Users can trace from any level back to atomic data (L0)

### Principle V: Design for Diverse Data Publics
- ✅ **Compliance**: Information seekers (L0-L1), Analysts (L2-L3), Creators (L3-L4) all served

### Principle V: Target User Assumption ⚠️ CRITICAL
- ⚠️ **REVIEW REQUIRED**: Constitution v1.2.0 explicitly states users have NO familiarity with data structures
- **Action Items**:
  1. Replace "graph" terminology with domain-appropriate language
  2. Replace "table" with "organized information" or "categorized items"
  3. Replace "vector" with "list of values" or "collection"
  4. Replace "datum" with "single answer" or "result"
  5. Ensure UI labels use domain questions, not technical queries

### Gate Status: ✅ PASS (with v1.2.0 adaptation required)

## Project Structure

### Documentation (this feature)

```text
specs/003-level-dataviz-display/
├── plan.md              # This file (updated for constitution v1.2.0)
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
intuitiveness/
├── ui/
│   ├── __init__.py          # Exports all UI components
│   ├── level_displays.py    # Level-specific display functions (L0-L4)
│   ├── entity_tabs.py       # Entity/relationship tabbed views
│   ├── ascent_forms.py      # Ascent UI forms (004-ascent-precision)
│   ├── decision_tree.py     # Navigation tree component
│   ├── drag_drop.py         # Relationship builder
│   └── json_visualizer.py   # JSON export visualization
├── streamlit_app.py         # Main application
└── __init__.py              # Package exports
```

**Structure Decision**: Single Python package with UI components in dedicated `ui/` subdirectory

## Complexity Tracking

> **No violations detected** - feature aligns with all constitution principles

## Constitution v1.2.0 Adaptation Required

The following terminology changes are required to comply with the new Target User Assumption:

| Technical Term | Domain-Friendly Alternative |
|---------------|---------------------------|
| Graph | "Connected information" or "Your data relationships" |
| Table | "Organized information" or "Categorized items" |
| Vector | "List of values" or "Collection of items" |
| Datum | "Single answer" or "Result" or "Computed value" |
| Entity | "Item" or "Thing" (context-dependent) |
| Relationship | "Connection" or "Link" |
| Node | "Item" |
| Edge | "Connection" |
| Schema | Never expose to users |

### UI Label Examples (Before/After)

| Before (Technical) | After (Domain-Friendly) |
|-------------------|------------------------|
| "Entity Tabs" | "Browse by Category" |
| "Relationship Tabs" | "View Connections" |
| "Graph Visualization" | "How Your Data Connects" |
| "Domain Table" | "Items by Category" |
| "Vector Display" | "Your Selected Values" |
| "Atomic Metric" | "Your Computed Result" |
| "L3→L2 Transition" | "Explore categories" |

## Research Reference

See [research.md](./research.md) for:
- Decision: Leverage existing L3→L2 implementation
- Decision: Ascent shows source level, descent shows target level
- Decision: Extract shared display components for mode consistency
- Decision: Paginate tables >50 rows for performance
