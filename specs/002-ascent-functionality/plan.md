# Implementation Plan: Ascent Functionality (Reverse Navigation)

**Branch**: `002-ascent-functionality` | **Date**: 2025-12-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-ascent-functionality/spec.md`

## Summary

Implement reverse navigation (ascent) from L0→L1→L2→L3 with:
- NavigationTree displayed as a Directed Acyclic Graph (DAG)
- Cumulative output export at any exit point
- Infinite exploration within L0-L3 boundaries
- Time-travel navigation with branch preservation

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: streamlit>=1.28.0, pandas, networkx, neo4j>=5.0.0, sentence-transformers, streamlit-agraph>=0.0.45
**Storage**: Session state (Streamlit st.session_state), pickle for session persistence, JSON for export
**Testing**: pytest (unit), behave (BDD)
**Target Platform**: Web application (Streamlit)
**Project Type**: Single project with subpackages
**Performance Goals**: <30 seconds for default enrichment operations (SC-001)
**Constraints**: Must handle datasets up to 10,000 items (SC-005)
**Scale/Scope**: Single-user exploratory data analysis tool

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Intuitiveness Through Abstraction Levels | ✅ PASS | Feature implements L0→L1→L2→L3 ascent, respecting all 5 levels |
| I. Navigation Rules | ✅ PASS | Spec defines step-by-step traversal, L4 entry-only, infinite exploration |
| II. Descent-Ascent Cycle | ✅ PASS | Feature completes the Ascent portion of the cycle |
| III. Complexity Quantification | ✅ PASS | AscentOperation tracks complexity changes |
| IV. Human-Data Interaction Granularity | ✅ PASS | NavigationTree provides full traceability from L0 ground truth |
| V. Design for Diverse Data Publics | ✅ PASS | Multiple exit points serve different user needs |

**Quality Gates**:
- ✅ Data integrity preserved during ascent (FR-005, SC-003)
- ✅ Ascent dimensions justified by user needs (custom enrichment functions)
- ✅ Each level independently testable (acceptance scenarios in spec)

## Project Structure

### Documentation (this feature)

```text
specs/002-ascent-functionality/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── navigation_api.py
│   └── export_api.py
├── checklists/          # Quality validation
│   └── requirements.md
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
intuitiveness/
├── __init__.py           # Package exports
├── complexity.py         # Level0-4 Dataset classes
├── redesign.py           # Redesigner with ascent methods
├── navigation.py         # NavigationTree, NavigationSession, NavigationTreeNode
├── ascent/
│   ├── __init__.py
│   ├── enrichment.py     # EnrichmentFunction, EnrichmentRegistry
│   └── dimensions.py     # DimensionDefinition, DimensionRegistry, RelationshipDefinition
├── ui/                   # NEW for this feature
│   ├── __init__.py
│   ├── decision_tree.py  # DAG visualization component
│   ├── drag_drop.py      # Relationship builder
│   └── json_visualizer.py # Export viewer
├── export/               # NEW for this feature
│   ├── __init__.py
│   └── json_export.py    # NavigationExport, OutputSummary
└── streamlit_app.py      # Main app with sidebar integration
```

**Structure Decision**: Single project with subpackages. UI components in `intuitiveness/ui/`, export functionality in `intuitiveness/export/`. Existing ascent logic in `intuitiveness/ascent/` is extended.

## New Requirements from Spec Changes (2025-12-04)

### FR-019: Cumulative Output Export
On exit at any level, export ALL accumulated outputs:
- Exit at L3: Graph + NavigationTree
- Exit at L2: Graph + Domain-labeled Table + NavigationTree
- Exit at L1: Graph + Domain-labeled Table + Vector + NavigationTree
- Exit at L0: Graph + Domain-labeled Table + Vector + Datum + NavigationTree

### FR-020: L2→L3 Entity Selection
For L2→L3 ascent, allow user to select extra table entity from available columns in raw original data.

### FR-021: NavigationTree as DAG
NavigationTree MUST be displayed as a Directed Acyclic Graph (DAG) recording:
- (a) Each navigation step taken
- (b) Decision made at each step (entity, label, operation)
- (c) Generated output snapshot at every step

### FR-022: Infinite Exploration
Navigation can continue indefinitely within L0-L3 boundaries.

## Complexity Tracking

No constitution violations to justify.

## Phases

### Phase 0: Research (Complete)
- Enrichment function patterns from research paper
- DAG visualization libraries (streamlit-agraph)
- Session state persistence patterns

### Phase 1: Design
- Data model updates for NavigationTree DAG structure
- Contract updates for cumulative export
- Quickstart scenarios for new navigation rules

### Phase 2: Tasks (Generated by /speckit.tasks)
- Implementation tasks organized by user story
