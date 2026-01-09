# Implementation Plan: Dataset Redesign Package

**Branch**: `001-dataset-redesign-package` | **Date**: 2025-12-02 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-dataset-redesign-package/spec.md`

## Summary

A Python package (`intuitiveness`) that enables users to navigate, reduce, and reconstruct dataset complexity across five abstraction levels (L0-L4). The package implements the Data Redesign Method's Descent-Ascent cycle, allowing transformation of "data swamps" into "intuitive datasets" tailored to specific user data literacy levels. Key features include:

- **Descent operations**: Reduce complexity from L4→L0 through linking, querying, selecting, and aggregating
- **Ascent operations**: Reconstruct complexity from L0→L4 by adding user-aligned dimensions
- **Complexity measurement**: Automatically detect and quantify dataset complexity levels
- **Step-by-step navigation**: Explore the data hierarchy freely with horizontal and vertical traversal
- **Data lineage**: Full traceability from any output value back to source data

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: pandas (DataFrames), networkx (graphs), typing (type hints), dataclasses (entities)
**Storage**: N/A (in-memory operations, optional pickle/JSON serialization for session persistence)
**Testing**: pytest with pytest-cov for coverage
**Target Platform**: Cross-platform (Linux, macOS, Windows) via PyPI distribution
**Project Type**: Single project (Python library package)
**Performance Goals**: Data lineage tracing < 1 second for datasets up to 100,000 rows (SC-006)
**Constraints**: Pure Python (no compiled extensions), minimal dependencies for easy installation
**Scale/Scope**: Datasets up to 100,000 rows, navigation sessions up to 100 steps (SC-008)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Intuitiveness Through Abstraction Levels ✅
- **Requirement**: Dataset designable at five levels (L0-L4)
- **Compliance**: FR-001 to FR-009 implement all level transitions; ComplexityLevel enum represents all five levels
- **Navigation Rules**: FR-018 to FR-024 implement all navigation rules (L4 entry-only, step-by-step movement, infinite exploration, exit anytime)

### Principle II: Descent-Ascent Cycle ✅
- **Requirement**: All redesign must follow Descent (L4→L0) then Ascent (L0→L4)
- **Compliance**: US1 (Descent) and US2 (Ascent) implement full cycle; US4 combines them into complete workflow

### Principle III: Complexity Quantification ✅
- **Requirement**: Complexity measured by extractable relationships with documented reduction bounds
- **Compliance**: FR-010 to FR-012 implement complexity detection, calculation, and reduction reporting; SC-004 verifies 75-100% bounds

### Principle IV: Human-Data Interaction Granularity ✅
- **Requirement**: Upper levels constructible from lower; L0 as ground truth anchor
- **Compliance**: FR-013 (data lineage) ensures traceability; ascent operations reconstruct from L0 upward

### Principle V: Design for Diverse Data Publics ✅
- **Requirement**: Accommodate information seekers (L0-1), analysts (L2-3), creators (L3-4)
- **Compliance**: Navigation (US5) enables all user types to access appropriate levels; complexity measurement (US3) helps match users to levels

**GATE STATUS: PASSED** - All five constitutional principles satisfied

## Project Structure

### Documentation (this feature)

```text
specs/001-dataset-redesign-package/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── api.md           # Python API contract
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
intuitiveness/
├── __init__.py          # Package exports
├── models/
│   ├── __init__.py
│   ├── dataset.py       # Dataset wrapper with complexity tracking
│   ├── complexity.py    # ComplexityLevel enum and complexity order calculations
│   ├── lineage.py       # DataLineage tracking
│   └── navigation.py    # NavigationSession state management
├── operations/
│   ├── __init__.py
│   ├── descent.py       # DescentOperation implementations (L4→L3→L2→L1→L0)
│   └── ascent.py        # AscentOperation implementations (L0→L1→L2→L3)
├── navigation/
│   ├── __init__.py
│   ├── navigator.py     # Step-by-step navigation engine
│   └── history.py       # Navigation history tracking
└── utils/
    ├── __init__.py
    └── validation.py    # Input validation and error handling

tests/
├── unit/
│   ├── test_dataset.py
│   ├── test_complexity.py
│   ├── test_descent.py
│   ├── test_ascent.py
│   └── test_navigation.py
├── integration/
│   ├── test_full_cycle.py
│   └── test_navigation_session.py
└── conftest.py          # Shared fixtures
```

**Structure Decision**: Single Python package (`intuitiveness`) with clear separation between models (data structures), operations (transformations), navigation (exploration), and utils (validation). Tests mirror source structure.
