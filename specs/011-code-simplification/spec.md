# 011-code-simplification

## Overview

**Feature**: Code Simplification - Intuitiveness Package Refactoring
**Status**: Complete
**Completed**: 2026-01-09
**Dependencies**: All previous specs (001-010)

## Problem Statement

The intuitiveness package accumulated technical debt through organic growth across 10+ features (001-010):

1. **Monolithic files**: `streamlit_app.py` (4,938 lines), `quality_dashboard.py` (2,200+ lines)
2. **Scattered session state**: 30+ keys defined across multiple files
3. **Code duplication**: 490+ lines of duplicated patterns
4. **Hidden dependencies**: Singleton registries with global state
5. **Poor spec traceability**: No clear mapping between code and specs

## Solution

Conservative refactoring in three phases:

### Phase 0: Foundation & Utilities
- Created `intuitiveness/utils/` module
- Centralized session state keys in `SessionStateKeys` class
- Consolidated common utilities

### Phase 1: Architecture Decomposition
- **P1.2**: Split `NavigationSession` into focused modules (`navigation/` package)
- **P1.3**: Split `quality_dashboard.py` into UI components (`ui/quality/` package)

### Phase 2: Design Pattern Improvements
- **P2.1**: Replaced singleton pattern with dependency injection
- **P2.2**: Created unified `AscentController` for ascent operations

## User Stories

### US-1: Developer Adds New Feature
**As a** developer
**I want** to find code by spec reference
**So that** I can quickly locate and modify functionality

**Acceptance Criteria**:
- Each module has spec traceability comments
- File structure mirrors spec structure (001-010)
- New features go in new modules with spec reference

### US-2: Developer Writes Tests
**As a** developer
**I want** isolated test instances
**So that** tests don't leak state

**Acceptance Criteria**:
- `create_fresh()` method on registries
- `reset_instance()` for test cleanup
- No auto-registration on module import

### US-3: Developer Understands Code
**As a** developer
**I want** focused modules with single responsibility
**So that** I can understand code without reading thousands of lines

**Acceptance Criteria**:
- No module exceeds 500 lines
- Each module has clear docstring with spec reference
- Public API documented in `__init__.py`

## Functional Requirements

### FR-001: Module Organization
Modules must be organized to reflect spec structure:

| Spec | Implementation Module |
|------|----------------------|
| 001-004 | `navigation/transformations.py`, `app/ascent_controller.py` |
| 002-003 | `navigation/session.py`, `app/free_mode.py` |
| 005 | `navigation/serialization.py`, `utils/session_manager.py` |
| 009-010 | `ui/quality/`, `quality/` |

### FR-002: Dependency Injection Support
Registries must support three patterns:
1. `get_instance()` - singleton for backward compatibility
2. `create_fresh(with_defaults)` - isolated instance for testing
3. `reset_instance()` - cleanup for test isolation

### FR-003: Backward Compatibility
All existing public APIs must continue to work:
- Import paths unchanged or aliased
- Function signatures unchanged
- Session state keys unchanged

### FR-004: Spec Traceability
Each new module must include:
- Docstring with spec reference (e.g., "Implements US-4 from 010-quality-ds-workflow")
- Contract reference if applicable
- Phase reference from this plan

## Technical Design

### Package Structure

```
intuitiveness/
├── utils/
│   ├── __init__.py
│   ├── common.py           # Shared utilities
│   ├── session_manager.py  # SessionStateKeys class
│   └── serialization.py    # Export/import utilities
├── navigation/
│   ├── __init__.py         # Public API (NavigationSession)
│   ├── session.py          # Core session (~400 lines)
│   ├── state.py            # State management
│   ├── tree.py             # Tree operations
│   ├── history.py          # Linear history
│   ├── exceptions.py       # Navigation errors
│   └── serialization.py    # Session export (005)
├── ui/
│   └── quality/
│       ├── __init__.py     # Package exports
│       ├── utils.py        # Session keys, score colors
│       ├── state.py        # Report history
│       ├── upload.py       # US-1 Step 1
│       ├── assessment.py   # US-1 Step 2
│       ├── suggestions.py  # US-1 Step 3, FR-002
│       └── readiness.py    # US-4, FR-001
├── app/
│   └── ascent_controller.py # Unified ascent logic
└── ascent/
    ├── enrichment.py       # EnrichmentRegistry (DI)
    └── dimensions.py       # DimensionRegistry (DI)
```

### Registry Pattern

```python
class EnrichmentRegistry:
    _instance: Optional['EnrichmentRegistry'] = None
    _defaults_registered: bool = False

    def __init__(self, auto_register_defaults: bool = False):
        self._functions = {}
        if auto_register_defaults:
            self._register_builtin_defaults()

    @classmethod
    def get_instance(cls) -> 'EnrichmentRegistry':
        """Singleton access (backward compatible)."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtin_defaults()
            cls._defaults_registered = True
        return cls._instance

    @classmethod
    def create_fresh(cls, with_defaults: bool = True) -> 'EnrichmentRegistry':
        """Create isolated instance (for testing)."""
        return cls(auto_register_defaults=with_defaults)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for test cleanup)."""
        cls._instance = None
        cls._defaults_registered = False
```

## Verification

### Unit Tests
```bash
pytest tests/unit/test_navigation.py -v
pytest tests/unit/test_enrichment.py -v
```

### Import Verification
```python
# All modules must import without error
from intuitiveness.utils.session_manager import SessionStateKeys
from intuitiveness.navigation import NavigationSession
from intuitiveness.ui.quality import render_file_upload
from intuitiveness.ascent.enrichment import EnrichmentRegistry
from intuitiveness.app.ascent_controller import AscentController
```

### Smoke Test
```bash
streamlit run intuitiveness/streamlit_app.py
# Upload test0, navigate L4→L0→L3
# Upload test1, assess quality, apply suggestions
```

## Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| streamlit_app.py lines | 4,938 | ~2,000 | 60% reduction |
| quality_dashboard.py lines | 2,216 | ~400 | 82% reduction |
| Duplicated code | 490+ lines | <100 lines | 80% reduction |
| Session state key locations | 15+ files | 1 file | Centralized |
| Test isolation | None | create_fresh() | Full isolation |

## Related Specs

- 001-dataset-redesign-package: Descent operations
- 002-ascent-functionality: Navigation, ascent operations
- 003-level-dataviz-display: Level displays
- 004-ascent-precision: Domain categorization
- 005-session-persistence: Session export/import
- 009-quality-data-platform: Quality assessment
- 010-quality-ds-workflow: DS Co-Pilot workflow
