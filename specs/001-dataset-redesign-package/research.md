# Research: Dataset Redesign Package

**Feature**: 001-dataset-redesign-package
**Date**: 2025-12-02
**Status**: Complete

## Research Summary

No NEEDS CLARIFICATION items were identified in the Technical Context. The spec and assumptions provide clear guidance. This document captures technology decisions and best practices research.

---

## Decision 1: Graph Library for Level 3 Representation

**Decision**: Use `networkx` for graph structures at Level 3

**Rationale**:
- Pure Python, no compiled dependencies (aligns with constraints)
- Well-documented, widely adopted in data science community
- Supports directed/undirected graphs, multigraphs, and attribute storage on nodes/edges
- Integrates well with pandas for node/edge data extraction

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| igraph | Requires C compilation, violates pure Python constraint |
| graph-tool | GPL license, C++ dependency |
| Custom implementation | Unnecessary; networkx is mature and sufficient |

---

## Decision 2: DataFrame Library for Level 2 Representation

**Decision**: Use `pandas` for tabular data at Level 2

**Rationale**:
- Industry standard for Python tabular data
- Excellent column selection, filtering, and aggregation support
- Natural fit for L2 (table) operations
- Spec assumptions explicitly mention pandas

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| polars | Newer, less adoption; pandas sufficient for target scale |
| numpy structured arrays | Less ergonomic for named columns |

---

## Decision 3: Session Persistence for Navigation

**Decision**: Use `pickle` for session serialization with optional JSON export

**Rationale**:
- Pickle handles complex Python objects (DataFrames, graphs) natively
- JSON export provides human-readable alternative for debugging
- No external storage dependency (file-based, in-memory)

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| SQLite | Overkill for session state; adds dependency |
| Redis | Requires external service; violates minimal dependency constraint |

---

## Decision 4: Complexity Calculation Algorithm

**Decision**: Implement complexity formulas from constitution directly

**Rationale**:
- L0: C(0) = 0 (single value, no relationships)
- L1: C(1) = 1 (one dimension)
- L2: C(2^n) = 2^(rows × cols) theoretical, practical = row × col combinations
- L3: C(2^ng(2^n-1)) = relationship count from graph edges
- L4: C(∞) = undefined, report as "indefinable"

**Implementation Notes**:
- For L2, report practical complexity as `n_rows × n_cols` rather than theoretical exponential
- For L3, count edges in graph as proxy for relationship complexity
- Reduction percentage = 1 - (new_complexity / old_complexity)

---

## Decision 5: Lineage Tracking Approach

**Decision**: Attach lineage metadata to each Dataset wrapper

**Rationale**:
- Each Dataset carries its transformation history
- Lineage stored as linked list of (operation, source_ref) pairs
- Source references use row/column indices for O(1) lookup

**Performance Consideration**:
- SC-006 requires < 1 second for 100K rows
- Index-based references enable O(1) lookup per cell
- History chain traversal is O(depth), typically < 10 for normal workflows

---

## Decision 6: Navigation State Machine

**Decision**: Implement NavigationSession as finite state machine

**Rationale**:
- States: ENTRY (L4), EXPLORING (L1-L3), EXITED
- Transitions enforce L4 entry-only rule
- History is append-only list of (level, node_id) tuples

**State Transitions**:
```
ENTRY (L4) → EXPLORING (L3) [descend]
EXPLORING (L3) → EXPLORING (L2) [descend]
EXPLORING (L2) → EXPLORING (L3) [ascend]
EXPLORING (L2) → EXPLORING (L2) [horizontal]
EXPLORING (*) → EXITED [exit]
EXPLORING (*) → ENTRY (L4) [BLOCKED]
```

---

## Best Practices Applied

### Python Package Structure
- Use `src/` layout avoided; flat `intuitiveness/` at repo root for simpler imports
- `__init__.py` exports public API only
- Private modules prefixed with `_` where needed

### Testing Strategy
- Unit tests per module in `tests/unit/`
- Integration tests for full descent-ascent cycle
- Fixtures in `conftest.py` for reusable test data

### Type Hints
- All public functions fully typed
- Use `typing.Protocol` for operation interfaces
- Enable mypy strict mode

---

## Open Items

None. All technical decisions resolved.
