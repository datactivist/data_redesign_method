# Tasks: Dataset Redesign Package

**Input**: Design documents from `/specs/001-dataset-redesign-package/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Tests are NOT explicitly requested in the feature specification. Test tasks are omitted.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

---

## Implementation Status Summary

**Last Updated**: 2025-12-02

| Phase | Total Tasks | Completed | Remaining | Status |
|-------|-------------|-----------|-----------|--------|
| Phase 1: Setup | 8 | 2 | 6 | Partial (flat structure used) |
| Phase 2: Foundational | 7 | 3 | 4 | Partial (missing lineage, validation) |
| Phase 3: US1 Descent | 9 | 5 | 4 | **Core done** (missing descend() API, lineage) |
| Phase 4: US2 Ascent | 7 | 4 | 3 | **Core done** (missing ascend() API, lineage) |
| Phase 5: US3 Complexity | 5 | 0 | 5 | Not started |
| Phase 6: US4 Full Cycle | 5 | 0 | 5 | Not started |
| Phase 7: US5 Navigation | 12 | 12 | 0 | **COMPLETE** |
| Phase 8: Polish | 5 | 0 | 5 | Not started |
| **TOTAL** | **58** | **26** | **32** | **45% complete** |

**Key Existing Implementation** (`intuitiveness/`):
- `complexity.py`: ComplexityLevel enum, Dataset ABC, Level0-4Dataset classes
- `redesign.py`: Redesigner class with reduce_complexity() and increase_complexity()
- `interactive.py`: InteractiveRedesigner, TransitionQuestions, Neo4jDataModel
- `navigation.py`: NavigationSession, NavigationState, NavigationStep, NavigationHistory (**NEW**)
- `utils.py`: Helper functions (load_csv_as_df, graph_to_dataframe)

**App Integration**:
- `app.py`: Added navigation explorer mode with mode toggle in sidebar

**Critical Gaps**:
1. No public `descend()` / `ascend()` functions (uses Redesigner methods)
2. No lineage tracking (DataLineage, SourceReference, trace_lineage)
3. No complexity measurement (measure_complexity)
4. No tests directory

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Project initialization and basic structure

**NOTE**: Existing `intuitiveness/` package uses flat structure. Tasks below reflect gaps from planned architecture.

- [x] T001 ~~Create package directory structure per plan.md~~ **EXISTING**: Flat structure in `intuitiveness/` with complexity.py, redesign.py, interactive.py, utils.py
- [ ] T002 Initialize Python package with pyproject.toml including dependencies: pandas, networkx, pytest, pytest-cov
- [x] T003 [P] ~~Create `intuitiveness/__init__.py`~~ **EXISTING**: Has exports but missing descend, ascend, measure_complexity, NavigationSession, trace_lineage
- [ ] T004 [P] Create `intuitiveness/models/__init__.py` with model exports (SKIPPED - using flat structure)
- [ ] T005 [P] Create `intuitiveness/operations/__init__.py` with operation exports (SKIPPED - using flat structure)
- [ ] T006 [P] Create `intuitiveness/navigation/__init__.py` with navigation exports (SKIPPED - using flat structure)
- [ ] T007 [P] Create `intuitiveness/utils/__init__.py` with utility exports (SKIPPED - using flat structure)
- [ ] T008 [P] Create `tests/conftest.py` with shared pytest fixtures for sample DataFrames, graphs, and L4 sources

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T009 ~~Implement ComplexityLevel enum~~ **EXISTING**: `intuitiveness/complexity.py` has ComplexityLevel with LEVEL_0 to LEVEL_4 (different naming than spec)
- [ ] T010 Implement base exception classes in `intuitiveness/utils/validation.py`: IntuitivenessError, ValidationError, OperationError, NavigationError, SessionNotFoundError
- [ ] T011 Implement input validation functions in `intuitiveness/utils/validation.py`: validate_non_empty(), validate_level_match(), validate_scalar(), validate_sequence(), validate_dataframe(), validate_graph(), validate_sources_dict()
- [x] T012 ~~Implement Dataset dataclass~~ **EXISTING**: `intuitiveness/complexity.py` has Dataset ABC + Level0Dataset, Level1Dataset, Level2Dataset, Level3Dataset, Level4Dataset classes
- [x] T013 ~~Implement Dataset factory methods~~ **EXISTING**: Each Level*Dataset class has its own constructor (different pattern than factory methods)
- [ ] T014 Implement SourceReference dataclass in `intuitiveness/models/lineage.py` with fields: dataset_id, row_indices, column_name, node_id
- [ ] T015 Implement DataLineage dataclass in `intuitiveness/models/lineage.py` with fields: source_ref, operation, parameters, parent, timestamp

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Reduce Dataset Complexity (Priority: P1)

**Goal**: Implement descent operations to reduce complexity from L4→L3→L2→L1→L0

**Independent Test**: Load complex dataset, progressively reduce to single atomic value

### Implementation for User Story 1

**NOTE**: Descent operations exist via `Redesigner.reduce_complexity()` in `intuitiveness/redesign.py`. Missing: public descend() function, lineage tracking, complexity order calculation.

- [ ] T016 [US1] Implement complexity order calculation in `intuitiveness/models/complexity.py`: calculate_complexity_order(data, level) returning C(0), C(1), C(rows*cols), C(edges), or infinity
- [x] T017 [US1] ~~Implement abstract DescentOperation base class~~ **EXISTING**: `Redesigner.reduce_complexity()` dispatches to level-specific methods
- [x] T018 [P] [US1] ~~Implement LinkOperation (L4→L3)~~ **EXISTING**: `Redesigner._reduce_4_to_3()` accepts builder_func parameter
- [x] T019 [P] [US1] ~~Implement QueryOperation (L3→L2)~~ **EXISTING**: `Redesigner._reduce_3_to_2()` accepts query_func parameter
- [x] T020 [P] [US1] ~~Implement SelectOperation (L2→L1)~~ **EXISTING**: `Redesigner._reduce_2_to_1()` accepts column and filter_query parameters
- [x] T021 [P] [US1] ~~Implement AggregateOperation (L1→L0)~~ **EXISTING**: `Redesigner._reduce_1_to_0()` accepts aggregation method (sum, mean, count, min, max) and callable
- [ ] T022 [US1] Implement descend() public function in `intuitiveness/operations/descent.py` that dispatches to correct operation based on dataset level
- [ ] T023 [US1] Implement lineage attachment in descent operations: create DataLineage with source_ref, operation name, and parameters
- [x] T024 [US1] ~~Implement complexity reduction validation~~ **EXISTING**: `Redesigner.reduce_complexity()` validates target_level < current_level

**Checkpoint**: User Story 1 complete - descent operations functional

---

## Phase 4: User Story 2 - Increase Dataset Complexity (Priority: P2)

**Goal**: Implement ascent operations to increase complexity from L0→L1→L2→L3

**Independent Test**: Start from single value, progressively add dimensions to reach desired level

### Implementation for User Story 2

**NOTE**: Ascent operations partially exist via `Redesigner.increase_complexity()` in `intuitiveness/redesign.py`. Missing: L0→L1 specific operation, public ascend() function, lineage tracking.

- [x] T025 [US2] ~~Implement abstract AscentOperation base class~~ **EXISTING**: `Redesigner.increase_complexity()` dispatches based on target_level
- [ ] T026 [P] [US2] Implement EnrichOperation (L0→L1) in `intuitiveness/operations/ascent.py` accepting source and selection_criteria parameters (PARTIAL: existing goes L0/L1→L2 directly)
- [x] T027 [P] [US2] ~~Implement DimensionOperation (L1→L2)~~ **EXISTING**: `Redesigner.increase_complexity()` to LEVEL_2 via enrichment_func
- [x] T028 [P] [US2] ~~Implement HierarchyOperation (L2→L3)~~ **EXISTING**: `Redesigner.increase_complexity()` to LEVEL_3 via enrichment_func
- [ ] T029 [US2] Implement ascend() public function in `intuitiveness/operations/ascent.py` that dispatches to correct operation based on dataset level
- [ ] T030 [US2] Implement lineage attachment in ascent operations: create DataLineage linking to source dataset
- [x] T031 [US2] ~~Implement ascent validation~~ **EXISTING**: `Redesigner.increase_complexity()` validates target_level > current_level, raises NotImplementedError for L4

**Checkpoint**: User Story 2 complete - ascent operations functional

---

## Phase 5: User Story 3 - Measure Dataset Complexity (Priority: P3)

**Goal**: Implement complexity measurement and reporting for any dataset

**Independent Test**: Pass various dataset types and verify correct level identification

### Implementation for User Story 3

- [ ] T032 [US3] Implement level detection logic in `intuitiveness/models/complexity.py`: detect_level(data) returning ComplexityLevel based on data structure
- [ ] T033 [US3] Implement complexity formula mapping in `intuitiveness/models/complexity.py`: get_complexity_formula(level) returning "C(0)", "C(1)", "C(2^n)", "C(2^ng(2^n-1))", "C(∞)"
- [ ] T034 [US3] Implement dimension extraction in `intuitiveness/models/complexity.py`: get_dimensions(data, level) returning rows, columns, nodes, edges, sources as applicable
- [ ] T035 [US3] Implement reduction percentage calculation in `intuitiveness/models/complexity.py`: calculate_reduction(old_complexity, new_complexity)
- [ ] T036 [US3] Implement measure_complexity() public function in `intuitiveness/models/complexity.py` returning dict with level, level_name, complexity_order, complexity_formula, dimensions, reduction_from_l4

**Checkpoint**: User Story 3 complete - complexity measurement functional

---

## Phase 6: User Story 4 - Execute Full Descent-Ascent Cycle (Priority: P4)

**Goal**: Enable complete descent-ascent workflow with full data lineage tracing

**Independent Test**: Start with raw L4 data, descend to L0, ascend to target level, trace any value to source

### Implementation for User Story 4

- [ ] T037 [US4] Implement trace_lineage() public function in `intuitiveness/models/lineage.py` accepting dataset, optional row_index, optional column_name
- [ ] T038 [US4] Implement lineage chain traversal in `intuitiveness/models/lineage.py`: walk parent references to build transformation history list
- [ ] T039 [US4] Implement source location extraction in `intuitiveness/models/lineage.py`: resolve SourceReference to original data coordinates
- [ ] T040 [US4] Implement operation chaining support in `intuitiveness/operations/descent.py` and `intuitiveness/operations/ascent.py`: ensure Dataset output from one operation is valid input for next
- [ ] T041 [US4] Add performance optimization for lineage tracing in `intuitiveness/models/lineage.py`: index-based O(1) lookup for datasets up to 100K rows

**Checkpoint**: User Story 4 complete - full cycle with lineage tracing functional

---

## Phase 7: User Story 5 - Navigate Dataset Hierarchy (Priority: P5)

**Goal**: Implement step-by-step navigation through abstraction levels with L4 entry-only constraint

**Independent Test**: Enter at L4, navigate through levels (except returning to L4), exit and resume

### Implementation for User Story 5

**NOTE**: Navigation implemented in `intuitiveness/navigation.py` (flat structure) and integrated into `app.py` with mode toggle.

- [x] T042 [US5] ~~Implement NavigationState enum~~ **DONE**: `intuitiveness/navigation.py` - NavigationState with ENTRY, EXPLORING, EXITED
- [x] T043 [US5] ~~Implement NavigationStep dataclass~~ **DONE**: `intuitiveness/navigation.py` - NavigationStep with level, node_id, action, timestamp
- [x] T044 [US5] ~~Implement NavigationHistory class~~ **DONE**: `intuitiveness/navigation.py` - NavigationHistory with append(), get_path(), get_path_dicts()
- [x] T045 [US5] ~~Implement NavigationSession class~~ **DONE**: `intuitiveness/navigation.py` - NavigationSession with L4 entry validation
- [x] T046 [US5] ~~Implement NavigationSession.descend()~~ **DONE**: Uses Redesigner.reduce_complexity(), records history
- [x] T047 [US5] ~~Implement NavigationSession.ascend()~~ **DONE**: Blocks L3→L4 with NavigationError
- [x] T048 [US5] ~~Implement NavigationSession.move_horizontal()~~ **DONE**: Records horizontal movement in history
- [x] T049 [US5] ~~Implement NavigationSession.get_available_moves()~~ **DONE**: Returns dict with descend/ascend/horizontal options
- [x] T050 [US5] ~~Implement NavigationSession.get_history()~~ **DONE**: Returns list of step dictionaries
- [x] T051 [US5] ~~Implement NavigationSession.exit()~~ **DONE**: Sets state to EXITED, records exit step
- [x] T052 [US5] ~~Implement NavigationSession.save() and load()~~ **DONE**: Pickle serialization with path support
- [x] T053 [US5] ~~Implement NavigationSession.resume()~~ **DONE**: Class-level session storage with resume functionality

**Checkpoint**: User Story 5 complete - navigation with L4 entry-only constraint functional

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T054 [P] Add type hints to all public functions and classes across package
- [ ] T055 [P] Add docstrings with examples to all public API (Dataset, descend, ascend, measure_complexity, NavigationSession, trace_lineage)
- [ ] T056 [P] Create README.md in `intuitiveness/` with installation and basic usage
- [ ] T057 Run quickstart.md examples and verify all code works as documented
- [ ] T058 Verify all validation error messages include actionable suggestions per FR-015

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - US1 (Phase 3): No dependency on other stories - can start after Phase 2
  - US2 (Phase 4): No dependency on US1 (independent ascent operations)
  - US3 (Phase 5): No dependency on US1/US2 (independent measurement)
  - US4 (Phase 6): Depends on US1 (descent) and US2 (ascent) for full cycle
  - US5 (Phase 7): Depends on US1 descent operations for navigation moves
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational - No dependencies on other stories
- **US2 (P2)**: Can start after Foundational - No dependencies on other stories
- **US3 (P3)**: Can start after Foundational - No dependencies on other stories
- **US4 (P4)**: Depends on US1 and US2 completion (combines both)
- **US5 (P5)**: Depends on US1 completion (uses descent for navigation moves)

### Within Each User Story

- Models before operations (T016 before T017-T024)
- Base classes before concrete implementations (T017 before T018-T021)
- Public function after all operations implemented (T022 after T018-T021)
- Lineage attachment after core operations (T023 after T022)

### Parallel Opportunities

All Setup tasks marked [P] can run in parallel:
- T003, T004, T005, T006, T007, T008

Within US1 (Phase 3), descent operations can run in parallel:
- T018, T019, T020, T021

Within US2 (Phase 4), ascent operations can run in parallel:
- T026, T027, T028

Within US5 (Phase 7), navigation methods can run in parallel after T045:
- T046, T047, T048, T049, T050

All Polish tasks marked [P] can run in parallel:
- T054, T055, T056

---

## Parallel Example: User Story 1

```bash
# After T017 (base class) completes, launch all concrete operations together:
Task: "Implement LinkOperation (L4→L3) in intuitiveness/operations/descent.py"
Task: "Implement QueryOperation (L3→L2) in intuitiveness/operations/descent.py"
Task: "Implement SelectOperation (L2→L1) in intuitiveness/operations/descent.py"
Task: "Implement AggregateOperation (L1→L0) in intuitiveness/operations/descent.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Descent)
4. **STOP and VALIDATE**: Test descent from L4→L0
5. Deploy/demo if ready - users can already reduce dataset complexity

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Deploy (MVP: descent operations)
3. Add User Story 2 → Deploy (ascent operations)
4. Add User Story 3 → Deploy (complexity measurement)
5. Add User Story 4 → Deploy (full cycle with lineage)
6. Add User Story 5 → Deploy (navigation)

### Parallel Team Strategy

With multiple developers after Foundational phase:
- Developer A: User Story 1 (Descent)
- Developer B: User Story 2 (Ascent)
- Developer C: User Story 3 (Complexity Measurement)
- Then: Developer A+B+C collaborate on US4 (depends on US1+US2)
- Finally: US5 (Navigation) after US1 complete

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
