# Tasks: Ascent Functionality (Reverse Navigation)

**Input**: Design documents from `/specs/002-ascent-functionality/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Not explicitly requested in specification - tests are OPTIONAL.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md structure:
- **Source**: `intuitiveness/` at repository root
- **Ascent subpackage**: `intuitiveness/ascent/`
- **UI components**: `intuitiveness/ui/` (NEW)
- **Export functionality**: `intuitiveness/export/` (NEW)
- **Main app**: `intuitiveness/streamlit_app.py`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create new UI and export subpackage structures

- [X] T001 Create UI subpackage with init file at `intuitiveness/ui/__init__.py`
- [X] T002 [P] Create export subpackage with init file at `intuitiveness/export/__init__.py`
- [X] T003 [P] Verify streamlit-agraph>=0.0.45 in requirements.txt (add if missing)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### NavigationTree DAG Structure (FR-021)

- [ ] T004 Extend Level0Dataset with `parent_data` and `aggregation_method` in `intuitiveness/complexity.py`
- [X] T005 [P] Create NavigationTreeNode dataclass with `decision_description` and `output_snapshot` fields in `intuitiveness/navigation.py`
- [X] T006 Create NavigationTree class with `nodes`, `root_id`, `current_id` in `intuitiveness/navigation.py`
- [X] T007 Implement NavigationTree.branch() method creating child nodes in `intuitiveness/navigation.py`
- [X] T008 Implement NavigationTree.restore() method for time-travel (FR-017) in `intuitiveness/navigation.py`
- [X] T009 [P] Implement NavigationTree.get_current_path() returning node IDs in `intuitiveness/navigation.py`
- [X] T010 [P] Implement NavigationTree.get_visualization() for DAG rendering in `intuitiveness/navigation.py`
- [ ] T011 Implement NavigationTree.render_as_dag() returning agraph nodes/edges in `intuitiveness/navigation.py`
- [X] T012 Implement NavigationTree.export_to_json() for FR-015 in `intuitiveness/navigation.py`

### NavigationSession Core

- [X] T013 Refactor NavigationSession to use NavigationTree instead of linear history in `intuitiveness/navigation.py`
- [X] T014 Add `accumulated_outputs: Dict[int, Any]` to NavigationSession for FR-019 in `intuitiveness/navigation.py`
- [X] T015 Add `raw_data_columns: List[str]` to NavigationSession for FR-020 in `intuitiveness/navigation.py`
- [X] T016 Implement `on_transition()` method to track accumulated outputs in `intuitiveness/navigation.py`
- [X] T017 Implement `get_available_options()` returning NavigationOption list in `intuitiveness/navigation.py`

### Ascent Core Entities

- [ ] T018 [P] Create AscentOperation dataclass with integrity validation in `intuitiveness/ascent/__init__.py`
- [X] T019 [P] Create EnrichmentFunction dataclass with `__call__` method in `intuitiveness/ascent/enrichment.py`
- [X] T020 Create EnrichmentRegistry singleton with `register()`, `get()`, `list_for_transition()` in `intuitiveness/ascent/enrichment.py`
- [X] T021 [P] Create DimensionDefinition dataclass with `classify()` method in `intuitiveness/ascent/dimensions.py`
- [X] T022 Create DimensionRegistry singleton with `register()`, `get()`, `list_for_transition()` in `intuitiveness/ascent/dimensions.py`
- [X] T023 [P] Create RelationshipDefinition dataclass with `to_networkx_edge()` in `intuitiveness/ascent/dimensions.py`

### Export Entities (FR-019)

- [X] T024 [P] Create OutputSummary dataclass with `to_dict()` in `intuitiveness/export/json_export.py`
- [X] T025 [P] Create CumulativeOutputs dataclass for cumulative export in `intuitiveness/export/json_export.py`
- [X] T026 [P] Create NavigationNodeExport dataclass with `decision_description`, `output_snapshot` in `intuitiveness/export/json_export.py`
- [X] T027 Create NavigationExport dataclass with `cumulative_outputs` field in `intuitiveness/export/json_export.py`
- [X] T028 Implement NavigationExport.create() factory method in `intuitiveness/export/json_export.py`
- [X] T029 Implement convert_to_jsoncrack_format() helper in `intuitiveness/export/json_export.py`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Enrich Datum to Vector (L0 → L1) (Priority: P1) 🎯 MVP

**Goal**: Enable users to ascend from L0 (scalar) to L1 (vector) using enrichment functions

**Independent Test**: Compute any L0 metric via descent, then ascend to see a vector that aggregates back to original value

### Implementation for User Story 1

- [ ] T030 [US1] Register default `source_expansion` enrichment (re-expand original vector) in `intuitiveness/ascent/enrichment.py`
- [ ] T031 [US1] Register default `naming_signatures` enrichment (extract naming features) in `intuitiveness/ascent/enrichment.py`
- [ ] T032 [US1] Implement `_increase_0_to_1()` method in Redesigner in `intuitiveness/redesign.py`
- [ ] T033 [US1] Update `increase_complexity()` to handle L0→L1 via EnrichmentRegistry in `intuitiveness/redesign.py`
- [ ] T034 [US1] Implement `ascend()` method in NavigationSession for L0→L1 in `intuitiveness/navigation.py`
- [ ] T035 [US1] Add NavigationTree.branch() call with `decision_description` in ascend() for L0→L1 in `intuitiveness/navigation.py`
- [ ] T036 [US1] Add data integrity validation: row count must be preserved (FR-005, SC-003) in `intuitiveness/redesign.py`
- [ ] T037 [US1] Handle edge case: enrichment produces no data with informative message in `intuitiveness/navigation.py`
- [ ] T038 [US1] Update accumulated_outputs on L0→L1 transition (FR-019) in `intuitiveness/navigation.py`

**Checkpoint**: User Story 1 complete - L0→L1 ascent functional with tree tracking

---

## Phase 4: User Story 2 - Add Dimensions to Create Table (L1 → L2) (Priority: P2)

**Goal**: Transform L1 vector into L2 table by adding categorical dimensions

**Independent Test**: Take any L1 vector, add categorical dimension, produce L2 table

### Implementation for User Story 2

- [ ] T039 [P] [US2] Register default `business_object` dimension (revenue, volume, ETP, other) in `intuitiveness/ascent/dimensions.py`
- [ ] T040 [P] [US2] Register default `calculated_flag` dimension (derived vs raw) in `intuitiveness/ascent/dimensions.py`
- [ ] T041 [P] [US2] Register default `weight_flag` dimension (weighted vs unweighted) in `intuitiveness/ascent/dimensions.py`
- [ ] T042 [US2] Implement `_increase_1_to_2()` method in Redesigner in `intuitiveness/redesign.py`
- [ ] T043 [US2] Update `increase_complexity()` to handle L1→L2 via DimensionRegistry in `intuitiveness/redesign.py`
- [ ] T044 [US2] Extend `ascend()` in NavigationSession for L1→L2 with dimensions param in `intuitiveness/navigation.py`
- [ ] T045 [US2] Handle partial classification with "Unknown" fallback in `intuitiveness/ascent/dimensions.py`
- [ ] T046 [US2] Add NavigationTree.branch() call with `decision_description` in ascend() for L1→L2 in `intuitiveness/navigation.py`
- [ ] T047 [US2] Update accumulated_outputs on L1→L2 transition (FR-019) in `intuitiveness/navigation.py`

**Checkpoint**: User Stories 1 AND 2 complete - L0→L1→L2 ascent chain functional

---

## Phase 5: User Story 3 - Group into Hierarchical Relationships (L2 → L3) (Priority: P3)

**Goal**: Add hierarchical dimensions to L2 table via drag-and-drop to create L3 structure

**Independent Test**: Take L2 table, define relationships via drag-drop, produce L3 graph

### Implementation for User Story 3

- [ ] T048 [P] [US3] Register default `client_segment` dimension (B2B, B2C, Government) in `intuitiveness/ascent/dimensions.py`
- [ ] T049 [P] [US3] Register default `financial_view` dimension (Revenue, Cost, Margin) in `intuitiveness/ascent/dimensions.py`
- [ ] T050 [P] [US3] Register default `lifecycle_view` dimension (Acquisition, Retention, Churn) in `intuitiveness/ascent/dimensions.py`
- [ ] T051 [US3] Create drag-drop relationship builder component in `intuitiveness/ui/drag_drop.py`
- [ ] T052 [US3] Implement drag_drop.render() using streamlit-agraph in `intuitiveness/ui/drag_drop.py`
- [ ] T053 [US3] Implement `_increase_2_to_3()` method in Redesigner accepting relationships in `intuitiveness/redesign.py`
- [ ] T054 [US3] Update `increase_complexity()` to handle L2→L3 in `intuitiveness/redesign.py`
- [ ] T055 [US3] Implement `get_available_graph_entities()` returning L4 columns (FR-020) in `intuitiveness/navigation.py`
- [ ] T056 [US3] Extend `ascend()` in NavigationSession for L2→L3 with relationships param in `intuitiveness/navigation.py`
- [ ] T057 [US3] Block L3→L4 ascent with clear error message (FR-004) in `intuitiveness/navigation.py`
- [ ] T058 [US3] Implement duplicate detection query for items with identical dimensions (SC-005) in `intuitiveness/redesign.py`
- [ ] T059 [US3] Add NavigationTree.branch() call with `decision_description` in ascend() for L2→L3 in `intuitiveness/navigation.py`
- [ ] T060 [US3] Update accumulated_outputs on L2→L3 transition (FR-019) in `intuitiveness/navigation.py`

**Checkpoint**: User Stories 1, 2, AND 3 complete - Full L0→L1→L2→L3 ascent chain functional

---

## Phase 6: User Story 4 - Interactive Decision-Tree Navigation (Priority: P2)

**Goal**: Display persistent DAG sidebar with time-travel navigation and level-specific options

**Independent Test**: Navigate to any level (L0-L3), verify DAG displays correct options per FR-011-FR-014

### Implementation for User Story 4

- [X] T061 [US4] Create decision tree DAG component using streamlit-agraph in `intuitiveness/ui/decision_tree.py`
- [X] T062 [US4] Implement render() method displaying nodes with decision_description and output_snapshot in `intuitiveness/ui/decision_tree.py`
- [X] T063 [US4] Add node click handler for time-travel calling NavigationTree.restore() in `intuitiveness/ui/decision_tree.py`
- [X] T064 [US4] Style current node highlighted (green), others gray in `intuitiveness/ui/decision_tree.py`
- [X] T065 [US4] Implement branch preservation when time-travel creates new path (FR-018) in `intuitiveness/navigation.py`
- [X] T066 [US4] Implement `get_cumulative_export()` returning all accumulated outputs (FR-019) in `intuitiveness/navigation.py`
- [X] T067 [US4] Implement `exit()` method returning cumulative export in `intuitiveness/navigation.py`
- [X] T068 [US4] Create JSON visualizer component using st.json() in `intuitiveness/ui/json_visualizer.py`
- [X] T069 [US4] Implement download button for JSON export (FR-015) in `intuitiveness/ui/json_visualizer.py`
- [X] T070 [US4] At L3: implement options "Exit" or "Descend to L2" (FR-011) in `intuitiveness/navigation.py`
- [X] T071 [US4] At L2: implement options "Exit", "Descend to L1", "Ascend to L3" (FR-012) in `intuitiveness/navigation.py`
- [X] T072 [US4] At L1: implement options "Exit", "Descend to L0", "Ascend to L2" (FR-013) in `intuitiveness/navigation.py`
- [X] T073 [US4] At L0: implement options "Exit", "Ascend to L1" (FR-014) in `intuitiveness/navigation.py`
- [X] T074 [US4] Integrate decision-tree DAG sidebar into streamlit_app.py in `intuitiveness/streamlit_app.py`
- [X] T075 [US4] Integrate navigation options panel per level in `intuitiveness/streamlit_app.py`
- [X] T076 [US4] Add exit button with cumulative JSON export and visualization in `intuitiveness/streamlit_app.py`

**Checkpoint**: All user stories complete - Full navigation with DAG visualization functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T077 [P] Add progress indicators for large dataset enrichment (1000+ items) in `intuitiveness/redesign.py`
- [ ] T078 [P] Add logging for ascent operations in `intuitiveness/navigation.py`
- [X] T079 [P] Update `intuitiveness/ui/__init__.py` with __all__ exports
- [X] T080 [P] Update `intuitiveness/export/__init__.py` with __all__ exports
- [X] T081 Update `intuitiveness/__init__.py` to export ui and export subpackages
- [ ] T082 Performance optimization: ensure <30s for default enrichment (SC-001) in `intuitiveness/ascent/enrichment.py`
- [X] T083 Verify infinite exploration works (FR-022): ascend/descend repeatedly without limit
- [ ] T084 Run quickstart.md validation scenarios end-to-end
- [X] T085 Verify all FR-001 through FR-022 functional requirements implemented

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1 (P1) → US2 (P2) → US3 (P3) recommended sequential order
  - US4 (P2) can run parallel after US1
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2)**: Depends on US1 output (L1) to test L1→L2 ascent
- **User Story 3 (P3)**: Depends on US2 output (L2) to test L2→L3 ascent
- **User Story 4 (P2)**: Can start after Foundational, full integration needs US1-US3

### Within Each User Story

- Register defaults before implementing transitions
- Redesigner methods before NavigationSession methods
- Core logic before UI integration
- Validation and edge cases after happy path

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- Foundational tasks marked [P] can run in parallel
- Within each user story, tasks marked [P] can run in parallel
- US4 UI work can run parallel to US2/US3 after US1 completes

---

## Parallel Example: Foundational Phase

```bash
# Launch in parallel (different files):
Task T005: "Create NavigationTreeNode in intuitiveness/navigation.py"
Task T018: "Create AscentOperation in intuitiveness/ascent/__init__.py"
Task T019: "Create EnrichmentFunction in intuitiveness/ascent/enrichment.py"
Task T021: "Create DimensionDefinition in intuitiveness/ascent/dimensions.py"
Task T23: "Create RelationshipDefinition in intuitiveness/ascent/dimensions.py"
Task T024: "Create OutputSummary in intuitiveness/export/json_export.py"
Task T025: "Create CumulativeOutputs in intuitiveness/export/json_export.py"
Task T026: "Create NavigationNodeExport in intuitiveness/export/json_export.py"
```

## Parallel Example: User Story 2

```bash
# Launch dimension registrations in parallel:
Task T039: "Register business_object dimension"
Task T040: "Register calculated_flag dimension"
Task T041: "Register weight_flag dimension"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (3 tasks)
2. Complete Phase 2: Foundational (26 tasks)
3. Complete Phase 3: User Story 1 (9 tasks)
4. **STOP and VALIDATE**: Test L0→L1 ascent independently
5. Can demo basic "unfold datum to vector" capability

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 (L0→L1) → Test → Demo MVP
3. Add User Story 2 (L1→L2) → Test → Demo with dimensions
4. Add User Story 4 (DAG Navigation) → Test → Demo interactive UI
5. Add User Story 3 (L2→L3) → Test → Demo full ascent chain
6. Each story adds value without breaking previous stories

### Recommended Single-Developer Order

1. Setup (T001-T003)
2. Foundational (T004-T029) - critical path
3. User Story 1 (T030-T038) - MVP
4. User Story 2 (T039-T047)
5. User Story 4 basics (T061-T069, T074-T076) - parallel-safe
6. User Story 3 (T048-T060)
7. User Story 4 remaining (T070-T073)
8. Polish (T077-T085)

---

## Summary

| Phase | Task Count | Key Deliverables |
|-------|------------|------------------|
| Setup | 3 | New subpackages |
| Foundational | 26 | NavigationTree DAG, CumulativeOutputs, registries |
| User Story 1 | 9 | L0→L1 ascent with enrichment |
| User Story 2 | 9 | L1→L2 ascent with dimensions |
| User Story 3 | 13 | L2→L3 ascent with drag-drop |
| User Story 4 | 16 | DAG sidebar, time-travel, export |
| Polish | 9 | Performance, validation, cleanup |
| **Total** | **85** | Full ascent functionality |

### Per-Story Breakdown

- **US1**: 9 tasks (MVP)
- **US2**: 9 tasks
- **US3**: 13 tasks
- **US4**: 16 tasks

### Parallel Opportunities

- 23 tasks marked [P] can run in parallel within their phase

### New Requirements Coverage

- **FR-019** (Cumulative Export): T014, T016, T025, T038, T047, T060, T066, T067
- **FR-020** (Entity Selection): T015, T055
- **FR-021** (DAG Display): T005, T011, T026, T035, T046, T059, T062
- **FR-022** (Infinite Exploration): T083

---

## Notes

- [P] tasks = different files, no dependencies within phase
- [Story] label maps task to specific user story for traceability
- Tests not generated as not explicitly requested in specification
- streamlit-json-viewer doesn't exist on PyPI - using built-in st.json()
- Stop at any checkpoint to validate story independently
- FR-005, SC-003: Always validate row count preserved during ascent
