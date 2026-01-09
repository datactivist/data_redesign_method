# Tasks: Ascent Phase Precision

**Input**: Design documents from `/specs/004-ascent-precision/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ascent_api.py

**Tests**: Manual verification via Streamlit UI (consistent with existing features - no automated tests required)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Package**: `intuitiveness/` at repository root
- **UI Components**: `intuitiveness/ui/`
- **Ascent Logic**: `intuitiveness/ascent/`
- **Main App**: `intuitiveness/streamlit_app.py`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the new ascent forms module and establish shared utilities

- [X] T001 Create `intuitiveness/ui/ascent_forms.py` module with file header and imports
- [X] T002 [P] Add session state keys for ascent form states in `intuitiveness/ui/ascent_forms.py`
- [X] T003 [P] Create helper function `_get_ascent_form_state()` for managing form state in session

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Extract domain categorization into reusable component (required for US2, enables FR-009)

**⚠️ CRITICAL**: The domain categorization UI must be refactored before US2 can be implemented

- [X] T004 Extract domain categorization UI from L3→L2 descent into `_render_domain_categorization_inputs()` shared function in `intuitiveness/ui/ascent_forms.py`
- [X] T005 Add domain input field component (comma-separated input, parsing, validation)
- [X] T006 Add semantic/keyword toggle component with threshold slider (disabled when keyword mode)
- [X] T007 Refactor L3→L2 descent in `intuitiveness/streamlit_app.py` to use new shared `_render_domain_categorization_inputs()` function
- [X] T008 Verify L3→L2 descent still works identically after refactoring (manual test)

**Checkpoint**: Shared domain categorization component ready - User Story 2 can now use identical UI

---

## Phase 3: User Story 1 - Unfold Datum to Source Vector (Priority: P1) 🎯 MVP

**Goal**: Enable users at L0 to unfold the aggregated datum back to its source vector (deterministic operation)

**Independent Test**: Navigate L3→L2→L1→L0 (aggregate with median), then ascend L0→L1 and verify original vector is restored

### Implementation for User Story 1

- [X] T009 [US1] Implement `render_l0_to_l1_unfold_form()` in `intuitiveness/ui/ascent_forms.py`
  - Display aggregation method used (FR-002)
  - Show source vector preview (first 5-10 values)
  - Confirmation button
  - Block with message if no parent_data exists (FR-003)
- [X] T010 [US1] Add unfold form integration to `render_ascend_options()` in `intuitiveness/streamlit_app.py` for Free Navigation mode
- [X] T011 [US1] Add unfold form integration to guided mode ascent in `intuitiveness/streamlit_app.py`
- [X] T012 [US1] Verify unfold preserves original column name (FR-004) - add test scenario from quickstart.md
- [X] T013 [US1] Verify orphan datum handling (FR-003) - test with manually created L0 dataset

**Checkpoint**: User Story 1 complete - L0→L1 unfold works in both Guided and Free Navigation modes

---

## Phase 4: User Story 2 - Enrich Vector with Domain Columns (Priority: P2)

**Goal**: Enable users at L1 to add domain categorization columns to create a 2D table (L2)

**Independent Test**: Create vector at L1, ascend to L2 with domains "Electronics, Clothing, Food", verify table has domain column with categorized values

### Implementation for User Story 2

- [X] T014 [US2] Implement `render_l1_to_l2_domain_form()` in `intuitiveness/ui/ascent_forms.py`
  - Reuse `_render_domain_categorization_inputs()` from Phase 2 (FR-009)
  - Domain input field (FR-005)
  - Semantic/keyword toggle (FR-006)
  - Threshold slider 0.1-0.9 (FR-007)
  - "Apply Domain Enrichment" button
- [X] T015 [US2] Add validation for at least 1 domain before submission
- [X] T016 [US2] Integrate L1→L2 form into `render_ascend_options()` in `intuitiveness/streamlit_app.py` for Free Navigation
- [X] T017 [US2] Integrate L1→L2 form into guided mode ascent in `intuitiveness/streamlit_app.py`
- [X] T018 [US2] Ensure "Unmatched" domain assigned to non-matching values (FR-008)
- [X] T019 [US2] Verify semantic matching produces same results as L3→L2 descent (SC-005) - test per quickstart.md Scenario 4
- [X] T020 [US2] Verify keyword matching works correctly (threshold slider disabled) - test per quickstart.md Scenario 5

**Checkpoint**: User Story 2 complete - L1→L2 domain enrichment works with same UI as L3→L2 descent

---

## Phase 5: User Story 3 - Build Graph from Table with Extra Entity (Priority: P2)

**Goal**: Enable users at L2 to build a graph by extracting an entity column and defining relationships

**Independent Test**: Navigate to L2 with table containing "department" column, ascend to L3 defining Department entity, verify graph has department nodes connected to row entities

### Implementation for User Story 3

- [X] T021 [US3] Implement `render_l2_to_l3_entity_form()` in `intuitiveness/ui/ascent_forms.py`
  - Entity column dropdown selector (FR-011)
  - Entity type name text input (FR-012)
  - Relationship type text input (FR-012)
  - "Build Graph" button
- [X] T022 [US3] Add column analysis to show unique value count for selected column
- [X] T023 [US3] Add warning when selected column has only 1 unique value (edge case from spec)
- [X] T024 [US3] Add validation: all fields required before submission
- [X] T025 [US3] Integrate L2→L3 form into `render_ascend_options()` in `intuitiveness/streamlit_app.py` for Free Navigation
- [X] T026 [US3] Integrate L2→L3 form into guided mode ascent in `intuitiveness/streamlit_app.py`
- [X] T027 [US3] Verify nodes created for each unique entity value (FR-013) - test per quickstart.md Scenario 6
- [X] T028 [US3] Verify edges connect table rows to entity nodes (FR-014)
- [X] T029 [US3] Add orphan node validation to ensure no disconnected nodes in result (FR-015, SC-006)

**Checkpoint**: User Story 3 complete - L2→L3 graph building creates connected graphs with no orphan nodes

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T030 Ensure ascent options are clearly labeled and discoverable within 5 seconds (SC-007)
- [X] T031 Add info tooltips explaining each ascent operation in the navigation panel
- [X] T032 Verify performance: L0→L1 < 2s, L1→L2 < 30s, L2→L3 < 60s (SC-001, SC-002, SC-003)
- [X] T033 Run complete round-trip test per quickstart.md Scenario 1 (L3→L0→L3)
- [X] T034 Test mid-ascent cancellation preserves navigation state (edge case from spec)
- [X] T035 Test null values in vector become "Unknown" during domain categorization (edge case from spec)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS User Story 2
- **User Story 1 (Phase 3)**: Depends on Setup only - can run parallel with Phase 2
- **User Story 2 (Phase 4)**: Depends on Foundational (Phase 2) completion
- **User Story 3 (Phase 5)**: Depends on Setup only - can run parallel with Phase 2
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Independent - only needs Setup phase
- **User Story 2 (P2)**: Depends on Foundational (shared domain categorization component)
- **User Story 3 (P2)**: Independent - only needs Setup phase

### Within Each User Story

- Form implementation first
- Then Free Navigation integration
- Then Guided Mode integration
- Then verification tests
- Story complete before moving to next priority

### Parallel Opportunities

```
Phase 1 (Setup)
    ↓
    ├── Phase 2 (Foundational) ──→ Phase 4 (US2)
    │                                   ↓
    │                              [US2 Complete]
    │
    ├── Phase 3 (US1) ─────────→ [US1 Complete]
    │
    └── Phase 5 (US3) ─────────→ [US3 Complete]
                                        ↓
                                 Phase 6 (Polish)
```

---

## Parallel Example: Initial Phase

```bash
# After Setup (Phase 1) completes, these can run in parallel:

# Developer A - Foundational:
Task T004: "Extract domain categorization UI into shared function"
Task T005: "Add domain input field component"

# Developer B - User Story 1:
Task T009: "Implement render_l0_to_l1_unfold_form()"
Task T010: "Add unfold form to Free Navigation"

# Developer C - User Story 3:
Task T021: "Implement render_l2_to_l3_entity_form()"
Task T022: "Add column analysis for unique values"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 3: User Story 1 (L0→L1 Unfold)
3. **STOP and VALIDATE**: Test L0→L1 unfold independently
4. Deploy/demo - users can now trace aggregations back to source data

### Incremental Delivery

1. Setup → Foundation → US1 ready (MVP!)
2. Add US2 → Test domain enrichment → Demo L1→L2
3. Add US3 → Test graph building → Demo L2→L3
4. Polish phase → Full round-trip validation
5. Each story adds value without breaking previous stories

### Recommended Order (Single Developer)

1. T001-T003 (Setup)
2. T009-T013 (US1 - delivers MVP immediately)
3. T004-T008 (Foundational - needed for US2)
4. T014-T020 (US2)
5. T021-T029 (US3)
6. T030-T035 (Polish)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Manual testing via Streamlit UI per existing feature patterns
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Key constraint: No orphan nodes allowed in L2→L3 results (Design Principle #1)
