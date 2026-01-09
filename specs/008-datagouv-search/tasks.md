# Tasks: Data.gouv.fr Search Integration

**Input**: Design documents from `/specs/008-datagouv-search/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are included as the project uses Playwright MCP for E2E testing (per CLAUDE.md).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Project root**: `intuitiveness/` for application code
- **Services**: `intuitiveness/services/` for API client wrappers
- **UI Components**: `intuitiveness/ui/` for Streamlit components
- **Styles**: `intuitiveness/styles/` for CSS styling
- **Tests**: `tests/e2e/` and `tests/unit/` for test files
- **Skill Library**: `skills/data-gouv/lib/` (existing, read-only)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and service layer structure

- [ ] T001 Create services directory and __init__.py in intuitiveness/services/__init__.py
- [ ] T002 [P] Create search styles module in intuitiveness/styles/search.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**Critical**: The DataGouvSearchService wrapper is the foundation for all user stories.

- [ ] T003 Implement DataGouvSearchService class in intuitiveness/services/datagouv_client.py
- [ ] T004 Add data classes (SearchResult, DatasetInfo, ResourceInfo, exceptions) in intuitiveness/services/datagouv_client.py
- [ ] T005 [P] Add search-related translations to intuitiveness/ui/i18n.py
- [ ] T006 Update intuitiveness/ui/__init__.py to export new components

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Search Open Data by Intent (Priority: P1)

**Goal**: Users can search data.gouv.fr by natural language intent and see matching datasets

**Independent Test**: Enter a search query and verify dataset results appear with relevant metadata

### Tests for User Story 1

- [ ] T007 [P] [US1] E2E test for search query submission in tests/e2e/test_datagouv_search.py
- [ ] T008 [P] [US1] Unit test for DataGouvSearchService.search() in tests/unit/test_datagouv_client.py

### Implementation for User Story 1

- [ ] T009 [US1] Create render_search_bar() function in intuitiveness/ui/datagouv_search.py
- [ ] T010 [US1] Create render_dataset_cards() function in intuitiveness/ui/datagouv_search.py
- [ ] T011 [US1] Add CSS styling for search bar and dataset cards in intuitiveness/styles/search.py
- [ ] T012 [US1] Initialize session state keys for search in intuitiveness/ui/datagouv_search.py

**Checkpoint**: Users can search and see results - Story 1 complete

---

## Phase 4: User Story 2 - Select and Load Dataset (Priority: P1)

**Goal**: Users can select a dataset resource and load it into the L4 workflow

**Independent Test**: Click a dataset, select a CSV resource, verify data loads into workflow

### Tests for User Story 2

- [ ] T013 [P] [US2] E2E test for dataset selection and loading in tests/e2e/test_datagouv_search.py
- [ ] T014 [P] [US2] Unit test for load_resource() in tests/unit/test_datagouv_client.py

### Implementation for User Story 2

- [ ] T015 [US2] Create render_resource_selector() function in intuitiveness/ui/datagouv_search.py
- [ ] T016 [US2] Create render_loading_state() function in intuitiveness/ui/datagouv_search.py
- [ ] T017 [US2] Implement load_resource() method in intuitiveness/services/datagouv_client.py
- [ ] T018 [US2] Add session state transitions for dataset selection in intuitiveness/ui/datagouv_search.py
- [ ] T019 [US2] Handle loaded DataFrame integration with existing workflow in intuitiveness/ui/datagouv_search.py

**Checkpoint**: Users can search, select, and load datasets - Stories 1 & 2 complete

---

## Phase 5: User Story 3 - Bilingual Interface (Priority: P2)

**Goal**: Search interface works seamlessly in both English and French

**Independent Test**: Toggle language and verify all search text changes appropriately

### Tests for User Story 3

- [ ] T020 [P] [US3] E2E test for language toggle on search interface in tests/e2e/test_datagouv_search.py

### Implementation for User Story 3

- [ ] T021 [US3] Verify all UI text uses t() function in intuitiveness/ui/datagouv_search.py
- [ ] T022 [US3] Add complete translation coverage check for search keys in intuitiveness/ui/i18n.py

**Checkpoint**: Search interface fully bilingual - Story 3 complete

---

## Phase 6: User Story 4 - Combined Entry Points (Priority: P3)

**Goal**: Both file upload and search entry points coexist seamlessly

**Independent Test**: Verify both paths (upload and search) lead to same workflow state

### Tests for User Story 4

- [ ] T023 [P] [US4] E2E test for combined entry points in tests/e2e/test_datagouv_search.py

### Implementation for User Story 4

- [ ] T024 [US4] Create render_search_interface() orchestrator in intuitiveness/ui/datagouv_search.py
- [ ] T025 [US4] Modify render_methodology_intro() to integrate search in intuitiveness/streamlit_app.py
- [ ] T026 [US4] Add "Or upload your own files" fallback section in intuitiveness/streamlit_app.py
- [ ] T027 [US4] Handle API unavailable graceful degradation with render_error_state() in intuitiveness/ui/datagouv_search.py

**Checkpoint**: Both entry points functional - All stories complete

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements and validation

- [ ] T028 [P] Add error handling for network failures in intuitiveness/services/datagouv_client.py
- [ ] T029 [P] Add file size warnings for large datasets (>50MB) in intuitiveness/ui/datagouv_search.py
- [ ] T030 [P] Implement is_available() API health check in intuitiveness/services/datagouv_client.py
- [ ] T031 Run quickstart.md validation (manual test of complete workflow)
- [ ] T032 Update exports in intuitiveness/services/__init__.py and intuitiveness/ui/__init__.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1 & US2 are both P1 - implement sequentially (US2 depends on US1 UI)
  - US3 can start after US1 (needs search UI to verify)
  - US4 depends on US1 & US2 (integration task)
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P1)**: Depends on US1 (needs search results to select from)
- **User Story 3 (P2)**: Depends on US1 (needs search UI to verify translations)
- **User Story 4 (P3)**: Depends on US1 & US2 (integrates search into main app)

### Within Each User Story

- Tests written FIRST (TDD approach)
- Core functions before orchestration
- Service layer before UI layer
- Commit after each task or logical group

### Parallel Opportunities

- T001, T002 (Setup) can run in parallel
- T005, T006 (Foundational) can run in parallel with T003, T004
- T007, T008 (US1 tests) can run in parallel
- T013, T014 (US2 tests) can run in parallel
- T028, T029, T030, T032 (Polish) can all run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "E2E test for search query submission in tests/e2e/test_datagouv_search.py"
Task: "Unit test for DataGouvSearchService.search() in tests/unit/test_datagouv_client.py"

# Then implement in sequence:
Task: "Create render_search_bar() function in intuitiveness/ui/datagouv_search.py"
Task: "Create render_dataset_cards() function in intuitiveness/ui/datagouv_search.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 & 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Search capability)
4. Complete Phase 4: User Story 2 (Load capability)
5. **STOP and VALIDATE**: Test search-to-load workflow end-to-end
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational -> Foundation ready
2. Add User Story 1 -> Test independently -> Users can search
3. Add User Story 2 -> Test independently -> Users can load datasets (MVP!)
4. Add User Story 3 -> Test independently -> Full bilingual support
5. Add User Story 4 -> Test independently -> Seamless dual entry points
6. Polish phase -> Production-ready

---

## Summary

| Phase | Tasks | Parallel Tasks |
|-------|-------|----------------|
| Setup | 2 | 1 |
| Foundational | 4 | 2 |
| US1 - Search | 6 | 2 |
| US2 - Load | 7 | 2 |
| US3 - Bilingual | 3 | 1 |
| US4 - Combined | 5 | 1 |
| Polish | 5 | 4 |
| **Total** | **32** | **13** |

**MVP Scope**: User Stories 1 & 2 (13 tasks after foundational)
**Full Feature**: All 4 User Stories (32 tasks total)
