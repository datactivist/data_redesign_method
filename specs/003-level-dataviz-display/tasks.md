# Tasks: Level-Specific Data Visualization Display

**Input**: Design documents from `/specs/003-level-dataviz-display/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/display_api.py
**Constitution**: v1.2.0 (Target User Assumption - NO technical data terms in UI)

**Tests**: Tests are NOT explicitly requested in the feature specification - omitted.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project structure and package initialization

- [X] T001 Create `intuitiveness/ui/` package directory with `__init__.py`
- [X] T002 [P] Create `intuitiveness/ui/level_displays.py` with stub functions
- [X] T003 [P] Create `intuitiveness/ui/entity_tabs.py` with stub functions

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core display infrastructure that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement `LevelDisplayConfig` dataclass in `intuitiveness/ui/level_displays.py` (from contracts/display_api.py)
- [X] T005 [P] Implement `DisplayType` enum in `intuitiveness/ui/level_displays.py`
- [X] T006 [P] Implement `NavigationDirection` enum in `intuitiveness/ui/level_displays.py`
- [X] T007 Implement `LEVEL_DISPLAY_MAPPING` dict mapping levels 0-4 to DisplayType values
- [X] T008 Implement `EntityTabData` dataclass in `intuitiveness/ui/entity_tabs.py` with validation
- [X] T009 [P] Implement `RelationshipTabData` dataclass in `intuitiveness/ui/entity_tabs.py` with validation
- [X] T010 Implement `get_display_level()` function in `intuitiveness/ui/level_displays.py` per FR-012

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Descent Navigation Visualization (Priority: P1) MVP

**Goal**: Users navigating DOWN (L4→L3→L2→L1→L0) see appropriate visualizations at each step

**Independent Test**: Navigate from L4 to L0 in Guided Mode, verify each step shows correct visualization

### Implementation for User Story 1

#### L4 Display (Your Uploaded Files)
- [X] T011 [US1] Implement `render_l4_file_list()` in `intuitiveness/ui/level_displays.py` showing file names, item counts, categories (FR-001)
- [X] T012 [US1] Add file preview functionality to `render_l4_file_list()` showing first few items per file (FR-002)
- [X] T013 [US1] Update `render_upload_step()` in `intuitiveness/streamlit_app.py` to use `render_l4_file_list()`

#### L3→L2 Display (Browse by Category + Connection Tabs)
- [X] T014 [US1] Implement `extract_entity_tabs()` in `intuitiveness/ui/entity_tabs.py` to extract items by category (FR-004)
- [X] T015 [P] [US1] Implement `extract_relationship_tabs()` in `intuitiveness/ui/entity_tabs.py` to extract connections by type (FR-005)
- [X] T016 [US1] Implement `render_entity_relationship_tabs()` in `intuitiveness/ui/entity_tabs.py` using st.tabs() (FR-006, FR-007)
- [X] T017 [US1] Refactor `render_domains_step()` in `intuitiveness/streamlit_app.py` to use shared entity_tabs.py functions

#### L2→L1 Display (Items by Category)
- [X] T018 [US1] Implement `render_l2_domain_table()` in `intuitiveness/ui/level_displays.py` showing categorized items (FR-008, FR-009)
- [X] T019 [US1] Update `render_features_step()` in `intuitiveness/streamlit_app.py` to use `render_l2_domain_table()`

#### L1→L0 Display (Your Selected Values)
- [X] T020 [US1] Implement `render_l1_vector()` in `intuitiveness/ui/level_displays.py` showing value list with context (FR-010, FR-011)
- [X] T021 [US1] Update `render_features_step()` in `intuitiveness/streamlit_app.py` to use `render_l1_vector()`

#### L0 Display (Your Computed Result)
- [X] T022 [US1] Implement `render_l0_datum()` in `intuitiveness/ui/level_displays.py` showing single result prominently
- [X] T023 [US1] Update `render_aggregation_step()` in `intuitiveness/streamlit_app.py` to use `render_l0_datum()`

**Checkpoint**: At this point, descent navigation (L4→L0) should show correct visualizations at each level

---

## Phase 4: User Story 2 - Ascent Navigation Visualization (Priority: P2)

**Goal**: Users navigating UP (L0→L1→L2→L3) see LOWER level visualization (source context)

**Independent Test**: Ascend from L0 to L3 in Guided Mode, verify each step shows previous level's visualization

### Implementation for User Story 2

- [X] T024 [US2] Update `render_ascent_options()` in `intuitiveness/streamlit_app.py` to show source level visualization using `get_display_level()` (FR-012)
- [X] T025 [US2] Add L0→L1 ascent visualization showing result being expanded in `render_ascent_options()`
- [X] T026 [US2] Add L1→L2 ascent visualization showing values being enriched in `render_ascent_options()`
- [X] T027 [US2] Add L2→L3 ascent visualization showing categorized items becoming connected in `render_ascent_options()`
- [X] T028 [US2] Add navigation direction indicator (exploring deeper vs building up) to UI per FR-013

**Checkpoint**: At this point, both descent AND ascent navigations show correct visualizations

---

## Phase 5: User Story 3 - Free Navigation Mode Visualization (Priority: P2)

**Goal**: Free Navigation mode shows identical visualizations to Guided Mode

**Independent Test**: Switch to Free Navigation mode, perform descent and ascent operations, compare to Guided Mode

### Implementation for User Story 3

- [X] T029 [US3] Audit `render_free_navigation()` in `intuitiveness/streamlit_app.py` to identify display function calls
- [X] T030 [US3] Refactor `render_current_data()` to use shared display functions from `intuitiveness/ui/level_displays.py` (FR-014)
- [X] T031 [US3] Refactor `render_current_data()` to use shared tab functions from `intuitiveness/ui/entity_tabs.py` (FR-014)
- [X] T032 [US3] Verify mode consistency when switching between Guided and Free Navigation (FR-015)

**Checkpoint**: All user stories should now be independently functional with consistent visualization across modes

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Performance optimization and edge case handling

- [X] T033 [P] Add pagination for category tabs with >50 items in `intuitiveness/ui/entity_tabs.py` (SC-004) - Built into `render_entity_relationship_tabs()` with `max_rows=50` parameter
- [X] T034 [P] Add pagination for connection tabs with >50 items in `intuitiveness/ui/entity_tabs.py` (SC-004) - Built into `render_entity_relationship_tabs()` with `max_rows=50` parameter
- [X] T035 Handle empty connected info state (no connections) - show only category tabs in `render_entity_relationship_tabs()` - Handled at line 239-241
- [X] T036 [P] Handle empty category state - show "No items matched this category" message - Implemented in `render_l2_domain_table()` at line 161-163
- [X] T037 Add loading indicators for large data processing to meet SC-004 (<2 second load) - Using st.spinner() in streamlit_app.py
- [ ] T038 Run quickstart.md visual verification scenarios manually
- [X] T039 Update `intuitiveness/__init__.py` to export new UI components - Already complete

---

## Phase 7: Constitution v1.2.0 Compliance (NEW)

**Purpose**: Update all UI labels to use domain language per Target User Assumption

**CRITICAL**: Users have NO familiarity with data structures. All technical terms must be replaced.

### Terminology Updates Required

| Technical Term | Domain Alternative |
|---------------|-------------------|
| Graph | "Connected information" or "How your data connects" |
| Table | "Organized information" or "Items by category" |
| Vector | "List of values" or "Your selected values" |
| Datum | "Your result" or "Computed answer" |
| Entity | "Item" or "Category" |
| Relationship | "Connection" or "Link" |
| Node | "Item" |
| Edge | "Connection" |
| Row/Column | "Item" / "Category of information" |
| Domain | "Category" |
| Ascend/Descend | "Build up" / "Explore deeper" |

### Implementation Tasks

- [X] T040 [US1] Update L4 display labels: "rows" → "items", "columns" → "categories" in `intuitiveness/ui/level_displays.py`
- [X] T041 [US1] Update L3 display labels: "entities" → "items", "relationships" → "connections" in `intuitiveness/ui/entity_tabs.py`
- [X] T042 [US1] Update L2 display labels: "domain" → "category", "table" → "organized information" in `intuitiveness/ui/level_displays.py`
- [X] T043 [US1] Update L1 display labels: "vector" → "selected values", "column" → "value type" in `intuitiveness/ui/level_displays.py`
- [X] T044 [US1] Update L0 display labels: "datum" → "result", "metric" → "computed answer" in `intuitiveness/ui/level_displays.py`
- [X] T045 [US2] Update ascent labels: "ascend" → "build up" in `intuitiveness/streamlit_app.py`
- [X] T046 [US1] Update descent labels: "descend" → "explore deeper" in `intuitiveness/streamlit_app.py`
- [X] T047 Update mode labels: "Guided Mode" → "Step-by-Step", "Free Navigation" → "Free Exploration" in `intuitiveness/streamlit_app.py`
- [X] T048 Audit all st.header(), st.subheader(), st.write() calls for technical terms
- [ ] T049 Run constitution compliance checklist from quickstart.md

**Checkpoint**: All UI labels use domain-friendly language per constitution v1.2.0

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can proceed in priority order (P1 → P2 → P2)
  - US2 and US3 can run in parallel after US1 completes
- **Polish (Phase 6)**: Depends on all user stories being complete
- **Constitution Compliance (Phase 7)**: Can run after Phase 6 or in parallel with Phase 6

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Depends on US1 display functions existing but can be independently tested
- **User Story 3 (P2)**: Depends on US1/US2 display functions existing but verifies mode consistency

### Within Each User Story

- Models/dataclasses before display functions
- Display functions before streamlit_app.py integration
- Core implementation before edge case handling
- Story complete before moving to next priority

### Parallel Opportunities

- T002, T003 can run in parallel (different files)
- T005, T006 can run in parallel (different enums, same file but no conflict)
- T008, T009 can run in parallel (different dataclasses)
- T014, T015 can run in parallel (category vs connection extraction)
- T033, T034, T036 can run in parallel (different edge cases)
- T040-T047 can run in parallel (different files/labels)

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (Descent Visualization)
4. **STOP and VALIDATE**: Run quickstart.md scenarios 1-5 for descent
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test descent navigation → Deploy/Demo (MVP!)
3. Add User Story 2 → Test ascent navigation → Deploy/Demo
4. Add User Story 3 → Verify mode consistency → Deploy/Demo
5. **NEW**: Add Phase 7 → Verify domain language compliance → Final release
6. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Existing L3→L2 tabbed display in `render_domains_step()` provides reference implementation (per research.md)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- **Constitution v1.2.0**: All user-facing labels must pass "domain curious mind" test - would a non-technical domain expert understand this?
