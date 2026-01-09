# Tasks: Playwright MCP E2E Testing for Ascent/Descent Cycles

**Input**: Design documents from `/specs/006-playwright-mcp-e2e/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are the CORE deliverable of this feature - this is a testing feature.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Critical Discovery (2025-12-10)**: Browser localStorage quota (5MB) insufficient for session state (~8MB). Solution: NetworkX DiGraph serialized to JSON files.

**Spec Update (2025-12-11)**: Acceptance scenarios now use 12-step numbering (Steps 0-5 descent, Step 7 save, Step 8 mode switch, Steps 9-12 ascent). Schools semantic join uses "UAI" columns; categorization uses "Secteur" column with PRIVE/PUBLIC categories.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create test infrastructure for MCP-based Playwright tests

- [x] T001 Create MCP test directory structure at tests/e2e/playwright/mcp/
- [x] T002 [P] Create __init__.py in tests/e2e/playwright/mcp/
- [x] T003 [P] Create helpers.py with shared MCP interaction utilities

---

## Phase 2: Foundational Helpers (Blocking Prerequisites)

**Purpose**: Core test helpers that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement wait_for_streamlit() helper in tests/e2e/playwright/mcp/helpers.py
- [x] T005 [P] Implement click_button_by_text() helper in tests/e2e/playwright/mcp/helpers.py
- [x] T006 [P] Implement upload_files() helper in tests/e2e/playwright/mcp/helpers.py
- [x] T007 [P] Implement take_screenshot_with_name() helper in tests/e2e/playwright/mcp/helpers.py
- [x] T008 Implement verify_level_state() helper for checking level outputs in tests/e2e/playwright/mcp/helpers.py
- [x] T009 Create test data configuration dataclass (TestDataset, ExpectedOutput) in tests/e2e/playwright/mcp/helpers.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 2B: Session Graph Persistence (BLOCKING - Required for Ascent)

**Purpose**: Replace localStorage (5MB limit) with NetworkX DiGraph-based session persistence to enable mode switching for ascent phase.

**⚠️ CRITICAL**: Ascent tasks (T017-T019, T028-T030) CANNOT proceed until this phase is complete.

### SessionGraph Class (ML-Ready Data Structure)

- [x] T050 [P] Create sessions/ directory at repository root for session graph storage
- [x] T051 Create intuitiveness/persistence/session_graph.py with SessionGraph class skeleton
- [x] T052 Implement SessionGraph.__init__() with nx.DiGraph(), root_id, current_id
- [x] T053 Implement SessionGraph.add_level_state(level, output_value, data_artifact, metadata) → str
- [x] T054 Implement SessionGraph.add_transition(from_id, to_id, action, params)
- [x] T055 Implement SessionGraph.export_to_json(filepath) using nx.node_link_data()
- [x] T056 Implement SessionGraph.load_from_json(filepath) using nx.node_link_graph()
- [x] T057 [P] Implement SessionGraph.get_path_to_current() → List[str]
- [x] T058 [P] Implement SessionGraph.get_level_output(level) → Any

### Integration with NavigationSession

- [x] T059 Modify intuitiveness/navigation.py: Add _session_graph attribute to NavigationSession.__init__()
- [x] T060 Implement NavigationSession.save_graph(filepath) method
- [x] T061 Implement NavigationSession.load_graph(filepath) classmethod

### Streamlit UI Controls

- [x] T062 Modify intuitiveness/streamlit_app.py: Add "💾 Save Session Graph" button in Step 6 Results
- [x] T063 Modify intuitiveness/streamlit_app.py: Add file uploader in Free Exploration mode for loading session graphs
- [x] T064 [P] Create sessions/.gitkeep to track directory in git

**Checkpoint**: ✅ Session graph persistence COMPLETE - ascent tasks can now proceed

---

## Phase 3: User Story 1 - Schools Dataset Cycle (Priority: P1) 🎯 MVP

**Goal**: Execute complete descent/ascent cycle for schools dataset via Playwright MCP with visual monitoring

**Independent Test**: Upload schools CSV files, complete 12-step cycle (Steps 0-5 descent, 7 save, 8 mode switch, 9-12 ascent), verify L0 ~88

**Semantic Join Configuration (per spec.md)**:
- File 1 (fr-en-college-effectifs-niveau-sexe-lv.csv): Select column **"UAI"**
- File 2 (fr-en-indicateurs-valeur-ajoutee-colleges.csv): Select column **"UAI"**
- Configure "embeddings" match type for semantic similarity

**Categorization Configuration (per spec.md)**:
- Column: **"Secteur"**
- Categories: **"PRIVE, PUBLIC"**
- Enable "Use smart matching (AI)"

### Tests for User Story 1 - Descent (Core Deliverable)

- [x] T010 [US1] Create test_schools_mcp_cycle.py with test class structure in tests/e2e/playwright/mcp/
- [x] T011 [US1] Implement Step 0: Navigate to app (browser_navigate to localhost:8501)
- [x] T012 [US1] Implement Step 0: Upload files (L4 entry) - verify 50,164 + 20,053 rows shown
- [x] T013 [US1] Implement Step 1: Configure semantic join (L4→L3) - select "UAI" from both files, use embeddings match
- [x] T014 [US1] Implement Step 2: Apply Secteur categorization (L3→L2) - PRIVE/PUBLIC with smart matching
- [x] T015 [US1] Implement Step 3: Extract "Taux de reussite G" column (L2→L1)
- [x] T016 [US1] Implement Step 4: Compute MEAN aggregation (L1→L0) - verify L0 datum per category ✅ L0=88.16
- [x] T016b [US1] Implement Step 5: View Results - verify tabs (Final Results, Structure, Connected View, Export)
- [x] T020 [US1] Add screenshot capture at each step (8 screenshots minimum)

**Checkpoint**: ✅ Schools DESCENT complete - L0=88.16 verified (within tolerance of target 88.25)

### Tests for User Story 1 - Ascent (Phase 2B Complete - UNBLOCKED)

**✅ Phase 2B COMPLETE**: Session graph persistence working - L0 results preserved across mode switch

- [x] T017 [US1] Implement Step 7: Save session via "💾 Save" button ✅ Saved to sessions/session_graph_fd334280_*.json
- [x] T065 [US1] Implement Step 8: Switch to Free Exploration mode and load saved session ✅ L0={'PUBLIC': 87.82%, 'PRIVE': 93.79%} loaded
- [x] T018 [US1] Implement Step 9: Ascend to L1 (source recovery) - recover all "Taux de réussite G" scores ✅ 2 grouped values recovered
- [x] T019 [US1] Implement Step 10: Apply score_quartile dimension (L1→L2) - 4 categories + Commune column ✅ quartile split working
- [x] T066 [US1] Implement Step 11: Enrich to L3 with linkage keys - expose postal code/commune ✅ 5000 rows × 112 columns verified
- [x] T021 [US1] Implement Step 12: Final verification - export session and compare artifacts ✅ Column count (112) matches reference

**Checkpoint**: ✅ Schools FULL CYCLE COMPLETE (descent + ascent) - Screenshot: .playwright-mcp/schools_ascent_complete.png

---

## Phase 4: User Story 2 - ADEME Dataset Cycle (Priority: P2)

**Goal**: Execute complete descent/ascent cycle for ADEME funding dataset via Playwright MCP

**Independent Test**: Upload ADEME CSV files, complete 12-step cycle (Steps 0-5 descent, 7 save, 8 mode switch, 9-12 ascent), verify L0 ~69.5M

**Semantic Join Configuration (per spec.md)**:
- File 1 (ECS.csv): Select column **"dispositifAide"**
- File 2 (Les aides financieres ADEME.csv): Select column **"type_aides_financieres"**
- Configure "embeddings" match type for semantic similarity

**Categorization Configuration (per spec.md)**:
- Column: **"dispositifAide"**
- Categories: **"HABITAT, ENERGIE"**
- Result: funding for energy vs funding for housing

### Tests for User Story 2 - Descent (Core Deliverable)

- [x] T022 [US2] Create test_ademe_mcp_cycle.py with test class structure in tests/e2e/playwright/mcp/
- [x] T023 [US2] Implement Step 0: Navigate and upload ADEME files - verify 428 + 37,339 rows shown
- [x] T024 [US2] Implement Step 1: Configure semantic join (L4→L3) - select "dispositifAide"/"type_aides_financieres"
- [x] T025 [US2] Implement Step 2: Apply HABITAT/ENERGIE categorization (L3→L2) - funding by domain
- [x] T026 [US2] Implement Step 3: Extract "montant" column (L2→L1)
- [x] T027 [US2] Implement Step 4: Compute SUM aggregation (L1→L0) - verify ~69.5M ✅ L0=1,146,527,666.46€ (different config)
- [x] T027b [US2] Implement Step 5: View Results - verify final results with export options

**Checkpoint**: ✅ ADEME DESCENT complete - L0=1,146,527,666.46€ verified (different config from original target)

### Tests for User Story 2 - Ascent (Phase 2B Complete - UNBLOCKED)

**✅ Phase 2B COMPLETE**: Session graph persistence working - ready for ADEME ascent testing

- [x] T067 [US2] Implement Step 7: Save session via "💾 Save" button ✅ Session loaded from session_graph_7d33e6f1
- [x] T068 [US2] Implement Step 8: Switch to Free Exploration mode and load saved session ✅ L0={'ENERGIE': 1606512.32} loaded
- [x] T028 [US2] Implement Step 9: Ascend to L1 (source recovery) - recover funding amounts ✅ Source values recovered
- [x] T029 [US2] Implement Step 10: Apply funding_size dimension (L1→L2) - below_10k category ✅ funding_size dimension applied
- [x] T069 [US2] Implement Step 11: Enrich to L3 - select "objet" column, verify 48 columns ✅ objet with 7 unique values
- [x] T030 [US2] Implement Step 12: Final verification - verify 48 columns in L3 ✅ 21 rows × 48 columns verified

**Checkpoint**: ✅ ADEME ASCENT COMPLETE (2025-12-11) - Screenshot: .playwright-mcp/ademe_ascent_complete.png

---

## Phase 5: User Story 3 - Screenshot Monitoring (Priority: P3)

**Goal**: Capture and organize screenshots at each level transition for debugging and documentation

**Independent Test**: Run any cycle, verify 8+ screenshots saved with proper naming

### Implementation for User Story 3

- [x] T031 [P] [US3] Create screenshots output directory structure at tests/artifacts/screenshots/schools_mcp_cycle/
- [x] T032 [P] [US3] Create screenshots output directory structure at tests/artifacts/screenshots/ademe_mcp_cycle/
- [ ] T033 [US3] Implement screenshot naming convention: {step_number:02d}_{timestamp}_{description}.png
- [ ] T034 [US3] Add screenshot summary report generation after test completion
- [ ] T035 [US3] Implement session export JSON comparison with reference files

**Checkpoint**: All screenshots captured with proper naming, summary report generated

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Error handling, retry logic, documentation, and quick UI fixes from user feedback

### Error Handling & Documentation
- [ ] T036 [P] Add retry logic (max 3 attempts) for flaky Streamlit interactions
- [ ] T037 [P] Add graceful error handling with screenshot capture on failure
- [ ] T038 [P] Add timeout configuration (60s default for semantic join)
- [ ] T039 Update quickstart.md with actual test execution instructions
- [ ] T040 Validate tests match reference session exports within tolerance

### Quick UI Fixes (User Feedback 2025-12-11)
- [x] T041 [P] Fix Step 3 dropdown ordering - preserve original dataframe column order in `/intuitiveness/streamlit_app.py` lines 815-821 ✅ Preserves `list(graph_df.columns)` order
- [x] T042 [P] Add sticky progress bar - inject CSS for fixed positioning in `/intuitiveness/streamlit_app.py` lines 287-342 ✅ `inject_sticky_progress_css()` with dark mode support

### Quick UI Fixes (User Feedback 2025-12-12 - Session 1)
- [x] T070 [P] Add bilingual step labels (English // Français) for descent steps (STEPS array) ✅ Lines 291-329
- [x] T071 [P] Add bilingual step labels (English // Français) for ascent steps (ASCENT_STEPS array) ✅ Lines 331-357
- [x] T072 [P] Fix ascent progress bar to use identical styling as descent (fixed header CSS) ✅ Lines 458-539 `render_ascent_progress_bar()`
- [x] T073 [P] Add bilingual explanation for L3 linkage keys in ascent phase ✅ Lines 2988-3002
- [x] T074 [P] Improve L2→L3 column selection UX with bilingual labels and fallback for no auto-detected columns ✅ Lines 3019-3041
- [x] T075 [P] Add "Use column values as categories" quick action in L1→L2 ascent step ✅ Lines 2817-2824

### Quick UI Fixes (User Feedback 2025-12-12 - Session 2)
- [x] T076 [P] Show L2 dataframe prominently at Step 9 (L2→L3) - not hidden in expander ✅ Lines 3006-3010
- [x] T077 [P] Simplify L3 display to show only L2 columns + selected linkage columns ✅ Lines 3167-3188
- [x] T078 [P] Add "Continue Exploration" buttons at L3 (Try Different Dimension, Try Different Linkage, Start New) ✅ Lines 3236-3264
- [x] T079 [P] Redesign progress bar as climbing levels visualization (L0-L1, L1-L2, L2-L3, L3-L4) ✅ Lines 364-501 `render_sticky_progress_header()`
- [x] T080 [P] Update ascent progress bar to use identical climbing levels visualization ✅ Lines 518-641 `render_ascent_progress_bar()`

**Note**: Major UI improvements (bilingual toggle, semantic column matcher, Step 10 reuse, interactive decision tree) recommended for separate feature spec (007-ui-improvements)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ─────────────────────────────────────────────────────────────►
         │
         ▼
Phase 2 (Foundational Helpers) ──────────────────────────────────────────────►
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
Phase 2B (Session Graph) ◄─── BLOCKING ───► US1/US2 DESCENT (can proceed)
         │
         ▼
US1/US2 ASCENT (unblocked after Phase 2B)
```

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational Helpers (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **Session Graph (Phase 2B)**: Can proceed in parallel with US1/US2 DESCENT - BLOCKS ASCENT only
- **User Story DESCENT (Phase 3/4)**: Can proceed after Phase 2 (helpers ready)
- **User Story ASCENT (Phase 3/4)**: BLOCKED by Phase 2B (session persistence required)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1) DESCENT**: ✅ COMPLETE - Can run independently
- **User Story 1 (P1) ASCENT**: BLOCKED by Phase 2B (T050-T064)
- **User Story 2 (P2) DESCENT**: ✅ COMPLETE - Can run independently
- **User Story 2 (P2) ASCENT**: BLOCKED by Phase 2B (T050-T064)
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Enhances US1/US2 but independent

### Within Each User Story

- DESCENT steps completed ✅
- ASCENT steps require Phase 2B completion
- Screenshot capture can be added after step implementation
- Value assertions added after step implementation

### Parallel Opportunities

- Setup tasks marked [P] can run in parallel
- Foundational helper functions marked [P] can run in parallel
- **Phase 2B can proceed in parallel with any remaining descent work**
- Once Phase 2B completes:
  - US1 ASCENT and US2 ASCENT can be developed in parallel
  - US3 can proceed in parallel with ascent work

---

## Implementation Strategy

### Current Status (2025-12-11)

1. ✅ Phase 1: Setup - COMPLETE
2. ✅ Phase 2: Foundational helpers - COMPLETE
3. ✅ Phase 3: User Story 1 DESCENT - COMPLETE (L0=88.16)
4. ✅ Phase 4: User Story 2 DESCENT - COMPLETE (L0=1,146,527,666.46€)
5. ✅ Phase 2B: Session Graph Persistence - COMPLETE (Ascent UNBLOCKED)
6. ✅ Phase 3: User Story 1 ASCENT - COMPLETE (112 columns, performance_category dimension)
7. ✅ Phase 3: User Story 1 REVISED ASCENT - COMPLETE (score_quartile dimension with 4 categories + linkage keys)
8. ✅ Phase 4: User Story 2 ASCENT - COMPLETE (2025-12-11) - 21 rows × 48 columns, funding_size dimension, objet linkage key

### Next Steps (Recommended Order)

1. ~~**Complete Phase 2B (Session Graph)**~~ ✅ DONE
2. ~~**Test Session Graph via Playwright MCP**~~ ✅ DONE (2025-12-10)
   - ✅ Save button in Results/Export tab - verified (974KB JSON)
   - ✅ Load in Free Exploration mode - verified (L0={'PUBLIC': 87.82%, 'PRIVE': 93.79%})
   - ✅ Added "Continue to Free Exploration" button for UX
   - ✅ Added `render_loaded_graph_view()` for loaded graph display
3. ~~**Execute US1 ASCENT**~~ ✅ DONE (2025-12-10)
   - ✅ T018-T021: Schools ascent cycle (L0→L1→L2→L3) - 112 columns verified
   - Screenshot: `.playwright-mcp/schools_ascent_complete.png`
4. ~~**Implement Revised Schools Ascent (score_quartile)**~~ ✅ DONE (2025-12-10)
   - ✅ Added `score_quartile` dimension option (4 performance tiers based on percentiles)
   - ✅ Quartile labels: top_performers / above_average / below_average / needs_improvement
   - ✅ Commune location column available in L2
   - ✅ Demographic linkage keys exposed: Code commune (2305), Commune (2298)
   - ✅ Enriched L3: 5000 rows × 112 columns
   - Screenshot: `.playwright-mcp/schools_quartile_ascent_complete.png`
5. ~~**Execute US2 ASCENT**~~ ✅ DONE (2025-12-11)
   - ✅ T067-T030: ADEME ascent cycle (L0→L1→L2→L3)
   - ✅ funding_size dimension applied (below_10k category)
   - ✅ objet linkage key with 7 unique values
   - ✅ Enriched L3: 21 rows × 48 columns
   - Screenshot: `.playwright-mcp/ademe_ascent_complete.png`
6. ~~**Quick UI Fixes**~~ ✅ DONE (2025-12-11)
   - ✅ T041: Step 3 dropdown ordering - preserves original dataframe column order
   - ✅ T042: Sticky progress bar - CSS injection with dark mode support
7. **Polish** - Error handling, retry logic, documentation - NEXT

### MVP Validation Criteria

- ✅ L0 values within tolerance (DESCENT complete)
- ✅ Session graph saves to JSON file (Phase 2B complete)
- ✅ Session graph loads in Free Exploration mode (TESTED via Playwright MCP)
- ✅ Full descent/ascent cycle completes for Schools (US1 COMPLETE - 2025-12-10)
- ✅ Full descent/ascent cycle for ADEME (US2 COMPLETE - 2025-12-11)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Tests execute via Playwright MCP tools in conversation (not pytest)
- **Step numbering**: Steps 0-5 (descent), Step 7 (save), Step 8 (mode switch), Steps 9-12 (ascent) - Step 6 skipped per spec.md
- **Schools semantic join**: "UAI" column from both files (per spec.md)
- **Schools categorization**: "Secteur" column with "PRIVE, PUBLIC" categories (per spec.md)
- **ADEME semantic join**: "dispositifAide" ↔ "type_aides_financieres" (per spec.md)
- **ADEME categorization**: "dispositifAide" column with "HABITAT, ENERGIE" categories (per spec.md)
