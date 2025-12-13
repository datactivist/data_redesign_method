# Tasks: Data Scientist Co-Pilot

**Input**: Design documents from `/specs/010-quality-ds-workflow/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/quality_api.yaml, research.md, quickstart.md

**Tests**: Included based on spec requirements and quickstart.md test scenarios.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Quality module**: `intuitiveness/quality/`
- **UI module**: `intuitiveness/ui/`
- **Tests**: `tests/unit/`, `tests/integration/`, `tests/e2e/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and data model foundations

- [ ] T001 Create feature branch `010-quality-ds-workflow` from main
- [ ] T002 [P] Create test artifacts directory at `tests/artifacts/quality_ds/`
- [ ] T003 [P] Add new data models to `intuitiveness/quality/models.py`:
  - `SyntheticBenchmarkReport`
  - `ModelBenchmarkResult`
  - `TransformationResult`
  - `TransformationLog`
  - `ExportPackage`
  - `ReadinessIndicator`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Create `intuitiveness/quality/benchmark.py` with module structure and imports
- [ ] T005 [P] Create `intuitiveness/quality/exporter.py` with module structure and imports
- [ ] T006 [P] Add session state keys for transformed datasets in `intuitiveness/ui/quality_dashboard.py`:
  - `SESSION_KEY_TRANSFORMED_DF`
  - `SESSION_KEY_TRANSFORMATION_LOG`
  - `SESSION_KEY_BENCHMARK_REPORT`
- [ ] T007 Update `intuitiveness/quality/__init__.py` with new module exports

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - 60-Second Data Prep (Priority: P1) MVP

**Goal**: Upload messy CSV, click "Apply All Suggestions", export clean DataFrame with Python code snippet

**Independent Test**: Upload a messy CSV with missing values, skewed distributions, and low-importance features, then verify the exported clean CSV has these issues resolved.

### Tests for User Story 1

- [ ] T008 [P] [US1] Unit test for `apply_all_suggestions()` in `tests/unit/test_assessor_apply_all.py`
- [ ] T009 [P] [US1] Unit test for export functions in `tests/unit/test_exporter.py`
- [ ] T010 [P] [US1] Integration test for upload→assess→fix→export workflow in `tests/integration/test_quality_workflow.py`

### Implementation for User Story 1

- [ ] T011 [US1] Implement `apply_all_suggestions()` in `intuitiveness/quality/assessor.py`:
  - Accept DataFrame, suggestions list, optional target_column
  - Apply suggestions sequentially with error handling
  - Track accuracy before/after each transformation
  - Return (transformed_df, TransformationLog)
  - Performance: <5 seconds for 10 suggestions

- [ ] T012 [US1] Implement `export_dataset()` in `intuitiveness/quality/exporter.py`:
  - Accept DataFrame, format (csv/pickle/parquet), dataset_name, transformation_log
  - Return ExportPackage with binary data and metadata

- [ ] T013 [P] [US1] Implement `generate_python_snippet()` in `intuitiveness/quality/exporter.py`:
  - Generate Python code for loading exported data
  - Include train/test split boilerplate
  - Include commented modeling starter code

- [ ] T014 [US1] Add "Apply All Suggestions" button component to `intuitiveness/ui/quality_dashboard.py`:
  - Display count of suggestions to apply
  - Show progress during application
  - Update session state with transformed DataFrame
  - Refresh quality display after application

- [ ] T015 [US1] Add export section to `intuitiveness/ui/quality_dashboard.py`:
  - Format selector (CSV/Pickle/Parquet)
  - Download button with proper MIME type
  - Copyable Python code snippet with syntax highlighting
  - Warning if exporting without applying suggestions

**Checkpoint**: User Story 1 complete - Upload→Assess→Fix→Export workflow functional

---

## Phase 4: User Story 2 - Synthetic Data Validation (Priority: P1)

**Goal**: Benchmark train-on-synthetic/test-on-real performance to prove synthetic data quality before use

**Independent Test**: Provide imbalanced dataset, generate balanced synthetic data, verify benchmark report shows transfer gap metrics.

### Tests for User Story 2

- [ ] T016 [P] [US2] Unit test for `benchmark_synthetic()` in `tests/unit/test_benchmark.py`
- [ ] T017 [P] [US2] Unit test for `generate_balanced_synthetic()` in `tests/unit/test_benchmark.py`
- [ ] T018 [P] [US2] Integration test for synthetic validation workflow in `tests/integration/test_synthetic_validation.py`

### Implementation for User Story 2

- [ ] T019 [US2] Implement `benchmark_synthetic()` in `intuitiveness/quality/benchmark.py`:
  - Split real data: 80% train, 20% held-out test
  - Generate synthetic from train split only
  - Benchmark LogisticRegression, RandomForest, XGBoost
  - Calculate transfer gap per model
  - Return SyntheticBenchmarkReport with recommendation
  - Performance: <30 seconds for 5000 rows

- [ ] T020 [US2] Implement `generate_balanced_synthetic()` in `intuitiveness/quality/benchmark.py`:
  - Generate synthetic samples per class
  - Default: match largest class size
  - Support custom samples_per_class parameter
  - Return balanced DataFrame (original + synthetic)

- [ ] T021 [US2] Add benchmark section to `intuitiveness/ui/quality_dashboard.py`:
  - "Generate Balanced Synthetic" button with class balance preview
  - "Validate Synthetic" button to run benchmark
  - Benchmark results display:
    - Real→Real accuracy (baseline)
    - Synthetic→Real accuracy (transfer)
    - Transfer gap percentage per model
    - Mean/max/min transfer gap
  - Recommendation indicator (safe/caution/not recommended)
  - Color-coded: green (<10%), yellow (10-15%), red (>15%)

**Checkpoint**: User Story 2 complete - Synthetic validation proves data quality before use

---

## Phase 5: User Story 3 - Before/After Improvement Benchmarks (Priority: P2)

**Goal**: Show proof that transformations improved model accuracy, with per-transformation breakdown

**Independent Test**: Apply transformations with a target column, verify before/after accuracy metrics displayed.

### Tests for User Story 3

- [ ] T022 [P] [US3] Unit test for accuracy tracking in `apply_all_suggestions()` in `tests/unit/test_accuracy_tracking.py`

### Implementation for User Story 3

- [ ] T023 [US3] Enhance `apply_all_suggestions()` in `intuitiveness/quality/assessor.py`:
  - Add `quick_benchmark()` helper using RandomForest
  - Calculate accuracy_before and accuracy_after per transformation
  - Detect and flag accuracy degradations
  - Calculate total_accuracy_improvement in TransformationLog

- [ ] T024 [US3] Add before/after comparison display to `intuitiveness/ui/quality_dashboard.py`:
  - Bar chart showing accuracy progression
  - Per-transformation breakdown table:
    - Transformation type
    - Column affected
    - Accuracy delta (e.g., "+2.3%")
  - Warning styling for negative deltas
  - Summary: "Total improvement: +X.X%"

**Checkpoint**: User Story 3 complete - Users see proof of improvement per transformation

---

## Phase 6: User Story 4 - Traffic Light Readiness Indicator (Priority: P2)

**Goal**: Instant go/no-go visual indicator based on usability score

**Independent Test**: Upload datasets of varying quality, verify correct traffic light assignment (green/yellow/red).

### Tests for User Story 4

- [ ] T025 [P] [US4] Unit test for `get_readiness_indicator()` in `tests/unit/test_readiness_indicator.py`

### Implementation for User Story 4

- [ ] T026 [US4] Implement `get_readiness_indicator()` in `intuitiveness/quality/assessor.py`:
  - Accept usability score, n_suggestions, estimated_improvement
  - Return ReadinessIndicator with status, color, title, message
  - Thresholds: 80+ green, 60-79 yellow, <60 red
  - Dynamic messaging based on fixable state

- [ ] T027 [US4] Add traffic light indicator component to `intuitiveness/ui/quality_dashboard.py`:
  - Large, prominent visual indicator at top of results
  - Color-coded: green (#22c55e), yellow (#eab308), red (#ef4444)
  - Status title: "READY FOR MODELING" / "FIXABLE" / "NEEDS WORK"
  - Actionable message:
    - Green: "Export and start training!"
    - Yellow: "N automated fixes will improve score to X"
    - Red: "Significant data issues. Review recommendations below."
  - Use CSS styling for visual prominence

**Checkpoint**: User Story 4 complete - Users instantly know if data is modeling-ready

---

## Phase 7: User Story 5 - Edge Case Augmentation (Priority: P3)

**Goal**: Targeted synthetic generation for rare classes (e.g., fraud cases)

**Independent Test**: Request synthetic samples for minority class, verify generated samples match class distribution.

### Tests for User Story 5

- [ ] T028 [P] [US5] Unit test for class-targeted synthetic generation in `tests/unit/test_targeted_synthetic.py`

### Implementation for User Story 5

- [ ] T029 [US5] Implement `generate_targeted_synthetic()` in `intuitiveness/quality/benchmark.py`:
  - Accept DataFrame, target_column, target_class_value, n_samples
  - Filter to target class, generate synthetic samples
  - Validate distribution similarity (within 15% deviation)
  - Return synthetic samples with validation metrics

- [ ] T030 [US5] Add targeted augmentation section to `intuitiveness/ui/quality_dashboard.py`:
  - Class distribution display (original data)
  - Class selector for rare class
  - Sample count input (default: match majority class)
  - "Augment Rare Cases" button
  - Generated samples preview
  - Distribution comparison chart (original vs augmented)

**Checkpoint**: User Story 5 complete - Users can boost rare classes for better model performance

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T031 [P] Add fallback mode when TabPFN unavailable in all benchmark functions
- [ ] T032 [P] Add progress indicators for long operations (benchmark, apply all, export)
- [ ] T033 Add confirmation dialog when exporting without applying suggestions
- [ ] T034 [P] Add minimum row check (50 rows) with user-friendly warning
- [ ] T035 [P] Update `intuitiveness/quality/__init__.py` with all new exports
- [ ] T036 Run all quickstart.md scenarios and validate
- [ ] T037 [P] E2E test: full upload→assess→fix→export workflow in `tests/e2e/test_quality_dashboard.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 and US2 are both P1 - can proceed in parallel
  - US3 depends on US1 (uses `apply_all_suggestions`)
  - US4 can proceed independently after Foundational
  - US5 can proceed independently after Foundational
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

```
Foundational (T004-T007)
    │
    ├──▶ US1: 60-Second Data Prep (T008-T015)
    │         └──▶ US3: Before/After Benchmarks (T022-T024)
    │
    ├──▶ US2: Synthetic Validation (T016-T021)
    │         └──▶ US5: Edge Case Augmentation (T028-T030)
    │
    └──▶ US4: Traffic Light Indicator (T025-T027) [independent]
```

### Parallel Opportunities

**Within Setup/Foundational**:
- T002, T003 can run in parallel
- T004, T005, T006 can run in parallel

**Within User Story 1**:
- T008, T009, T010 (tests) can run in parallel
- T013 can run in parallel with T012

**Within User Story 2**:
- T016, T017, T018 (tests) can run in parallel
- T019, T020 can run sequentially (T020 uses T019's structure)

**Across User Stories**:
- After Foundational: US1, US2, US4 can all start in parallel
- US3 waits for US1 completion
- US5 can start after US2 (reuses synthetic generation)

---

## Parallel Example: Launching User Stories

```bash
# After Foundational phase completes, launch P1 stories in parallel:

# Agent A: User Story 1 - 60-Second Data Prep
Task: "Implement apply_all_suggestions() in intuitiveness/quality/assessor.py"
Task: "Implement export_dataset() in intuitiveness/quality/exporter.py"

# Agent B: User Story 2 - Synthetic Validation
Task: "Implement benchmark_synthetic() in intuitiveness/quality/benchmark.py"
Task: "Implement generate_balanced_synthetic() in intuitiveness/quality/benchmark.py"

# Agent C: User Story 4 - Traffic Light Indicator
Task: "Implement get_readiness_indicator() in intuitiveness/quality/assessor.py"
Task: "Add traffic light indicator component to quality_dashboard.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 4)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 6: User Story 4 - Traffic Light Indicator (quick visual win)
4. Complete Phase 3: User Story 1 - 60-Second Data Prep
5. **STOP and VALIDATE**: Test core workflow independently
6. Deploy/demo if ready - "Upload→See Traffic Light→Fix→Export" works!

### Recommended Full Delivery Order

1. **Setup + Foundational** → Foundation ready
2. **US4 (Traffic Light)** → Quick visual improvement
3. **US1 (60-Second Data Prep)** → Core value proposition (MVP!)
4. **US2 (Synthetic Validation)** → Differentiator feature
5. **US3 (Before/After Benchmarks)** → Proof of improvement
6. **US5 (Edge Case Augmentation)** → Advanced feature
7. **Polish** → Production-ready

### Parallel Team Strategy

With 3 developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - **Developer A**: US1 (60-Second Data Prep) → US3 (Before/After)
   - **Developer B**: US2 (Synthetic Validation) → US5 (Edge Case)
   - **Developer C**: US4 (Traffic Light) → Polish tasks
3. Stories complete and integrate independently

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Tasks** | 37 |
| **Setup Phase** | 3 tasks |
| **Foundational Phase** | 4 tasks |
| **User Story 1 (P1)** | 8 tasks (3 test, 5 impl) |
| **User Story 2 (P1)** | 6 tasks (3 test, 3 impl) |
| **User Story 3 (P2)** | 3 tasks (1 test, 2 impl) |
| **User Story 4 (P2)** | 3 tasks (1 test, 2 impl) |
| **User Story 5 (P3)** | 3 tasks (1 test, 2 impl) |
| **Polish Phase** | 7 tasks |
| **Parallel Opportunities** | 18 tasks marked [P] |
| **MVP Scope** | US1 + US4 (14 tasks) |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Performance targets: <60 seconds full workflow, <30 seconds benchmark
