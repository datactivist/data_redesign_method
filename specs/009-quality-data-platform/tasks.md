# Tasks: Quality Data Platform

**Input**: Design documents from `/specs/009-quality-data-platform/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/quality_api.yaml

**Tests**: Not explicitly requested - implementation tasks only.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Project structure**: Extends existing `intuitiveness/` package
- New modules: `intuitiveness/quality/`, `intuitiveness/catalog/`
- UI components: `intuitiveness/ui/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependencies, and TabPFN authentication

- [x] T001 Install TabPFN dependencies: `pip install --upgrade tabpfn-client shap` in myenv311
- [x] T002 [P] Create `intuitiveness/quality/__init__.py` with module exports
- [x] T003 [P] Create `intuitiveness/catalog/__init__.py` with module exports
- [x] T004 [P] Create TabPFN authentication utility in `intuitiveness/quality/tabpfn_auth.py`
- [x] T005 Create base models for quality module in `intuitiveness/quality/models.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create QualityReport dataclass in `intuitiveness/quality/models.py` (from data-model.md)
- [x] T007 [P] Create FeatureProfile dataclass in `intuitiveness/quality/models.py`
- [x] T008 [P] Create Dataset dataclass in `intuitiveness/catalog/models.py`
- [x] T009 Create TabPFN wrapper with fallback logic in `intuitiveness/quality/tabpfn_wrapper.py` (API primary, local fallback)
- [x] T010 Create JSON storage utility for catalog in `intuitiveness/catalog/storage.py`
- [x] T011 Create catalog index structure in `intuitiveness/catalog/search.py`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Dataset Quality Assessment (Priority: P1) 🎯 MVP

**Goal**: Data scientists can upload a dataset and receive an automated quality report with usability score, feature importance, and anomaly indicators in under 30 seconds.

**Independent Test**: Upload a single CSV file and receive a quality report with usability score (0-100), per-feature predictive power scores, and flagged anomalous rows.

### Implementation for User Story 1

- [x] T012 [US1] Implement `assess_dataset()` function in `intuitiveness/quality/assessor.py` - TabPFN cross-validation scoring
- [x] T013 [US1] Implement `compute_usability_score()` function in `intuitiveness/quality/assessor.py` - composite formula: 40% prediction_quality + 30% data_completeness + 20% feature_diversity + 10% size_appropriateness
- [x] T014 [US1] Implement feature importance via TabPFN ablation in `intuitiveness/quality/assessor.py`
- [x] T015 [US1] Implement SHAP value computation for interpretability in `intuitiveness/quality/assessor.py`
- [x] T016 [US1] Implement task type auto-detection (classification vs regression) in `intuitiveness/quality/assessor.py`
- [x] T017 [US1] Implement missing value and categorical feature handling in `intuitiveness/quality/assessor.py`
- [x] T018 [US1] Implement dataset sampling for >10,000 rows with user notification in `intuitiveness/quality/assessor.py`
- [x] T019 [US1] Implement quality report generation in `intuitiveness/quality/report.py`
- [x] T020 [US1] Create Quality Assessment UI component in `intuitiveness/ui/quality_dashboard.py` - file upload, assessment trigger, report display
- [x] T021 [US1] Add progress feedback for assessments >5 seconds in `intuitiveness/ui/quality_dashboard.py`
- [x] T022 [US1] Add download functionality for quality reports in `intuitiveness/ui/quality_dashboard.py`

**Checkpoint**: User Story 1 is fully functional - upload CSV → get quality report with score

---

## Phase 4: User Story 2 - Feature Engineering Suggestions (Priority: P2)

**Goal**: After assessment, users receive actionable feature engineering recommendations that improve predictive power.

**Independent Test**: Provide a dataset with target column, receive ranked recommendations for feature combinations, transformations, and removals with expected impact scores.

### Implementation for User Story 2

- [x] T023 [US2] Create FeatureSuggestion dataclass in `intuitiveness/quality/models.py`
- [x] T024 [US2] Implement `suggest_features()` function in `intuitiveness/quality/feature_engineer.py` - ablation-based importance + SHAP analysis
- [x] T025 [US2] Implement suggestion types: remove (low importance), transform (skewed distributions), combine (high correlation) in `intuitiveness/quality/feature_engineer.py`
- [x] T026 [US2] Implement expected impact computation for each suggestion in `intuitiveness/quality/feature_engineer.py`
- [x] T027 [US2] Implement `apply_suggestion()` function in `intuitiveness/quality/feature_engineer.py` - applies transformation to DataFrame
- [x] T028 [US2] Add Feature Suggestions UI section in `intuitiveness/ui/quality_dashboard.py` - display ranked suggestions with apply buttons
- [x] T029 [US2] Implement re-assessment after applying suggestion to show score change in `intuitiveness/ui/quality_dashboard.py`

**Checkpoint**: User Story 2 complete - users can view and apply feature suggestions, see score improvements

---

## Phase 5: User Story 3 - Curated Dataset Catalog (Priority: P2)

**Goal**: Data scientists can browse, filter, and sort a catalog of assessed datasets by usability score, domain, and size.

**Independent Test**: Have 5+ datasets in catalog, successfully filter by usability score >70 and domain tag.

### Implementation for User Story 3

- [x] T030 [US3] Create DatasetSummary and DatasetDetail dataclasses in `intuitiveness/catalog/models.py`
- [x] T031 [US3] Implement `add_dataset()` function in `intuitiveness/catalog/storage.py` - with auto_assess option
- [x] T032 [US3] Implement `get_dataset()` function in `intuitiveness/catalog/storage.py`
- [x] T033 [US3] Implement `update_dataset()` function in `intuitiveness/catalog/storage.py`
- [x] T034 [US3] Implement `delete_dataset()` function in `intuitiveness/catalog/storage.py`
- [x] T035 [US3] Implement `filter_datasets()` function in `intuitiveness/catalog/search.py` - filter by min_score, domains, min_rows, max_rows
- [x] T036 [US3] Implement `search_datasets()` function in `intuitiveness/catalog/search.py` - full-text search in name/description
- [x] T037 [US3] Implement catalog sorting by usability_score, name, created_at, row_count in `intuitiveness/catalog/search.py`
- [x] T038 [US3] Create Catalog Browser UI in `intuitiveness/ui/catalog_browser.py` - list view with filters and sorting
- [x] T039 [US3] Create Dataset Detail view in `intuitiveness/ui/catalog_browser.py` - full quality report and download options
- [x] T040 [US3] Add "Add to Catalog" functionality in `intuitiveness/ui/quality_dashboard.py` - save assessed dataset to catalog

**Checkpoint**: User Story 3 complete - users can browse/filter catalog, view dataset details

---

## Phase 6: User Story 4 - Anomaly Detection Application (Priority: P3)

**Goal**: Domain experts can identify unusual records using TabPFN density estimation with interpretable explanations.

**Independent Test**: Upload dataset with known injected anomalies, verify flagged records include injected anomalies at high detection rate.

### Implementation for User Story 4

- [x] T041 [US4] Create AnomalyRecord dataclass in `intuitiveness/quality/models.py`
- [x] T042 [US4] Implement `detect_anomalies()` function in `intuitiveness/quality/anomaly_detector.py` - LOF-based density estimation with percentile thresholds
- [x] T043 [US4] Implement `explain_anomaly()` function in `intuitiveness/quality/anomaly_detector.py` - per-feature contribution to anomaly score
- [x] T044 [US4] Implement anomaly ranking by density percentile in `intuitiveness/quality/anomaly_detector.py`
- [x] T045 [US4] Add Anomaly Detection UI section in `intuitiveness/ui/quality_dashboard.py` - ranked list of anomalous rows
- [x] T046 [US4] Add anomaly detail view with feature attributions in `intuitiveness/ui/quality_dashboard.py`
- [x] T047 [US4] Add anomaly export functionality in `intuitiveness/ui/quality_dashboard.py`

**Checkpoint**: User Story 4 complete - users can detect, investigate, and export anomalies

---

## Phase 7: User Story 5 - Synthetic Data Generation (Priority: P3)

**Goal**: Researchers can generate synthetic samples that preserve statistical properties for data augmentation or privacy-preserving sharing.

**Independent Test**: Generate 100 synthetic rows from 500-row dataset, verify mean/std/correlations within 10% tolerance.

### Implementation for User Story 5

- [x] T048 [US5] Create SyntheticDataMetrics dataclass in `intuitiveness/quality/models.py`
- [x] T049 [US5] Implement `generate_synthetic()` function in `intuitiveness/quality/synthetic_generator.py` - TabPFN unsupervised model with Gaussian copula fallback
- [x] T050 [US5] Implement `validate_synthetic()` function in `intuitiveness/quality/synthetic_generator.py` - compare distributions and correlations
- [x] T051 [US5] Implement correlation preservation validation in `intuitiveness/quality/synthetic_generator.py` - Frobenius norm comparison
- [x] T052 [US5] Implement distribution similarity via KS-test in `intuitiveness/quality/synthetic_generator.py`
- [x] T053 [US5] Add Synthetic Data Generation UI section in `intuitiveness/ui/quality_dashboard.py` - n_samples input, temperature slider
- [x] T054 [US5] Add synthetic data quality metrics display in `intuitiveness/ui/quality_dashboard.py`
- [x] T055 [US5] Add synthetic data download as CSV in `intuitiveness/ui/quality_dashboard.py`

**Checkpoint**: User Story 5 complete - users can generate and download validated synthetic data

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T056 [P] Add edge case handling for <50 rows in `intuitiveness/quality/assessor.py`
- [x] T057 [P] Add edge case handling for >500 features in `intuitiveness/quality/assessor.py`
- [x] T058 [P] Add handling for only-categorical or only-numerical datasets in `intuitiveness/quality/assessor.py`
- [x] T059 [P] Add high-cardinality categorical handling (>100 unique values) in `intuitiveness/quality/assessor.py`
- [x] T060 [P] Add TabPFN timeout handling and graceful fallback in `intuitiveness/quality/tabpfn_wrapper.py`
- [x] T061 Integrate Quality Dashboard into main Streamlit app in `intuitiveness/streamlit_app.py`
- [x] T062 Integrate Catalog Browser into main Streamlit app in `intuitiveness/streamlit_app.py`
- [x] T063 [P] Update requirements.txt with new dependencies: tabpfn-client, tabpfn-extensions, shap
- [x] T064 Run quickstart.md validation scenarios

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 (P1): Start first - core value proposition
  - US2 (P2): Can start after US1 (uses quality report)
  - US3 (P2): Can start in parallel with US2 (catalog is independent)
  - US4 (P3): Can start after US1 (uses TabPFN density)
  - US5 (P3): Can start after US1 (uses TabPFN generation)
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: No dependencies on other stories - can start after Phase 2
- **User Story 2 (P2)**: Uses QualityReport from US1 but can be tested independently
- **User Story 3 (P2)**: Independent of US1/US2 - can start in parallel after Phase 2
- **User Story 4 (P3)**: Uses TabPFN wrapper from US1 but can be tested independently
- **User Story 5 (P3)**: Uses TabPFN wrapper from US1 but can be tested independently

### Within Each User Story

- Models before services
- Services before UI components
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 1**: T002, T003, T004 can run in parallel (different files)
- **Phase 2**: T007, T008 can run in parallel (different model files)
- **After Phase 2**: US3 can run in parallel with US2, US4, US5 (catalog is independent)
- **Phase 8**: All edge case handlers can run in parallel (different concerns)

---

## Parallel Example: User Story 1

```bash
# After Phase 2 completes, launch assessor implementation:
Task: "Implement assess_dataset() in intuitiveness/quality/assessor.py"
Task: "Implement compute_usability_score() in intuitiveness/quality/assessor.py"

# These can run in parallel (different aspects of same module):
Task: "Implement feature importance via TabPFN ablation"
Task: "Implement SHAP value computation for interpretability"
```

---

## Parallel Example: After US1

```bash
# Once US1 is complete, these user stories can proceed in parallel:
Developer A: US2 - Feature Engineering Suggestions
Developer B: US3 - Curated Dataset Catalog (independent)
Developer C: US4 - Anomaly Detection (uses TabPFN from US1)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (install deps, create modules)
2. Complete Phase 2: Foundational (models, TabPFN wrapper, storage)
3. Complete Phase 3: User Story 1 (quality assessment)
4. **STOP and VALIDATE**: Upload CSV → Get quality report with usability score
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 + 3 → Test independently → Deploy/Demo (P2 complete)
4. Add User Story 4 + 5 → Test independently → Deploy/Demo (full feature)
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (critical path)
   - After US1 complete:
     - Developer A: User Story 2
     - Developer B: User Story 3 (independent)
     - Developer C: User Story 4 or 5
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- TabPFN-client is primary (cloud API), tabpfn is fallback (local GPU)
- Performance goal: Quality assessment in <30s for datasets up to 5,000 rows
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
