# Feature Specification: Data Scientist Co-Pilot

**Feature Branch**: `010-quality-ds-workflow`
**Created**: 2025-12-13
**Status**: Draft
**Input**: User description: "Transform the quality assessment module into a data scientist co-pilot that gets messy CSV data modeling-ready in 60 seconds. Upload messy CSV → Get modeling-ready DataFrame in 60 seconds. TabPFN-powered quality scoring. One-click fixes with proof. Synthetic data that actually works."

## Vision Statement

**The Problem**: Data scientists spend 80% of their time on data cleaning and preparation. They hate it. They want to model, not wrangle. Current tools either assess quality (but don't fix it) or transform data (but don't prove it works).

**The Solution**: Intuitiveness becomes the "data scientist co-pilot" — a 60-second workflow that:
1. **Assesses** data quality with TabPFN (zero-shot, no training)
2. **Fixes** common issues with one click (auto-apply suggestions)
3. **Proves** improvements work (before/after benchmarks)
4. **Validates** synthetic data before use (train-on-synthetic, test-on-real)
5. **Exports** modeling-ready data (clean CSV + Python code)

**The Tagline**: *"Upload messy CSV → Get modeling-ready DataFrame in 60 seconds."*

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 60-Second Data Prep (Priority: P1)

A data scientist receives a messy CSV from a colleague. They upload it to Intuitiveness, see instant quality assessment, click "Apply All Suggestions" to fix common issues, and export a clean DataFrame ready for `model.fit()`. Total time: under 60 seconds.

**Why this priority**: This is the core value proposition — transforming hours of manual data wrangling into a 60-second automated workflow. Without this, we're just another assessment tool.

**Independent Test**: Can be fully tested by uploading a messy CSV with missing values, skewed distributions, and low-importance features, then verifying that the exported clean CSV has these issues resolved.

**Acceptance Scenarios**:

1. **Given** a data scientist uploads a CSV with 2,000 rows, **When** they click "Assess Quality", **Then** they receive a quality score and traffic light indicator (green/yellow/red) within 30 seconds.

2. **Given** a quality assessment shows 3 fixable issues, **When** the user clicks "Apply All Suggestions", **Then** all suggestions are applied in under 5 seconds and the updated score is displayed.

3. **Given** suggestions have been applied, **When** the user clicks "Export Clean CSV", **Then** they receive a downloadable file with all transformations applied.

4. **Given** a user wants to continue in Jupyter, **When** they click "Copy Python Code", **Then** they receive a code snippet that loads the clean data and splits it for modeling.

---

### User Story 2 - Synthetic Data Validation (Priority: P1)

A data scientist has an imbalanced dataset (98% normal, 2% fraud). Before using synthetic data to balance classes, they want PROOF that models trained on synthetic data will perform well on real data. They run the validation pipeline to see transfer performance before committing.

**Why this priority**: Synthetic data is useless if it doesn't work. This is the differentiator — nobody else proves synthetic data quality before use. This solves the class imbalance problem that affects 40% of ML projects.

**Independent Test**: Can be tested by providing an imbalanced dataset, generating balanced synthetic data, and verifying that the benchmark report shows train-on-synthetic/test-on-real performance metrics.

**Acceptance Scenarios**:

1. **Given** an imbalanced dataset with 98/2 class split, **When** the user requests balanced synthetic generation, **Then** the system generates samples to achieve 50/50 balance.

2. **Given** synthetic data is generated, **When** the user runs "Validate Synthetic", **Then** they see a benchmark report showing:
   - Real→Real accuracy (baseline)
   - Synthetic→Real accuracy (transfer)
   - Transfer gap percentage
   - Recommendation (safe to use / not recommended)

3. **Given** the transfer gap is less than 10%, **When** displayed to the user, **Then** the system shows "Safe to use for data augmentation" with a green indicator.

4. **Given** the transfer gap exceeds 15%, **When** displayed to the user, **Then** the system shows a warning and suggests adjusting generation parameters.

---

### User Story 3 - Before/After Improvement Benchmarks (Priority: P2)

A data scientist applies feature engineering suggestions but wants to see PROOF that changes improved their data. They see a before/after comparison showing how each transformation affected model accuracy, not just the quality score.

**Why this priority**: Data scientists trust numbers, not promises. Showing ROI of each transformation builds confidence and helps them understand which changes matter most.

**Independent Test**: Can be tested by applying transformations to a dataset with a target column and verifying that before/after accuracy metrics are displayed.

**Acceptance Scenarios**:

1. **Given** a dataset with a designated target column, **When** the user applies a single suggestion, **Then** they see the accuracy improvement attributed to that change (e.g., "Log transform on 'price' → +2.3% accuracy").

2. **Given** multiple suggestions are available, **When** the user clicks "Apply All", **Then** they see cumulative improvement with per-transformation breakdown.

3. **Given** a transformation degrades accuracy, **When** displayed to user, **Then** it shows a warning with the negative impact (e.g., "-1.2% accuracy — consider reverting").

---

### User Story 4 - Traffic Light Readiness Indicator (Priority: P2)

A data scientist glances at their assessed dataset and immediately knows if it's ready for modeling (green), needs minor fixes (yellow), or has serious issues (red). No need to interpret complex scores.

**Why this priority**: Instant go/no-go decisions reduce cognitive load. Data scientists want to know "can I start modeling?" not "what does 72/100 mean?"

**Independent Test**: Can be tested by uploading datasets of varying quality and verifying correct traffic light assignment.

**Acceptance Scenarios**:

1. **Given** a dataset with usability score ≥ 80, **When** assessment completes, **Then** user sees a green "READY FOR MODELING" indicator with message "Export and start training!"

2. **Given** a dataset with usability score 60-79, **When** assessment completes, **Then** user sees a yellow "FIXABLE" indicator with message "N automated fixes will improve score to X".

3. **Given** a dataset with usability score < 60, **When** assessment completes, **Then** user sees a red "NEEDS WORK" indicator with message "Significant data issues. Review recommendations below."

---

### User Story 5 - Edge Case Augmentation (Priority: P3)

A data scientist working on fraud detection has only 50 fraud cases in their dataset. They want to generate more synthetic fraud samples (not random samples) to improve model performance on rare events.

**Why this priority**: Rare event prediction is a common ML challenge. Targeted augmentation of minority classes is more valuable than random synthetic generation.

**Independent Test**: Can be tested by requesting synthetic samples similar to a specific class and verifying that generated samples have similar feature distributions to that class.

**Acceptance Scenarios**:

1. **Given** a dataset with rare class (< 5% of rows), **When** user requests "Augment Rare Cases", **Then** they can specify how many synthetic samples to generate for the rare class only.

2. **Given** rare case augmentation is complete, **When** user views synthetic samples, **Then** the samples have feature distributions similar to original rare cases (within 15% deviation).

---

### Edge Cases

- What happens when a dataset has fewer than 50 rows? (Minimum threshold for reliable TabPFN assessment — show warning)
- What happens when no target column is specified? (Allow assessment but disable accuracy-based benchmarks)
- What happens when TabPFN is not available (no GPU, no API)? (Fall back to heuristic scoring with warning about reduced accuracy)
- How does the system handle datasets with only categorical features? (TabPFN handles this natively)
- What happens when synthetic generation fails to preserve correlations? (Show warning, suggest alternative parameters)
- What happens when the user exports but hasn't applied any suggestions? (Confirm dialog: "Export original data without improvements?")

---

## Requirements *(mandatory)*

### Functional Requirements

**Core Workflow (P0)**

- **FR-001**: System MUST display a traffic light readiness indicator (green/yellow/red) after quality assessment with clear messaging for each state
- **FR-002**: System MUST provide an "Apply All Suggestions" button that applies all high-confidence transformations in a single click
- **FR-003**: System MUST allow export of the transformed dataset as CSV with all applied suggestions
- **FR-004**: System MUST provide a copyable Python code snippet for loading exported data in Jupyter/Colab
- **FR-005**: System MUST complete the upload→assess→fix→export workflow in under 60 seconds for datasets up to 5,000 rows

**Synthetic Data Validation (P0)**

- **FR-006**: System MUST benchmark synthetic data by training models on synthetic and testing on held-out real data
- **FR-007**: System MUST report the "transfer gap" (accuracy difference between real→real and synthetic→real)
- **FR-008**: System MUST recommend whether synthetic data is safe to use based on transfer gap threshold (default: 10%)
- **FR-009**: System MUST support class-balanced synthetic generation (equal samples per class regardless of original distribution)

**Improvement Benchmarks (P1)**

- **FR-010**: System MUST show before/after accuracy comparison when suggestions are applied
- **FR-011**: System MUST attribute accuracy changes to specific transformations (e.g., "+2.3% from log transform")
- **FR-012**: System MUST warn users when a transformation degrades accuracy

**Edge Case Augmentation (P2)**

- **FR-013**: System MUST allow targeted synthetic generation for specific class values (e.g., "generate 100 more fraud cases")
- **FR-014**: System MUST validate that generated samples match the target class distribution

**Export & Integration (P1)**

- **FR-015**: System MUST export clean DataFrame as CSV, Pickle, or Parquet formats
- **FR-016**: System MUST generate Python code snippet with train/test split for immediate modeling
- **FR-017**: System MUST preserve a record of all applied transformations in the export metadata

### Key Entities

- **AssessedDataset**: Original uploaded data with quality assessment results (usability score, feature profiles, issues detected)
- **TransformedDataset**: Dataset after applying suggestions, with before/after metrics and transformation log
- **SyntheticBenchmark**: Results of train-on-synthetic/test-on-real validation (transfer gap, per-model metrics, recommendation)
- **TransformationResult**: Record of a single transformation with accuracy delta and confidence
- **ExportPackage**: Clean dataset file + Python code snippet + transformation metadata

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users complete the upload→assess→fix→export workflow in under 60 seconds for datasets up to 5,000 rows (measured via session timing)
- **SC-002**: 90% of users correctly interpret the traffic light indicator on first use (validated via user testing)
- **SC-003**: Before/after benchmarks show accuracy improvement in 70% of cases where suggestions are applied (measured via logged metrics)
- **SC-004**: Synthetic-to-real transfer gap is less than 10% for 80% of balanced synthetic generations (measured via benchmark reports)
- **SC-005**: Data scientists report saving at least 2 hours of data preparation time per dataset compared to manual workflow (validated via user surveys)
- **SC-006**: 80% of users who generate synthetic data run the validation benchmark before using it (measured via usage analytics)
- **SC-007**: Exported Python code snippets execute without errors in fresh Jupyter environments (validated via automated testing)

---

## Dependencies & Assumptions

### Dependencies

- **009-quality-data-platform**: This feature extends the quality assessment module from spec 009
- **TabPFN availability**: Core functionality requires TabPFN (local or cloud API) — fallback mode provides reduced functionality

### Assumptions

- Data scientists are the primary users (technical, comfortable with Python, impatient with data prep)
- Most datasets are tabular CSV files under 10,000 rows (TabPFN sweet spot)
- Users have a clear target column for supervised assessment (unsupervised mode is secondary)
- Python/Jupyter is the primary modeling environment (other export formats are lower priority)
- Users prefer automated defaults over manual configuration (minimize decisions)

---

## Out of Scope

- Real-time data streaming assessment
- Integration with specific ML frameworks (PyTorch, TensorFlow)
- Automated model training (we prepare data, user trains models)
- Privacy guarantees for synthetic data (future enhancement)
- Support for datasets over 10,000 rows (TabPFN limitation)
