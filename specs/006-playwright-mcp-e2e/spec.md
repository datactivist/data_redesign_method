# Feature Specification: Playwright MCP E2E Testing for Ascent/Descent Cycles

**Feature Branch**: `006-playwright-mcp-e2e`
**Created**: 2025-12-09
**Status**: Draft
**Input**: User description: "Develop new specs for final testing for a full ascent/descent cycle using Playwright MCP for visual monitoring. The test should reproduce the exact transformations from test0_schools_session_export.json and test1_ademe_session_export.json through the Streamlit interface."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Visual Monitoring of Schools Data Cycle (Priority: P1)

As a developer/tester, I want to visually monitor the complete descent/ascent cycle for the French middle schools dataset through the Streamlit interface, so I can verify that each level transition produces the expected data transformations and the system behavior matches the reference session export. **IMPORTANT** here the expected final output should be a dataset where we have a way to measure how the location of the middle school influences the "Taux de réussite".

**Why this priority**: This is the primary test case that validates the core descent/ascent functionality. The schools dataset has well-defined expected outputs and demonstrates the complete 8-step cycle (L4→L3→L2→L1→L0→L1→L2→L3).

**Independent Test**: Can be fully tested by uploading 2 CSV files (fr-en-college-effectifs-niveau-sexe-lv.csv and fr-en-indicateurs-valeur-ajoutee-colleges.csv), performing the complete cycle, and verifying the final L0 datum is around 88 (mean success rate).

**Acceptance Scenarios** (Guided Workflow - 6 Steps):

1. **Step 0 - Upload Data**: **Given** the Streamlit app is running at localhost:8501, **When** I click "Browse files" and upload both schools CSV files, **Then** the system shows 2 files accepted with row counts (50,164 and 20,053 rows).

2. **Step 1 - Define Items (L4→L3)**: **Given** files are uploaded, **When** I click "Next →" to enter the semantic join wizard, select columns from File 1 "UAI" and columns from File 2 "UAI" to create row vectors, configure "embeddings" match type, then click "Continue →" to build the joined dataset, **Then** the system creates an L3 linked table by vectorizing each row using selected columns and matching rows by semantic similarity.

3. **Step 2 - Define Categories (L3→L2)**: **Given** L3 linked table exists and I click "Continue →", **When** I select the column "Secteur" for categorization, enter "PRIVE, PUBLIC" as categories, enable "Use smart matching (AI)", and click "🔄 Categorize Data", **Then** the system creates L2 categories with PRIVE and PUBLIC distribution.

4. **Step 3 - Select Values (L2→L1)**: **Given** L2 categorized table exists and I click "Continue →", **When** I select "Taux de reussite G" from the column dropdown and click "Extract Values", **Then** the system creates L1 vector.

5. **Step 4 - Choose Computation (L1→L0)**: **Given** L1 vector exists and I click "Continue →", **When** I select "mean" from the aggregation dropdown and click "Compute Metrics", **Then** the system displays L0 datum for the two categorized groups (PRIVE and PUBLIC).

6. **Step 5 - View Results**: **Given** L0 datum computed and I click "View Results →", **When** I view the Results step, **Then** the system shows final results with tabs for Final Results, Structure, Connected View, and Export.

7. **Save Session Before Mode Switch**: **Given** descent is complete (Step 6 Results), **When** I click "💾 Save" in the sidebar to save the current session, **Then** the system saves all the artifacts in memory (a complete memory is made of all the artifacts present in the folder "/Users/arthursarazin/Documents/data_redesign_method/tests/artifacts/20251208_domain_specific_v2/test0_schools", excluded the one containing "ascent" in their names).

8. **Switch to Free Exploration and Load Session**: **Given** session saved, **When** I switch to "Free Exploration" mode and load the saved session, **Then** the system restores the L0 result and shows navigation tree with ascent to L1 option.

9. **Ascent to L1 (L0→L1)**: **Given** Free Exploration mode active with L0 result loaded, **When** I click "Ascend to L1" in the navigation, **Then** the system recovers all individual "Taux de réussite G" scores for both categories (PRIVE and PUBLIC) into a single L1 vector with scores. 

10. **Apply Quartile Dimension (L1→L2)**: **Given** L1 vector with 410 scores, **When** I select "score_quartile" dimension with data-driven percentile boundaries (25th/50th/75th) and click "Apply Dimension", **Then** the system creates L2 with four categories (top_performers / above_average / below_average / needs_improvement) plus the "Commune" location column. Values at boundary go to upper category (e.g., score == 50th goes to above_average)

11. **Enrich to L3 with Linkage Key (L2→L3)**: **Given** L2 categorized with quartiles and location, **When** I click "Enrich Data" to expand back to L3, **Then** the system exposes all avaible columns in original raw datasets, then click postal code/commune exposed as a linkage key for future demographic joins (using only existing columns).

12. **Final Verification**: **Given** the full cycle completes, **When** I export the session, **Then** I get all generated artifacts that are like the one present here : "/Users/arthursarazin/Documents/data_redesign_method/tests/artifacts/20251208_domain_specific_v2/test0_schools" (files with "ascent" on their name)

---

### User Story 2 - Visual Monitoring of ADEME Funding Cycle (Priority: P2)

As a developer/tester, I want to visually monitor the complete descent/ascent cycle for the ADEME environmental funding dataset through the Streamlit interface, so I can verify the system handles the descent/ascent cycle correctly and produces a useful redesign of the original datasets. Here it produces financial aggregations correctly and produces the expected total funding amount. **IMPORTANT** here the expected final output should be a dataset where we can know where ADEME spent most of its funding on 

**Why this priority**: This test case validates the system with a different data domain (financial data) and uses SUM aggregation instead of MEAN, ensuring the system works across different data types and aggregation methods.

**Independent Test**: Can be fully tested by uploading 2 CSV files (ECS.csv and Les aides financieres ADEME.csv), performing the complete cycle, and verifying the final L0 datum equals 69,586,180.93 euros (total funding).

**Acceptance Scenarios** (Guided Workflow - 6 Steps):

1. **Step 0 - Upload Data**: **Given** the Streamlit app is running at localhost:8501, **When** I click "Browse files" and upload both ADEME CSV files (ECS.csv, Les aides financieres ADEME.csv), **Then** the system shows 2 files accepted with row counts (428 and 37,339 rows).

2. **Step 1 - Define Items (L4→L3)**: **Given** files are uploaded, **When** I click "Next →" to enter the semantic join wizard, select columns from File 1 "dispositifAide" and columns from File 2 "type_aides_financieres" to create row vectors, configure "embeddings" match type, then click "Continue →" to build the joined dataset, **Then** the system creates an L3 linked table by vectorizing each row using selected columns as coordinates and matching rows by semantic similarity.

3. **Step 2 - Define Categories (L3→L2)**: **Given** L3 linked table exists, **When** I select "dispositifAide" column, enter "HABITAT,ENERGIE" as categories, and click "🔄 Categorize Data", **Then** the system creates L2 categories with funding for energy and funding for housing.

4. **Step 3 - Select Values (L2→L1)**: **Given** L2 categorized table exists, **When** I select "montant" from the column dropdown and click "Extract Values" **Then** the system creates L1 vector

5. **Step 4 - Choose Computation (L1→L0)**: **Given** L1 vector exists, **When** I select "sum" from the aggregation dropdown and click "Compute Metrics", **Then** the system displays L0 datum approximately 69,586,180.93 euros.

6. **Step 5 - View Results**: **Given** L0 datum computed, **When** I click "View Results →" and view the Results step, **Then** the system shows final results with export options.

7. **Save Session Before Mode Switch**: **Given** descent is complete (Step 6 Results), **When** I click "💾 Save" in the sidebar to save the current session, **Then** the system saves **AND export** all the artifacts in memory (a complete memory is made of all the artifacts **like the one** present in the folder "/Users/arthursarazin/Documents/data_redesign_method/tests/artifacts/20251208_domain_specific_v2/test0_schools", excluded the one containing "ascent" in their names).

8. **Switch to Free Exploration and Load Session**: **Given** session saved, **When** I switch to "Free Exploration" mode and load the saved session, **Then** the system restores the L0 result and shows navigation tree with ascent to L1 option.

9. **Ascent to L1 (L0→L1)**: **Given** Free Exploration mode active with L0 result, **When** I click "Ascend to L1" in the navigation, **Then** the system recovers the L1 source values (~450 individual funding amounts).

10. **Apply Dimension (L1→L2)**: **Given** L1 values recovered, **When** I select "funding_size" dimension with 10k euro threshold and click "Apply Dimension", **Then** the system creates L2 with above_10k (~301) and below_10k (~149) categories.

11. **Enrich to L3 (L2→L3)**: **Given** L2 categorized, **When** I click "Enrich Data" to expand back to L3, **Then** the system displays all avaible colmumns in original raw dataset that can be linked, **Then** I click "objet" from the "Les aides financières ADEME.csv". It produces enriched L3 table with 48 columns.

12. **Final Verification**: **Given** the full cycle completes, **When** I export the session, **Then** the L0 datum equals 69,586,180.93 (±0.01 tolerance) and enriched L3 has 48 columns.

---

### User Story 3 - Interactive Test Monitoring with Screenshots (Priority: P3)

As a developer/tester, I want the test execution to capture screenshots at each level transition and export session data, so I can review the visual state of the interface at each step and debug any failures.

**Why this priority**: This supports debugging and documentation of test runs, but is not required for the core test verification logic.

**Independent Test**: Can be tested by running any cycle and verifying screenshots are saved to the designated directory with proper naming convention.

**Acceptance Scenarios**:

1. **Given** a test cycle is in progress, **When** each level transition occurs, **Then** a timestamped screenshot is captured showing the current interface state.

2. **Given** a test cycle completes, **When** the final state is reached, **Then** a session export JSON file is generated with the complete navigation tree.

3. **Given** screenshots are captured, **When** viewing the test output directory, **Then** screenshots are named with step number, timestamp, and descriptive step name (e.g., "01_143052_initial_state.png").

---

### Edge Cases

- What happens when the Streamlit app is not running or unreachable?
  - Test should fail gracefully with clear error message indicating connection failure

- What happens when CSV files have encoding issues (UTF-8 vs Latin-1)?
  - The smart CSV loader should auto-detect encoding and load successfully

- What happens when semantic join produces zero matches?
  - Test should detect empty result and report validation failure

- What happens when the interface elements change their selectors?
  - Test should use flexible selectors (text-based) and report which element could not be found

- What happens when processing takes longer than expected (large files)?
  - Test should have configurable timeouts and wait for Streamlit spinner to disappear

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Test execution MUST use Playwright MCP tools to enable real-time visual monitoring of the Streamlit interface during test runs.

- **FR-002**: Test MUST upload multiple CSV files simultaneously through the Streamlit file upload widget.

- **FR-003**: Test MUST navigate through the Guided Workflow's "Define Items" step (Step 1) to configure semantic joins between data sources. The user selects multiple columns from each file to form row vectors—each row is vectorized using selected column values as coordinates (multi-dimensional point). Semantic matching compares these row vectors between files to find similar records. This is a multi-column vectorization approach, NOT a simple column-value lookup.

- **FR-004**: Test MUST apply categorization transformations at L2 level with configurable dimension names and categorization logic.

- **FR-005**: Test MUST extract specific columns to create L1 vectors with optional grouping and aggregation.

- **FR-006**: Test MUST compute L0 datum using specified aggregation method (MEAN, SUM, etc.).

- **FR-007**: Test MUST execute the ascent phase, reversing from L0 back through L1, L2, to L3 with alternative dimensions.

- **FR-008**: Test MUST capture screenshots at each major step with descriptive filenames.

- **FR-009**: Test MUST wait for Streamlit spinners/loading indicators to complete before proceeding to next action.

- **FR-010**: Test MUST validate that output values match expected results from reference session exports.

- **FR-011**: Test MUST support iterative debugging by continuing execution even when individual assertions fail, collecting all errors for review.

- **FR-012**: Test MUST export the session state as JSON at the end of each cycle for comparison with reference exports.

### Key Entities

- **Session Export**: JSON document capturing the complete navigation tree, cumulative outputs at each level, and design choices made during the cycle. Contains nodes with level, action, timestamp, parameters, and output snapshots.

- **Level State**: Current position in the abstraction hierarchy (L4=raw files, L3=linked table, L2=categorized, L1=vector, L0=datum). Each state has associated data and available actions.

- **Transformation Step**: A discrete action that moves between levels (descend/ascend) with specific parameters (join columns, categorization rules, aggregation methods).

- **Row Vector (L4→L3 Semantic Join)**: During semantic join, each row from a file is converted into a multi-dimensional vector. The user selects N columns from File 1 and M columns from File 2. Each row becomes a point in vector space where column values are coordinates: `vector = (col1_value, col2_value, ...)`. Semantic similarity is computed between row vectors from both files to find matches. This is NOT a simple column-value lookup—it's a multi-column vectorization approach.

- **Screenshot Artifact**: PNG image captured at each step, stored with sequential numbering and descriptive names for debugging and documentation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Test successfully completes the full 8-step descent/ascent cycle for schools dataset within 5 minutes, producing L0 value within 0.01 of observed 88.25 (approximation is OK)

- **SC-002**: Test successfully completes the full 8-step descent/ascent cycle for ADEME dataset within 5 minutes, producing L0 value within 0.01 of expected 69,586,180.93.

- **SC-003**: Each level transition produces output matching the reference session export's row counts within 5% tolerance (to account for minor data variations).

- **SC-004**: Test captures minimum 8 screenshots per cycle, one for each major step (upload, L3 join, L2 categorize, L1 extract, L0 compute, ascent L1, ascent L2, ascent L3).

- **SC-005**: Test iteration continues until full cycle completes without interface errors, with maximum 3 retry attempts per step.

- **SC-006**: Final session export JSON contains valid navigation tree with all 8 nodes and matching structure to reference exports.

## Clarifications

### Session 2025-12-09

- Q: How does the L4→L3 semantic join actually work when selecting columns? → A: Selected columns become vector coordinates; each row is vectorized as a multi-dimensional point for semantic comparison.
- Q: How should the ascent phase (L0→L1→L2→L3) be accessed after descent completion? → A: Descent uses Step-by-Step mode (Steps 1-6), then switch to Free Exploration mode for ascent (Steps 7-9) since ascent steps only exist in Free Exploration.
- Q: Do the ascent steps currently exist in the Step-by-Step workflow UI? → A: No, ascent steps exist only in Free Exploration mode. Test protocol is hybrid: Step-by-Step for descent, Free Exploration for ascent.
- Q: Does session state persist when switching from Step-by-Step to Free Exploration mode? → A: No, session state does NOT persist automatically. Must save session first (💾 Save), then switch to Free Exploration and load the saved session.

### Session 2025-12-10

- Q: How should quartile boundaries be determined for L2 score categorization (Schools ascent)? → A: Data-driven percentiles (25th/50th/75th from actual score distribution).
- Q: What is the source for the location column at L2 (Schools ascent)? → A: Use existing "Commune" column from L3 (city name already available from descent join).
- Q: How should L3 demographic linkage work for Schools ascent? → A: Use only existing columns; expose postal code/commune as linkage key for future demographic joins, but no external data is added during ascent.
- Q: Which score column for L1 vector in Schools ascent? → A: "Taux de réussite G" (global success rate per school) - all 410 individual school scores as one ungrouped vector.
- Q: How should quartile categories be labeled at L2? → A: Descriptive performance labels: top_performers / above_average / below_average / needs_improvement.

## Assumptions

- The Streamlit app is running on localhost:8501 before test execution begins
- Test data files are available in the test_data/test0 and test_data/test1 directories
- Playwright MCP server is properly configured and accessible
- The interface wizard/discovery flow follows the same structure as when reference exports were created
- Network latency and file processing times are within reasonable bounds (< 60 seconds per step)
