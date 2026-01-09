# Quickstart: Playwright MCP E2E Testing

**Feature**: 006-playwright-mcp-e2e
**Date**: 2025-12-09
**Source of Truth**: `/intuitiveness/streamlit_app.py`

## Overview

This guide explains how to run visual E2E tests for the descent/ascent cycle using Playwright MCP tools. The tests execute in real-time, allowing you to visually monitor each step in the browser.

## Prerequisites

1. **Streamlit App Running**
   ```bash
   source myenv311/bin/activate
   streamlit run intuitiveness/streamlit_app.py --server.port 8501
   ```

2. **Playwright MCP Server Connected**
   - Ensure your Claude environment has access to Playwright MCP tools
   - The browser will be controlled via MCP protocol

3. **Test Data Available**
   - `test_data/test0/` - Schools dataset (2 CSV files)
   - `test_data/test1/` - ADEME dataset (2 CSV files)

## Guided Workflow Steps

The Streamlit app uses a **6-step Guided Workflow** for the descent phase:

| Step | Name | Action | Key Button |
|------|------|--------|------------|
| 0 | Upload Data | Upload CSV files | "Browse files" |
| 1 | Define Items | Configure semantic join | "🔨 Build Connected Information" |
| 2 | Define Categories | Apply categorization | "🔄 Categorize Data" |
| 3 | Select Values | Extract column | "Extract Values" |
| 4 | Choose Computation | Select aggregation | "Compute Metrics" |
| 5 | Results | View & export | Export buttons |

## Running Tests

### Option 1: Interactive Execution (Recommended)

Simply ask Claude to execute the test:

```text
"Run the schools dataset E2E test using Playwright MCP"
```

or

```text
"Execute the full descent/ascent cycle for ADEME data"
```

Claude will:
1. Navigate to the Streamlit app
2. Upload the test files
3. Execute each step of the cycle
4. Take screenshots at each level
5. Report pass/fail status with evidence

### Option 2: Step-by-Step Execution

You can also run individual steps:

```text
# Step 0: Upload Data
"Navigate to the Streamlit app at localhost:8501"
"Click Browse files and upload the schools CSV files"

# Step 1: Define Items (L4→L3)
"Click Next to go to Define Items step"
"Enter context for AI discovery, click Generate Structure"
"Select Smart Match mode and click Build Connected Information"

# Step 2: Define Categories (L3→L2)
"Click Continue to go to Define Categories"
"Select nombre_eleves_total column, enter 'downtown, countryside' as categories"
"Click Categorize Data"

# Step 3: Select Values (L2→L1)
"Click Continue to Select Values"
"Select 'Taux de reussite G' column and click Extract Values"

# Step 4: Choose Computation (L1→L0)
"Click Continue to Choose Computation"
"Select 'mean' aggregation and click Compute Metrics"

# Step 5: View Results
"Click View Results to see the final output"
```

## Expected Results

### Schools Dataset (test0)

| Step | UI Action | Expected Output |
|------|-----------|-----------------|
| Step 0 | Upload files | 2 files: 50,164 + 20,053 rows |
| Step 1 | Build Connected Information | ~410 connected items |
| Step 2 | Categorize Data | downtown: ~281, countryside: ~129 |
| Step 3 | Extract Values | 410 values (Taux de reussite G) |
| Step 4 | Compute Metrics | L0 datum: **88.25** (mean) |
| Step 5 | View Results | Summary with export options |

### ADEME Dataset (test1)

| Step | UI Action | Expected Output |
|------|-----------|-----------------|
| Step 0 | Upload files | 2 files: 428 + 37,339 rows |
| Step 1 | Build Connected Information | ~500 connected items |
| Step 2 | Categorize Data | single_funding: ~412, multiple_funding: ~88 |
| Step 3 | Extract Values | ~450 values (montant) |
| Step 4 | Compute Metrics | L0 datum: **69,586,180.93** (sum) |
| Step 5 | View Results | Summary with export options |

## Screenshots

Screenshots are saved with naming convention:
```
{step_number:02d}_{description}.png
```

Example:
- `01_initial_state.png` - App loaded
- `02_l4_files_uploaded.png` - After file upload
- `03_l3_joined.png` - After semantic join
- `04_l2_categorized.png` - After categorization
- `05_l1_vector.png` - After extraction
- `06_l0_datum.png` - After aggregation
- `07_results.png` - Final results page

## Key UI Elements

### Buttons to Click

| Button Text | When to Click | Effect |
|-------------|---------------|--------|
| "Browse files" | Step 0 | Opens file chooser |
| "Next →" / "Continue →" | After each step | Proceeds to next step |
| "← Back" | Any step | Returns to previous step |
| "Generate Structure" | Step 1 (AI mode) | Generates data model |
| "🔨 Build Connected Information" | Step 1 | Executes semantic join |
| "🔄 Categorize Data" | Step 2 | Applies categorization |
| "Extract Values" | Step 3 | Extracts column vector |
| "Compute Metrics" | Step 4 | Computes aggregation |
| "View Results →" | Step 4 | Goes to results page |

### Form Elements

| Element | Selector | Purpose |
|---------|----------|---------|
| File uploader | `[data-testid="stFileUploader"]` | Upload CSV files |
| Dropdown | `[data-testid="stSelectbox"]` | Select columns/methods |
| Text input | `[data-testid="stTextInput"]` | Enter categories |
| Checkbox | `[data-testid="stCheckbox"]` | Enable smart matching |
| Slider | `[data-testid="stSlider"]` | Adjust threshold |

## Troubleshooting

### App Not Loading

```text
"Check if Streamlit is running at localhost:8501"
"Take a screenshot of the current browser state"
```

### File Upload Fails

```text
"Get browser snapshot to see current state"
"Click the Browse files button first, then upload"
```

### Semantic Join Takes Too Long

The semantic join uses embeddings and may take 30-60 seconds for large files.

```text
"Wait 60 seconds for the join to complete"
```

### Element Not Found

If a button or element can't be found:

```text
"Get browser snapshot to see available elements"
"Try scrolling down the page"
```

### Wrong Button Name

The app uses emojis in button names. If text-based selection fails:

```text
"Get browser snapshot to see exact button text"
```

## Success Criteria

Test passes if:

1. All 6 guided workflow steps complete without errors
2. L0 datum value within 0.01 of expected (88.25 for schools, 69586180.93 for ADEME)
3. Category distributions approximately match expected counts
4. All screenshots captured successfully
5. Export functionality available in Results step

## Reference Files

- Spec: `specs/006-playwright-mcp-e2e/spec.md`
- Data Model: `specs/006-playwright-mcp-e2e/data-model.md`
- MCP Sequence: `specs/006-playwright-mcp-e2e/contracts/playwright-mcp-sequence.md`
- Research: `specs/006-playwright-mcp-e2e/research.md`
- Session Exports: `tests/artifacts/20251208_domain_specific_v2/`
- **Source of Truth**: `intuitiveness/streamlit_app.py`
