# Playwright MCP Call Sequence Contract

**Feature**: 006-playwright-mcp-e2e
**Date**: 2025-12-09
**Source of Truth**: `/intuitiveness/streamlit_app.py`

This document defines the exact sequence of Playwright MCP tool calls for executing a complete descent/ascent test cycle, based on the actual Streamlit app UI.

---

## UI Overview: Guided Workflow Steps

The Streamlit app uses a **Guided Workflow** with these steps (from `streamlit_app.py` lines 243-280):

| Step | Name | Level Transition | Key Button |
|------|------|------------------|------------|
| 0 | Upload Data | → L4 | "Browse files" |
| 1 | Define Items | L4 → L3 | "🔨 Build Connected Information" |
| 2 | Define Categories | L3 → L2 | "🔄 Categorize Data" |
| 3 | Select Values | L2 → L1 | "Extract Values" |
| 4 | Choose Computation | L1 → L0 | "Compute Metrics" |
| 5 | Results | L0 display | Export buttons |

---

## Pre-requisites

1. Streamlit app running: `streamlit run intuitiveness/streamlit_app.py --server.port 8501`
2. Test data files available in `test_data/test0/` or `test_data/test1/`
3. Playwright MCP server connected

---

## Sequence: Schools Dataset (test0)

### Step 1: Navigate to App

```yaml
tool: browser_navigate
params:
  url: "http://localhost:8501"
expect:
  - Page loads with Streamlit app
  - Title shows "Intuitiveness" or similar
```

### Step 2: Wait for App Ready

```yaml
tool: browser_wait_for
params:
  text: "Browse files"
expect:
  - File upload widget visible
```

### Step 3: Take Initial Screenshot

```yaml
tool: browser_take_screenshot
params:
  filename: "01_initial_state.png"
```

### Step 4: Click Browse Files

```yaml
tool: browser_click
params:
  element: "Browse files button"
  ref: "button:has-text('Browse files')"
expect:
  - File chooser dialog opens
```

### Step 5: Upload Files (L4 Entry)

```yaml
tool: browser_file_upload
params:
  paths:
    - "/Users/arthursarazin/Documents/data_redesign_method/test_data/test0/fr-en-college-effectifs-niveau-sexe-lv.csv"
    - "/Users/arthursarazin/Documents/data_redesign_method/test_data/test0/fr-en-indicateurs-valeur-ajoutee-colleges.csv"
expect:
  - Files accepted
  - File list shows 2 files with row counts
```

### Step 6: Take L4 Screenshot

```yaml
tool: browser_take_screenshot
params:
  filename: "02_l4_files_uploaded.png"
```

### Step 7: Click Next to Define Items (Step 1)

```yaml
tool: browser_click
params:
  element: "Next button"
  ref: "button:has-text('Next')"
  # OR "button:has-text('→')"
expect:
  - "Define Items" page loads
  - Entity definition options visible
```

### Step 8: Wait for Entity Definition Page

```yaml
tool: browser_wait_for
params:
  text: "Define Items"
  # OR "AI-Assisted" or "Manual"
```

### Step 9: Configure AI-Assisted Entity Discovery

```yaml
# Enter context/description for AI discovery
tool: browser_type
params:
  element: "Context textarea"
  ref: "[data-testid='stTextArea'] textarea"
  text: "Middle schools in France with student counts and performance scores"

# Click Generate Structure
tool: browser_click
params:
  element: "Generate Structure button"
  ref: "button:has-text('Generate Structure')"

tool: browser_wait_for
params:
  textGone: "Analyzing"
  time: 30
```

### Step 10: Configure Semantic Join (L4→L3)

```yaml
# Select connection mode
tool: browser_click
params:
  element: "Smart Match radio option"
  ref: "input[value='🧠 Smart Match']"
  # OR click on text "🧠 Smart Match (AI)"

# Wait for match configuration
tool: browser_wait_for
params:
  text: "threshold"
```

### Step 11: Build Connected Information

```yaml
tool: browser_click
params:
  element: "Build Connected Information button"
  ref: "button:has-text('Build Connected Information')"

# Wait for semantic join to complete (may take 30-60 seconds)
tool: browser_wait_for
params:
  time: 60
  textGone: "Building"
```

### Step 12: Take L3 Screenshot

```yaml
tool: browser_take_screenshot
params:
  filename: "03_l3_joined.png"
```

### Step 13: Verify L3 State

```yaml
tool: browser_snapshot
expect:
  - Shows connected items count
  - Graph or table visualization visible
```

### Step 14: Click Continue to Define Categories (Step 2)

```yaml
tool: browser_click
params:
  element: "Continue button"
  ref: "button:has-text('Continue')"
  # OR "button:has-text('→')"
```

### Step 15: Wait for Categories Page

```yaml
tool: browser_wait_for
params:
  text: "Define Categories"
  # OR "Browse Your Connected Information"
```

### Step 16: Select Column for Categorization

```yaml
tool: browser_select_option
params:
  element: "Column selector dropdown"
  ref: "[data-testid='stSelectbox']"
  values: ["nombre_eleves_total"]
```

### Step 17: Enter Categories

```yaml
tool: browser_type
params:
  element: "Categories input"
  ref: "[data-testid='stTextInput'] input"
  text: "downtown, countryside"
```

### Step 18: Enable Smart Matching

```yaml
tool: browser_click
params:
  element: "Smart matching checkbox"
  ref: "[data-testid='stCheckbox']"
  # Check if not already checked
```

### Step 19: Apply Categorization

```yaml
tool: browser_click
params:
  element: "Categorize Data button"
  ref: "button:has-text('Categorize Data')"

tool: browser_wait_for
params:
  text: "downtown"
  # Wait for categories to appear in results
```

### Step 20: Take L2 Screenshot

```yaml
tool: browser_take_screenshot
params:
  filename: "04_l2_categorized.png"
```

### Step 21: Click Continue to Select Values (Step 3)

```yaml
tool: browser_click
params:
  element: "Continue button"
  ref: "button:has-text('Continue')"
```

### Step 22: Wait for Values Page

```yaml
tool: browser_wait_for
params:
  text: "Select Values"
  # OR "Select Column to Extract"
```

### Step 23: Select Column to Extract

```yaml
tool: browser_select_option
params:
  element: "Column selector dropdown"
  ref: "[data-testid='stSelectbox']"
  values: ["Taux de reussite G"]
```

### Step 24: Extract Values

```yaml
tool: browser_click
params:
  element: "Extract Values button"
  ref: "button:has-text('Extract Values')"

tool: browser_wait_for
params:
  text: "values"
  # Wait for L1 vector to display
```

### Step 25: Take L1 Screenshot

```yaml
tool: browser_take_screenshot
params:
  filename: "05_l1_vector.png"
```

### Step 26: Click Continue to Choose Computation (Step 4)

```yaml
tool: browser_click
params:
  element: "Continue button"
  ref: "button:has-text('Continue')"
```

### Step 27: Wait for Aggregation Page

```yaml
tool: browser_wait_for
params:
  text: "Choose Computation"
  # OR "Select calculation method"
```

### Step 28: Select Aggregation Method

```yaml
tool: browser_select_option
params:
  element: "Aggregation method selector"
  ref: "[data-testid='stSelectbox']"
  values: ["mean"]
```

### Step 29: Compute Metrics

```yaml
tool: browser_click
params:
  element: "Compute Metrics button"
  ref: "button:has-text('Compute Metrics')"

tool: browser_wait_for
params:
  text: "88"
  # Wait for L0 datum to display
```

### Step 30: Take L0 Screenshot

```yaml
tool: browser_take_screenshot
params:
  filename: "06_l0_datum.png"
```

### Step 31: Verify L0 Value

```yaml
tool: browser_snapshot
expect:
  - Shows value approximately 88.25
  - Shows "mean" aggregation type
```

### Step 32: Click View Results (Step 5)

```yaml
tool: browser_click
params:
  element: "View Results button"
  ref: "button:has-text('View Results')"
```

### Step 33: Take Results Screenshot

```yaml
tool: browser_take_screenshot
params:
  filename: "07_results.png"
```

### Step 34: Verify Results Page

```yaml
tool: browser_snapshot
expect:
  - Shows "Descent complete!" message
  - Shows tabs: Final Results, Structure, Connected View, Export
  - Shows summary metrics
```

---

## Ascent Phase (Free Navigation Mode)

**Note**: Ascent requires switching to Free Navigation Mode. This is accessed via a mode toggle or by starting a new session with Free Navigation enabled.

### Step 35: Switch to Free Navigation (if needed)

```yaml
# Look for Free Navigation toggle or option
tool: browser_snapshot
# Identify the navigation mode selector

tool: browser_click
params:
  element: "Free Navigation mode"
  ref: "button:has-text('Free Navigation')"
  # OR sidebar option
```

### Step 36-40: Ascent Steps (L0→L1→L2→L3)

The ascent follows similar patterns using:
- "Ascend" or "Build up" buttons
- Dimension configuration forms
- "Apply" buttons

---

## Error Handling

For each step, if an error occurs:

1. Take screenshot with suffix `_error.png`
2. Get browser_snapshot for debugging
3. Log error details
4. Attempt retry (max 3 times)
5. If retry fails, report failure with last known state

---

## Validation Criteria

| Step | Verification |
|------|--------------|
| L4 Entry | 2 files shown, row counts visible |
| L3 Join | Connected items count displayed |
| L2 Categorize | Category names and counts shown |
| L1 Extract | Vector values displayed |
| L0 Aggregate | Value ~88.25 for schools, ~69M for ADEME |
| Results | Export options available |

---

## Key UI Elements (from streamlit_app.py)

### Buttons
- "Browse files" - File upload trigger
- "Next →" / "Continue →" - Step navigation
- "← Back" - Previous step
- "Generate Structure" - AI entity discovery
- "🔨 Build Connected Information" - Execute L4→L3 join
- "🔄 Categorize Data" - Execute L3→L2 categorization
- "Extract Values" - Execute L2→L1 extraction
- "Compute Metrics" - Execute L1→L0 aggregation
- "View Results →" - Go to results page
- "🔄 Start New Analysis" - Reset workflow

### Streamlit Widgets
- `[data-testid="stFileUploader"]` - File upload area
- `[data-testid="stSelectbox"]` - Dropdown selectors
- `[data-testid="stTextInput"]` - Text input fields
- `[data-testid="stTextArea"]` - Multiline text input
- `[data-testid="stCheckbox"]` - Checkbox options
- `[data-testid="stSlider"]` - Slider controls
- `[data-testid="stSpinner"]` - Loading indicator (wait until hidden)
