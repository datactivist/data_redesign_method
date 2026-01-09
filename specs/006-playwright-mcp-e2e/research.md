# Research: Playwright MCP E2E Testing

**Feature**: 006-playwright-mcp-e2e
**Date**: 2025-12-09
**Status**: Complete

## Research Questions

### Q1: How to use Playwright MCP for visual test monitoring?

**Decision**: Execute Playwright MCP tools directly in conversation flow

**Rationale**:
- User requirement: "I WANT THIS TEST TO BE RUN THROUGH PLAYWRIGHT MCP SO I CAN SEE AND MONITOR TESTING"
- MCP tools execute in user's browser session, enabling real-time visual observation
- Each tool call shows in conversation, providing step-by-step progress

**Available Tools**:
| Tool | Purpose | Usage |
|------|---------|-------|
| `browser_navigate` | Navigate to URL | Start test at localhost:8501 |
| `browser_file_upload` | Upload files | Upload CSV test data |
| `browser_click` | Click elements | Interact with buttons, options |
| `browser_snapshot` | Get accessibility tree | Verify page state |
| `browser_take_screenshot` | Capture PNG | Document visual state |
| `browser_wait_for` | Wait for conditions | Handle async operations |
| `browser_type` | Enter text | Fill input fields |
| `browser_select_option` | Select dropdown | Choose options |

**Alternatives Rejected**:
- pytest-playwright: Runs headless, no visual monitoring
- Manual testing: Not reproducible, not documented

---

### Q2: What is the correct Streamlit element selector strategy?

**Decision**: Use text-based selectors with Streamlit-specific testids as fallback

**Rationale**:
- Streamlit generates semi-stable `data-testid` attributes
- Text content (button labels, headers) is stable and domain-meaningful
- Combining both provides resilience to minor UI changes

**Selector Patterns for Intuitiveness App** (based on actual streamlit_app.py):

```text
# File Upload (Step 0)
[data-testid="stFileUploader"]
button:has-text("Browse files")

# Navigation Buttons
button:has-text("Next")           # Move to next step
button:has-text("Continue")       # Alternative next
button:has-text("Back")           # Previous step
button:has-text("View Results")   # Go to results

# Step 1 - Define Items (L4→L3)
button:has-text("Generate Structure")        # AI entity discovery
button:has-text("Build Connected Information")  # Execute semantic join
# Note: Contains emoji 🔨

# Step 2 - Define Categories (L3→L2)
button:has-text("Categorize Data")  # Execute categorization
# Note: Contains emoji 🔄

# Step 3 - Select Values (L2→L1)
button:has-text("Extract Values")   # Execute extraction

# Step 4 - Choose Computation (L1→L0)
button:has-text("Compute Metrics")  # Execute aggregation

# Step 5 - Results
button:has-text("Start New Analysis")  # Reset workflow
# Note: Contains emoji 🔄

# Streamlit Widgets
[data-testid="stSelectbox"]       # Dropdown selectors
[data-testid="stTextInput"]       # Single-line text input
[data-testid="stTextArea"]        # Multi-line text input
[data-testid="stSlider"]          # Slider controls
[data-testid="stCheckbox"]        # Checkbox options
[data-testid="stRadio"]           # Radio button groups

# Loading State
[data-testid="stSpinner"]         # Wait until hidden

# Data Display
[data-testid="stMetric"]          # L0 datum display
[data-testid="stDataFrame"]       # Table displays
[data-testid="stExpanderHeader"]  # Expandable sections

# Tabs
[data-testid="stTab"]             # Tab headers
```

**Source**: Analyzed `intuitiveness/streamlit_app.py` render_*_step() functions (lines 484-1000).

---

### Q3: What are the exact expected values from session exports?

**Decision**: Use exact values from reference exports as test assertions

**Schools Dataset (test0)**:
| Level | Metric | Expected Value |
|-------|--------|----------------|
| L4 | Source files | 2 (50164 + 20053 rows) |
| L3 | Joined table | 410 rows, 111 columns |
| L2 | Categories | downtown: 281, countryside: 129 |
| L1 | Vector | 410 values (Taux de reussite G) |
| L0 | Datum | 88.25365853658536 (MEAN) |
| Ascent L2 | Categories | above_median: 208, below_median: 202 |
| Ascent L3 | Enriched | 410 rows, 112 columns |

**ADEME Dataset (test1)**:
| Level | Metric | Expected Value |
|-------|--------|----------------|
| L4 | Source files | 2 (428 + 37339 rows) |
| L3 | Joined table | 500 rows, 47 columns |
| L2 | Categories | single_funding: 412, multiple_funding: 88 |
| L1 | Vector | 450 values (montant grouped by recipient) |
| L0 | Datum | 69586180.93 (SUM) |
| Ascent L2 | Categories | above_10k: 301, below_10k: 149 |
| Ascent L3 | Enriched | 450 rows, 48 columns |

**Source**: Parsed from session export JSON files in `tests/artifacts/20251208_domain_specific_v2/`

---

### Q4: What is the UI navigation flow for descent/ascent?

**Decision**: Follow Guided Workflow Mode for descent, Free Navigation Mode for ascent

**Source of Truth**: `intuitiveness/streamlit_app.py` (lines 243-280 for STEPS, 484-1000 for step renderers)

**Guided Workflow Steps (Descent L4→L0)**:

| Step | Function | Level Transition | Key UI Elements |
|------|----------|------------------|-----------------|
| 0 | `render_upload_step()` | → L4 | File uploader, "Browse files" button |
| 1 | `render_entities_step()` | L4 → L3 | Manual/AI method, "Generate Structure", "🔨 Build Connected Information" |
| 2 | `render_domains_step()` | L3 → L2 | Column selector, categories input, "🔄 Categorize Data" |
| 3 | `render_features_step()` | L2 → L1 | Column selector, "Extract Values" |
| 4 | `render_aggregation_step()` | L1 → L0 | Aggregation dropdown (count/sum/mean/min/max), "Compute Metrics" |
| 5 | `render_results_step()` | L0 display | Tabs: Final Results, Structure, Connected View, Export |

**Detailed Descent Flow**:

1. **Step 0 - Upload Data (→ L4)**:
   - Click "Browse files" button to open file chooser
   - Upload CSV files (multiple selection supported)
   - System shows files with row/column counts
   - Click "Next →" to proceed

2. **Step 1 - Define Items (L4 → L3)**:
   - Choose method: "Manual (specify items)" or "AI-Assisted"
   - For AI-Assisted: Enter context description, click "Generate Structure"
   - Select connection mode: "🔑 Exact Match" or "🧠 Smart Match (AI)"
   - Configure semantic join threshold (default 0.85)
   - Click "🔨 Build Connected Information" to execute join
   - Click "Continue →" after L3 is created

3. **Step 2 - Define Categories (L3 → L2)**:
   - Browse connected data in tabs (All, Items, Connections)
   - Select column for categorization via dropdown
   - Enter categories as comma-separated text (e.g., "downtown, countryside")
   - Optionally enable "Use smart matching (AI)" checkbox
   - Adjust "Matching strictness" slider (0.1-0.9)
   - Click "🔄 Categorize Data" to execute
   - Click "Continue →" after L2 is created

4. **Step 3 - Select Values (L2 → L1)**:
   - View L2 domain tables
   - Select column to extract via dropdown
   - Click "Extract Values" to create L1 vector
   - Click "Continue →" after L1 is created

5. **Step 4 - Choose Computation (L1 → L0)**:
   - View L1 vectors per domain
   - Select aggregation method: count, sum, mean, min, max
   - Click "Compute Metrics" to create L0 datum
   - Click "View Results →" to see final results

6. **Step 5 - Results (L0 Display)**:
   - View "Descent complete!" summary
   - Explore tabs: Final Results, Structure, Connected View, Export
   - Export session via Export tab
   - Click "🔄 Start New Analysis" to reset

**Ascent Flow (Free Navigation Mode)**:

The ascent phase (L0→L1→L2→L3) is handled via **Free Navigation Mode** (`render_free_navigation_main()` at lines 1861-2260):

- Switch to Free Navigation via mode selector
- Use navigation tree in sidebar
- Each level offers "Ascend" / "Build up" options
- Configure alternative dimensions during ascent
- Tree-based time-travel allows exploring branches

**Source**: Direct analysis of `streamlit_app.py` render functions.

---

### Q5: How does the intuitiveness package support the cycle?

**Decision**: Tests interact via UI only; package internals are transparent to user

**Key Package Components Used by Streamlit App**:

| Component | Purpose | UI Trigger |
|-----------|---------|------------|
| `Level4Dataset` | Holds raw uploaded files | File upload widget |
| `semantic_table_join()` | L4→L3 via embeddings | "Create Join" button |
| `Level3Dataset` | Graph with linked entities | Join completion |
| `Level2Dataset` | Categorized table | "Apply Categorization" |
| `Level1Dataset` | Vector of values | "Extract Values" |
| `Level0Dataset` | Single aggregated datum | "Compute Metrics" |
| `NavigationSession` | Tracks descent/ascent path | All navigation |
| `ascent.operations` | L0→L1→L2→L3 reconstruction | Ascend buttons |

**Embedding Model**: `intfloat/multilingual-e5-small` (see `descent/semantic_join.py:37`)

**Constitution Compliance**: Tests verify user-facing behavior without requiring users to understand these internals.

---

### Q6: Why did single-column semantic matching fail? (2025-12-09)

**Problem Discovered**: L4→L3 semantic join produced 731,142 rows instead of expected ~410 rows.

**Root Cause Analysis**:
The original test used single-column matching:
- Left column: "Dénomination principale" (64 unique values like "COLLEGE", "LYCEE")
- Right column: "Nom de l'établissement" (4,122 unique school names)

The 64 unique values were **generic type names**, not unique identifiers:
| Value | Count | Problem |
|-------|-------|---------|
| "COLLEGE" | 32,214 | Matches to ANY school with "COLLEGE" in name |
| "LYCEE GENERAL" | 8,450 | Matches to ANY lycée |
| ... | ... | ... |

When semantic matching assigned `semantic_id` based on these generic values:
- 32,214 rows labeled "COLLEGE" in left file
- N schools containing "COLLEGE" in right file
- Result: 32,214 × N = **cartesian product explosion**

**Decision**: Use multi-column row vectorization

**Correct Approach**:
The L4→L3 semantic join must use **multiple columns** to create unique row identities:

```
User selects columns:
  File 1: ["Appellation officielle", "Commune"]
  File 2: ["Nom de l'établissement", "Commune"]

Each ROW becomes a vector:
  Row 1 File 1: vector = ("COLLEGE JEAN MOULIN", "PARIS")
  Row 1 File 2: vector = ("COLLEGE JEAN MOULIN PARIS", "PARIS")

Semantic matching compares row vectors:
  similarity(row1_file1, row1_file2) = 0.95 → MATCH
```

**Column Selection Strategy**:

| Dataset | File 1 Columns | File 2 Columns | Unique Identity |
|---------|---------------|----------------|-----------------|
| Schools | "Appellation officielle" + "Commune" | "Nom de l'établissement" + "Commune" | School name + location |
| ADEME | "dispositifAide" + "nomBeneficiaire" | "type_aides_financieres" + "nom_programme" | Funding + recipient |

**Expected Results After Fix**:
- Schools: ~410 matched rows (NOT 731,142)
- ADEME: ~500 matched rows

**UI Impact**:
Tests must click **multiple columns** in the join wizard per file, not just one column.

**Source**: User clarification during test execution (2025-12-09)

---

## Conclusions

All research questions resolved. Ready for Phase 1 design.

**Key Findings**:
1. Playwright MCP provides all necessary tools for visual E2E testing
2. Streamlit element selectors follow predictable patterns
3. Reference session exports provide exact expected values for assertions
4. UI flow is well-defined through Free Navigation Mode
5. Technical complexity is properly hidden from user-facing tests
6. **L4→L3 semantic join requires multi-column row vectorization** - single generic columns cause join explosion
