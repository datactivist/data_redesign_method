# Implementation Plan: Playwright MCP E2E Testing + Revised Schools Ascent

**Branch**: `006-playwright-mcp-e2e` | **Date**: 2025-12-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/006-playwright-mcp-e2e/spec.md`

## Summary

Develop E2E tests using Playwright MCP for visual monitoring of complete descent/ascent cycles through the Streamlit interface. The plan includes a **revised Schools ascent design** with quartile-based categorization, location data, and demographic linkage keys.

**Key Changes (Session 2025-12-10)**:
- Schools L1: All 410 individual scores as ungrouped vector (not grouped by Secteur)
- Schools L2: Quartile categories (top_performers/above_average/below_average/needs_improvement) + Commune location
- Schools L3: Enriched table with postal code as linkage key for future demographic joins

**Key Changes (Session 2025-12-11 - Spec Update)**:
- **12-Step Workflow**: Both user stories now specify 12 acceptance scenarios (Steps 0-5 descent, Step 7 save, Step 8 mode switch, Steps 9-12 ascent)
- **Session Persistence Required**: Must save session before switching to Free Exploration mode (session state does NOT persist automatically)
- **Expected Outputs Clarified**: Schools should produce dataset measuring middle school influence on "Taux de réussite"; ADEME should show where funding was spent most
- **Hybrid Test Protocol**: Step-by-Step mode for descent (Steps 0-5), Free Exploration mode for ascent (Steps 9-12)

## Technical Context

**Language/Version**: Python 3.11 (existing `myenv311` virtual environment)
**Primary Dependencies**:
- Playwright MCP (browser automation via Model Context Protocol)
- Streamlit (existing `intuitiveness/streamlit_app.py`)
- NetworkX (graph-based session persistence) - ALREADY IN PROJECT
- HuggingFace `sentence-transformers` (semantic matching)
- pandas, numpy (data manipulation)
**Storage**: JSON files (session graphs) + CSV files (test data)
**Testing**: Playwright MCP tools executed in conversation (NOT pytest)
**Target Platform**: macOS (localhost:8501)
**Project Type**: Single project - test suite + session persistence + ascent UI enhancement
**Performance Goals**: Complete descent/ascent cycle in <5 minutes per dataset
**Constraints**: Must match reference exports within ±0.01 tolerance (L0 values)
**Scale/Scope**: 2 test datasets (schools: 70k rows combined, ADEME: 38k rows combined); ~1MB per session graph file

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Intuitiveness Through Abstraction Levels** | PASS | Tests verify all 5 levels (L4→L0→L3) with proper navigation |
| **II. Descent-Ascent Cycle** | PASS | Tests execute full descent then ascent with alternative dimensions (quartiles for Schools, funding_size for ADEME) |
| **III. Complexity Quantification** | PASS | Row counts at each level are validated against reference |
| **IV. Human-Data Interaction Granularity** | PASS | Tests verify L0 ground truth anchors the entire cycle |
| **V. Design for Diverse Data Publics** | PASS | Tests use domain terms (schools, funding, scores, quartiles), not technical terms |
| **Target User Assumption** | PASS | Test steps mirror non-technical user workflows (upload → click → verify) |

**Quality Gates:**
- Each level transition preserves data integrity (validated by row counts)
- Ascent dimensions are user-need driven (quartile performance for schools, funding threshold for ADEME)
- Final dataset is independently testable (column count verification)
- User-facing terms: "Categories", "Values", "Computation" (not tables/vectors)

## Project Structure

### Documentation (this feature)

```text
specs/006-playwright-mcp-e2e/
├── plan.md              # This file
├── spec.md              # Feature specification (updated with clarifications)
├── research.md          # Phase 0 output - multi-column vectorization research
├── data-model.md        # Phase 1 output - SessionGraph entity definition
├── quickstart.md        # Phase 1 output - Test execution guide
├── contracts/           # Phase 1 output - Expected I/O contracts
└── tasks.md             # Phase 2 output - Task breakdown
```

### Source Code (repository root)

```text
intuitiveness/
├── persistence/
│   ├── session_graph.py     # SessionGraph class (NetworkX DiGraph)
│   ├── serializers.py       # Graph serialization utilities
│   └── session_store.py     # localStorage fallback
├── navigation.py            # NavigationSession with save_graph/load_graph
└── streamlit_app.py         # UI with ascent handlers

tests/
├── e2e/
│   └── playwright/
│       └── mcp/
│           ├── __init__.py
│           ├── helpers.py
│           ├── test_schools_mcp_cycle.py
│           └── test_ademe_mcp_cycle.py
└── artifacts/
    ├── screenshots/
    │   ├── schools_mcp_cycle/
    │   └── ademe_mcp_cycle/
    └── 20251208_domain_specific_v2/
        ├── test0_schools/    # Reference: schools session export
        └── test1_ademe/      # Reference: ADEME session export

sessions/                     # Session graph storage
└── session_graph_*.json      # Serialized NetworkX DiGraphs
```

**Structure Decision**: Single project with session persistence enhancement. Tests execute via Playwright MCP in conversation, not pytest runner.

---

## Revised Schools Ascent Design (Session 2025-12-10)

Based on clarifications, the Schools ascent phase has been redesigned:

### New Ascent Flow (Schools - test0)

| Step | Level | Action | Output |
|------|-------|--------|--------|
| **Descent (unchanged)** ||||
| 1-6 | L4→L0 | Standard descent | L0 = 88.25 (mean score) |
| **Ascent (revised)** ||||
| 7 | L0→L1 | Source recovery | 410 individual "Taux de réussite G" scores (ungrouped) |
| 8 | L1→L2 | Quartile categorization | 4 categories: top_performers / above_average / below_average / needs_improvement |
| 8b | L1→L2 | Add location | "Commune" column from existing L3 data |
| 9 | L2→L3 | Enrichment | Expose postal code/commune as linkage key |

### Quartile Boundaries

- **Method**: Data-driven percentiles (25th/50th/75th from actual score distribution)
- **Categories**:
  - `top_performers`: scores >= 75th percentile
  - `above_average`: 50th <= scores < 75th percentile
  - `below_average`: 25th <= scores < 50th percentile
  - `needs_improvement`: scores < 25th percentile

### Data Constraints

- **L3 Enrichment**: Uses ONLY existing columns from descent join
- **No external data**: Postal code is exposed as linkage key but no demographic data is added
- **Score column**: "Taux de réussite G" (global success rate)
- **Location column**: "Commune" (already in L3 from descent)

---

## Complete 12-Step Workflow (from spec.md)

### Schools Dataset (US1) - Full Cycle

| Step | Mode | Action | Level | Expected Output |
|------|------|--------|-------|-----------------|
| 0 | Guided | Upload 2 CSV files | → L4 | 50,164 + 20,053 rows shown |
| 1 | Guided | Semantic join (UAI columns) | L4→L3 | Linked table created |
| 2 | Guided | Categorize by "Secteur" | L3→L2 | PRIVE/PUBLIC categories |
| 3 | Guided | Extract "Taux de reussite G" | L2→L1 | L1 vector created |
| 4 | Guided | Compute MEAN | L1→L0 | L0 datum per category |
| 5 | Guided | View Results | L0 | Final results tabs |
| 7 | Guided | Save Session | - | JSON saved to sessions/ |
| 8 | Switch | Load in Free Exploration | - | L0 restored with nav tree |
| 9 | Free | Ascend to L1 | L0→L1 | Grouped scores recovered |
| 10 | Free | Apply score_quartile dimension | L1→L2 | 4 performance categories + Commune |
| 11 | Free | Enrich to L3 | L2→L3 | All columns + linkage keys exposed |
| 12 | Free | Export session | - | Artifacts match reference |

### ADEME Dataset (US2) - Full Cycle

| Step | Mode | Action | Level | Expected Output |
|------|------|--------|-------|-----------------|
| 0 | Guided | Upload 2 CSV files | → L4 | 428 + 37,339 rows shown |
| 1 | Guided | Semantic join (dispositifAide/type_aides) | L4→L3 | Linked table created |
| 2 | Guided | Categorize by funding type | L3→L2 | HABITAT/ENERGIE categories |
| 3 | Guided | Extract "montant" | L2→L1 | L1 vector created |
| 4 | Guided | Compute SUM | L1→L0 | L0 datum ~69.5M or ~1.14B |
| 5 | Guided | View Results | L0 | Final results tabs |
| 7 | Guided | Save Session | - | JSON saved to sessions/ |
| 8 | Switch | Load in Free Exploration | - | L0 restored with nav tree |
| 9 | Free | Ascend to L1 | L0→L1 | ~450 funding amounts |
| 10 | Free | Apply funding_size dimension | L1→L2 | above_10k / below_10k split |
| 11 | Free | Enrich to L3 | L2→L3 | 48 columns with "objet" linked |
| 12 | Free | Export session | - | Artifacts match reference |

**Note**: Step 6 is skipped in numbering (spec uses Steps 0-5, then 7-12).

---

## Implementation Tasks Overview

### Phase 1: Quartile Dimension Handler

Modify `render_loaded_graph_view()` in `streamlit_app.py`:
- Add "score_quartile" dimension option for Schools dataset
- Compute percentile boundaries from L1 data
- Apply descriptive labels (not Q1/Q2/Q3/Q4)
- Include Commune column in L2 output

### Phase 2: L3 Linkage Key Exposure

Modify L2→L3 enrichment handler:
- Preserve postal code/commune columns prominently
- Document that these serve as demographic linkage keys
- No external data fetching

### Phase 3: E2E Test Updates

Update test scenarios for revised ascent:
- Verify 4 quartile categories created
- Verify Commune column present in L2
- Verify linkage key columns in L3

---

## Complexity Tracking

No constitution violations requiring justification. The feature is a test suite with session persistence and ascent UI enhancement, not a new abstraction layer.

