# Implementation Plan: Data.gouv.fr Search Integration

**Branch**: `008-datagouv-search` | **Date**: 2025-12-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/008-datagouv-search/spec.md`

## Summary

Add a search-first interface to the Data Redesign Method application, enabling users to discover and load French open data from data.gouv.fr directly. The interface presents a prominent search bar with "Redesign any data for your intent" messaging (bilingual EN/FR), displays search results as cards, and loads selected CSV resources into the existing L4 workflow.

**Technical Approach**: Integrate the existing `DataGouvAPI` library from `skills/data-gouv/lib/datagouv.py` into a new Streamlit UI component. Replace the methodology intro section with a search-first experience while preserving the file upload fallback.

## Technical Context

**Language/Version**: Python 3.11 (existing `myenv311` virtual environment)
**Primary Dependencies**: Streamlit >=1.28.0, pandas, requests (all already installed)
**Storage**: Session state for search results; local file cache for downloaded CSVs (~/.cache/datagouv)
**Testing**: Playwright MCP for E2E tests (existing pattern from 006-playwright-mcp-e2e)
**Target Platform**: Web browser (Streamlit app)
**Project Type**: Single project (existing Streamlit application)
**Performance Goals**: Search results in <3s, dataset load <60s (excluding network download)
**Constraints**: Must preserve existing file upload workflow; graceful degradation when API unavailable
**Scale/Scope**: Single page feature addition; ~4 new UI components

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| **I. Intuitiveness Through Abstraction Levels** | Users MUST enter at L4 | PASS | Search results load as L4 datasets, same as file upload |
| **II. Descent-Ascent Cycle** | All redesign follows Descent-Ascent | PASS | Loaded datasets enter existing workflow at L4 |
| **III. Complexity Quantification** | Complexity measurable between levels | PASS | No change to level transitions |
| **IV. Human-Data Interaction Granularity** | Upper levels constructible from lower | PASS | No change to level structure |
| **V. Design for Diverse Data Publics** | Non-technical users (domain experts) | PASS | Search by natural language intent; no technical terms exposed |
| **V. Target User Assumption** | Shield users from technical complexity | PASS | Users search by domain questions ("vaccination France"), not API syntax |

**Domain-Friendly Labels Check**:
- "Redesign any data for your intent" - domain-focused, intent-driven
- Search results show dataset titles/descriptions, not technical IDs
- No exposure to API structure, pagination, or technical metadata

## Project Structure

### Documentation (this feature)

```text
specs/008-datagouv-search/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API interfaces)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
intuitiveness/
├── streamlit_app.py          # Main app - ADD search integration
├── ui/
│   ├── __init__.py           # MODIFY - export new components
│   ├── datagouv_search.py    # NEW - Search bar and results component
│   └── i18n.py               # MODIFY - add search translations
├── services/
│   └── datagouv_client.py    # NEW - Wrapper around DataGouvAPI
└── styles/
    └── search.py             # NEW - Search component styling

skills/data-gouv/lib/
└── datagouv.py               # EXISTING - API client (already downloaded)

tests/
├── e2e/
│   └── test_datagouv_search.py  # NEW - Playwright E2E tests
└── unit/
    └── test_datagouv_client.py  # NEW - Unit tests for client wrapper
```

**Structure Decision**: Single project extension. New components integrate into existing `intuitiveness/ui/` pattern. Service layer wraps external library for testability.

## Complexity Tracking

No violations requiring justification. Feature extends existing L4 entry point pattern.

---

## Phase 0: Research

### Research Tasks

1. **Data.gouv.fr API Capabilities**: Understand search endpoint parameters, response structure, rate limits
2. **Existing DataGouvAPI Usage**: Review the skill library's methods and caching behavior
3. **Streamlit Search Patterns**: Best practices for search interfaces in Streamlit
4. **i18n Extension**: Pattern for adding new translation keys to existing system

### Research Findings

See [research.md](./research.md) for detailed findings.

---

## Phase 1: Design & Contracts

### Data Model

See [data-model.md](./data-model.md) for entity definitions.

### API Contracts

See [contracts/](./contracts/) for interface specifications.

### Quickstart

See [quickstart.md](./quickstart.md) for developer setup guide.
