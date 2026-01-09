# Implementation Plan: Data Scientist Co-Pilot

**Branch**: `010-quality-ds-workflow` | **Date**: 2025-12-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/010-quality-ds-workflow/spec.md`

## Summary

Transform the existing quality assessment module into a "data scientist co-pilot" — a 60-second workflow that gets messy CSV data modeling-ready. Key additions: synthetic-to-real validation pipeline, one-click "Apply All Suggestions" with before/after benchmarks, traffic light readiness indicator, and export-and-go functionality (clean CSV + Python code snippet).

This feature extends `009-quality-data-platform` by adding:
1. **Synthetic validation loop** — benchmark train-on-synthetic/test-on-real transfer
2. **Batch transformation** — apply all suggestions with accuracy tracking
3. **Export workflow** — DataFrame export + code generation
4. **UX improvements** — traffic light indicator, progress feedback

## Technical Context

**Language/Version**: Python 3.11 (existing `myenv311` virtual environment)
**Primary Dependencies**:
- Streamlit >=1.28.0 (UI framework)
- TabPFN (quality assessment, synthetic generation)
- pandas (data manipulation)
- scikit-learn (benchmarking models: LogisticRegression, RandomForest, XGBoost)
- numpy (numerical operations)

**Storage**:
- Session state for transformed datasets
- Local file export (CSV, Pickle, Parquet)
- No persistent database required

**Testing**:
- pytest for unit tests
- Playwright MCP for E2E UI tests

**Target Platform**: Streamlit web application (localhost + Streamlit Cloud)
**Project Type**: Single Python package with Streamlit UI

**Performance Goals**:
- Upload→Assess→Fix→Export in under 60 seconds for 5,000 rows
- Synthetic validation benchmark in under 30 seconds

**Constraints**:
- TabPFN works best on datasets 50-10,000 rows, up to 500 features
- Must provide fallback when TabPFN unavailable

**Scale/Scope**:
- Single-user local application
- Datasets up to 10,000 rows × 500 features

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Compliance | Notes |
|-----------|------------|-------|
| **I. Intuitiveness Through Abstraction Levels** | ✅ PASS | Quality assessment operates at L2 (Table level) — users upload CSV, receive assessment, export clean DataFrame |
| **II. Descent-Ascent Cycle** | ✅ PASS | Feature doesn't modify core descent-ascent; operates as a parallel "quality prep" workflow |
| **III. Complexity Quantification** | ✅ PASS | Usability score (0-100) quantifies data quality; traffic light simplifies interpretation |
| **IV. Human-Data Interaction Granularity** | ✅ PASS | Users can trace from usability score → feature profiles → individual transformations |
| **V. Design for Diverse Data Publics** | ✅ PASS | Traffic light (green/yellow/red) + "Apply All" button designed for non-technical domain experts |

**Target User Assumption Check**:
- ✅ Domain terminology used: "modeling-ready", "data quality", not "DataFrame normalization"
- ✅ One-click actions: "Apply All Suggestions", "Export Clean CSV"
- ✅ No technical jargon in UI: Traffic light instead of numeric scores

**Quality Gates**:
- ✅ Data integrity preserved through transformations (logged in metadata)
- ✅ Export includes transformation record for auditability
- ✅ Testable via Playwright E2E (upload CSV → export clean CSV)

## Project Structure

### Documentation (this feature)

```text
specs/010-quality-ds-workflow/
├── plan.md              # This file
├── research.md          # Phase 0 output - technical research
├── data-model.md        # Phase 1 output - entity definitions
├── quickstart.md        # Phase 1 output - developer guide
├── contracts/           # Phase 1 output - API contracts
│   └── quality_api.yaml # Internal API contract
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
intuitiveness/
├── quality/                      # Existing quality module (009)
│   ├── __init__.py              # Module exports
│   ├── assessor.py              # TabPFN assessment (EXTEND)
│   ├── synthetic_generator.py   # Synthetic generation (EXTEND)
│   ├── feature_engineer.py      # Suggestions (EXTEND)
│   ├── report.py                # Quality report (EXTEND)
│   ├── models.py                # Data models (EXTEND)
│   ├── anomaly_detector.py      # Anomaly detection (unchanged)
│   ├── tabpfn_wrapper.py        # TabPFN wrapper (unchanged)
│   ├── tabpfn_auth.py           # TabPFN auth (unchanged)
│   ├── benchmark.py             # NEW: Synthetic validation pipeline
│   └── exporter.py              # NEW: DataFrame export + code gen
│
├── ui/
│   └── quality_dashboard.py     # Quality UI (MAJOR EXTEND)
│
└── streamlit_app.py             # Main app (minor updates)

tests/
├── unit/
│   ├── test_benchmark.py        # NEW: Benchmark tests
│   └── test_exporter.py         # NEW: Export tests
├── integration/
│   └── test_quality_workflow.py # NEW: End-to-end quality workflow
└── artifacts/
    └── quality_ds/              # Test artifacts for this feature
```

**Structure Decision**: Extends existing `intuitiveness/quality/` module with 2 new files (`benchmark.py`, `exporter.py`) and major updates to `quality_dashboard.py`. Follows single-project structure consistent with existing codebase.

## Implementation Phases

### Phase 0: Research (Complete)
- [x] Analyze existing quality module
- [x] Review TabPFN capabilities for synthetic validation
- [x] Design benchmark methodology
- Output: `research.md`

### Phase 1: Design & Contracts
- [ ] Define data models for benchmark results
- [ ] Design export package structure
- [ ] Create API contracts for internal functions
- Output: `data-model.md`, `contracts/`, `quickstart.md`

### Phase 2: Tasks (via /speckit.tasks)
- [ ] Generate implementation tasks
- Output: `tasks.md`

## Complexity Tracking

> No constitution violations requiring justification.

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| New files | 2 new files in quality/ | Keeps benchmark and export logic isolated; follows single responsibility |
| UI changes | Major update to quality_dashboard.py | Centralizes all quality UX changes in one file |
| No new dependencies | Uses existing sklearn, pandas | Benchmark uses models already available in environment |
