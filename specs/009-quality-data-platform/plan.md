# Implementation Plan: Quality Data Platform

**Branch**: `009-quality-data-platform` | **Date**: 2025-12-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/009-quality-data-platform/spec.md`

## Summary

Build a curated open data platform that leverages TabPFN (Nature 2025) to provide dataset quality assessment, feature engineering suggestions, anomaly detection, and synthetic data generation. The platform transforms raw datasets into ML-ready assets with usability scores, enabling data scientists to quickly identify high-quality datasets and non-technical domain experts to understand their data better.

## Technical Context

**Language/Version**: Python 3.11 (existing `myenv311` virtual environment)
**Primary Dependencies**:
- Streamlit >=1.28.0 (existing UI framework)
- TabPFN 2.0 (via `tabpfn` PyPI package - Nature 2025 release)
- pandas, numpy, scikit-learn (existing)
- SHAP (for interpretable feature importance)

**Storage**:
- JSON files for quality reports and catalog metadata (aligns with existing session persistence)
- CSV files for datasets and synthetic data exports
- Session state for in-progress assessments

**Testing**: pytest with Playwright MCP for E2E testing (existing pattern)

**Target Platform**: Streamlit web application (existing), deployable to Streamlit Cloud

**Project Type**: Single project (extends existing `intuitiveness` package)

**Performance Goals**:
- Quality assessment in <30 seconds for datasets up to 5,000 rows (TabPFN: ~2.8s inference)
- Catalog search <2 seconds for 1,000 datasets

**Constraints**:
- TabPFN optimal range: 50-10,000 rows, up to 500 features
- Consumer-grade GPU recommended but not required (CPU fallback ~10x slower)
- Memory: ~1KB per cell for TabPFN inference

**Scale/Scope**:
- Initial catalog: 10-100 datasets
- Target: 1,000+ datasets with quality scores

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status | Evidence |
|-----------|------|--------|----------|
| I. Abstraction Levels | Feature addresses L2-L4 datasets | PASS | Quality reports work on tables (L2) and multi-level datasets (L3-L4) |
| II. Descent-Ascent | Feature supports data transformation cycle | PASS | Feature engineering suggestions guide ascent dimension selection |
| III. Complexity Quantification | Complexity is measurable | PASS | Usability score (0-100) quantifies dataset complexity for ML tasks |
| IV. Human-Data Interaction | Ground truth traceable | PASS | SHAP values trace predictions to individual features (L0 granules) |
| V. Diverse Data Publics | Serves non-technical users | PASS | Quality scores and recommendations use domain language, not ML jargon |

**Target User Assumption Check**:
- Quality reports use plain language ("This feature helps predict the outcome" vs "high SHAP value")
- Recommendations are actionable without ML knowledge ("Consider combining City and Region")
- Anomaly explanations reference domain concepts ("This row has unusual Age for its Income level")

## Project Structure

### Documentation (this feature)

```text
specs/009-quality-data-platform/
├── plan.md              # This file
├── research.md          # Phase 0 output - TabPFN integration patterns
├── data-model.md        # Phase 1 output - entities and relationships
├── quickstart.md        # Phase 1 output - getting started guide
├── contracts/           # Phase 1 output - API contracts
│   └── quality_api.yaml # Assessment and catalog endpoints
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
intuitiveness/
├── quality/                    # NEW: Quality assessment module
│   ├── __init__.py
│   ├── assessor.py            # TabPFN-based quality scoring
│   ├── feature_engineer.py    # Feature suggestion engine
│   ├── anomaly_detector.py    # Density-based anomaly detection
│   ├── synthetic_generator.py # Synthetic data generation
│   └── report.py              # Quality report generation
├── catalog/                    # NEW: Dataset catalog module
│   ├── __init__.py
│   ├── models.py              # Catalog entities
│   ├── storage.py             # JSON-based catalog persistence
│   └── search.py              # Filtering and sorting
├── ui/
│   ├── quality_dashboard.py   # NEW: Quality assessment UI
│   ├── catalog_browser.py     # NEW: Catalog browsing UI
│   └── ... (existing)
└── ... (existing modules)

tests/
├── unit/
│   ├── test_quality_assessor.py
│   ├── test_feature_engineer.py
│   ├── test_anomaly_detector.py
│   └── test_synthetic_generator.py
├── integration/
│   └── test_catalog_flow.py
└── contract/
    └── test_quality_api.py
```

**Structure Decision**: Extends existing `intuitiveness` package with two new submodules (`quality/` and `catalog/`). This preserves the single-project architecture while adding the quality platform capabilities. New UI components integrate with existing Streamlit app structure.

## Complexity Tracking

> No constitution violations requiring justification.

| Decision | Rationale |
|----------|-----------|
| New `quality/` module | Separates TabPFN logic from existing navigation/redesign code |
| New `catalog/` module | Catalog is a distinct concern from quality assessment |
| JSON storage for catalog | Matches existing session persistence pattern, avoids new DB dependency |
