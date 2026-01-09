# Data Model: Quality Data Platform

**Feature**: 009-quality-data-platform
**Date**: 2025-12-13
**Status**: Complete

## Entity Overview

```
┌─────────────────┐       ┌─────────────────┐
│    Dataset      │1─────*│  QualityReport  │
└────────┬────────┘       └────────┬────────┘
         │                         │
         │                         │1
         │                         │
         │                         ▼
         │                ┌─────────────────┐
         │                │  FeatureProfile │
         │                └─────────────────┘
         │                         *
         │1                        │
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  SyntheticData  │       │  AnomalyRecord  │
└─────────────────┘       └─────────────────┘
```

## Entities

### Dataset

The core entity representing a tabular data file in the catalog.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK, auto-generated | Unique identifier |
| name | string | required, max 200 chars | Human-readable dataset name |
| description | string | optional, max 2000 chars | Dataset purpose and contents |
| domain_tags | string[] | optional | Domain categories (e.g., "healthcare", "finance") |
| file_path | string | required | Path to CSV file |
| row_count | integer | required, ≥0 | Number of rows in dataset |
| feature_count | integer | required, ≥0 | Number of columns (excluding target) |
| target_column | string | optional | Designated target for supervised tasks |
| usability_score | float | 0-100, nullable | Overall ML-readiness score |
| latest_report_id | UUID | FK to QualityReport | Most recent quality assessment |
| created_at | datetime | auto | When dataset was added |
| updated_at | datetime | auto | Last modification time |

**Validation Rules**:
- `row_count` must be ≥50 for quality assessment
- `feature_count` must be ≤500 for TabPFN compatibility
- `usability_score` is null until first assessment

**State Transitions**:
```
[Created] → [Assessed] → [Re-assessed]
              ↓
         [Archived]
```

---

### QualityReport

Results of a TabPFN-based quality assessment.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK, auto-generated | Unique identifier |
| dataset_id | UUID | FK to Dataset, required | Assessed dataset |
| usability_score | float | 0-100, required | Composite quality score |
| prediction_quality | float | 0-100, required | TabPFN cross-validation accuracy |
| data_completeness | float | 0-100, required | (1 - missing_ratio) * 100 |
| feature_diversity | float | 0-100, required | Entropy of feature types |
| size_appropriateness | float | 0-100, required | Penalty for extreme sizes |
| target_column | string | required | Column used for assessment |
| task_type | enum | "classification" or "regression" | Detected or specified task |
| assessment_time_seconds | float | ≥0 | Time taken for assessment |
| created_at | datetime | auto | When assessment was run |

**Relationships**:
- Has many `FeatureProfile` (one per feature)
- Has many `AnomalyRecord` (flagged rows)
- Has many `FeatureSuggestion` (engineering recommendations)

---

### FeatureProfile

Per-feature statistics and importance scores.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK, auto-generated | Unique identifier |
| report_id | UUID | FK to QualityReport, required | Parent report |
| feature_name | string | required | Column name |
| feature_type | enum | "numeric", "categorical", "boolean", "datetime" | Detected type |
| missing_count | integer | ≥0 | Number of missing values |
| missing_ratio | float | 0-1 | Fraction of missing values |
| unique_count | integer | ≥0 | Number of unique values |
| importance_score | float | 0-1 | TabPFN ablation importance |
| shap_mean | float | any | Mean absolute SHAP value |
| distribution_skew | float | any | Skewness for numeric features |
| suggested_transform | string | optional | Recommended transformation |

**Validation Rules**:
- `missing_ratio` = `missing_count` / dataset.row_count
- `importance_score` normalized across all features (sum = 1)

---

### AnomalyRecord

Flagged rows with unusual density scores.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK, auto-generated | Unique identifier |
| report_id | UUID | FK to QualityReport, required | Parent report |
| row_index | integer | ≥0 | Original row number in dataset |
| anomaly_score | float | any | Log-density score (lower = more anomalous) |
| percentile | float | 0-100 | Density percentile (lower = more anomalous) |
| top_contributors | json | required | Top 3 features contributing to anomaly |

**Example `top_contributors`**:
```json
[
  {"feature": "Age", "contribution": 0.45, "reason": "Unusually high for Income level"},
  {"feature": "Income", "contribution": 0.30, "reason": "Outlier value"},
  {"feature": "City", "contribution": 0.15, "reason": "Rare category"}
]
```

---

### FeatureSuggestion

Recommended feature engineering actions.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK, auto-generated | Unique identifier |
| report_id | UUID | FK to QualityReport, required | Parent report |
| suggestion_type | enum | "remove", "transform", "combine" | Type of recommendation |
| target_features | string[] | required, 1-2 elements | Feature(s) involved |
| description | string | required | Plain-language explanation |
| expected_impact | float | -100 to +100 | Expected change in usability score |
| confidence | float | 0-1 | Model confidence in suggestion |

**Example Suggestions**:
```json
[
  {
    "suggestion_type": "remove",
    "target_features": ["ID"],
    "description": "This identifier column doesn't help predictions",
    "expected_impact": 2.5,
    "confidence": 0.95
  },
  {
    "suggestion_type": "transform",
    "target_features": ["Income"],
    "description": "Apply log transform - this feature is heavily right-skewed",
    "expected_impact": 5.0,
    "confidence": 0.80
  },
  {
    "suggestion_type": "combine",
    "target_features": ["City", "State"],
    "description": "Combine into a single location feature - they are highly correlated",
    "expected_impact": 3.0,
    "confidence": 0.70
  }
]
```

---

### SyntheticData

Generated synthetic samples for a dataset.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK, auto-generated | Unique identifier |
| dataset_id | UUID | FK to Dataset, required | Source dataset |
| n_samples | integer | ≥1 | Number of generated rows |
| file_path | string | required | Path to synthetic CSV |
| mean_correlation_error | float | 0-1 | Quality metric: correlation preservation |
| distribution_similarity | float | 0-1 | Quality metric: KS-test average |
| generation_time_seconds | float | ≥0 | Time taken to generate |
| created_at | datetime | auto | When generated |

**Validation Rules**:
- `mean_correlation_error` < 0.1 for "high quality" synthetic data
- `distribution_similarity` > 0.9 for "high quality" synthetic data

---

## Catalog Index

For fast search and filtering, maintain an in-memory index:

```python
@dataclass
class CatalogIndex:
    """In-memory index for fast catalog queries."""
    datasets: Dict[UUID, Dataset]
    by_domain: Dict[str, List[UUID]]  # domain_tag -> dataset IDs
    by_score: List[Tuple[float, UUID]]  # sorted by usability_score desc

    def filter(self,
               min_score: float = None,
               domains: List[str] = None,
               min_rows: int = None,
               max_rows: int = None) -> List[Dataset]:
        """Filter datasets by criteria."""
        ...

    def search(self, query: str) -> List[Dataset]:
        """Full-text search in name and description."""
        ...
```

---

## JSON Storage Schema

All entities are persisted as JSON files following this structure:

```
catalog/
├── catalog.json           # Dataset index
├── datasets/
│   ├── {dataset_id}/
│   │   ├── metadata.json  # Dataset entity
│   │   ├── data.csv       # Actual data file
│   │   └── reports/
│   │       ├── {report_id}.json  # QualityReport + nested entities
│   │       └── ...
│   └── ...
└── synthetic/
    ├── {synthetic_id}.json  # SyntheticData metadata
    └── {synthetic_id}.csv   # Synthetic data file
```
