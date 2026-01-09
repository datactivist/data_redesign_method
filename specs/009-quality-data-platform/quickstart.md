# Quickstart: Quality Data Platform

**Feature**: 009-quality-data-platform
**Date**: 2025-12-13

## Prerequisites

1. Python 3.11 with the existing `myenv311` virtual environment
2. TabPFN package installed (see Setup below)
3. Streamlit application running

## Setup

### 1. Activate Environment

```bash
source myenv311/bin/activate
```

### 2. Install New Dependencies

```bash
# Primary: Cloud API (recommended - no GPU needed)
pip install --upgrade tabpfn-client shap

# Optional: Local TabPFN (requires GPU for optimal speed)
pip install tabpfn shap
```

### 3. Authenticate with TabPFN API (first time only)

```python
import tabpfn_client

# This opens a browser for authentication
token = tabpfn_client.get_access_token()

# Save token for future sessions
tabpfn_client.set_access_token(token)
print("TabPFN API authenticated!")
```

### 4. Verify Installation

```python
from tabpfn_client import TabPFNClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Quick test
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = TabPFNClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f"TabPFN ready! Test accuracy: {accuracy:.2%}")
```

## Quick Usage Examples

### Assess a Dataset

```python
import pandas as pd
from intuitiveness.quality import assess_dataset

# Load your data
df = pd.read_csv("my_dataset.csv")

# Run quality assessment
report = assess_dataset(
    df=df,
    target_column="target",
    task_type="auto"  # or "classification" / "regression"
)

# View results
print(f"Usability Score: {report.usability_score}/100")
print(f"Prediction Quality: {report.prediction_quality}/100")
print(f"Data Completeness: {report.data_completeness}/100")

# Top features
for fp in sorted(report.feature_profiles, key=lambda x: -x.importance_score)[:5]:
    print(f"  {fp.feature_name}: {fp.importance_score:.2f}")
```

### Get Feature Engineering Suggestions

```python
from intuitiveness.quality import suggest_features, apply_suggestion

# Get suggestions based on quality report
suggestions = suggest_features(report, max_suggestions=5)

for s in suggestions:
    print(f"[{s.suggestion_type}] {s.description}")
    print(f"  Expected impact: +{s.expected_impact:.1f} points")

# Apply a suggestion
if suggestions:
    df_improved = apply_suggestion(df, suggestions[0])
    print("Suggestion applied!")
```

### Detect Anomalies

```python
from intuitiveness.quality import detect_anomalies

# Find unusual rows
anomalies = detect_anomalies(df, percentile_threshold=2.0)

print(f"Found {len(anomalies)} anomalous rows")
for a in anomalies[:3]:
    print(f"\nRow {a.row_index} (percentile: {a.percentile:.1f}%):")
    for contrib in a.top_contributors:
        print(f"  - {contrib['feature']}: {contrib['reason']}")
```

### Generate Synthetic Data

```python
from intuitiveness.quality import generate_synthetic

# Generate 100 synthetic samples
synthetic_df, metrics = generate_synthetic(
    df=df,
    n_samples=100,
    preserve_correlations=True
)

print(f"Generated {len(synthetic_df)} samples")
print(f"Correlation preservation: {1 - metrics.mean_correlation_error:.1%}")
print(f"Distribution similarity: {metrics.distribution_similarity:.1%}")

# Save to file
synthetic_df.to_csv("synthetic_data.csv", index=False)
```

### Browse the Catalog

```python
from intuitiveness.catalog import filter_datasets, add_dataset

# Add a dataset to catalog
dataset = add_dataset(
    name="Customer Churn Dataset",
    file_path="data/churn.csv",
    description="Telecom customer churn prediction",
    domain_tags=["telecom", "classification"],
    target_column="Churn",
    auto_assess=True  # Automatically run quality assessment
)

print(f"Added: {dataset.name} (ID: {dataset.id})")
print(f"Usability Score: {dataset.usability_score}")

# Search for high-quality datasets
top_datasets = filter_datasets(
    min_score=70,
    domains=["healthcare"],
    sort_by="usability_score",
    limit=10
)

for ds in top_datasets:
    print(f"{ds.name}: {ds.usability_score}/100")
```

## Using in Streamlit UI

The quality platform is integrated into the main Streamlit app:

```bash
streamlit run app.py
```

Navigate to:
- **Quality Assessment**: Upload or select a dataset → Get instant quality report
- **Feature Suggestions**: View and apply engineering recommendations
- **Anomaly Detection**: Identify and investigate unusual records
- **Synthetic Data**: Generate privacy-preserving synthetic samples
- **Dataset Catalog**: Browse, filter, and discover high-quality datasets

## Troubleshooting

### TabPFN API Fallback

If you don't have a GPU, TabPFN will use CPU (slower) or you can use the API:

```python
from tabpfn import TabPFNClassifier

# Use API for faster inference without GPU
clf = TabPFNClassifier(use_api=True)
```

### Dataset Too Large

TabPFN works best with 50-10,000 rows. For larger datasets:

```python
# Sample before assessment
if len(df) > 10000:
    df_sample = df.sample(n=10000, random_state=42)
    report = assess_dataset(df_sample, target_column="target")
    print("Note: Assessment based on 10,000-row sample")
```

### Missing Target Column

For unsupervised assessment (no target):

```python
# Use density estimation only
anomalies = detect_anomalies(df)  # No target needed
```

## Next Steps

1. **Add your datasets** to the catalog
2. **Run assessments** to get usability scores
3. **Apply suggestions** to improve data quality
4. **Share synthetic data** for privacy-preserving collaboration
