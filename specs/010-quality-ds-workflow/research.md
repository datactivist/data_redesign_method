# Research: Data Scientist Co-Pilot

**Feature**: 010-quality-ds-workflow
**Date**: 2025-12-13

## Research Questions

### RQ1: How to benchmark synthetic data quality?

**Decision**: Train-on-synthetic, test-on-real methodology

**Rationale**:
- Standard approach in ML literature for evaluating synthetic data utility
- Measures practical value: "Can I train models on synthetic data?"
- Produces interpretable metric: transfer gap (% accuracy drop)

**Implementation**:
```python
def benchmark_synthetic(df, target_col, n_synthetic, models):
    # 1. Split real data: 80% train, 20% test (held out)
    real_train, real_test = train_test_split(df, test_size=0.2)

    # 2. Generate synthetic data from real_train only
    synthetic_train = generate_synthetic(real_train, n_synthetic)

    # 3. Benchmark both approaches
    for model in models:
        # Real → Real (baseline)
        model.fit(real_train[X_cols], real_train[target_col])
        real_score = model.score(real_test[X_cols], real_test[target_col])

        # Synthetic → Real (transfer)
        model.fit(synthetic_train[X_cols], synthetic_train[target_col])
        synth_score = model.score(real_test[X_cols], real_test[target_col])

        transfer_gap = (real_score - synth_score) / real_score * 100

    return BenchmarkReport(...)
```

**Alternatives Considered**:
- Statistical tests only (KS-test, correlation): Rejected — doesn't measure ML utility
- Train on mixed data: Rejected — harder to interpret, less clear signal
- Cross-validation on synthetic only: Rejected — doesn't test transfer to real data

---

### RQ2: Which models to use for benchmarking?

**Decision**: LogisticRegression, RandomForest, XGBoost (3 models)

**Rationale**:
- Covers linear, tree-based, and boosting paradigms
- All available in scikit-learn (already in environment)
- Fast enough for <30 second benchmark constraint
- Representative of common production models

**Configuration**:
```python
BENCHMARK_MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False)
}
```

**Alternatives Considered**:
- TabPFN only: Rejected — users want to know if data works with their models
- Neural networks: Rejected — too slow for 30-second constraint
- Single model (RF only): Rejected — doesn't show model-agnostic quality

---

### RQ3: How to implement "Apply All Suggestions"?

**Decision**: Sequential application with rollback capability

**Rationale**:
- Suggestions may have dependencies (e.g., impute before transform)
- Need to track individual accuracy deltas for transparency
- Rollback needed if transformation degrades quality

**Implementation**:
```python
def apply_all_suggestions(df, suggestions, target_col):
    results = []
    current_df = df.copy()
    baseline_accuracy = quick_benchmark(current_df, target_col)

    for suggestion in sorted(suggestions, key=lambda s: s.priority):
        try:
            new_df = apply_suggestion(current_df, suggestion)
            new_accuracy = quick_benchmark(new_df, target_col)
            delta = new_accuracy - baseline_accuracy

            results.append(TransformationResult(
                suggestion=suggestion,
                accuracy_delta=delta,
                success=True
            ))

            current_df = new_df
            baseline_accuracy = new_accuracy
        except Exception as e:
            results.append(TransformationResult(
                suggestion=suggestion,
                error=str(e),
                success=False
            ))

    return current_df, results
```

**Alternatives Considered**:
- Parallel application: Rejected — transformations may conflict
- No accuracy tracking: Rejected — users need proof of improvement
- Apply without confirmation: Rejected — need to show preview first

---

### RQ4: What export formats to support?

**Decision**: CSV (primary), Pickle (fast), Parquet (efficient)

**Rationale**:
- CSV: Universal, human-readable, works in any environment
- Pickle: Fast Python serialization, preserves dtypes exactly
- Parquet: Efficient storage, good for larger datasets, preserves types

**Implementation**:
```python
def export_dataset(df, format, metadata):
    if format == 'csv':
        return df.to_csv(index=False)
    elif format == 'pickle':
        return pickle.dumps(df)
    elif format == 'parquet':
        return df.to_parquet()
```

**Code Snippet Template**:
```python
JUPYTER_SNIPPET = '''
# Load your modeling-ready data
import pandas as pd

df = pd.read_csv('{filename}')
X = df.drop('{target_col}', axis=1)
y = df['{target_col}']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start modeling!
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# print(f"Accuracy: {{model.score(X_test, y_test):.2%}}")
'''
```

---

### RQ5: Traffic light threshold design?

**Decision**: Score-based with dynamic messaging

| Score Range | Traffic Light | Message |
|-------------|---------------|---------|
| 80-100 | 🟢 READY | "Export and start training!" |
| 60-79 | 🟡 FIXABLE | "N fixes will improve score to X" |
| 0-59 | 🔴 NEEDS WORK | "Review recommendations below" |

**Rationale**:
- 80+ threshold: High confidence data won't cause modeling issues
- 60-79 threshold: Common issues that are automatable
- <60 threshold: Significant problems requiring manual review

**Alternatives Considered**:
- Binary (ready/not ready): Rejected — too coarse, misses "fixable" state
- 5-level scale: Rejected — too complex, cognitive overload
- Dynamic thresholds based on dataset: Rejected — inconsistent UX

---

### RQ6: Class balancing in synthetic generation?

**Decision**: Stratified generation with per-class sample counts

**Implementation**:
```python
def generate_balanced_synthetic(df, target_col, samples_per_class=None):
    classes = df[target_col].unique()

    if samples_per_class is None:
        # Default: equal to largest class
        samples_per_class = df[target_col].value_counts().max()

    balanced_samples = []
    for cls in classes:
        class_df = df[df[target_col] == cls]
        n_to_generate = samples_per_class - len(class_df)

        if n_to_generate > 0:
            synthetic = generate_synthetic(class_df, n_to_generate)
            balanced_samples.append(synthetic)

    return pd.concat([df] + balanced_samples, ignore_index=True)
```

**Alternatives Considered**:
- SMOTE: Rejected — creates interpolated samples, not true synthetic
- Undersampling majority: Rejected — loses data
- Random oversampling: Rejected — duplicates existing samples

---

## Technology Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Benchmark methodology | Train-synthetic/Test-real | ML utility measurement |
| Benchmark models | LR, RF, XGB | Fast, representative, available |
| Suggestion application | Sequential with rollback | Dependency handling, transparency |
| Export formats | CSV, Pickle, Parquet | Universal to efficient spectrum |
| Traffic light thresholds | 80/60 split | Intuitive, actionable |
| Class balancing | Per-class generation | Preserves class distributions |

## Dependencies Confirmed

All required dependencies are already in `myenv311`:
- ✅ scikit-learn (LogisticRegression, RandomForest)
- ✅ xgboost (XGBClassifier)
- ✅ pandas (DataFrame operations)
- ✅ numpy (numerical operations)
- ✅ streamlit (UI components)
