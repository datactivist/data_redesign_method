# Research: Quality Data Platform

**Feature**: 009-quality-data-platform
**Date**: 2025-12-13
**Status**: Complete

## Research Topics

### 1. TabPFN Integration for Quality Assessment

**Decision**: Use `tabpfn-client` (cloud API) as primary, with `tabpfn` (local) as fallback

**Rationale**:
- TabPFN is a tabular foundation model published in Nature (January 2025)
- Outperforms all baselines on datasets with 50-10,000 samples (up to 50,000 for TabPFN-2.5)
- Single forward pass inference (~2.8s classification, ~4.8s regression)
- `tabpfn-client` provides cloud-based inference without GPU requirements
- 100M daily credits sufficient for platform usage
- Same sklearn-compatible interface between local and client versions

**Package Options**:
| Package | When to Use | Pros | Cons |
|---------|-------------|------|------|
| `tabpfn-client` | Streamlit Cloud, no GPU | Cloud compute, fast | Data sent to servers, requires auth |
| `tabpfn` | Local with GPU | Privacy, no auth needed | Requires GPU for optimal speed |

**Alternatives Considered**:
- AutoML (AutoGluon, H2O): Slower (4h vs 2.8s), more complex setup
- XGBoost/CatBoost: No density estimation or synthetic generation
- Custom neural networks: Require training per dataset

**Key Implementation Details** (from PriorLabs GitHub):

```python
# Option 1: Cloud API (recommended for Streamlit Cloud)
from tabpfn_client import TabPFNClassifier, TabPFNRegressor
import tabpfn_client

# Authentication (first time)
token = tabpfn_client.get_access_token()
tabpfn_client.set_access_token(token)

# Usage (sklearn-compatible)
classifier = TabPFNClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict_proba(X_test)

# Option 2: Local (requires GPU)
from tabpfn import TabPFNClassifier, TabPFNRegressor

# Enable KV cache for faster predictions
classifier = TabPFNClassifier(fit_mode='fit_with_cache')
classifier.fit(X_train, y_train)
predictions = classifier.predict_proba(X_test)
```

**Unsupervised Capabilities** (from TabPFN ecosystem):
- Data imputation
- Synthetic tabular data generation
- Outlier detection via density estimation
- Learned embedding extraction

**API Limits** (tabpfn-client):
- Daily credits: 100,000,000
- Cost formula: `max((n_train + n_test) * n_cols * n_estimators, 5000)`
- Max cells per request: 20,000,000
- Max test samples for regression full output: 500

### 2. Usability Score Computation

**Decision**: Composite score based on TabPFN predictive performance + data quality metrics

**Rationale**:
- Pure prediction accuracy doesn't capture "usability" for data scientists
- Need to combine ML-readiness with data hygiene factors
- Score should be interpretable without ML knowledge

**Formula**:
```
usability_score = 0.4 * prediction_quality + 0.3 * data_completeness + 0.2 * feature_diversity + 0.1 * size_appropriateness
```

Where:
- `prediction_quality`: TabPFN cross-validation accuracy (normalized 0-100)
- `data_completeness`: (1 - missing_ratio) * 100
- `feature_diversity`: Entropy of feature types (numeric, categorical, mixed)
- `size_appropriateness`: Penalty for <50 or >10,000 rows

**Alternatives Considered**:
- Pure TabPFN accuracy: Doesn't capture data quality issues
- Statistical tests only: Misses predictive power
- Manual checklists: Not automated, inconsistent

### 3. Feature Engineering Suggestions

**Decision**: Use TabPFN ablation studies + SHAP values to identify impactful features

**Rationale**:
- Remove features one-by-one and measure prediction drop (ablation)
- SHAP values show feature contribution to predictions
- Combine with domain-agnostic heuristics (log transform for skewed, interaction for correlated)

**Implementation Approach**:
1. Baseline TabPFN score
2. For each feature: compute score without it (importance = baseline - ablated)
3. Compute SHAP values for interpretability
4. Generate suggestions:
   - "Remove X" if importance < threshold
   - "Transform X" if distribution is heavily skewed
   - "Combine X and Y" if correlation > 0.8 and both important

**Alternatives Considered**:
- Feature selection algorithms (RFE, Lasso): Slower, less interpretable
- AutoML feature engineering: Black box, hard to explain to users

### 4. Anomaly Detection via Density Estimation

**Decision**: Use TabPFN's density estimation with percentile-based thresholds

**Rationale**:
- TabPFN can estimate p(x|D) using its generative capabilities
- Density estimation handles mixed feature types naturally
- Percentile thresholds (e.g., bottom 2%) are interpretable

**Implementation**:
```python
# Compute log-likelihood for each row
log_densities = tabpfn_model.predict_density(X)

# Flag rows below 2nd percentile as anomalies
threshold = np.percentile(log_densities, 2)
anomalies = log_densities < threshold

# For each anomaly, compute per-feature contribution
# by masking features and measuring density change
```

**Alternatives Considered**:
- Isolation Forest: Less interpretable, doesn't use TabPFN
- Autoencoders: Require training, poor on small datasets
- Statistical outliers (z-score): Only works for numerical features

### 5. Synthetic Data Generation

**Decision**: Use TabPFN's autoregressive generation following the factorized joint distribution

**Rationale**:
- TabPFN learns p(x,y|D) during in-context learning
- Can sample from this distribution feature-by-feature
- Preserves correlations and relationships between features

**Implementation** (from Nature paper):
```python
# Generate synthetic samples
# Factorization: p(x,y|D) = Π p(xⱼ|x<ⱼ,D) · p(y|x,D)

synthetic_rows = []
for _ in range(n_samples):
    row = {}
    for feature in feature_order:
        # Predict distribution for this feature given previous features
        dist = tabpfn_model.predict(row, target=feature)
        row[feature] = sample_from(dist)
    synthetic_rows.append(row)
```

**Quality Validation**:
- Compare feature means/stds (within 10% tolerance)
- Compare correlation matrix (Frobenius norm < 0.1)
- KS-test for distribution similarity

**Alternatives Considered**:
- SMOTE: Only for classification, doesn't preserve correlations
- GANs: Require training, poor on small datasets
- Copulas: Complex, less accurate

### 6. Catalog Storage and Search

**Decision**: JSON-based catalog with in-memory indexing

**Rationale**:
- Matches existing session persistence pattern in `intuitiveness`
- No additional database dependency
- Sufficient for 1,000+ datasets (target scale)
- Easy to version control and backup

**Structure**:
```json
{
  "catalog_version": "1.0",
  "datasets": [
    {
      "id": "uuid",
      "name": "Dataset Name",
      "description": "...",
      "domain_tags": ["healthcare", "classification"],
      "file_path": "path/to/data.csv",
      "row_count": 5000,
      "feature_count": 25,
      "usability_score": 85,
      "quality_report_path": "path/to/report.json",
      "created_at": "2025-12-13T10:00:00Z",
      "updated_at": "2025-12-13T10:00:00Z"
    }
  ]
}
```

**Search Implementation**:
- Load catalog into memory at startup
- Filter using pandas-like operations
- Sort by usability_score, name, date
- <2s for 1,000 datasets (in-memory)

**Alternatives Considered**:
- SQLite: Additional dependency, overkill for metadata
- PostgreSQL: Requires server, too complex
- Elasticsearch: Overkill for scale

## Dependencies to Add

```
# requirements.txt additions
tabpfn-client>=0.1.0   # TabPFN cloud API (primary - no GPU needed)
tabpfn>=2.0.0          # TabPFN local (optional - requires GPU)
shap>=0.42.0           # SHAP explanations for interpretability
```

**Installation Commands**:
```bash
# Primary (cloud API - recommended)
pip install --upgrade tabpfn-client shap

# Optional (local with GPU)
pip install tabpfn shap
```

**Authentication Setup** (one-time for tabpfn-client):
```python
import tabpfn_client
token = tabpfn_client.get_access_token()  # Opens browser for auth
# Save token for subsequent sessions
tabpfn_client.set_access_token(token)
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| TabPFN not available on Streamlit Cloud (no GPU) | Use TabPFN API at priorlabs.ai, or fall back to CPU (slower) |
| Large datasets exceed TabPFN limits | Sample 10,000 rows with stratification, inform user |
| SHAP computation slow on many features | Limit to top 20 features, use SHAP sampling |
