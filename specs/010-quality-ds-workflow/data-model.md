# Data Model: Data Scientist Co-Pilot

**Feature**: 010-quality-ds-workflow
**Date**: 2025-12-13

## Entity Definitions

### 1. SyntheticBenchmarkReport

Results of train-on-synthetic/test-on-real validation.

```python
@dataclass
class SyntheticBenchmarkReport:
    """Results of synthetic data validation pipeline."""

    # Identification
    dataset_name: str
    target_column: str
    timestamp: datetime

    # Generation parameters
    n_synthetic_samples: int
    generation_method: str  # 'tabpfn' | 'gaussian_copula'
    class_balanced: bool

    # Benchmark results per model
    model_results: List[ModelBenchmarkResult]

    # Aggregate metrics
    mean_transfer_gap: float  # Average across models
    max_transfer_gap: float   # Worst case
    min_transfer_gap: float   # Best case

    # Recommendation
    recommendation: str  # 'safe_to_use' | 'use_with_caution' | 'not_recommended'
    recommendation_reason: str

    @property
    def is_safe(self) -> bool:
        """Transfer gap < 10% is considered safe."""
        return self.mean_transfer_gap < 0.10
```

### 2. ModelBenchmarkResult

Per-model benchmark metrics.

```python
@dataclass
class ModelBenchmarkResult:
    """Benchmark results for a single model."""

    model_name: str  # 'LogisticRegression' | 'RandomForest' | 'XGBoost'

    # Real → Real (baseline)
    real_accuracy: float
    real_f1: float
    real_precision: float
    real_recall: float

    # Synthetic → Real (transfer)
    synthetic_accuracy: float
    synthetic_f1: float
    synthetic_precision: float
    synthetic_recall: float

    # Computed metrics
    @property
    def transfer_gap(self) -> float:
        """Percentage drop from real to synthetic training."""
        if self.real_accuracy == 0:
            return 1.0
        return (self.real_accuracy - self.synthetic_accuracy) / self.real_accuracy

    @property
    def transfer_gap_percent(self) -> str:
        """Human-readable transfer gap."""
        return f"{self.transfer_gap * 100:.1f}%"
```

### 3. TransformationResult

Record of a single transformation applied.

```python
@dataclass
class TransformationResult:
    """Result of applying a single transformation."""

    suggestion: FeatureSuggestion  # From existing feature_engineer.py
    applied_at: datetime

    # Success/failure
    success: bool
    error: Optional[str] = None

    # Accuracy tracking (if target column specified)
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None

    @property
    def accuracy_delta(self) -> Optional[float]:
        """Change in accuracy from this transformation."""
        if self.accuracy_before is None or self.accuracy_after is None:
            return None
        return self.accuracy_after - self.accuracy_before

    @property
    def accuracy_delta_percent(self) -> Optional[str]:
        """Human-readable accuracy change."""
        delta = self.accuracy_delta
        if delta is None:
            return None
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta * 100:.1f}%"
```

### 4. TransformationLog

Complete log of all transformations applied.

```python
@dataclass
class TransformationLog:
    """Log of all transformations applied to a dataset."""

    dataset_name: str
    original_shape: Tuple[int, int]  # (rows, cols)
    final_shape: Tuple[int, int]

    # Individual results
    results: List[TransformationResult]

    # Aggregate metrics
    total_applied: int
    total_failed: int

    # Accuracy tracking
    initial_accuracy: Optional[float] = None
    final_accuracy: Optional[float] = None

    @property
    def total_accuracy_improvement(self) -> Optional[float]:
        """Total accuracy improvement from all transformations."""
        if self.initial_accuracy is None or self.final_accuracy is None:
            return None
        return self.final_accuracy - self.initial_accuracy

    def to_metadata(self) -> Dict[str, Any]:
        """Export as metadata for inclusion in exported files."""
        return {
            'original_shape': self.original_shape,
            'final_shape': self.final_shape,
            'transformations_applied': self.total_applied,
            'accuracy_improvement': self.total_accuracy_improvement,
            'transformations': [
                {
                    'type': r.suggestion.suggestion_type,
                    'column': r.suggestion.column_name,
                    'delta': r.accuracy_delta_percent
                }
                for r in self.results if r.success
            ]
        }
```

### 5. ExportPackage

Bundle of exported data and metadata.

```python
@dataclass
class ExportPackage:
    """Package containing exported dataset and supporting files."""

    # Core data
    dataset: pd.DataFrame
    format: str  # 'csv' | 'pickle' | 'parquet'

    # Metadata
    dataset_name: str
    target_column: Optional[str]
    transformation_log: Optional[TransformationLog]
    quality_report: Optional[QualityReport]  # From existing models.py

    # Code generation
    @property
    def python_snippet(self) -> str:
        """Generate Python code snippet for loading data."""
        filename = f"{self.dataset_name}_clean.{self.format}"
        target = self.target_column or 'target'

        return f'''# Load your modeling-ready data
import pandas as pd

df = pd.read_csv('{filename}')
X = df.drop('{target}', axis=1)
y = df['{target}']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start modeling!
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# print(f"Accuracy: {{model.score(X_test, y_test):.2%}}")
'''

    def export(self) -> bytes:
        """Export dataset in specified format."""
        if self.format == 'csv':
            return self.dataset.to_csv(index=False).encode('utf-8')
        elif self.format == 'pickle':
            import pickle
            return pickle.dumps(self.dataset)
        elif self.format == 'parquet':
            import io
            buffer = io.BytesIO()
            self.dataset.to_parquet(buffer, index=False)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unknown format: {self.format}")
```

### 6. ReadinessIndicator

Traffic light readiness status.

```python
@dataclass
class ReadinessIndicator:
    """Traffic light readiness indicator for dataset quality."""

    status: str  # 'ready' | 'fixable' | 'needs_work'
    color: str   # 'green' | 'yellow' | 'red'
    score: float  # 0-100 usability score

    # Messaging
    title: str
    message: str

    # Fixable details (if status == 'fixable')
    n_fixes_available: int = 0
    estimated_score_after_fixes: float = 0.0

    @classmethod
    def from_score(cls, score: float, n_suggestions: int = 0,
                   estimated_improvement: float = 0.0) -> 'ReadinessIndicator':
        """Create indicator from usability score."""
        if score >= 80:
            return cls(
                status='ready',
                color='green',
                score=score,
                title='🟢 READY FOR MODELING',
                message='Export and start training!'
            )
        elif score >= 60:
            return cls(
                status='fixable',
                color='yellow',
                score=score,
                title='🟡 FIXABLE',
                message=f'{n_suggestions} automated fixes will improve score to {score + estimated_improvement:.0f}',
                n_fixes_available=n_suggestions,
                estimated_score_after_fixes=score + estimated_improvement
            )
        else:
            return cls(
                status='needs_work',
                color='red',
                score=score,
                title='🔴 NEEDS WORK',
                message='Significant data issues. Review recommendations below.'
            )
```

## Entity Relationships

```
QualityReport (existing)
    │
    ├── contains → FeatureProfile[] (existing)
    │
    ├── generates → FeatureSuggestion[] (existing)
    │                    │
    │                    └── applied to → TransformationResult
    │                                          │
    │                                          └── collected in → TransformationLog
    │
    └── produces → ReadinessIndicator


SyntheticBenchmarkReport
    │
    ├── contains → ModelBenchmarkResult[]
    │
    └── uses → synthetic data from SyntheticGenerator (existing)


ExportPackage
    │
    ├── contains → DataFrame (transformed)
    │
    ├── includes → TransformationLog
    │
    └── includes → QualityReport
```

## State Transitions

### Dataset Quality State Machine

```
[Uploaded] ──assess──> [Assessed]
                           │
                           ├── score >= 80 ──> [Ready] ──export──> [Exported]
                           │
                           └── score < 80 ──> [Fixable]
                                                  │
                                                  ├── apply_all ──> [Transformed]
                                                  │                      │
                                                  │                      └── re-assess ──> [Assessed]
                                                  │
                                                  └── export_anyway ──> [Exported]
```

### Synthetic Validation State Machine

```
[Dataset Ready] ──generate_synthetic──> [Synthetic Generated]
                                              │
                                              └── validate ──> [Benchmarked]
                                                                   │
                                                                   ├── gap < 10% ──> [Safe to Use]
                                                                   │
                                                                   └── gap >= 10% ──> [Not Recommended]
```
