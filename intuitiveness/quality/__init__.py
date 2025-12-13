"""
Quality Data Platform - Quality Assessment Module

This module provides TabPFN-based dataset quality assessment, feature engineering
suggestions, anomaly detection, and synthetic data generation.

Primary Components:
- assessor: Dataset quality assessment with usability scores
- feature_engineer: Feature engineering suggestions
- anomaly_detector: Density-based anomaly detection
- synthetic_generator: Synthetic data generation
- report: Quality report generation and export
"""

from intuitiveness.quality.models import (
    QualityReport,
    FeatureProfile,
    FeatureSuggestion,
    AnomalyRecord,
    SyntheticDataMetrics,
    # New models for 010-quality-ds-workflow
    SyntheticBenchmarkReport,
    ModelBenchmarkResult,
    TransformationResult,
    TransformationLog,
    ReadinessIndicator,
    ExportPackage,
)

from intuitiveness.quality.assessor import (
    assess_dataset,
    compute_usability_score,
    apply_all_suggestions,
    get_readiness_indicator,
    quick_benchmark,
)

from intuitiveness.quality.feature_engineer import (
    suggest_features,
    apply_suggestion,
)

from intuitiveness.quality.anomaly_detector import (
    detect_anomalies,
    explain_anomaly,
    get_anomaly_summary,
)

from intuitiveness.quality.synthetic_generator import (
    generate_synthetic,
    validate_synthetic,
    check_tabpfn_auth,
    get_synthetic_summary,
)

from intuitiveness.quality.benchmark import (
    benchmark_synthetic,
    generate_balanced_synthetic,
    generate_targeted_synthetic,
)

from intuitiveness.quality.exporter import (
    export_dataset,
    export_to_bytes,
    export_with_metadata,
    generate_python_snippet,
    get_mime_type,
)

__all__ = [
    # Models
    "QualityReport",
    "FeatureProfile",
    "FeatureSuggestion",
    "AnomalyRecord",
    "SyntheticDataMetrics",
    # New models for 010-quality-ds-workflow
    "SyntheticBenchmarkReport",
    "ModelBenchmarkResult",
    "TransformationResult",
    "TransformationLog",
    "ReadinessIndicator",
    "ExportPackage",
    # Assessment functions
    "assess_dataset",
    "compute_usability_score",
    "apply_all_suggestions",
    "get_readiness_indicator",
    "quick_benchmark",
    # Feature engineering functions
    "suggest_features",
    "apply_suggestion",
    # Anomaly detection functions
    "detect_anomalies",
    "explain_anomaly",
    "get_anomaly_summary",
    # Synthetic data functions
    "generate_synthetic",
    "validate_synthetic",
    "check_tabpfn_auth",
    "get_synthetic_summary",
    # Benchmark functions (new)
    "benchmark_synthetic",
    "generate_balanced_synthetic",
    "generate_targeted_synthetic",
    # Export functions (new)
    "export_dataset",
    "export_to_bytes",
    "export_with_metadata",
    "generate_python_snippet",
    "get_mime_type",
]
