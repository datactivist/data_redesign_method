"""
Quality Data Platform - Dataset Exporter

Export transformed datasets with metadata and Python code snippets
for immediate use in Jupyter notebooks.
"""

import io
import logging
from typing import Optional, Literal

import pandas as pd

from intuitiveness.quality.models import (
    ExportPackage,
    TransformationLog,
)

logger = logging.getLogger(__name__)


def generate_python_snippet(
    filename: str,
    target_column: str = "target",
    format: Literal["csv", "pickle", "parquet"] = "csv",
    task_type: Literal["classification", "regression"] = "classification",
) -> str:
    """
    Generate robust Python code snippet for loading exported data.

    The generated code:
    - Auto-detects the best task type (classification vs regression) at runtime
    - Handles TabPFN's 10-class limit for classification
    - Falls back to sklearn if TabPFN is unavailable or fails
    - Provides clear feedback about model selection

    Args:
        filename: Name of exported file.
        target_column: Target column name.
        format: File format.
        task_type: Suggested task type (user can override at runtime).

    Returns:
        Python code snippet as string.
    """
    if format == "csv":
        load_code = f"df = pd.read_csv('{filename}')"
    elif format == "pickle":
        load_code = f"df = pd.read_pickle('{filename}')"
    elif format == "parquet":
        load_code = f"df = pd.read_parquet('{filename}')"
    else:
        load_code = f"df = pd.read_csv('{filename}')"

    # Robust modeling code that handles all edge cases
    model_code = f'''# ═══════════════════════════════════════════════════════════════════════════════
# SMART MODEL SELECTION - Auto-detects best approach for your data
# ═══════════════════════════════════════════════════════════════════════════════

def smart_model_fit(X_train, y_train, X_test, y_test, suggested_task="{task_type}"):
    """
    Automatically selects and fits the best model for your data.
    - Detects if target is continuous (regression) or categorical (classification)
    - Uses TabPFN when possible (fast, accurate, no tuning needed)
    - Falls back to sklearn GradientBoosting when TabPFN can't handle the data
    """
    import numpy as np

    n_unique = y_train.nunique()
    n_total = len(y_train)
    is_numeric = np.issubdtype(y_train.dtype, np.number)

    # Smart task detection
    if suggested_task == "regression":
        task = "regression"
    elif n_unique > 20 and is_numeric:
        # Many unique numeric values → likely continuous → regression
        task = "regression"
        print(f"ℹ️  Target has {{n_unique}} unique numeric values → using REGRESSION")
    elif n_unique <= 10:
        task = "classification"
    else:
        # 10 < unique <= 20: classification but might need sklearn
        task = "classification"

    print(f"📊 Task: {{task.upper()}} | Unique values: {{n_unique}} | Samples: {{n_total}}")
    print("-" * 60)

    # Try TabPFN first, fall back to sklearn
    model = None
    used_tabpfn = False

    if task == "classification":
        # Classification path
        if n_unique > 10:
            print(f"⚠️  {{n_unique}} classes exceeds TabPFN limit (10). Using sklearn.")
        else:
            try:
                from tabpfn import TabPFNClassifier
                model = TabPFNClassifier()
                model.fit(X_train, y_train)
                used_tabpfn = True
                print("✅ Using TabPFN Classifier (same as quality assessment)")
            except ImportError:
                print("⚠️  TabPFN not installed. Install with: pip install tabpfn")
            except Exception as e:
                print(f"⚠️  TabPFN failed: {{e}}")

        if model is None:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            print("✅ Using sklearn GradientBoostingClassifier (robust fallback)")

        # Evaluate
        accuracy = model.score(X_test, y_test)
        print(f"\\n🎯 Accuracy: {{accuracy:.2%}}")

        if used_tabpfn and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            print(f"📈 Confidence range: {{proba.max(axis=1).min():.1%}} - {{proba.max(axis=1).max():.1%}}")

    else:
        # Regression path
        try:
            from tabpfn import TabPFNRegressor
            model = TabPFNRegressor()
            model.fit(X_train, y_train)
            used_tabpfn = True
            print("✅ Using TabPFN Regressor (same as quality assessment)")
        except ImportError:
            print("⚠️  TabPFN not installed. Install with: pip install tabpfn")
        except Exception as e:
            print(f"⚠️  TabPFN failed: {{e}}")

        if model is None:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            print("✅ Using sklearn GradientBoostingRegressor (robust fallback)")

        # Evaluate
        y_pred = model.predict(X_test)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Compatible with all sklearn versions
        print(f"\\n🎯 R² Score: {{r2:.3f}}")
        print(f"📏 RMSE: {{rmse:.3f}}")

    return model

# Run the smart model fitting
model = smart_model_fit(X_train, y_train, X_test, y_test)'''

    return f'''# ═══════════════════════════════════════════════════════════════════════════════
# MODELING-READY DATA - Generated by Data Scientist Co-Pilot
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd

# Load your clean data
{load_code}

# Prepare features and target
X = df.drop('{target_column}', axis=1)
y = df['{target_column}']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📁 Dataset: {{len(df)}} rows, {{len(df.columns)}} columns")
print(f"🎯 Target: '{target_column}'")
print(f"🔀 Split: {{len(X_train)}} train / {{len(X_test)}} test")
print()

{model_code}

# ═══════════════════════════════════════════════════════════════════════════════
# NEXT STEPS:
# - model.predict(X_new)           → Get predictions
# - model.predict_proba(X_new)     → Get probabilities (classification only)
# - model.feature_importances_     → See which features matter (sklearn only)
# ═══════════════════════════════════════════════════════════════════════════════
'''


def export_to_bytes(
    df: pd.DataFrame,
    format: Literal["csv", "pickle", "parquet"] = "csv",
) -> bytes:
    """
    Export DataFrame to bytes in specified format.

    Args:
        df: DataFrame to export.
        format: Export format.

    Returns:
        Binary data.

    Raises:
        ValueError: If format is unknown.
    """
    if format == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif format == "pickle":
        import pickle
        return pickle.dumps(df)
    elif format == "parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()
    else:
        raise ValueError(f"Unknown format: {format}")


def get_mime_type(format: Literal["csv", "pickle", "parquet"]) -> str:
    """Get MIME type for download."""
    mime_types = {
        "csv": "text/csv",
        "pickle": "application/octet-stream",
        "parquet": "application/octet-stream",
    }
    return mime_types.get(format, "application/octet-stream")


def export_dataset(
    df: pd.DataFrame,
    format: Literal["csv", "pickle", "parquet"] = "csv",
    dataset_name: str = "dataset",
    target_column: Optional[str] = None,
    transformation_log: Optional[TransformationLog] = None,
    task_type: Literal["classification", "regression"] = "classification",
) -> ExportPackage:
    """
    Export transformed dataset in specified format with metadata.

    This is the main export function that creates an ExportPackage
    containing all information needed for immediate modeling.

    Args:
        df: Dataset to export.
        format: Export format (csv, pickle, parquet).
        dataset_name: Name for exported file.
        target_column: Target column for Python snippet.
        transformation_log: Log of applied transformations.
        task_type: Task type for appropriate model code generation.

    Returns:
        ExportPackage with data and metadata.
    """
    logger.info(f"Exporting dataset '{dataset_name}' as {format} (task_type={task_type})")

    package = ExportPackage(
        dataset_name=dataset_name,
        format=format,
        target_column=target_column,
        transformation_log=transformation_log,
        row_count=len(df),
        column_count=len(df.columns),
        task_type=task_type,
    )

    logger.info(f"Export package created: {package.filename}")

    return package


def export_with_metadata(
    df: pd.DataFrame,
    format: Literal["csv", "pickle", "parquet"] = "csv",
    dataset_name: str = "dataset",
    target_column: Optional[str] = None,
    transformation_log: Optional[TransformationLog] = None,
    include_metadata_file: bool = True,
    task_type: Literal["classification", "regression"] = "classification",
) -> dict:
    """
    Export dataset with optional separate metadata file.

    Returns a dictionary with binary data and metadata.

    Args:
        df: Dataset to export.
        format: Export format.
        dataset_name: Name for exported file.
        target_column: Target column for Python snippet.
        transformation_log: Log of applied transformations.
        include_metadata_file: Whether to include separate metadata JSON.
        task_type: Task type for appropriate model code generation.

    Returns:
        Dictionary with 'data', 'metadata', 'python_snippet', 'filename'.
    """
    package = export_dataset(
        df=df,
        format=format,
        dataset_name=dataset_name,
        target_column=target_column,
        transformation_log=transformation_log,
        task_type=task_type,
    )

    result = {
        "data": export_to_bytes(df, format),
        "filename": package.filename,
        "python_snippet": package.python_snippet,
        "mime_type": get_mime_type(format),
    }

    if include_metadata_file and transformation_log:
        import json
        result["metadata"] = json.dumps(transformation_log.to_metadata(), indent=2)
        result["metadata_filename"] = f"{dataset_name}_metadata.json"

    return result
