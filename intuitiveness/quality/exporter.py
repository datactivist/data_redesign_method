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
) -> str:
    """
    Generate Python code snippet for loading exported data.

    Args:
        filename: Name of exported file.
        target_column: Target column name.
        format: File format.

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

    return f'''# Load your modeling-ready data
import pandas as pd

{load_code}
X = df.drop('{target_column}', axis=1)
y = df['{target_column}']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start modeling!
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# print(f"Accuracy: {{model.score(X_test, y_test):.2%}}")
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

    Returns:
        ExportPackage with data and metadata.
    """
    logger.info(f"Exporting dataset '{dataset_name}' as {format}")

    package = ExportPackage(
        dataset_name=dataset_name,
        format=format,
        target_column=target_column,
        transformation_log=transformation_log,
        row_count=len(df),
        column_count=len(df.columns),
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

    Returns:
        Dictionary with 'data', 'metadata', 'python_snippet', 'filename'.
    """
    package = export_dataset(
        df=df,
        format=format,
        dataset_name=dataset_name,
        target_column=target_column,
        transformation_log=transformation_log,
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
