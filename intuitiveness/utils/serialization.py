"""
Serialization Utilities Module

Provides base classes and utilities for dataclass serialization,
eliminating repetitive to_dict/from_dict implementations.

Created: 2026-01-09 (Phase 0 - Code Simplification)
Supports: 005-session-persistence, 010-quality-ds-workflow (FR-003, FR-004)
"""

from dataclasses import dataclass, fields, asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Type, TypeVar, Optional, get_type_hints
from uuid import UUID
import json
import io
import pandas as pd


T = TypeVar('T')


# =============================================================================
# SERIALIZABLE DATACLASS MIXIN
# =============================================================================

class SerializableDataclass:
    """
    Mixin for dataclasses that provides automatic to_dict/from_dict.

    Eliminates manual serialization code by using dataclasses.asdict()
    and automatic reconstruction.

    Example:
        >>> @dataclass
        ... class MyModel(SerializableDataclass):
        ...     name: str
        ...     value: int = 0
        >>>
        >>> obj = MyModel(name="test", value=42)
        >>> d = obj.to_dict()
        >>> obj2 = MyModel.from_dict(d)
        >>> obj == obj2
        True

    Handles special types:
    - datetime: Converted to/from ISO format strings
    - UUID: Converted to/from string
    - tuple: Converted to/from list (JSON doesn't have tuples)
    - Nested dataclasses: Recursively serialized
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to dictionary for JSON serialization.

        Special handling for datetime, UUID, and nested dataclasses.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = _serialize_value(value)
        return result

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create dataclass instance from dictionary.

        Attempts to reconstruct special types (datetime, UUID, etc.)
        based on field type hints.
        """
        if data is None:
            raise ValueError(f"Cannot create {cls.__name__} from None")

        # Get type hints for deserialization
        hints = get_type_hints(cls) if hasattr(cls, '__annotations__') else {}

        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                value = data[f.name]
                field_type = hints.get(f.name, type(value))
                kwargs[f.name] = _deserialize_value(value, field_type)
            elif f.default is not f.default_factory:
                # Field has default, skip it
                pass
            elif f.default_factory is not None:
                # Field has default_factory, skip it
                pass
            else:
                # Required field missing
                kwargs[f.name] = None

        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def _serialize_value(value: Any) -> Any:
    """Serialize a single value to JSON-compatible format."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, tuple):
        return list(_serialize_value(v) for v in value)
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, set):
        return list(value)
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        return asdict(value)
    if hasattr(value, '__dict__'):
        # Custom object - try to serialize attributes
        return {k: _serialize_value(v) for k, v in value.__dict__.items()
                if not k.startswith('_')}
    return value


def _deserialize_value(value: Any, target_type: Type) -> Any:
    """Deserialize a value to the target type."""
    if value is None:
        return None

    # Handle Optional types
    origin = getattr(target_type, '__origin__', None)
    if origin is type(None):
        return None

    # Handle datetime
    if target_type == datetime or (hasattr(target_type, '__origin__') and
                                    getattr(target_type, '__args__', (None,))[0] == datetime):
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return value
        return value

    # Handle UUID
    if target_type == UUID:
        if isinstance(value, str):
            return UUID(value)
        return value

    # Handle tuple
    if origin is tuple or target_type == tuple:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return value

    # Handle list
    if origin is list:
        return list(value) if value else []

    # Handle set
    if origin is set:
        return set(value) if value else set()

    # Handle dict
    if origin is dict:
        return dict(value) if value else {}

    # Handle nested dataclasses
    if is_dataclass(target_type) and isinstance(value, dict):
        if hasattr(target_type, 'from_dict'):
            return target_type.from_dict(value)
        return target_type(**value)

    return value


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

class ExportFormat:
    """Export format constants."""
    CSV = "csv"
    JSON = "json"
    PICKLE = "pickle"
    PARQUET = "parquet"


def export_dataframe_to_bytes(
    df: pd.DataFrame,
    format: str = ExportFormat.CSV,
    **kwargs
) -> bytes:
    """
    Export DataFrame to bytes in specified format.

    Consolidates export logic from quality/exporter.py.

    Args:
        df: DataFrame to export
        format: Export format (csv, json, pickle, parquet)
        **kwargs: Additional arguments for the export function

    Returns:
        Bytes representation of the DataFrame

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> bytes_data = export_dataframe_to_bytes(df, format='csv')
        >>> bytes_data.decode('utf-8')
        ',a\\n0,1\\n1,2\\n2,3\\n'
    """
    buffer = io.BytesIO()

    if format == ExportFormat.CSV:
        df.to_csv(buffer, **kwargs)
    elif format == ExportFormat.JSON:
        df.to_json(buffer, **kwargs)
    elif format == ExportFormat.PICKLE:
        df.to_pickle(buffer, **kwargs)
    elif format == ExportFormat.PARQUET:
        df.to_parquet(buffer, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    buffer.seek(0)
    return buffer.getvalue()


def get_mime_type(format: str) -> str:
    """
    Get MIME type for export format.

    Consolidated from quality/exporter.py (line 218).

    Args:
        format: Export format string

    Returns:
        MIME type string
    """
    mime_types = {
        ExportFormat.CSV: "text/csv",
        ExportFormat.JSON: "application/json",
        ExportFormat.PICKLE: "application/octet-stream",
        ExportFormat.PARQUET: "application/octet-stream",
    }
    return mime_types.get(format, "application/octet-stream")


def get_file_extension(format: str) -> str:
    """
    Get file extension for export format.

    Args:
        format: Export format string

    Returns:
        File extension with dot (e.g., '.csv')
    """
    extensions = {
        ExportFormat.CSV: ".csv",
        ExportFormat.JSON: ".json",
        ExportFormat.PICKLE: ".pkl",
        ExportFormat.PARQUET: ".parquet",
    }
    return extensions.get(format, ".dat")


# =============================================================================
# JSON SERIALIZATION HELPERS
# =============================================================================

class DataclassJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that handles dataclasses and special types.

    Example:
        >>> @dataclass
        ... class MyData:
        ...     timestamp: datetime
        >>>
        >>> data = MyData(timestamp=datetime.now())
        >>> json.dumps(data, cls=DataclassJSONEncoder)
    """

    def default(self, obj: Any) -> Any:
        if is_dataclass(obj) and not isinstance(obj, type):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        return super().default(obj)


def to_json(obj: Any, indent: int = 2) -> str:
    """
    Convert any object to JSON string with proper handling.

    Args:
        obj: Object to serialize (dataclass, dict, list, etc.)
        indent: JSON indentation

    Returns:
        JSON string
    """
    return json.dumps(obj, cls=DataclassJSONEncoder, indent=indent)


def from_json(json_str: str) -> Any:
    """
    Parse JSON string to Python object.

    Args:
        json_str: JSON string

    Returns:
        Parsed Python object (dict, list, etc.)
    """
    return json.loads(json_str)


# =============================================================================
# REPORT EXPORT UTILITIES (010-quality-ds-workflow)
# =============================================================================

def generate_python_code_snippet(
    filename: str,
    target_column: Optional[str] = None,
    transformations: Optional[list] = None,
) -> str:
    """
    Generate Python code snippet for loading exported data.

    Implements FR-004 (010-quality-ds-workflow): Copy Python Code.

    Args:
        filename: Name of the exported CSV file
        target_column: Optional target column for train/test split
        transformations: Optional list of applied transformations

    Returns:
        Python code as string
    """
    code_lines = [
        "# Load data exported from Intuitiveness",
        "import pandas as pd",
        "",
        f'df = pd.read_csv("{filename}")',
        f"print(f\"Loaded {{len(df)}} rows, {{len(df.columns)}} columns\")",
    ]

    if transformations:
        code_lines.append("")
        code_lines.append("# Applied transformations:")
        for t in transformations[:5]:  # Limit to first 5
            code_lines.append(f"# - {t}")

    if target_column:
        code_lines.extend([
            "",
            "# Prepare for modeling",
            f"X = df.drop(columns=['{target_column}'])",
            f"y = df['{target_column}']",
            "",
            "# Train/test split",
            "from sklearn.model_selection import train_test_split",
            "X_train, X_test, y_train, y_test = train_test_split(",
            "    X, y, test_size=0.2, random_state=42",
            ")",
            "",
            "print(f\"Training set: {len(X_train)} rows\")",
            "print(f\"Test set: {len(X_test)} rows\")",
        ])

    return "\n".join(code_lines)
