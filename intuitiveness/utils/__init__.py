"""
Utilities Package

Consolidated utilities for the intuitiveness package.

Phase 0 (Code Simplification) - Created 2026-01-09
"""

from .common import (
    # Alert utilities
    format_alert_message,

    # HTML/CSS builders
    build_html_card,
    build_html_badge,

    # Type detection (consolidated from assessor.py, benchmark.py)
    detect_task_type,
    detect_feature_type,

    # Text formatting (consolidated from datagouv_client.py)
    format_filesize,
    parse_iso_datetime,
    truncate_text,

    # Color utilities (consolidated from quality_dashboard.py, report.py)
    score_to_color,
    delta_to_color,

    # Validation
    is_valid_dataframe,

    # Constants
    MIN_ROWS_FOR_ASSESSMENT,
    MAX_ROWS_FOR_TABPFN,
    MAX_FEATURES_FOR_TABPFN,
    HIGH_CARDINALITY_THRESHOLD,
    TRAFFIC_LIGHT_GREEN_THRESHOLD,
    TRAFFIC_LIGHT_YELLOW_THRESHOLD,
)

__all__ = [
    # Alert utilities
    'format_alert_message',

    # HTML/CSS builders
    'build_html_card',
    'build_html_badge',

    # Type detection
    'detect_task_type',
    'detect_feature_type',

    # Text formatting
    'format_filesize',
    'parse_iso_datetime',
    'truncate_text',

    # Color utilities
    'score_to_color',
    'delta_to_color',

    # Validation
    'is_valid_dataframe',

    # Constants
    'MIN_ROWS_FOR_ASSESSMENT',
    'MAX_ROWS_FOR_TABPFN',
    'MAX_FEATURES_FOR_TABPFN',
    'HIGH_CARDINALITY_THRESHOLD',
    'TRAFFIC_LIGHT_GREEN_THRESHOLD',
    'TRAFFIC_LIGHT_YELLOW_THRESHOLD',

    # Session state manager
    'SessionStateKeys',
    'SessionStateManager',
    'session',
    'init_session_state',

    # Serialization utilities
    'SerializableDataclass',
    'ExportFormat',
    'export_dataframe_to_bytes',
    'get_mime_type',
    'get_file_extension',
    'DataclassJSONEncoder',
    'to_json',
    'from_json',
    'generate_python_code_snippet',
]

# Session state manager
from .session_manager import (
    SessionStateKeys,
    SessionStateManager,
    session,
    init_session_state,
)

# Serialization utilities
from .serialization import (
    SerializableDataclass,
    ExportFormat,
    export_dataframe_to_bytes,
    get_mime_type,
    get_file_extension,
    DataclassJSONEncoder,
    to_json,
    from_json,
    generate_python_code_snippet,
)
