"""
Quality Dashboard - File Upload Components

Phase 1.3 - Code Simplification (011-code-simplification)
Extracted from quality_dashboard.py

Spec Traceability:
------------------
- 010-quality-ds-workflow: US-1 Step 1 (Upload CSV)

Contains:
- File upload component with auto-delimiter detection
- Target column selection
"""

import streamlit as st
import pandas as pd
import csv
from typing import Optional

from intuitiveness.ui.alert import error
from intuitiveness.ui.quality.utils import (
    SESSION_KEY_QUALITY_DF,
    SESSION_KEY_QUALITY_FILE_NAME,
)


def render_file_upload() -> Optional[pd.DataFrame]:
    """
    Render file upload component with auto-delimiter detection.

    Spec: 010-quality-ds-workflow US-1 Step 1

    Features:
    - Auto-detects delimiter (comma, semicolon, tab, pipe)
    - Handles encoding issues gracefully
    - Stores uploaded DataFrame in session state

    Returns:
        Uploaded DataFrame or None
    """
    uploaded_file = st.file_uploader(
        "Upload a CSV file for quality assessment",
        type=["csv"],
        help="Upload a tabular dataset (50-10,000 rows recommended)",
        key="quality_file_uploader",
    )

    if uploaded_file is not None:
        try:
            # Auto-detect delimiter (handles comma, semicolon, tab, etc.)
            # First, try to sniff the delimiter from the first few lines
            sample = uploaded_file.read(8192).decode('utf-8', errors='replace')
            uploaded_file.seek(0)  # Reset file position

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
                sep = dialect.delimiter
            except csv.Error:
                # Fallback: check if semicolon is more common than comma
                if sample.count(';') > sample.count(','):
                    sep = ';'
                else:
                    sep = ','

            df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8', on_bad_lines='warn')
            st.session_state[SESSION_KEY_QUALITY_DF] = df
            st.session_state[SESSION_KEY_QUALITY_FILE_NAME] = uploaded_file.name
            return df
        except Exception as e:
            error(f"Failed to read CSV file: {e}")
            return None

    return st.session_state.get(SESSION_KEY_QUALITY_DF)


def render_target_selection(df: pd.DataFrame) -> Optional[str]:
    """
    Render target column selection dropdown.

    Spec: 010-quality-ds-workflow US-1 Step 1

    Args:
        df: DataFrame to select target from

    Returns:
        Selected target column name
    """
    columns = list(df.columns)

    # Try to guess a reasonable default
    default_idx = 0
    for i, col in enumerate(columns):
        col_lower = col.lower()
        if any(word in col_lower for word in ["target", "label", "class", "y", "outcome"]):
            default_idx = i
            break

    target = st.selectbox(
        "Select target column",
        options=columns,
        index=default_idx,
        help="The column you want to predict (for classification or regression)",
        key="quality_target_column",
    )

    return target
