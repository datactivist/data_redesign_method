"""
Quality Dashboard - Assessment Components

Phase 1.3 - Code Simplification (011-code-simplification)
Extracted from quality_dashboard.py

Spec Traceability:
------------------
- 010-quality-ds-workflow: US-1 Step 2 (Run Assessment)

Contains:
- Assessment button with progress tracking
- TabPFN explainer and API cost estimate
"""

import streamlit as st
import pandas as pd
import time
from typing import Optional, Callable

from intuitiveness.ui.alert import info, warning, error
from intuitiveness.ui.quality.state import save_report_to_history
from intuitiveness.ui.quality.utils import SESSION_KEY_QUALITY_DF


def render_assessment_button(
    df: pd.DataFrame,
    target_column: str,
    on_complete: Optional[Callable] = None,
) -> None:
    """
    Render assessment button with progress tracking and TabPFN explainer.

    Spec: 010-quality-ds-workflow US-1 Step 2

    Args:
        df: DataFrame to assess
        target_column: Target column name
        on_complete: Optional callback when assessment completes
    """
    from intuitiveness.quality.assessor import (
        assess_dataset,
        detect_task_type,
        MIN_ROWS_FOR_ASSESSMENT,
    )
    from intuitiveness.quality.tabpfn_wrapper import estimate_api_consumption

    # Validate minimum rows
    if len(df) < MIN_ROWS_FOR_ASSESSMENT:
        warning(
            f"Dataset has only {len(df)} rows. "
            f"Minimum {MIN_ROWS_FOR_ASSESSMENT} rows required for reliable assessment."
        )
        return

    if len(df) > 10000:
        info(
            f"Dataset has {len(df):,} rows. "
            "A representative sample of 5,000 rows will be used for assessment."
        )

    # Detect task type BEFORE calculating estimates
    task_type = detect_task_type(df[target_column]) if target_column in df.columns else "classification"

    # Calculate API consumption estimate
    n_features = len(df.columns) - 1  # Exclude target
    n_unique_values = df[target_column].nunique() if target_column in df.columns else 2
    n_rows = min(len(df), 5000)  # Capped at 5000 for assessment

    api_estimate = estimate_api_consumption(
        n_rows=n_rows,
        n_features=n_features,
        n_classes=n_unique_values,
        task_type=task_type,
    )

    # Educational TabPFN explainer with API estimate
    _render_tabpfn_explainer(api_estimate, task_type)

    # Assessment button
    col1, col2 = st.columns([1, 3])
    with col1:
        run_assessment = st.button(
            "Run Assessment",
            type="primary",
            use_container_width=True,
            key="run_quality_assessment",
        )

    if run_assessment:
        _run_assessment(df, target_column, on_complete)


def _render_tabpfn_explainer(api_estimate, task_type: str) -> None:
    """Render the TabPFN methodology explainer."""
    with st.expander("What is TabPFN? (API usage estimate)", expanded=False):
        st.markdown("""
**TabPFN** (Tabular Prior-data Fitted Network) is a foundation model for tabular data from the
[Nature paper](https://www.nature.com/articles/s41586-024-07544-w) by Hollmann et al. (2024).

**How it works:**
- **Pre-trained on 100 million synthetic datasets** using structural causal models
- **In-Context Learning (ICL)**: Receives your entire dataset as context, predicts in a single forward pass
- **No training on your data**: Unlike traditional ML, TabPFN doesn't train - it learned patterns during pre-training
- **Ensemble of 8 estimators**: Predictions are averaged for robustness
- **5,140x faster** than tuned baselines while matching accuracy

**Optimal for:**
- 10,000 samples (rows)
- 500 features (columns)
- 10 classes (for classification)
        """)

        st.markdown("---")
        st.markdown("**API Cost Estimate for Your Dataset:**")
        st.markdown(f"*Formula*: `max((train_rows + test_rows) x cols x 8, 5000)`")

        # Show API cost breakdown
        col_a, col_b = st.columns(2)
        with col_a:
            # Show task-type-aware info
            if task_type == "regression":
                target_info = f"{api_estimate.n_classes} unique values (continuous -> **regression**)"
            else:
                target_info = f"{api_estimate.n_classes} classes (**classification**)"

            st.markdown(f"""
**Dataset Info:**
- {api_estimate.n_rows:,} rows x {api_estimate.n_features} features
- Target: {target_info}
- {api_estimate.total_cells:,} total cells
            """)
        with col_b:
            st.markdown(f"""
**Cost Breakdown:**
- CV (5-fold): {api_estimate.cv_calls} x {api_estimate.cost_per_cv_call:,} = **{api_estimate.total_cv_cost:,}**
- Feature importance: {api_estimate.feature_importance_calls} calls = **{api_estimate.total_feature_importance_cost:,}**
- SHAP: ~{api_estimate.shap_calls} calls = **{api_estimate.total_shap_cost:,}**
            """)

        st.markdown(f"### Total API Cost: ~{api_estimate.total_api_cost:,} units")

        if not api_estimate.is_optimal:
            st.warning("Warning: " + " | ".join(api_estimate.warnings))
        else:
            st.success("Dataset is within TabPFN optimal limits")


def _run_assessment(
    df: pd.DataFrame,
    target_column: str,
    on_complete: Optional[Callable] = None,
) -> None:
    """Execute the assessment with progress tracking."""
    from intuitiveness.quality.assessor import assess_dataset

    progress_bar = st.progress(0, text="Starting assessment...")
    status_text = st.empty()

    # Track timing for estimated time remaining
    start_time = time.time()
    last_progress = 0.0

    def progress_callback(message: str, progress: float):
        nonlocal last_progress
        elapsed = time.time() - start_time

        # Estimate remaining time based on progress
        if progress > 0.05:  # Need some progress to estimate
            estimated_total = elapsed / progress
            estimated_remaining = estimated_total - elapsed

            if estimated_remaining > 60:
                time_str = f"~{estimated_remaining / 60:.0f} min remaining"
            elif estimated_remaining > 5:
                time_str = f"~{estimated_remaining:.0f}s remaining"
            else:
                time_str = "Almost done..."

            progress_bar.progress(progress, text=f"{message} ({time_str})")
        else:
            progress_bar.progress(progress, text=message)

        last_progress = progress

    try:
        report = assess_dataset(
            df=df,
            target_column=target_column,
            task_type="auto",
            compute_shap=True,  # Enable SHAP for maximum interpretability
            progress_callback=progress_callback,
        )

        # Save to history instead of overwriting
        save_report_to_history(report)
        total_time = time.time() - start_time
        progress_bar.progress(1.0, text=f"Assessment complete! ({total_time:.1f}s)")
        time.sleep(0.8)
        progress_bar.empty()
        status_text.empty()

        if on_complete:
            on_complete(report)

        st.rerun()

    except Exception as e:
        progress_bar.empty()
        error(f"Assessment failed: {e}")
