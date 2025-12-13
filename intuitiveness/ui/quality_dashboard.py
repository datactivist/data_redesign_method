"""
Quality Data Platform - Quality Dashboard UI

Streamlit UI component for dataset quality assessment, including:
- File upload
- Assessment trigger with progress
- Quality report display
- Report download
"""

import streamlit as st
import pandas as pd
import io
import json
import time
from pathlib import Path
from typing import Optional, Callable

from intuitiveness.ui.layout import card, spacer
from intuitiveness.ui.header import render_page_header, render_section_header
from intuitiveness.ui.metric_card import render_metric_card, render_metric_card_row
from intuitiveness.ui.alert import info, success, warning, error
from intuitiveness.ui.button import primary_button, secondary_button

# Session state keys
SESSION_KEY_QUALITY_REPORT = "quality_report"
SESSION_KEY_QUALITY_DF = "quality_df"
SESSION_KEY_QUALITY_FILE_NAME = "quality_file_name"
SESSION_KEY_ASSESSMENT_PROGRESS = "assessment_progress"
SESSION_KEY_APPLIED_SUGGESTIONS = "applied_suggestions"
# New session state keys for 010-quality-ds-workflow
SESSION_KEY_TRANSFORMED_DF = "transformed_df"
SESSION_KEY_TRANSFORMATION_LOG = "transformation_log"
SESSION_KEY_BENCHMARK_REPORT = "benchmark_report"
SESSION_KEY_EXPORT_FORMAT = "export_format"
# P0 FIX: Report history for versioning (Mx. Context Keeper)
SESSION_KEY_QUALITY_REPORTS_HISTORY = "quality_reports_history"
SESSION_KEY_CURRENT_REPORT_INDEX = "current_report_index"


def _save_report_to_history(report) -> None:
    """
    P0 FIX: Save report to history instead of overwriting.

    This preserves the original assessment so users can compare
    before/after quality scores across transformation iterations.
    """
    if SESSION_KEY_QUALITY_REPORTS_HISTORY not in st.session_state:
        st.session_state[SESSION_KEY_QUALITY_REPORTS_HISTORY] = []

    # Append to history
    st.session_state[SESSION_KEY_QUALITY_REPORTS_HISTORY].append(report)
    # Update current index
    st.session_state[SESSION_KEY_CURRENT_REPORT_INDEX] = (
        len(st.session_state[SESSION_KEY_QUALITY_REPORTS_HISTORY]) - 1
    )
    # Also set current report (for backward compatibility)
    st.session_state[SESSION_KEY_QUALITY_REPORT] = report


def _get_initial_report():
    """Get the initial (first) quality report from history."""
    history = st.session_state.get(SESSION_KEY_QUALITY_REPORTS_HISTORY, [])
    return history[0] if history else None


def _get_current_report():
    """Get the current (latest) quality report."""
    return st.session_state.get(SESSION_KEY_QUALITY_REPORT)


def _clear_report_history() -> None:
    """Clear all report history (for starting fresh)."""
    st.session_state.pop(SESSION_KEY_QUALITY_REPORTS_HISTORY, None)
    st.session_state.pop(SESSION_KEY_CURRENT_REPORT_INDEX, None)
    st.session_state.pop(SESSION_KEY_QUALITY_REPORT, None)


def _score_color(score: float) -> str:
    """Get color based on score value."""
    if score >= 80:
        return "#22c55e"  # green
    elif score >= 60:
        return "#eab308"  # yellow
    elif score >= 40:
        return "#f97316"  # orange
    else:
        return "#ef4444"  # red


def _score_label(score: float) -> str:
    """Get label based on score value."""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    else:
        return "Poor"


def render_file_upload() -> Optional[pd.DataFrame]:
    """
    Render file upload component.

    Returns:
        Uploaded DataFrame or None.
    """
    uploaded_file = st.file_uploader(
        "Upload a CSV file for quality assessment",
        type=["csv"],
        help="Upload a tabular dataset (50-10,000 rows recommended)",
        key="quality_file_uploader",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state[SESSION_KEY_QUALITY_DF] = df
            st.session_state[SESSION_KEY_QUALITY_FILE_NAME] = uploaded_file.name
            return df
        except Exception as e:
            error(f"Failed to read CSV file: {e}")
            return None

    return st.session_state.get(SESSION_KEY_QUALITY_DF)


def render_target_selection(df: pd.DataFrame) -> Optional[str]:
    """
    Render target column selection.

    Args:
        df: DataFrame to select target from.

    Returns:
        Selected target column name.
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


def render_assessment_button(
    df: pd.DataFrame,
    target_column: str,
    on_complete: Optional[Callable] = None,
) -> None:
    """
    Render assessment button with progress.

    Args:
        df: DataFrame to assess.
        target_column: Target column name.
        on_complete: Optional callback when assessment completes.
    """
    from intuitiveness.quality.assessor import (
        assess_dataset,
        MIN_ROWS_FOR_ASSESSMENT,
    )

    # Validate
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

    col1, col2 = st.columns([1, 3])
    with col1:
        run_assessment = st.button(
            "🔍 Run Assessment",
            type="primary",
            use_container_width=True,
            key="run_quality_assessment",
        )

    if run_assessment:
        progress_bar = st.progress(0, text="Starting assessment...")

        def progress_callback(message: str, progress: float):
            progress_bar.progress(progress, text=message)

        try:
            report = assess_dataset(
                df=df,
                target_column=target_column,
                task_type="auto",
                compute_shap=True,  # Enable SHAP for maximum interpretability
                progress_callback=progress_callback,
            )

            # P0 FIX: Save to history instead of overwriting
            _save_report_to_history(report)
            progress_bar.progress(1.0, text="Assessment complete!")
            time.sleep(0.5)
            progress_bar.empty()

            if on_complete:
                on_complete(report)

            st.rerun()

        except Exception as e:
            progress_bar.empty()
            error(f"Assessment failed: {e}")


def render_feature_suggestions(report) -> None:
    """
    Render feature engineering suggestions section.

    Args:
        report: QualityReport instance.
    """
    from intuitiveness.quality.feature_engineer import suggest_features, apply_suggestion

    render_section_header(
        "Feature Engineering Suggestions",
        "Recommendations to improve your dataset's quality"
    )

    # Get the current DataFrame
    df = st.session_state.get(SESSION_KEY_QUALITY_DF)
    if df is None:
        info("Upload a dataset to get feature engineering suggestions.")
        return

    # Generate suggestions
    suggestions = suggest_features(report, df=df, max_suggestions=5)

    if not suggestions:
        success("No suggestions - your dataset looks well-prepared!")
        return

    # Track applied suggestions
    applied = st.session_state.get(SESSION_KEY_APPLIED_SUGGESTIONS, set())

    for i, suggestion in enumerate(suggestions):
        suggestion_key = f"{suggestion.suggestion_type}_{'-'.join(suggestion.target_features)}"
        is_applied = suggestion_key in applied

        # Suggestion card
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 2])

            with col1:
                # Type badge
                type_colors = {
                    "remove": "#ef4444",
                    "transform": "#3b82f6",
                    "combine": "#8b5cf6",
                }
                color = type_colors.get(suggestion.suggestion_type, "#64748b")
                st.markdown(
                    f"""
                    <div style="
                        background: {color}20;
                        color: {color};
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        font-weight: 600;
                        text-transform: uppercase;
                        text-align: center;
                    ">
                        {suggestion.suggestion_type}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 4px;">
                        <strong>{', '.join(suggestion.target_features)}</strong>
                    </div>
                    <div style="color: #64748b; font-size: 14px;">
                        {suggestion.description}
                    </div>
                    <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">
                        Expected impact: <strong>+{suggestion.expected_impact:.1f}</strong> pts &middot;
                        Confidence: <strong>{suggestion.confidence:.0%}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:
                if is_applied:
                    st.markdown(
                        '<div style="color: #22c55e; font-weight: 600;">✓ Applied</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(
                        "Apply",
                        key=f"apply_suggestion_{i}",
                        use_container_width=True,
                    ):
                        # Apply the suggestion
                        try:
                            new_df = apply_suggestion(df, suggestion)
                            st.session_state[SESSION_KEY_QUALITY_DF] = new_df

                            # Mark as applied
                            if SESSION_KEY_APPLIED_SUGGESTIONS not in st.session_state:
                                st.session_state[SESSION_KEY_APPLIED_SUGGESTIONS] = set()
                            st.session_state[SESSION_KEY_APPLIED_SUGGESTIONS].add(suggestion_key)

                            # Clear the report to force re-assessment
                            st.session_state.pop(SESSION_KEY_QUALITY_REPORT, None)

                            success(f"Applied suggestion: {suggestion.suggestion_type} on {', '.join(suggestion.target_features)}")
                            st.rerun()
                        except Exception as e:
                            error(f"Failed to apply suggestion: {e}")

            st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # Re-assess button
    if applied:
        spacer(8)
        if st.button("🔄 Re-assess with Changes", use_container_width=True):
            st.session_state.pop(SESSION_KEY_QUALITY_REPORT, None)
            st.rerun()


# ============================================================================
# NEW UI COMPONENTS FOR 010-quality-ds-workflow
# Data Scientist Co-Pilot Feature
# ============================================================================


def render_readiness_indicator(report) -> None:
    """
    Display traffic light readiness indicator.

    Provides instant go/no-go visual for data scientists.

    Args:
        report: QualityReport instance.
    """
    from intuitiveness.quality.assessor import get_readiness_indicator
    from intuitiveness.quality.feature_engineer import suggest_features

    df = st.session_state.get(SESSION_KEY_QUALITY_DF)
    suggestions = suggest_features(report, df=df, max_suggestions=10) if df is not None else []

    # Estimate improvement from suggestions
    estimated_improvement = sum(s.expected_impact for s in suggestions)

    indicator = get_readiness_indicator(
        score=report.usability_score,
        n_suggestions=len(suggestions),
        estimated_improvement=estimated_improvement,
    )

    # Color mapping
    colors = {
        "green": "#22c55e",
        "yellow": "#eab308",
        "red": "#ef4444",
    }
    bg_colors = {
        "green": "#dcfce7",
        "yellow": "#fef3c7",
        "red": "#fee2e2",
    }

    color = colors.get(indicator.color, "#64748b")
    bg_color = bg_colors.get(indicator.color, "#f1f5f9")

    # Emoji mapping
    emojis = {
        "ready": "🟢",
        "fixable": "🟡",
        "needs_work": "🔴",
    }
    emoji = emojis.get(indicator.status, "⚪")

    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            border: 3px solid {color};
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin: 16px 0;
        ">
            <div style="font-size: 48px; margin-bottom: 8px;">{emoji}</div>
            <div style="
                font-size: 24px;
                font-weight: bold;
                color: {color};
                margin-bottom: 8px;
            ">
                {indicator.title}
            </div>
            <div style="
                font-size: 16px;
                color: #475569;
            ">
                {indicator.message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_apply_all_button(report) -> None:
    """
    Render one-click apply all suggestions button.

    Args:
        report: QualityReport instance.
    """
    from intuitiveness.quality.assessor import apply_all_suggestions
    from intuitiveness.quality.feature_engineer import suggest_features

    df = st.session_state.get(SESSION_KEY_QUALITY_DF)
    if df is None:
        return

    # Get suggestions
    suggestions = suggest_features(report, df=df, max_suggestions=10)

    if not suggestions:
        return

    render_section_header(
        "One-Click Fix",
        f"Apply all {len(suggestions)} suggested transformations"
    )

    # Show preview of what will be applied
    st.markdown("**Transformations to apply:**")
    for i, s in enumerate(suggestions[:5], 1):
        impact_color = "#22c55e" if s.expected_impact > 0 else "#ef4444"
        st.markdown(
            f"- **{s.suggestion_type.title()}** on `{', '.join(s.target_features)}` "
            f"(<span style='color: {impact_color}'>+{s.expected_impact:.1f} pts</span>)",
            unsafe_allow_html=True,
        )
    if len(suggestions) > 5:
        st.markdown(f"- *...and {len(suggestions) - 5} more*")

    spacer(12)

    col1, col2 = st.columns([1, 3])
    with col1:
        apply_button = st.button(
            "⚡ Apply All Suggestions",
            type="primary",
            use_container_width=True,
            key="apply_all_suggestions_button",
        )

    if apply_button:
        with st.spinner("Applying transformations..."):
            try:
                transformed_df, log = apply_all_suggestions(
                    df=df,
                    suggestions=suggestions,
                    target_column=report.target_column,
                )

                log.dataset_name = st.session_state.get(SESSION_KEY_QUALITY_FILE_NAME, "dataset")

                # Save to session state
                st.session_state[SESSION_KEY_TRANSFORMED_DF] = transformed_df
                st.session_state[SESSION_KEY_TRANSFORMATION_LOG] = log
                st.session_state[SESSION_KEY_QUALITY_DF] = transformed_df

                # Clear old report to trigger re-assessment
                st.session_state.pop(SESSION_KEY_QUALITY_REPORT, None)

                success(
                    f"Applied {log.total_applied} transformations! "
                    f"Accuracy improved by {log.total_accuracy_improvement:.1%}" if log.total_accuracy_improvement else
                    f"Applied {log.total_applied} transformations!"
                )
                st.rerun()

            except Exception as e:
                error(f"Failed to apply suggestions: {e}")


def render_before_after_comparison() -> None:
    """
    Display before/after accuracy comparison.

    Shows ROI of each transformation.
    """
    log = st.session_state.get(SESSION_KEY_TRANSFORMATION_LOG)
    if log is None:
        return

    render_section_header(
        "Transformation Impact",
        "Before/after accuracy for each change"
    )

    # Summary metrics
    if log.total_accuracy_improvement is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            improvement_color = "#22c55e" if log.total_accuracy_improvement >= 0 else "#ef4444"
            st.markdown(
                f"""
                <div style="text-align: center; padding: 16px; background: #f8fafc; border-radius: 8px;">
                    <div style="font-size: 28px; font-weight: bold; color: {improvement_color};">
                        {log.total_accuracy_improvement:+.1%}
                    </div>
                    <div style="color: #64748b; font-size: 14px;">Total Improvement</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 16px; background: #f8fafc; border-radius: 8px;">
                    <div style="font-size: 28px; font-weight: bold; color: #3b82f6;">
                        {log.total_applied}
                    </div>
                    <div style="color: #64748b; font-size: 14px;">Transformations Applied</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 16px; background: #f8fafc; border-radius: 8px;">
                    <div style="font-size: 28px; font-weight: bold; color: #64748b;">
                        {log.final_shape[1]}
                    </div>
                    <div style="color: #64748b; font-size: 14px;">Final Features</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    spacer(16)

    # Per-transformation breakdown
    if log.results:
        st.markdown("**Per-Transformation Breakdown:**")
        breakdown_data = []
        for r in log.results:
            delta = r.accuracy_delta_percent or "N/A"
            status = "✓" if r.success else "✗"
            breakdown_data.append({
                "Status": status,
                "Type": r.suggestion_type.title(),
                "Columns": ", ".join(r.target_features),
                "Impact": delta,
            })

        st.dataframe(
            pd.DataFrame(breakdown_data),
            use_container_width=True,
            hide_index=True,
        )


def render_export_section(report) -> None:
    """
    Render export controls with format selection and code snippet.

    Args:
        report: QualityReport instance.
    """
    from intuitiveness.quality.exporter import export_dataset, export_to_bytes, get_mime_type

    render_section_header(
        "Export & Go",
        "Download your modeling-ready data with Python code snippet"
    )

    # Get the current (possibly transformed) DataFrame
    df = st.session_state.get(SESSION_KEY_TRANSFORMED_DF) or st.session_state.get(SESSION_KEY_QUALITY_DF)
    log = st.session_state.get(SESSION_KEY_TRANSFORMATION_LOG)
    file_name = st.session_state.get(SESSION_KEY_QUALITY_FILE_NAME, "dataset")
    dataset_name = Path(file_name).stem if file_name else "dataset"

    if df is None:
        info("No dataset available to export.")
        return

    # Warning if exporting without applying suggestions
    if log is None:
        from intuitiveness.quality.feature_engineer import suggest_features
        suggestions = suggest_features(report, df=df, max_suggestions=5)
        if suggestions:
            warning(
                f"You have {len(suggestions)} unapplied suggestions. "
                "Consider clicking 'Apply All Suggestions' first for better results."
            )

    # Format selection and download
    col1, col2 = st.columns([1, 2])

    with col1:
        format_option = st.selectbox(
            "Export Format",
            options=["csv", "parquet", "pickle"],
            index=0,
            help="CSV is universal, Parquet is efficient, Pickle preserves dtypes",
            key="export_format_select",
        )

    with col2:
        spacer(28)  # Align with selectbox
        package = export_dataset(
            df=df,
            format=format_option,
            dataset_name=dataset_name,
            target_column=report.target_column,
            transformation_log=log,
        )

        export_data = export_to_bytes(df, format_option)
        st.download_button(
            label=f"📥 Download {package.filename}",
            data=export_data,
            file_name=package.filename,
            mime=get_mime_type(format_option),
            use_container_width=True,
            key="export_download_button",
        )

    spacer(16)

    # Python code snippet
    st.markdown("**Python Code Snippet:**")
    st.markdown("*Copy this to your Jupyter notebook to start modeling immediately*")

    st.code(package.python_snippet, language="python")

    # Copy button hint
    st.markdown(
        '<div style="color: #94a3b8; font-size: 13px;">💡 Hover over the code and click the copy icon in the top-right corner</div>',
        unsafe_allow_html=True,
    )


def render_benchmark_section(report) -> None:
    """
    Render synthetic data validation benchmark UI.

    Proves synthetic data quality with train-on-synthetic/test-on-real methodology.

    Args:
        report: QualityReport instance.
    """
    from intuitiveness.quality.benchmark import benchmark_synthetic, generate_balanced_synthetic

    render_section_header(
        "Synthetic Data Validation",
        "Prove your synthetic data works before using it"
    )

    df = st.session_state.get(SESSION_KEY_QUALITY_DF)
    if df is None:
        info("Upload a dataset to validate synthetic data.")
        return

    # Show class distribution
    target_col = report.target_column
    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts()
        st.markdown("**Current Class Distribution:**")

        # Display as horizontal bar
        for cls, count in class_counts.items():
            pct = count / len(df) * 100
            st.markdown(
                f"- `{cls}`: {count:,} samples ({pct:.1f}%)"
            )

        # Check for imbalance
        max_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else 100
        if max_ratio > 3:
            warning(f"Class imbalance detected (ratio: {max_ratio:.1f}:1). Consider balanced synthetic generation.")

    spacer(12)

    # Benchmark controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        n_synthetic = st.slider(
            "Synthetic Samples",
            min_value=100,
            max_value=min(5000, len(df) * 3),
            value=min(1000, len(df)),
            step=100,
            help="Number of synthetic samples to generate",
            key="benchmark_n_synthetic",
        )

    with col2:
        class_balanced = st.checkbox(
            "Class Balanced",
            value=True,
            help="Generate equal samples per class",
            key="benchmark_class_balanced",
        )

    with col3:
        spacer(28)
        run_benchmark = st.button(
            "🧪 Validate",
            use_container_width=True,
            key="run_benchmark_button",
        )

    if run_benchmark:
        with st.spinner("Running synthetic validation benchmark..."):
            try:
                benchmark_report = benchmark_synthetic(
                    df=df,
                    target_column=target_col,
                    n_synthetic=n_synthetic,
                    class_balanced=class_balanced,
                    dataset_name=st.session_state.get(SESSION_KEY_QUALITY_FILE_NAME, "dataset"),
                )
                st.session_state[SESSION_KEY_BENCHMARK_REPORT] = benchmark_report
            except Exception as e:
                error(f"Benchmark failed: {e}")
                return

    # Display benchmark results
    benchmark_report = st.session_state.get(SESSION_KEY_BENCHMARK_REPORT)
    if benchmark_report is None:
        info("Click 'Validate' to run the synthetic data benchmark.")
        return

    spacer(16)

    # Recommendation indicator
    rec_colors = {
        "safe_to_use": "#22c55e",
        "use_with_caution": "#eab308",
        "not_recommended": "#ef4444",
    }
    rec_emojis = {
        "safe_to_use": "✅",
        "use_with_caution": "⚠️",
        "not_recommended": "❌",
    }
    rec_color = rec_colors.get(benchmark_report.recommendation, "#64748b")
    rec_emoji = rec_emojis.get(benchmark_report.recommendation, "❓")

    st.markdown(
        f"""
        <div style="
            background: {rec_color}20;
            border-left: 4px solid {rec_color};
            padding: 16px;
            border-radius: 0 8px 8px 0;
            margin: 16px 0;
        ">
            <div style="font-size: 18px; font-weight: bold; color: {rec_color};">
                {rec_emoji} {benchmark_report.recommendation.replace('_', ' ').title()}
            </div>
            <div style="color: #475569; margin-top: 8px;">
                {benchmark_report.recommendation_reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        gap_color = "#22c55e" if benchmark_report.mean_transfer_gap < 0.10 else (
            "#eab308" if benchmark_report.mean_transfer_gap < 0.15 else "#ef4444"
        )
        st.markdown(
            f"""
            <div style="text-align: center; padding: 16px; background: #f8fafc; border-radius: 8px;">
                <div style="font-size: 28px; font-weight: bold; color: {gap_color};">
                    {benchmark_report.mean_transfer_gap:.1%}
                </div>
                <div style="color: #64748b; font-size: 14px;">Mean Transfer Gap</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 16px; background: #f8fafc; border-radius: 8px;">
                <div style="font-size: 28px; font-weight: bold; color: #3b82f6;">
                    {benchmark_report.n_synthetic_samples:,}
                </div>
                <div style="color: #64748b; font-size: 14px;">Synthetic Samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 16px; background: #f8fafc; border-radius: 8px;">
                <div style="font-size: 28px; font-weight: bold; color: #64748b;">
                    {len(benchmark_report.model_results)}
                </div>
                <div style="color: #64748b; font-size: 14px;">Models Tested</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    spacer(16)

    # Per-model results table
    if benchmark_report.model_results:
        st.markdown("**Per-Model Results:**")
        model_data = []
        for r in benchmark_report.model_results:
            model_data.append({
                "Model": r.model_name,
                "Real→Real Acc": f"{r.real_accuracy:.1%}",
                "Synth→Real Acc": f"{r.synthetic_accuracy:.1%}",
                "Transfer Gap": r.transfer_gap_percent,
            })

        st.dataframe(
            pd.DataFrame(model_data),
            use_container_width=True,
            hide_index=True,
        )


def render_quality_report(report) -> None:
    """
    Render the quality report display.

    Args:
        report: QualityReport instance.
    """
    from intuitiveness.quality.report import (
        generate_report_summary,
        export_report_json,
        export_report_html,
        get_score_interpretation,
    )

    # TRAFFIC LIGHT INDICATOR - First thing users see
    render_readiness_indicator(report)

    spacer(16)

    # Main score card
    with card():
        col1, col2 = st.columns([1, 2])

        with col1:
            score = report.usability_score
            st.markdown(
                f"""
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 14px; color: #64748b; margin-bottom: 8px;">
                        USABILITY SCORE
                    </div>
                    <div style="font-size: 64px; font-weight: bold; color: {_score_color(score)};">
                        {score:.0f}
                    </div>
                    <div style="font-size: 18px; color: {_score_color(score)};">
                        {_score_label(score)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="padding: 10px 0;">
                    <p style="color: #475569; line-height: 1.6;">
                        {get_score_interpretation(score)}
                    </p>
                    <p style="color: #94a3b8; font-size: 14px; margin-top: 12px;">
                        📊 <strong>{report.row_count:,}</strong> rows &middot;
                        📋 <strong>{report.feature_count}</strong> features &middot;
                        🎯 <strong>{report.task_type.title()}</strong> &middot;
                        ⏱️ <strong>{report.assessment_time_seconds:.1f}s</strong>
                        {' &middot; 📊 <em>Sampled</em>' if report.sampled else ''}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    spacer(16)

    # Sub-scores
    render_section_header("Score Breakdown", "How the usability score is calculated")

    render_metric_card_row([
        {
            "label": "Prediction Quality",
            "value": f"{report.prediction_quality:.0f}",
            "suffix": "/100",
            "description": "TabPFN cross-validation accuracy",
            "color": _score_color(report.prediction_quality),
        },
        {
            "label": "Data Completeness",
            "value": f"{report.data_completeness:.0f}",
            "suffix": "/100",
            "description": "Percentage of non-missing values",
            "color": _score_color(report.data_completeness),
        },
        {
            "label": "Feature Diversity",
            "value": f"{report.feature_diversity:.0f}",
            "suffix": "/100",
            "description": "Mix of feature types",
            "color": _score_color(report.feature_diversity),
        },
        {
            "label": "Size Appropriateness",
            "value": f"{report.size_appropriateness:.0f}",
            "suffix": "/100",
            "description": "Optimal row count for ML",
            "color": _score_color(report.size_appropriateness),
        },
    ])

    spacer(24)

    # Feature analysis
    render_section_header("Feature Analysis", "Importance and statistics for each feature")

    # Create feature table with SHAP values
    feature_data = []
    for fp in sorted(report.feature_profiles, key=lambda x: -x.importance_score):
        feature_data.append({
            "Feature": fp.feature_name,
            "Type": fp.feature_type.title(),
            "Importance": f"{fp.importance_score:.3f}",
            "SHAP": f"{fp.shap_mean:.3f}" if fp.shap_mean > 0 else "-",
            "Missing %": f"{fp.missing_ratio:.1%}",
            "Skew": f"{fp.distribution_skew:.2f}" if fp.feature_type == "numeric" else "-",
            "Unique": fp.unique_count,
            "Suggestion": fp.suggested_transform or "-",
        })

    if feature_data:
        feature_df = pd.DataFrame(feature_data)
        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Importance": st.column_config.TextColumn("Importance", width="small"),
                "SHAP": st.column_config.TextColumn("SHAP Mean", width="small", help="Mean absolute SHAP value - higher means more influential"),
                "Missing %": st.column_config.TextColumn("Missing", width="small"),
                "Skew": st.column_config.TextColumn("Skew", width="small", help="Distribution skewness (for numeric features)"),
                "Unique": st.column_config.NumberColumn("Unique", width="small"),
                "Suggestion": st.column_config.TextColumn("Transform", width="small"),
            },
        )

        # Add SHAP interpretation note
        st.markdown(
            """
            <div style="color: #64748b; font-size: 13px; margin-top: 8px;">
                💡 <strong>SHAP values</strong> show each feature's average contribution to predictions.
                Higher values indicate features that have more influence on the model's decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    spacer(24)

    # ONE-CLICK FIX - Apply All Suggestions
    with card():
        render_apply_all_button(report)

    spacer(24)

    # BEFORE/AFTER COMPARISON - Show transformation impact
    log = st.session_state.get(SESSION_KEY_TRANSFORMATION_LOG)
    if log is not None:
        with card():
            render_before_after_comparison()
        spacer(24)

    # EXPORT & GO - Main export section with code snippet
    with card():
        render_export_section(report)

    spacer(24)

    # Feature engineering suggestions (detailed view)
    render_feature_suggestions(report)

    spacer(24)

    # SYNTHETIC DATA VALIDATION - Benchmark section
    with card():
        render_benchmark_section(report)

    spacer(24)

    # Anomaly detection section
    with card():
        render_anomaly_detection()

    spacer(24)

    # Synthetic data generation section (original)
    with card():
        render_synthetic_generation()

    spacer(24)

    # Add to Catalog section
    render_add_to_catalog(report)

    spacer(24)

    # Download section (legacy - kept for compatibility)
    render_section_header("Export Report", "Download assessment results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # JSON download
        json_data = export_report_json(report)
        json_str = json.dumps(json_data, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"quality_report_{report.target_column}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        # HTML download
        html_content = export_report_html(report)
        st.download_button(
            label="📥 Download HTML",
            data=html_content,
            file_name=f"quality_report_{report.target_column}.html",
            mime="text/html",
            use_container_width=True,
        )

    with col3:
        # Text summary download
        text_summary = generate_report_summary(report)
        st.download_button(
            label="📥 Download Text",
            data=text_summary,
            file_name=f"quality_report_{report.target_column}.txt",
            mime="text/plain",
            use_container_width=True,
        )


SESSION_KEY_ANOMALIES = "anomalies"
SESSION_KEY_SYNTHETIC_DF = "synthetic_df"
SESSION_KEY_SYNTHETIC_METRICS = "synthetic_metrics"


def render_anomaly_detection() -> None:
    """
    Render anomaly detection section.

    Uses Local Outlier Factor to detect unusual records with
    interpretable feature attributions.
    """
    from intuitiveness.quality.anomaly_detector import (
        detect_anomalies,
        get_anomaly_summary,
    )

    render_section_header(
        "Anomaly Detection",
        "Identify unusual records in your dataset"
    )

    df = st.session_state.get(SESSION_KEY_QUALITY_DF)
    if df is None:
        info("Upload a dataset to detect anomalies.")
        return

    # Detection controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        percentile_threshold = st.slider(
            "Percentile Threshold",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Flag records below this percentile as anomalies",
            key="anomaly_percentile_slider",
        )

    with col2:
        max_anomalies = st.slider(
            "Maximum Anomalies",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Maximum number of anomalies to return",
            key="anomaly_max_slider",
        )

    with col3:
        detect_button = st.button(
            "🔍 Detect",
            use_container_width=True,
            key="detect_anomalies_button",
        )

    if detect_button:
        with st.spinner("Detecting anomalies..."):
            try:
                anomalies = detect_anomalies(
                    df=df,
                    percentile_threshold=percentile_threshold,
                    max_anomalies=max_anomalies,
                )
                st.session_state[SESSION_KEY_ANOMALIES] = anomalies
            except Exception as e:
                error(f"Anomaly detection failed: {e}")
                return

    # Display results
    anomalies = st.session_state.get(SESSION_KEY_ANOMALIES)
    if anomalies is None:
        info("Click 'Detect' to find anomalous records.")
        return

    if not anomalies:
        success("No anomalies detected at this threshold!")
        return

    # Summary
    summary = get_anomaly_summary(anomalies, df)
    st.markdown(
        f"""
        <div style="
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            margin: 16px 0;
            border-radius: 0 8px 8px 0;
        ">
            <div style="font-weight: 600; color: #92400e;">
                Found {summary['total_anomalies']} anomalous records ({summary['percentage']:.2f}% of dataset)
            </div>
            <div style="color: #92400e; font-size: 14px; margin-top: 4px;">
                Severity: {summary['severity_distribution'].get('severe', 0)} severe,
                {summary['severity_distribution'].get('moderate', 0)} moderate,
                {summary['severity_distribution'].get('mild', 0)} mild
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top contributing features
    if summary['top_contributing_features']:
        st.markdown("**Top Contributing Features:**")
        contrib_text = ", ".join(
            f"`{f['feature']}` ({f['count']}x)"
            for f in summary['top_contributing_features'][:3]
        )
        st.markdown(contrib_text)

    spacer(12)

    # Anomaly table
    anomaly_data = []
    for a in anomalies[:20]:  # Show top 20
        row_data = df.iloc[a.row_index]
        top_reason = a.top_contributors[0] if a.top_contributors else {}

        anomaly_data.append({
            "Row": a.row_index,
            "Score": f"{a.anomaly_score:.2f}",
            "Percentile": f"{a.percentile:.2f}%",
            "Top Factor": top_reason.get("feature", "-"),
            "Reason": top_reason.get("reason", "-")[:50],
        })

    if anomaly_data:
        anomaly_df = pd.DataFrame(anomaly_data)
        st.dataframe(
            anomaly_df,
            use_container_width=True,
            hide_index=True,
        )

    # Expandable detail for each anomaly
    with st.expander("View Anomaly Details"):
        selected_idx = st.selectbox(
            "Select anomaly to inspect",
            options=[a.row_index for a in anomalies[:20]],
            format_func=lambda x: f"Row {x} (score: {next(a.anomaly_score for a in anomalies if a.row_index == x):.2f})",
            key="anomaly_detail_select",
        )

        if selected_idx is not None:
            anomaly = next((a for a in anomalies if a.row_index == selected_idx), None)
            if anomaly:
                # Show row data
                st.markdown("**Record Values:**")
                row_data = df.iloc[anomaly.row_index].to_frame().T
                st.dataframe(row_data, use_container_width=True)

                # Show contributing factors
                st.markdown("**Contributing Factors:**")
                for contrib in anomaly.top_contributors:
                    st.markdown(
                        f"- **{contrib['feature']}**: {contrib['reason']} "
                        f"(contribution: {contrib['contribution']:.2f})"
                    )

    # Export button
    spacer(8)
    if st.button("📥 Export Anomalies CSV", key="export_anomalies"):
        export_data = []
        for a in anomalies:
            row_dict = df.iloc[a.row_index].to_dict()
            row_dict["_anomaly_score"] = a.anomaly_score
            row_dict["_percentile"] = a.percentile
            row_dict["_top_factor"] = a.top_contributors[0]["feature"] if a.top_contributors else ""
            export_data.append(row_dict)

        export_df = pd.DataFrame(export_data)
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="anomalies.csv",
            mime="text/csv",
            key="download_anomalies_csv",
        )


def render_synthetic_generation() -> None:
    """
    Render synthetic data generation section.

    Uses TabPFN's unsupervised model for high-fidelity synthetic data
    or falls back to Gaussian copula.
    """
    from intuitiveness.quality.synthetic_generator import (
        generate_synthetic,
        get_synthetic_summary,
        check_tabpfn_auth,
    )

    render_section_header(
        "Synthetic Data Generation",
        "Generate synthetic samples preserving statistical properties"
    )

    df = st.session_state.get(SESSION_KEY_QUALITY_DF)
    if df is None:
        info("Upload a dataset to generate synthetic data.")
        return

    # Check TabPFN auth status
    is_auth, auth_msg = check_tabpfn_auth()
    if not is_auth:
        st.markdown(
            f"""
            <div style="
                background: #dbeafe;
                border-left: 4px solid #3b82f6;
                padding: 12px 16px;
                margin: 8px 0;
                border-radius: 0 8px 8px 0;
                font-size: 13px;
                color: #1e40af;
            ">
                <strong>Note:</strong> Using statistical generation method.
                For TabPFN-powered generation, set up HuggingFace authentication.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Generation controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        n_samples = st.slider(
            "Number of Samples",
            min_value=10,
            max_value=min(1000, len(df) * 5),
            value=min(100, len(df)),
            step=10,
            help="Number of synthetic samples to generate",
            key="synthetic_n_samples",
        )

    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Higher = more diverse, Lower = more similar to original",
            key="synthetic_temperature",
        )

    with col3:
        generate_button = st.button(
            "🔮 Generate",
            use_container_width=True,
            key="generate_synthetic_button",
        )

    if generate_button:
        progress_text = st.empty()
        progress_text.markdown("*Generating synthetic data...*")

        try:
            synthetic_df, metrics = generate_synthetic(
                df=df,
                n_samples=n_samples,
                temperature=temperature,
                n_permutations=3,
            )
            st.session_state[SESSION_KEY_SYNTHETIC_DF] = synthetic_df
            st.session_state[SESSION_KEY_SYNTHETIC_METRICS] = metrics
            progress_text.empty()
            success(f"Generated {n_samples} synthetic samples in {metrics.generation_time_seconds:.1f}s")
        except Exception as e:
            progress_text.empty()
            error(f"Synthetic generation failed: {e}")
            return

    # Display results
    synthetic_df = st.session_state.get(SESSION_KEY_SYNTHETIC_DF)
    metrics = st.session_state.get(SESSION_KEY_SYNTHETIC_METRICS)

    if synthetic_df is None or metrics is None:
        info("Click 'Generate' to create synthetic samples.")
        return

    spacer(12)

    # Quality metrics
    render_metric_card_row([
        {
            "label": "Samples Generated",
            "value": str(metrics.n_samples),
            "description": "Synthetic records",
            "color": "#3b82f6",
        },
        {
            "label": "Correlation Preservation",
            "value": f"{(1 - metrics.mean_correlation_error):.0%}",
            "description": "Feature relationships maintained",
            "color": "#22c55e" if metrics.mean_correlation_error < 0.15 else "#f59e0b",
        },
        {
            "label": "Distribution Similarity",
            "value": f"{metrics.distribution_similarity:.0%}",
            "description": "Statistical fidelity (KS test)",
            "color": "#22c55e" if metrics.distribution_similarity > 0.85 else "#f59e0b",
        },
        {
            "label": "Generation Time",
            "value": f"{metrics.generation_time_seconds:.1f}s",
            "description": "Processing time",
            "color": "#64748b",
        },
    ])

    spacer(16)

    # Preview
    st.markdown("**Synthetic Data Preview:**")
    st.dataframe(synthetic_df.head(10), use_container_width=True)

    # Comparison with original
    with st.expander("Compare with Original Data"):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:5]

        comparison_data = []
        for col in numeric_cols:
            comparison_data.append({
                "Feature": col,
                "Original Mean": f"{df[col].mean():.3f}",
                "Synthetic Mean": f"{synthetic_df[col].mean():.3f}",
                "Original Std": f"{df[col].std():.3f}",
                "Synthetic Std": f"{synthetic_df[col].std():.3f}",
            })

        if comparison_data:
            st.dataframe(
                pd.DataFrame(comparison_data),
                use_container_width=True,
                hide_index=True,
            )

    # Download
    spacer(8)
    csv_buffer = io.StringIO()
    synthetic_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Synthetic Data (CSV)",
        data=csv_buffer.getvalue(),
        file_name="synthetic_data.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_synthetic_csv",
    )


def render_add_to_catalog(report) -> None:
    """
    Render the 'Add to Catalog' section.

    Allows users to save assessed datasets to the catalog for future reference.

    Args:
        report: QualityReport instance.
    """
    from intuitiveness.catalog.storage import get_storage
    from intuitiveness.catalog.models import Dataset
    from uuid import uuid4
    from datetime import datetime

    render_section_header(
        "Save to Catalog",
        "Add this assessed dataset to your catalog"
    )

    df = st.session_state.get(SESSION_KEY_QUALITY_DF)
    file_name = st.session_state.get(SESSION_KEY_QUALITY_FILE_NAME, "dataset")

    if df is None:
        info("No dataset available to save.")
        return

    # Dataset metadata form
    col1, col2 = st.columns(2)

    with col1:
        dataset_name = st.text_input(
            "Dataset Name",
            value=Path(file_name).stem if file_name else "My Dataset",
            help="A descriptive name for this dataset",
            key="catalog_dataset_name",
        )

    with col2:
        domain_tags_input = st.text_input(
            "Domain Tags",
            placeholder="e.g., healthcare, classification, public",
            help="Comma-separated tags for categorization",
            key="catalog_domain_tags",
        )

    description = st.text_area(
        "Description",
        placeholder="Brief description of this dataset and its intended use...",
        help="Optional description for documentation",
        key="catalog_description",
        height=80,
    )

    spacer(8)

    if st.button("📁 Add to Catalog", use_container_width=True, key="add_to_catalog_button"):
        if not dataset_name:
            warning("Please provide a dataset name.")
            return

        try:
            storage = get_storage()

            # Parse domain tags
            domain_tags = [
                tag.strip()
                for tag in domain_tags_input.split(",")
                if tag.strip()
            ] if domain_tags_input else []

            # Create dataset entry
            dataset = Dataset(
                id=uuid4(),
                name=dataset_name,
                description=description or None,
                source_path=file_name,
                row_count=len(df),
                feature_count=len(df.columns),
                target_column=report.target_column,
                usability_score=report.usability_score,
                domain_tags=domain_tags,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                quality_reports=[],  # Will be populated by storage
            )

            # Add to storage
            dataset_id = storage.add_dataset(dataset)

            # Store the latest quality report
            from intuitiveness.quality.report import export_report_json
            report_dict = export_report_json(report)
            storage.add_quality_report(dataset_id, report_dict)

            success(f"Dataset '{dataset_name}' added to catalog!")

        except Exception as e:
            error(f"Failed to add to catalog: {e}")


def render_quality_dashboard() -> None:
    """
    Render the complete quality assessment dashboard.
    """
    render_page_header(
        "Dataset Quality Assessment",
        "Analyze your dataset's ML-readiness with TabPFN-powered quality scoring",
    )

    spacer(16)

    # Check for existing report in session
    report = st.session_state.get(SESSION_KEY_QUALITY_REPORT)

    if report is not None:
        # Show report with option to start over
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 New Assessment", use_container_width=True):
                # P0 FIX: Clear entire history when starting completely fresh
                _clear_report_history()
                st.session_state.pop(SESSION_KEY_QUALITY_DF, None)
                st.session_state.pop(SESSION_KEY_QUALITY_FILE_NAME, None)
                st.session_state.pop(SESSION_KEY_TRANSFORMED_DF, None)
                st.session_state.pop(SESSION_KEY_TRANSFORMATION_LOG, None)
                st.session_state.pop(SESSION_KEY_BENCHMARK_REPORT, None)
                st.session_state.pop(SESSION_KEY_APPLIED_SUGGESTIONS, None)
                st.rerun()

        render_quality_report(report)
        return

    # File upload
    with card():
        render_section_header("Upload Dataset", "Upload a CSV file to begin assessment")
        df = render_file_upload()

    if df is None:
        info(
            "Upload a CSV file to get started. "
            "The assessment works best with datasets of 50-10,000 rows."
        )
        return

    spacer(16)

    # Show data preview
    with card():
        render_section_header(
            "Data Preview",
            f"{len(df):,} rows × {len(df.columns)} columns"
        )
        st.dataframe(df.head(10), use_container_width=True)

    spacer(16)

    # Target selection and assessment
    with card():
        render_section_header("Configure Assessment", "Select the target column for prediction")

        target_column = render_target_selection(df)

        if target_column:
            # Show target stats
            target_series = df[target_column]
            n_unique = target_series.nunique()
            n_missing = target_series.isna().sum()

            st.markdown(
                f"""
                <div style="color: #64748b; font-size: 14px; margin: 12px 0;">
                    Target: <strong>{target_column}</strong> &middot;
                    {n_unique} unique values &middot;
                    {n_missing} missing ({n_missing/len(df):.1%})
                </div>
                """,
                unsafe_allow_html=True,
            )

            spacer(16)
            render_assessment_button(df, target_column)
