"""
Quality Data Platform - ML Diagnostic Visualizations

P0 FIX: Standard ML graphics that build trust with data scientists.

This module provides visualization functions for:
- Confusion Matrix (classification)
- ROC Curve with AUC (binary classification)
- Feature Importance Bar Chart
- Class Distribution
- Prediction Confidence Distribution

These visualizations address ML engineer feedback: "The common machine learning
graphics are not there, so how do I know that my dataset is ready for ML?"
"""

import logging
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from intuitiveness.quality.models import QualityReport, FeatureProfile

logger = logging.getLogger(__name__)


# Color palette consistent with UI
COLORS = {
    "primary": "#3b82f6",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "muted": "#64748b",
    "background": "#f8fafc",
}


def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """
    Create an interactive confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Names for each class.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize for color intensity
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Get class names
    if class_names is None:
        unique_labels = sorted(set(y_true) | set(y_pred))
        class_names = [str(label) for label in unique_labels]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale="Blues",
        showscale=True,
        hovertemplate=(
            "Predicted: %{x}<br>"
            "Actual: %{y}<br>"
            "Count: %{z}<extra></extra>"
        ),
    ))

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Add annotations with metrics
    annotations = []
    for i, name in enumerate(class_names):
        annotations.append(dict(
            x=1.15,
            y=name,
            text=f"P:{precision[i]:.2f} R:{recall[i]:.2f}",
            font=dict(size=10, color=COLORS["muted"]),
            showarrow=False,
            xref="paper",
            yref="y",
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        annotations=annotations,
        width=500,
        height=450,
        margin=dict(l=80, r=100, t=60, b=60),
    )

    return fig


def create_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
) -> go.Figure:
    """
    Create ROC curve with AUC score for binary classification.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})',
        line=dict(color=COLORS["primary"], width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
    ))

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color=COLORS["muted"], width=2, dash='dash'),
    ))

    # Add AUC annotation
    fig.add_annotation(
        x=0.6,
        y=0.3,
        text=f"<b>AUC = {roc_auc:.3f}</b>",
        font=dict(size=20, color=COLORS["primary"]),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=COLORS["primary"],
        borderwidth=2,
        borderpad=10,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
        width=500,
        height=450,
        showlegend=True,
        legend=dict(x=0.6, y=0.1),
    )

    return fig


def create_feature_importance_chart(
    report: QualityReport,
    top_n: int = 15,
    title: str = "Feature Importance",
) -> go.Figure:
    """
    Create horizontal bar chart of feature importance.

    Args:
        report: QualityReport with feature profiles.
        top_n: Number of top features to show.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    # Sort features by importance
    features = sorted(
        report.feature_profiles,
        key=lambda x: x.importance_score,
        reverse=True
    )[:top_n]

    # Prepare data
    names = [f.feature_name for f in features][::-1]  # Reverse for horizontal bar
    scores = [f.importance_score for f in features][::-1]

    # Color by importance level
    colors = []
    for score in scores:
        if score >= 0.7:
            colors.append(COLORS["success"])
        elif score >= 0.3:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["danger"])

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation='h',
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Importance Score",
        yaxis_title="",
        xaxis=dict(range=[0, 1.1]),
        height=max(350, len(features) * 25 + 100),
        margin=dict(l=150, r=60, t=60, b=40),
    )

    return fig


def create_class_distribution(
    y: pd.Series,
    title: str = "Class Distribution",
) -> go.Figure:
    """
    Create bar chart showing class distribution.

    Highlights class imbalance issues.

    Args:
        y: Target series.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    # Count classes
    class_counts = y.value_counts().sort_index()

    # Determine if imbalanced
    total = len(y)
    percentages = class_counts / total * 100

    # Color by balance (green if ~balanced, red if severely imbalanced)
    colors = []
    for pct in percentages:
        if pct < 5 or pct > 95:
            colors.append(COLORS["danger"])  # Severe imbalance
        elif pct < 20 or pct > 80:
            colors.append(COLORS["warning"])  # Moderate imbalance
        else:
            colors.append(COLORS["success"])  # Balanced

    fig = go.Figure(go.Bar(
        x=[str(c) for c in class_counts.index],
        y=class_counts.values,
        marker_color=colors,
        text=[f"{count:,}<br>({pct:.1f}%)" for count, pct in zip(class_counts.values, percentages)],
        textposition='outside',
        hovertemplate="<b>Class %{x}</b><br>Count: %{y:,}<extra></extra>",
    ))

    # Add imbalance warning annotation if needed
    max_pct = percentages.max()
    if max_pct > 80:
        fig.add_annotation(
            x=0.5,
            y=1.1,
            xref="paper",
            yref="paper",
            text="⚠️ Class imbalance detected - consider synthetic data augmentation",
            font=dict(size=12, color=COLORS["warning"]),
            showarrow=False,
            bgcolor="rgba(245, 158, 11, 0.1)",
            borderpad=5,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Class",
        yaxis_title="Count",
        height=350,
    )

    return fig


def create_prediction_confidence_histogram(
    probabilities: np.ndarray,
    title: str = "Prediction Confidence Distribution",
) -> go.Figure:
    """
    Create histogram of prediction confidence (max probability).

    Shows how certain the model is about its predictions.

    Args:
        probabilities: Prediction probabilities (shape: n_samples x n_classes).
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    # Get max probability for each prediction (confidence)
    if probabilities.ndim == 1:
        max_proba = probabilities
    else:
        max_proba = probabilities.max(axis=1)

    fig = go.Figure(go.Histogram(
        x=max_proba,
        nbinsx=20,
        marker_color=COLORS["primary"],
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    # Add reference lines
    mean_conf = np.mean(max_proba)
    fig.add_vline(
        x=mean_conf,
        line_dash="dash",
        line_color=COLORS["success"],
        annotation_text=f"Mean: {mean_conf:.2f}",
        annotation_position="top",
    )

    # Add uncertain zone highlight
    fig.add_vrect(
        x0=0,
        x1=0.6,
        fillcolor="rgba(239, 68, 68, 0.1)",
        line_width=0,
        annotation_text="Low confidence zone",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Max Probability (Confidence)",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1]),
        height=350,
    )

    return fig


def create_score_evolution_chart(
    scores: List[float],
    labels: Optional[List[str]] = None,
    title: str = "Quality Score Evolution",
) -> go.Figure:
    """
    Create line chart showing quality score evolution across assessments.

    Args:
        scores: List of usability scores.
        labels: Optional labels for each assessment point.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    if len(scores) < 2:
        # Not enough data for evolution chart
        return None

    if labels is None:
        labels = [f"Assessment {i+1}" for i in range(len(scores))]

    # Determine colors based on score improvement
    colors = []
    for i, score in enumerate(scores):
        if i == 0:
            colors.append(COLORS["muted"])
        elif score >= scores[i-1]:
            colors.append(COLORS["success"])
        else:
            colors.append(COLORS["danger"])

    fig = go.Figure()

    # Line trace
    fig.add_trace(go.Scatter(
        x=labels,
        y=scores,
        mode='lines+markers',
        marker=dict(size=12, color=colors),
        line=dict(width=3, color=COLORS["primary"]),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
    ))

    # Add score annotations
    for i, (label, score) in enumerate(zip(labels, scores)):
        fig.add_annotation(
            x=label,
            y=score + 3,
            text=f"{score:.0f}",
            font=dict(size=14, color=colors[i]),
            showarrow=False,
        )

    # Add horizontal reference lines
    fig.add_hline(y=80, line_dash="dot", line_color=COLORS["success"],
                  annotation_text="Ready (80+)", annotation_position="right")
    fig.add_hline(y=60, line_dash="dot", line_color=COLORS["warning"],
                  annotation_text="Fixable (60+)", annotation_position="right")

    # Calculate overall improvement
    improvement = scores[-1] - scores[0]
    improvement_text = f"+{improvement:.1f}" if improvement >= 0 else f"{improvement:.1f}"
    improvement_color = COLORS["success"] if improvement >= 0 else COLORS["danger"]

    fig.add_annotation(
        x=1,
        y=1.15,
        xref="paper",
        yref="paper",
        text=f"Total Change: <b>{improvement_text}</b>",
        font=dict(size=14, color=improvement_color),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="",
        yaxis_title="Usability Score",
        yaxis=dict(range=[0, 110]),
        height=400,
        showlegend=False,
    )

    return fig


def create_shap_summary_plot(
    feature_profiles: List[FeatureProfile],
    top_n: int = 15,
    title: str = "SHAP Feature Impact",
) -> go.Figure:
    """
    Create a SHAP-style summary plot showing feature impact.

    This is a simplified version since we only have mean SHAP values,
    not the full distribution.

    Args:
        feature_profiles: List of FeatureProfile with shap_mean values.
        top_n: Number of top features to show.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    # Filter features with SHAP values
    features_with_shap = [f for f in feature_profiles if f.shap_mean > 0]

    if not features_with_shap:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="SHAP values not available for this assessment",
            font=dict(size=14, color=COLORS["muted"]),
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=350,
        )
        return fig

    # Sort by SHAP value
    sorted_features = sorted(
        features_with_shap,
        key=lambda x: x.shap_mean,
        reverse=True
    )[:top_n]

    # Prepare data
    names = [f.feature_name for f in sorted_features][::-1]
    shap_values = [f.shap_mean for f in sorted_features][::-1]

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=shap_values,
        y=names,
        orientation='h',
        marker=dict(
            color=shap_values,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Impact"),
        ),
        text=[f"{s:.3f}" for s in shap_values],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Mean |SHAP| Value",
        yaxis_title="",
        height=max(350, len(sorted_features) * 25 + 100),
        margin=dict(l=150, r=60, t=60, b=40),
    )

    return fig


def create_all_diagnostics(
    report: QualityReport,
    df: pd.DataFrame,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, go.Figure]:
    """
    Create all diagnostic visualizations for a quality report.

    Args:
        report: QualityReport instance.
        df: Original DataFrame.
        y_true: True labels (if available).
        y_pred: Predicted labels (if available).
        y_proba: Predicted probabilities (if available).

    Returns:
        Dictionary of visualization name to Plotly Figure.
    """
    visualizations = {}

    # Feature Importance (always available)
    visualizations["feature_importance"] = create_feature_importance_chart(report)

    # SHAP Summary (if SHAP values available)
    shap_chart = create_shap_summary_plot(report.feature_profiles)
    if shap_chart:
        visualizations["shap_summary"] = shap_chart

    # Class Distribution (for classification)
    if report.task_type == "classification" and report.target_column in df.columns:
        y = df[report.target_column]
        visualizations["class_distribution"] = create_class_distribution(y)

    # Confusion Matrix (if predictions available)
    if y_true is not None and y_pred is not None:
        visualizations["confusion_matrix"] = create_confusion_matrix(y_true, y_pred)

    # ROC Curve (for binary classification with probabilities)
    if (y_true is not None and y_proba is not None and
            report.task_type == "classification" and len(np.unique(y_true)) == 2):
        # Use positive class probability
        if y_proba.ndim > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        visualizations["roc_curve"] = create_roc_curve(y_true, y_proba_pos)

    # Prediction Confidence (if probabilities available)
    if y_proba is not None:
        visualizations["confidence_distribution"] = create_prediction_confidence_histogram(y_proba)

    return visualizations
