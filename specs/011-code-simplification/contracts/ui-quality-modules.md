# Contract: UI Quality Module Structure

## Purpose

Define the module boundaries and public APIs for the `ui/quality/` package.

## Package Overview

The `ui/quality/` package decomposes the monolithic `quality_dashboard.py` into focused modules aligned with spec 010-quality-ds-workflow user stories.

## Module Responsibilities

### utils.py - Session State & Utilities

**Responsibility**: Centralized session state keys and score utilities

**Public API**:
```python
# Session state keys (constants)
SESSION_KEY_QUALITY_REPORT: str
SESSION_KEY_QUALITY_DF: str
SESSION_KEY_QUALITY_FILE_NAME: str
SESSION_KEY_ASSESSMENT_PROGRESS: str
SESSION_KEY_APPLIED_SUGGESTIONS: str
SESSION_KEY_TRANSFORMED_DF: str
SESSION_KEY_TRANSFORMATION_LOG: str
SESSION_KEY_BENCHMARK_REPORT: str
SESSION_KEY_EXPORT_FORMAT: str
SESSION_KEY_QUALITY_REPORTS_HISTORY: str
SESSION_KEY_CURRENT_REPORT_INDEX: str

# Functions
def get_score_color(score: float) -> str:
    """Return hex color for score (green/yellow/red)."""

def get_score_label(score: float) -> str:
    """Return label for score (Excellent/Good/Needs Work)."""
```

**Invariants**:
- Session keys are string constants, not computed
- Color thresholds: green (≥80), yellow (60-79), red (<60)

---

### state.py - Report History Management

**Responsibility**: Track report versions for before/after comparison

**Public API**:
```python
def save_report_to_history(report: QualityReport) -> None:
    """Save report to session state history."""

def get_initial_report() -> Optional[QualityReport]:
    """Get first report in history (baseline)."""

def get_current_report() -> Optional[QualityReport]:
    """Get most recent report."""

def clear_report_history() -> None:
    """Clear all reports from history."""

def render_quality_score_evolution() -> None:
    """Display score evolution chart."""
```

**Spec Traceability**: US-3 (Before/After Benchmarks)

---

### upload.py - File Upload (US-1 Step 1)

**Responsibility**: CSV upload and target column selection

**Public API**:
```python
def render_file_upload() -> Optional[pd.DataFrame]:
    """
    Render file uploader widget.

    Returns:
        DataFrame if file uploaded, None otherwise

    Side Effects:
        Sets SESSION_KEY_QUALITY_DF in session state
    """

def render_target_selection(df: pd.DataFrame) -> Optional[str]:
    """
    Render target column selector.

    Args:
        df: The uploaded DataFrame

    Returns:
        Selected column name, or None if not selected
    """
```

**Spec Traceability**: US-1 (60-Second Data Prep) - Step 1

---

### assessment.py - Assessment Button (US-1 Step 2)

**Responsibility**: Quality assessment with progress indicator

**Public API**:
```python
def render_assessment_button(
    df: pd.DataFrame,
    target_column: str
) -> Optional[QualityReport]:
    """
    Render assessment button and execute assessment.

    Args:
        df: DataFrame to assess
        target_column: Target column for ML

    Returns:
        QualityReport if assessment complete, None otherwise

    Side Effects:
        Sets SESSION_KEY_QUALITY_REPORT in session state
        Updates SESSION_KEY_ASSESSMENT_PROGRESS during execution
    """
```

**Spec Traceability**: US-1 (60-Second Data Prep) - Step 2

---

### suggestions.py - Feature Suggestions (US-1 Step 3)

**Responsibility**: Display and apply feature engineering suggestions

**Public API**:
```python
def render_feature_suggestions(
    report: QualityReport,
    df: pd.DataFrame
) -> List[FeatureSuggestion]:
    """
    Render suggestion cards with apply buttons.

    Args:
        report: Quality assessment report
        df: Current DataFrame

    Returns:
        List of suggestions (for display tracking)

    Side Effects:
        May update SESSION_KEY_TRANSFORMED_DF when user applies
    """

def render_apply_all_button(
    suggestions: List[FeatureSuggestion],
    df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Render "Apply All" button (FR-002).

    Args:
        suggestions: List of suggestions to apply
        df: Current DataFrame

    Returns:
        Transformed DataFrame if applied, None otherwise

    Side Effects:
        Sets SESSION_KEY_TRANSFORMED_DF
        Updates SESSION_KEY_APPLIED_SUGGESTIONS
    """
```

**Spec Traceability**: US-1 Step 3, FR-002 (One-Click Apply All)

---

### readiness.py - Traffic Light Indicator (US-4)

**Responsibility**: Go/no-go visual indicator with methodology transparency

**Public API**:
```python
def render_readiness_indicator(report: QualityReport) -> None:
    """
    Display traffic light readiness indicator.

    Colors:
    - Green: Score ≥80 (Ready for ML)
    - Yellow: Score 60-79 (Fixable)
    - Red: Score <60 (Needs Work)

    Args:
        report: Quality assessment report
    """

def render_tabpfn_methodology(report: QualityReport) -> None:
    """
    Display TabPFN methodology explanation.

    Shows:
    - What TabPFN is (Nature paper reference)
    - Per-fold CV scores
    - Feature handling details
    - SHAP computation status

    Args:
        report: Report with tabpfn_diagnostics
    """
```

**Spec Traceability**: US-4 (Traffic Light), FR-001 (Readiness Indicator)

---

## Package __init__.py Contract

The package must re-export all public APIs:

```python
from intuitiveness.ui.quality import (
    # Session keys
    SESSION_KEY_QUALITY_REPORT,
    SESSION_KEY_QUALITY_DF,
    # ... all keys

    # Utilities
    get_score_color,
    get_score_label,

    # State
    save_report_to_history,
    get_initial_report,
    get_current_report,
    clear_report_history,
    render_quality_score_evolution,

    # Upload (US-1 Step 1)
    render_file_upload,
    render_target_selection,

    # Assessment (US-1 Step 2)
    render_assessment_button,

    # Suggestions (US-1 Step 3)
    render_feature_suggestions,
    render_apply_all_button,

    # Readiness (US-4)
    render_readiness_indicator,
    render_tabpfn_methodology,
)
```

## Backward Compatibility

The original `quality_dashboard.py` must import from the package and re-export for backward compatibility:

```python
# quality_dashboard.py
from intuitiveness.ui.quality import (
    render_file_upload,
    render_assessment_button,
    # ... etc
)

# Backward compatibility aliases
_save_report_to_history = save_report_to_history
_score_color = get_score_color
```
