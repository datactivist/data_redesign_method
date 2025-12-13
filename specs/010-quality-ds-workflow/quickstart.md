# Developer Quickstart: Data Scientist Co-Pilot

**Feature**: 010-quality-ds-workflow
**Prerequisites**: Python 3.11, `myenv311` virtual environment

## Quick Start

### 1. Activate Environment

```bash
cd /Users/arthursarazin/Documents/data_redesign_method
source myenv311/bin/activate
```

### 2. Run the App

```bash
streamlit run intuitiveness/streamlit_app.py
```

### 3. Test the Workflow

1. Navigate to **Quality Dashboard** section
2. Upload a CSV file with a target column
3. Click **Assess Quality**
4. See the traffic light indicator (🟢/🟡/🔴)
5. Click **Apply All Suggestions** (if yellow)
6. Click **Export Clean CSV**
7. Copy the Python code snippet

## File Structure

```
intuitiveness/quality/
├── benchmark.py      # NEW: Synthetic validation
├── exporter.py       # NEW: Export & code gen
├── assessor.py       # EXTEND: apply_all_suggestions
├── models.py         # EXTEND: new data models
└── ...

intuitiveness/ui/
└── quality_dashboard.py  # MAJOR UPDATE
```

## Key APIs

### Benchmark Synthetic Data

```python
from intuitiveness.quality.benchmark import benchmark_synthetic

report = benchmark_synthetic(
    df=my_dataframe,
    target_column='fraud',
    n_synthetic=5000,
    class_balanced=True
)

print(f"Transfer gap: {report.mean_transfer_gap:.1%}")
print(f"Safe to use: {report.is_safe}")
```

### Apply All Suggestions

```python
from intuitiveness.quality.assessor import apply_all_suggestions

clean_df, log = apply_all_suggestions(
    df=my_dataframe,
    suggestions=quality_report.suggestions,
    target_column='price'
)

print(f"Applied {log.total_applied} transformations")
print(f"Accuracy improved by {log.total_accuracy_improvement:.1%}")
```

### Export Dataset

```python
from intuitiveness.quality.exporter import export_dataset

package = export_dataset(
    df=clean_df,
    format='csv',
    dataset_name='my_data',
    transformation_log=log
)

# Get Python snippet
print(package.python_snippet)

# Get binary data for download
csv_bytes = package.export()
```

### Get Readiness Indicator

```python
from intuitiveness.quality.assessor import get_readiness_indicator

indicator = get_readiness_indicator(
    score=quality_report.usability_score,
    n_suggestions=len(quality_report.suggestions),
    estimated_improvement=15.0
)

print(f"{indicator.title}")  # 🟡 FIXABLE
print(f"{indicator.message}")  # "3 automated fixes will improve score to 85"
```

## Testing

### Run Unit Tests

```bash
pytest tests/unit/test_benchmark.py -v
pytest tests/unit/test_exporter.py -v
```

### Run Integration Tests

```bash
pytest tests/integration/test_quality_workflow.py -v
```

### Run E2E Tests (Playwright)

```bash
# Requires Playwright MCP server running
pytest tests/e2e/test_quality_dashboard.py -v
```

## Common Issues

### TabPFN Not Available

If TabPFN is not installed or no GPU available:
- Benchmark will fall back to sklearn-only models
- Synthetic generation will use Gaussian copula
- A warning will be shown in the UI

### Slow Benchmark

If benchmark takes >30 seconds:
- Reduce `n_synthetic` samples
- Use fewer models: `models=['RandomForest']`
- Sample your dataset before benchmarking

### Export Fails

If export fails:
- Check disk space for large Parquet files
- For Pickle, ensure no lambda functions in DataFrame
- CSV always works as fallback

## Development Workflow

1. **Implement `benchmark.py`** — Core synthetic validation logic
2. **Implement `exporter.py`** — Export and code generation
3. **Extend `assessor.py`** — Add `apply_all_suggestions`
4. **Extend `models.py`** — Add new data models
5. **Update `quality_dashboard.py`** — Add UI components
6. **Write tests** — Unit, integration, E2E
7. **Run Playwright tests** — Full workflow validation
