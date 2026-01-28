"""
Quality Workflow Package - 60-Second Data Prep Workflow.

Implements Spec 010 (Data Scientist Co-Pilot)

Provides orchestration for the complete 60-second workflow:
- Traffic light readiness indicators
- One-click "Apply All" transformations
- Export clean CSV + Python code
- Synthetic data validation (benchmarking)

Usage:
    from intuitiveness.quality.workflow import run_60_second_workflow

    result = run_60_second_workflow(
        df=messy_data,
        target_column="target",
        auto_apply_suggestions=True
    )

    # Check if meets 60s target
    print(f"Completed in {result.total_time:.1f}s")
    print(f"Readiness: {result.readiness_status.message}")

    # Export
    with open("clean_data.csv", "wb") as f:
        f.write(result.export_csv)
"""

from intuitiveness.quality.workflow.traffic_light import (
    ReadinessStatus,
    get_readiness_status,
    estimate_score_improvement,
    GREEN_THRESHOLD,
    YELLOW_THRESHOLD,
)

from intuitiveness.quality.workflow.sixty_second import (
    WorkflowResult,
    run_60_second_workflow,
    quick_export,
)

__all__ = [
    # Traffic light
    'ReadinessStatus',
    'get_readiness_status',
    'estimate_score_improvement',
    'GREEN_THRESHOLD',
    'YELLOW_THRESHOLD',

    # 60-second workflow
    'WorkflowResult',
    'run_60_second_workflow',
    'quick_export',
]
