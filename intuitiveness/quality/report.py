"""
Quality Data Platform - Report Generation

Generate and export quality reports in various formats.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from intuitiveness.quality.models import QualityReport
# Import consolidated utility (Phase 2 - 011-code-simplification)
from intuitiveness.utils import score_to_color

logger = logging.getLogger(__name__)


def generate_report_summary(report: QualityReport) -> str:
    """
    Generate a human-readable text summary of a quality report.

    Args:
        report: QualityReport instance.

    Returns:
        Formatted text summary.
    """
    lines = [
        "=" * 60,
        "DATASET QUALITY REPORT",
        "=" * 60,
        "",
        f"Assessment Date: {report.created_at.strftime('%Y-%m-%d %H:%M')}",
        f"Target Column: {report.target_column}",
        f"Task Type: {report.task_type.title()}",
        f"Rows: {report.row_count:,}" + (" (sampled)" if report.sampled else ""),
        f"Features: {report.feature_count}",
        f"Assessment Time: {report.assessment_time_seconds:.1f}s",
        "",
        "-" * 60,
        "SCORES",
        "-" * 60,
        "",
        f"📊 USABILITY SCORE: {report.usability_score:.0f}/100",
        "",
        f"  • Prediction Quality: {report.prediction_quality:.0f}/100",
        f"  • Data Completeness: {report.data_completeness:.0f}/100",
        f"  • Feature Diversity: {report.feature_diversity:.0f}/100",
        f"  • Size Appropriateness: {report.size_appropriateness:.0f}/100",
        "",
    ]

    # Top features
    if report.feature_profiles:
        lines.extend([
            "-" * 60,
            "TOP FEATURES (by importance)",
            "-" * 60,
            "",
        ])
        top_features = report.get_top_features(5)
        for i, fp in enumerate(top_features, 1):
            lines.append(
                f"  {i}. {fp.feature_name}: {fp.importance_score:.2f} "
                f"({fp.feature_type}, {fp.missing_ratio:.1%} missing)"
            )
        lines.append("")

    # Low importance features
    low_importance = report.get_low_importance_features()
    if low_importance:
        lines.extend([
            "-" * 60,
            "LOW IMPORTANCE FEATURES (consider removing)",
            "-" * 60,
            "",
        ])
        for fp in low_importance[:3]:
            lines.append(f"  • {fp.feature_name}: {fp.importance_score:.3f}")
        lines.append("")

    # Suggestions
    if report.suggestions:
        lines.extend([
            "-" * 60,
            "SUGGESTIONS",
            "-" * 60,
            "",
        ])
        for s in report.suggestions[:5]:
            lines.append(
                f"  [{s.suggestion_type.upper()}] {s.description} "
                f"(+{s.expected_impact:.1f} pts, confidence: {s.confidence:.0%})"
            )
        lines.append("")

    # Anomalies
    if report.anomalies:
        lines.extend([
            "-" * 60,
            f"ANOMALIES ({len(report.anomalies)} detected)",
            "-" * 60,
            "",
        ])
        for a in report.anomalies[:3]:
            lines.append(
                f"  Row {a.row_index}: percentile {a.percentile:.1f}%"
            )
            for contrib in a.top_contributors[:2]:
                lines.append(f"    - {contrib['feature']}: {contrib['reason']}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def export_report_json(
    report: QualityReport,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Export quality report as JSON.

    Args:
        report: QualityReport instance.
        output_path: Optional path to save JSON file.

    Returns:
        Report as dictionary.
    """
    data = report.to_dict()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Report exported to {output_path}")

    return data


def export_report_html(
    report: QualityReport,
    output_path: Optional[Path] = None,
) -> str:
    """
    Export quality report as HTML.

    Args:
        report: QualityReport instance.
        output_path: Optional path to save HTML file.

    Returns:
        HTML string.
    """
    # Score color - use consolidated utility (011-code-simplification)
    def get_color(score: float) -> str:
        return score_to_color(score, thresholds=(40, 60, 80), colors=("#ef4444", "#f97316", "#eab308", "#22c55e"))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Quality Report - {report.target_column}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1e293b; }}
        .score-card {{ background: #f8fafc; border-radius: 12px; padding: 24px; margin: 20px 0; }}
        .main-score {{ font-size: 48px; font-weight: bold; color: {get_color(report.usability_score)}; }}
        .sub-scores {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-top: 20px; }}
        .sub-score {{ background: white; padding: 16px; border-radius: 8px; }}
        .sub-score-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
        .sub-score-value {{ font-size: 24px; font-weight: bold; }}
        .feature-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .feature-table th, .feature-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        .feature-table th {{ background: #f1f5f9; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
        .badge-numeric {{ background: #dbeafe; color: #1e40af; }}
        .badge-categorical {{ background: #dcfce7; color: #166534; }}
        .badge-boolean {{ background: #fef3c7; color: #92400e; }}
        .meta {{ color: #64748b; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>Dataset Quality Report</h1>

    <p class="meta">
        Target: <strong>{report.target_column}</strong> |
        Task: <strong>{report.task_type.title()}</strong> |
        Rows: <strong>{report.row_count:,}</strong> |
        Features: <strong>{report.feature_count}</strong> |
        Assessed: <strong>{report.created_at.strftime('%Y-%m-%d %H:%M')}</strong>
    </p>

    <div class="score-card">
        <div>Usability Score</div>
        <div class="main-score">{report.usability_score:.0f}<span style="font-size: 24px; color: #94a3b8;">/100</span></div>

        <div class="sub-scores">
            <div class="sub-score">
                <div class="sub-score-label">Prediction Quality</div>
                <div class="sub-score-value" style="color: {get_color(report.prediction_quality)}">{report.prediction_quality:.0f}</div>
            </div>
            <div class="sub-score">
                <div class="sub-score-label">Data Completeness</div>
                <div class="sub-score-value" style="color: {get_color(report.data_completeness)}">{report.data_completeness:.0f}</div>
            </div>
            <div class="sub-score">
                <div class="sub-score-label">Feature Diversity</div>
                <div class="sub-score-value" style="color: {get_color(report.feature_diversity)}">{report.feature_diversity:.0f}</div>
            </div>
            <div class="sub-score">
                <div class="sub-score-label">Size Appropriateness</div>
                <div class="sub-score-value" style="color: {get_color(report.size_appropriateness)}">{report.size_appropriateness:.0f}</div>
            </div>
        </div>
    </div>

    <h2>Feature Analysis</h2>
    <table class="feature-table">
        <thead>
            <tr>
                <th>Feature</th>
                <th>Type</th>
                <th>Importance</th>
                <th>Missing</th>
                <th>Unique</th>
            </tr>
        </thead>
        <tbody>
"""

    for fp in sorted(report.feature_profiles, key=lambda x: -x.importance_score):
        badge_class = f"badge-{fp.feature_type}"
        html += f"""            <tr>
                <td>{fp.feature_name}</td>
                <td><span class="badge {badge_class}">{fp.feature_type}</span></td>
                <td>{fp.importance_score:.3f}</td>
                <td>{fp.missing_ratio:.1%}</td>
                <td>{fp.unique_count}</td>
            </tr>
"""

    html += """        </tbody>
    </table>
"""

    if report.suggestions:
        html += """    <h2>Suggestions</h2>
    <ul>
"""
        for s in report.suggestions:
            html += f"""        <li><strong>[{s.suggestion_type.upper()}]</strong> {s.description} <em>(+{s.expected_impact:.1f} pts)</em></li>
"""
        html += """    </ul>
"""

    html += f"""
    <p class="meta">Assessment completed in {report.assessment_time_seconds:.1f} seconds</p>
</body>
</html>
"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        logger.info(f"HTML report exported to {output_path}")

    return html


def get_score_interpretation(score: float) -> str:
    """
    Get human-readable interpretation of a usability score.

    Args:
        score: Usability score 0-100.

    Returns:
        Interpretation string.
    """
    if score >= 80:
        return "Excellent - This dataset is highly suitable for machine learning tasks with minimal preparation needed."
    elif score >= 60:
        return "Good - This dataset is suitable for ML with some data cleaning or feature engineering recommended."
    elif score >= 40:
        return "Fair - This dataset may work for ML but requires significant preparation or additional data collection."
    else:
        return "Poor - This dataset needs substantial work before being suitable for machine learning."
