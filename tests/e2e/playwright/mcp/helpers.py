"""
Playwright MCP E2E Test Helpers

This module provides dataclasses and configurations for running
visual E2E tests via Playwright MCP tools.

The test configurations match the reference session exports:
- test0_schools: Schools dataset (L0 = 88.25)
- test1_ademe: ADEME funding dataset (L0 = 69,586,180.93)
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# =============================================================================
# Configuration Dataclasses (T009)
# =============================================================================

@dataclass
class SourceFile:
    """CSV file configuration for upload."""
    filename: str
    path: str
    expected_rows: int
    expected_columns: int


@dataclass
class JoinConfig:
    """Parameters for L4->L3 semantic join."""
    left_column: str
    right_column: str
    threshold: float = 0.85
    model: str = "intfloat/multilingual-e5-small"


@dataclass
class CategorizeConfig:
    """Parameters for L3->L2 categorization."""
    dimension_name: str
    column: str
    rules: dict[str, str]


@dataclass
class ExtractConfig:
    """Parameters for L2->L1 extraction."""
    column: str
    group_by: Optional[str] = None
    group_agg: Optional[str] = None


@dataclass
class AggregateConfig:
    """Parameters for L1->L0 aggregation."""
    method: str  # mean, sum, count, min, max


@dataclass
class DimensionConfig:
    """Parameters for ascent L1->L2 dimension application."""
    name: str
    column: str
    rules: dict[str, str]


@dataclass
class DescentConfig:
    """Configuration for descent phase (L3->L0)."""
    categorization: CategorizeConfig
    extraction: ExtractConfig
    aggregation: AggregateConfig


@dataclass
class AscentConfig:
    """Configuration for ascent phase (L0->L3)."""
    l1_recovery: str = "source_values"
    l2_dimension: Optional[DimensionConfig] = None
    l3_enrichment: str = "automatic"


@dataclass
class ExpectedOutput:
    """Expected values at each level for verification."""
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    datum_value: Optional[float] = None
    categories: Optional[dict[str, int]] = None
    tolerance: float = 0.01


@dataclass
class TestDataset:
    """Complete test dataset configuration."""
    name: str
    source_files: list[SourceFile]
    join_config: JoinConfig
    descent_config: DescentConfig
    ascent_config: AscentConfig
    expected_outputs: dict[str, ExpectedOutput] = field(default_factory=dict)


# =============================================================================
# Pre-configured Test Datasets
# =============================================================================

# Base path for test data
TEST_DATA_BASE = Path("/Users/arthursarazin/Documents/data_redesign_method/test_data")
SCREENSHOTS_BASE = Path("/Users/arthursarazin/Documents/data_redesign_method/tests/artifacts/screenshots")

# Schools Dataset (test0) - L0 = 88.25365853658536
SCHOOLS_DATASET = TestDataset(
    name="schools",
    source_files=[
        SourceFile(
            filename="fr-en-college-effectifs-niveau-sexe-lv.csv",
            path=str(TEST_DATA_BASE / "test0" / "fr-en-college-effectifs-niveau-sexe-lv.csv"),
            expected_rows=50164,
            expected_columns=23,
        ),
        SourceFile(
            filename="fr-en-indicateurs-valeur-ajoutee-colleges.csv",
            path=str(TEST_DATA_BASE / "test0" / "fr-en-indicateurs-valeur-ajoutee-colleges.csv"),
            expected_rows=20053,
            expected_columns=89,
        ),
    ],
    join_config=JoinConfig(
        left_column="Patronyme",
        right_column="Nom de l'etablissement",
        threshold=0.85,
        model="intfloat/multilingual-e5-small",
    ),
    descent_config=DescentConfig(
        categorization=CategorizeConfig(
            dimension_name="location_type",
            column="nombre_eleves_total",
            rules={"downtown": ">200", "countryside": "<=200"},
        ),
        extraction=ExtractConfig(
            column="Taux de reussite G",
            group_by=None,
        ),
        aggregation=AggregateConfig(method="mean"),
    ),
    ascent_config=AscentConfig(
        l2_dimension=DimensionConfig(
            name="performance_category",
            column="extracted_value",
            rules={"above_median": ">median", "below_median": "<=median"},
        ),
    ),
    expected_outputs={
        "L4": ExpectedOutput(row_count=2),  # 2 files
        "L3": ExpectedOutput(row_count=410, column_count=111),
        "L2_descent": ExpectedOutput(
            row_count=410,
            categories={"downtown": 281, "countryside": 129},
        ),
        "L1": ExpectedOutput(row_count=410),
        "L0": ExpectedOutput(datum_value=88.25365853658536),
        "L1_ascent": ExpectedOutput(row_count=410),
        "L2_ascent": ExpectedOutput(
            row_count=410,
            categories={"above_median": 208, "below_median": 202},
        ),
        "L3_ascent": ExpectedOutput(row_count=410, column_count=112),
    },
)

# ADEME Dataset (test1) - L0 = 69586180.93
ADEME_DATASET = TestDataset(
    name="ademe",
    source_files=[
        SourceFile(
            filename="ECS.csv",
            path=str(TEST_DATA_BASE / "test1" / "ECS.csv"),
            expected_rows=428,
            expected_columns=7,
        ),
        SourceFile(
            filename="Les aides financieres ADEME.csv",
            path=str(TEST_DATA_BASE / "test1" / "Les aides financieres ADEME.csv"),
            expected_rows=37339,
            expected_columns=41,
        ),
    ],
    join_config=JoinConfig(
        left_column="dispositifAide",
        right_column="type_aides_financieres",
        threshold=0.75,
        model="intfloat/multilingual-e5-small",
    ),
    descent_config=DescentConfig(
        categorization=CategorizeConfig(
            dimension_name="funding_frequency",
            column="nomBeneficiaire",
            rules={"single_funding": "count==1", "multiple_funding": "count>1"},
        ),
        extraction=ExtractConfig(
            column="montant",
            group_by="nomBeneficiaire",
            group_agg="sum",
        ),
        aggregation=AggregateConfig(method="sum"),
    ),
    ascent_config=AscentConfig(
        l2_dimension=DimensionConfig(
            name="funding_size",
            column="extracted_value",
            rules={"above_10k": ">10000", "below_10k": "<=10000"},
        ),
    ),
    expected_outputs={
        "L4": ExpectedOutput(row_count=2),  # 2 files
        "L3": ExpectedOutput(row_count=500, column_count=47),
        "L2_descent": ExpectedOutput(
            row_count=500,
            categories={"single_funding": 412, "multiple_funding": 88},
        ),
        "L1": ExpectedOutput(row_count=450),  # Unique recipients
        "L0": ExpectedOutput(datum_value=69586180.93),
        "L1_ascent": ExpectedOutput(row_count=450),
        "L2_ascent": ExpectedOutput(
            row_count=450,
            categories={"above_10k": 301, "below_10k": 149},
        ),
        "L3_ascent": ExpectedOutput(row_count=450, column_count=48),
    },
)


# =============================================================================
# Helper Functions for MCP Test Execution (T004-T008)
# =============================================================================

def get_screenshot_path(dataset_name: str, step_number: int, description: str) -> str:
    """
    Generate screenshot path with proper naming convention (T007, T033).

    Format: {step_number:02d}_{description}.png

    Args:
        dataset_name: "schools" or "ademe"
        step_number: 1-9 step number
        description: Short description like "initial_state", "l4_uploaded"

    Returns:
        Absolute path for the screenshot file
    """
    filename = f"{step_number:02d}_{description}.png"
    return str(SCREENSHOTS_BASE / f"{dataset_name}_mcp_cycle" / filename)


def verify_datum_value(actual: float, expected: float, tolerance: float = 0.01) -> tuple[bool, str]:
    """
    Verify L0 datum value is within tolerance (T008).

    Args:
        actual: The actual computed value
        expected: The expected value from session export
        tolerance: Acceptable relative difference (default 1%)

    Returns:
        Tuple of (passed, message)
    """
    if expected == 0:
        passed = actual == 0
        diff = actual
    else:
        diff = abs(actual - expected) / abs(expected)
        passed = diff <= tolerance

    if passed:
        return True, f"PASS: {actual:.2f} matches expected {expected:.2f} (diff: {diff:.4%})"
    else:
        return False, f"FAIL: {actual:.2f} != expected {expected:.2f} (diff: {diff:.4%}, tolerance: {tolerance:.2%})"


def verify_row_count(actual: int, expected: int) -> tuple[bool, str]:
    """
    Verify row count matches expected (T008).

    Args:
        actual: The actual row count
        expected: The expected row count

    Returns:
        Tuple of (passed, message)
    """
    passed = actual == expected
    if passed:
        return True, f"PASS: {actual} rows matches expected"
    else:
        return False, f"FAIL: {actual} rows != expected {expected} rows"


def verify_category_distribution(
    actual: dict[str, int],
    expected: dict[str, int]
) -> tuple[bool, str]:
    """
    Verify category distribution matches expected (T008).

    Args:
        actual: Actual category counts
        expected: Expected category counts

    Returns:
        Tuple of (passed, message)
    """
    messages = []
    all_passed = True

    for category, expected_count in expected.items():
        actual_count = actual.get(category, 0)
        if actual_count == expected_count:
            messages.append(f"  {category}: {actual_count} PASS")
        else:
            messages.append(f"  {category}: {actual_count} != {expected_count} FAIL")
            all_passed = False

    # Check for unexpected categories
    for category in actual:
        if category not in expected:
            messages.append(f"  {category}: {actual[category]} (unexpected)")
            all_passed = False

    status = "PASS" if all_passed else "FAIL"
    return all_passed, f"{status}: Category distribution\n" + "\n".join(messages)


def format_step_summary(
    step_number: int,
    step_name: str,
    level_from: str,
    level_to: str,
    status: str,
    details: Optional[str] = None
) -> str:
    """
    Format a step summary for reporting.

    Args:
        step_number: Step number (1-9)
        step_name: Human-readable step name
        level_from: Source level (e.g., "L4")
        level_to: Target level (e.g., "L3")
        status: "PASS", "FAIL", or "SKIP"
        details: Optional additional details

    Returns:
        Formatted summary string
    """
    emoji = {"PASS": "[OK]", "FAIL": "[X]", "SKIP": "[-]"}.get(status, "[ ]")
    summary = f"Step {step_number} {emoji}: {step_name} ({level_from} -> {level_to})"
    if details:
        summary += f"\n  {details}"
    return summary


# =============================================================================
# Test Step Reference (for MCP execution guidance)
# =============================================================================

SCHOOLS_TEST_STEPS = [
    {"step": 1, "action": "navigate", "level": "start", "target": "L4", "description": "Navigate to app"},
    {"step": 2, "action": "upload", "level": "L4", "target": "L4", "description": "Upload schools files"},
    {"step": 3, "action": "join", "level": "L4", "target": "L3", "description": "Configure semantic join"},
    {"step": 4, "action": "categorize", "level": "L3", "target": "L2", "description": "Apply location_type"},
    {"step": 5, "action": "extract", "level": "L2", "target": "L1", "description": "Extract Taux de reussite G"},
    {"step": 6, "action": "aggregate", "level": "L1", "target": "L0", "description": "Compute MEAN"},
    {"step": 7, "action": "ascend", "level": "L0", "target": "L1", "description": "Recover source values"},
    {"step": 8, "action": "dimension", "level": "L1", "target": "L2", "description": "Apply performance_category"},
    {"step": 9, "action": "enrich", "level": "L2", "target": "L3", "description": "Complete enrichment"},
]

ADEME_TEST_STEPS = [
    {"step": 1, "action": "navigate", "level": "start", "target": "L4", "description": "Navigate to app"},
    {"step": 2, "action": "upload", "level": "L4", "target": "L4", "description": "Upload ADEME files"},
    {"step": 3, "action": "join", "level": "L4", "target": "L3", "description": "Configure semantic join"},
    {"step": 4, "action": "categorize", "level": "L3", "target": "L2", "description": "Apply funding_frequency"},
    {"step": 5, "action": "extract", "level": "L2", "target": "L1", "description": "Group montant by recipient"},
    {"step": 6, "action": "aggregate", "level": "L1", "target": "L0", "description": "Compute SUM"},
    {"step": 7, "action": "ascend", "level": "L0", "target": "L1", "description": "Recover source values"},
    {"step": 8, "action": "dimension", "level": "L1", "target": "L2", "description": "Apply funding_size"},
    {"step": 9, "action": "enrich", "level": "L2", "target": "L3", "description": "Complete enrichment"},
]
