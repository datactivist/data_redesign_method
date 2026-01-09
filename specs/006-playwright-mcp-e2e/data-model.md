# Data Model: Playwright MCP E2E Testing

**Feature**: 006-playwright-mcp-e2e
**Date**: 2025-12-09

## Entities

### TestCycle

Represents a complete descent/ascent test execution.

| Field | Type | Description |
|-------|------|-------------|
| cycle_id | string | Unique identifier for the test run |
| dataset_name | string | "schools" or "ademe" |
| start_time | datetime | When test started |
| end_time | datetime | When test completed (nullable) |
| status | enum | pending, running, passed, failed |
| steps | list[TestStep] | Ordered list of steps executed |
| screenshots | list[Screenshot] | Screenshots captured during test |

### TestStep

Represents a single level transition in the cycle.

| Field | Type | Description |
|-------|------|-------------|
| step_id | string | Unique step identifier |
| step_number | integer | 1-8 (position in cycle) |
| action | enum | upload, join, categorize, extract, aggregate, ascend_l1, ascend_l2, ascend_l3 |
| source_level | integer | Level before action (0-4) |
| target_level | integer | Level after action (0-4) |
| parameters | dict | Action-specific parameters |
| expected_output | ExpectedOutput | What to verify |
| actual_output | dict | What was observed (nullable) |
| status | enum | pending, passed, failed, skipped |
| error_message | string | Error details if failed (nullable) |
| timestamp | datetime | When step was executed |

### ExpectedOutput

Defines what to verify at each step.

| Field | Type | Description |
|-------|------|-------------|
| row_count | integer | Expected number of rows (nullable) |
| column_count | integer | Expected number of columns (nullable) |
| datum_value | float | Expected L0 value (nullable) |
| categories | dict[string, int] | Expected category distribution (nullable) |
| tolerance | float | Acceptable deviation percentage (default 0.05) |

### Screenshot

Captures visual state at a point in time.

| Field | Type | Description |
|-------|------|-------------|
| screenshot_id | string | Unique identifier |
| step_id | string | Associated test step |
| filename | string | File path relative to screenshots dir |
| timestamp | datetime | When captured |
| description | string | Human-readable step description |

### TestDataset

Reference data for test execution.

| Field | Type | Description |
|-------|------|-------------|
| name | string | "schools" or "ademe" |
| source_files | list[SourceFile] | CSV files to upload |
| join_config | JoinConfig | Semantic join parameters |
| descent_config | DescentConfig | L3→L2→L1→L0 parameters |
| ascent_config | AscentConfig | L0→L1→L2→L3 parameters |
| expected_outputs | dict[string, ExpectedOutput] | Expected outputs per level |

### SourceFile

CSV file to upload for testing.

| Field | Type | Description |
|-------|------|-------------|
| filename | string | File name |
| path | string | Absolute path |
| expected_rows | integer | Row count for verification |
| expected_columns | integer | Column count |

### JoinConfig

Parameters for L4→L3 semantic join.

| Field | Type | Description |
|-------|------|-------------|
| left_column | string | Column name from first file |
| right_column | string | Column name from second file |
| threshold | float | Similarity threshold (0.0-1.0) |
| model | string | Embedding model name |

### DescentConfig

Parameters for descent phase (L3→L0).

| Field | Type | Description |
|-------|------|-------------|
| categorization | CategorizeConfig | L3→L2 config |
| extraction | ExtractConfig | L2→L1 config |
| aggregation | AggregateConfig | L1→L0 config |

### AscentConfig

Parameters for ascent phase (L0→L3).

| Field | Type | Description |
|-------|------|-------------|
| l1_recovery | string | "source_values" (automatic) |
| l2_dimension | DimensionConfig | New categorization for L2 |
| l3_enrichment | string | "automatic" |

---

## Relationships

```text
TestCycle 1──* TestStep (ordered by step_number)
TestCycle 1──* Screenshot
TestStep 1──1 ExpectedOutput
TestStep 1──* Screenshot
TestDataset 1──* SourceFile
TestDataset 1──1 JoinConfig
TestDataset 1──1 DescentConfig
TestDataset 1──1 AscentConfig
```

## Validation Rules

1. **TestCycle.steps**: Must contain exactly 8 steps for complete cycle
2. **TestStep.step_number**: Must be sequential 1-8
3. **TestStep.source_level/target_level**: Must differ by exactly 1
4. **ExpectedOutput.tolerance**: Default 0.05 (5%), max 0.10 (10%)
5. **JoinConfig.threshold**: Must be between 0.0 and 1.0
6. **Screenshot.filename**: Must match pattern `{step_number:02d}_{timestamp}_{description}.png`

## State Transitions

### TestCycle Status

```text
pending → running → passed
                  → failed
```

### TestStep Status

```text
pending → passed (if actual matches expected within tolerance)
        → failed (if mismatch or error)
        → skipped (if previous step failed)
```

---

## Test Dataset Definitions

### Schools Dataset

```yaml
name: schools
source_files:
  - filename: fr-en-college-effectifs-niveau-sexe-lv.csv
    expected_rows: 50164
  - filename: fr-en-indicateurs-valeur-ajoutee-colleges.csv
    expected_rows: 20053

join_config:
  left_column: Dénomination principale
  right_column: Nom de l'établissement
  threshold: 0.85
  model: intfloat/multilingual-e5-small

descent_config:
  categorization:
    dimension_name: location_type
    column: nombre_eleves_total
    rules: {downtown: ">200", countryside: "<=200"}
  extraction:
    column: Taux de reussite G
    group_by: null
  aggregation:
    method: mean

ascent_config:
  l2_dimension:
    name: performance_category
    column: extracted_value
    rules: {above_median: ">median", below_median: "<=median"}

expected_outputs:
  L3: {row_count: 410, column_count: 111}
  L2: {row_count: 410, categories: {downtown: 281, countryside: 129}}
  L1: {row_count: 410}
  L0: {datum_value: 88.25365853658536}
  ascent_L2: {categories: {above_median: 208, below_median: 202}}
  ascent_L3: {row_count: 410, column_count: 112}
```

### ADEME Dataset

```yaml
name: ademe
source_files:
  - filename: ECS.csv
    expected_rows: 428
  - filename: Les aides financieres ADEME.csv
    expected_rows: 37339

join_config:
  left_column: dispositifAide
  right_column: type_aides_financieres
  threshold: 0.75
  model: intfloat/multilingual-e5-small

descent_config:
  categorization:
    dimension_name: funding_frequency
    column: nomBeneficiaire
    rules: {single_funding: "count==1", multiple_funding: "count>1"}
  extraction:
    column: montant
    group_by: nomBeneficiaire
    group_agg: sum
  aggregation:
    method: sum

ascent_config:
  l2_dimension:
    name: funding_size
    column: extracted_value
    rules: {above_10k: ">10000", below_10k: "<=10000"}

expected_outputs:
  L3: {row_count: 500, column_count: 47}
  L2: {row_count: 500, categories: {single_funding: 412, multiple_funding: 88}}
  L1: {row_count: 450}
  L0: {datum_value: 69586180.93}
  ascent_L2: {categories: {above_10k: 301, below_10k: 149}}
  ascent_L3: {row_count: 450, column_count: 48}
```
