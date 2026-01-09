# Contract: Ascent Operations

**Feature**: Ascent Functionality
**Entity**: AscentOperation
**Date**: 2025-12-02

## Interface Contract

### AscentOperation

```python
@dataclass
class AscentOperation:
    id: str                                      # REQUIRED: UUID
    source_level: ComplexityLevel                # REQUIRED: L0, L1, or L2
    target_level: ComplexityLevel                # REQUIRED: L1, L2, or L3
    enrichment_function: str                     # REQUIRED: name of function used
    dimensions_added: List[str]                  # REQUIRED: may be empty for L0→L1
    timestamp: datetime                          # REQUIRED: when operation occurred
    source_data_hash: str                        # REQUIRED: integrity check
    result_data_hash: str                        # REQUIRED: integrity check
    row_count_before: int                        # REQUIRED: items before
    row_count_after: int                         # REQUIRED: items after
```

### Validation Rules

1. `target_level.value == source_level.value + 1`
2. `target_level.value <= 3` (cannot target L4)
3. `row_count_before == row_count_after` (data integrity - FR-005)
4. `source_data_hash` must match actual input data
5. `id` must be valid UUID

## Behavior Specifications

### Scenario: Perform L0 to L1 ascent

```gherkin
Given a Level0Dataset with value=523 and parent_data available
And an enrichment function "source_expansion"
When ascent is performed
Then an AscentOperation is created
And source_level is LEVEL_0
And target_level is LEVEL_1
And row_count_after equals length of parent_data
And a Level1Dataset is returned with 523 items
```

### Scenario: Perform L1 to L2 ascent with dimensions

```gherkin
Given a Level1Dataset with 100 items
And dimensions ["business_object", "is_calculated"]
When ascent is performed
Then an AscentOperation is created
And dimensions_added is ["business_object", "is_calculated"]
And row_count_before equals 100
And row_count_after equals 100
And a Level2Dataset is returned with 100 rows and 3 columns
```

### Scenario: Attempt L3 to L4 ascent (blocked)

```gherkin
Given a Level3Dataset
When user attempts to ascend to L4
Then a NavigationError is raised
And message contains "L4 is entry-only"
And no AscentOperation is created
```

### Scenario: Data integrity validation

```gherkin
Given an AscentOperation with row_count_before=100, row_count_after=100
When validate_integrity() is called
Then True is returned

Given an AscentOperation with row_count_before=100, row_count_after=99
When validate_integrity() is called
Then False is returned
```

## Redesigner.increase_complexity Contract

```python
@staticmethod
def increase_complexity(
    dataset: Dataset,
    target_level: ComplexityLevel,
    enrichment_func: Optional[EnrichmentFunction] = None,
    dimensions: Optional[List[DimensionDefinition]] = None,
    **kwargs
) -> Dataset:
    """
    Increases complexity (Ascent).

    Args:
        dataset: Source dataset (L0, L1, or L2)
        target_level: Must be source + 1, max L3
        enrichment_func: Function to enrich data (optional if defaults available)
        dimensions: Dimensions to add (L1→L2 and L2→L3)

    Returns:
        Dataset at target_level

    Raises:
        ValueError: If target is not source + 1
        ValueError: If target is L4
        ValueError: If required enrichment_func missing
        NavigationError: If ascent fails validation
    """
```

### Dispatch Logic

| Source | Target | Required Parameters |
|--------|--------|---------------------|
| L0 | L1 | `enrichment_func` (or default with parent_data) |
| L1 | L2 | `enrichment_func` or `dimensions` |
| L2 | L3 | `enrichment_func` or `dimensions` |

## NavigationSession.ascend Contract

```python
def ascend(self, **params) -> 'NavigationSession':
    """
    Move up one level.

    Params:
        enrichment_func: str - Name of enrichment function (optional)
        dimensions: List[str] - Names of dimensions to add (optional)

    Returns:
        Self for method chaining

    Raises:
        NavigationError: If session exited
        NavigationError: If at L4 (already at top)
        NavigationError: If at L3 (L4 is entry-only)
        NavigationError: If enrichment fails
    """
```

## Error Handling

| Error Case | Exception | Message Pattern |
|------------|-----------|-----------------|
| At L4 | `NavigationError` | "Already at L4, cannot ascend" |
| At L3 | `NavigationError` | "L4 is entry-only; cannot return" |
| No enrichment function | `ValueError` | "Ascent requires enrichment function or defaults" |
| Row count mismatch | `NavigationError` | "Data integrity error: row count changed" |
| Session exited | `NavigationError` | "Session has exited. Use resume()" |
