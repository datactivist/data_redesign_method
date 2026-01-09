# Contract: Enrichment Functions

**Feature**: Ascent Functionality
**Entity**: EnrichmentFunction
**Date**: 2025-12-02

## Interface Contract

### EnrichmentFunction

```python
@dataclass
class EnrichmentFunction:
    name: str                           # REQUIRED: unique identifier
    description: str                    # REQUIRED: user-facing description
    source_level: ComplexityLevel       # REQUIRED: L0, L1, or L2
    target_level: ComplexityLevel       # REQUIRED: must be source_level + 1
    func: Callable[[Any], Any]          # REQUIRED: the enrichment callable
    requires_context: bool = False      # OPTIONAL: whether parent data needed
```

### Validation Rules

1. `target_level.value == source_level.value + 1` (single-step ascent only)
2. `target_level.value <= 3` (cannot ascend to L4)
3. `name` must be unique within registry
4. `func` must accept data matching source level type:
   - L0 source: scalar or Level0Dataset
   - L1 source: pd.Series or Level1Dataset
   - L2 source: pd.DataFrame or Level2Dataset

### Output Requirements

| Source | Target | Output Type |
|--------|--------|-------------|
| L0 | L1 | pd.Series |
| L1 | L2 | pd.DataFrame |
| L2 | L3 | pd.DataFrame with additional columns |

## EnrichmentRegistry Contract

### Methods

```python
class EnrichmentRegistry:
    def register(self, func: EnrichmentFunction) -> None:
        """
        Register an enrichment function.

        Raises:
            ValueError: If name already registered
            ValueError: If target_level != source_level + 1
        """

    def get(self, name: str) -> EnrichmentFunction:
        """
        Retrieve enrichment function by name.

        Raises:
            KeyError: If name not found
        """

    def list_for_transition(
        self,
        source: ComplexityLevel,
        target: ComplexityLevel
    ) -> List[EnrichmentFunction]:
        """
        List all functions available for a specific transition.

        Returns:
            List of matching EnrichmentFunction objects (may be empty)
        """

    def get_defaults(
        self,
        source: ComplexityLevel,
        target: ComplexityLevel
    ) -> List[EnrichmentFunction]:
        """
        Get default (built-in) functions for a transition.

        Returns:
            At least 2 functions per valid transition (per FR-006)
        """
```

### Default Functions Required (FR-006)

| Transition | Minimum Defaults |
|------------|------------------|
| L0 → L1 | 2 |
| L1 → L2 | 2 |
| L2 → L3 | 2 |

## Behavior Specifications

### Scenario: Register valid enrichment function

```gherkin
Given an EnrichmentRegistry instance
When I register an EnrichmentFunction with source_level=L0, target_level=L1
Then the function is stored in the registry
And list_for_transition(L0, L1) includes this function
```

### Scenario: Reject invalid level transition

```gherkin
Given an EnrichmentRegistry instance
When I try to register a function with source_level=L0, target_level=L2
Then a ValueError is raised with message containing "single-step"
```

### Scenario: Block L3 to L4 ascent

```gherkin
Given an EnrichmentRegistry instance
When I try to register a function with source_level=L3, target_level=L4
Then a ValueError is raised with message "L4 is entry-only"
```

### Scenario: Get defaults for L0→L1

```gherkin
Given an EnrichmentRegistry with defaults loaded
When I call get_defaults(L0, L1)
Then I receive at least 2 EnrichmentFunction objects
And each has source_level=L0 and target_level=L1
```

## Error Handling

| Error Case | Exception | Message Pattern |
|------------|-----------|-----------------|
| Duplicate name | `ValueError` | "Function '{name}' already registered" |
| Invalid transition | `ValueError` | "Target must be source + 1" |
| L4 target | `ValueError` | "L4 is entry-only, cannot ascend to L4" |
| Unknown function | `KeyError` | "No enrichment function named '{name}'" |
| Enrichment returns wrong type | `TypeError` | "Expected {expected}, got {actual}" |
