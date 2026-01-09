# Contract: Dimension Definitions

**Feature**: Ascent Functionality
**Entity**: DimensionDefinition
**Date**: 2025-12-02

## Interface Contract

### DimensionDefinition

```python
@dataclass
class DimensionDefinition:
    name: str                                    # REQUIRED: column name to create
    description: str                             # REQUIRED: user-facing description
    possible_values: List[str]                   # REQUIRED: known categories
    classifier: Callable[[Any], str]             # REQUIRED: classification function
    default_value: str = "Unknown"               # OPTIONAL: fallback category
```

### Validation Rules

1. `name` must be a valid Python identifier (for DataFrame column)
2. `possible_values` must be non-empty
3. `default_value` should be in `possible_values`
4. `classifier` must accept a single item and return a string

### Built-in Dimension Types

#### For L1 → L2 Transitions

| Dimension Name | Possible Values | Classification Logic |
|----------------|-----------------|----------------------|
| `business_object` | ["revenue", "volume", "ETP", "other"] | Pattern match on item name |
| `is_calculated` | ["calculated", "raw"] | Check for formula indicators |
| `has_weight` | ["weighted", "unweighted"] | Check for weight suffix |
| `has_rse` | ["has_rse", "no_rse"] | Check for RSE indicators |

#### For L2 → L3 Transitions

| Dimension Name | Possible Values | Classification Logic |
|----------------|-----------------|----------------------|
| `client_segment` | ["B2B", "B2C", "Government", "Mixed", "Unknown"] | Business rules |
| `sales_location` | ["Domestic", "Export", "Both", "Unknown"] | Geographic patterns |
| `product_segment` | ["ProductA", "ProductB", "Services", "Unknown"] | Product line rules |
| `financial_view` | ["Revenue", "Cost", "Margin", "Unknown"] | Financial classification |
| `lifecycle_view` | ["Acquisition", "Retention", "Churn", "Unknown"] | Lifecycle stage |

## Behavior Specifications

### Scenario: Apply dimension to vector

```gherkin
Given a DimensionDefinition with name="business_object"
And a Level1Dataset with items ["revenue_total", "volume_sales", "ETP_count"]
When the dimension classifier is applied to each item
Then a new column "business_object" is created
And values are ["revenue", "volume", "ETP"]
```

### Scenario: Handle unknown items

```gherkin
Given a DimensionDefinition with default_value="other"
And an item "xyz_unknown_metric"
When the classifier cannot determine a category
Then the value "other" is assigned
And no error is raised
```

### Scenario: Apply multiple dimensions

```gherkin
Given dimensions [business_object, is_calculated, has_weight]
And a Level1Dataset with 100 items
When all dimensions are applied
Then a Level2Dataset is created with 3 new columns
And each row has values for all 3 dimensions
```

## DimensionRegistry Contract

```python
class DimensionRegistry:
    def register(self, dimension: DimensionDefinition) -> None:
        """
        Register a dimension definition.
        """

    def get(self, name: str) -> DimensionDefinition:
        """
        Retrieve dimension by name.

        Raises:
            KeyError: If name not found
        """

    def list_for_transition(
        self,
        source: ComplexityLevel,
        target: ComplexityLevel
    ) -> List[DimensionDefinition]:
        """
        List dimensions applicable to a transition.
        L1→L2: categorical dimensions
        L2→L3: analytic dimensions
        """

    def get_defaults(
        self,
        transition: Tuple[ComplexityLevel, ComplexityLevel]
    ) -> List[DimensionDefinition]:
        """
        Get default dimensions for a transition.
        """
```

## Error Handling

| Error Case | Exception | Message Pattern |
|------------|-----------|-----------------|
| Invalid column name | `ValueError` | "'{name}' is not a valid identifier" |
| Empty possible_values | `ValueError` | "possible_values cannot be empty" |
| Classifier returns non-string | `TypeError` | "Classifier must return str" |
| Classifier raises exception | Caught | Returns default_value |
