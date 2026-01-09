# API Contract: Dataset Redesign Package

**Package**: `intuitiveness`
**Version**: 0.1.0
**Date**: 2025-12-02

## Public API Overview

```python
from intuitiveness import (
    # Core classes
    Dataset,
    ComplexityLevel,

    # Operations
    descend,
    ascend,
    measure_complexity,

    # Navigation
    NavigationSession,

    # Utilities
    trace_lineage,
)
```

---

## Core Classes

### Dataset

```python
@dataclass
class Dataset:
    """Wrapper for data at any complexity level (L0-L4)."""

    data: Any
    level: ComplexityLevel
    complexity_order: float
    lineage: Optional[DataLineage] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "Dataset":
        """Create L0 Dataset from a single value."""

    @classmethod
    def from_series(cls, series: pd.Series) -> "Dataset":
        """Create L1 Dataset from a pandas Series."""

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Dataset":
        """Create L2 Dataset from a pandas DataFrame."""

    @classmethod
    def from_graph(cls, graph: nx.Graph) -> "Dataset":
        """Create L3 Dataset from a networkx Graph."""

    @classmethod
    def from_sources(cls, sources: Dict[str, Any]) -> "Dataset":
        """Create L4 Dataset from disparate data sources."""
```

### ComplexityLevel

```python
class ComplexityLevel(Enum):
    """Enumeration of the five abstraction levels."""

    DATUM = 0       # L0: Single entity-attribute-value
    VECTOR = 1      # L1: Single dimension
    TABLE = 2       # L2: Two-dimensional
    LINKABLE = 3    # L3: Multi-level with relationships
    UNLINKABLE = 4  # L4: Multi-level without relationships
```

---

## Descent Operations

### descend()

```python
def descend(
    dataset: Dataset,
    *,
    # L4 → L3: Linking
    linking_function: Optional[Callable[[Dict], nx.Graph]] = None,

    # L3 → L2: Query
    entity_type: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,

    # L2 → L1: Selection
    column: Optional[str] = None,
    row_filter: Optional[Callable[[pd.Series], bool]] = None,

    # L1 → L0: Aggregation
    aggregation: Optional[str] = None,  # "count", "sum", "mean", "min", "max"
    custom_aggregator: Optional[Callable[[Sequence], Any]] = None,
) -> Dataset:
    """
    Reduce dataset complexity by one level.

    Parameters depend on source level:
    - L4 → L3: Requires `linking_function`
    - L3 → L2: Requires `entity_type`, optional `filters`
    - L2 → L1: Requires `column`, optional `row_filter`
    - L1 → L0: Requires `aggregation` or `custom_aggregator`

    Returns:
        New Dataset at one level lower with lineage attached.

    Raises:
        ValueError: If required parameters missing for level transition.
        TypeError: If dataset level is L0 (cannot descend further).

    Example:
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table = Dataset.from_dataframe(df)
        >>> vector = descend(table, column="a")
        >>> value = descend(vector, aggregation="sum")
        >>> value.data
        6
    """
```

---

## Ascent Operations

### ascend()

```python
def ascend(
    dataset: Dataset,
    source: Dataset,
    *,
    # L0 → L1: Enrichment
    selection_criteria: Optional[Dict[str, Any]] = None,

    # L1 → L2: Dimensioning
    dimensions: Optional[List[str]] = None,

    # L2 → L3: Hierarchical grouping
    groupings: Optional[List[str]] = None,
    relationships: Optional[Dict[str, str]] = None,
) -> Dataset:
    """
    Increase dataset complexity by one level using source data.

    Parameters depend on target level:
    - L0 → L1: Requires `source` (L1+), `selection_criteria`
    - L1 → L2: Requires `source` (L2+), `dimensions`
    - L2 → L3: Requires `groupings`, `relationships`

    Returns:
        New Dataset at one level higher with lineage attached.

    Raises:
        ValueError: If required parameters missing for level transition.
        TypeError: If dataset level is L3 (cannot ascend to L4).

    Example:
        >>> value = Dataset.from_value(6)
        >>> source_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> source = Dataset.from_dataframe(source_df)
        >>> vector = ascend(value, source, selection_criteria={"column": "a"})
        >>> vector.level
        ComplexityLevel.VECTOR
    """
```

---

## Complexity Measurement

### measure_complexity()

```python
def measure_complexity(dataset: Dataset) -> Dict[str, Any]:
    """
    Analyze and report dataset complexity.

    Returns:
        {
            "level": ComplexityLevel,
            "level_name": str,           # "DATUM", "VECTOR", etc.
            "complexity_order": float,   # Numeric complexity value
            "complexity_formula": str,   # e.g., "C(2^n)" for L2
            "dimensions": {
                "rows": int | None,
                "columns": int | None,
                "nodes": int | None,      # For L3
                "edges": int | None,      # For L3
                "sources": int | None,    # For L4
            },
            "reduction_from_l4": float | None,  # Percentage if descended from L4
        }

    Example:
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table = Dataset.from_dataframe(df)
        >>> info = measure_complexity(table)
        >>> info["level_name"]
        "TABLE"
        >>> info["complexity_order"]
        6.0  # 3 rows × 2 columns
    """
```

---

## Navigation API

### NavigationSession

```python
class NavigationSession:
    """Stateful exploration of dataset hierarchy."""

    def __init__(self, dataset: Dataset) -> None:
        """
        Initialize navigation session.

        Args:
            dataset: Must be L4 (UNLINKABLE) - the entry point.

        Raises:
            ValueError: If dataset is not L4.
        """

    @property
    def current_level(self) -> ComplexityLevel:
        """Current abstraction level."""

    @property
    def current_node(self) -> Any:
        """Current node/data at this position."""

    @property
    def state(self) -> NavigationState:
        """Current state: ENTRY, EXPLORING, or EXITED."""

    def descend(self, **params) -> "NavigationSession":
        """
        Move down one level.

        Parameters same as descend() function for current level transition.

        Returns:
            Self for chaining.

        Raises:
            NavigationError: If at L1 (cannot descend to L0 in navigation mode).
        """

    def ascend(self) -> "NavigationSession":
        """
        Move up one level.

        Returns:
            Self for chaining.

        Raises:
            NavigationError: If at L3 (cannot return to L4).
        """

    def move_horizontal(self, node_id: str) -> "NavigationSession":
        """
        Move to related node at same level.

        Args:
            node_id: Identifier of target node.

        Returns:
            Self for chaining.

        Raises:
            NavigationError: If node_id not found or not related.
        """

    def get_available_moves(self) -> Dict[str, List[str]]:
        """
        List valid moves from current position.

        Returns:
            {
                "descend": [...],     # Available nodes at lower level
                "ascend": [...],      # Available nodes at higher level (empty if L3)
                "horizontal": [...],  # Related nodes at same level
            }
        """

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get navigation path.

        Returns:
            List of {"level": ..., "node_id": ..., "action": ..., "timestamp": ...}
        """

    def exit(self) -> None:
        """
        End navigation session, preserving position.
        """

    @classmethod
    def resume(cls, session_id: str) -> "NavigationSession":
        """
        Resume a previously exited session.

        Args:
            session_id: UUID from exited session.

        Returns:
            Restored NavigationSession.

        Raises:
            SessionNotFoundError: If session expired or not found.
        """

    def save(self, path: str) -> None:
        """Save session to file for later resumption."""

    @classmethod
    def load(cls, path: str) -> "NavigationSession":
        """Load session from file."""
```

---

## Lineage Utilities

### trace_lineage()

```python
def trace_lineage(
    dataset: Dataset,
    *,
    row_index: Optional[int] = None,
    column_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Trace a value back to its source.

    Args:
        dataset: Dataset to trace from.
        row_index: Specific row (for L2+).
        column_name: Specific column (for L1+).

    Returns:
        List of transformation steps from current to source:
        [
            {
                "operation": str,
                "parameters": dict,
                "source_dataset_id": str,
                "source_location": {"row": int, "column": str, ...},
                "timestamp": datetime,
            },
            ...  # Oldest transformation last
        ]

    Example:
        >>> # After: value = descend(descend(table, column="a"), aggregation="sum")
        >>> history = trace_lineage(value)
        >>> len(history)
        2
        >>> history[0]["operation"]
        "aggregate"
        >>> history[1]["operation"]
        "select"
    """
```

---

## Exceptions

```python
class IntuitivenessError(Exception):
    """Base exception for all package errors."""

class ValidationError(IntuitivenessError):
    """Raised when input validation fails."""

class OperationError(IntuitivenessError):
    """Raised when an operation cannot be performed."""

class NavigationError(IntuitivenessError):
    """Raised when navigation action is invalid."""

class SessionNotFoundError(NavigationError):
    """Raised when trying to resume non-existent session."""
```

---

## Type Aliases

```python
from typing import TypeAlias

# Common callback types
LinkingFunction: TypeAlias = Callable[[Dict[str, Any]], nx.Graph]
RowFilter: TypeAlias = Callable[[pd.Series], bool]
CustomAggregator: TypeAlias = Callable[[Sequence[Any]], Any]

# Data types at each level
L0Data: TypeAlias = Union[int, float, str, bool, None]
L1Data: TypeAlias = Union[List, pd.Series, np.ndarray]
L2Data: TypeAlias = pd.DataFrame
L3Data: TypeAlias = nx.Graph
L4Data: TypeAlias = Dict[str, Union[L0Data, L1Data, L2Data, L3Data]]
```
