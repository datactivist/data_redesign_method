from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union
import pandas as pd
import networkx as nx

class ComplexityLevel(Enum):
    LEVEL_0 = 0  # Data Point
    LEVEL_1 = 1  # Variable / Vector
    LEVEL_2 = 2  # Table / Matrix
    LEVEL_3 = 3  # Multi-level Table / Knowledge Graph
    LEVEL_4 = 4  # Unlinkable Tables / Raw Data

class Dataset(ABC):
    """Abstract base class for all datasets."""
    
    @property
    @abstractmethod
    def complexity_level(self) -> ComplexityLevel:
        pass
    
    @abstractmethod
    def get_data(self) -> Any:
        pass
    
    def __repr__(self):
        return f"<{self.__class__.__name__} (Level {self.complexity_level.value})>"

class Level4Dataset(Dataset):
    """
    Level 4: Unlinkable Tables / Raw Data.
    Represents a collection of raw data sources that are not yet unified.
    """
    def __init__(self, data_sources: Dict[str, Any]):
        self._data = data_sources

    @property
    def complexity_level(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_4

    def get_data(self) -> Dict[str, Any]:
        return self._data

class Level3Dataset(Dataset):
    """
    Level 3: Multi-level Table / Knowledge Graph.
    Represents data with complex relationships, modeled as a graph or related tables.
    """
    def __init__(self, data: Union[nx.Graph, pd.DataFrame]):
        self._data = data

    @property
    def complexity_level(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_3

    def get_data(self) -> Union[nx.Graph, pd.DataFrame]:
        return self._data

class Level2Dataset(Dataset):
    """
    Level 2: Single Table.
    Represents a flat, tabular dataset (e.g., a pandas DataFrame).
    """
    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self._df = df
        self.name = name

    @property
    def complexity_level(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_2

    def get_data(self) -> pd.DataFrame:
        return self._df

class Level1Dataset(Dataset):
    """
    Level 1: Variable / Vector.
    Represents a single series of values (e.g., a pandas Series or list).
    """
    def __init__(self, series: pd.Series, name: str = "variable"):
        self._series = series
        self.name = name

    @property
    def complexity_level(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_1

    def get_data(self) -> pd.Series:
        return self._series

class Level0Dataset(Dataset):
    """
    Level 0: Data Point.
    Represents a single scalar value.

    Extended to support ascent by storing parent reference.
    """
    def __init__(
        self,
        value: Any,
        description: str = "value",
        parent_data: Optional[pd.Series] = None,
        aggregation_method: Optional[str] = None
    ):
        self._value = value
        self.description = description
        self._parent_data = parent_data
        self._aggregation_method = aggregation_method

    @property
    def complexity_level(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_0

    def get_data(self) -> Any:
        return self._value

    @property
    def has_parent(self) -> bool:
        """Check if parent data is available for ascent."""
        return self._parent_data is not None

    def get_parent_data(self) -> Optional[pd.Series]:
        """Get the parent L1 data that was aggregated to produce this L0."""
        return self._parent_data

    @property
    def aggregation_method(self) -> Optional[str]:
        """Get the aggregation method used to produce this L0."""
        return self._aggregation_method
