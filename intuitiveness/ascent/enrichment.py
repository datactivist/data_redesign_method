"""
Enrichment functions for ascending through abstraction levels.

EnrichmentFunction: Callable that transforms data from a lower level to a higher level.
EnrichmentRegistry: Manages available enrichment functions and provides defaults.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from ..complexity import ComplexityLevel


@dataclass
class EnrichmentFunction:
    """
    A callable that takes data from a lower level and produces enriched data
    for a higher level.

    Attributes:
        name: Unique identifier for this function
        description: User-facing description
        source_level: The level this function operates on (L0, L1, or L2)
        target_level: The level this function produces (must be source + 1)
        func: The enrichment callable
        requires_context: Whether parent data is needed for enrichment
    """
    name: str
    description: str
    source_level: ComplexityLevel
    target_level: ComplexityLevel
    func: Callable[[Any], Any]
    requires_context: bool = False

    def __post_init__(self):
        """Validate that target is exactly source + 1."""
        if self.target_level.value != self.source_level.value + 1:
            raise ValueError(
                f"Target level must be source + 1. "
                f"Got source={self.source_level.name}, target={self.target_level.name}"
            )
        if self.target_level == ComplexityLevel.LEVEL_4:
            raise ValueError("L4 is entry-only, cannot ascend to L4")

    def __call__(self, data: Any, context: Optional[Any] = None) -> Any:
        """Execute the enrichment function."""
        if self.requires_context and context is None:
            raise ValueError(f"{self.name} requires context data")
        if self.requires_context:
            return self.func(data, context)
        return self.func(data)


class EnrichmentRegistry:
    """
    Registry of available enrichment functions.
    Provides defaults and allows custom registration.
    """

    _instance: Optional['EnrichmentRegistry'] = None

    def __init__(self):
        self._functions: Dict[str, EnrichmentFunction] = {}
        self._defaults: Dict[tuple, List[str]] = {}  # (source, target) -> list of default names

    @classmethod
    def get_instance(cls) -> 'EnrichmentRegistry':
        """Get the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def register(self, func: EnrichmentFunction, is_default: bool = False) -> None:
        """
        Register an enrichment function.

        Args:
            func: The EnrichmentFunction to register
            is_default: Whether this is a default function for its transition

        Raises:
            ValueError: If name already registered
        """
        if func.name in self._functions:
            raise ValueError(f"Function '{func.name}' already registered")

        self._functions[func.name] = func

        if is_default:
            key = (func.source_level, func.target_level)
            if key not in self._defaults:
                self._defaults[key] = []
            self._defaults[key].append(func.name)

    def get(self, name: str) -> EnrichmentFunction:
        """
        Retrieve enrichment function by name.

        Raises:
            KeyError: If name not found
        """
        if name not in self._functions:
            raise KeyError(f"No enrichment function named '{name}'")
        return self._functions[name]

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
        return [
            func for func in self._functions.values()
            if func.source_level == source and func.target_level == target
        ]

    def get_defaults(
        self,
        source: ComplexityLevel,
        target: ComplexityLevel
    ) -> List[EnrichmentFunction]:
        """
        Get default (built-in) functions for a transition.

        Returns:
            List of default EnrichmentFunction objects
        """
        key = (source, target)
        if key not in self._defaults:
            return []
        return [self._functions[name] for name in self._defaults[key]]

    def list_all(self) -> List[EnrichmentFunction]:
        """List all registered functions."""
        return list(self._functions.values())


# =============================================================================
# Default Enrichment Functions for L0 → L1
# =============================================================================

def _source_expansion(data: Any, context: pd.Series) -> pd.Series:
    """
    Re-expand to original vector from parent data.

    Args:
        data: The L0 scalar value (used for validation)
        context: The parent L1 series that was aggregated

    Returns:
        The original series
    """
    return context


def _naming_signatures(data: Any, context: pd.Series) -> pd.Series:
    """
    Extract naming features from each item in the parent series.

    For each item, creates a tuple of:
    - first_word: First word of the name
    - word_count: Total number of words
    - has_underscore: Whether name contains underscore
    - char_count: Total character count

    Args:
        data: The L0 scalar value
        context: The parent L1 series

    Returns:
        Series of naming signature tuples
    """
    def extract_signature(name):
        if not isinstance(name, str):
            name = str(name)
        words = name.replace('_', ' ').split()
        return {
            'original': name,
            'first_word': words[0] if words else '',
            'word_count': len(words),
            'has_underscore': '_' in name,
            'char_count': len(name)
        }

    return context.apply(extract_signature)


# =============================================================================
# Default Enrichment Functions for L1 → L2 (will be registered from dimensions.py)
# =============================================================================

def _vector_to_dataframe(series: pd.Series) -> pd.DataFrame:
    """
    Convert a vector to a single-column DataFrame.

    Args:
        series: The L1 vector

    Returns:
        DataFrame with the series as a column
    """
    return pd.DataFrame({'value': series})


# =============================================================================
# Register default functions
# =============================================================================

def _register_defaults():
    """Register all default enrichment functions."""
    registry = EnrichmentRegistry.get_instance()

    # L0 → L1 defaults
    try:
        registry.register(
            EnrichmentFunction(
                name='source_expansion',
                description='Re-expand to the original vector that was aggregated',
                source_level=ComplexityLevel.LEVEL_0,
                target_level=ComplexityLevel.LEVEL_1,
                func=_source_expansion,
                requires_context=True
            ),
            is_default=True
        )
    except ValueError:
        pass  # Already registered

    try:
        registry.register(
            EnrichmentFunction(
                name='naming_signatures',
                description='Extract naming features (first word, word count, patterns) from each item',
                source_level=ComplexityLevel.LEVEL_0,
                target_level=ComplexityLevel.LEVEL_1,
                func=_naming_signatures,
                requires_context=True
            ),
            is_default=True
        )
    except ValueError:
        pass  # Already registered


# Auto-register defaults on module import
_register_defaults()
