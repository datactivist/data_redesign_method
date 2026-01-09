# Contract: Registry Dependency Injection

## Purpose

Define the interface contract for registries that support dependency injection.

## Applies To

- `intuitiveness/ascent/enrichment.py` - EnrichmentRegistry
- `intuitiveness/ascent/dimensions.py` - DimensionRegistry

## Interface Contract

### Required Class Methods

```python
@classmethod
def get_instance(cls) -> 'Registry':
    """
    Get singleton instance with defaults registered.

    Behavior:
    - First call creates instance and registers defaults
    - Subsequent calls return same instance
    - Thread-safe not guaranteed (Streamlit is single-threaded)

    Returns:
        Registry: The singleton instance
    """

@classmethod
def create_fresh(cls, with_defaults: bool = True) -> 'Registry':
    """
    Create isolated instance (not singleton).

    Use cases:
    - Unit tests requiring isolation
    - Per-session registries
    - Custom configurations

    Args:
        with_defaults: If True, register built-in defaults

    Returns:
        Registry: New isolated instance
    """

@classmethod
def reset_instance(cls) -> None:
    """
    Reset singleton state for test cleanup.

    Behavior:
    - Sets _instance to None
    - Sets _defaults_registered to False
    - Next get_instance() creates fresh singleton
    """
```

### Required Instance Methods

```python
def register(self, item: Any, is_default: bool = False) -> None:
    """
    Register an item in the registry.

    Args:
        item: The item to register
        is_default: Whether this is a built-in default

    Raises:
        ValueError: If item with same name already registered
    """

def get(self, name: str) -> Any:
    """
    Retrieve item by name.

    Args:
        name: Unique identifier

    Returns:
        The registered item

    Raises:
        KeyError: If name not found
    """

def list_all(self) -> List[Any]:
    """
    List all registered items.

    Returns:
        List of all items (may be empty)
    """
```

### Required Class Attributes

```python
_instance: Optional['Registry'] = None  # Singleton instance
_defaults_registered: bool = False       # Track if defaults loaded
```

## Invariants

1. **Lazy Initialization**: Defaults are not registered at module import time
2. **Singleton Isolation**: `create_fresh()` returns instance independent of singleton
3. **Idempotent Reset**: Multiple `reset_instance()` calls are safe
4. **Registration Uniqueness**: Same name cannot be registered twice

## Test Verification

```python
def test_create_fresh_isolation():
    """Fresh instances don't share state with singleton."""
    singleton = Registry.get_instance()
    fresh = Registry.create_fresh()

    # Modify fresh
    fresh.register(custom_item)

    # Singleton unchanged
    assert custom_item not in singleton.list_all()

def test_reset_clears_singleton():
    """Reset allows new singleton creation."""
    instance1 = Registry.get_instance()
    Registry.reset_instance()
    instance2 = Registry.get_instance()

    assert instance1 is not instance2
```

## Migration Guide

### From Singleton Pattern

Before (singleton only):
```python
registry = EnrichmentRegistry.get_instance()
```

After (with DI support):
```python
# Production (backward compatible)
registry = EnrichmentRegistry.get_instance()

# Testing (isolated)
registry = EnrichmentRegistry.create_fresh(with_defaults=True)

# Dependency injection
def my_function(registry: EnrichmentRegistry = None):
    registry = registry or EnrichmentRegistry.get_instance()
```
