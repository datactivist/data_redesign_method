# Persistence API Contract

**Feature**: 005-session-persistence
**Date**: 2025-12-04

## Module: `intuitiveness.persistence`

### SessionStore

Main class for managing session persistence.

```python
class SessionStore:
    """
    Manages session state persistence to browser localStorage.

    Usage:
        store = SessionStore()

        # Save current session
        store.save()

        # Load session on app start
        if store.has_saved_session():
            store.load()

        # Clear session
        store.clear()
    """

    def __init__(self, storage_key: str = "data_redesign_session"):
        """
        Initialize the session store.

        Args:
            storage_key: localStorage key name
        """
        pass

    def save(self) -> SaveResult:
        """
        Save current st.session_state to localStorage.

        Returns:
            SaveResult with success status and any warnings

        Raises:
            StorageQuotaExceeded: If data exceeds localStorage limit
        """
        pass

    def load(self) -> LoadResult:
        """
        Load session from localStorage into st.session_state.

        Returns:
            LoadResult with success status and loaded data summary

        Raises:
            SessionCorrupted: If stored data is invalid
            VersionMismatch: If schema version is incompatible
        """
        pass

    def has_saved_session(self) -> bool:
        """
        Check if a saved session exists in localStorage.

        Returns:
            True if valid session data exists
        """
        pass

    def clear(self) -> None:
        """
        Remove all session data from localStorage and reset session_state.
        """
        pass

    def get_session_info(self) -> Optional[SessionInfo]:
        """
        Get metadata about saved session without loading it.

        Returns:
            SessionInfo with timestamp, step, file count, or None if no session
        """
        pass
```

### Data Classes

```python
@dataclass
class SaveResult:
    """Result of a save operation."""
    success: bool
    bytes_saved: int
    warnings: List[str]  # e.g., "Large file skipped: data.csv"

@dataclass
class LoadResult:
    """Result of a load operation."""
    success: bool
    wizard_step: int
    file_count: int
    dataset_levels: List[str]  # e.g., ["l4", "l3"]
    warnings: List[str]

@dataclass
class SessionInfo:
    """Metadata about a saved session."""
    timestamp: datetime
    wizard_step: int
    file_count: int
    total_size_bytes: int
    version: str
```

### Exceptions

```python
class PersistenceError(Exception):
    """Base class for persistence errors."""
    pass

class StorageQuotaExceeded(PersistenceError):
    """Raised when localStorage quota is exceeded."""
    def __init__(self, required_bytes: int, available_bytes: int):
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes

class SessionCorrupted(PersistenceError):
    """Raised when stored session data is invalid."""
    def __init__(self, reason: str):
        self.reason = reason

class VersionMismatch(PersistenceError):
    """Raised when session schema version is incompatible."""
    def __init__(self, stored_version: str, current_version: str):
        self.stored_version = stored_version
        self.current_version = current_version
```

---

## Module: `intuitiveness.persistence.serializers`

### DataFrame Serializer

```python
def serialize_dataframe(df: pd.DataFrame) -> str:
    """
    Serialize a pandas DataFrame to compressed base64 string.

    Args:
        df: DataFrame to serialize

    Returns:
        Base64-encoded compressed JSON string
    """
    pass

def deserialize_dataframe(data: str) -> pd.DataFrame:
    """
    Deserialize a compressed base64 string to DataFrame.

    Args:
        data: Base64-encoded compressed JSON string

    Returns:
        Reconstructed DataFrame

    Raises:
        ValueError: If data is invalid or corrupted
    """
    pass
```

### Graph Serializer

```python
def serialize_graph(G: nx.Graph) -> str:
    """
    Serialize a networkx Graph to compressed base64 string.

    Args:
        G: Graph to serialize

    Returns:
        Base64-encoded compressed JSON string
    """
    pass

def deserialize_graph(data: str) -> nx.Graph:
    """
    Deserialize a compressed base64 string to Graph.

    Args:
        data: Base64-encoded compressed JSON string

    Returns:
        Reconstructed Graph

    Raises:
        ValueError: If data is invalid or corrupted
    """
    pass
```

---

## Module: `intuitiveness.ui.recovery_banner`

### Recovery Banner Component

```python
def render_recovery_banner(session_info: SessionInfo) -> RecoveryAction:
    """
    Render a session recovery banner at the top of the app.

    Shows:
    - "Welcome back! Your previous session was saved [time ago]"
    - "You were at Step X with Y files"
    - [Continue] [Start Fresh] buttons

    Args:
        session_info: Metadata about saved session

    Returns:
        RecoveryAction.CONTINUE or RecoveryAction.START_FRESH
    """
    pass

class RecoveryAction(Enum):
    CONTINUE = "continue"
    START_FRESH = "start_fresh"
    DISMISS = "dismiss"
```

---

## Integration Points

### App Initialization

```python
# In streamlit_app.py main()

from intuitiveness.persistence import SessionStore
from intuitiveness.ui.recovery_banner import render_recovery_banner

def main():
    store = SessionStore()

    # Check for saved session on first load
    if 'session_loaded' not in st.session_state:
        st.session_state.session_loaded = True

        if store.has_saved_session():
            info = store.get_session_info()
            action = render_recovery_banner(info)

            if action == RecoveryAction.CONTINUE:
                result = store.load()
                if not result.success:
                    st.warning(f"Could not restore session: {result.warnings}")
            elif action == RecoveryAction.START_FRESH:
                store.clear()

    # Auto-save on state changes
    if should_auto_save():
        store.save()
```

### Auto-Save Triggers

Auto-save should occur after:
- File upload completed
- Wizard step navigation
- Form submission (checkbox, text input changes)
- Connection definition changes
- Semantic matching completion

Use debouncing to avoid excessive saves (e.g., save at most once per 2 seconds).
