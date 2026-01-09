"""
Navigation Exceptions

Phase 1.2 - Code Simplification (011-code-simplification)
Extracted from navigation.py

Spec Traceability:
------------------
- 002-ascent-functionality: Navigation error handling
"""


class NavigationError(Exception):
    """Raised when a navigation action is invalid."""
    pass


class SessionNotFoundError(NavigationError):
    """Raised when trying to resume a non-existent session."""
    pass
