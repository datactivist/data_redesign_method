"""
Navigation State and Action Enums

Phase 1.2 - Code Simplification (011-code-simplification)
Extracted from navigation.py

Spec Traceability:
------------------
- 002-ascent-functionality: Navigation state management
"""

from enum import Enum


class NavigationState(Enum):
    """
    State of a navigation session.

    ENTRY: Just started, at L4 (entry point)
    EXPLORING: Actively navigating (L1-L3)
    EXITED: Session ended, can be resumed
    """
    ENTRY = "entry"
    EXPLORING = "exploring"
    EXITED = "exited"


class NavigationAction(Enum):
    """Types of navigation actions."""
    ENTRY = "entry"
    DESCEND = "descend"
    ASCEND = "ascend"
    RESTORE = "restore"
    EXIT = "exit"
