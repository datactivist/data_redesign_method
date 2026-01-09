"""
Navigation History Module

Phase 1.2 - Code Simplification (011-code-simplification)
Extracted from navigation.py

Spec Traceability:
------------------
- 002-ascent-functionality: Linear navigation history tracking

Contains:
- NavigationStep: Single step in navigation history
- NavigationHistory: Append-only log of steps
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from intuitiveness.complexity import ComplexityLevel


@dataclass
class NavigationStep:
    """
    A single step in the navigation history.

    Attributes:
        level: The complexity level at this step
        node_id: Identifier of the node/data at this position
        action: The action taken to reach this step (descend, ascend, entry, exit, resume)
        timestamp: When this step occurred
    """
    level: ComplexityLevel
    node_id: str
    action: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "level_name": self.level.name,
            "node_id": self.node_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat()
        }


class NavigationHistory:
    """
    Append-only log of navigation steps.

    Tracks the complete path a user has taken through the data hierarchy.
    """

    def __init__(self):
        self._steps: List[NavigationStep] = []

    def append(self, step: NavigationStep) -> None:
        """Add a step to the history."""
        self._steps.append(step)

    def get_path(self) -> List[NavigationStep]:
        """Get the full navigation path."""
        return self._steps.copy()

    def get_path_dicts(self) -> List[Dict[str, Any]]:
        """Get the navigation path as a list of dictionaries."""
        return [step.to_dict() for step in self._steps]

    @property
    def length(self) -> int:
        """Number of steps taken."""
        return len(self._steps)

    @property
    def current_step(self) -> Optional[NavigationStep]:
        """Get the most recent step."""
        return self._steps[-1] if self._steps else None

    def __len__(self) -> int:
        return len(self._steps)
