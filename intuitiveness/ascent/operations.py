"""
Ascent operation tracking for the Data Redesign Method.

AscentOperation: Records an ascent action for history and traceability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional
import hashlib
import uuid

from ..complexity import ComplexityLevel


@dataclass
class AscentOperation:
    """
    Records an ascent action including source level, target level,
    enrichment function used, and resulting data structure.

    Attributes:
        id: UUID for this operation
        source_level: Where we started (L0, L1, or L2)
        target_level: Where we ended (L1, L2, or L3)
        enrichment_function: Name of EnrichmentFunction used
        dimensions_added: Names of dimensions added (L1→L2, L2→L3)
        timestamp: When operation occurred
        source_data_hash: Hash of input data for integrity
        result_data_hash: Hash of output data
        row_count_before: Items before ascent
        row_count_after: Items after ascent (should match)
    """
    source_level: ComplexityLevel
    target_level: ComplexityLevel
    enrichment_function: str
    dimensions_added: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    source_data_hash: str = ""
    result_data_hash: str = ""
    row_count_before: int = 0
    row_count_after: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Validate the operation."""
        if self.target_level.value != self.source_level.value + 1:
            raise ValueError(
                f"Ascent must be to adjacent level. "
                f"Got source={self.source_level.name}, target={self.target_level.name}"
            )
        if self.target_level == ComplexityLevel.LEVEL_4:
            raise ValueError("L4 is entry-only, cannot ascend to L4")

    def validate_integrity(self) -> bool:
        """Check that row counts match (ascent should not add/remove items)."""
        return self.row_count_before == self.row_count_after

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'source_level': self.source_level.value,
            'source_level_name': self.source_level.name,
            'target_level': self.target_level.value,
            'target_level_name': self.target_level.name,
            'enrichment_function': self.enrichment_function,
            'dimensions_added': self.dimensions_added,
            'timestamp': self.timestamp.isoformat(),
            'source_data_hash': self.source_data_hash,
            'result_data_hash': self.result_data_hash,
            'row_count_before': self.row_count_before,
            'row_count_after': self.row_count_after,
            'integrity_valid': self.validate_integrity()
        }

    @staticmethod
    def compute_hash(data: Any) -> str:
        """Compute a hash for data integrity verification."""
        try:
            # Convert to string representation for hashing
            data_str = str(data)
            return hashlib.md5(data_str.encode()).hexdigest()[:16]
        except Exception:
            return "hash_error"

    @classmethod
    def create(
        cls,
        source_level: ComplexityLevel,
        target_level: ComplexityLevel,
        enrichment_function: str,
        source_data: Any,
        result_data: Any,
        dimensions_added: Optional[List[str]] = None
    ) -> 'AscentOperation':
        """
        Factory method to create an AscentOperation with computed hashes.

        Args:
            source_level: Starting level
            target_level: Ending level
            enrichment_function: Name of function used
            source_data: Input data
            result_data: Output data
            dimensions_added: List of dimension names added

        Returns:
            AscentOperation with all fields populated
        """
        import pandas as pd

        # Compute row counts
        if hasattr(source_data, '__len__'):
            row_before = len(source_data)
        elif isinstance(source_data, (int, float)):
            row_before = 1  # Scalar
        else:
            row_before = 1

        if isinstance(result_data, pd.DataFrame):
            row_after = len(result_data)
        elif isinstance(result_data, pd.Series):
            row_after = len(result_data)
        elif hasattr(result_data, '__len__'):
            row_after = len(result_data)
        else:
            row_after = 1

        return cls(
            source_level=source_level,
            target_level=target_level,
            enrichment_function=enrichment_function,
            dimensions_added=dimensions_added or [],
            source_data_hash=cls.compute_hash(source_data),
            result_data_hash=cls.compute_hash(result_data),
            row_count_before=row_before,
            row_count_after=row_after
        )
