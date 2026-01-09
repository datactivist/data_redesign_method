"""
Quality Data Platform - Catalog Data Models

Data models for the dataset catalog including Dataset, DatasetSummary,
and DatasetDetail entities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4


@dataclass
class Dataset:
    """
    Core entity representing a tabular data file in the catalog.

    Attributes:
        id: Unique identifier.
        name: Human-readable dataset name.
        description: Dataset purpose and contents.
        domain_tags: Domain categories (e.g., "healthcare", "finance").
        file_path: Path to CSV file.
        row_count: Number of rows in dataset.
        feature_count: Number of columns (excluding target).
        target_column: Designated target for supervised tasks.
        usability_score: Overall ML-readiness score (0-100).
        latest_report_id: Most recent quality assessment.
        created_at: When dataset was added.
        updated_at: Last modification time.
    """

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    domain_tags: List[str] = field(default_factory=list)
    file_path: str = ""
    row_count: int = 0
    feature_count: int = 0
    target_column: Optional[str] = None
    usability_score: Optional[float] = None
    latest_report_id: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "domain_tags": self.domain_tags,
            "file_path": self.file_path,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "target_column": self.target_column,
            "usability_score": self.usability_score,
            "latest_report_id": str(self.latest_report_id) if self.latest_report_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid4()),
            name=data.get("name", ""),
            description=data.get("description", ""),
            domain_tags=data.get("domain_tags", []),
            file_path=data.get("file_path", ""),
            row_count=data.get("row_count", 0),
            feature_count=data.get("feature_count", 0),
            target_column=data.get("target_column"),
            usability_score=data.get("usability_score"),
            latest_report_id=UUID(data["latest_report_id"])
            if data.get("latest_report_id")
            else None,
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", datetime.now()),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else data.get("updated_at", datetime.now()),
        )


@dataclass
class DatasetSummary:
    """
    Lightweight dataset summary for catalog listings.

    Attributes:
        id: Unique identifier.
        name: Human-readable dataset name.
        domain_tags: Domain categories.
        row_count: Number of rows.
        feature_count: Number of features.
        usability_score: Overall ML-readiness score.
    """

    id: UUID
    name: str
    domain_tags: List[str] = field(default_factory=list)
    row_count: int = 0
    feature_count: int = 0
    usability_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "domain_tags": self.domain_tags,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "usability_score": self.usability_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSummary":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id"),
            name=data.get("name", ""),
            domain_tags=data.get("domain_tags", []),
            row_count=data.get("row_count", 0),
            feature_count=data.get("feature_count", 0),
            usability_score=data.get("usability_score"),
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "DatasetSummary":
        """Create summary from full dataset."""
        return cls(
            id=dataset.id,
            name=dataset.name,
            domain_tags=dataset.domain_tags,
            row_count=dataset.row_count,
            feature_count=dataset.feature_count,
            usability_score=dataset.usability_score,
        )


@dataclass
class DatasetDetail:
    """
    Full dataset details including latest quality report.

    Inherits all Dataset fields plus the latest quality report.
    """

    dataset: Dataset
    latest_report: Optional[Dict[str, Any]] = None  # QualityReport.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = self.dataset.to_dict()
        result["latest_report"] = self.latest_report
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetDetail":
        """Create from dictionary."""
        latest_report = data.pop("latest_report", None)
        dataset = Dataset.from_dict(data)
        return cls(dataset=dataset, latest_report=latest_report)
