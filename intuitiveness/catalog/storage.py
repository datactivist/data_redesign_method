"""
Quality Data Platform - Catalog Storage

JSON-based catalog persistence for dataset metadata and quality reports.
Follows the same pattern as existing session persistence in intuitiveness.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime

from intuitiveness.catalog.models import Dataset, DatasetSummary, DatasetDetail

logger = logging.getLogger(__name__)

# Default catalog directory
DEFAULT_CATALOG_DIR = Path.home() / ".intuitiveness" / "catalog"


class CatalogStorage:
    """
    JSON-based storage for the dataset catalog.

    Structure:
        catalog/
        ├── catalog.json           # Dataset index
        ├── datasets/
        │   ├── {dataset_id}/
        │   │   ├── metadata.json  # Dataset entity
        │   │   └── reports/
        │   │       └── {report_id}.json
        │   └── ...
        └── synthetic/
            └── {synthetic_id}.json
    """

    def __init__(self, catalog_dir: Optional[Path] = None):
        """
        Initialize catalog storage.

        Args:
            catalog_dir: Directory for catalog files. Defaults to ~/.intuitiveness/catalog
        """
        self.catalog_dir = Path(catalog_dir) if catalog_dir else DEFAULT_CATALOG_DIR
        self.catalog_file = self.catalog_dir / "catalog.json"
        self.datasets_dir = self.catalog_dir / "datasets"
        self.synthetic_dir = self.catalog_dir / "synthetic"

        self._ensure_directories()
        self._catalog_cache: Dict[str, Dataset] = {}
        self._load_catalog()

    def _ensure_directories(self) -> None:
        """Create catalog directories if they don't exist."""
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)
        self.synthetic_dir.mkdir(exist_ok=True)

    def _load_catalog(self) -> None:
        """Load catalog index from disk."""
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, "r") as f:
                    data = json.load(f)
                    for ds_data in data.get("datasets", []):
                        dataset = Dataset.from_dict(ds_data)
                        self._catalog_cache[str(dataset.id)] = dataset
                logger.info(f"Loaded {len(self._catalog_cache)} datasets from catalog")
            except Exception as e:
                logger.error(f"Failed to load catalog: {e}")
                self._catalog_cache = {}
        else:
            self._catalog_cache = {}

    def _save_catalog(self) -> None:
        """Save catalog index to disk."""
        data = {
            "catalog_version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "datasets": [ds.to_dict() for ds in self._catalog_cache.values()],
        }
        with open(self.catalog_file, "w") as f:
            json.dump(data, f, indent=2)

    def _get_dataset_dir(self, dataset_id: UUID) -> Path:
        """Get directory for a specific dataset."""
        return self.datasets_dir / str(dataset_id)

    def _get_reports_dir(self, dataset_id: UUID) -> Path:
        """Get reports directory for a dataset."""
        return self._get_dataset_dir(dataset_id) / "reports"

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def add_dataset(
        self,
        name: str,
        file_path: str,
        description: str = "",
        domain_tags: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        row_count: int = 0,
        feature_count: int = 0,
    ) -> Dataset:
        """
        Add a new dataset to the catalog.

        Args:
            name: Human-readable dataset name.
            file_path: Path to the CSV file.
            description: Dataset description.
            domain_tags: Domain categories.
            target_column: Target column for supervised tasks.
            row_count: Number of rows.
            feature_count: Number of features.

        Returns:
            Created Dataset instance.
        """
        dataset = Dataset(
            name=name,
            file_path=file_path,
            description=description,
            domain_tags=domain_tags or [],
            target_column=target_column,
            row_count=row_count,
            feature_count=feature_count,
        )

        # Create dataset directory
        dataset_dir = self._get_dataset_dir(dataset.id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self._get_reports_dir(dataset.id).mkdir(exist_ok=True)

        # Save metadata
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2)

        # Update catalog cache and save
        self._catalog_cache[str(dataset.id)] = dataset
        self._save_catalog()

        logger.info(f"Added dataset: {name} (ID: {dataset.id})")
        return dataset

    def get_dataset(self, dataset_id: UUID) -> Optional[Dataset]:
        """
        Get a dataset by ID.

        Args:
            dataset_id: Dataset UUID.

        Returns:
            Dataset if found, None otherwise.
        """
        return self._catalog_cache.get(str(dataset_id))

    def update_dataset(self, dataset: Dataset) -> Dataset:
        """
        Update an existing dataset.

        Args:
            dataset: Dataset with updated fields.

        Returns:
            Updated Dataset instance.
        """
        dataset.updated_at = datetime.now()

        # Update metadata file
        dataset_dir = self._get_dataset_dir(dataset.id)
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2)

        # Update cache and save catalog
        self._catalog_cache[str(dataset.id)] = dataset
        self._save_catalog()

        logger.info(f"Updated dataset: {dataset.name} (ID: {dataset.id})")
        return dataset

    def delete_dataset(self, dataset_id: UUID) -> bool:
        """
        Remove a dataset from the catalog.

        Args:
            dataset_id: Dataset UUID.

        Returns:
            True if deleted, False if not found.
        """
        if str(dataset_id) not in self._catalog_cache:
            return False

        # Remove from cache
        del self._catalog_cache[str(dataset_id)]

        # Remove directory
        import shutil
        dataset_dir = self._get_dataset_dir(dataset_id)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        # Save catalog
        self._save_catalog()

        logger.info(f"Deleted dataset: {dataset_id}")
        return True

    # =========================================================================
    # Report Storage
    # =========================================================================

    def save_report(self, dataset_id: UUID, report: Dict[str, Any]) -> str:
        """
        Save a quality report for a dataset.

        Args:
            dataset_id: Dataset UUID.
            report: Quality report dictionary.

        Returns:
            Report ID.
        """
        report_id = report.get("id", str(UUID()))
        reports_dir = self._get_reports_dir(dataset_id)
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_file = reports_dir / f"{report_id}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Update dataset with latest report
        dataset = self.get_dataset(dataset_id)
        if dataset:
            dataset.latest_report_id = UUID(report_id)
            dataset.usability_score = report.get("usability_score")
            self.update_dataset(dataset)

        logger.info(f"Saved report {report_id} for dataset {dataset_id}")
        return report_id

    def get_report(self, dataset_id: UUID, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a quality report by ID.

        Args:
            dataset_id: Dataset UUID.
            report_id: Report ID.

        Returns:
            Report dictionary if found, None otherwise.
        """
        report_file = self._get_reports_dir(dataset_id) / f"{report_id}.json"
        if report_file.exists():
            with open(report_file, "r") as f:
                return json.load(f)
        return None

    def get_latest_report(self, dataset_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get the most recent quality report for a dataset.

        Args:
            dataset_id: Dataset UUID.

        Returns:
            Latest report dictionary if exists, None otherwise.
        """
        dataset = self.get_dataset(dataset_id)
        if dataset and dataset.latest_report_id:
            return self.get_report(dataset_id, str(dataset.latest_report_id))
        return None

    # =========================================================================
    # Listing and Search
    # =========================================================================

    def list_datasets(self) -> List[Dataset]:
        """Get all datasets in the catalog."""
        return list(self._catalog_cache.values())

    def list_summaries(self) -> List[DatasetSummary]:
        """Get lightweight summaries of all datasets."""
        return [DatasetSummary.from_dataset(ds) for ds in self._catalog_cache.values()]

    def get_dataset_detail(self, dataset_id: UUID) -> Optional[DatasetDetail]:
        """
        Get full dataset details including latest report.

        Args:
            dataset_id: Dataset UUID.

        Returns:
            DatasetDetail if found, None otherwise.
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None

        latest_report = self.get_latest_report(dataset_id)
        return DatasetDetail(dataset=dataset, latest_report=latest_report)

    def count(self) -> int:
        """Get the number of datasets in the catalog."""
        return len(self._catalog_cache)


# Global storage instance
_storage: Optional[CatalogStorage] = None


def get_storage(catalog_dir: Optional[Path] = None) -> CatalogStorage:
    """
    Get or create the global catalog storage instance.

    Args:
        catalog_dir: Optional custom catalog directory.

    Returns:
        CatalogStorage instance.
    """
    global _storage
    if _storage is None or catalog_dir is not None:
        _storage = CatalogStorage(catalog_dir)
    return _storage
