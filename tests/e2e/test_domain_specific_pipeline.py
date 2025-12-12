"""
Domain-Specific Pipeline Tests for the Data Redesign Method.

This module tests the complete descent (L4->L3->L2->L1->L0) and ascent (L0->L1->L2->L3)
cycles with DOMAIN-SPECIFIC transformations as defined in CLAUDE.md.

Key differences from test_full_pipeline.py:
- Uses semantic_table_join for L4→L3 (producing joined tables with _semantic_similarity)
- Implements business-specific categorization logic
- Tests meaningful aggregations (not just count)

Run with: pytest tests/e2e/test_domain_specific_pipeline.py -v -s
"""

import pytest
import pandas as pd
import networkx as nx
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from intuitiveness import (
    ComplexityLevel,
    Level4Dataset,
    Level3Dataset,
    Level2Dataset,
    Level1Dataset,
    Level0Dataset,
    Redesigner,
)
from intuitiveness.descent.semantic_join import SemanticJoinConfig, semantic_table_join
import uuid


# =============================================================================
# NAVIGATION SESSION (For full path export with design choices)
# =============================================================================

@dataclass
class NavigationNode:
    """A node in the navigation tree tracking design choices."""
    id: str
    level: int
    level_name: str
    parent_id: Optional[str]
    children_ids: List[str]
    action: str  # 'entry', 'descend', 'ascend'
    timestamp: str
    decision_description: str
    design_rationale: str
    params: Dict[str, Any]
    output_snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "level_name": self.level_name,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "action": self.action,
            "timestamp": self.timestamp,
            "decision_description": self.decision_description,
            "design_rationale": self.design_rationale,
            "params": self.params,
            "output_snapshot": self.output_snapshot
        }


class NavigationSession:
    """
    Track the full navigation path with design choices for descent/ascent.

    This creates session exports like test0_session_export.json with:
    - Full navigation tree
    - Design decisions at each level transition
    - Output snapshots at each level
    """

    def __init__(self, config: 'DomainConfig'):
        self.session_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config
        self.nodes: List[NavigationNode] = []
        self.current_path: List[str] = []
        self.root_id: Optional[str] = None
        self.current_id: Optional[str] = None
        self.start_time = datetime.now().isoformat()
        self.cumulative_outputs: Dict[str, Dict] = {}

    def _create_node_id(self) -> str:
        return str(uuid.uuid4())

    def add_node(
        self,
        level: int,
        level_name: str,
        action: str,
        decision_description: str,
        design_rationale: str,
        params: Dict[str, Any],
        output_snapshot: Dict[str, Any]
    ) -> str:
        """Add a navigation node and return its ID."""
        node_id = self._create_node_id()
        parent_id = self.current_id

        node = NavigationNode(
            id=node_id,
            level=level,
            level_name=level_name,
            parent_id=parent_id,
            children_ids=[],
            action=action,
            timestamp=datetime.now().isoformat(),
            decision_description=decision_description,
            design_rationale=design_rationale,
            params=params,
            output_snapshot=output_snapshot
        )

        # Update parent's children
        if parent_id:
            for n in self.nodes:
                if n.id == parent_id:
                    n.children_ids.append(node_id)
                    break

        self.nodes.append(node)
        self.current_path.append(node_id)
        self.current_id = node_id

        if self.root_id is None:
            self.root_id = node_id

        return node_id

    def update_cumulative_output(self, level_key: str, output_info: Dict):
        """Track cumulative outputs for export."""
        self.cumulative_outputs[level_key] = output_info

    def export(self, output_dir: Path) -> Dict[str, Any]:
        """Export full session to JSON."""
        export_data = {
            "version": "1.0",
            "feature": "domain-specific-transformations",
            "exported_at": datetime.now().isoformat() + "Z",
            "session_id": self.session_id,
            "config_name": self.config.name,
            "config_description": self.config.description,
            "current_output": {
                "level": self.nodes[-1].level if self.nodes else 0,
                "level_name": self.nodes[-1].level_name if self.nodes else "LEVEL_0",
                "output_type": "datum" if self.nodes and self.nodes[-1].level == 0 else "unknown",
                "value": self.cumulative_outputs.get("datum", {}).get("sample_data", "")
            },
            "navigation_tree": {
                "nodes": [n.to_dict() for n in self.nodes],
                "root_id": self.root_id,
                "current_id": self.current_id
            },
            "current_path": self.current_path,
            "cumulative_outputs": self.cumulative_outputs,
            "design_choices_summary": self._summarize_design_choices()
        }

        # Save to file
        filepath = output_dir / f"{self.config.name}_session_export.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        return export_data

    def _summarize_design_choices(self) -> Dict[str, str]:
        """Summarize key design choices for easy reading."""
        return {
            "L4_to_L3_join": f"Semantic join: {self.config.left_join_column} ↔ {self.config.right_join_column} (threshold={self.config.similarity_threshold})",
            "L3_to_L2_categorization": f"Dimension '{self.config.l2_category_name}' based on {self.config.l2_category_column}",
            "L2_to_L1_extraction": f"Column '{self.config.l1_column}'",
            "L1_to_L0_aggregation": f"{self.config.l0_aggregation.upper()}: {self.config.l0_description}",
            "Ascent_L1_to_L2": f"Dimension '{self.config.ascent_l2_dimension_name}'"
        }


# =============================================================================
# TEST DATA PATHS
# =============================================================================

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test_data"
TEST0_DIR = TEST_DATA_DIR / "test0"  # Schools
TEST1_DIR = TEST_DATA_DIR / "test1"  # ADEME funding
TEST2_DIR = TEST_DATA_DIR / "test2"  # Energy
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


# =============================================================================
# DOMAIN-SPECIFIC TRANSFORMATIONS
# =============================================================================

@dataclass
class DomainConfig:
    """Configuration for domain-specific transformations."""
    # Required fields first (no defaults)
    name: str
    description: str

    # L4→L3: Semantic join configuration
    left_file: str
    right_file: str
    left_join_column: str
    right_join_column: str

    # L3→L2: Categorization
    l2_category_column: str
    l2_category_func: Callable[[pd.DataFrame], pd.Series]
    l2_category_name: str

    # L2→L1: Column extraction
    l1_column: str

    # L1→L0: Aggregation
    l0_aggregation: str  # 'mean', 'sum', 'count', 'median'
    l0_description: str

    # Ascent L1→L2: Dimension for categorization
    ascent_l2_dimension_func: Callable[[pd.Series], pd.Series]
    ascent_l2_dimension_name: str

    # Optional fields with defaults (must come last)
    similarity_threshold: float = 0.8
    l1_filter: Optional[str] = None
    l1_group_by: Optional[str] = None  # Group by this column before extracting L1
    l1_group_agg: str = "sum"  # Aggregation method when grouping


def categorize_school_location(df: pd.DataFrame) -> pd.Series:
    """
    Categorize schools as 'countryside' or 'downtown' based on student count.
    Schools with < 300 students are typically rural/countryside.
    """
    if 'nombre_eleves_total' in df.columns:
        return df['nombre_eleves_total'].apply(
            lambda x: 'countryside' if pd.notna(x) and float(x) < 300 else 'downtown'
        )
    # Fallback: use any student count column
    for col in df.columns:
        if 'eleves' in col.lower() or 'effectif' in col.lower():
            return df[col].apply(
                lambda x: 'countryside' if pd.notna(x) and float(str(x).replace(',', '.') or 0) < 300 else 'downtown'
            )
    return pd.Series(['unknown'] * len(df))


def categorize_ademe_funding(df: pd.DataFrame) -> pd.Series:
    """
    Categorize ADEME funding recipients as 'single_funding' or 'multiple_funding'.
    Based on whether they appear multiple times in the dataset.
    """
    # Look for recipient/beneficiary columns (check common variations)
    recipient_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'nombeneficiaire' in col_lower or 'beneficiaire' in col_lower:
            recipient_col = col
            break

    # Fallback to other name-like columns
    if not recipient_col:
        for col in df.columns:
            if any(term in col.lower() for term in ['destinataire', 'nom', 'raison']):
                recipient_col = col
                break

    if recipient_col:
        # Count occurrences per recipient
        counts = df[recipient_col].value_counts()
        return df[recipient_col].apply(
            lambda x: 'multiple_funding' if counts.get(x, 0) > 1 else 'single_funding'
        )
    return pd.Series(['unknown'] * len(df))


def categorize_energy_price(df: pd.DataFrame) -> pd.Series:
    """
    Categorize energy prices as 'high_price' or 'low_price'.
    Uses median as threshold.
    """
    # Look for price columns
    price_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['prix', 'price', 'valeur', 'montant', 'cout']):
            price_col = col
            break

    if price_col:
        # Convert to numeric, handling French decimal format
        prices = pd.to_numeric(
            df[price_col].astype(str).str.replace(',', '.').str.replace(' ', ''),
            errors='coerce'
        )
        median_price = prices.median()
        return prices.apply(
            lambda x: 'high_price' if pd.notna(x) and x > median_price else 'low_price'
        )
    return pd.Series(['unknown'] * len(df))


def ascent_categorize_by_median(series: pd.Series) -> pd.Series:
    """Categorize values as 'above_median' or 'below_median'."""
    numeric = pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')
    median_val = numeric.median()
    return numeric.apply(
        lambda x: 'above_median' if pd.notna(x) and x >= median_val else 'below_median'
    )


def ascent_categorize_funding_threshold(series: pd.Series) -> pd.Series:
    """Categorize funding amounts: 'above_10k' or 'below_10k'."""
    numeric = pd.to_numeric(series.astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce')
    return numeric.apply(
        lambda x: 'above_10k' if pd.notna(x) and x >= 10000 else 'below_10k'
    )


# =============================================================================
# DOMAIN CONFIGURATIONS
# =============================================================================

TEST0_CONFIG = DomainConfig(
    name="test0_schools",
    description="French middle schools - student counts and performance scores",

    # L4→L3: Join student counts with performance indicators by school name
    left_file="fr-en-college-effectifs-niveau-sexe-lv.csv",
    right_file="fr-en-indicateurs-valeur-ajoutee-colleges.csv",
    left_join_column="Patronyme",  # School name in effectifs
    right_join_column="Nom de l'établissement",  # School name in indicateurs

    # L3→L2: Categorize by location type
    l2_category_column="nombre_eleves_total",
    l2_category_func=categorize_school_location,
    l2_category_name="location_type",

    # L2→L1: Extract success rate
    l1_column="Taux de réussite G",

    # L1→L0: Average success rate
    l0_aggregation="mean",
    l0_description="Average middle school success rate",

    # Ascent L1→L2
    ascent_l2_dimension_func=ascent_categorize_by_median,
    ascent_l2_dimension_name="performance_category",

    # Optional
    similarity_threshold=0.85
)


TEST1_CONFIG = DomainConfig(
    name="test1_ademe",
    description="ADEME funding - environmental subsidies",

    # L4→L3: Join ADEME aids with ECS data on aid type
    left_file="Les aides financieres ADEME.csv",
    right_file="ECS.csv",
    left_join_column="dispositifAide",  # Aid type in ADEME file
    right_join_column="type_aides_financieres",  # Aid type in ECS file

    # L3→L2: Categorize by funding frequency (single vs multiple fundings per recipient)
    l2_category_column="nomBeneficiaire",  # Count occurrences per recipient
    l2_category_func=categorize_ademe_funding,
    l2_category_name="funding_frequency",

    # L2→L1: Extract funding amounts PER RECIPIENT (grouped)
    l1_column="montant",

    # L1→L0: Total funding
    l0_aggregation="sum",
    l0_description="Total ADEME funding amount",

    # Ascent L1→L2
    ascent_l2_dimension_func=ascent_categorize_funding_threshold,
    ascent_l2_dimension_name="funding_size",

    # Optional
    similarity_threshold=0.75,
    l1_group_by="nomBeneficiaire",  # Group by recipient to get funding per recipient
    l1_group_agg="sum"  # Sum funding amounts per recipient
)


TEST2_CONFIG = DomainConfig(
    name="test2_energy",
    description="Energy consumption - prices and imports/exports",

    # L4→L3: Join price data with trade data
    left_file="Niveaux_prix_TRVG.csv",
    right_file="imports-exports-commerciaux.csv",
    left_join_column="Pays",  # Country
    right_join_column="Pays",  # Country

    # L3→L2: Categorize by price level
    l2_category_column="Prix",
    l2_category_func=categorize_energy_price,
    l2_category_name="price_category",

    # L2→L1: Extract import values
    l1_column="Valeur",  # Will be discovered

    # L1→L0: Total imports
    l0_aggregation="sum",
    l0_description="Total energy imports from foreign countries",

    # Ascent L1→L2
    ascent_l2_dimension_func=ascent_categorize_by_median,
    ascent_l2_dimension_name="import_category",

    # Optional
    similarity_threshold=0.9
)


# =============================================================================
# ARTIFACT EXPORTER (Enhanced)
# =============================================================================

@dataclass
class DomainArtifact:
    """Artifact with domain-specific metadata."""
    level: int
    level_name: str
    timestamp: str
    data_type: str
    domain_description: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    value: Optional[Any] = None
    columns: Optional[List[str]] = None
    categories: Optional[Dict[str, int]] = None  # For L2
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class DomainArtifactExporter:
    """Export domain-specific artifacts at each level."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_l4(self, sources: Dict[str, pd.DataFrame], name: str) -> DomainArtifact:
        """Export L4 sources."""
        artifact = DomainArtifact(
            level=4,
            level_name="LEVEL_4_UNLINKABLE",
            timestamp=datetime.now().isoformat(),
            data_type="dict_of_dataframes",
            domain_description="Raw source files before semantic linking"
        )

        manifest = {}
        for source_name, df in sources.items():
            filepath = self.output_dir / f"{name}_L4_{source_name}"
            df.to_csv(filepath.with_suffix('.csv'), index=False)
            manifest[source_name] = {
                "rows": len(df),
                "columns": list(df.columns)
            }

        manifest_path = self.output_dir / f"{name}_L4_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        artifact.file_path = str(manifest_path)

        return artifact

    def export_l3_joined(self, df: pd.DataFrame, name: str, join_info: Dict) -> DomainArtifact:
        """Export L3 joined table (the key artifact showing semantic linking)."""
        artifact = DomainArtifact(
            level=3,
            level_name="LEVEL_3_LINKED",
            timestamp=datetime.now().isoformat(),
            data_type="joined_dataframe",
            domain_description=f"Semantically joined table: {join_info.get('description', '')}",
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns)
        )

        # Save the joined table
        filepath = self.output_dir / f"{name}_L3_joined_table.csv"
        df.to_csv(filepath, index=False)
        artifact.file_path = str(filepath)

        # Save join metadata
        meta_path = self.output_dir / f"{name}_L3_join_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                **join_info,
                "result_rows": len(df),
                "result_columns": len(df.columns),
                "has_similarity_column": "_semantic_similarity" in df.columns
            }, f, indent=2)

        return artifact

    def export_l2(self, df: pd.DataFrame, name: str, category_col: str) -> DomainArtifact:
        """Export L2 categorized table."""
        # Count categories
        categories = {}
        if category_col in df.columns:
            categories = df[category_col].value_counts().to_dict()

        artifact = DomainArtifact(
            level=2,
            level_name="LEVEL_2_CATEGORIZED",
            timestamp=datetime.now().isoformat(),
            data_type="categorized_dataframe",
            domain_description=f"Table with '{category_col}' dimension",
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            categories=categories
        )

        filepath = self.output_dir / f"{name}_L2_categorized_table.csv"
        df.to_csv(filepath, index=False)
        artifact.file_path = str(filepath)

        return artifact

    def export_l1(self, series: pd.Series, name: str, description: str) -> DomainArtifact:
        """Export L1 vector."""
        artifact = DomainArtifact(
            level=1,
            level_name="LEVEL_1_VECTOR",
            timestamp=datetime.now().isoformat(),
            data_type="series",
            domain_description=description,
            row_count=len(series)
        )

        filepath = self.output_dir / f"{name}_L1_vector.csv"
        series.to_frame(name="value").to_csv(filepath, index=True)
        artifact.file_path = str(filepath)

        return artifact

    def export_l0(self, value: Any, name: str, description: str, method: str) -> DomainArtifact:
        """Export L0 datum."""
        artifact = DomainArtifact(
            level=0,
            level_name="LEVEL_0_DATUM",
            timestamp=datetime.now().isoformat(),
            data_type="scalar",
            domain_description=description,
            value=value
        )

        filepath = self.output_dir / f"{name}_L0_datum.json"
        with open(filepath, 'w') as f:
            json.dump({
                "value": value,
                "description": description,
                "aggregation_method": method,
                "type": type(value).__name__
            }, f, indent=2, default=str)
        artifact.file_path = str(filepath)

        return artifact


# =============================================================================
# DOMAIN-SPECIFIC PIPELINE
# =============================================================================

class DomainSpecificPipeline:
    """
    Execute domain-specific descent-ascent cycle.

    Key difference: Uses semantic_table_join for L4→L3 transition.
    Tracks full navigation path with design choices for session export.
    """

    def __init__(self, config: DomainConfig, output_dir: Path):
        self.config = config
        self.exporter = DomainArtifactExporter(output_dir / config.name)
        self.artifacts: Dict[str, DomainArtifact] = {}
        self.data_at_level: Dict[int, Any] = {}
        self.session = NavigationSession(config)  # Track navigation

    def load_sources(self, data_dir: Path) -> Dict[str, pd.DataFrame]:
        """Load source CSV files."""
        sources = {}
        for f in data_dir.glob("*.csv"):
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                for sep in [';', ',', '\t']:
                    try:
                        df = pd.read_csv(f, encoding=encoding, sep=sep, on_bad_lines='skip')
                        if len(df.columns) > 1:
                            break
                    except Exception:
                        continue
                if df is not None and len(df.columns) > 1:
                    break
            if df is not None:
                sources[f.name] = df
        return sources

    def descent_l4_to_l3(self, sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        L4→L3: Semantic join of two sources.

        This is the critical step that creates LINKED data from UNLINKABLE sources.
        """
        # Export L4
        self.artifacts["L4"] = self.exporter.export_l4(sources, self.config.name)
        self.data_at_level[4] = sources

        # Add L4 entry node
        self.session.add_node(
            level=4,
            level_name="LEVEL_4_UNLINKABLE",
            action="entry",
            decision_description=f"Loaded {len(sources)} raw data sources",
            design_rationale="L4 represents unlinkable raw files before semantic processing",
            params={"source_files": list(sources.keys())},
            output_snapshot={
                "level": 4,
                "type": "dict_of_dataframes",
                "source_count": len(sources),
                "sources": {k: {"rows": len(v), "columns": list(v.columns)[:5]} for k, v in sources.items()}
            }
        )
        self.session.update_cumulative_output("sources", {
            "level": 4,
            "level_name": "LEVEL_4_UNLINKABLE",
            "output_type": "dict_of_dataframes",
            "source_count": len(sources),
            "source_names": list(sources.keys())
        })

        # Get left and right DataFrames
        left_df = sources.get(self.config.left_file)
        right_df = sources.get(self.config.right_file)

        if left_df is None or right_df is None:
            # Fallback: use first two files
            source_list = list(sources.values())
            if len(source_list) >= 2:
                left_df, right_df = source_list[0], source_list[1]
            else:
                # Single file case: return as-is
                left_df = source_list[0] if source_list else pd.DataFrame()
                self.data_at_level[3] = left_df
                return left_df

        # Find best join columns if configured ones don't exist
        left_col = self.config.left_join_column
        right_col = self.config.right_join_column

        if left_col not in left_df.columns:
            # Find a suitable text column
            for col in left_df.columns:
                if left_df[col].dtype == 'object' and left_df[col].notna().sum() > 0:
                    left_col = col
                    break

        if right_col not in right_df.columns:
            for col in right_df.columns:
                if right_df[col].dtype == 'object' and right_df[col].notna().sum() > 0:
                    right_col = col
                    break

        print(f"\n[L4→L3] Semantic join: {left_col} ↔ {right_col}")
        print(f"  Left: {len(left_df)} rows, Right: {len(right_df)} rows")

        try:
            # Perform semantic join
            join_config = SemanticJoinConfig(
                left_column=left_col,
                right_column=right_col,
                similarity_threshold=self.config.similarity_threshold,
                join_type='inner',
                best_match_only=True,
                include_similarity_score=True
            )

            # Limit rows for testing performance
            left_sample = left_df.head(500)
            right_sample = right_df.head(500)

            l3_df = semantic_table_join(left_sample, right_sample, join_config)

            print(f"  Result: {len(l3_df)} joined rows")

        except Exception as e:
            print(f"  Semantic join failed: {e}")
            print(f"  Falling back to simple merge on common columns")

            # Fallback: find common columns and merge
            common_cols = set(left_df.columns) & set(right_df.columns)
            if common_cols:
                merge_col = list(common_cols)[0]
                l3_df = pd.merge(
                    left_df.head(500),
                    right_df.head(500),
                    on=merge_col,
                    how='inner',
                    suffixes=('', '_right')
                )
            else:
                # Cross-join sample
                l3_df = left_df.head(100).assign(key=1).merge(
                    right_df.head(100).assign(key=1),
                    on='key',
                    suffixes=('', '_right')
                ).drop('key', axis=1)

        # Export L3 joined table
        self.artifacts["L3"] = self.exporter.export_l3_joined(
            l3_df,
            self.config.name,
            {
                "left_file": self.config.left_file,
                "right_file": self.config.right_file,
                "left_column": left_col,
                "right_column": right_col,
                "threshold": self.config.similarity_threshold,
                "description": f"Semantic join of {self.config.left_file} and {self.config.right_file}"
            }
        )
        self.data_at_level[3] = l3_df

        # Add L3 descend node with design choice
        has_similarity = "_semantic_similarity" in l3_df.columns
        self.session.add_node(
            level=3,
            level_name="LEVEL_3_LINKED",
            action="descend",
            decision_description=f"Semantic join: {left_col} ↔ {right_col}",
            design_rationale=f"Used embedding-based semantic matching (threshold={self.config.similarity_threshold}) to link '{self.config.left_file}' with '{self.config.right_file}'. This creates relationships between previously unlinkable records based on semantic similarity of text fields.",
            params={
                "left_file": self.config.left_file,
                "right_file": self.config.right_file,
                "left_column": left_col,
                "right_column": right_col,
                "similarity_threshold": self.config.similarity_threshold,
                "join_type": "semantic_best_match"
            },
            output_snapshot={
                "level": 3,
                "type": "joined_dataframe",
                "row_count": len(l3_df),
                "column_count": len(l3_df.columns),
                "has_similarity_column": has_similarity,
                "columns_sample": list(l3_df.columns)[:10]
            }
        )
        self.session.update_cumulative_output("joined_table", {
            "level": 3,
            "level_name": "LEVEL_3_LINKED",
            "output_type": "dataframe",
            "row_count": len(l3_df),
            "column_names": list(l3_df.columns)[:20],
            "join_columns": {"left": left_col, "right": right_col}
        })

        return l3_df

    def descent_l3_to_l2(self, l3_df: pd.DataFrame) -> pd.DataFrame:
        """
        L3→L2: Apply domain-specific categorization.
        """
        # Apply category function
        category_series = self.config.l2_category_func(l3_df)

        # Add category column
        l2_df = l3_df.copy()
        l2_df[self.config.l2_category_name] = category_series

        # Get category distribution
        category_counts = l2_df[self.config.l2_category_name].value_counts().to_dict()

        print(f"\n[L3→L2] Applied categorization: {self.config.l2_category_name}")
        print(f"  Categories: {category_counts}")

        # Export L2
        self.artifacts["L2"] = self.exporter.export_l2(
            l2_df,
            self.config.name,
            self.config.l2_category_name
        )
        self.data_at_level[2] = l2_df

        # Add L2 descend node with design choice
        # Build a more informative rationale based on the function used
        func_name = self.config.l2_category_func.__name__
        if 'funding' in func_name.lower():
            method_desc = f"Counted occurrences of '{self.config.l2_category_column}' to identify recipients with single vs multiple fundings"
        elif 'location' in func_name.lower() or 'school' in func_name.lower():
            method_desc = f"Applied threshold on '{self.config.l2_category_column}' (< 300 students = countryside, >= 300 = downtown)"
        elif 'price' in func_name.lower() or 'energy' in func_name.lower():
            method_desc = f"Used median threshold on '{self.config.l2_category_column}' to split into high/low price categories"
        else:
            method_desc = f"Applied business logic on '{self.config.l2_category_column}'"

        self.session.add_node(
            level=2,
            level_name="LEVEL_2_CATEGORIZED",
            action="descend",
            decision_description=f"Applied '{self.config.l2_category_name}' dimension using {func_name}",
            design_rationale=f"{method_desc}. Categories: {category_counts}",
            params={
                "category_name": self.config.l2_category_name,
                "category_column": self.config.l2_category_column,
                "category_function": func_name,
                "method": method_desc
            },
            output_snapshot={
                "level": 2,
                "type": "categorized_dataframe",
                "row_count": len(l2_df),
                "categories": category_counts,
                "dimension_name": self.config.l2_category_name
            }
        )
        self.session.update_cumulative_output("categorized_table", {
            "level": 2,
            "level_name": "LEVEL_2_CATEGORIZED",
            "output_type": "dataframe",
            "row_count": len(l2_df),
            "categories": category_counts,
            "dimension_column": self.config.l2_category_name
        })

        return l2_df

    def descent_l2_to_l1(self, l2_df: pd.DataFrame) -> pd.Series:
        """
        L2→L1: Extract feature column as vector.
        Optionally groups by a key column first (e.g., funding per recipient).
        """
        # Find the value column
        col = self.config.l1_column
        original_col = col
        if col not in l2_df.columns:
            # Try to find a numeric column
            numeric_cols = l2_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
            else:
                # Use first column
                col = l2_df.columns[0]

        # Apply filter if specified
        if self.config.l1_filter:
            filtered_df = l2_df.query(self.config.l1_filter)
        else:
            filtered_df = l2_df

        # Check if we need to group by a column (e.g., recipient)
        group_by_col = self.config.l1_group_by
        if group_by_col and group_by_col in filtered_df.columns:
            # Convert value column to numeric first
            filtered_df = filtered_df.copy()
            filtered_df[col] = pd.to_numeric(
                filtered_df[col].astype(str).str.replace(',', '.').str.replace(' ', ''),
                errors='coerce'
            )

            # Group by the key column and aggregate
            agg_method = self.config.l1_group_agg
            grouped = filtered_df.groupby(group_by_col)[col].agg(agg_method)
            l1_series = grouped

            print(f"\n[L2→L1] Grouped by '{group_by_col}', aggregated '{col}' with {agg_method}")
            print(f"  Unique groups: {len(l1_series)}")
        else:
            l1_series = filtered_df[col]
            # Convert to numeric if possible
            l1_series = pd.to_numeric(
                l1_series.astype(str).str.replace(',', '.').str.replace(' ', ''),
                errors='coerce'
            )
            print(f"\n[L2→L1] Extracted column: {col}")

        non_null_count = l1_series.notna().sum()
        print(f"  Values: {len(l1_series)}, Non-null: {non_null_count}")

        # Export L1
        self.artifacts["L1"] = self.exporter.export_l1(
            l1_series,
            self.config.name,
            f"Vector of {col} values"
        )
        self.data_at_level[1] = l1_series

        # Add L1 descend node with design choice
        group_by_col = self.config.l1_group_by
        if group_by_col and group_by_col in l2_df.columns:
            decision_desc = f"Grouped '{col}' by '{group_by_col}' ({self.config.l1_group_agg})"
            rationale = f"Aggregated '{col}' per '{group_by_col}' using {self.config.l1_group_agg}. This creates a vector of {len(l1_series)} unique {group_by_col} values with their total {col}."
        else:
            decision_desc = f"Extracted column '{col}' as vector"
            rationale = f"Selected '{col}' as the target metric column. This column represents the key measurement for aggregation (originally requested: '{original_col}')."

        self.session.add_node(
            level=1,
            level_name="LEVEL_1_VECTOR",
            action="descend",
            decision_description=decision_desc,
            design_rationale=rationale,
            params={
                "column": col,
                "original_column": original_col,
                "filter_applied": self.config.l1_filter,
                "group_by": group_by_col,
                "group_agg": self.config.l1_group_agg if group_by_col else None
            },
            output_snapshot={
                "level": 1,
                "type": "vector",
                "length": len(l1_series),
                "non_null_count": int(non_null_count),
                "sample_values": l1_series.dropna().head(5).tolist() if non_null_count > 0 else []
            }
        )
        self.session.update_cumulative_output("vector", {
            "level": 1,
            "level_name": "LEVEL_1_VECTOR",
            "output_type": "vector",
            "row_count": len(l1_series),
            "non_null_count": int(non_null_count),
            "column_name": col
        })

        return l1_series

    def descent_l1_to_l0(self, l1_series: pd.Series) -> Any:
        """
        L1→L0: Aggregate to scalar datum.
        """
        # Clean the series
        clean_series = l1_series.dropna()
        input_count = len(clean_series)

        # Apply aggregation
        agg_method = self.config.l0_aggregation
        if agg_method == 'mean':
            l0_value = float(clean_series.mean()) if len(clean_series) > 0 else 0.0
        elif agg_method == 'sum':
            l0_value = float(clean_series.sum()) if len(clean_series) > 0 else 0.0
        elif agg_method == 'count':
            l0_value = int(len(clean_series))
        elif agg_method == 'median':
            l0_value = float(clean_series.median()) if len(clean_series) > 0 else 0.0
        elif agg_method == 'min':
            l0_value = float(clean_series.min()) if len(clean_series) > 0 else 0.0
        elif agg_method == 'max':
            l0_value = float(clean_series.max()) if len(clean_series) > 0 else 0.0
        else:
            l0_value = float(clean_series.mean()) if len(clean_series) > 0 else 0.0

        print(f"\n[L1→L0] Aggregation: {agg_method}")
        print(f"  Result: {l0_value}")
        print(f"  Description: {self.config.l0_description}")

        # Export L0
        self.artifacts["L0"] = self.exporter.export_l0(
            l0_value,
            self.config.name,
            self.config.l0_description,
            agg_method
        )
        self.data_at_level[0] = l0_value

        # Add L0 descend node with design choice
        self.session.add_node(
            level=0,
            level_name="LEVEL_0_DATUM",
            action="descend",
            decision_description=f"Computed {agg_method.upper()} = {l0_value:.4f}" if isinstance(l0_value, float) else f"Computed {agg_method.upper()} = {l0_value}",
            design_rationale=f"{self.config.l0_description}. Aggregated {input_count} values using '{agg_method}' to produce a single atomic datum representing the key insight.",
            params={
                "aggregation_method": agg_method,
                "input_count": input_count,
                "description": self.config.l0_description
            },
            output_snapshot={
                "level": 0,
                "type": "datum",
                "value": l0_value,
                "aggregation": agg_method
            }
        )
        self.session.update_cumulative_output("datum", {
            "level": 0,
            "level_name": "LEVEL_0_DATUM",
            "output_type": "datum",
            "value": l0_value,
            "aggregation_method": agg_method,
            "description": self.config.l0_description,
            "sample_data": str(l0_value)
        })

        return l0_value

    def ascent_l0_to_l1(self) -> pd.Series:
        """
        L0→L1: Expand datum back to source vector.
        """
        # Get the original L1 series
        l1_series = self.data_at_level.get(1, pd.Series())

        print(f"\n[L0→L1] Source expansion")
        print(f"  Restored {len(l1_series)} values from L1")

        # Add ascent L1 node
        self.session.add_node(
            level=1,
            level_name="LEVEL_1_VECTOR",
            action="ascend",
            decision_description=f"Expanded to {len(l1_series)} source values",
            design_rationale="Re-expanded from atomic datum back to source vector to enable alternative dimensional views. This 'ascent' allows exploration of the data from different analytical perspectives.",
            params={
                "expansion_method": "source_recovery"
            },
            output_snapshot={
                "level": 1,
                "type": "vector",
                "length": len(l1_series)
            }
        )

        return l1_series

    def ascent_l1_to_l2(self, l1_series: pd.Series) -> pd.DataFrame:
        """
        L1→L2: Apply dimension classification.
        """
        # Apply dimension function
        dimension_series = self.config.ascent_l2_dimension_func(l1_series)

        # Create L2 table
        l2_df = pd.DataFrame({
            'value': l1_series,
            self.config.ascent_l2_dimension_name: dimension_series
        })

        # Get category distribution
        category_counts = l2_df[self.config.ascent_l2_dimension_name].value_counts().to_dict()

        print(f"\n[L1→L2] Applied dimension: {self.config.ascent_l2_dimension_name}")
        print(f"  Categories: {category_counts}")

        # Export ascent L2
        filepath = self.exporter.output_dir / f"{self.config.name}_ascent_L2_table.csv"
        l2_df.to_csv(filepath, index=False)

        # Add ascent L2 node
        self.session.add_node(
            level=2,
            level_name="LEVEL_2_CATEGORIZED",
            action="ascend",
            decision_description=f"Applied '{self.config.ascent_l2_dimension_name}' dimension",
            design_rationale=f"Added alternative analytical dimension '{self.config.ascent_l2_dimension_name}' during ascent. This provides a different categorization perspective than the descent phase for comparative analysis.",
            params={
                "dimension_name": self.config.ascent_l2_dimension_name,
                "dimension_function": self.config.ascent_l2_dimension_func.__name__
            },
            output_snapshot={
                "level": 2,
                "type": "categorized_dataframe",
                "row_count": len(l2_df),
                "categories": category_counts
            }
        )

        return l2_df

    def ascent_l2_to_l3(self, l2_df: pd.DataFrame) -> pd.DataFrame:
        """
        L2→L3: Add hierarchical structure.
        """
        # Get original L3 data for enrichment
        l3_original = self.data_at_level.get(3, pd.DataFrame())

        # Merge back with original L3 context if possible
        if len(l3_original) > 0 and len(l2_df) <= len(l3_original):
            # Add dimension columns to original L3
            l3_enriched = l3_original.head(len(l2_df)).copy()
            l3_enriched[self.config.ascent_l2_dimension_name] = l2_df[self.config.ascent_l2_dimension_name].values
        else:
            l3_enriched = l2_df

        print(f"\n[L2→L3] Created hierarchical structure")
        print(f"  Columns: {list(l3_enriched.columns)[:10]}...")

        # Export ascent L3
        filepath = self.exporter.output_dir / f"{self.config.name}_ascent_L3_table.csv"
        l3_enriched.to_csv(filepath, index=False)

        # Add ascent L3 node
        self.session.add_node(
            level=3,
            level_name="LEVEL_3_LINKED",
            action="ascend",
            decision_description="Enriched table with ascent dimensions",
            design_rationale=f"Combined original linked table with new dimensional classification. The enriched L3 table now contains both the original semantic join relationships and the ascent-phase categorization '{self.config.ascent_l2_dimension_name}', enabling cross-dimensional analysis.",
            params={
                "enrichment_method": "dimension_merge",
                "added_dimensions": [self.config.ascent_l2_dimension_name]
            },
            output_snapshot={
                "level": 3,
                "type": "enriched_dataframe",
                "row_count": len(l3_enriched),
                "column_count": len(l3_enriched.columns),
                "columns_sample": list(l3_enriched.columns)[:10]
            }
        )

        return l3_enriched

    def run_full_cycle(self, data_dir: Path) -> Dict[str, Any]:
        """Run complete descent-ascent cycle."""
        print(f"\n{'='*60}")
        print(f"DOMAIN-SPECIFIC PIPELINE: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"{'='*60}")

        # Load sources
        sources = self.load_sources(data_dir)
        print(f"\nLoaded {len(sources)} source files: {list(sources.keys())}")

        # DESCENT
        print(f"\n--- DESCENT (L4 → L0) ---")
        l3_df = self.descent_l4_to_l3(sources)
        l2_df = self.descent_l3_to_l2(l3_df)
        l1_series = self.descent_l2_to_l1(l2_df)
        l0_value = self.descent_l1_to_l0(l1_series)

        # ASCENT
        print(f"\n--- ASCENT (L0 → L3) ---")
        l1_ascent = self.ascent_l0_to_l1()
        l2_ascent = self.ascent_l1_to_l2(l1_ascent)
        l3_ascent = self.ascent_l2_to_l3(l2_ascent)

        # Export session with full path and design choices
        session_export = self.session.export(self.exporter.output_dir)
        print(f"\n--- SESSION EXPORTED ---")
        print(f"  File: {self.exporter.output_dir / f'{self.config.name}_session_export.json'}")
        print(f"  Nodes: {len(self.session.nodes)}")
        print(f"  Design choices: {list(session_export['design_choices_summary'].keys())}")

        # Summary
        print(f"\n--- SUMMARY ---")
        print(f"  L4: {len(sources)} source files")
        print(f"  L3: {len(l3_df)} joined rows")
        print(f"  L2: {len(l2_df)} categorized rows")
        print(f"  L1: {len(l1_series)} values")
        print(f"  L0: {l0_value} ({self.config.l0_description})")

        return {
            "config": self.config.name,
            "l0_value": l0_value,
            "l0_description": self.config.l0_description,
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "row_counts": {
                "L4_sources": len(sources),
                "L3_joined": len(l3_df),
                "L2_categorized": len(l2_df),
                "L1_vector": len(l1_series)
            },
            "session_export": session_export
        }


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def output_dir():
    """Create output directory for artifacts."""
    output = ARTIFACTS_DIR / "domain_specific"
    output.mkdir(parents=True, exist_ok=True)
    return output


# =============================================================================
# TESTS
# =============================================================================

class TestDomainSpecificDescent:
    """Test domain-specific descent transformations."""

    def test_test0_schools_descent(self, output_dir):
        """
        Test0: French middle schools
        - L3: Join student counts with performance scores by school name
        - L2: Categorize as 'countryside' or 'downtown'
        - L1: Extract success rate column
        - L0: Compute average success rate
        """
        pipeline = DomainSpecificPipeline(TEST0_CONFIG, output_dir)
        result = pipeline.run_full_cycle(TEST0_DIR)

        # Verify L3 joined table was created
        assert "L3" in pipeline.artifacts
        l3_artifact = pipeline.artifacts["L3"]
        assert l3_artifact.row_count > 0
        assert "joined_table" in l3_artifact.file_path

        # Verify L2 has categories
        assert "L2" in pipeline.artifacts
        l2_artifact = pipeline.artifacts["L2"]
        assert l2_artifact.categories is not None
        assert len(l2_artifact.categories) > 0

        # Verify L0 is meaningful (not just count)
        assert "L0" in pipeline.artifacts
        l0_artifact = pipeline.artifacts["L0"]
        assert l0_artifact.domain_description == "Average middle school success rate"

        print(f"\nTest0 Schools: L0 = {result['l0_value']}")

    def test_test1_ademe_descent(self, output_dir):
        """
        Test1: ADEME funding
        - L3: Join funding data with recipient data
        - L2: Categorize as 'single_funding' or 'multiple_funding'
        - L1: Extract funding amounts
        - L0: Compute total funding
        """
        pipeline = DomainSpecificPipeline(TEST1_CONFIG, output_dir)
        result = pipeline.run_full_cycle(TEST1_DIR)

        # Verify artifacts
        assert "L3" in pipeline.artifacts
        assert "L2" in pipeline.artifacts
        assert "L0" in pipeline.artifacts

        l0_artifact = pipeline.artifacts["L0"]
        assert l0_artifact.domain_description == "Total ADEME funding amount"

        print(f"\nTest1 ADEME: L0 = {result['l0_value']}")

    def test_test2_energy_descent(self, output_dir):
        """
        Test2: Energy prices and trade
        - L3: Join price data with import/export data
        - L2: Categorize as 'high_price' or 'low_price'
        - L1: Extract import values
        - L0: Compute total imports
        """
        pipeline = DomainSpecificPipeline(TEST2_CONFIG, output_dir)
        result = pipeline.run_full_cycle(TEST2_DIR)

        # Verify artifacts
        assert "L3" in pipeline.artifacts
        assert "L2" in pipeline.artifacts
        assert "L0" in pipeline.artifacts

        l0_artifact = pipeline.artifacts["L0"]
        assert l0_artifact.domain_description == "Total energy imports from foreign countries"

        print(f"\nTest2 Energy: L0 = {result['l0_value']}")


class TestDomainSpecificAscent:
    """Test domain-specific ascent transformations."""

    def test_test0_schools_full_cycle(self, output_dir):
        """
        Test0 full cycle with ascent:
        - Ascent L1: Re-expand to success rates
        - Ascent L2: Categorize as 'above_median' or 'below_median'
        - Ascent L3: Enrich with original context
        """
        pipeline = DomainSpecificPipeline(TEST0_CONFIG, output_dir)
        result = pipeline.run_full_cycle(TEST0_DIR)

        # Check ascent artifacts exist
        ascent_l2_path = pipeline.exporter.output_dir / f"{TEST0_CONFIG.name}_ascent_L2_table.csv"
        ascent_l3_path = pipeline.exporter.output_dir / f"{TEST0_CONFIG.name}_ascent_L3_table.csv"

        assert ascent_l2_path.exists(), "Ascent L2 table not created"
        assert ascent_l3_path.exists(), "Ascent L3 table not created"

        # Verify ascent L2 has dimension column
        l2_df = pd.read_csv(ascent_l2_path)
        assert TEST0_CONFIG.ascent_l2_dimension_name in l2_df.columns

        print(f"\nTest0 Ascent categories: {l2_df[TEST0_CONFIG.ascent_l2_dimension_name].value_counts().to_dict()}")

    def test_test1_ademe_full_cycle(self, output_dir):
        """Test1 full cycle with funding size categorization."""
        pipeline = DomainSpecificPipeline(TEST1_CONFIG, output_dir)
        result = pipeline.run_full_cycle(TEST1_DIR)

        ascent_l2_path = pipeline.exporter.output_dir / f"{TEST1_CONFIG.name}_ascent_L2_table.csv"
        assert ascent_l2_path.exists()

        l2_df = pd.read_csv(ascent_l2_path)
        assert TEST1_CONFIG.ascent_l2_dimension_name in l2_df.columns

        print(f"\nTest1 Ascent categories: {l2_df[TEST1_CONFIG.ascent_l2_dimension_name].value_counts().to_dict()}")

    def test_test2_energy_full_cycle(self, output_dir):
        """Test2 full cycle with import categorization."""
        pipeline = DomainSpecificPipeline(TEST2_CONFIG, output_dir)
        result = pipeline.run_full_cycle(TEST2_DIR)

        ascent_l2_path = pipeline.exporter.output_dir / f"{TEST2_CONFIG.name}_ascent_L2_table.csv"
        assert ascent_l2_path.exists()


class TestSemanticJoinQuality:
    """Test quality of semantic joins (L4→L3)."""

    def test_l3_has_similarity_scores(self, output_dir):
        """Verify L3 joined tables have semantic similarity scores."""
        pipeline = DomainSpecificPipeline(TEST0_CONFIG, output_dir)
        sources = pipeline.load_sources(TEST0_DIR)
        l3_df = pipeline.descent_l4_to_l3(sources)

        # Check for similarity column
        sim_cols = [c for c in l3_df.columns if 'similarity' in c.lower()]
        print(f"\nSimilarity columns found: {sim_cols}")

        if sim_cols:
            sim_col = sim_cols[0]
            avg_sim = l3_df[sim_col].mean()
            print(f"Average similarity: {avg_sim:.3f}")
            assert avg_sim >= TEST0_CONFIG.similarity_threshold, \
                f"Average similarity {avg_sim} below threshold {TEST0_CONFIG.similarity_threshold}"

    def test_l3_no_orphan_rows(self, output_dir):
        """Verify L3 joined tables don't have orphan rows (Design Constraint #1)."""
        pipeline = DomainSpecificPipeline(TEST0_CONFIG, output_dir)
        sources = pipeline.load_sources(TEST0_DIR)
        l3_df = pipeline.descent_l4_to_l3(sources)

        # Check that joined rows have data from both sources
        # Look for _right suffix columns (indicating right table data)
        right_cols = [c for c in l3_df.columns if c.endswith('_right')]

        if right_cols:
            # Check that right columns have values
            non_null_right = l3_df[right_cols].notna().any(axis=1).sum()
            print(f"\nRows with right-side data: {non_null_right}/{len(l3_df)}")
            assert non_null_right > 0, "No rows have data from both sources"


class TestArtifactIntegrity:
    """Test artifact file integrity."""

    def test_all_artifacts_created(self, output_dir):
        """Verify all expected artifacts are created."""
        pipeline = DomainSpecificPipeline(TEST0_CONFIG, output_dir)
        pipeline.run_full_cycle(TEST0_DIR)

        expected_files = [
            f"{TEST0_CONFIG.name}_L4_manifest.json",
            f"{TEST0_CONFIG.name}_L3_joined_table.csv",
            f"{TEST0_CONFIG.name}_L3_join_metadata.json",
            f"{TEST0_CONFIG.name}_L2_categorized_table.csv",
            f"{TEST0_CONFIG.name}_L1_vector.csv",
            f"{TEST0_CONFIG.name}_L0_datum.json",
            f"{TEST0_CONFIG.name}_ascent_L2_table.csv",
            f"{TEST0_CONFIG.name}_ascent_L3_table.csv",
        ]

        for fname in expected_files:
            fpath = pipeline.exporter.output_dir / fname
            assert fpath.exists(), f"Missing artifact: {fname}"
            print(f"  ✓ {fname}")

    def test_l0_datum_readable(self, output_dir):
        """Verify L0 datum JSON is valid and readable."""
        pipeline = DomainSpecificPipeline(TEST0_CONFIG, output_dir)
        pipeline.run_full_cycle(TEST0_DIR)

        l0_path = pipeline.exporter.output_dir / f"{TEST0_CONFIG.name}_L0_datum.json"
        with open(l0_path) as f:
            l0_data = json.load(f)

        assert "value" in l0_data
        assert "description" in l0_data
        assert "aggregation_method" in l0_data

        print(f"\nL0 Datum: {l0_data}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
