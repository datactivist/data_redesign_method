"""
AI-Powered Relationship Discovery Module

This module provides automatic discovery of relationships between CSV files using
a three-tier approach:
- Tier 1: Column name heuristics (fastest)
- Tier 2: Value overlap detection (fast)
- Tier 3: Semantic embeddings (when needed)

Feature: 002-ascent-functionality
Date: 2025-12-04
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np


@dataclass
class EntitySuggestion:
    """AI-suggested entity from CSV analysis."""
    id: str
    source_file: str
    suggested_name: str
    key_column: str
    property_columns: List[str]
    row_count: int
    confidence: float  # 0.0-1.0
    reasoning: str  # Natural language explanation
    accepted: Optional[bool] = None
    user_edited_name: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Get the name to display (user-edited or suggested)."""
        return self.user_edited_name or self.suggested_name


@dataclass
class RelationshipSuggestion:
    """AI-suggested relationship between entities."""
    id: str
    start_entity_id: str
    end_entity_id: str
    start_entity_name: str
    end_entity_name: str
    start_column: str
    end_column: str
    discovery_method: str  # "name_match", "value_overlap", "semantic"
    confidence: float  # 0.0-1.0
    natural_description: str  # "Schools connect to Students through..."
    matching_values_count: int
    sample_matches: List[str] = field(default_factory=list)
    accepted: Optional[bool] = None


@dataclass
class DiscoveryResult:
    """Complete discovery results for all uploaded files."""
    entity_suggestions: List[EntitySuggestion] = field(default_factory=list)
    relationship_suggestions: List[RelationshipSuggestion] = field(default_factory=list)
    analysis_time_ms: float = 0.0


class RelationshipDiscovery:
    """
    Discovers relationships between CSV files using AI-powered analysis.

    Uses three-tier approach optimized for large datasets (100K+ rows):
    - Tier 1: Column name heuristics (~5ms)
    - Tier 2: Value overlap with sampling (~100ms)
    - Tier 3: Semantic embeddings (~2s, only when needed)
    """

    # Common ID column patterns
    ID_PATTERNS = [
        'id', 'code', 'key', 'num', 'ref', 'uai', 'siren', 'siret',
        'numero', 'identifiant', 'matricule'
    ]

    # Common ID suffixes
    ID_SUFFIXES = ['_id', '_code', '_key', '_num', '_ref', 'id', 'code']

    # French-English common column translations
    TRANSLATIONS = {
        'nom': 'name', 'prenom': 'firstname', 'ville': 'city',
        'departement': 'department', 'region': 'region',
        'date': 'date', 'annee': 'year', 'mois': 'month'
    }

    def __init__(
        self,
        dataframes: Dict[str, pd.DataFrame],
        sample_size: int = 1000,
        min_overlap_threshold: float = 0.05,
        semantic_threshold: float = 0.7
    ):
        """
        Initialize the discovery engine.

        Args:
            dataframes: Dict mapping filename to DataFrame
            sample_size: Max values to sample for overlap detection
            min_overlap_threshold: Minimum Jaccard similarity for relationship
            semantic_threshold: Minimum cosine similarity for semantic matching
        """
        self.dataframes = dataframes
        self.sample_size = sample_size
        self.min_overlap_threshold = min_overlap_threshold
        self.semantic_threshold = semantic_threshold
        self._semantic_model = None  # Lazy-loaded

    def discover_all(self) -> DiscoveryResult:
        """
        Run complete discovery process.

        Returns:
            DiscoveryResult with entity and relationship suggestions
        """
        import time
        start_time = time.time()

        # Step 1: Discover entities
        entities = self.discover_entities()

        # Step 2: Discover relationships
        relationships = self.discover_relationships(entities)

        elapsed_ms = (time.time() - start_time) * 1000

        return DiscoveryResult(
            entity_suggestions=entities,
            relationship_suggestions=relationships,
            analysis_time_ms=elapsed_ms
        )

    def discover_entities(self) -> List[EntitySuggestion]:
        """
        Analyze each CSV and suggest entity names + key columns.

        Returns:
            List of EntitySuggestion for each uploaded file
        """
        suggestions = []

        for i, (filename, df) in enumerate(self.dataframes.items()):
            # Generate friendly name from filename
            friendly_name = self._humanize_filename(filename)

            # Detect key column
            key_col, key_confidence, key_reasoning = self._detect_key_column(df)

            # Build suggestion
            suggestion = EntitySuggestion(
                id=f"entity_{i}_{hash(filename) % 10000}",
                source_file=filename,
                suggested_name=friendly_name,
                key_column=key_col,
                property_columns=list(df.columns),
                row_count=len(df),
                confidence=key_confidence,
                reasoning=key_reasoning
            )
            suggestions.append(suggestion)

        return suggestions

    def discover_relationships(
        self,
        entities: List[EntitySuggestion]
    ) -> List[RelationshipSuggestion]:
        """
        Find relationships between entities using three-tier discovery.

        Args:
            entities: List of discovered entities

        Returns:
            List of RelationshipSuggestion between entity pairs
        """
        suggestions = []

        # Compare each pair of entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                df1 = self.dataframes[entity1.source_file]
                df2 = self.dataframes[entity2.source_file]

                # Find best matching columns
                best_match = self._find_best_column_match(
                    df1, df2, entity1, entity2
                )

                if best_match:
                    col1, col2, method, confidence, match_count, samples = best_match

                    # Generate natural description
                    description = self._generate_natural_description(
                        entity1.display_name, entity2.display_name,
                        col1, col2, method, match_count
                    )

                    suggestion = RelationshipSuggestion(
                        id=f"rel_{entity1.id}_{entity2.id}",
                        start_entity_id=entity1.id,
                        end_entity_id=entity2.id,
                        start_entity_name=entity1.display_name,
                        end_entity_name=entity2.display_name,
                        start_column=col1,
                        end_column=col2,
                        discovery_method=method,
                        confidence=confidence,
                        natural_description=description,
                        matching_values_count=match_count,
                        sample_matches=samples
                    )
                    suggestions.append(suggestion)

        # Sort by confidence (highest first)
        suggestions.sort(key=lambda x: x.confidence, reverse=True)

        return suggestions

    def _find_best_column_match(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        entity1: EntitySuggestion,
        entity2: EntitySuggestion
    ) -> Optional[Tuple[str, str, str, float, int, List[str]]]:
        """
        Find the best matching columns between two DataFrames.

        Returns:
            Tuple of (col1, col2, method, confidence, match_count, sample_matches)
            or None if no good match found
        """
        best_match = None
        best_score = 0.0

        for col1 in df1.columns:
            for col2 in df2.columns:
                # Tier 1: Name heuristics (instant)
                name_match, name_score = self._tier1_name_heuristics(col1, col2)

                if name_match:
                    # Verify with value overlap
                    overlap, match_count, samples = self._tier2_value_overlap(
                        df1, col1, df2, col2
                    )
                    if overlap > 0:
                        # Boost score based on actual overlap
                        combined_score = name_score * 0.4 + min(overlap * 2, 0.6)
                        if combined_score > best_score:
                            best_score = combined_score
                            best_match = (col1, col2, "name_match", combined_score,
                                        match_count, samples)
                else:
                    # Tier 2: Value overlap only
                    overlap, match_count, samples = self._tier2_value_overlap(
                        df1, col1, df2, col2
                    )
                    if overlap >= self.min_overlap_threshold:
                        score = min(0.85, 0.3 + overlap)
                        if score > best_score:
                            best_score = score
                            best_match = (col1, col2, "value_overlap", score,
                                        match_count, samples)

        # Only return if we found a reasonable match
        if best_match and best_score >= 0.3:
            return best_match

        return None

    def _tier1_name_heuristics(
        self,
        col1: str,
        col2: str
    ) -> Tuple[bool, float]:
        """
        Check if columns match by name heuristics.

        Returns:
            Tuple of (is_match, confidence_score)
        """
        # Normalize column names
        n1 = self._normalize_column_name(col1)
        n2 = self._normalize_column_name(col2)

        # Exact match after normalization
        if n1 == n2:
            return True, 0.95

        # One contains the other (for things like "school_id" vs "id")
        if len(n1) > 3 and len(n2) > 3:
            if n1 in n2 or n2 in n1:
                return True, 0.8

        # Common ID pattern matching
        # e.g., "student_id" and "id_student" both have "student" + "id"
        for pattern in self.ID_PATTERNS:
            if pattern in n1 and pattern in n2:
                # Extract the non-ID part
                rest1 = n1.replace(pattern, '')
                rest2 = n2.replace(pattern, '')
                if rest1 == rest2 or not rest1 or not rest2:
                    return True, 0.85

        # Check for common suffix patterns
        for suffix in self.ID_SUFFIXES:
            if n1.endswith(suffix) and n2.endswith(suffix):
                prefix1 = n1[:-len(suffix)]
                prefix2 = n2[:-len(suffix)]
                if prefix1 == prefix2:
                    return True, 0.8

        # French-English translation matching
        t1 = self._translate_column_name(n1)
        t2 = self._translate_column_name(n2)
        if t1 == t2 and t1 != n1:
            return True, 0.75

        return False, 0.0

    def _tier2_value_overlap(
        self,
        df1: pd.DataFrame,
        col1: str,
        df2: pd.DataFrame,
        col2: str
    ) -> Tuple[float, int, List[str]]:
        """
        Compute value overlap (Jaccard similarity) on sampled values.

        Returns:
            Tuple of (jaccard_similarity, matching_count, sample_matches)
        """
        # Sample unique values
        vals1 = self._sample_unique_values(df1, col1)
        vals2 = self._sample_unique_values(df2, col2)

        if not vals1 or not vals2:
            return 0.0, 0, []

        # Compute intersection
        intersection = vals1 & vals2
        union = vals1 | vals2

        if not union:
            return 0.0, 0, []

        jaccard = len(intersection) / len(union)

        # Get sample matches (up to 5)
        sample_matches = list(intersection)[:5]

        return jaccard, len(intersection), sample_matches

    def _sample_unique_values(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Set[str]:
        """
        Sample unique values from a column for fast comparison.

        Uses stratified sampling (head, middle, tail) for better coverage.
        """
        try:
            unique_vals = df[column].dropna().unique()
        except Exception:
            return set()

        # Convert to strings for comparison
        if len(unique_vals) <= self.sample_size:
            return set(str(v).strip().lower() for v in unique_vals if str(v).strip())

        # Stratified sampling: head, middle, tail
        indices = np.linspace(0, len(unique_vals) - 1, self.sample_size, dtype=int)
        sampled = [unique_vals[i] for i in indices]

        return set(str(v).strip().lower() for v in sampled if str(v).strip())

    def _detect_key_column(
        self,
        df: pd.DataFrame
    ) -> Tuple[str, float, str]:
        """
        Detect the most likely key (identifier) column.

        Returns:
            Tuple of (column_name, confidence, reasoning)
        """
        if df.empty or len(df.columns) == 0:
            return df.columns[0] if len(df.columns) > 0 else "", 0.5, "Empty file"

        candidates = []

        for col in df.columns:
            col_lower = col.lower()
            uniqueness = df[col].nunique() / len(df) if len(df) > 0 else 0

            # Priority 1: Named as ID with high uniqueness
            is_id_name = any(
                pattern in col_lower for pattern in self.ID_PATTERNS
            ) or any(
                col_lower.endswith(suffix) for suffix in self.ID_SUFFIXES
            )

            if is_id_name and uniqueness > 0.5:
                candidates.append((col, 0.95, uniqueness,
                    f"'{col}' looks like an identifier with {uniqueness:.0%} unique values"))

            # Priority 2: High uniqueness (>90%)
            elif uniqueness > 0.9:
                candidates.append((col, 0.8, uniqueness,
                    f"'{col}' has {uniqueness:.0%} unique values"))

            # Priority 3: Named as ID but lower uniqueness
            elif is_id_name and uniqueness > 0.1:
                candidates.append((col, 0.7, uniqueness,
                    f"'{col}' is named like an identifier ({uniqueness:.0%} unique)"))

        if candidates:
            # Sort by confidence then uniqueness
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return candidates[0][0], candidates[0][1], candidates[0][3]

        # Fallback: first column
        first_col = df.columns[0]
        return first_col, 0.5, f"Using first column '{first_col}' as identifier"

    def _humanize_filename(self, filename: str) -> str:
        """
        Convert technical filename to friendly entity name.

        Examples:
            "fr-en-college-effectifs-niveau-sexe-lv.csv" -> "College Effectifs"
            "data_253ecdc0.csv" -> "Data"
        """
        # Remove extension
        name = filename.rsplit('.', 1)[0]

        # Remove common prefixes
        prefixes_to_remove = [
            'fr-en-', 'fr_en_', 'data_', 'export_', 'raw_',
            'dataset_', 'file_', 'table_'
        ]
        for prefix in prefixes_to_remove:
            if name.lower().startswith(prefix):
                name = name[len(prefix):]

        # Remove hash-like suffixes (e.g., _253ecdc0)
        name = re.sub(r'_[a-f0-9]{6,}$', '', name)

        # Split on common delimiters
        words = re.split(r'[-_\s]+', name)

        # Filter and capitalize meaningful words
        meaningful = []
        skip_words = {'fr', 'en', 'de', 'la', 'le', 'les', 'du', 'des', 'et'}

        for word in words:
            if len(word) > 2 and word.lower() not in skip_words:
                # Capitalize first letter, keep rest as-is for acronyms
                if word.isupper() and len(word) <= 5:
                    meaningful.append(word)  # Keep acronyms
                else:
                    meaningful.append(word.capitalize())

        # Take first 3 meaningful words
        if meaningful:
            return ' '.join(meaningful[:3])

        return "Data"

    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name for comparison."""
        return name.lower().replace('_', '').replace('-', '').replace(' ', '')

    def _translate_column_name(self, name: str) -> str:
        """Translate common French column names to English."""
        result = name
        for fr, en in self.TRANSLATIONS.items():
            result = result.replace(fr, en)
        return result

    def _tier3_semantic_match(
        self,
        df1: pd.DataFrame,
        col1: str,
        df2: pd.DataFrame,
        col2: str,
        sample_size: int = 500
    ) -> Tuple[float, int, List[Tuple[str, str, float]]]:
        """
        Find semantically similar values between two columns using embeddings.

        This is Tier 3 of the discovery system - used when columns don't share
        exact values but may have semantically related content.

        Uses HF Inference API for embeddings (no local model needed).

        Args:
            df1: First DataFrame
            col1: Column name in df1
            df2: Second DataFrame
            col2: Column name in df2
            sample_size: Maximum values to sample from each column

        Returns:
            Tuple of (average_similarity, match_count, matched_pairs)
            where matched_pairs is list of (val1, val2, similarity_score)
        """
        try:
            from intuitiveness.models import get_batch_similarities

            # Sample unique values from each column
            vals1 = df1[col1].dropna().astype(str).unique()
            vals2 = df2[col2].dropna().astype(str).unique()

            # Filter out empty/short values
            vals1 = [v.strip() for v in vals1 if len(str(v).strip()) > 1][:sample_size]
            vals2 = [v.strip() for v in vals2 if len(str(v).strip()) > 1][:sample_size]

            if not vals1 or not vals2:
                return 0.0, 0, []

            # Get pairwise similarities via HF Inference API
            similarities = get_batch_similarities(vals1, vals2)

            if similarities is None:
                return 0.0, 0, []

            # Find best matches above threshold
            matches = []
            for i, v1 in enumerate(vals1):
                best_j = int(similarities[i].argmax())
                best_score = float(similarities[i][best_j])
                if best_score >= self.semantic_threshold:
                    matches.append((v1, vals2[best_j], best_score))

            # Sort by similarity score (highest first)
            matches.sort(key=lambda x: x[2], reverse=True)

            avg_sim = float(np.mean([m[2] for m in matches])) if matches else 0.0
            return avg_sim, len(matches), matches[:10]

        except Exception as e:
            import streamlit as st
            st.warning(f"Semantic match error: {e}")
            return 0.0, 0, []

    def _generate_natural_description(
        self,
        entity1_name: str,
        entity2_name: str,
        col1: str,
        col2: str,
        method: str,
        match_count: int
    ) -> str:
        """
        Generate a natural language description of the relationship.
        """
        if method == "name_match":
            if col1.lower() == col2.lower():
                return (f"{entity1_name} connects to {entity2_name} "
                       f"through matching '{col1}' values ({match_count:,} matches)")
            else:
                return (f"{entity1_name} connects to {entity2_name} "
                       f"through '{col1}' and '{col2}' ({match_count:,} matches)")

        elif method == "value_overlap":
            return (f"{entity1_name} may relate to {entity2_name} "
                   f"(similar values in '{col1}' and '{col2}', {match_count:,} matches)")

        elif method == "semantic":
            return (f"{entity1_name} appears related to {entity2_name} "
                   f"based on content similarity ({match_count:,} potential matches)")

        return f"{entity1_name} connects to {entity2_name}"


def run_discovery(dataframes: Dict[str, pd.DataFrame]) -> DiscoveryResult:
    """
    Convenience function to run discovery on uploaded files.

    Args:
        dataframes: Dict mapping filename to DataFrame

    Returns:
        DiscoveryResult with entity and relationship suggestions
    """
    discovery = RelationshipDiscovery(dataframes)
    return discovery.discover_all()


def run_semantic_match(
    df1: pd.DataFrame,
    col1: str,
    df2: pd.DataFrame,
    col2: str,
    threshold: float = 0.7,
    sample_size: int = 500
) -> Tuple[float, int, List[Tuple[str, str, float]]]:
    """
    Convenience function to run semantic matching between two columns.

    Use this when columns don't share exact values but may have
    semantically similar content.

    Args:
        df1: First DataFrame
        col1: Column name in df1
        df2: Second DataFrame
        col2: Column name in df2
        threshold: Minimum similarity score (0.0-1.0)
        sample_size: Max values to sample from each column

    Returns:
        Tuple of (average_similarity, match_count, matched_pairs)
        where matched_pairs is list of (val1, val2, similarity_score)

    Example:
        >>> avg_sim, count, matches = run_semantic_match(
        ...     schools_df, 'school_name',
        ...     students_df, 'institution',
        ...     threshold=0.7
        ... )
        >>> for val1, val2, score in matches[:5]:
        ...     print(f"'{val1}' ↔ '{val2}' ({score:.0%} similar)")
    """
    discovery = RelationshipDiscovery(
        dataframes={},  # Empty, we'll call the method directly
        semantic_threshold=threshold,
        sample_size=sample_size
    )
    return discovery._tier3_semantic_match(df1, col1, df2, col2, sample_size)
