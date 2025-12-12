"""Descent operations module for L4→L3→L2→L1→L0 complexity reduction."""

from .semantic_join import semantic_table_join, SemanticJoinConfig

__all__ = ['semantic_table_join', 'SemanticJoinConfig']
