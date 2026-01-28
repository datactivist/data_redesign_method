"""
Ascent UI Forms Package

Implements Spec 011: Code Simplification
Extracted from ui/ascent_forms.py (1,755 → 3 modules)

This package contains focused form modules for ascent operations:
- l0_to_l1_form: Unfold datum to source vector (deterministic)
- l1_to_l2_form: Domain enrichment with categorization
- l2_to_l3_form: Graph building with entity extraction
- shared: Common utilities for all ascent forms

Each module is self-contained with <400 lines of focused logic.
"""

from intuitiveness.ui.ascent.l0_to_l1_form import render_l0_to_l1_unfold_form
from intuitiveness.ui.ascent.l1_to_l2_form import render_l1_to_l2_domain_form
from intuitiveness.ui.ascent.l2_to_l3_form import render_l2_to_l3_entity_form
from intuitiveness.ui.ascent.shared import (
    SESSION_KEY_L0_TO_L1_FORM,
    SESSION_KEY_L1_TO_L2_FORM,
    SESSION_KEY_L2_TO_L3_FORM,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_DOMAINS,
    L0ToL1FormState,
    L1ToL2FormState,
    L2ToL3FormState,
)

__all__ = [
    "render_l0_to_l1_unfold_form",
    "render_l1_to_l2_domain_form",
    "render_l2_to_l3_entity_form",
    "SESSION_KEY_L0_TO_L1_FORM",
    "SESSION_KEY_L1_TO_L2_FORM",
    "SESSION_KEY_L2_TO_L3_FORM",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_DOMAINS",
    "L0ToL1FormState",
    "L1ToL2FormState",
    "L2ToL3FormState",
]
