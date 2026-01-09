# Research: Ascent Phase Precision

**Feature**: 004-ascent-precision
**Date**: 2025-12-04

## Research Summary

This document consolidates findings from exploring the existing codebase to understand current ascent implementations and gaps.

## Decision 1: L0→L1 Unfold Implementation

**Decision**: Reuse existing `_increase_0_to_1()` with `source_expansion` enrichment function.

**Rationale**:
- Level0Dataset already stores `parent_data` (the source vector) when created via L1→L0 descent
- The `source_expansion` enrichment function in `enrichment.py` restores this parent data
- This is deterministic - no user input needed for unfold operation

**Alternatives Considered**:
- Creating new unfold-specific function → Rejected: `source_expansion` already does exactly this
- Storing parent data differently → Rejected: Current `parent_data` attribute is well-designed

**Key Code Locations**:
- `complexity.py:98-130` - Level0Dataset with `parent_data`, `has_parent`, `get_parent_data()`
- `ascent/enrichment.py` - EnrichmentRegistry and `source_expansion` function
- `redesign.py:160-244` - `_increase_0_to_1()` method

## Decision 2: L1→L2 Domain Categorization

**Decision**: Reuse the domain categorization pattern from L3→L2 descent (same UI, same classifiers).

**Rationale**:
- FR-009 explicitly requires reusing L3→L2 domain categorization logic
- The L3→L2 descent in `streamlit_app.py` already has:
  - Domain input (comma-separated)
  - Semantic matching toggle
  - Similarity threshold slider
  - "Unmatched" fallback domain
- DimensionRegistry in `dimensions.py` supports L1→L2 transitions

**Alternatives Considered**:
- Creating separate L1→L2 categorization UI → Rejected: Violates FR-009 (must reuse)
- Using only built-in classifiers → Rejected: User needs to specify custom domains

**Key Code Locations**:
- `streamlit_app.py:2400-2572` - L3→L2 domain categorization UI (to be refactored into shared component)
- `ascent/dimensions.py` - DimensionDefinition, DimensionRegistry
- `ascent/dimensions.py:181-220` - Built-in classifiers (`_classify_business_object`, `_classify_pattern`)

## Decision 3: L2→L3 Graph Building

**Decision**: Create UI for users to select entity column and define relationship type.

**Rationale**:
- L2→L3 ascent requires user input to define the "extra dimension" entity
- The entity column's unique values become new nodes
- Relationships connect existing table rows to these new entity nodes
- Must ensure no orphan nodes (Design Principle #1)

**Alternatives Considered**:
- Auto-detect entity columns → Rejected: User must explicitly choose the dimension to add
- Use existing RelationshipDefinition drag-drop UI → Partially accepted: Can reuse patterns but need simpler flow

**Key Code Locations**:
- `ascent/dimensions.py:342-412` - `create_graph_from_relationships()`, `create_dimension_groups()`
- `ascent/dimensions.py:581-634` - RelationshipDefinition dataclass
- `redesign.py:297-373` - `_increase_2_to_3()` method

## Decision 4: UI Component Architecture

**Decision**: Create `ui/ascent_forms.py` module with reusable ascent form components.

**Rationale**:
- Ascent forms need to be used in both Guided Mode and Free Navigation Mode
- Current `render_ascend_options()` is tightly coupled to Free Navigation
- Separating into reusable components enables consistency (FR-009, SC-005)

**Components to Create**:
1. `render_l0_to_l1_unfold_form()` - Simple confirmation (deterministic)
2. `render_l1_to_l2_domain_form()` - Domain input, semantic toggle, threshold
3. `render_l2_to_l3_entity_form()` - Entity column selection, relationship type

## Existing Infrastructure Summary

| Component | Status | Location |
|-----------|--------|----------|
| Level0Dataset.parent_data | ✅ Exists | complexity.py:98-130 |
| Level0Dataset.has_parent | ✅ Exists | complexity.py:117-118 |
| Level0Dataset.get_parent_data() | ✅ Exists | complexity.py:120-122 |
| EnrichmentRegistry | ✅ Exists | ascent/enrichment.py |
| source_expansion enrichment | ✅ Exists | ascent/enrichment.py |
| DimensionRegistry | ✅ Exists | ascent/dimensions.py |
| DimensionDefinition | ✅ Exists | ascent/dimensions.py |
| _increase_0_to_1() | ✅ Exists | redesign.py:160-244 |
| _increase_1_to_2() | ✅ Exists | redesign.py:246-295 |
| _increase_2_to_3() | ✅ Exists | redesign.py:297-373 |
| Domain categorization UI | ✅ Exists (L3→L2) | streamlit_app.py:2400-2572 |
| Ascent forms module | ❌ Missing | ui/ascent_forms.py (NEW) |

## Gaps to Address

1. **L0→L1 UI Enhancement**: Current UI doesn't clearly show "unfold" as restoring the source distribution
2. **L1→L2 Ascent UI**: Need to add domain categorization to L1→L2 ascent (currently minimal)
3. **L2→L3 Ascent UI**: Need entity column selection and relationship definition UI
4. **Shared Components**: Extract domain categorization into reusable component for both descent and ascent
5. **Orphan Node Prevention**: Add validation to ensure L2→L3 doesn't create disconnected nodes
