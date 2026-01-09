# Quickstart: Ascent Phase Precision

**Feature**: 004-ascent-precision
**Date**: 2025-12-04

## Overview

This feature enables precise ascent operations between abstraction levels:
- **L0→L1**: Unfold a datum to reveal its source vector
- **L1→L2**: Enrich a vector with domain categorization
- **L2→L3**: Build a graph by defining extra-dimensional entities

## Integration Scenarios

### Scenario 1: Complete Round-Trip (L3 → L0 → L3)

**Purpose**: Verify the full descent-ascent cycle preserves data integrity.

```
1. Start at L3 with a knowledge graph
2. Descend to L2 (extract domain table)
3. Descend to L1 (select column as vector)
4. Descend to L0 (aggregate to single value)
5. Ascend to L1 (unfold - should restore vector)
6. Ascend to L2 (add domain categorization)
7. Ascend to L3 (define entity and relationships)
```

**Expected Outcome**: User can navigate the full cycle. The L1 vector after unfold matches the original L1 from descent. Final L3 graph has no orphan nodes.

---

### Scenario 2: L0→L1 Unfold Operation

**Purpose**: Verify deterministic unfold restores parent vector.

**Setup**:
1. Load a dataset at L3
2. Navigate: L3 → L2 → L1 → L0 (using aggregation like "median")

**Test Steps**:
1. At L0, click "Ascend to L1"
2. System shows unfold confirmation with:
   - Aggregation method used (e.g., "median")
   - Preview of source vector
3. Confirm unfold
4. Verify the restored L1 vector matches the original

**Acceptance**:
- [x] Original aggregation method is displayed
- [x] Source vector preview is shown
- [x] Restored vector has same values and column name
- [x] Operation completes within 2 seconds (SC-001)

---

### Scenario 3: L0→L1 Blocked for Orphan Datum

**Purpose**: Verify system blocks unfold when no parent exists.

**Setup**:
1. Create or load a Level0Dataset directly (not from aggregation)

**Test Steps**:
1. At L0, click "Ascend to L1"
2. System should display blocking message

**Acceptance**:
- [x] Clear error message explains why ascent is blocked
- [x] Suggests alternative actions or explains missing parent data

---

### Scenario 4: L1→L2 Domain Enrichment

**Purpose**: Verify domain categorization creates proper 2D table.

**Setup**:
1. Navigate to L1 with a vector of mixed values (e.g., product names)

**Test Steps**:
1. At L1, click "Ascend to L2"
2. Enter domains: "Electronics, Clothing, Food"
3. Toggle semantic matching ON
4. Set threshold to 0.5
5. Click "Apply Domain Enrichment"
6. Verify resulting 2D table

**Acceptance**:
- [x] Domain input field accepts comma-separated values
- [x] Semantic matching toggle works
- [x] Threshold slider ranges from 0.1 to 0.9
- [x] Resulting table has original values + "domain" column
- [x] Values not matching any domain are labeled "Unmatched"
- [x] Operation completes within 30 seconds (SC-002)
- [x] Same UI/behavior as L3→L2 descent categorization (SC-005)

---

### Scenario 5: L1→L2 with Keyword Matching

**Purpose**: Verify keyword-based categorization alternative.

**Setup**:
1. Navigate to L1 with a vector containing clear keywords

**Test Steps**:
1. At L1, click "Ascend to L2"
2. Enter domains: "Sales, Marketing, Engineering"
3. Toggle semantic matching OFF (keyword mode)
4. Click "Apply Domain Enrichment"

**Acceptance**:
- [x] Keyword matching uses simple text containment
- [x] Threshold slider is disabled in keyword mode
- [x] Results match expected keyword categorization

---

### Scenario 6: L2→L3 Graph Building

**Purpose**: Verify graph creation from table with new entity type.

**Setup**:
1. Navigate to L2 with a table containing a categorical column (e.g., "department")

**Test Steps**:
1. At L2, click "Ascend to L3"
2. Select entity column: "department"
3. Enter entity type name: "Department"
4. Enter relationship type: "BELONGS_TO"
5. Click "Build Graph"
6. Verify resulting graph

**Acceptance**:
- [x] Column selector shows available categorical columns
- [x] Entity type name input is required
- [x] Relationship type input is required
- [x] Resulting graph has nodes for each unique department value
- [x] Each original row connects to its department node
- [x] No orphan nodes exist (SC-006)
- [x] Operation completes within 60 seconds (SC-003)

---

### Scenario 7: L2→L3 Single-Value Column Warning

**Purpose**: Verify system warns about low-cardinality entity columns.

**Setup**:
1. Navigate to L2 with a table where one column has only 1 unique value

**Test Steps**:
1. At L2, click "Ascend to L3"
2. Select the single-value column
3. System should show warning

**Acceptance**:
- [x] Warning indicates only 1 node will be created
- [x] User can still proceed if desired
- [x] Resulting graph has the single entity node connected to all rows

---

## UI Component Verification

### Ascent Forms Module (`ui/ascent_forms.py`)

| Form | Inputs | Validation |
|------|--------|------------|
| L0→L1 Unfold | Confirmation button | Has parent_data |
| L1→L2 Domain | Domains, method toggle, threshold | At least 1 domain |
| L2→L3 Entity | Column selector, entity name, rel type | All fields required |

### Reused Components

| Component | Original Location | Reused In |
|-----------|-------------------|-----------|
| Domain input field | L3→L2 descent | L1→L2 ascent |
| Semantic toggle | L3→L2 descent | L1→L2 ascent |
| Threshold slider | L3→L2 descent | L1→L2 ascent |

---

## Performance Benchmarks

| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| L0→L1 Unfold | < 2 seconds | Time from click to L1 display |
| L1→L2 Domain | < 30 seconds | Time from form submit to L2 display |
| L2→L3 Graph | < 60 seconds | Time from form submit to L3 display |

---

## Error Cases to Test

1. **L0 without parent_data**: Should block with clear message
2. **L1→L2 empty domains**: Should show validation error
3. **L1→L2 invalid threshold**: Should clamp to valid range
4. **L2→L3 missing column**: Should show validation error
5. **L2→L3 empty entity name**: Should show validation error
6. **L2→L3 orphan nodes**: Should prevent or warn
