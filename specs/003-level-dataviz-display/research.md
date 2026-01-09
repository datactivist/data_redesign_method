# Research: Level-Specific Data Visualization Display

**Feature**: 003-level-dataviz-display
**Date**: 2025-12-04
**Status**: Complete
**Constitution**: v1.2.0 (Target User Assumption integrated)

## Research Questions

### Q1: What visualization components already exist?

**Decision**: Leverage existing implementation from 002-ascent-functionality

**Rationale**: The L3→L2 entity/relationship tabbed display was already implemented in `render_domains_step()` of `streamlit_app.py`. This implementation:
- Extracts items by type from connected information
- Extracts connections by type from links
- Displays as tabs using `st.tabs()`
- Shows item tables with id, name, type, and properties
- Shows connection tables with start_name, connection, end_name

**Alternatives Considered**:
- Building from scratch: Rejected - existing implementation already meets FR-004, FR-005, FR-007
- Using external visualization library: Rejected - st.tabs() provides adequate tabbed interface

### Q2: How should ascent visualizations differ from descent?

**Decision**: Ascent shows LOWER level data (source); Descent shows HIGHER level data (target)

**Rationale**: Per FR-012, during ascent the user needs to see what they're transforming FROM (the source level). This provides context for the enrichment operation.

| Operation | User sees | Purpose (Domain Language) |
|-----------|-----------|---------------------------|
| Descent L4→L3 | Raw files (L4) | "Here are your uploaded files" |
| Descent L3→L2 | Connected info + category tabs (L3) | "Browse your information by category" |
| Descent L2→L1 | Categorized items (L2) | "Items grouped by category" |
| Descent L1→L0 | List of values (L1) | "Your selected values" |
| Ascent L0→L1 | Single result (L0) | "The value you're expanding" |
| Ascent L1→L2 | List of values (L1) | "Values being enriched" |
| Ascent L2→L3 | Categorized items (L2) | "Items becoming connected" |

**Alternatives Considered**:
- Show target level preview: Rejected - users need source context for ascent decisions
- Show both levels: Rejected - would clutter UI

### Q3: How to ensure mode consistency (FR-014)?

**Decision**: Extract display logic into reusable components callable from both modes

**Rationale**: Creating shared display functions ensures Guided Mode and Free Navigation Mode render identical visualizations. The display logic should be:
1. Independent of navigation mode
2. Parameterized by level and data
3. Callable from any render function

**Alternatives Considered**:
- Duplicate code in each mode: Rejected - violates DRY, risks inconsistency
- Mode-specific styling: Rejected - visual consistency is a requirement (SC-006)

### Q4: Performance considerations for large datasets?

**Decision**: Implement pagination/truncation for displays >50 rows

**Rationale**: SC-004 requires category tabs to load <2 seconds for 5,000 items. Rendering all rows in large displays would degrade performance.

**Implementation**:
- Use `st.dataframe()` with `use_container_width=True`
- Display first 50-100 rows by default
- Add "Show all" expander for full data
- Show count summaries (e.g., "5,518 items")

### Q5: How to comply with constitution v1.2.0 Target User Assumption?

**Decision**: Use domain-native language throughout UI, never expose technical data structure terms

**Rationale**: Per constitution v1.2.0, users have NO familiarity with data structures. They are domain curious minds who approach data with domain questions, not technical queries.

**Implementation**:
1. **Avoid technical terms**: "graph," "table," "vector," "datum," "entity," "node," "edge," "schema"
2. **Use domain alternatives**:
   - Graph → "Connected information" or "How your data connects"
   - Table → "Organized information" or "Items by category"
   - Vector → "List of values" or "Your selected values"
   - Datum → "Your result" or "Computed answer"
   - Entity → "Item" or "Thing" (context-dependent)
   - Relationship → "Connection" or "Link"
3. **Frame actions as questions**:
   - Instead of "Define domains" → "How would you like to categorize?"
   - Instead of "Extract vector" → "Which values interest you?"
   - Instead of "Compute metric" → "What answer are you looking for?"

**Alternatives Considered**:
- Glossary/tooltip approach: Rejected - still requires users to learn terminology
- Progressive disclosure: Rejected - adds friction; users shouldn't need to "level up"

## Findings Summary

| Topic | Finding |
|-------|---------|
| Existing implementation | L3→L2 tabs already implemented - refactor for reuse |
| Ascent visualization | Show source level, not target level |
| Mode consistency | Extract shared display components |
| Performance | Paginate large displays, show counts |
| Constitution v1.2.0 | Use domain language, never technical terms |

## Impact on Design

1. **data-model.md**: Define LevelDisplay configuration with domain-friendly labels
2. **contracts/display_api.py**: Define display function interfaces with domain terminology
3. **quickstart.md**: Visual verification scenarios using domain language
4. **UI Components**: All labels must pass "domain curious mind" test - would a non-technical domain expert understand this?
