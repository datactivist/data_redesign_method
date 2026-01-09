# Feature Specification: Level-Specific Data Visualization Display

**Feature Branch**: `003-level-dataviz-display`
**Created**: 2025-12-04
**Status**: Draft
**Input**: User description: "Define data visualization displays at each navigation level for both Guided and Free Navigation modes, showing appropriate visualizations during descent and ascent operations."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Descent Navigation Visualization (Priority: P1)

When users navigate DOWN through abstraction levels (L4→L3→L2→L1→L0), they need to see appropriate visualizations at each step that help them understand the data transformation happening.

**Why this priority**: This is the primary navigation flow. Users must understand what data they're working with at each level to make informed decisions during descent.

**Independent Test**: Can be tested by navigating from L4 to L0 in Guided Mode and verifying each step shows the correct visualization.

**Acceptance Scenarios**:

1. **Given** user is at L4 (Raw Data), **When** they view the L4 step, **Then** they see their uploaded raw dataset files displayed as a list with file names and basic metadata (row count, column count)
2. **Given** user is at L3→L2 transition, **When** they view this step, **Then** they see:
   - The knowledge graph visualization showing entities and relationships
   - Tabbed tables showing: one tab per entity type with all entities of that type, plus one tab per relationship type showing linked entity pairs
3. **Given** user is at L2→L1 transition, **When** they view this step, **Then** they see the domain-categorized table showing items classified by domain labels
4. **Given** user is at L1→L0 transition, **When** they view this step, **Then** they see the vector (list of values) extracted from the selected column

---

### User Story 2 - Ascent Navigation Visualization (Priority: P2)

When users navigate UP through abstraction levels (L0→L1→L2→L3), they need to see the visualization from the LOWER level they came from, helping them understand the starting point of the transformation.

**Why this priority**: Ascent is the reverse operation that completes the navigation cycle. Showing the lower level's visualization provides context for the enrichment/expansion operation.

**Independent Test**: Can be tested by ascending from L0 to L3 and verifying each step shows the previous level's visualization.

**Acceptance Scenarios**:

1. **Given** user is ascending from L0→L1, **When** they view this step, **Then** they see the L0 datum (single value) they are expanding
2. **Given** user is ascending from L1→L2, **When** they view this step, **Then** they see the L1 vector they are enriching with dimensions
3. **Given** user is ascending from L2→L3, **When** they view this step, **Then** they see the L2 domain table they are transforming into a graph

---

### User Story 3 - Free Navigation Mode Visualization (Priority: P2)

Users in Free Navigation mode need the same visualization consistency as Guided Mode, allowing them to see context-appropriate displays regardless of their navigation direction.

**Why this priority**: Free Navigation is an alternative mode that should maintain visual consistency with Guided Mode for a unified user experience.

**Independent Test**: Can be tested by switching to Free Navigation mode and performing both descent and ascent operations.

**Acceptance Scenarios**:

1. **Given** user is in Free Navigation mode at any level, **When** they choose to descend, **Then** they see the same visualization as Guided Mode descent for that transition
2. **Given** user is in Free Navigation mode at any level, **When** they choose to ascend, **Then** they see the same visualization as Guided Mode ascent for that transition (lower level's visualization)

---

### Edge Cases

- What happens when graph has no relationships? Display only entity tabs, no relationship tabs
- What happens when a domain has no matching items? Display empty state with "No items matched this domain" message
- What happens when raw data upload fails? Display error message and allow retry
- How does system handle very large datasets (>10,000 rows)? Display first 50-100 rows with pagination or "Load more" option

## Requirements *(mandatory)*

### Functional Requirements

#### L4 Display (Raw Data)
- **FR-001**: System MUST display uploaded raw dataset files as a list showing file name, row count, and column count
- **FR-002**: System MUST allow users to preview each raw file's first few rows

#### L3→L2 Display (Graph to Table Transition)
- **FR-003**: System MUST display the knowledge graph as an interactive visualization showing entities and relationships
- **FR-004**: System MUST display tabbed views with one tab per entity type, showing all entities of that type in a table
- **FR-005**: System MUST display one tab per relationship type, showing pairs of linked entities with their relationship type
- **FR-006**: System MUST show entity tables side-by-side or adjacent to the graph visualization
- **FR-007**: Each entity tab MUST show columns: id, name, type, and any additional properties

#### L2→L1 Display (Table to Vector Transition)
- **FR-008**: System MUST display the domain-categorized table showing which domain each item was classified into
- **FR-009**: System MUST show domain labels clearly for each item in the table

#### L1→L0 Display (Vector to Datum Transition)
- **FR-010**: System MUST display the vector as a list or series of values
- **FR-011**: System MUST show the column name from which the vector was extracted

#### Ascent Visualization Rule
- **FR-012**: During ascent operations, system MUST display the visualization from the LOWER level (source level) instead of the target level
- **FR-013**: System MUST clearly indicate the direction of navigation (ascending vs descending) to users

#### Mode Consistency
- **FR-014**: Guided Mode and Free Navigation Mode MUST display identical visualizations for the same level transitions
- **FR-015**: System MUST maintain visualization consistency when users switch between navigation modes

### Key Entities

- **LevelDisplay**: Represents the visualization configuration for a specific abstraction level (L0-L4)
- **NavigationDirection**: Indicates whether user is ascending (L0→L3) or descending (L4→L0)
- **EntityTab**: A tabbed view showing all items of a single entity type from the graph
- **RelationshipTab**: A tabbed view showing linked entity pairs for a specific relationship type

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can identify which abstraction level they are viewing within 3 seconds of page load
- **SC-002**: 100% of descent transitions show the higher level's data being transformed
- **SC-003**: 100% of ascent transitions show the lower level's data being enriched
- **SC-004**: Entity/relationship tabs load within 2 seconds for graphs with up to 5,000 nodes
- **SC-005**: Users report understanding "what data they're working with" in 90% of usability tests
- **SC-006**: Visual consistency between Guided and Free Navigation modes verified at all level transitions

## Assumptions

- **Target User Assumption (Constitution v1.2.0)**: Users have NO familiarity with data structures. They are domain curious minds—experts in their own field who want to explore data relevant to their domain. The system MUST shield users from technical complexity; they interact with domain concepts and natural language, never requiring understanding of tables, graphs, vectors, or schemas.
- The existing Level0-Level4 Dataset classes provide appropriate `get_data()` methods (internal implementation)
- Visualization components already exist for connected information display
- Users do NOT need to understand abstraction levels—the UI presents data in domain-native language without exposing L0-L4 terminology
- Maximum expected dataset size is 10,000 items for typical use cases
