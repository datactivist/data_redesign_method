# Feature Specification: Ascent Functionality (Reverse Navigation)

**Feature Branch**: `002-ascent-functionality`
**Created**: 2025-12-02
**Status**: Draft
**Input**: User description: "I didn't develop yet the level up from L0 to L1, L1 to L2, and L2 to L3. Propose me some reverse functionalities"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Enrich Datum to Vector (L0 → L1) (Priority: P1)

A data analyst has computed an atomic metric (L0) such as "count of revenue indicators = 523" and now wants to understand the structural identity of each item that contributed to this metric. They need to "re-expand" the datum into a vector of related data points while maintaining the constraint established at L0.

**Why this priority**: This is the foundational ascent operation. Without L0→L1, no further ascent is possible. It enables the "reconstruct from ground truth" workflow described in the research paper.

**Independent Test**: Can be fully tested by computing any L0 metric, then ascending to see a vector of values that, when aggregated, would produce the original L0 value.

**Acceptance Scenarios**:

1. **Given** a Level 0 datum (e.g., sum=523), **When** user requests ascent to L1 with a feature extraction function, **Then** system produces a Level 1 vector where the aggregation of that vector equals the original L0 value.
2. **Given** a Level 0 datum with description "count of indicators", **When** user ascends with naming signature extraction, **Then** system produces a vector of naming features (first word, word count, character patterns) for each contributing item.
3. **Given** a Level 0 datum, **When** user attempts ascent without providing an enrichment function, **Then** system displays available enrichment options or prompts user to define one.

---

### User Story 2 - Add Dimensions to Create Table (L1 → L2) (Priority: P2)

A data analyst has a vector of values (L1) such as indicator names or feature signatures and wants to categorize them by adding dimensions to create a table. They need to transform a one-dimensional list into a multi-dimensional table by adding category columns.

**Why this priority**: This enables classification and categorization, which is essential for organizing data into meaningful groups before further analysis.

**Independent Test**: Can be fully tested by taking any L1 vector and adding at least one categorical dimension to produce an L2 table.

**Acceptance Scenarios**:

1. **Given** a Level 1 vector of indicator names, **When** user adds a "business_object" dimension (revenue, volume, ETP), **Then** system produces a Level 2 table with each indicator classified by business object.
2. **Given** a Level 1 vector, **When** user adds multiple dimensions (calculated flag, weight flag, RSE flag), **Then** system produces a Level 2 table with all specified dimension columns populated.
3. **Given** a Level 1 vector, **When** user provides a classification function for a dimension, **Then** system applies that function to categorize each vector element.
4. **Given** a Level 1 vector, **When** user requests auto-classification, **Then** system suggests possible dimension categories based on data patterns.

---

### User Story 3 - Group into Hierarchical Relationships (L2 → L3) (Priority: P3)

A data analyst has a categorized table (L2) and wants to add hierarchical/analytic dimensions to create a multi-level structure (L3). They need to enrich the table with additional grouping dimensions that enable cross-cutting analysis.

**Why this priority**: This is the final ascent step that creates the most intuitive, multi-dimensional view of the data for business users.

**Independent Test**: Can be fully tested by taking any L2 table with basic categories and adding analytic dimensions to produce an L3 multi-level structure.

**Acceptance Scenarios**:

1. **Given** a Level 2 table with business object categories, **When** user adds analytic dimensions (client segmentation, sales location, product segmentation), **Then** system produces a Level 3 structure with hierarchical groupings.
2. **Given** a Level 2 table, **When** user uses the drag-and-drop interface to draw connections between entities, **Then** system creates linkable multi-level tables with the specified relationships.
3. **Given** a Level 2 table, **When** user adds "financial_view" and "lifecycle_view" dimensions, **Then** system enables filtering and grouping by these analytic perspectives.
4. **Given** a Level 3 structure, **When** user queries for items sharing identical analytic dimensions, **Then** system returns clusters of potential duplicates.

---

### User Story 4 - Interactive Decision-Tree Navigation (Priority: P2)

A user navigating in "Free Navigation Mode" sees a persistent decision-tree visualization showing their current position and all navigation options (ascend/descend/exit). At every node, the user can see exactly where they are in the data hierarchy and what options are available.

**Why this priority**: Provides a consistent user experience between descent and ascent, making the system more intuitive.

**Independent Test**: Can be tested by navigating to any level (L0, L1, L2, L3) and verifying the decision-tree displays correct options.

**Acceptance Scenarios**:

1. **Given** a user at Level 3 (graph), **When** viewing the decision-tree, **Then** system displays options: "Exit (with graph + path)" or "Descend to L2".
2. **Given** a user at Level 2 (domain table), **When** viewing the decision-tree, **Then** system displays options: "Exit (with table + path)", "Descend to L1", or "Ascend to L3 (specify relationships)".
3. **Given** a user at Level 1 (vector), **When** viewing the decision-tree, **Then** system displays options: "Exit (with vector + path)", "Descend to L0", or "Ascend to L2 (add domain)".
4. **Given** a user at Level 0 (datum), **When** viewing the decision-tree, **Then** system displays options: "Exit (with datum + path)" or "Ascend to L1 (unfold datum)".
5. **Given** a user at any level who selects "Exit", **Then** system exports both the current output (graph/table/vector/datum) AND the decision-tree path taken.

---

### Edge Cases

- What happens when user tries to ascend from L3 to L4? System must block this and explain L4 is entry-only.
- What happens when the enrichment function produces no data? System must handle gracefully with informative message.
- What happens when dimension values cannot be determined for some items? System must allow partial classification with "Unknown" category.
- How does system handle very large vectors (1000+ items) during enrichment? System must provide progress feedback.

---

## Navigation Rules *(mandatory)*

### Complete Navigation Flow

Users always start from Level 4 (entry point) and descend to Level 3. From there, the following rules govern all navigation:

```
L4 (Entry) → L3 (Graph)
     ↓
L3: Exit OR Descend to L2
     ↓
L2: Exit OR Descend to L1 OR Ascend to L3
     ↓
L1: Exit OR Descend to L0 OR Ascend to L2
     ↓
L0: Exit OR Ascend to L1
```

### Level-Specific Navigation Options

| Level | Exit Output | Descend To | Ascend To | Ascent Action |
|-------|-------------|------------|-----------|---------------|
| L3 (Graph) | Graph + NavigationTree | L2 | BLOCKED (L4 is entry-only) | N/A |
| L2 (Domain Table) | Graph + Domain-labeled Table + NavigationTree | L1 | L3 | Add extra table entity from available columns in raw data, then recreate graph |
| L1 (Vector) | Graph + Domain-labeled Table + Vector + NavigationTree | L0 | L2 | Add domain label to vector |
| L0 (Datum) | Graph + Domain-labeled Table + Vector + Datum + NavigationTree | N/A | L1 | Unfold datum to vector |

### Cumulative Output on Exit

On exit at any level, the system exports **all outputs accumulated during the session**:

- **Exit at L3**: Graph + NavigationTree
- **Exit at L2**: Graph + Domain-labeled Table + NavigationTree
- **Exit at L1**: Graph + Domain-labeled Table + Vector + NavigationTree
- **Exit at L0**: Graph + Domain-labeled Table + Vector + Datum + NavigationTree

### Infinite Exploration

The navigation process can continue indefinitely within these boundaries:
- No ascending to L4 (entry-only)
- Ascend/descend options available at L3, L2, L1
- Exit possible at every node with cumulative outputs and NavigationTree
- Ascend only available at L0 (no descend below L0)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support ascending from Level 0 (Datum) to Level 1 (Vector) by applying an enrichment function that expands a scalar into a series of related values.
- **FR-002**: System MUST support ascending from Level 1 (Vector) to Level 2 (Table) by adding one or more categorical dimensions to create columns.
- **FR-003**: System MUST support ascending from Level 2 (Table) to Level 3 (Linkable) by adding hierarchical/analytic dimensions that enable multi-level grouping.
- **FR-004**: System MUST prevent ascending from Level 3 to Level 4, displaying a clear message that L4 is entry-only.
- **FR-005**: System MUST preserve data integrity during ascent - the relationship between the lower level and higher level data must be traceable.
- **FR-006**: System MUST provide default enrichment options for common ascent patterns (e.g., naming signature extraction for L0→L1).
- **FR-007**: System MUST allow users to provide custom enrichment/classification functions for each ascent operation.
- **FR-008**: System MUST update the navigation history when ascent operations are performed.
- **FR-009**: System MUST display a persistent sidebar with the decision-tree structure in Free Navigation Mode, showing the user's current position, navigation path taken, and available options (ascend/descend/exit) at every level.
- **FR-010**: System MUST validate that enrichment functions produce valid output for the target level before completing the ascent.
- **FR-011**: At L3 (graph output): System MUST offer "Exit with graph and path" or "Descend to L2 (domain table)".
- **FR-012**: At L2 (domain table output): System MUST offer "Exit with table and path", "Descend to L1 (vector)", or "Ascend to L3 (specify relationships to recreate graph)".
- **FR-013**: At L1 (vector output): System MUST offer "Exit with vector and path", "Descend to L0 (datum)", or "Ascend to L2 (add domain dimensions)".
- **FR-014**: At L0 (datum output): System MUST offer "Exit with datum and path" or "Ascend to L1 (unfold datum to vector)".
- **FR-015**: On exit, system MUST export a JSON file containing the navigation path and current output, rendered with a JSON Crack-style interactive tree visualization.
- **FR-016**: For L2→L3 ascent, system MUST provide a visual drag-and-drop interface where users can draw connections between entities to define relationships.
- **FR-017**: Sidebar decision-tree nodes MUST be clickable, restoring the user to that previous state (time-travel navigation) while preserving the full navigation history.
- **FR-018**: When user time-travels back and takes a different path, system MUST preserve both branches, creating a true tree structure with multiple exploration paths.
- **FR-019**: On exit at any level, system MUST export all cumulative outputs accumulated during the session (not just the current level's output).
- **FR-020**: For L2→L3 ascent, system MUST allow user to select an extra table entity from available columns in the raw original data to recreate the graph structure.
- **FR-021**: NavigationTree MUST be displayed as a Directed Acyclic Graph (DAG) and record: (a) each navigation step taken, (b) the specific decision made at each step (entity selected, label added, operation performed), and (c) the generated output snapshot at every step.
- **FR-022**: Navigation process MUST support infinite exploration within the defined boundaries - users can ascend and descend repeatedly between L0-L3 without limit.

### Key Entities

- **EnrichmentFunction**: A callable that takes data from a lower level and produces enriched data for a higher level. For L0→L1, takes a scalar and returns a vector. For L1→L2, takes a vector and returns a DataFrame with added columns.
- **DimensionDefinition**: Specifies a categorical dimension to add during ascent, including name, possible values, and classification logic.
- **AscentOperation**: Records an ascent action including source level, target level, enrichment function used, and resulting data structure.
- **NavigationTree**: A visualization and tracking structure displayed as a **Directed Acyclic Graph (DAG)** that records: (1) each navigation step taken, (2) the decision made at each step (e.g., "make graph with entity X", "add label Y", "perform operation Z"), and (3) the generated output at every step. Supports multiple exploration branches when user time-travels and takes different paths; displayed in persistent sidebar as a DAG and exported on exit with all accumulated outputs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can ascend from any level (L0, L1, L2) to the next higher level (up to L3) in under 30 seconds when using default enrichment options.
- **SC-002**: 90% of users can successfully complete a full ascent cycle (L0→L1→L2→L3) on their first attempt using the guided workflow.
- **SC-003**: Ascent operations preserve data consistency - aggregating an L1 vector produced from L0→L1 ascent returns the original L0 value.
- **SC-004**: System provides at least 2 default enrichment options for each ascent transition (L0→L1, L1→L2, L2→L3).
- **SC-005**: Users can identify duplicate indicators using L2→L3 ascent by finding items with identical analytic dimensions within 5 minutes for datasets up to 10,000 items.

## Clarifications

### Session 2025-12-03

- Q: How should Free Navigation Mode display navigation options at each level? → A: Display a decision-tree structure showing current position with ascend/descend/exit options at every node. At L3 (graph): exit or descend to L2. At L2 (domain table): exit, descend to L1, or ascend to L3. At L1 (vector): exit, descend to L0, or ascend to L2. At L0 (datum): exit or ascend to L1. Exit always includes the decision-tree path and current output.
- Q: How should the decision-tree visualization be rendered in the UI? → A: Persistent sidebar showing the full navigation tree.
- Q: When user exits, what format for export? → A: JSON export with JSON Crack-style interactive visualization rendering.
- Q: For L2→L3 ascent, how should users define relationships? → A: Visual drag-and-drop interface to draw connections between entities.
- Q: Should sidebar decision-tree nodes be interactive? → A: Yes, clicking a node restores that previous state (time-travel navigation).
- Q: When user time-travels back and takes a different path, what happens to the original branch? → A: Preserve both branches (tree structure with multiple paths).

### Session 2025-12-04

- Q: What is the complete navigation flow? → A: Users always start from L4, descend to L3. From L3: exit or descend to L2. From L2: exit, descend to L1, or ascend to L3 (add extra table entity from raw data columns, recreate graph). From L1: exit, descend to L0, or ascend to L2 (add domain to vector). From L0: exit or ascend to L1. Process is infinite within these boundaries.
- Q: What outputs are exported on exit? → A: Cumulative outputs from all levels visited. Exit at L3: Graph + NavigationTree. Exit at L2: Graph + Domain-labeled Table + NavigationTree. Exit at L1: Graph + Domain-labeled Table + Vector + NavigationTree. Exit at L0: Graph + Domain-labeled Table + Vector + Datum + NavigationTree.
- Q: What does the NavigationTree visualize? → A: Each step taken, the decision made at each step (e.g., "make graph with entity X", "add label Y", "perform operation Z"), and the generated output at every step.
- Q: What shape should the NavigationTree take? → A: A Directed Acyclic Graph (DAG) structure, supporting branching exploration paths while preventing cycles.

## Assumptions

- **Target User Assumption (Constitution v1.2.0)**: Users have NO familiarity with data structures. They are domain curious minds—experts in their own field who want to explore data relevant to their domain. The system MUST shield users from technical complexity; they interact with domain concepts and natural language, never requiring understanding of tables, graphs, vectors, or schemas.
- Users have already explored deeper from their starting point (they have reached a computed result, list of values, or categorized items).
- The original source data is preserved in session and can be referenced during enrichment.
- Default enrichment functions will use common patterns (naming patterns, category classification, organizational dimensions).
- The UI will follow consistent interaction patterns across all navigation directions.
