# Feature Specification: Ascent Phase Precision

**Feature Branch**: `004-ascent-precision`
**Created**: 2025-12-04
**Status**: Draft
**Input**: User description: "Let's add more precision to the ascent phase. When going from L0 to L1, there is only one option which is to unfold the datum into the vector that previously built him. From L1 to L2, it is all about adding domain-related columns to a vector and get a 2D table. The user should be able to choose which domains they want, which classifier to more precisely understand data. So that's the same function than the one we use to go from L3 to L2. From L2 to L3, we need to add connection with an extra-dimension entity from which we can rebuild a graph."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Unfold Datum to Source Vector (Priority: P1)

A user at L0 (single aggregated value like median, average, or count) wants to see the underlying distribution that produced this value. The system "unfolds" the datum by displaying the original vector (L1) from which the aggregation was computed.

**Why this priority**: This is the most constrained ascent operation - there's only one possible outcome (the source vector), making it the simplest to implement and most deterministic. It provides immediate value by letting users understand how their aggregated metric was derived.

**Independent Test**: Can be fully tested by descending from L1→L0 with any aggregation, then ascending back and verifying the original vector is restored. Delivers transparency into aggregation computations.

**Acceptance Scenarios**:

1. **Given** a user is at L0 with a datum produced from averaging a vector, **When** they choose to ascend to L1, **Then** the system displays the original vector with all its values that were averaged
2. **Given** a user is at L0 with a count aggregation, **When** they choose to ascend, **Then** the system shows the full list of items that were counted
3. **Given** a user is at L0 where no parent vector exists (orphan datum), **When** they try to ascend, **Then** the system informs them that ascent is not possible and explains why

---

### User Story 2 - Enrich Vector with Domain Columns (Priority: P2)

A user at L1 (vector/series of values) wants to transform it into a 2D table (L2) by adding domain-categorization columns. This is like adding an axis to a 1D distribution to create a structured table. The user selects domains and a classifier method to categorize each value in the vector.

**Why this priority**: This operation reuses the domain categorization logic from L3→L2 descent, ensuring consistency across navigation directions. It enables users to add contextual meaning to raw vectors.

**Independent Test**: Can be fully tested by creating a vector at L1, ascending to L2, selecting domains, and verifying each vector value is categorized into a table with domain columns.

**Acceptance Scenarios**:

1. **Given** a user is at L1 with a vector of values, **When** they choose to ascend to L2 and specify domains (e.g., "Revenue, Volume, ETP"), **Then** the system creates a 2D table where each original value has a domain classification column
2. **Given** a user is at L1, **When** they enable semantic matching and set a similarity threshold, **Then** the domain categorization uses AI-powered semantic similarity
3. **Given** a user is at L1 with a vector, **When** they choose keyword-based categorization, **Then** values are matched to domains via simple text matching
4. **Given** a user is at L1 with values that don't match any domain, **When** categorization is applied, **Then** those values are assigned to an "Unmatched" domain

---

### User Story 3 - Build Graph from Table with Extra Entity (Priority: P2)

A user at L2 (domain-categorized table) wants to enrich it into a graph (L3) by defining an extra-dimensional entity that creates relationships between table rows. The user specifies an entity type and how it connects to existing data, rebuilding a knowledge graph.

**Why this priority**: This completes the full bidirectional navigation capability, allowing users to go from tables back to graphs. It requires user input to define the new entity dimension and connection logic.

**Independent Test**: Can be fully tested by creating a domain table at L2, ascending to L3 by defining a new entity type and relationship, and verifying a connected graph is produced.

**Acceptance Scenarios**:

1. **Given** a user is at L2 with a domain-categorized table, **When** they choose to ascend to L3 and define an extra entity type (e.g., "Department" from a "department_name" column), **Then** the system creates nodes for this entity and connects table rows to them
2. **Given** a user selects a column for the extra entity, **When** the column has unique values, **Then** each unique value becomes a node that connects to related rows
3. **Given** a user defines multiple relationship types during ascent, **When** the graph is built, **Then** all specified relationships are created with appropriate edge types
4. **Given** a user is at L2 with no suitable columns for entity extraction, **When** they try to ascend, **Then** the system suggests column options or warns that manual entity definition is needed

---

### Edge Cases

- What happens when the L0 datum was created manually (not from aggregation)? → Ascent blocked with explanation
- What happens when vector values contain special characters or null values during domain categorization? → Nulls become "Unknown", special chars preserved
- What happens when the user defines an entity column with only one unique value? → System warns but allows (creates single-node graph)
- What happens when the user cancels mid-ascent? → Navigation state is preserved, user returns to current level

## Requirements *(mandatory)*

### Functional Requirements

#### L0→L1 Ascent (Unfold)
- **FR-001**: System MUST display the source vector (parent data) from which the L0 datum was aggregated
- **FR-002**: System MUST show the aggregation method that was used (e.g., "median", "average", "count")
- **FR-003**: System MUST block ascent when no parent vector exists and display an explanatory message
- **FR-004**: System MUST preserve the column name from the original vector when unfolding

#### L1→L2 Ascent (Domain Enrichment)
- **FR-005**: System MUST allow users to specify domain names (comma-separated input)
- **FR-006**: System MUST support both semantic matching and keyword-based categorization methods
- **FR-007**: System MUST allow users to set a similarity threshold for semantic matching (0.1 to 0.9)
- **FR-008**: System MUST assign "Unmatched" domain to values that don't meet the similarity threshold
- **FR-009**: System MUST reuse the existing domain categorization logic from L3→L2 descent
- **FR-010**: System MUST create a 2D table with the original vector values plus a "domain" column

#### L2→L3 Ascent (Graph Enrichment)
- **FR-011**: System MUST allow users to select a column to extract as a new entity type
- **FR-012**: System MUST allow users to define the relationship type connecting existing entities to the new entity
- **FR-013**: System MUST create nodes for each unique value in the selected entity column
- **FR-014**: System MUST create edges connecting original table rows (as entities) to the new entity nodes
- **FR-015**: System MUST ensure the resulting graph has no orphan nodes (all nodes have at least one relationship)

### Key Entities

- **Datum (L0)**: A single scalar value with metadata about its aggregation origin (parent vector reference, aggregation method)
- **Vector (L1)**: A 1D series of values with a name/description attribute
- **Domain Table (L2)**: A 2D table with rows categorized by domain columns
- **Knowledge Graph (L3)**: A NetworkX graph with typed nodes and labeled relationships

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can unfold any aggregated datum to its source vector within 2 seconds
- **SC-002**: Users can complete domain enrichment (L1→L2) in under 30 seconds including domain input
- **SC-003**: Users can build a graph from a table (L2→L3) in under 60 seconds including entity definition
- **SC-004**: 100% of ascent operations from L0 with parent data successfully restore the original vector
- **SC-005**: Domain categorization in L1→L2 ascent produces identical results to L3→L2 descent for the same inputs
- **SC-006**: Resulting graphs from L2→L3 ascent have no orphan nodes (per design principle #1)
- **SC-007**: Users can identify ascent options and their meanings within 5 seconds of viewing the navigation panel

## Assumptions

- **Target User Assumption (Constitution v1.2.0)**: Users have NO familiarity with data structures. They are domain curious minds—experts in their own field who want to explore data relevant to their domain. The system MUST shield users from technical complexity; they interact with domain concepts and natural language, never requiring understanding of tables, graphs, vectors, or schemas.
- The existing domain categorization logic (semantic matching) is available and functional (internal implementation)
- Computed results store a reference to their source values when created via summarization
- The application runs in its existing environment with access to current UI components
- Users do NOT need to understand abstraction levels—the UI presents navigation in domain-native terms
