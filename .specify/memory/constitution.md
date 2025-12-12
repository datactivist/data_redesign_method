<!--
================================================================================
SYNC IMPACT REPORT
================================================================================
Version change: 1.1.0 → 1.2.0

Modified principles:
- Principle V: Material expansion to explicitly state target users have NO
  familiarity with data structures - they are domain curious minds, not
  technically literate users

Added sections:
- New "Target User Assumption" subsection under Principle V

Removed sections: None

Templates requiring updates:
- .specify/templates/plan-template.md: ✅ No updates needed (user stories are
  already written in plain language)
- .specify/templates/spec-template.md: ✅ No updates needed
- .specify/templates/tasks-template.md: ✅ No updates needed

Follow-up TODOs: None
================================================================================
-->

# Data Redesign Method Constitution

## Core Principles

### I. Intuitiveness Through Abstraction Levels

Every dataset MUST be designable at five levels of abstraction (L0-L4). Designers MUST ensure users can navigate between levels to match their data literacy and needs:

- **Level 0** (Datum): Single entity-attribute-value triplet. Zero complexity. Machine domain.
- **Level 1** (Vector): Single entity with multiple attributes OR single attribute with multiple entities. Complexity C(1).
- **Level 2** (Table): Single dataset with multiple entities and attributes. Complexity C(2^n).
- **Level 3** (Linkable Multi-level): Multiple linkable datasets with hierarchical entities/attributes. Complexity C(2^ng(2^n-1)).
- **Level 4** (Unlinkable): Multi-level datasets with indefinable complexity. Human conceptual domain.

**Navigation Rules (Step-by-Step Traversal):**

Users MUST be able to navigate the abstraction hierarchy following these rules:

1. **Entry Point**: L4 serves as the primary and exclusive entry point—users begin at the highest conceptual level
2. **Step-by-Step Movement**: At each step, users MUST be able to either:
   - Move **horizontally** between related nodes at the same level, OR
   - Move **up or down** one level at a time (L3↔L2↔L1)
3. **No Return to L4**: Once a user leaves L4, they CANNOT return to L4—L4 is entry-only
4. **Infinite Exploration**: The number of navigation steps is not finite—users may continue exploring as long as needed
5. **Exit Anytime**: Users may exit the navigation process at any point that matches their analytical needs

**Rationale**: Datasets that adapt to user data literacy unlock broader access to data, information, knowledge, and reuse potential. Step-by-step navigation with bidirectional level movement (except for L4) ensures users can explore context freely without getting lost, while the L4 entry-only constraint maintains a clear conceptual starting point.

### II. Descent-Ascent Cycle

All dataset redesign MUST follow the Descent-Ascent cycle:

1. **Descent** (L4→L0): Reduce complexity by progressively isolating entities, extracting features, and reaching atomic metrics.
2. **Ascent** (L0→L4): Reconstruct complexity intentionally by adding dimensions aligned with user needs.

**Rationale**: Descending to Level 0 sanitizes data; ascending with intentional dimension selection creates datasets tailored to specific user needs. This transforms "data swamps" into "intuitive datasets."

### III. Complexity Quantification

Dataset complexity MUST be measured by the number of extractable relationships. Complexity reduction between levels follows quantifiable bounds:

- L4→L3: Nearly 100% reduction (indefinable to measurable)
- L3→L2: Reduction ∈ [75%, 100%)
- L2→L1: Reduction ∈ [75%, 100%)
- L1→L0: Exactly 100% reduction

**Rationale**: Quantifiable complexity enables evidence-based decisions about which abstraction level best serves a given user's data literacy and analytical needs.

### IV. Human-Data Interaction Granularity

Following the recursive principle of granular computing, datasets at upper abstraction levels MUST be constructible by human extrapolation from lower levels. The fundamental information granule (Level 0) serves as the ground truth anchor.

**Rationale**: This ensures end-to-end interpretability—users can trace from any derived insight back to atomic data points, maintaining trust and auditability.

### V. Design for Diverse Data Publics

Intuitive datasets MUST accommodate the full spectrum of data users:

- **Information seekers**: Users looking for single facts (Level 0-1 access)
- **Analysts**: Users exploring relationships within structured data (Level 2-3 access)
- **Creators**: Users building data products requiring full context (Level 3-4 access)

**Target User Assumption:**

Users of intuitive datasets are **NOT expected to have any familiarity with data structures**. Target users are:

- **Domain curious minds**: Experts in their own field (healthcare, finance, urban planning, etc.) who want to explore data relevant to their domain
- **Non-technical explorers**: Users who understand their domain deeply but have zero exposure to databases, tables, graphs, or other data concepts
- **Question-driven**: Users approach data with domain questions ("How does X affect Y in my field?"), not technical queries

The framework MUST shield users from all technical complexity. Users should interact with data using domain concepts and natural language, never requiring understanding of underlying data structures like tables, graphs, vectors, or schemas.

**Rationale**: Open data producers have historically assumed users possess technical data literacy. This assumption excludes the vast majority of domain experts who could extract immense value from data if it were presented in intuitive, domain-native terms. The Data Redesign Method exists specifically to bridge this gap.

## Complexity Levels

The five abstraction levels form the structural backbone of the framework:

| Level | Description | Complexity Order | Domain |
|-------|-------------|------------------|--------|
| L0 | Single datum (entity-attribute-value) | C(0) | Machine storage |
| L1 | Single vector (1 entity × N attributes OR N entities × 1 attribute) | C(1) | Basic interpretation |
| L2 | Single table (N entities × M attributes) | C(2^n) | Analytical exploration |
| L3 | Linkable multi-level datasets | C(2^ng(2^n-1)) | Complex analysis |
| L4 | Unlinkable multi-level datasets | C(∞) | Human conceptual linking |

**Implementation Note**: The `intuitiveness` Python library provides programmatic support for navigating between these levels. All technical implementation details MUST remain invisible to end users.

## Design Workflow

### Redesign Process

1. **Ingest**: Load raw data as Level 4 (assume no linkable structure exists)
2. **Descend**: Apply hierarchical decomposition to reach Level 0 ground truth
3. **Define Metric**: Establish the atomic metric that captures the core user need
4. **Ascend**: Reconstruct levels by adding dimensions aligned with user requirements
5. **Validate**: Confirm the final abstraction level matches target user's domain understanding (NOT technical literacy)

### Quality Gates

- Each transition between levels MUST preserve data integrity
- Ascent dimensions MUST be explicitly justified by user needs
- Final dataset MUST be independently testable at its target level
- User-facing interfaces MUST use domain terminology, not technical data terms

## Governance

This constitution establishes non-negotiable principles for the Data Redesign Method project. All contributions MUST comply with these principles.

### Amendment Procedure

1. Proposed amendments require documentation of rationale and impact
2. Amendments affecting core principles (I-V) require major version increment
3. New sections or material expansions require minor version increment
4. Clarifications and non-semantic changes require patch version increment

### Versioning Policy

- Format: MAJOR.MINOR.PATCH (semantic versioning)
- Breaking changes to principles: increment MAJOR
- Additive changes: increment MINOR
- Fixes/clarifications: increment PATCH

### Compliance Review

- All PRs/code reviews MUST verify alignment with the five abstraction levels
- Documentation MUST reference which complexity level(s) a feature addresses
- User-facing features MUST be evaluated against the Target User Assumption (non-technical domain experts)
- See README.md for runtime development guidance

**Version**: 1.2.0 | **Ratified**: 2025-11-24 | **Last Amended**: 2025-12-04
