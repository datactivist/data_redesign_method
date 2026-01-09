# Feature Specification: Streamlit Minimalist Design Makeup

**Feature Branch**: `007-streamlit-design-makeup`
**Created**: 2025-12-12
**Status**: Draft
**Input**: Apply minimalist design makeup to the Streamlit interface following Gael Penessot's DataGyver philosophy: use config.toml for theming, st.html() for CSS injection, hide raw Streamlit chrome, implement custom typography with IBM Plex Sans, create warm neutral color palette, add refined metric cards, simplify progress indicators, and ensure the framework becomes invisible to users.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Professional Interface First Impression (Priority: P1)

A user opens the Intuitiveness application and immediately perceives it as a polished, professional data tool rather than a generic Streamlit prototype. The interface feels intentionally designed with cohesive typography, colors, and spacing that instill confidence in the tool's capabilities.

**Why this priority**: First impressions determine user trust and adoption. If the interface looks like a default Streamlit app, users may perceive the underlying functionality as less reliable or professional.

**Independent Test**: Can be tested by launching the app and visually verifying that default Streamlit elements (hamburger menu, footer, "Made with Streamlit" badge) are hidden, custom fonts are loaded, and the color palette is consistent.

**Acceptance Scenarios**:

1. **Given** a user navigates to the application URL, **When** the page loads, **Then** no default Streamlit chrome elements (hamburger menu, footer, "Made with Streamlit" badge) are visible
2. **Given** the application is loaded, **When** the user views any text, **Then** the custom typography (IBM Plex Sans) is displayed consistently throughout
3. **Given** the application is loaded, **When** the user views the interface, **Then** all colors belong to the defined warm neutral palette with consistent accent colors

---

### User Story 2 - Streamlined Progress Tracking (Priority: P2)

A user navigating through the descent-ascent workflow sees a clean, minimal progress indicator that communicates their current position without visual clutter. The indicator uses subtle visual cues rather than heavy animations or complex graphics.

**Why this priority**: Progress indicators are constantly visible during the workflow. A cluttered or distracting indicator degrades the overall experience, while a refined one reinforces the professional aesthetic.

**Independent Test**: Can be tested by navigating through workflow steps and verifying the progress indicator displays current state clearly with minimal visual elements and no distracting animations.

**Acceptance Scenarios**:

1. **Given** a user is on any workflow step, **When** they view the progress indicator, **Then** it clearly shows their current level (L4-L0) using simple visual markers
2. **Given** a user transitions between levels, **When** the progress indicator updates, **Then** the transition is subtle (no flashy animations) and maintains readability
3. **Given** a user views the sidebar, **When** they look at the progress section, **Then** it occupies minimal vertical space while remaining informative

---

### User Story 3 - Refined Data Presentation (Priority: P2)

When viewing computed metrics or data summaries (L0 datum, aggregation results), the user sees information presented in elegant metric cards with clear hierarchy rather than raw Streamlit widgets or plain text.

**Why this priority**: Data presentation is the core value of the application. Well-designed metric cards make insights more scannable and memorable.

**Independent Test**: Can be tested by completing a descent workflow to L0 and verifying that results are displayed in styled card components with clear visual hierarchy.

**Acceptance Scenarios**:

1. **Given** a user reaches the L0 results view, **When** they view computed metrics, **Then** each metric is displayed in a styled card with label, value, and optional contextual information
2. **Given** metric cards are displayed, **When** the user scans the interface, **Then** the visual hierarchy guides attention to the primary value first
3. **Given** multiple metrics are displayed, **When** viewed together, **Then** they align in a consistent grid layout with appropriate spacing

---

### User Story 4 - Consistent Component Styling (Priority: P3)

Throughout the application, all interactive elements (buttons, inputs, expanders, tabs) follow a consistent visual language that matches the overall minimalist aesthetic.

**Why this priority**: Consistency reinforces professionalism and reduces cognitive load. Users should never encounter jarring visual inconsistencies.

**Independent Test**: Can be tested by navigating through all major screens and verifying that buttons, inputs, and containers share consistent styling (border radius, padding, hover states).

**Acceptance Scenarios**:

1. **Given** a user interacts with any button, **When** they hover over it, **Then** the button displays a subtle, consistent hover effect
2. **Given** the application displays expanders or containers, **When** viewed, **Then** they use consistent border styling and shadow depth
3. **Given** the application displays form inputs, **When** viewed, **Then** they share consistent border radius and focus states with other components

---

### Edge Cases

- What happens when custom fonts fail to load? Fallback to system fonts that maintain similar visual weight (sans-serif stack)
- How does the interface handle very long text in metric cards? Text truncation with tooltips or graceful wrapping that maintains card proportions
- What happens on mobile/narrow viewports? Responsive adjustments that preserve readability while adapting layout
- How does the design handle error states? Error messages use accent colors consistently with the palette while remaining prominent

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a `.streamlit/config.toml` file that defines the application theme (colors, fonts, border radius)
- **FR-002**: System MUST hide default Streamlit UI chrome (hamburger menu, footer, header, "Made with Streamlit" badge) via CSS injection
- **FR-003**: System MUST load and apply IBM Plex Sans font family consistently across all text elements
- **FR-004**: System MUST implement a warm neutral color palette with defined primary, secondary, background, and accent colors
- **FR-005**: System MUST provide a reusable metric card component for displaying L0 datum and aggregation results
- **FR-006**: System MUST simplify the progress indicator to use minimal visual elements while clearly showing current workflow position
- **FR-007**: System MUST centralize all CSS customizations in a dedicated styles module for maintainability
- **FR-008**: System MUST ensure all interactive components (buttons, inputs, expanders) have consistent styling
- **FR-009**: System MUST provide graceful fallbacks when custom fonts fail to load
- **FR-010**: System MUST maintain responsive behavior for different viewport sizes

### Key Entities

- **Theme Configuration**: Centralized definition of colors, fonts, spacing, and border radius values used throughout the application
- **Style Module**: Collection of CSS injection strings organized by purpose (chrome hiding, typography, components)
- **Metric Card**: Reusable display component for presenting key-value data with optional contextual information

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Zero default Streamlit chrome elements visible when application loads (hamburger menu, footer, badge hidden)
- **SC-002**: 100% of text elements use the specified custom font family when fonts load successfully
- **SC-003**: All color values used in the interface belong to the defined palette (verifiable by CSS inspection)
- **SC-004**: Progress indicator height reduced by at least 40% compared to current implementation while maintaining clarity
- **SC-005**: Users can identify their current workflow level within 2 seconds of viewing the progress indicator
- **SC-006**: Metric cards display with consistent dimensions and spacing across all results views
- **SC-007**: All CSS customizations consolidated in 3 or fewer module files
- **SC-008**: Application maintains usability on viewports as narrow as 768px

## Assumptions

- IBM Plex Sans is available via Google Fonts CDN for loading
- Users have modern browsers that support CSS custom properties and Google Fonts
- The existing Streamlit version (1.x) supports `st.html()` for CSS injection
- The current inline CSS in `streamlit_app.py` can be refactored without breaking existing functionality
- The warm neutral palette will use tones similar to #fafaf9 (background), #1c1917 (text), #2563eb (accent)

## Out of Scope

- Complete redesign of data visualizations (charts, graphs)
- Changes to the underlying workflow logic or data processing
- Dark mode implementation (future enhancement)
- Custom iconography or illustration work
- Accessibility audit (should be separate feature)
