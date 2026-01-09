# Feature Specification: Data.gouv.fr Search Integration

**Feature Branch**: `008-datagouv-search`
**Created**: 2025-12-12
**Status**: Draft
**Input**: User description: "Integration with data.gouv.fr: Add a search bar interface at the start of the application with the message 'Redesign any data for your intent' with French translation. Users can search French open data directly from data.gouv.fr and select datasets to redesign."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Search Open Data by Intent (Priority: P1)

A data analyst arrives at the application without their own CSV files. They have an intent or question (e.g., "vaccination rates in France") and want to find relevant open data from data.gouv.fr to redesign for their needs.

**Why this priority**: This is the core value proposition - enabling users to discover and work with French open data without needing to navigate data.gouv.fr separately. It transforms the app from "upload your files" to "find and redesign any data."

**Independent Test**: Can be fully tested by entering a search query and verifying dataset results appear with relevant metadata.

**Acceptance Scenarios**:

1. **Given** I am on the application homepage without uploaded files, **When** I view the main interface, **Then** I see a prominent search bar with the message "Redesign any data for your intent" (or "Redesignez toute donnee selon votre intention" in French).

2. **Given** I am viewing the search interface, **When** I type "vaccination" and submit, **Then** I see a list of matching datasets from data.gouv.fr with titles, descriptions, and last update dates.

3. **Given** search results are displayed, **When** I hover over a dataset card, **Then** I see additional metadata (organization, format, number of resources).

---

### User Story 2 - Select and Load Dataset (Priority: P1)

After finding a relevant dataset, the user selects it to begin the redesign workflow. The selected dataset loads into the application as the L4 starting point.

**Why this priority**: Essential to complete the search-to-redesign flow. Without this, search has no actionable outcome.

**Independent Test**: Can be tested by selecting a dataset and verifying it loads into the workflow with correct data.

**Acceptance Scenarios**:

1. **Given** I see search results, **When** I click on a dataset card, **Then** I see the available resources (files) within that dataset.

2. **Given** I am viewing dataset resources, **When** I select a CSV resource, **Then** the file downloads and loads into the application automatically.

3. **Given** a dataset has loaded, **When** the loading completes, **Then** I am taken to the redesign workflow with the data displayed at L4 level.

4. **Given** a dataset is loading, **When** I wait, **Then** I see a progress indicator showing download status.

---

### User Story 3 - Bilingual Interface (Priority: P2)

French and English-speaking users can use the search interface in their preferred language. The interface respects the existing language toggle.

**Why this priority**: Supports the existing internationalization pattern and ensures accessibility for both French and international users.

**Independent Test**: Can be tested by toggling language and verifying all search-related text changes appropriately.

**Acceptance Scenarios**:

1. **Given** the language is set to English, **When** I view the search interface, **Then** I see "Redesign any data for your intent" and English labels.

2. **Given** the language is set to French, **When** I view the search interface, **Then** I see "Redesignez toute donnee selon votre intention" and French labels.

3. **Given** search results are displayed, **When** I toggle the language, **Then** the interface labels update but dataset titles remain in their original language.

---

### User Story 4 - Combined Entry Points (Priority: P3)

Users can choose between uploading their own files OR searching data.gouv.fr. Both entry points lead to the same redesign workflow.

**Why this priority**: Maintains backward compatibility while adding the new search capability.

**Independent Test**: Can be tested by verifying both the file upload and search paths lead to the same workflow state.

**Acceptance Scenarios**:

1. **Given** I am on the homepage, **When** I view the interface, **Then** I see both the search bar and the file upload option clearly presented.

2. **Given** I choose to upload my own files, **When** I complete the upload, **Then** the workflow proceeds as before.

3. **Given** I choose to search data.gouv.fr, **When** I select and load a dataset, **Then** the workflow proceeds identically to file upload.

---

### Edge Cases

- What happens when a search returns no results? Display a helpful message with suggestions.
- What happens when the data.gouv.fr API is unavailable? Show an error message and suggest using file upload instead.
- What happens when a selected CSV file is too large? Show file size before download and warn for files over 50MB.
- What happens when the CSV format is incompatible? Use the existing smart_load_csv function for automatic format detection.
- What happens when a dataset has no CSV resources? Indicate available formats and suggest alternative datasets.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display a search bar prominently on the initial application view when no data is loaded.
- **FR-002**: System MUST query the data.gouv.fr API when the user submits a search query.
- **FR-003**: System MUST display search results as cards showing dataset title, description (truncated), organization, and last modified date.
- **FR-004**: System MUST allow users to click on a dataset to view its resources (files).
- **FR-005**: System MUST allow users to select a CSV resource to load into the redesign workflow.
- **FR-006**: System MUST download and parse the selected CSV using automatic format detection (encoding, delimiter).
- **FR-007**: System MUST support the existing bilingual interface (English/French) for all search-related text.
- **FR-008**: System MUST display the message "Redesign any data for your intent" (EN) / "Redesignez toute donnee selon votre intention" (FR) above the search bar.
- **FR-009**: System MUST show loading indicators during search and download operations.
- **FR-010**: System MUST handle API errors gracefully with user-friendly messages.
- **FR-011**: System MUST preserve the existing file upload functionality alongside the new search feature.
- **FR-012**: System MUST cache downloaded datasets locally to avoid re-downloading on page refresh.

### Key Entities

- **SearchQuery**: The user's intent expressed as a text string, used to query data.gouv.fr.
- **DatasetResult**: A dataset returned from data.gouv.fr, containing title, description, organization, resources, and metadata.
- **Resource**: A file within a dataset (typically CSV), with URL, format, size, and last modified date.
- **LoadedData**: The parsed CSV data that enters the redesign workflow at L4 level.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can search and load a dataset from data.gouv.fr in under 60 seconds (excluding download time for large files).
- **SC-002**: Search results appear within 3 seconds of submitting a query.
- **SC-003**: 90% of datasets with CSV resources successfully load into the workflow on first attempt.
- **SC-004**: Users understand both entry points (upload vs search) without confusion, measured by task completion rate.
- **SC-005**: Bilingual users can complete the search workflow entirely in their preferred language.
- **SC-006**: Zero workflow disruption when data.gouv.fr is temporarily unavailable (graceful fallback to upload).

## Assumptions

- The data.gouv.fr public API remains available and free to use.
- The existing DataGouvAPI library (from data-gouv-skill) handles API communication.
- Most datasets of interest have CSV resources available.
- The existing smart_load_csv function can handle French CSV formats (semicolon separators, comma decimals).
- The application already has a bilingual infrastructure (EN/FR) that can be extended.
- Users have sufficient internet bandwidth to download datasets (typical range 1-50MB).

## Out of Scope

- Integration with other open data portals (data.europa.eu, etc.) - future enhancement.
- Dataset preview before download - users see metadata only.
- Saved searches or search history - users search fresh each time.
- Dataset recommendations based on user history - no personalization.
- Downloading non-CSV formats (JSON, XML, Excel) - CSV only for initial release.
