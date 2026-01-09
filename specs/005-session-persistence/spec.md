# Feature Specification: Session Persistence

**Feature Branch**: `005-session-persistence`
**Created**: 2025-12-04
**Status**: Draft
**Input**: User description: "Can the results from previous actions always be kept in cache. It is tiring for the user to re-upload everything and tick boxes everytime."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Resume After Browser Refresh (Priority: P1)

A user is working through the Data Redesign wizard, has uploaded files and completed Step 2 (semantic matching). They accidentally refresh the browser or close the tab. When they return to the application, all their previous work should be preserved - uploaded files, wizard progress, column selections, and semantic matching results.

**Why this priority**: This is the core pain point described by the user. Losing all progress on browser refresh is extremely frustrating and wastes time.

**Independent Test**: User uploads 2 files, completes Step 1 and Step 2, refreshes browser, and verifies all data and selections are intact.

**Acceptance Scenarios**:

1. **Given** user has uploaded files and completed wizard Step 2, **When** user refreshes the browser, **Then** all uploaded files are still available and wizard shows Step 2 as current step with previous selections preserved.
2. **Given** user has uploaded files and defined column connections, **When** user closes and reopens the browser tab, **Then** all connections and semantic matching results are restored.
3. **Given** user has completed the full 6-step process, **When** user refreshes, **Then** the application shows the completed state with all datasets preserved.

---

### User Story 2 - Preserve Checkbox and Form Selections (Priority: P1)

A user has made numerous selections throughout the wizard - selecting columns, choosing connection methods (common key vs AI embeddings), defining categories. These selections should persist across page reloads and navigation between steps.

**Why this priority**: Equally critical to file persistence; re-selecting options is tedious and error-prone.

**Independent Test**: User makes selections in Step 2, navigates to Step 3, goes back to Step 2, and verifies all selections are preserved.

**Acceptance Scenarios**:

1. **Given** user has selected columns to connect in Step 2, **When** user navigates away and returns, **Then** column selections remain checked.
2. **Given** user has chosen "AI embeddings" as connection method, **When** user refreshes, **Then** the method choice is preserved and semantic results are displayed.
3. **Given** user has entered domain categories in Step 3, **When** user navigates back to Step 2 then forward again, **Then** domain inputs retain their values.

---

### User Story 3 - Start Fresh Option (Priority: P2)

While persistence is the default behavior, users need the ability to intentionally start over with a clean slate - clearing all cached data and beginning the wizard from scratch.

**Why this priority**: Important for flexibility but secondary to core persistence functionality.

**Independent Test**: User with cached session clicks "Start Fresh" and verifies all previous data is cleared.

**Acceptance Scenarios**:

1. **Given** user has a persisted session with uploaded files, **When** user clicks "Start Fresh" or "Clear Session", **Then** all cached data is removed and wizard resets to Step 1.
2. **Given** user clicks "Start Fresh", **When** they re-upload files, **Then** the wizard behaves as if starting for the first time with no legacy data.

---

### User Story 4 - Session Recovery Notification (Priority: P3)

When a user returns to the application with persisted data, they should see a clear indication that their previous session was recovered, with the option to continue or start fresh.

**Why this priority**: Improves user experience by providing clarity, but core functionality works without it.

**Independent Test**: User with persisted data opens the app and sees recovery notification with continue/reset options.

**Acceptance Scenarios**:

1. **Given** user has persisted session data, **When** user opens the application, **Then** a notification displays "Welcome back! Your previous session has been restored." with options to continue or start fresh.
2. **Given** user sees recovery notification, **When** user clicks "Continue", **Then** they proceed with restored data at their last position.

---

### Edge Cases

- What happens when browser storage is full? System should gracefully handle storage limits and notify user if data cannot be saved.
- What happens when user has corrupted cached data? System should detect corruption and offer to start fresh rather than crash.
- What happens when application version changes? System should handle schema changes gracefully, preserving what it can or prompting user to start fresh if incompatible.
- What happens with very large files (100MB+)? System should store file references or compress data to stay within storage limits.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST persist uploaded file data across browser refreshes and tab closures.
- **FR-002**: System MUST persist the current wizard step position so users resume where they left off.
- **FR-003**: System MUST persist all form selections including checkboxes, radio buttons, and text inputs made during the wizard.
- **FR-004**: System MUST persist semantic matching results so users don't have to re-run AI analysis.
- **FR-005**: System MUST persist the joined L3 dataset and all intermediate transformation results.
- **FR-006**: System MUST provide a "Start Fresh" or "Clear Session" option to intentionally reset all persisted data.
- **FR-007**: System MUST detect and load persisted data automatically when user opens the application.
- **FR-008**: System MUST handle storage errors gracefully, notifying users if data cannot be saved.
- **FR-009**: System MUST persist domain categorizations and entity mappings created during the workflow.
- **FR-010**: System SHOULD notify users when a previous session is recovered.

### Key Entities

- **Session State**: The complete state of user's progress including current step, all selections, and transformation results.
- **Uploaded Files**: Raw file data or references that were uploaded by the user.
- **Wizard Progress**: The step number, completion status of each step, and navigation history.
- **User Selections**: All checkbox states, dropdown values, text inputs, and connection definitions.
- **Transformation Results**: Semantic matching outputs, joined tables, and computed datasets at each level (L4, L3, L2, L1, L0).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can refresh the browser at any wizard step and resume with 100% of their data intact.
- **SC-002**: Users can close and reopen the browser within 24 hours and find their complete session restored.
- **SC-003**: Session recovery happens automatically in under 2 seconds when user opens the application.
- **SC-004**: 95% reduction in time spent re-uploading files and re-selecting options (from ~5 minutes to ~10 seconds for returning users).
- **SC-005**: "Start Fresh" option clears all data and resets the wizard within 1 second.
- **SC-006**: Users report significantly reduced frustration with the workflow in qualitative feedback.

## Assumptions

- Browser local storage or equivalent mechanism has sufficient capacity for typical use cases (files under 50MB total).
- Users are working on a single device/browser; cross-device sync is out of scope.
- Session data is stored locally in the browser, not on a server.
- Default session persistence duration is 7 days; after that, data may be cleared.
- File content is stored, not just file references, since users may not have original files available on return.
