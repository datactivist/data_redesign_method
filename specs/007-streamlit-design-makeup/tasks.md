# Tasks: Streamlit Minimalist Design Makeup

**Branch**: `007-streamlit-design-makeup` | **Date**: 2025-12-13 | **Spec**: [spec.md](./spec.md)

## Task Overview

| Phase | Focus | Tasks | Dependencies |
|-------|-------|-------|--------------|
| 1 | Setup & Configuration | 2 | None |
| 2 | Foundational Styles | 3 | Phase 1 |
| 3 | US1 - Professional Interface | 3 | Phase 2 |
| 4 | US2 - Progress Tracking | 2 | Phase 3 |
| 5 | US3 - Metric Cards | 2 | Phase 3 |
| 6 | US4 - Component Styling | 2 | Phase 3 |
| 7 | Integration & Polish | 3 | Phases 4-6 |
| **8** | **Klein Blue Landing Page** | **3** | **Phase 3** |
| **9** | **Sarazin & Mourey Tutorial** | **4** | **Phase 8** |
| **10** | **Flow Integration** | **3** | **Phase 9** |

---

## Phase 1: Setup & Configuration

### Task 1.1: Create Streamlit config.toml

**User Story**: US1 (Professional Interface)
**Priority**: P1
**Requirement**: FR-001

**Description**: Create `.streamlit/config.toml` with warm neutral theme settings.

**Implementation**:
```toml
[theme]
primaryColor = "#2563eb"
backgroundColor = "#fafaf9"
secondaryBackgroundColor = "#f5f5f4"
textColor = "#1c1917"
font = "sans serif"
```

**Acceptance Criteria**:
- [x] File exists at `.streamlit/config.toml`
- [x] App loads with new background colors
- [x] Primary color appears on interactive elements

**Files**: `.streamlit/config.toml` (CREATE)

---

### Task 1.2: Create styles module directory structure

**User Story**: All
**Priority**: P1
**Requirement**: FR-002

**Description**: Create the `intuitiveness/styles/` module structure with empty files.

**Implementation**:
```bash
mkdir -p intuitiveness/styles
touch intuitiveness/styles/__init__.py
touch intuitiveness/styles/chrome.py
touch intuitiveness/styles/typography.py
touch intuitiveness/styles/palette.py
touch intuitiveness/styles/components.py
touch intuitiveness/styles/progress.py
```

**Acceptance Criteria**:
- [x] Directory `intuitiveness/styles/` exists
- [x] All 6 module files created
- [x] `__init__.py` is importable

**Files**:
- `intuitiveness/styles/__init__.py` (CREATE)
- `intuitiveness/styles/chrome.py` (CREATE)
- `intuitiveness/styles/typography.py` (CREATE)
- `intuitiveness/styles/palette.py` (CREATE)
- `intuitiveness/styles/components.py` (CREATE)
- `intuitiveness/styles/progress.py` (CREATE)

---

## Phase 2: Foundational Styles

### Task 2.1: Implement palette.py with color tokens

**User Story**: US1
**Priority**: P1
**Requirement**: FR-003
**Contract**: `contracts/css-contracts.md#palette_css`

**Description**: Define the COLORS dictionary and PALETTE_CSS string with CSS custom properties.

**Implementation**: Follow `data-model.md#color-palette` specification.

**Acceptance Criteria**:
- [x] COLORS dict exported with all 15 color tokens
- [x] PALETTE_CSS string defines CSS custom properties
- [x] Colors match specification exactly

**Files**: `intuitiveness/styles/palette.py` (MODIFY)

---

### Task 2.2: Implement typography.py with font loading

**User Story**: US1
**Priority**: P1
**Requirement**: FR-002
**Contract**: `contracts/css-contracts.md#typography_css`

**Description**: Define typography tokens and CSS for IBM Plex Sans loading.

**Implementation**:
- Include `@import` for Google Fonts
- Define TYPOGRAPHY dict with font sizes, weights
- Export TYPOGRAPHY_CSS string

**Acceptance Criteria**:
- [x] TYPOGRAPHY dict exported with all tokens
- [x] TYPOGRAPHY_CSS includes Google Fonts import
- [x] Fallback font stack specified
- [x] Headings styled with correct weights

**Files**: `intuitiveness/styles/typography.py` (MODIFY)

---

### Task 2.3: Implement chrome.py to hide Streamlit UI

**User Story**: US1
**Priority**: P1
**Requirement**: FR-001
**Contract**: `contracts/css-contracts.md#hide_chrome_css`

**Description**: CSS to hide hamburger menu, footer, header, and "Made with Streamlit" badge.

**Implementation**:
```css
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.viewerBadge_container__r5tak { display: none; }
.block-container { padding-top: 2rem !important; }
```

**Acceptance Criteria**:
- [x] HIDE_CHROME_CSS exported
- [x] No hamburger menu visible
- [x] No footer visible
- [x] No "Made with Streamlit" badge
- [x] Content area adjusted for hidden header

**Files**: `intuitiveness/styles/chrome.py` (MODIFY)

---

## Phase 3: US1 - Professional Interface Integration

### Task 3.1: Implement styles/__init__.py integration

**User Story**: US1
**Priority**: P1
**Requirement**: FR-001, FR-002, FR-003
**Contract**: `contracts/css-contracts.md#integration-contract`

**Description**: Create the integration module that combines all CSS and exports `inject_all_styles()`.

**Implementation**:
```python
from .chrome import HIDE_CHROME_CSS
from .typography import TYPOGRAPHY_CSS
from .palette import PALETTE_CSS, COLORS
from .components import COMPONENT_CSS
from .progress import PROGRESS_CSS

ALL_STYLES = f"""<style>
{PALETTE_CSS}
{TYPOGRAPHY_CSS}
{HIDE_CHROME_CSS}
{COMPONENT_CSS}
{PROGRESS_CSS}
</style>"""

def inject_all_styles():
    import streamlit as st
    st.markdown(ALL_STYLES, unsafe_allow_html=True)
```

**Acceptance Criteria**:
- [x] All style constants importable from `intuitiveness.styles`
- [x] `inject_all_styles()` function available
- [x] Single CSS injection point

**Files**: `intuitiveness/styles/__init__.py` (MODIFY)

---

### Task 3.2: Integrate styles into streamlit_app.py

**User Story**: US1
**Priority**: P1
**Requirement**: FR-001

**Description**: Add `inject_all_styles()` call after `st.set_page_config()` in main app.

**Implementation**:
```python
from intuitiveness.styles import inject_all_styles

def main():
    st.set_page_config(...)
    inject_all_styles()  # Add this line
    # ... rest of app
```

**Acceptance Criteria**:
- [x] Import statement added
- [x] `inject_all_styles()` called early in main()
- [x] App loads with all custom styles applied

**Files**: `intuitiveness/streamlit_app.py` (MODIFY ~5 lines)

---

### Task 3.3: Verify US1 with Playwright MCP

**User Story**: US1
**Priority**: P1
**Requirement**: SC-001, SC-002, SC-003

**Description**: Visual verification that professional interface elements are applied.

**Test Cases**:
1. Navigate to app
2. Verify no hamburger menu visible
3. Verify no footer visible
4. Verify IBM Plex Sans font rendered
5. Verify warm background colors

**Acceptance Criteria**:
- [x] All chrome elements hidden
- [x] Custom fonts load successfully
- [x] Color palette applied consistently

**Files**: None (testing only)

---

## Phase 4: US2 - Progress Tracking

### Task 4.1: Implement progress.py styles

**User Story**: US2
**Priority**: P2
**Requirement**: FR-004
**Contract**: `contracts/css-contracts.md#progress_css`

**Description**: Create CSS for simplified text-based progress indicator.

**Implementation**: Follow contract specification for `.progress-indicator`, `.progress-level`, `.completed`, `.current` classes.

**Acceptance Criteria**:
- [x] PROGRESS_CSS exported
- [x] Progress levels styled correctly
- [x] Current level highlighted
- [x] Completed levels show checkmark style

**Files**: `intuitiveness/styles/progress.py` (MODIFY)

---

### Task 4.2: Refactor progress indicator in streamlit_app.py

**User Story**: US2
**Priority**: P2
**Requirement**: FR-004

**Description**: Replace complex HTML progress indicator with text-based version using `render_minimal_progress()`.

**Implementation**:
- Remove `inject_right_sidebar_css()` function (~80 lines)
- Remove `render_progress_bar()` HTML injection
- Remove `render_ascent_progress_bar()` HTML injection
- Add new `render_minimal_progress()` function from quickstart.md

**Acceptance Criteria**:
- [x] Old progress functions removed
- [x] New minimal progress indicator works
- [x] Descent mode shows L4→L0 correctly
- [x] Ascent mode shows L0→L3 correctly
- [x] ~80 lines of code removed

**Files**: `intuitiveness/streamlit_app.py` (MODIFY ~100 lines)

---

## Phase 5: US3 - Metric Cards

### Task 5.1: Create metric_card.py component

**User Story**: US3
**Priority**: P2
**Requirement**: FR-005
**Contract**: `contracts/css-contracts.md#render_metric_card`

**Description**: Create reusable metric card component for L0 result display.

**Implementation**: Follow quickstart.md Step 5 specification.

**Acceptance Criteria**:
- [x] `render_metric_card()` function created
- [x] Accepts label, value, delta, description parameters
- [x] Uses COLORS from palette
- [x] Renders styled HTML card

**Files**:
- `intuitiveness/ui/__init__.py` (MODIFY - add export)
- `intuitiveness/ui/metric_card.py` (CREATE)

---

### Task 5.2: Integrate metric cards in L0 display

**User Story**: US3
**Priority**: P2
**Requirement**: FR-005

**Description**: Use `render_metric_card()` to display L0 datum results.

**Implementation**: Replace existing L0 result display with metric card component.

**Acceptance Criteria**:
- [x] L0 results display as styled cards
- [x] Card shows label and value clearly
- [x] Optional delta/description shown when available

**Files**: `intuitiveness/streamlit_app.py` (MODIFY ~20 lines)

---

## Phase 6: US4 - Component Styling

### Task 6.1: Implement components.py button styles

**User Story**: US4
**Priority**: P3
**Requirement**: FR-006, FR-007
**Contract**: `contracts/css-contracts.md#component_css`

**Description**: CSS for primary/secondary buttons, expanders, and inputs.

**Implementation**: Follow contract specification for button, expander, input selectors.

**Acceptance Criteria**:
- [x] COMPONENT_CSS exported
- [x] Primary buttons use accent color
- [x] Secondary buttons have transparent background
- [x] Expanders have consistent border radius
- [x] Inputs have consistent styling

**Files**: `intuitiveness/styles/components.py` (MODIFY)

---

### Task 6.2: Verify component styling

**User Story**: US4
**Priority**: P3
**Requirement**: SC-006

**Description**: Visual verification of component styling across app screens.

**Test Cases**:
1. Check "Next" button styling (primary)
2. Check "Back" button styling (secondary)
3. Check expander appearance
4. Check text input fields

**Acceptance Criteria**:
- [x] All buttons styled consistently
- [x] All interactive elements match design system

**Files**: None (testing only)

---

## Phase 7: Integration & Polish

### Task 7.1: Remove remaining inline CSS

**User Story**: All
**Priority**: P2
**Requirement**: FR-001

**Description**: Search for and remove any remaining inline CSS from streamlit_app.py.

**Implementation**:
- Search for `st.markdown("""<style>` patterns
- Move necessary styles to appropriate style modules
- Remove duplicate CSS definitions

**Acceptance Criteria**:
- [x] No inline `<style>` blocks in streamlit_app.py
- [x] All CSS centralized in styles/ module
- [x] App functionality unchanged

**Files**: `intuitiveness/streamlit_app.py` (MODIFY)

---

### Task 7.2: Responsive testing at 768px

**User Story**: All
**Priority**: P2
**Requirement**: SC-007

**Description**: Verify layout works at minimum 768px viewport width.

**Test Cases**:
1. Resize browser to 768px width
2. Verify sidebar collapses properly
3. Verify content remains readable
4. Verify no horizontal scrolling

**Acceptance Criteria**:
- [x] App usable at 768px width
- [x] No layout breaks
- [x] Content readable

**Files**: None (testing only)

---

### Task 7.3: Performance verification

**User Story**: All
**Priority**: P2
**Requirement**: SC-008

**Description**: Verify page load time remains under 3 seconds.

**Test Cases**:
1. Clear browser cache
2. Navigate to app
3. Measure time to interactive
4. Verify fonts load correctly

**Acceptance Criteria**:
- [x] Page loads in under 3 seconds
- [x] Google Fonts CDN accessible
- [x] No console errors

**Files**: None (testing only)

---

## Phase 8: Klein Blue Landing Page

### Task 8.1: Create Klein Blue CSS Module

**User Story**: US5 (Single-Purpose Landing)
**Priority**: P1
**Requirement**: FR-011

**Description**: Create the Klein Blue landing page CSS with International Klein Blue (#002fa7) gradient, geometric overlays, and Outfit font family.

**Implementation**:
```css
/* Klein Blue gradient background */
.klein-landing {
    background: linear-gradient(180deg, #002fa7 0%, #001d6e 100%);
    border-radius: 0;
    padding: 80px 40px 60px 40px;
    position: relative;
    overflow: hidden;
}

/* Outfit typography */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Frosted glass search input */
.klein-search-input input {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}
```

**Acceptance Criteria**:
- [x] Klein Blue gradient applied (#002fa7 → #001d6e)
- [x] Outfit font loaded from Google Fonts
- [x] Geometric overlays with subtle animation
- [x] Frosted glass search input styling

**Files**: `intuitiveness/ui/datagouv_search.py` (MODIFY - add `_get_klein_blue_landing_css()`)

---

### Task 8.2: Implement Single-Purpose Landing Layout

**User Story**: US5
**Priority**: P1
**Requirement**: FR-012

**Description**: Center the landing page around "Redesign any data for your intent" message with search bar. Remove all other headers when on landing.

**Implementation**:
```python
def render_search_bar() -> Optional[str]:
    st.markdown(_get_klein_blue_landing_css(), unsafe_allow_html=True)
    st.markdown("""
    <div class="klein-landing">
        <h1 class="klein-headline">
            Redesign <span class="accent">any data</span><br>for your intent
        </h1>
        <p class="klein-tagline">Search French open data from data.gouv.fr</p>
    </div>
    """, unsafe_allow_html=True)
```

**Acceptance Criteria**:
- [x] "Redesign any data for your intent" headline displayed
- [x] Search bar integrated into landing hero
- [x] No extra headers visible on landing page
- [x] Accent color on "any data" text

**Files**:
- `intuitiveness/ui/datagouv_search.py` (MODIFY)
- `intuitiveness/streamlit_app.py` (MODIFY - conditional header hiding)

---

### Task 8.3: Verify Landing Page with Playwright MCP

**User Story**: US5
**Priority**: P1
**Requirement**: SC-009

**Description**: Visual verification that landing page displays correctly.

**Test Cases**:
1. Navigate to app with no data loaded
2. Verify Klein Blue gradient background
3. Verify "Redesign any data for your intent" headline visible
4. Verify search bar is functional
5. Verify no redundant headers visible

**Acceptance Criteria**:
- [x] Landing page renders with Klein Blue theme
- [x] Search functionality works
- [x] No Streamlit default headers visible
- [x] Responsive at 768px width

**Files**: None (testing only)

---

## Phase 9: Sarazin & Mourey Tutorial

### Task 9.1: Create Tutorial Content Structure

**User Story**: US6 (Educational Onboarding)
**Priority**: P1
**Requirement**: FR-013

**Description**: Define the tutorial content based on Sarazin & Mourey's 5-level abstraction framework. Tutorial explains the descent-ascent methodology.

**Content Outline**:
1. **Introduction**: "Transform chaos into clarity"
2. **The 5 Levels of Abstraction**:
   - L4: Raw Dataset (files, columns, rows)
   - L3: Entity Graph (entities + relationships)
   - L2: Domain Categories (grouped meanings)
   - L1: Unified Vector (single dimension)
   - L0: Core Datum (one truth value)
3. **The Descent**: Sanitizing data by stripping dimensions
4. **The Ascent**: Rebuilding with intentional dimension selection
5. **Your Intent**: What question will you answer?

**Acceptance Criteria**:
- [x] Tutorial content written following research paper
- [x] Content split into digestible steps (max 5)
- [x] Each step explains one concept clearly

**Files**: `intuitiveness/ui/tutorial.py` (CREATE)

---

### Task 9.2: Create Tutorial UI Component

**User Story**: US6
**Priority**: P1
**Requirement**: FR-014

**Description**: Build a Streamlit component that renders the tutorial as interactive slides/cards.

**Implementation**:
```python
def render_tutorial() -> bool:
    """
    Render the Sarazin & Mourey method tutorial.

    Returns:
        True when user completes tutorial, False otherwise.
    """
    step = st.session_state.get('tutorial_step', 0)

    # Render current tutorial step
    _render_tutorial_step(step)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if step > 0:
            if st.button("← Back"):
                st.session_state.tutorial_step = step - 1
    with col2:
        if step < TUTORIAL_STEPS - 1:
            if st.button("Next →"):
                st.session_state.tutorial_step = step + 1
        else:
            if st.button("Start Redesigning →"):
                return True

    return False
```

**Acceptance Criteria**:
- [x] Tutorial displays as step-by-step cards
- [x] User can navigate forward/backward
- [x] Final step has "Start Redesigning" CTA
- [x] Uses Klein Blue accent colors

**Files**: `intuitiveness/ui/tutorial.py` (CREATE)

---

### Task 9.3: Add Tutorial Visual Assets

**User Story**: US6
**Priority**: P2
**Requirement**: FR-015

**Description**: Create minimal visual diagrams showing the 5 levels and descent-ascent flow using HTML/CSS (no external images).

**Implementation**:
- L4→L0 descent diagram (vertical flow)
- L0→L3 ascent diagram (vertical flow with branching)
- Level icons (circle badges with numbers)

**Acceptance Criteria**:
- [x] Descent diagram shows L4→L3→L2→L1→L0
- [x] Ascent diagram shows L0→L1→L2→L3
- [x] Diagrams use Klein Blue color palette
- [x] No external image dependencies

**Files**: `intuitiveness/ui/tutorial.py` (MODIFY)

---

### Task 9.4: Tutorial Session State Management

**User Story**: US6
**Priority**: P1
**Requirement**: FR-016

**Description**: Track tutorial completion state so users only see tutorial once (unless they reset).

**Implementation**:
```python
SESSION_KEY_TUTORIAL_COMPLETED = 'tutorial_completed'
SESSION_KEY_TUTORIAL_STEP = 'tutorial_step'

def is_tutorial_completed() -> bool:
    return st.session_state.get(SESSION_KEY_TUTORIAL_COMPLETED, False)

def mark_tutorial_completed():
    st.session_state[SESSION_KEY_TUTORIAL_COMPLETED] = True
```

**Acceptance Criteria**:
- [x] Tutorial completion persisted in session state
- [x] Users skip tutorial on subsequent visits (within session)
- [x] "Skip Tutorial" option available
- [x] Reset function clears tutorial state

**Files**: `intuitiveness/ui/tutorial.py` (MODIFY)

---

## Phase 10: Flow Integration

### Task 10.1: Implement Landing → Tutorial → Workflow Flow

**User Story**: US5, US6
**Priority**: P1
**Requirement**: FR-017

**Description**: Wire up the complete user flow from landing page through tutorial to main workflow.

**Flow Logic**:
```
1. User arrives → Show Landing Page (search)
2. User selects dataset → Check tutorial completion
3. If tutorial not completed → Show Tutorial
4. After tutorial → Start Descent workflow
5. Continue with existing ascent/descent logic
```

**Implementation**:
```python
def render_upload_step():
    # Show landing page with search
    selected_resource = render_search_interface()

    if selected_resource:
        # Load dataset
        df = load_resource(selected_resource)
        st.session_state.raw_data = df

        # Check if tutorial needed
        if not is_tutorial_completed():
            st.session_state.current_step = 'tutorial'
        else:
            st.session_state.current_step = 1  # Go to descent
```

**Acceptance Criteria**:
- [x] Flow transitions smoothly between phases
- [x] Tutorial appears after first dataset selection
- [x] Workflow resumes correctly after tutorial
- [x] Back navigation works through flow

**Files**: `intuitiveness/streamlit_app.py` (MODIFY)

---

### Task 10.2: Add Tutorial Skip/Replay Options

**User Story**: US6
**Priority**: P2
**Requirement**: FR-018

**Description**: Allow users to skip tutorial (experienced users) or replay it later.

**Implementation**:
- "Skip Tutorial" link on tutorial screen
- "View Tutorial" option in sidebar or settings
- Tutorial replay resets to step 0

**Acceptance Criteria**:
- [x] Skip option visible on tutorial screens
- [x] Replay option accessible from sidebar
- [x] Skip/Replay properly updates session state

**Files**:
- `intuitiveness/ui/tutorial.py` (MODIFY)
- `intuitiveness/streamlit_app.py` (MODIFY)

---

### Task 10.3: End-to-End Flow Testing with Playwright

**User Story**: US5, US6
**Priority**: P1
**Requirement**: SC-010

**Description**: Complete end-to-end test of the new user flow using Playwright MCP.

**Test Scenarios**:
1. **First Visit Flow**:
   - Navigate to app → See landing page
   - Search for dataset → Select one
   - See tutorial → Navigate through all steps
   - Complete tutorial → Land on descent workflow

2. **Return Visit Flow**:
   - Navigate to app → See landing page
   - Search and select dataset
   - Skip directly to descent (no tutorial)

3. **Tutorial Replay**:
   - Access sidebar option
   - See tutorial from step 1
   - Complete or skip

**Acceptance Criteria**:
- [x] First visit flow completes successfully
- [x] Return visit skips tutorial
- [x] Tutorial replay works correctly
- [x] All transitions are smooth

**Files**: None (testing only)

---

## Task Dependencies Graph

```
Phase 1: Setup
├── Task 1.1: config.toml
└── Task 1.2: Directory structure
         │
         ▼
Phase 2: Foundational
├── Task 2.1: palette.py
├── Task 2.2: typography.py
└── Task 2.3: chrome.py
         │
         ▼
Phase 3: US1 Integration
├── Task 3.1: __init__.py
├── Task 3.2: streamlit_app.py integration
└── Task 3.3: Verify US1
         │
         ├─────────────┬─────────────┬─────────────┐
         ▼             ▼             ▼             ▼
Phase 4: US2    Phase 5: US3   Phase 6: US4   Phase 8: Landing
Progress        Metric Cards   Components     Klein Blue
         │             │             │             │
         └─────────────┴─────────────┘             │
                       │                           │
                       ▼                           ▼
         Phase 7: Integration          Phase 9: Tutorial
         & Polish                      Sarazin & Mourey
                                              │
                                              ▼
                                      Phase 10: Flow
                                      Integration
```

---

## Estimated Effort

| Phase | Estimated Time | Status |
|-------|----------------|--------|
| Phase 1 | 15 min | ✅ Complete |
| Phase 2 | 45 min | ✅ Complete |
| Phase 3 | 30 min | ✅ Complete |
| Phase 4 | 45 min | ✅ Complete |
| Phase 5 | 30 min | ✅ Complete |
| Phase 6 | 20 min | ✅ Complete |
| Phase 7 | 45 min | ✅ Complete |
| **Phase 8** | **45 min** | **✅ Complete** |
| **Phase 9** | **90 min** | **✅ Complete** |
| **Phase 10** | **60 min** | **✅ Complete** |
| **Total** | **~7.5 hours** | |

---

## Success Criteria Mapping

| Criteria | Tasks |
|----------|-------|
| SC-001: No hamburger menu | Task 2.3, 3.3 |
| SC-002: No footer | Task 2.3, 3.3 |
| SC-003: IBM Plex Sans | Task 2.2, 3.3 |
| SC-004: Progress compact | Task 4.1, 4.2 |
| SC-005: Metric cards | Task 5.1, 5.2 |
| SC-006: Consistent buttons | Task 6.1, 6.2 |
| SC-007: Responsive 768px | Task 7.2 |
| SC-008: Load < 3s | Task 7.3 |
| **SC-009: Klein Blue landing** | **Task 8.1, 8.2, 8.3** |
| **SC-010: Tutorial completion** | **Task 9.1, 9.2, 9.4** |
| **SC-011: Flow integration** | **Task 10.1, 10.3** |
