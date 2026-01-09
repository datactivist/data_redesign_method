"""
Intuitiveness App Module

Spec-aligned modular structure for the Streamlit application.
Each module maps to specific specs for clear traceability.

Phase 1 - Code Simplification (011-code-simplification)
Created: 2026-01-09

Module Structure:
-----------------
- main.py             : App orchestration (~300 lines)
- initialization.py   : Session init, recovery (005-session-persistence)
- guided_mode.py      : Guided descent/ascent workflow (001-004)
- free_mode.py        : Free navigation with decision tree (002-003)
- sidebar.py          : Sidebar rendering components
- ascent_controller.py: Unified ascent logic for both modes (002-004)

Spec Traceability:
------------------
| Module              | Specs      | Key User Stories                    |
|---------------------|------------|-------------------------------------|
| guided_mode.py      | 001-004    | Descent wizard, ascent forms        |
| free_mode.py        | 002-003    | US-5 (Navigate Dataset Hierarchy)   |
| initialization.py   | 005        | Session persistence, recovery       |
| sidebar.py          | 005-010    | All sidebar components              |
"""

from intuitiveness.app.main import run_app
from intuitiveness.app.initialization import (
    init_app_config,
    init_styles,
    init_session_state,
    handle_mode_switching,
    handle_session_recovery,
    is_pure_landing_page,
    run_initialization,
)
from intuitiveness.app.guided_mode import (
    DESCENT_STEPS,
    ASCENT_STEPS,
    STEPS,
    render_guided_content,
    get_current_step,
    set_current_step,
    advance_step,
    go_back_step,
    is_descent_complete,
    get_current_level,
)
from intuitiveness.app.free_mode import (
    render_free_content,
    is_free_mode,
    switch_to_free_mode,
    switch_to_guided_mode,
    has_nav_session,
    get_nav_session,
    request_descend,
    request_ascend,
)
from intuitiveness.app.sidebar import render_sidebar
from intuitiveness.app.ascent_controller import (
    AscentController,
    AscentResult,
    AscentOutcome,
    get_ascent_controller,
)

__all__ = [
    # Main entry point
    'run_app',

    # Initialization (005)
    'init_app_config',
    'init_styles',
    'init_session_state',
    'handle_mode_switching',
    'handle_session_recovery',
    'is_pure_landing_page',
    'run_initialization',

    # Guided mode (001-004)
    'DESCENT_STEPS',
    'ASCENT_STEPS',
    'STEPS',
    'render_guided_content',
    'get_current_step',
    'set_current_step',
    'advance_step',
    'go_back_step',
    'is_descent_complete',
    'get_current_level',

    # Free mode (002-003)
    'render_free_content',
    'is_free_mode',
    'switch_to_free_mode',
    'switch_to_guided_mode',
    'has_nav_session',
    'get_nav_session',
    'request_descend',
    'request_ascend',

    # Sidebar
    'render_sidebar',

    # Ascent Controller (002-004)
    'AscentController',
    'AscentResult',
    'AscentOutcome',
    'get_ascent_controller',
]
