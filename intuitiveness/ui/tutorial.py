"""
Research Paper Viewer - Dialog Modal
=====================================

Clean modal dialog showing the Sarazin & Mourey research paper
using streamlit-extras pdf_viewer.

Feature: 007-streamlit-design-makeup
Reference: Intuitiveness methodology paper
"""

import streamlit as st
from pathlib import Path
from streamlit_extras.pdf_viewer import pdf_viewer


# =============================================================================
# Configuration
# =============================================================================

# Path to the research paper - use multiple fallback paths for different environments
def _find_paper_path() -> Path:
    """Find paper path across local and Streamlit Cloud environments."""
    candidates = [
        # Streamlit Cloud: runs from repo root
        Path.cwd() / "scientific_article" / "Intuitiveness.pdf",
        # Local development: relative to this file
        Path(__file__).parent.parent.parent / "scientific_article" / "Intuitiveness.pdf",
        # Alternative: absolute path resolution
        Path(__file__).resolve().parent.parent.parent / "scientific_article" / "Intuitiveness.pdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Return first candidate for error message
    return candidates[0]

PAPER_PATH = _find_paper_path()

# Session state keys
SESSION_KEY_TUTORIAL_COMPLETED = 'tutorial_completed'
SESSION_KEY_SHOW_TUTORIAL = 'show_tutorial'

# Keep old keys for backwards compatibility
SESSION_KEY_TUTORIAL_STEP = 'tutorial_step'
TUTORIAL_STEPS = 1  # Single step now


# =============================================================================
# State Management
# =============================================================================

def is_tutorial_completed() -> bool:
    """Check if user has viewed the paper."""
    return st.session_state.get(SESSION_KEY_TUTORIAL_COMPLETED, False)


def mark_tutorial_completed():
    """Mark paper as viewed."""
    st.session_state[SESSION_KEY_TUTORIAL_COMPLETED] = True
    st.session_state[SESSION_KEY_SHOW_TUTORIAL] = False


def skip_tutorial():
    """Skip viewing the paper."""
    mark_tutorial_completed()


def reset_tutorial():
    """Show paper again."""
    st.session_state[SESSION_KEY_TUTORIAL_COMPLETED] = False
    st.session_state[SESSION_KEY_SHOW_TUTORIAL] = True


def should_show_tutorial() -> bool:
    """Check if paper should be displayed."""
    if is_tutorial_completed():
        return False
    return st.session_state.get(SESSION_KEY_SHOW_TUTORIAL, True)


# =============================================================================
# Main Paper Dialog
# =============================================================================

@st.dialog("The Intuitiveness Method", width="large")
def show_tutorial_dialog():
    """
    Display the research paper in a modal dialog.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap');
    .paper-header {
        font-family: 'Outfit', sans-serif;
        text-align: center;
        margin-bottom: 1rem;
    }
    .paper-header h3 {
        color: #002fa7;
        margin: 0;
    }
    .paper-header p {
        color: #64748b;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
    </style>
    <div class="paper-header">
        <h3>Sarazin & Mourey Research Paper</h3>
        <p>Understanding the methodology behind intuitive data redesign</p>
    </div>
    """, unsafe_allow_html=True)

    # Debug info (collapsible)
    with st.expander("Debug Info", expanded=False):
        st.code(f"""
PAPER_PATH: {PAPER_PATH}
Exists: {PAPER_PATH.exists()}
CWD: {Path.cwd()}
__file__: {Path(__file__)}
File size: {PAPER_PATH.stat().st_size if PAPER_PATH.exists() else 'N/A'} bytes
        """)

    # Load and display PDF
    if PAPER_PATH.exists():
        try:
            pdf_bytes = PAPER_PATH.read_bytes()
            st.caption(f"Loaded {len(pdf_bytes)} bytes")
            pdf_viewer(pdf_bytes, height=500)
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            st.exception(e)
    else:
        st.error(f"Paper not found. Looking in: {PAPER_PATH}")
        st.caption(f"Current working directory: {Path.cwd()}")
        st.caption(f"File location: {Path(__file__).parent}")

    st.markdown("")

    # Close button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Redesigning", type="primary", use_container_width=True):
            mark_tutorial_completed()
            st.rerun()


def render_tutorial(on_complete=None) -> bool:
    """
    Render the paper dialog if needed.

    Returns:
        True if paper was viewed/skipped, False if showing.
    """
    if is_tutorial_completed():
        return True

    if should_show_tutorial():
        show_tutorial_dialog()
        return False

    return True


def render_tutorial_replay_button():
    """Render a button to view the paper again (for sidebar)."""
    if st.button("📄 View Paper", key="replay_tutorial", use_container_width=True):
        reset_tutorial()
        st.rerun()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'render_tutorial',
    'show_tutorial_dialog',
    'render_tutorial_replay_button',
    'is_tutorial_completed',
    'mark_tutorial_completed',
    'skip_tutorial',
    'reset_tutorial',
    'should_show_tutorial',
    'SESSION_KEY_TUTORIAL_COMPLETED',
    'SESSION_KEY_TUTORIAL_STEP',
    'SESSION_KEY_SHOW_TUTORIAL',
    'TUTORIAL_STEPS',
]
