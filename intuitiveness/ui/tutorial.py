"""
Research Paper Viewer - Dialog Modal
=====================================

Clean modal dialog showing the Sarazin & Mourey research paper
using Mozilla PDF.js viewer (Chrome-compatible).

Feature: 007-streamlit-design-makeup
Reference: Intuitiveness methodology paper
"""

import streamlit as st
from pathlib import Path
from urllib.parse import quote
from intuitiveness.ui.i18n import t


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

# GitHub raw URL for the PDF (used for Cloud deployment)
GITHUB_PDF_RAW_URL = "https://raw.githubusercontent.com/ArthurSrz/intuitiveness/main/scientific_article/Intuitiveness.pdf"

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
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap');
    .paper-header {{
        font-family: 'Outfit', sans-serif;
        text-align: center;
        margin-bottom: 1rem;
    }}
    .paper-header p {{
        color: #64748b;
        font-size: 0.95rem;
        margin: 0;
    }}
    </style>
    <div class="paper-header">
        <p>{t("tutorial_description")}</p>
    </div>
    """, unsafe_allow_html=True)

    # Display PDF using Mozilla PDF.js viewer (Chrome-compatible)
    # This avoids Chrome blocking base64 data URIs in iframes
    encoded_url = quote(GITHUB_PDF_RAW_URL, safe='')
    pdfjs_viewer_url = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={encoded_url}"

    pdf_display = f'''
        <iframe
            src="{pdfjs_viewer_url}"
            width="100%"
            height="600px"
            style="border: 1px solid #e2e8f0; border-radius: 8px;"
        ></iframe>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)

    # Download button (use local file if available, otherwise direct GitHub link)
    if PAPER_PATH.exists():
        pdf_bytes = PAPER_PATH.read_bytes()
        st.download_button(
            label=f"📥 {t('download_pdf')}",
            data=pdf_bytes,
            file_name="Intuitiveness_Sarazin_Mourey.pdf",
            mime="application/pdf",
        )
    else:
        # Fallback: direct link to GitHub
        st.markdown(f"[📥 {t('download_pdf')}]({GITHUB_PDF_RAW_URL})")

    st.markdown("")

    # Close button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(t("start_redesigning"), type="primary", use_container_width=True):
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
    if st.button(f"📄 {t('view_paper')}", key="replay_tutorial", use_container_width=True):
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
