"""
CSS for Streamlit sidebar customization.

SaaS-ready sidebar design: dissolves the typical Streamlit barrier
by using white background with subtle Klein Blue shadow instead of
gray background with hard border.

Following Gael Penessot's DataGyver philosophy.
"""

SIDEBAR_CSS = """
/* ==============================================
   SIDEBAR - SAAS-READY DESIGN
   White background + shadow replaces gray + border
   Creates seamless integration with main content
   ============================================== */

/* Sidebar container - dissolve the barrier */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(0, 47, 167, 0.06) !important;
}

/* Sidebar inner content area */
[data-testid="stSidebarContent"],
[data-testid="stSidebar"] > div:first-child {
    background: #ffffff !important;
    padding-top: 1.5rem !important;
}

/* Remove any default Streamlit gradients/overlays */
[data-testid="stSidebar"]::before,
[data-testid="stSidebar"]::after {
    display: none !important;
}

/* Sidebar section headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--color-text-primary, #0f172a) !important;
    font-weight: 600 !important;
    margin-bottom: 0.75rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid var(--color-border, #e2e8f0) !important;
}

/* Sidebar text styling */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--color-text-secondary, #475569) !important;
}

/* Sidebar widgets - minimal styling, no extra cards */
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stMultiSelect,
[data-testid="stSidebar"] .stRadio,
[data-testid="stSidebar"] .stCheckbox {
    background: transparent !important;
    padding: 0.5rem 0 !important;
    border-radius: 0 !important;
    border: none !important;
    margin-bottom: 0.5rem !important;
}

/* Sidebar buttons - full width, clean style */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    justify-content: center !important;
}

/* Sidebar dividers - ultra subtle gradient */
[data-testid="stSidebar"] hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(to right, transparent, var(--color-border, #e2e8f0), transparent) !important;
    margin: 1.25rem 0 !important;
}

/* Navigation-like items in sidebar */
[data-testid="stSidebar"] .row-widget {
    margin-bottom: 0.5rem !important;
}

/* Sidebar expanders - cleaner look */
[data-testid="stSidebar"] .stExpander {
    background: var(--color-bg-elevated, #ffffff) !important;
    border-color: var(--color-border, #e2e8f0) !important;
}

/* Custom navigation item styling (if using markdown links) */
[data-testid="stSidebar"] a {
    color: var(--color-text-primary, #0f172a) !important;
    text-decoration: none !important;
    display: block !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 0.375rem !important;
    transition: all 0.15s ease !important;
}

[data-testid="stSidebar"] a:hover {
    background: var(--color-accent-light, #f0f2fa) !important;
    color: var(--color-accent, #002fa7) !important;
}

/* Progress indicator in sidebar */
[data-testid="stSidebar"] .progress-indicator {
    background: var(--color-bg-elevated, #ffffff) !important;
    padding: 1rem !important;
    border-radius: 0.5rem !important;
    border: 1px solid var(--color-border, #e2e8f0) !important;
}

/* File uploader in sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: var(--color-bg-elevated, #ffffff) !important;
    border: 2px dashed var(--color-border, #e2e8f0) !important;
    border-radius: 0.5rem !important;
    padding: 1rem !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: var(--color-accent, #002fa7) !important;
}

/* Sidebar bottom spacing */
[data-testid="stSidebar"] > div:first-child > div:last-child {
    padding-bottom: 2rem !important;
}
"""
