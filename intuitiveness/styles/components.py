"""
CSS for Streamlit interactive components.

Styles buttons, expanders, inputs for visual consistency.
Deep customization following Gael Penessot's DataGyver philosophy.
"""

COMPONENT_CSS = """
/* ==============================================
   BUTTON STYLING - Modern, Non-Streamlit Feel
   ============================================== */

/* Primary buttons - solid accent color */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: var(--color-accent, #3b82f6) !important;
    border: none !important;
    font-weight: 500 !important;
    border-radius: 0.375rem !important;
    color: white !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: var(--color-accent-hover, #2563eb) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25) !important;
}

.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="baseButton-primary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

/* Secondary buttons - transparent with border */
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background: var(--color-bg-elevated, #ffffff) !important;
    border: 1px solid var(--color-border, #e2e8f0) !important;
    border-radius: 0.375rem !important;
    color: var(--color-text-primary, #0f172a) !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
    background: var(--color-bg-secondary, #f1f5f9) !important;
    border-color: var(--color-accent, #3b82f6) !important;
    color: var(--color-accent, #3b82f6) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):active {
    transform: translateY(0) !important;
}

/* Disabled button state */
.stButton > button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ==============================================
   EXPANDER/ACCORDION STYLING
   ============================================== */

/* Expander container - clean card look */
.stExpander {
    border: 1px solid var(--color-border, #e2e8f0) !important;
    border-radius: 0.5rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    background: var(--color-bg-elevated, #ffffff) !important;
    overflow: hidden !important;
}

/* Expander header - remove default Streamlit styling */
.stExpander [data-testid="stExpanderToggleIcon"] {
    /* Hide the default Streamlit arrow */
    display: none !important;
}

.stExpander summary {
    padding: 0.875rem 1rem !important;
    font-weight: 500 !important;
    color: var(--color-text-primary, #0f172a) !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    transition: background 0.15s ease !important;
}

.stExpander summary:hover {
    background: var(--color-bg-secondary, #f1f5f9) !important;
}

/* Custom chevron icon using CSS ::after */
.stExpander summary::after {
    content: "\\203A" !important;  /* Right angle bracket */
    font-size: 1.25rem !important;
    font-weight: 400 !important;
    color: var(--color-text-muted, #94a3b8) !important;
    transform: rotate(90deg) !important;
    transition: transform 0.2s ease !important;
    margin-left: auto !important;
}

.stExpander[open] summary::after {
    transform: rotate(-90deg) !important;
}

/* Expander content area */
.stExpander [data-testid="stExpanderDetails"] {
    padding: 0 1rem 1rem 1rem !important;
    border-top: 1px solid var(--color-border, #e2e8f0) !important;
}

/* Remove default focus outline, add subtle one */
.stExpander summary:focus {
    outline: none !important;
    box-shadow: inset 0 0 0 2px var(--color-accent-light, #eff6ff) !important;
}

/* Text inputs and select boxes */
.stTextInput input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    border-radius: 0.5rem !important;
    border-color: var(--color-border, #e7e5e4) !important;
}

.stTextInput input:focus,
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--color-accent, #2563eb) !important;
    box-shadow: 0 0 0 1px var(--color-accent, #2563eb) !important;
}

/* Radio buttons and checkboxes */
.stRadio > div,
.stCheckbox > div {
    gap: 0.5rem;
}

/* Alerts and info boxes */
.stAlert {
    border-radius: 0.5rem !important;
    border-left-width: 3px !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 0.5rem 0.5rem 0 0 !important;
    font-weight: 500;
}

/* Dividers */
.stDivider {
    border-color: var(--color-border, #e7e5e4) !important;
}

/* Sidebar styling */
.stSidebar [data-testid="stSidebarContent"] {
    background: var(--color-bg-secondary, #f5f5f4);
}

/* Code blocks */
.stCodeBlock {
    border-radius: 0.5rem !important;
}

/* DataFrames */
.stDataFrame {
    border-radius: 0.5rem !important;
}
"""
