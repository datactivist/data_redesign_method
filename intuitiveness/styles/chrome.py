"""
CSS to hide default Streamlit UI elements.

Comprehensive hiding of Streamlit chrome following Gael Penessot's DataGyver philosophy.
Goal: Make the app look like a custom web application, not a Streamlit app.
"""

HIDE_CHROME_CSS = """
/* ==============================================
   STREAMLIT CHROME HIDING - COMPLETE REMOVAL
   Goal: Zero Streamlit appearance, pure custom app
   ============================================== */

/* CRITICAL: Hide ALL header elements completely */
#MainMenu,
header,
.stApp > header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
.stToolbar,
.stDeployButton,
[data-testid="stDeployButton"],
button[data-testid="baseButton-header"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Hide footer completely - no "Made with Streamlit" */
footer,
.stApp > footer,
[data-testid="stFooter"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Hide ALL badges and branding */
.viewerBadge_container__r5tak,
.viewerBadge_link__qRIco,
[data-testid="stStreamlitBadge"],
.stAppViewBlockContainer [data-testid="stDecoration"],
[data-testid="stDecoration"] {
    display: none !important;
}

/* Hide sidebar collapse control */
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Hide fullscreen buttons */
button[title="View fullscreen"],
[data-testid="StyledFullScreenButton"] {
    display: none !important;
}

/* Hide running indicator */
[data-testid="stStatusWidget"],
.stStatusWidget {
    display: none !important;
}

/* Hide page title that appears during rerun */
.stApp [data-testid="stAppViewContainer"] > div:first-child > div:first-child {
    display: none !important;
}

/* ==============================================
   LAYOUT ADJUSTMENTS - SAAS OPTIMIZED
   Tighter spacing, wider content area
   ============================================== */

/* Adjust main content area padding - SaaS compact */
/* Right padding increased to avoid overlap with fixed right progress sidebar */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 1400px !important;
    padding-left: 2rem !important;
    padding-right: 5rem !important;
}

/* Clean up app header spacing */
.stApp > header {
    height: 0 !important;
    min-height: 0 !important;
}

/* Ensure main content doesn't have top gap */
.main .block-container {
    padding-top: 0.75rem !important;
}

/* ==============================================
   STREAMLIT ELEMENT REFINEMENTS
   ============================================== */

/* Softer dividers - not the harsh Streamlit default */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--color-border, #e2e8f0) !important;
    margin: 1.5rem 0 !important;
}

/* Remove default Streamlit element margins that feel "off" */
.stMarkdown {
    margin-bottom: 0 !important;
}

/* Smoother transitions for all interactive elements */
button, input, select, textarea, .stExpander {
    transition: all 0.2s ease !important;
}

/* ==============================================
   SAAS REFINEMENTS - Hide Default Dividers
   ============================================== */

/* Streamlit's default st.divider() - make it ultra subtle */
[data-testid="stHorizontalBlock"] hr,
.stDivider hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(to right, transparent, var(--color-border, #e2e8f0), transparent) !important;
    margin: 1.5rem 0 !important;
    opacity: 0.6 !important;
}

/* Custom section separator - gradient fade */
.section-separator {
    height: 1px;
    background: linear-gradient(to right, transparent, var(--color-border, #e2e8f0), transparent);
    margin: 2rem 0;
}

/* Remove excessive vertical gaps between elements */
.stMarkdown + .stMarkdown {
    margin-top: 0 !important;
}

/* Reduce header spacing */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    margin-top: 1rem !important;
    margin-bottom: 0.75rem !important;
}
"""
