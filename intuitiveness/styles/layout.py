"""
CSS for SaaS-ready layout components.

Provides card containers, section styling, and layout utilities
that replace raw Streamlit content blocks with professional
SaaS-style visual hierarchy.

Following Gael Penessot's DataGyver philosophy.
"""

LAYOUT_CSS = """
/* ==============================================
   CARD SYSTEM - Content Containers
   Replace raw content blocks with elevated cards
   ============================================== */

/* Base content card */
.content-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0, 47, 167, 0.08);
    margin-bottom: 16px;
}

/* Card with hover interaction */
.content-card-interactive {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0, 47, 167, 0.08);
    margin-bottom: 16px;
    transition: all 0.2s ease;
    cursor: pointer;
}

.content-card-interactive:hover {
    box-shadow: 0 4px 12px rgba(0, 47, 167, 0.12);
    transform: translateY(-2px);
}

/* Compact card variant */
.content-card-compact {
    background: #ffffff;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 2px rgba(0, 47, 167, 0.06);
    margin-bottom: 12px;
}

/* Card with accent border */
.content-card-accent {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0, 47, 167, 0.08);
    margin-bottom: 16px;
    border-left: 3px solid var(--color-accent, #002fa7);
}

/* ==============================================
   SECTION HEADERS - Replace st.header
   Clean typography with subtle accents
   ============================================== */

/* Section header with bottom accent */
.section-header {
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--color-text-primary, #0f172a);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--color-accent-light, #f0f2fa);
}

/* Section header with Klein Blue accent dot */
.section-header-dot {
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--color-text-primary, #0f172a);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-header-dot::before {
    content: '';
    width: 8px;
    height: 8px;
    background: var(--color-accent, #002fa7);
    border-radius: 50%;
}

/* ==============================================
   PAGE HEADER - Replace st.title
   Professional SaaS-style page headers
   ============================================== */

.page-header {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--color-border, #e2e8f0);
}

.page-header-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--color-text-primary, #0f172a);
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}

.page-header-subtitle {
    font-size: 1rem;
    color: var(--color-text-secondary, #475569);
    margin: 0;
    font-weight: 400;
}

/* Page header with Klein Blue gradient accent */
.page-header-accent {
    margin-bottom: 24px;
    padding: 20px 0;
    position: relative;
}

.page-header-accent::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 60px;
    height: 3px;
    background: linear-gradient(to right, var(--color-accent, #002fa7), transparent);
    border-radius: 2px;
}

/* ==============================================
   BREADCRUMB - Navigation trail
   ============================================== */

.breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.875rem;
    color: var(--color-text-muted, #94a3b8);
    margin-bottom: 8px;
}

.breadcrumb-item {
    color: var(--color-text-secondary, #475569);
}

.breadcrumb-item-active {
    color: var(--color-accent, #002fa7);
    font-weight: 500;
}

.breadcrumb-separator {
    color: var(--color-text-muted, #94a3b8);
}

/* ==============================================
   UTILITY CLASSES
   ============================================== */

/* Subtle background for content areas */
.bg-subtle {
    background: var(--color-bg-primary, #f8fafc);
    border-radius: 8px;
    padding: 16px;
}

/* Klein Blue accent text */
.text-accent {
    color: var(--color-accent, #002fa7) !important;
}

/* Flex row with gap */
.flex-row {
    display: flex;
    align-items: center;
    gap: 16px;
}

/* Flex column with gap */
.flex-col {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

/* Spacing utilities */
.mt-0 { margin-top: 0 !important; }
.mt-1 { margin-top: 8px !important; }
.mt-2 { margin-top: 16px !important; }
.mt-3 { margin-top: 24px !important; }
.mb-0 { margin-bottom: 0 !important; }
.mb-1 { margin-bottom: 8px !important; }
.mb-2 { margin-bottom: 16px !important; }
.mb-3 { margin-bottom: 24px !important; }
"""
