"""
CSS for custom alert components.

Replaces Streamlit's distinctive st.info/warning/success/error with
subtle, design-system-aligned alerts following Gael Penessot's approach.
"""

ALERTS_CSS = """
/* ==============================================
   CUSTOM ALERT STYLES
   Subtle left-border design, not full-colored backgrounds
   ============================================== */

/* ==============================================
   RESTYLE NATIVE STREAMLIT ALERTS
   Make st.info/warning/success/error look custom
   ============================================== */

/* Base styling for all native Streamlit alerts */
.stAlert {
    border-radius: 0.5rem !important;
    border: none !important;
    border-left: 3px solid !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    padding: 1rem 1.25rem !important;
    margin: 1rem 0 !important;
}

/* Info alerts - blue accent */
.stAlert[data-baseweb="notification"][kind="info"],
div[data-testid="stNotification"][data-type="info"] {
    border-left-color: var(--color-accent, #3b82f6) !important;
    background: linear-gradient(to right, rgba(59, 130, 246, 0.05), var(--color-bg-elevated, #ffffff)) !important;
}

/* Success alerts - green accent */
.stAlert[data-baseweb="notification"][kind="positive"],
div[data-testid="stNotification"][data-type="success"] {
    border-left-color: var(--color-success, #22c55e) !important;
    background: linear-gradient(to right, rgba(34, 197, 94, 0.05), var(--color-bg-elevated, #ffffff)) !important;
}

/* Warning alerts - amber accent */
.stAlert[data-baseweb="notification"][kind="warning"],
div[data-testid="stNotification"][data-type="warning"] {
    border-left-color: var(--color-warning, #f59e0b) !important;
    background: linear-gradient(to right, rgba(245, 158, 11, 0.05), var(--color-bg-elevated, #ffffff)) !important;
}

/* Error alerts - red accent */
.stAlert[data-baseweb="notification"][kind="negative"],
div[data-testid="stNotification"][data-type="error"] {
    border-left-color: var(--color-error, #ef4444) !important;
    background: linear-gradient(to right, rgba(239, 68, 68, 0.05), var(--color-bg-elevated, #ffffff)) !important;
}

/* Alert content text styling */
.stAlert p,
.stAlert div {
    color: var(--color-text-primary, #0f172a) !important;
    font-size: 0.9375rem !important;
}

/* Hide the default Streamlit alert icons if we want cleaner look */
/* Uncomment if you want to remove default icons:
.stAlert svg {
    display: none !important;
}
*/

/* Base alert style */
.custom-alert {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
    border-radius: 0.5rem;
    border-left: 3px solid;
    margin: 1rem 0;
    background: var(--color-bg-elevated, #ffffff);
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.custom-alert-icon {
    font-size: 1.25rem;
    flex-shrink: 0;
    line-height: 1.4;
}

.custom-alert-content {
    flex: 1;
    font-size: 0.9375rem;
    line-height: 1.5;
    color: var(--color-text-primary, #0f172a);
}

.custom-alert-title {
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.custom-alert-message {
    color: var(--color-text-secondary, #475569);
}

/* Info variant - blue accent */
.custom-alert-info {
    border-left-color: var(--color-accent, #3b82f6);
    background: linear-gradient(to right, rgba(59, 130, 246, 0.05), var(--color-bg-elevated, #ffffff));
}

.custom-alert-info .custom-alert-icon {
    color: var(--color-accent, #3b82f6);
}

/* Success variant - green accent */
.custom-alert-success {
    border-left-color: var(--color-success, #22c55e);
    background: linear-gradient(to right, rgba(34, 197, 94, 0.05), var(--color-bg-elevated, #ffffff));
}

.custom-alert-success .custom-alert-icon {
    color: var(--color-success, #22c55e);
}

/* Warning variant - amber accent */
.custom-alert-warning {
    border-left-color: var(--color-warning, #f59e0b);
    background: linear-gradient(to right, rgba(245, 158, 11, 0.05), var(--color-bg-elevated, #ffffff));
}

.custom-alert-warning .custom-alert-icon {
    color: var(--color-warning, #f59e0b);
}

/* Error variant - red accent */
.custom-alert-error {
    border-left-color: var(--color-error, #ef4444);
    background: linear-gradient(to right, rgba(239, 68, 68, 0.05), var(--color-bg-elevated, #ffffff));
}

.custom-alert-error .custom-alert-icon {
    color: var(--color-error, #ef4444);
}

/* Tip variant - purple accent (bonus) */
.custom-alert-tip {
    border-left-color: #8b5cf6;
    background: linear-gradient(to right, rgba(139, 92, 246, 0.05), var(--color-bg-elevated, #ffffff));
}

.custom-alert-tip .custom-alert-icon {
    color: #8b5cf6;
}
"""
