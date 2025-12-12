"""
CSS for progress indicator styling.

Implements simplified text-based progress indicator.
Based on css-contracts.md PROGRESS_CSS specification.
"""

PROGRESS_CSS = """
/* Progress indicator container */
.progress-indicator {
    font-family: var(--font-family, 'IBM Plex Sans', sans-serif);
    padding: 1rem;
}

/* Level markers */
.progress-level {
    display: flex;
    align-items: center;
    padding: 0.5rem 0;
    color: var(--color-text-muted, #a8a29e);
    font-size: 0.875rem;
}

.progress-level.completed {
    color: var(--color-success, #22c55e);
    text-decoration: line-through;
}

.progress-level.current {
    color: var(--color-accent, #2563eb);
    font-weight: 500;
}

.progress-level.current::before {
    content: "→ ";
}

/* Compact progress text in sidebar */
.stSidebar .progress-text {
    font-size: 0.875rem;
    line-height: 1.5;
}

.stSidebar .progress-text .completed {
    color: var(--color-success, #22c55e);
}

.stSidebar .progress-text .current {
    color: var(--color-accent, #2563eb);
    font-weight: 500;
}

.stSidebar .progress-text .pending {
    color: var(--color-text-muted, #a8a29e);
}
"""
