"""
Sarazin & Mourey Method Tutorial Component
==========================================

Interactive tutorial introducing the 5-level abstraction framework
and descent-ascent methodology for intuitive dataset redesign.

Feature: 007-streamlit-design-makeup (Phase 9)
Based on: Sarazin & Mourey research paper on intuitive datasets
"""

import streamlit as st
from typing import Optional, Callable
from dataclasses import dataclass


# =============================================================================
# Session State Keys
# =============================================================================

SESSION_KEY_TUTORIAL_COMPLETED = 'tutorial_completed'
SESSION_KEY_TUTORIAL_STEP = 'tutorial_step'


# =============================================================================
# Tutorial Content
# =============================================================================

@dataclass
class TutorialStep:
    """A single step in the tutorial."""
    title: str
    subtitle: str
    content: str
    diagram_html: str


# The 5 tutorial steps based on Sarazin & Mourey methodology
TUTORIAL_STEPS = [
    TutorialStep(
        title="Transform Chaos into Clarity",
        subtitle="The problem with raw data",
        content="""
        Raw datasets are often **chaotic**: too many columns, unclear relationships,
        and no clear path to insight.

        The **Sarazin & Mourey method** transforms any dataset into one that
        directly answers your question—by first stripping it down to its essence,
        then rebuilding it with purpose.
        """,
        diagram_html="""
        <div class="tutorial-diagram intro-diagram">
            <div class="chaos-cloud">
                <span class="chaos-item">columns</span>
                <span class="chaos-item">rows</span>
                <span class="chaos-item">nulls</span>
                <span class="chaos-item">types</span>
                <span class="chaos-item">formats</span>
            </div>
            <div class="arrow-down"></div>
            <div class="clarity-result">
                <span class="clarity-icon">✦</span>
                <span>Your Answer</span>
            </div>
        </div>
        """
    ),
    TutorialStep(
        title="The 5 Levels of Abstraction",
        subtitle="From raw files to pure meaning",
        content="""
        Every dataset exists at one of **5 abstraction levels**:

        - **L4 — Raw Dataset**: Files, columns, rows as they arrive
        - **L3 — Entity Graph**: Things and their relationships
        - **L2 — Domain Categories**: Grouped by meaning
        - **L1 — Unified Vector**: One dimension of truth
        - **L0 — Core Datum**: A single, irreducible value
        """,
        diagram_html="""
        <div class="tutorial-diagram levels-diagram">
            <div class="level-stack">
                <div class="level-item l4"><span class="level-badge">L4</span> Raw Dataset</div>
                <div class="level-connector"></div>
                <div class="level-item l3"><span class="level-badge">L3</span> Entity Graph</div>
                <div class="level-connector"></div>
                <div class="level-item l2"><span class="level-badge">L2</span> Domain Categories</div>
                <div class="level-connector"></div>
                <div class="level-item l1"><span class="level-badge">L1</span> Unified Vector</div>
                <div class="level-connector"></div>
                <div class="level-item l0"><span class="level-badge">L0</span> Core Datum</div>
            </div>
        </div>
        """
    ),
    TutorialStep(
        title="The Descent",
        subtitle="Sanitizing by stripping dimensions",
        content="""
        **Descent** is the process of moving from L4 down to L0.

        At each level, you strip away one dimension of complexity:
        - Remove irrelevant columns
        - Collapse categories
        - Aggregate values

        When you reach **L0**, you have a single truth—the dataset's essence.
        This process **sanitizes** your data by removing noise.
        """,
        diagram_html="""
        <div class="tutorial-diagram descent-diagram">
            <div class="descent-flow">
                <div class="descent-level active">L4</div>
                <div class="descent-arrow">↓ <span class="action">identify entities</span></div>
                <div class="descent-level">L3</div>
                <div class="descent-arrow">↓ <span class="action">categorize</span></div>
                <div class="descent-level">L2</div>
                <div class="descent-arrow">↓ <span class="action">unify</span></div>
                <div class="descent-level">L1</div>
                <div class="descent-arrow">↓ <span class="action">reduce</span></div>
                <div class="descent-level final">L0</div>
            </div>
            <div class="descent-label">Sanitized Data</div>
        </div>
        """
    ),
    TutorialStep(
        title="The Ascent",
        subtitle="Rebuilding with intent",
        content="""
        **Ascent** is the reverse: moving from L0 back up to L3.

        But here's the key insight: you choose **which dimensions to add back**.
        Each choice is intentional, driven by the question you want to answer.

        The result? A dataset **tailored to your specific need**—not a generic
        dump of everything, but precisely what you need to find your answer.
        """,
        diagram_html="""
        <div class="tutorial-diagram ascent-diagram">
            <div class="ascent-flow">
                <div class="ascent-level start">L0</div>
                <div class="ascent-arrow">↑ <span class="action">unfold</span></div>
                <div class="ascent-level">L1</div>
                <div class="ascent-arrow">↑ <span class="action">add domain</span></div>
                <div class="ascent-level">L2</div>
                <div class="ascent-arrow">↑ <span class="action">link entities</span></div>
                <div class="ascent-level final">L3</div>
            </div>
            <div class="ascent-label">Your Intuitive Dataset</div>
        </div>
        """
    ),
    TutorialStep(
        title="Your Intent Matters",
        subtitle="What question will you answer?",
        content="""
        The magic of this method: **your intent shapes the result**.

        Two analysts with the same raw data can create completely different
        L3 datasets—each perfectly suited to their unique question.

        As you work through the descent and ascent, ask yourself:
        *"What do I actually need to know?"*

        Let that question guide every choice.
        """,
        diagram_html="""
        <div class="tutorial-diagram intent-diagram">
            <div class="intent-center">
                <div class="intent-question">?</div>
                <div class="intent-label">Your Question</div>
            </div>
            <div class="intent-branches">
                <div class="intent-branch left">
                    <div class="branch-arrow">←</div>
                    <div class="branch-result">Dataset A</div>
                </div>
                <div class="intent-branch right">
                    <div class="branch-arrow">→</div>
                    <div class="branch-result">Dataset B</div>
                </div>
            </div>
            <div class="intent-note">Same data, different intents, different results</div>
        </div>
        """
    ),
]


# =============================================================================
# CSS Styles
# =============================================================================

def _get_tutorial_css() -> str:
    """Return CSS for tutorial component styling."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Tutorial container */
    .tutorial-container {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        max-width: 700px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }

    /* Progress indicator */
    .tutorial-progress {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-bottom: 2rem;
    }

    .progress-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #e2e8f0;
        transition: all 0.3s ease;
    }

    .progress-dot.active {
        background: #002fa7;
        transform: scale(1.2);
    }

    .progress-dot.completed {
        background: #002fa7;
        opacity: 0.5;
    }

    /* Step content */
    .tutorial-step {
        text-align: center;
        animation: fadeIn 0.4s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .tutorial-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }

    .tutorial-subtitle {
        font-size: 1.1rem;
        color: #002fa7;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    .tutorial-content {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.7;
        text-align: left;
        max-width: 600px;
        margin: 0 auto 2rem auto;
    }

    .tutorial-content strong {
        color: #1e293b;
        font-weight: 600;
    }

    /* Diagram base styles */
    .tutorial-diagram {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
    }

    /* Intro diagram - chaos to clarity */
    .chaos-cloud {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin-bottom: 1rem;
    }

    .chaos-item {
        background: #fee2e2;
        color: #991b1b;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        animation: float 3s ease-in-out infinite;
    }

    .chaos-item:nth-child(2) { animation-delay: 0.5s; }
    .chaos-item:nth-child(3) { animation-delay: 1s; }
    .chaos-item:nth-child(4) { animation-delay: 1.5s; }
    .chaos-item:nth-child(5) { animation-delay: 2s; }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }

    .arrow-down {
        width: 2px;
        height: 40px;
        background: linear-gradient(to bottom, #cbd5e1, #002fa7);
        margin: 1rem auto;
        position: relative;
    }

    .arrow-down::after {
        content: '▼';
        position: absolute;
        bottom: -12px;
        left: 50%;
        transform: translateX(-50%);
        color: #002fa7;
        font-size: 12px;
    }

    .clarity-result {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: #002fa7;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        max-width: 200px;
        margin: 0 auto;
    }

    .clarity-icon {
        font-size: 1.2rem;
    }

    /* Levels diagram */
    .level-stack {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0;
    }

    .level-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 20px;
        background: white;
        border-radius: 8px;
        font-weight: 500;
        color: #1e293b;
        min-width: 200px;
        justify-content: flex-start;
        border: 1px solid #e2e8f0;
    }

    .level-badge {
        background: #002fa7;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .level-item.l0 .level-badge { background: #002fa7; }
    .level-item.l1 .level-badge { background: #0041d1; }
    .level-item.l2 .level-badge { background: #0052ff; }
    .level-item.l3 .level-badge { background: #3b82f6; }
    .level-item.l4 .level-badge { background: #60a5fa; }

    .level-connector {
        width: 2px;
        height: 12px;
        background: #cbd5e1;
    }

    /* Descent diagram */
    .descent-flow, .ascent-flow {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0;
    }

    .descent-level, .ascent-level {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #64748b;
    }

    .descent-level.active, .ascent-level.start {
        border-color: #002fa7;
        color: #002fa7;
    }

    .descent-level.final, .ascent-level.final {
        background: #002fa7;
        border-color: #002fa7;
        color: white;
    }

    .descent-arrow, .ascent-arrow {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 8px 0;
        color: #94a3b8;
        font-size: 1.2rem;
    }

    .descent-arrow .action, .ascent-arrow .action {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
    }

    .descent-label, .ascent-label {
        margin-top: 1rem;
        font-weight: 600;
        color: #002fa7;
        font-size: 0.9rem;
    }

    /* Intent diagram */
    .intent-diagram {
        text-align: center;
    }

    .intent-center {
        margin-bottom: 1.5rem;
    }

    .intent-question {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #002fa7, #0041d1);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 auto 0.5rem auto;
        box-shadow: 0 4px 20px rgba(0, 47, 167, 0.3);
    }

    .intent-label {
        font-weight: 600;
        color: #1e293b;
    }

    .intent-branches {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-bottom: 1rem;
    }

    .intent-branch {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
    }

    .branch-arrow {
        font-size: 1.5rem;
        color: #002fa7;
    }

    .branch-result {
        background: white;
        border: 2px solid #002fa7;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        color: #002fa7;
    }

    .intent-note {
        font-size: 0.85rem;
        color: #64748b;
        font-style: italic;
    }

    /* Navigation buttons - handled by Streamlit */
    .tutorial-nav {
        display: flex;
        justify-content: space-between;
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Skip link */
    .tutorial-skip {
        text-align: center;
        margin-top: 1.5rem;
    }

    .tutorial-skip a {
        color: #64748b;
        font-size: 0.9rem;
        text-decoration: none;
    }

    .tutorial-skip a:hover {
        color: #002fa7;
        text-decoration: underline;
    }
    </style>
    """


# =============================================================================
# Session State Helpers
# =============================================================================

def is_tutorial_completed() -> bool:
    """Check if the user has completed the tutorial."""
    return st.session_state.get(SESSION_KEY_TUTORIAL_COMPLETED, False)


def mark_tutorial_completed():
    """Mark the tutorial as completed."""
    st.session_state[SESSION_KEY_TUTORIAL_COMPLETED] = True


def skip_tutorial():
    """Skip the tutorial without completing it."""
    st.session_state[SESSION_KEY_TUTORIAL_COMPLETED] = True
    st.session_state[SESSION_KEY_TUTORIAL_STEP] = 0


def reset_tutorial():
    """Reset tutorial state to show it again."""
    st.session_state[SESSION_KEY_TUTORIAL_COMPLETED] = False
    st.session_state[SESSION_KEY_TUTORIAL_STEP] = 0


def _get_tutorial_step() -> int:
    """Get the current tutorial step (0-indexed)."""
    return st.session_state.get(SESSION_KEY_TUTORIAL_STEP, 0)


def _set_tutorial_step(step: int):
    """Set the current tutorial step."""
    st.session_state[SESSION_KEY_TUTORIAL_STEP] = max(0, min(step, len(TUTORIAL_STEPS) - 1))


# =============================================================================
# Rendering Functions
# =============================================================================

def _render_progress_dots(current_step: int, total_steps: int):
    """Render the progress dots indicator."""
    dots_html = '<div class="tutorial-progress">'
    for i in range(total_steps):
        if i < current_step:
            css_class = "progress-dot completed"
        elif i == current_step:
            css_class = "progress-dot active"
        else:
            css_class = "progress-dot"
        dots_html += f'<div class="{css_class}"></div>'
    dots_html += '</div>'
    st.markdown(dots_html, unsafe_allow_html=True)


def _render_tutorial_step(step: TutorialStep):
    """Render a single tutorial step."""
    st.markdown(f"""
    <div class="tutorial-step">
        <h1 class="tutorial-title">{step.title}</h1>
        <p class="tutorial-subtitle">{step.subtitle}</p>
        <div class="tutorial-content">{step.content}</div>
        {step.diagram_html}
    </div>
    """, unsafe_allow_html=True)


def render_tutorial(on_complete: Optional[Callable] = None) -> bool:
    """
    Render the Sarazin & Mourey method tutorial.

    Args:
        on_complete: Optional callback when tutorial is completed.

    Returns:
        True when user completes or skips tutorial, False otherwise.
    """
    # Inject CSS
    st.markdown(_get_tutorial_css(), unsafe_allow_html=True)

    # Get current step
    current_step = _get_tutorial_step()
    total_steps = len(TUTORIAL_STEPS)

    # Container for tutorial
    st.markdown('<div class="tutorial-container">', unsafe_allow_html=True)

    # Progress dots
    _render_progress_dots(current_step, total_steps)

    # Current step content
    _render_tutorial_step(TUTORIAL_STEPS[current_step])

    # Navigation
    st.markdown('<div class="tutorial-nav">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if current_step > 0:
            if st.button("← Back", key="tutorial_back", use_container_width=True):
                _set_tutorial_step(current_step - 1)
                st.rerun()

    with col3:
        if current_step < total_steps - 1:
            if st.button("Next →", key="tutorial_next", type="primary", use_container_width=True):
                _set_tutorial_step(current_step + 1)
                st.rerun()
        else:
            if st.button("Start Redesigning →", key="tutorial_complete", type="primary", use_container_width=True):
                mark_tutorial_completed()
                if on_complete:
                    on_complete()
                st.rerun()
                return True

    st.markdown('</div>', unsafe_allow_html=True)

    # Skip link
    with col2:
        if st.button("Skip tutorial", key="tutorial_skip", type="secondary", use_container_width=True):
            skip_tutorial()
            if on_complete:
                on_complete()
            st.rerun()
            return True

    st.markdown('</div>', unsafe_allow_html=True)

    return False


def render_tutorial_replay_button():
    """Render a button to replay the tutorial (for sidebar)."""
    if st.button("📖 View Tutorial", key="replay_tutorial", use_container_width=True):
        reset_tutorial()
        st.session_state.show_tutorial = True
        st.rerun()
