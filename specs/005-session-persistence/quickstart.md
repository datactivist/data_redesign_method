# Quickstart: Session Persistence

**Feature**: 005-session-persistence
**Date**: 2025-12-04

## Overview

This guide explains how to integrate session persistence into the Data Redesign wizard.

## Installation

Add to requirements.txt:

```
streamlit-javascript>=0.1.5
```

Install:

```bash
pip install streamlit-javascript
```

## Basic Usage

### 1. Import the persistence module

```python
from intuitiveness.persistence import SessionStore
```

### 2. Initialize at app startup

```python
def main():
    store = SessionStore()

    # Check for existing session
    if store.has_saved_session():
        info = store.get_session_info()
        st.info(f"Welcome back! Session from {info.timestamp}")

        if st.button("Continue where you left off"):
            store.load()
        if st.button("Start fresh"):
            store.clear()

    # Rest of your app...
```

### 3. Auto-save after important actions

```python
# After file upload
if uploaded_files:
    process_files(uploaded_files)
    store.save()

# After wizard step change
if st.button("Next"):
    st.session_state.current_step += 1
    store.save()
```

## Integration Example

```python
import streamlit as st
from intuitiveness.persistence import SessionStore, RecoveryAction
from intuitiveness.ui.recovery_banner import render_recovery_banner

def main():
    st.set_page_config(page_title="Data Redesign", layout="wide")

    store = SessionStore()

    # Handle session recovery (only on first load)
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True

        if store.has_saved_session():
            info = store.get_session_info()
            action = render_recovery_banner(info)

            if action == RecoveryAction.CONTINUE:
                result = store.load()
                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            elif action == RecoveryAction.START_FRESH:
                store.clear()

            st.rerun()

    # Initialize default session state
    initialize_session_state()

    # Render main app
    render_wizard()

    # Auto-save after each interaction (debounced internally)
    store.save()

if __name__ == "__main__":
    main()
```

## Handling Large Files

If files exceed localStorage limits:

```python
result = store.save()

if not result.success:
    st.warning("Session too large to save completely.")
    for warning in result.warnings:
        st.caption(warning)
```

## Testing Persistence

1. Upload files and progress through wizard
2. Refresh browser (Cmd+R / Ctrl+R)
3. Verify recovery banner appears
4. Click "Continue" and verify all data restored
5. Test "Start Fresh" clears everything

## Troubleshooting

### Session not restoring?

- Check browser localStorage in DevTools (Application tab)
- Look for key `data_redesign_session`
- Verify no localStorage quota errors in console

### Data corrupted after restore?

- Clear localStorage: `localStorage.removeItem('data_redesign_session')`
- Or use "Start Fresh" button in app

### Very slow save/load?

- Large DataFrames (>10MB) may cause delays
- Consider reducing data before save
- Check compression is working (data should be ~70% smaller)
