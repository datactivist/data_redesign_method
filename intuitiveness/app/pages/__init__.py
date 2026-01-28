"""
App Pages Package

Extracted pages from streamlit_app.py (Spec 011: Code Simplification)
Breaking down 4,900-line monolith into focused page modules.

Each page module handles one major workflow step:
- upload.py: File upload and data.gouv.fr search
- discovery.py: L4→L3 connection wizard
- descent.py: L3→L2→L1→L0 guided workflow
- ascent.py: L0→L1→L2→L3 reconstruction workflow
- export.py: Session export and visualization

All pages follow single-responsibility principle (<300 lines each).
"""

__all__ = []
