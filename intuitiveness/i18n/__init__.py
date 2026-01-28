"""
Internationalization (i18n) Package

Implements Spec 011: Code Simplification
Extracted from ui/i18n.py (2,280 → JSON + 50 lines)

This package provides bilingual support (English/French) using JSON translation files:
- en.json: English translations (455 keys)
- fr.json: French translations (455 keys)
- translator.py: Translation loader and formatter

The old 2,280-line dictionary is now two clean JSON files.
"""

from intuitiveness.i18n.translator import t, get_language, set_language, SUPPORTED_LANGUAGES

__all__ = ["t", "get_language", "set_language", "SUPPORTED_LANGUAGES"]
