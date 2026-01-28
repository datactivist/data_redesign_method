"""
Translation Loader and Formatter

Implements Spec 011: Code Simplification (i18n JSON Extraction)
Replaces 2,280-line TRANSLATIONS dict with JSON loading

Loads translations from JSON files:
- en.json: English translations
- fr.json: French translations

Provides t() function for translation with placeholder substitution.
"""

import json
import os
from typing import Dict

# Cache for loaded translations
_TRANSLATIONS_CACHE: Dict[str, Dict[str, str]] = {}

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "fr": "Français",
}

# Default language
DEFAULT_LANGUAGE = "en"

# Current language (can be overridden by set_language)
_current_language = DEFAULT_LANGUAGE


def _load_translations(lang: str) -> Dict[str, str]:
    """
    Load translations for a given language from JSON file.

    Args:
        lang: Language code ('en' or 'fr')

    Returns:
        Dictionary of translations

    Raises:
        FileNotFoundError: If translation file doesn't exist
    """
    if lang in _TRANSLATIONS_CACHE:
        return _TRANSLATIONS_CACHE[lang]

    # Get path to JSON file
    i18n_dir = os.path.dirname(__file__)
    json_path = os.path.join(i18n_dir, f"{lang}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Translation file not found: {json_path}")

    # Load translations
    with open(json_path, 'r', encoding='utf-8') as f:
        translations = json.load(f)

    # Cache for future use
    _TRANSLATIONS_CACHE[lang] = translations

    return translations


def get_language() -> str:
    """
    Get current language setting.

    Returns:
        Language code ('en' or 'fr')
    """
    # Try to get from streamlit session state if available
    try:
        import streamlit as st
        if 'ui_language' in st.session_state:
            return st.session_state['ui_language']
    except ImportError:
        pass

    return _current_language


def set_language(lang: str) -> None:
    """
    Set current language.

    Args:
        lang: Language code ('en' or 'fr')

    Raises:
        ValueError: If language is not supported
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")

    global _current_language
    _current_language = lang

    # Also update streamlit session state if available
    try:
        import streamlit as st
        st.session_state['ui_language'] = lang
    except ImportError:
        pass


def t(key: str, **kwargs) -> str:
    """
    Translate a key to the current language with optional placeholder substitution.

    Args:
        key: Translation key
        **kwargs: Placeholder values for formatting

    Returns:
        Translated string with placeholders substituted

    Examples:
        >>> t('upload_success', filename='data.csv', rows=100, cols=5)
        'Loaded: data.csv (100 rows, 5 cols)'

        >>> t('found_data_types', count=3, conn=2)
        'Found 3 data types and 2 connections'
    """
    lang = get_language()

    try:
        translations = _load_translations(lang)
    except FileNotFoundError:
        # Fall back to English if language file not found
        if lang != DEFAULT_LANGUAGE:
            translations = _load_translations(DEFAULT_LANGUAGE)
        else:
            return f"[MISSING: {key}]"

    # Get translated text
    text = translations.get(key)

    if text is None:
        # Return key as fallback
        return f"[MISSING: {key}]"

    # Substitute placeholders
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError as e:
            # Missing placeholder - return text with error indicator
            return f"{text} [PLACEHOLDER ERROR: {e}]"

    return text
