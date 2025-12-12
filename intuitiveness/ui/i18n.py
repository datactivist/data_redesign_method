"""
Internationalization (i18n) module for Data Redesign Method UI.

Provides bilingual support (English/French) for all user-facing strings.
Feature: 006-playwright-mcp-e2e (UI improvements)
"""

import streamlit as st
from typing import Dict, Optional

# Session state key for language preference
SESSION_KEY_LANGUAGE = "ui_language"

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "fr": "Francais",
}

# Default language
DEFAULT_LANGUAGE = "en"

# Translation dictionary organized by functional area
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # ============================================
    # UPLOAD / STEP 1
    # ============================================
    "upload_header": {
        "en": "Upload Data",
        "fr": "Importer les donnees",
    },
    "upload_instruction": {
        "en": "Upload one or more CSV files to begin the redesign process.",
        "fr": "Importez un ou plusieurs fichiers CSV pour commencer le processus de redesign.",
    },
    "upload_success": {
        "en": "Loaded: {filename} ({rows} rows, {cols} cols)",
        "fr": "Charge: {filename} ({rows} lignes, {cols} colonnes)",
    },
    "upload_error": {
        "en": "Error loading {filename}: {error}",
        "fr": "Erreur lors du chargement de {filename}: {error}",
    },
    "upload_demo_data": {
        "en": "Or use demo data:",
        "fr": "Ou utilisez les donnees de demonstration:",
    },
    "load_demo_data": {
        "en": "Load Demo Data",
        "fr": "Charger les donnees de demo",
    },

    # ============================================
    # ENTITIES / STEP 2
    # ============================================
    "entities_header": {
        "en": "Define Items",
        "fr": "Definir les elements",
    },
    "connect_your_data": {
        "en": "Connect Your Data",
        "fr": "Connecter vos donnees",
    },
    "analyze_files": {
        "en": "I'll analyze your files and suggest how to connect them.",
        "fr": "Je vais analyser vos fichiers et suggerer comment les connecter.",
    },
    "found_data_types": {
        "en": "Found {count} data types and {conn} connections",
        "fr": "Trouve {count} types de donnees et {conn} connexions",
    },
    "error_analyzing": {
        "en": "Error analyzing files: {error}",
        "fr": "Erreur lors de l'analyse des fichiers: {error}",
    },
    "reset_analyze": {
        "en": "Reset and Re-analyze",
        "fr": "Reinitialiser et re-analyser",
    },
    "could_not_analyze": {
        "en": "Could not automatically analyze your files. Please continue manually.",
        "fr": "Impossible d'analyser automatiquement vos fichiers. Veuillez continuer manuellement.",
    },

    # ============================================
    # CONNECTIONS / STEP 3 (Semantic Matching)
    # ============================================
    "connections_header": {
        "en": "Define Connections",
        "fr": "Definir les connexions",
    },
    "build_connected_info": {
        "en": "Build Connected Information",
        "fr": "Construire les informations connectees",
    },
    "build_connected_info_desc": {
        "en": "Your connected information will be built instantly. No setup needed.",
        "fr": "Vos informations connectees seront construites instantanement. Aucune configuration necessaire.",
    },
    "how_to_connect": {
        "en": "How to connect your files:",
        "fr": "Comment connecter vos fichiers:",
    },
    "exact_matching_info": {
        "en": "Exact matching connects items that have the same identifier values.",
        "fr": "La correspondance exacte connecte les elements ayant les memes valeurs d'identifiant.",
    },
    "smart_matching": {
        "en": "Use smart matching (AI)",
        "fr": "Utiliser la correspondance intelligente (IA)",
    },
    "smart_matching_desc": {
        "en": "AI finds similar terms (e.g., 'School ID' matches 'UAI code')",
        "fr": "L'IA trouve des termes similaires (ex: 'ID Ecole' correspond a 'code UAI')",
    },
    "matching_strictness": {
        "en": "Matching strictness",
        "fr": "Rigueur de la correspondance",
    },
    "matching_strictness_help": {
        "en": "Lower = more matches, Higher = stricter matches",
        "fr": "Plus bas = plus de correspondances, Plus haut = correspondances plus strictes",
    },
    "build_button": {
        "en": "Build Connected Information",
        "fr": "Construire les informations connectees",
    },
    "connected_success": {
        "en": "Connected information built successfully!",
        "fr": "Informations connectees construites avec succes!",
    },
    "how_data_connects": {
        "en": "How Your Data Connects",
        "fr": "Comment vos donnees se connectent",
    },

    # ============================================
    # CATEGORIES / STEP 4
    # ============================================
    "categories_header": {
        "en": "Define Categories",
        "fr": "Definir les categories",
    },
    "using_selected_table": {
        "en": "Using selected table: {name} ({rows} rows)",
        "fr": "Utilisation de la table selectionnee: {name} ({rows} lignes)",
    },
    "no_data_available": {
        "en": "No data available for '{name}'",
        "fr": "Aucune donnee disponible pour '{name}'",
    },
    "no_columns_found": {
        "en": "No suitable columns found for categorization",
        "fr": "Aucune colonne adaptee trouvee pour la categorisation",
    },
    "enter_categories": {
        "en": "Enter the categories you want to group by:",
        "fr": "Entrez les categories par lesquelles vous souhaitez regrouper:",
    },
    "categories_input_placeholder": {
        "en": "e.g., PRIVE, PUBLIC or High, Medium, Low",
        "fr": "ex: PRIVE, PUBLIC ou Haut, Moyen, Bas",
    },
    "categorize_button": {
        "en": "Categorize Data",
        "fr": "Categoriser les donnees",
    },
    "categorizing_info": {
        "en": "Categorizing data by domains...",
        "fr": "Categorisation des donnees par domaines...",
    },

    # ============================================
    # VALUES / STEP 5
    # ============================================
    "values_header": {
        "en": "Extract Values",
        "fr": "Extraire les valeurs",
    },
    "select_column_extract": {
        "en": "Select a column to extract from your categorized data.",
        "fr": "Selectionnez une colonne a extraire de vos donnees categorisees.",
    },
    "no_categorized_data": {
        "en": "No categorized data available. Please complete the previous step.",
        "fr": "Aucune donnee categorisee disponible. Veuillez completer l'etape precedente.",
    },
    "extract_button": {
        "en": "Extract Values",
        "fr": "Extraire les valeurs",
    },

    # ============================================
    # COMPUTATION / STEP 6
    # ============================================
    "computation_header": {
        "en": "Compute Result",
        "fr": "Calculer le resultat",
    },
    "choose_calculation": {
        "en": "Choose how to calculate a final result from your values.",
        "fr": "Choisissez comment calculer un resultat final a partir de vos valeurs.",
    },
    "available_value_lists": {
        "en": "Available Value Lists",
        "fr": "Listes de valeurs disponibles",
    },
    "compute_button": {
        "en": "Compute Metrics",
        "fr": "Calculer les metriques",
    },

    # ============================================
    # RESULTS / STEP 7
    # ============================================
    "results_header": {
        "en": "View Results",
        "fr": "Voir les resultats",
    },
    "descent_complete": {
        "en": "Descent complete! Here are your results:",
        "fr": "Descente terminee! Voici vos resultats:",
    },
    "start_new_analysis": {
        "en": "Start New Analysis",
        "fr": "Demarrer une nouvelle analyse",
    },
    "start_redesign": {
        "en": "Start Redesign (Ascent)",
        "fr": "Demarrer le redesign (Remontee)",
    },
    "cannot_start_redesign": {
        "en": "Cannot start redesign - descent data incomplete",
        "fr": "Impossible de demarrer le redesign - donnees de descente incompletes",
    },

    # ============================================
    # NAVIGATION
    # ============================================
    "back_button": {
        "en": "Back",
        "fr": "Retour",
    },
    "continue_button": {
        "en": "Continue",
        "fr": "Continuer",
    },
    "view_results": {
        "en": "View Results",
        "fr": "Voir les resultats",
    },
    "configuration_complete": {
        "en": "Configuration complete! Moving to next step...",
        "fr": "Configuration terminee! Passage a l'etape suivante...",
    },
    "navigation_tree": {
        "en": "Navigation Tree",
        "fr": "Arbre de navigation",
    },

    # ============================================
    # SESSION MANAGEMENT
    # ============================================
    "save_session": {
        "en": "Save Session Graph",
        "fr": "Sauvegarder le graphe de session",
    },
    "load_session": {
        "en": "Load Saved Session",
        "fr": "Charger une session sauvegardee",
    },
    "session_saved": {
        "en": "Session saved to: {path}",
        "fr": "Session sauvegardee dans: {path}",
    },
    "session_save_failed": {
        "en": "Failed to save session graph: {error}",
        "fr": "Echec de la sauvegarde du graphe de session: {error}",
    },
    "no_session_data": {
        "en": "No active session data to save. Complete the workflow first.",
        "fr": "Aucune donnee de session active a sauvegarder. Completez d'abord le workflow.",
    },
    "session_loaded": {
        "en": "Session graph loaded successfully!",
        "fr": "Graphe de session charge avec succes!",
    },
    "session_load_failed": {
        "en": "Failed to load session graph: {error}",
        "fr": "Echec du chargement du graphe de session: {error}",
    },
    "loaded_session_summary": {
        "en": "Loaded Session Summary:",
        "fr": "Resume de la session chargee:",
    },
    "navigation_path": {
        "en": "Navigation Path:",
        "fr": "Chemin de navigation:",
    },
    "continue_free_exploration": {
        "en": "Continue to Free Exploration",
        "fr": "Continuer vers l'exploration libre",
    },

    # ============================================
    # ASCENT PHASE
    # ============================================
    "ascent_phase": {
        "en": "Ascent Phase",
        "fr": "Phase de remontee",
    },
    "step_9_header": {
        "en": "Step 9: Recover Source Values (L0 -> L1)",
        "fr": "Etape 9: Recuperer les valeurs sources (L0 -> L1)",
    },
    "recover_source_values": {
        "en": "Recover Source Values",
        "fr": "Recuperer les valeurs sources",
    },
    "l1_not_found": {
        "en": "L1 data not found in session graph",
        "fr": "Donnees L1 introuvables dans le graphe de session",
    },
    "session_graph_unavailable": {
        "en": "Session graph not available",
        "fr": "Graphe de session indisponible",
    },
    "step_10_header": {
        "en": "Step 10: Add New Dimension (L1 -> L2)",
        "fr": "Etape 10: Ajouter une nouvelle dimension (L1 -> L2)",
    },
    "apply_categorization": {
        "en": "Apply Categorization",
        "fr": "Appliquer la categorisation",
    },
    "categories_applied": {
        "en": "Applied categories: {summary}",
        "fr": "Categories appliquees: {summary}",
    },
    "added_commune": {
        "en": "Added Commune location ({count} unique)",
        "fr": "Commune ajoutee ({count} uniques)",
    },
    "step_11_header": {
        "en": "Step 11: Enrich to L3 with Linkage Key (L2 -> L3)",
        "fr": "Etape 11: Enrichir vers L3 avec cle de liaison (L2 -> L3)",
    },
    "dimension": {
        "en": "Dimension: {name}",
        "fr": "Dimension: {name}",
    },
    "location_data": {
        "en": "Location data: {count} unique Communes",
        "fr": "Donnees de localisation: {count} Communes uniques",
    },
    "available_columns": {
        "en": "Available columns from original datasets:",
        "fr": "Colonnes disponibles des jeux de donnees originaux:",
    },
    "select_linkage_keys": {
        "en": "Select Linkage Key Column(s):",
        "fr": "Selectionnez la/les colonne(s) cle de liaison:",
    },
    "complete_enrichment": {
        "en": "Complete Enrichment",
        "fr": "Completer l'enrichissement",
    },
    "enriched_l3": {
        "en": "Enriched L3 table: {rows} rows, {cols} columns",
        "fr": "Table L3 enrichie: {rows} lignes, {cols} colonnes",
    },
    "linkage_keys_selected": {
        "en": "Linkage keys selected: {keys}",
        "fr": "Cles de liaison selectionnees: {keys}",
    },
    "step_12_header": {
        "en": "Step 12: Final Verification - Ascent Complete!",
        "fr": "Etape 12: Verification finale - Remontee terminee!",
    },
    "l0_ground_truth": {
        "en": "L0 Ground Truth: {value}",
        "fr": "Verite terrain L0: {value}",
    },
    "new_dimension": {
        "en": "New Dimension: {name}",
        "fr": "Nouvelle dimension: {name}",
    },
    "selected_linkage_keys": {
        "en": "Selected Linkage Keys:",
        "fr": "Cles de liaison selectionnees:",
    },
    "demographic_linkage_keys": {
        "en": "Demographic Linkage Keys Available:",
        "fr": "Cles de liaison demographiques disponibles:",
    },
    "export_ascent_artifacts": {
        "en": "Export Ascent Artifacts",
        "fr": "Exporter les artefacts de remontee",
    },
    "start_new_ascent": {
        "en": "Start New Ascent",
        "fr": "Demarrer une nouvelle remontee",
    },

    # ============================================
    # FREE EXPLORATION MODE
    # ============================================
    "free_exploration": {
        "en": "Free Exploration",
        "fr": "Exploration libre",
    },
    "exploration_mode": {
        "en": "Exploration Mode",
        "fr": "Mode exploration",
    },
    "current_progress": {
        "en": "Current Progress",
        "fr": "Progression actuelle",
    },
    "quick_navigation": {
        "en": "Quick Navigation",
        "fr": "Navigation rapide",
    },
    "start_exploring": {
        "en": "Start exploring to see the exploration tree",
        "fr": "Commencez a explorer pour voir l'arbre d'exploration",
    },
    "reset_workflow": {
        "en": "Reset Workflow",
        "fr": "Reinitialiser le workflow",
    },
    "what_would_you_like": {
        "en": "What would you like to do?",
        "fr": "Que souhaitez-vous faire?",
    },
    "explore_deeper": {
        "en": "Explore deeper to {level}",
        "fr": "Explorer plus profondement vers {level}",
    },
    "build_up_to": {
        "en": "Build up to {level}",
        "fr": "Construire vers {level}",
    },

    # ============================================
    # EXPORT
    # ============================================
    "export_header": {
        "en": "Export Options",
        "fr": "Options d'export",
    },
    "export_csv_header": {
        "en": "Export Data as CSV",
        "fr": "Exporter les donnees en CSV",
    },
    "export_csv_info": {
        "en": "Download each level's data directly as CSV files.",
        "fr": "Telechargez les donnees de chaque niveau directement en fichiers CSV.",
    },

    # ============================================
    # LEVEL NAMES
    # ============================================
    "level_l4": {
        "en": "L4: Raw Files",
        "fr": "L4: Fichiers bruts",
    },
    "level_l3": {
        "en": "L3: Connected Graph",
        "fr": "L3: Graphe connecte",
    },
    "level_l2": {
        "en": "L2: Categorized Table",
        "fr": "L2: Table categorisee",
    },
    "level_l1": {
        "en": "L1: Value List",
        "fr": "L1: Liste de valeurs",
    },
    "level_l0": {
        "en": "L0: Single Value",
        "fr": "L0: Valeur unique",
    },

    # ============================================
    # SIDEBAR / MODE SWITCHING
    # ============================================
    "select_mode": {
        "en": "Select mode:",
        "fr": "Selectionnez le mode:",
    },
    "step_by_step": {
        "en": "Step-by-Step",
        "fr": "Pas a pas",
    },
    "step_by_step_help": {
        "en": "Step-by-Step: Follow a guided workflow. Free Exploration: Navigate freely.",
        "fr": "Pas a pas: Suivez un workflow guide. Exploration libre: Naviguez librement.",
    },
    "sidebar_session": {
        "en": "Session",
        "fr": "Session",
    },
    "save_button": {
        "en": "Save",
        "fr": "Sauvegarder",
    },
    "save_help": {
        "en": "Save current session to browser",
        "fr": "Sauvegarder la session actuelle dans le navigateur",
    },
    "saved_success": {
        "en": "Saved!",
        "fr": "Sauvegarde!",
    },
    "save_too_large": {
        "en": "Could not save (data too large)",
        "fr": "Impossible de sauvegarder (donnees trop volumineuses)",
    },
    "save_failed": {
        "en": "Save failed: {error}",
        "fr": "Echec de la sauvegarde: {error}",
    },
    "clear_button": {
        "en": "Clear",
        "fr": "Effacer",
    },
    "clear_help": {
        "en": "Clear all saved data",
        "fr": "Effacer toutes les donnees sauvegardees",
    },
    "switch_to_step_by_step": {
        "en": "Switch to Step-by-Step",
        "fr": "Passer au mode pas a pas",
    },
    "transform_data": {
        "en": "Transform your data from chaos to clarity through guided questions.",
        "fr": "Transformez vos donnees du chaos vers la clarte grace a des questions guidees.",
    },

    # ============================================
    # MISC / COMMON
    # ============================================
    "rows": {
        "en": "rows",
        "fr": "lignes",
    },
    "columns": {
        "en": "columns",
        "fr": "colonnes",
    },
    "unique_values": {
        "en": "unique values",
        "fr": "valeurs uniques",
    },
    "items": {
        "en": "items",
        "fr": "elements",
    },
    "selected": {
        "en": "Selected",
        "fr": "Selectionne",
    },
    "no_items_found": {
        "en": "No items found. Please check the previous step.",
        "fr": "Aucun element trouve. Veuillez verifier l'etape precedente.",
    },
    "browse_connected_info": {
        "en": "Browse Your Connected Information",
        "fr": "Parcourez vos informations connectees",
    },
    "graph_statistics": {
        "en": "Graph Statistics:",
        "fr": "Statistiques du graphe:",
    },
    "node_types": {
        "en": "Node Types:",
        "fr": "Types de noeuds:",
    },
    "no_items_matched": {
        "en": "No items matched this domain",
        "fr": "Aucun element ne correspond a ce domaine",
    },
    "no_metrics_computed": {
        "en": "No atomic metrics computed yet",
        "fr": "Aucune metrique atomique calculee pour l'instant",
    },
    "no_graph_built": {
        "en": "No knowledge graph built yet",
        "fr": "Aucun graphe de connaissances construit pour l'instant",
    },
    "language": {
        "en": "Language",
        "fr": "Langue",
    },
}


def get_language() -> str:
    """Get the current language from session state."""
    return st.session_state.get(SESSION_KEY_LANGUAGE, DEFAULT_LANGUAGE)


def set_language(lang: str) -> None:
    """Set the current language in session state."""
    if lang in SUPPORTED_LANGUAGES:
        st.session_state[SESSION_KEY_LANGUAGE] = lang


def t(key: str, **kwargs) -> str:
    """
    Get translated string for the given key.

    Args:
        key: Translation key from TRANSLATIONS dictionary
        **kwargs: Format arguments for string interpolation

    Returns:
        Translated string, or key itself if not found

    Example:
        t("upload_success", filename="data.csv", rows=100, cols=5)
        # Returns: "Loaded: data.csv (100 rows, 5 cols)"
    """
    lang = get_language()

    # Look up translation
    translation_dict = TRANSLATIONS.get(key)
    if translation_dict is None:
        # Key not found, return the key itself as fallback
        return key

    # Get translation for current language, fallback to English
    text = translation_dict.get(lang, translation_dict.get(DEFAULT_LANGUAGE, key))

    # Apply string formatting if kwargs provided
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            # If format fails, return unformatted text
            pass

    return text


def render_language_toggle() -> Optional[str]:
    """
    Render a language toggle in the sidebar.

    Returns:
        The currently selected language code, or None if not changed
    """
    current_lang = get_language()

    # Create language options as list for selectbox
    lang_options = list(SUPPORTED_LANGUAGES.keys())
    lang_labels = list(SUPPORTED_LANGUAGES.values())

    # Find current index
    current_index = lang_options.index(current_lang) if current_lang in lang_options else 0

    # Render selectbox
    selected_label = st.sidebar.selectbox(
        label=t("language"),
        options=lang_labels,
        index=current_index,
        key="language_selector",
    )

    # Find which language was selected
    selected_index = lang_labels.index(selected_label)
    selected_lang = lang_options[selected_index]

    # Update if changed
    if selected_lang != current_lang:
        set_language(selected_lang)
        st.rerun()

    return selected_lang


def render_language_toggle_compact() -> None:
    """
    Render a compact language toggle using radio buttons.
    More visible than dropdown, good for sidebar header.
    """
    current_lang = get_language()

    selected = st.sidebar.radio(
        label="",  # Empty label for compact display
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: SUPPORTED_LANGUAGES[x],
        index=list(SUPPORTED_LANGUAGES.keys()).index(current_lang),
        horizontal=True,
        key="language_radio",
    )

    if selected != current_lang:
        set_language(selected)
        st.rerun()
