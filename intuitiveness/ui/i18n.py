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

    # ============================================
    # DATA.GOUV.FR SEARCH (Feature 008)
    # ============================================
    "search_tagline": {
        "en": "Redesign any data for your intent",
        "fr": "Redesignez toute donnee selon votre intention",
    },
    "search_placeholder": {
        "en": "Search French open data...",
        "fr": "Rechercher des donnees ouvertes francaises...",
    },
    "search_button": {
        "en": "Search",
        "fr": "Rechercher",
    },
    "searching": {
        "en": "Searching data.gouv.fr...",
        "fr": "Recherche sur data.gouv.fr...",
    },
    "no_results": {
        "en": "No datasets found for '{query}'",
        "fr": "Aucun jeu de donnees trouve pour '{query}'",
    },
    "loading_dataset": {
        "en": "Loading dataset...",
        "fr": "Chargement du jeu de donnees...",
    },
    "dataset_loaded": {
        "en": "Dataset loaded! Starting redesign workflow.",
        "fr": "Jeu de donnees charge! Demarrage du workflow de redesign.",
    },
    "api_error": {
        "en": "Could not connect to data.gouv.fr. Please try uploading files instead.",
        "fr": "Connexion a data.gouv.fr impossible. Veuillez essayer d'importer des fichiers.",
    },
    "dataset_by": {
        "en": "by {org}",
        "fr": "par {org}",
    },
    "last_updated": {
        "en": "Updated {date}",
        "fr": "Mis a jour le {date}",
    },
    "resources_count": {
        "en": "{count} files available",
        "fr": "{count} fichiers disponibles",
    },
    "select_resource": {
        "en": "Select a file to load:",
        "fr": "Selectionnez un fichier a charger:",
    },
    "load_csv": {
        "en": "Load this CSV",
        "fr": "Charger ce CSV",
    },
    "file_size": {
        "en": "Size: {size}",
        "fr": "Taille: {size}",
    },
    "no_csv_available": {
        "en": "No CSV files available in this dataset",
        "fr": "Aucun fichier CSV disponible dans ce jeu de donnees",
    },
    "or_upload_files": {
        "en": "Or upload your own files:",
        "fr": "Ou importez vos propres fichiers:",
    },
    "upload_option": {
        "en": "Upload Files",
        "fr": "Importer des fichiers",
    },
    "search_option": {
        "en": "Search Open Data",
        "fr": "Rechercher des donnees ouvertes",
    },
    "results_found": {
        "en": "{count} datasets found",
        "fr": "{count} jeux de donnees trouves",
    },
    "load_more": {
        "en": "Load more results",
        "fr": "Charger plus de resultats",
    },
    "try_different_query": {
        "en": "Try a different search query or upload your own files below.",
        "fr": "Essayez une autre recherche ou importez vos propres fichiers ci-dessous.",
    },
    "csv_available": {
        "en": "CSV available",
        "fr": "CSV disponible",
    },
    "no_csv": {
        "en": "No CSV",
        "fr": "Pas de CSV",
    },
    "large_file_warning": {
        "en": "This file is large ({size}). Download may take a while.",
        "fr": "Ce fichier est volumineux ({size}). Le telechargement peut prendre du temps.",
    },
    "download_failed": {
        "en": "Could not download file. Please try again or choose another dataset.",
        "fr": "Impossible de telecharger le fichier. Veuillez reessayer ou choisir un autre jeu de donnees.",
    },
    "retry_button": {
        "en": "Retry",
        "fr": "Reessayer",
    },
    # Multi-dataset loading (008-datagouv-search)
    "datasets_loaded": {
        "en": "{count} dataset(s) loaded",
        "fr": "{count} jeu(x) de données chargé(s)",
    },
    "add_another_dataset": {
        "en": "Add another dataset",
        "fr": "Ajouter un autre jeu de données",
    },
    "continue_with_datasets": {
        "en": "Continue with datasets",
        "fr": "Continuer avec les données",
    },
    "search_more_datasets": {
        "en": "Search for more datasets:",
        "fr": "Rechercher d'autres jeux de données:",
    },

    # ============================================
    # TUTORIAL / PAPER MODAL
    # ============================================
    "tutorial_modal_title": {
        "en": "The Intuitiveness Method",
        "fr": "La méthode Intuitiveness",
    },
    "tutorial_description": {
        "en": "Understanding the methodology behind intuitive data redesign",
        "fr": "Comprendre la méthodologie du redesign intuitif de données",
    },
    "download_pdf": {
        "en": "Download PDF",
        "fr": "Télécharger le PDF",
    },
    "start_redesigning": {
        "en": "Start Redesigning",
        "fr": "Commencer le redesign",
    },
    "view_paper": {
        "en": "View Paper",
        "fr": "Voir l'article",
    },

    # ============================================
    # BUTTONS & NAVIGATION
    # ============================================
    "next_button": {
        "en": "Next",
        "fr": "Suivant",
    },
    "continue_button": {
        "en": "Continue",
        "fr": "Continuer",
    },
    "back_button": {
        "en": "Back",
        "fr": "Retour",
    },
    "cancel_button": {
        "en": "Cancel",
        "fr": "Annuler",
    },
    "confirm_button": {
        "en": "Confirm",
        "fr": "Confirmer",
    },
    "clear_all": {
        "en": "Clear all",
        "fr": "Tout effacer",
    },
    "load_more": {
        "en": "Load more results",
        "fr": "Charger plus de résultats",
    },
    "add_to_selection": {
        "en": "Add to selection",
        "fr": "Ajouter à la sélection",
    },

    # ============================================
    # RECOVERY BANNER
    # ============================================
    "welcome_back": {
        "en": "Welcome back!",
        "fr": "Bon retour !",
    },
    "continue_where_left": {
        "en": "Continue where I left off",
        "fr": "Reprendre où j'en étais",
    },
    "start_fresh": {
        "en": "Start fresh",
        "fr": "Recommencer à zéro",
    },
    "yes_start_fresh": {
        "en": "Yes, start fresh",
        "fr": "Oui, recommencer",
    },

    # ============================================
    # DATA DISPLAY
    # ============================================
    "no_data_display": {
        "en": "No data to display.",
        "fr": "Aucune donnée à afficher.",
    },
    "no_description_available": {
        "en": "No description available",
        "fr": "Aucune description disponible",
    },
    "failed_to_load": {
        "en": "Failed to load.",
        "fr": "Échec du chargement.",
    },
    "selected": {
        "en": "Selected",
        "fr": "Sélectionné",
    },

    # ============================================
    # TIME AGO STRINGS
    # ============================================
    "just_now": {
        "en": "just now",
        "fr": "à l'instant",
    },
    "minutes_ago": {
        "en": "{count} minute(s) ago",
        "fr": "il y a {count} minute(s)",
    },
    "hours_ago": {
        "en": "{count} hour(s) ago",
        "fr": "il y a {count} heure(s)",
    },
    "yesterday": {
        "en": "yesterday",
        "fr": "hier",
    },
    "days_ago": {
        "en": "{count} day(s) ago",
        "fr": "il y a {count} jour(s)",
    },
    "session_saved_ago": {
        "en": "Your previous session was saved {time_ago}.",
        "fr": "Votre session précédente a été sauvegardée {time_ago}.",
    },
    "at_step_with_files": {
        "en": "You were at **Step {step}** with **{count} file(s)** uploaded.",
        "fr": "Vous étiez à l'**Étape {step}** avec **{count} fichier(s)** chargé(s).",
    },
    "session_info": {
        "en": "Session version: {version} | Size: {size} KB",
        "fr": "Version de session: {version} | Taille: {size} KB",
    },
    "confirm_start_fresh": {
        "en": "This will clear all your uploaded files and progress. Are you sure?",
        "fr": "Cela effacera tous vos fichiers et votre progression. Êtes-vous sûr ?",
    },

    # ============================================
    # DATAGOUV SEARCH INTERFACE
    # ============================================
    "search_headline": {
        "en": "Redesign <span class=\"accent\">any data</span> for your intent",
        "fr": "Redesignez <span class=\"accent\">vos données</span> selon votre intention",
    },
    "search_datasets_label": {
        "en": "Search datasets",
        "fr": "Rechercher des jeux de données",
    },
    "search_placeholder": {
        "en": "Search French open data...",
        "fr": "Rechercher des données ouvertes...",
    },
    "search_button": {
        "en": "Search",
        "fr": "Rechercher",
    },
    "searching": {
        "en": "Searching data.gouv.fr...",
        "fr": "Recherche sur data.gouv.fr...",
    },
    "datasets_found": {
        "en": "{count} datasets found",
        "fr": "{count} jeux de données trouvés",
    },
    "click_to_add": {
        "en": "Click to add to selection",
        "fr": "Cliquez pour ajouter à la sélection",
    },
    "keywords_used": {
        "en": "Keywords: {keywords}",
        "fr": "Mots-clés: {keywords}",
    },
    "no_datasets_found": {
        "en": "No datasets found for",
        "fr": "Aucun jeu de données trouvé pour",
    },
    "try_different_keywords": {
        "en": "Try different keywords or a broader search term.",
        "fr": "Essayez des mots-clés différents ou un terme plus général.",
    },
    "loading": {
        "en": "Loading...",
        "fr": "Chargement...",
    },
    "loading_more": {
        "en": "Loading more...",
        "fr": "Chargement...",
    },
    "search_failed": {
        "en": "Search failed. Please try again.",
        "fr": "La recherche a échoué. Veuillez réessayer.",
    },
    "failed_load_more": {
        "en": "Failed to load more results.",
        "fr": "Impossible de charger plus de résultats.",
    },

    # ============================================
    # ASCENT FORMS
    # ============================================
    "enter_categories_label": {
        "en": "Enter categories (comma-separated):",
        "fr": "Entrez les catégories (séparées par des virgules) :",
    },
    "categories_label": {
        "en": "Categories:",
        "fr": "Catégories :",
    },
    "categories_help": {
        "en": "Enter categories separated by commas (e.g., 'Sales, Marketing, Engineering')",
        "fr": "Entrez les catégories séparées par des virgules (ex: 'Ventes, Marketing, Ingénierie')",
    },
    "use_smart_matching": {
        "en": "Use smart matching (AI)",
        "fr": "Utiliser la correspondance intelligente (IA)",
    },
    "smart_matching_help": {
        "en": "Use AI to find similar items (smarter but slower)",
        "fr": "Utiliser l'IA pour trouver des éléments similaires (plus intelligent mais plus lent)",
    },
    "matching_strictness_label": {
        "en": "Matching strictness:",
        "fr": "Rigueur de la correspondance :",
    },
    "matching_strictness_help": {
        "en": "How strict should matching be? (higher = fewer matches)",
        "fr": "Quelle rigueur pour la correspondance ? (plus haut = moins de correspondances)",
    },
    "expand_result_title": {
        "en": "Expand Result",
        "fr": "Développer le résultat",
    },
    "expand_result_info": {
        "en": "See all the values that were used to calculate this result. The original list of values is preserved and can be restored.",
        "fr": "Voir toutes les valeurs utilisées pour calculer ce résultat. La liste originale des valeurs est préservée et peut être restaurée.",
    },
    "cannot_expand": {
        "en": "Cannot expand",
        "fr": "Impossible de développer",
    },
    "cannot_expand_reason": {
        "en": "This value wasn't calculated from a list.",
        "fr": "Cette valeur n'a pas été calculée à partir d'une liste.",
    },
    "expansion_unavailable_info": {
        "en": "Expansion is only available for calculated results. This value was entered directly, so there is no source list to show.",
        "fr": "Le développement est uniquement disponible pour les résultats calculés. Cette valeur a été entrée directement, il n'y a donc pas de liste source à afficher.",
    },
    "calculation_method": {
        "en": "Calculation method",
        "fr": "Méthode de calcul",
    },
    "source_values_preview": {
        "en": "Source values preview",
        "fr": "Aperçu des valeurs sources",
    },
    "first_n_values": {
        "en": "first {n}",
        "fr": "les {n} premières",
    },
    "total_values": {
        "en": "Total values: {count}",
        "fr": "Total des valeurs : {count}",
    },
    "expand_to_source_values": {
        "en": "Expand to Source Values",
        "fr": "Développer vers les valeurs sources",
    },
    "add_categories_title": {
        "en": "Add Categories",
        "fr": "Ajouter des catégories",
    },
    "add_categories_info": {
        "en": "Organize your values into groups by assigning each value to a category. This creates a structured table from your list of values.",
        "fr": "Organisez vos valeurs en groupes en attribuant chaque valeur à une catégorie. Cela crée un tableau structuré à partir de votre liste de valeurs.",
    },
    "add_categories_desc": {
        "en": "Group your values into categories to create a structured table. Each value will be assigned to a category based on your matching method.",
        "fr": "Regroupez vos valeurs en catégories pour créer un tableau structuré. Chaque valeur sera assignée à une catégorie selon votre méthode de correspondance.",
    },
    "enter_at_least_one_category": {
        "en": "Please enter at least one category to proceed.",
        "fr": "Veuillez entrer au moins une catégorie pour continuer.",
    },
    "preview_categorization": {
        "en": "Preview categorization",
        "fr": "Aperçu de la catégorisation",
    },
    "categories_to_apply": {
        "en": "Categories to apply: {categories}",
        "fr": "Catégories à appliquer : {categories}",
    },
    "method_smart_matching": {
        "en": "Smart matching (AI)",
        "fr": "Correspondance intelligente (IA)",
    },
    "method_exact_matching": {
        "en": "Exact matching",
        "fr": "Correspondance exacte",
    },
    "strictness_label": {
        "en": "Strictness: {value}",
        "fr": "Rigueur : {value}",
    },
    "apply_categories_btn": {
        "en": "Apply Categories",
        "fr": "Appliquer les catégories",
    },
    "create_connections_title": {
        "en": "Create Connections",
        "fr": "Créer des connexions",
    },
    "create_connections_info": {
        "en": "Pick a column to create a connected view of your data. Each unique value becomes an item that connects to your data rows.",
        "fr": "Choisissez une colonne pour créer une vue connectée de vos données. Chaque valeur unique devient un élément qui se connecte à vos lignes de données.",
    },
    "create_connections_desc": {
        "en": "Select a column to extract as a new item type. Unique values in this column will become items linked to your data rows.",
        "fr": "Sélectionnez une colonne à extraire comme nouveau type d'élément. Les valeurs uniques dans cette colonne deviendront des éléments liés à vos lignes de données.",
    },
    "no_table_data": {
        "en": "No table data available for creating connections.",
        "fr": "Aucune donnée de table disponible pour créer des connexions.",
    },
    "no_columns_available": {
        "en": "No columns available for extraction.",
        "fr": "Aucune colonne disponible pour l'extraction.",
    },
    "select_column_extract": {
        "en": "Select column to extract:",
        "fr": "Sélectionnez la colonne à extraire :",
    },
    "select_column_help": {
        "en": "Choose which column's unique values will become connected items",
        "fr": "Choisissez quelle colonne dont les valeurs uniques deviendront des éléments connectés",
    },
    "single_value_warning": {
        "en": "This column has only **1 unique value**. All {rows} rows will be connected to this single item.",
        "fr": "Cette colonne n'a qu'**une seule valeur unique**. Toutes les {rows} lignes seront connectées à cet élément unique.",
    },
    "unique_values_info": {
        "en": "**{count}** unique values → **{count}** connected items",
        "fr": "**{count}** valeurs uniques → **{count}** éléments connectés",
    },
    "item_type_name_label": {
        "en": "Name for this type of item:",
        "fr": "Nom pour ce type d'élément :",
    },
    "item_type_placeholder": {
        "en": "e.g., Department, Category, Region",
        "fr": "ex: Département, Catégorie, Région",
    },
    "item_type_help": {
        "en": "Name for the new item type",
        "fr": "Nom pour le nouveau type d'élément",
    },
    "connection_type_label": {
        "en": "How should items be connected?",
        "fr": "Comment les éléments doivent-ils être connectés ?",
    },
    "connection_type_placeholder": {
        "en": "e.g., BELONGS_TO, HAS_CATEGORY, IN_REGION",
        "fr": "ex: APPARTIENT_A, A_CATEGORIE, DANS_REGION",
    },
    "connection_type_help": {
        "en": "Label for the connection between items",
        "fr": "Libellé pour la connexion entre les éléments",
    },
    "enter_item_type_name": {
        "en": "Please enter a name for the item type.",
        "fr": "Veuillez entrer un nom pour le type d'élément.",
    },
    "enter_connection_type": {
        "en": "Please enter a connection type.",
        "fr": "Veuillez entrer un type de connexion.",
    },
    "create_connections_btn": {
        "en": "Create Connections",
        "fr": "Créer les connexions",
    },
    "step_1_of_3_columns": {
        "en": "Step 1 of 3: Select Columns to Connect",
        "fr": "Étape 1 sur 3 : Sélectionner les colonnes à connecter",
    },
    "click_columns_instruction": {
        "en": "Click on columns that might link your files together (like IDs or codes):",
        "fr": "Cliquez sur les colonnes qui pourraient relier vos fichiers (comme des IDs ou des codes) :",
    },
    "no_files_to_analyze": {
        "en": "No files found to analyze.",
        "fr": "Aucun fichier trouvé à analyser.",
    },
    "columns_selected": {
        "en": "**{count} columns selected** from {files} files",
        "fr": "**{count} colonnes sélectionnées** parmi {files} fichiers",
    },
    "files_with_columns": {
        "en": "**{files} files** with **{columns} columns** - click to select",
        "fr": "**{files} fichiers** avec **{columns} colonnes** - cliquez pour sélectionner",
    },
    "legend_identifier": {
        "en": "Likely identifier",
        "fr": "Identifiant probable",
    },
    "legend_high_uniqueness": {
        "en": "High uniqueness",
        "fr": "Haute unicité",
    },
    "legend_click_select": {
        "en": "Click to select/deselect",
        "fr": "Cliquez pour sélectionner/désélectionner",
    },
    "selected_columns_count": {
        "en": "Selected columns ({count})",
        "fr": "Colonnes sélectionnées ({count})",
    },
    "from_file": {
        "en": "from",
        "fr": "de",
    },
    "clear_all_btn": {
        "en": "Clear All",
        "fr": "Tout effacer",
    },
    "continue_arrow": {
        "en": "Continue →",
        "fr": "Continuer →",
    },
    "select_at_least_2": {
        "en": "Select at least 2 columns",
        "fr": "Sélectionnez au moins 2 colonnes",
    },
    "step_2_of_3_connections": {
        "en": "Step 2 of 3: Finding Connections",
        "fr": "Étape 2 sur 3 : Recherche de connexions",
    },
    "select_columns_step1_first": {
        "en": "Please select at least 2 columns in Step 1 first.",
        "fr": "Veuillez d'abord sélectionner au moins 2 colonnes à l'étape 1.",
    },
    "back_to_step_1": {
        "en": "← Back to Step 1",
        "fr": "← Retour à l'étape 1",
    },
    "select_from_2_files": {
        "en": "Select columns from at least 2 different files to create connections between them.",
        "fr": "Sélectionnez des colonnes d'au moins 2 fichiers différents pour créer des connexions entre eux.",
    },
    "whats_happening": {
        "en": "What's happening here?",
        "fr": "Que se passe-t-il ici ?",
    },
    "whats_happening_desc": {
        "en": "I'm looking at your selected columns to find **matching items** between your files. For example, if both files have a school code, I'll use that to connect related information together.",
        "fr": "J'examine vos colonnes sélectionnées pour trouver des **éléments correspondants** entre vos fichiers. Par exemple, si les deux fichiers ont un code d'école, je l'utiliserai pour connecter les informations liées.",
    },
    "your_files_to_connect": {
        "en": "Your Files to Connect",
        "fr": "Vos fichiers à connecter",
    },
    "items_label": {
        "en": "Items:",
        "fr": "Éléments :",
    },
    "link_column_label": {
        "en": "Link column:",
        "fr": "Colonne de liaison :",
    },
    "preview_sample_values": {
        "en": "Preview sample values from {file}",
        "fr": "Aperçu des valeurs d'exemple de {file}",
    },
    "sample_values_link_column": {
        "en": "Sample values from the link column:",
        "fr": "Valeurs d'exemple de la colonne de liaison :",
    },
    "how_items_connected": {
        "en": "How Items Will Be Connected",
        "fr": "Comment les éléments seront connectés",
    },
    "matching": {
        "en": "matching",
        "fr": "correspondance",
    },
    "smart_matching_enabled": {
        "en": "Smart Matching Enabled",
        "fr": "Correspondance intelligente activée",
    },
    "smart_matching_desc": {
        "en": "I'll look for items in both files that share the **same or similar values** in your selected columns, then link them together automatically.",
        "fr": "Je rechercherai les éléments dans les deux fichiers qui partagent les **mêmes valeurs ou des valeurs similaires** dans vos colonnes sélectionnées, puis les lierai automatiquement.",
    },
    "how_strict_matching": {
        "en": "How strict should the matching be?",
        "fr": "Quelle rigueur pour la correspondance ?",
    },
    "strict_looser_desc": {
        "en": "Stricter = fewer connections but more accurate; Looser = more connections but may include mismatches",
        "fr": "Plus strict = moins de connexions mais plus précises ; Plus souple = plus de connexions mais peut inclure des erreurs",
    },
    "very_strict": {
        "en": "Very Strict (exact matches only)",
        "fr": "Très strict (correspondances exactes uniquement)",
    },
    "strict": {
        "en": "Strict (high confidence matches)",
        "fr": "Strict (correspondances haute confiance)",
    },
    "balanced_recommended": {
        "en": "Balanced (recommended)",
        "fr": "Équilibré (recommandé)",
    },
    "loose": {
        "en": "Loose (may include partial matches)",
        "fr": "Souple (peut inclure des correspondances partielles)",
    },
    "preview_connections": {
        "en": "Preview Connections",
        "fr": "Aperçu des connexions",
    },
    "finding_matching_items": {
        "en": "Finding matching items between your files...",
        "fr": "Recherche des éléments correspondants entre vos fichiers...",
    },
    "preview_not_available": {
        "en": "Preview not available. Matching will be performed in the next step.",
        "fr": "Aperçu non disponible. La correspondance sera effectuée à l'étape suivante.",
    },
    "error_during_preview": {
        "en": "Error during preview: {error}",
        "fr": "Erreur lors de l'aperçu : {error}",
    },
    "found_connections": {
        "en": "Found {count} connections!",
        "fr": "{count} connexions trouvées !",
    },
    "average_match_confidence": {
        "en": "Average match confidence: **{score}**",
        "fr": "Confiance moyenne des correspondances : **{score}**",
    },
    "sample_connections_found": {
        "en": "Sample connections found:",
        "fr": "Exemples de connexions trouvées :",
    },
    "no_connections_at_threshold": {
        "en": "No connections found at this strictness level. Try moving the slider to the left for looser matching.",
        "fr": "Aucune connexion trouvée à ce niveau de rigueur. Essayez de déplacer le curseur vers la gauche pour une correspondance plus souple.",
    },
    "back_arrow": {
        "en": "← Back",
        "fr": "← Retour",
    },
    "step_3_of_3_joined": {
        "en": "Step 3 of 3: Your Joined Dataset",
        "fr": "Étape 3 sur 3 : Votre jeu de données joint",
    },
    "joined_l3_desc": {
        "en": "Here's your unified L3 table built from semantic connections:",
        "fr": "Voici votre table L3 unifiée construite à partir de connexions sémantiques :",
    },
    "no_connections_defined": {
        "en": "No connections defined. Go back to Step 2 to connect your columns.",
        "fr": "Aucune connexion définie. Retournez à l'étape 2 pour connecter vos colonnes.",
    },
    "back_to_step_2": {
        "en": "← Back to Step 2",
        "fr": "← Retour à l'étape 2",
    },
    "building_joined_table": {
        "en": "Building joined table using row-vector semantic matching...",
        "fr": "Construction de la table jointe avec correspondance sémantique par vecteur-ligne...",
    },
    "building_joined_table_simple": {
        "en": "Building joined table...",
        "fr": "Construction de la table jointe...",
    },
    "could_not_create_joined": {
        "en": "Could not create joined table. Try lowering the matching strictness in Step 2.",
        "fr": "Impossible de créer la table jointe. Essayez de réduire la rigueur de correspondance à l'étape 2.",
    },
    "rows_label": {
        "en": "Rows",
        "fr": "Lignes",
    },
    "columns_label": {
        "en": "Columns",
        "fr": "Colonnes",
    },
    "similarity_threshold": {
        "en": "Similarity Threshold",
        "fr": "Seuil de similarité",
    },
    "connections_used": {
        "en": "Connections Used",
        "fr": "Connexions utilisées",
    },
    "matching_method": {
        "en": "Matching method:",
        "fr": "Méthode de correspondance :",
    },
    "row_vector_semantic": {
        "en": "Row-vector semantic matching (per spec FR-003)",
        "fr": "Correspondance sémantique par vecteur-ligne (selon spec FR-003)",
    },
    "exact_match": {
        "en": "exact match",
        "fr": "correspondance exacte",
    },
    "semantic": {
        "en": "semantic",
        "fr": "sémantique",
    },
    "preview_joined_l3": {
        "en": "Preview of joined L3 dataset:",
        "fr": "Aperçu du jeu de données L3 joint :",
    },
    "rows_to_preview": {
        "en": "Rows to preview",
        "fr": "Lignes à prévisualiser",
    },
    "column_details": {
        "en": "Column details",
        "fr": "Détails des colonnes",
    },
    "non_null": {
        "en": "non-null",
        "fr": "non-nuls",
    },
    "confirm_use_dataset": {
        "en": "Confirm & Use This Dataset",
        "fr": "Confirmer et utiliser ce jeu de données",
    },
    "dataset_confirmed": {
        "en": "Dataset confirmed! Your L3 table is ready.",
        "fr": "Jeu de données confirmé ! Votre table L3 est prête.",
    },

    # ============================================
    # ENTITY TABS
    # ============================================
    "found_items_connections": {
        "en": "Found {items} items across {categories} categories, with {connections} connections",
        "fr": "{items} éléments trouvés dans {categories} catégories, avec {connections} connexions",
    },
    "no_items_connections": {
        "en": "No items or connections found in your data.",
        "fr": "Aucun élément ni connexion trouvé dans vos données.",
    },
    "no_data_to_display": {
        "en": "No data to display.",
        "fr": "Aucune donnée à afficher.",
    },
    "items_with_connections": {
        "en": "{count} items with connections shown",
        "fr": "{count} éléments avec connexions affichés",
    },
    "each_record_shows": {
        "en": "Each record shows what it connects to",
        "fr": "Chaque enregistrement montre ses connexions",
    },
    "total_items_categories": {
        "en": "{count} total items across all categories",
        "fr": "{count} éléments au total dans toutes les catégories",
    },
    "total_connections_types": {
        "en": "{count} total connections across all types",
        "fr": "{count} connexions au total de tous types",
    },
    "entity_items_count": {
        "en": "{count} {type} items",
        "fr": "{count} éléments {type}",
    },
    "relationship_connections_count": {
        "en": "{count} {type} connections",
        "fr": "{count} connexions {type}",
    },
    "showing_first_of": {
        "en": "Showing first {max} of {total} items",
        "fr": "Affichage des {max} premiers sur {total} éléments",
    },
    "use_this_data": {
        "en": "Use this data",
        "fr": "Utiliser ces données",
    },
    "all_items_connections": {
        "en": "All (Items + Connections)",
        "fr": "Tout (Éléments + Connexions)",
    },
    "all_items_label": {
        "en": "All Items",
        "fr": "Tous les éléments",
    },
    "all_connections_label": {
        "en": "All Connections",
        "fr": "Toutes les connexions",
    },
    "items_suffix": {
        "en": "items",
        "fr": "éléments",
    },
    "connections_suffix": {
        "en": "connections",
        "fr": "connexions",
    },

    # ============================================
    # MAIN APP - BRANDING & NAVIGATION
    # ============================================
    "brand_tagline": {
        "en": "The next stage of open data",
        "fr": "La prochaine étape des données ouvertes",
    },
    "descent_progress": {
        "en": "DESCENT PROGRESS (L4 → L0)",
        "fr": "PROGRESSION DESCENTE (L4 → L0)",
    },
    "ascent_progress": {
        "en": "ASCENT PROGRESS (L0 → L3)",
        "fr": "PROGRESSION ASCENSION (L0 → L3)",
    },

    # ============================================
    # MAIN APP - WORKFLOW STEPS (DESCENT)
    # ============================================
    "step_upload_title": {
        "en": "Unlinkable datasets",
        "fr": "Données non-structurées",
    },
    "step_upload_desc": {
        "en": "Upload your raw data files (CSV format)",
        "fr": "Chargez vos fichiers de données bruts (format CSV)",
    },
    "step_entities_title": {
        "en": "Linkable data",
        "fr": "Données liables",
    },
    "step_entities_desc": {
        "en": "What are the main things you want to see in your connected information?",
        "fr": "Quels sont les éléments principaux que vous voulez voir dans vos informations connectées ?",
    },
    "step_domains_title": {
        "en": "Table",
        "fr": "Tableau de données",
    },
    "step_domains_desc": {
        "en": "What categories do you want to organize your data by?",
        "fr": "Par quelles catégories voulez-vous organiser vos données ?",
    },
    "step_features_title": {
        "en": "Vector",
        "fr": "Vecteur de données",
    },
    "step_features_desc": {
        "en": "What values do you want to extract?",
        "fr": "Quelles valeurs voulez-vous extraire ?",
    },
    "step_aggregation_title": {
        "en": "Datum",
        "fr": "Datum",
    },
    "step_aggregation_desc": {
        "en": "What computation do you want to run on your values?",
        "fr": "Quel calcul voulez-vous effectuer sur vos valeurs ?",
    },
    "step_results_title": {
        "en": "Analytic core",
        "fr": "Cœur analytique",
    },
    "step_results_desc": {
        "en": "View your computed results",
        "fr": "Visualisez vos résultats calculés",
    },

    # ============================================
    # MAIN APP - WORKFLOW STEPS (ASCENT)
    # ============================================
    "ascent_recover_title": {
        "en": "Datum",
        "fr": "Datum",
    },
    "ascent_recover_desc": {
        "en": "L0 → L1: Recover source values",
        "fr": "L0 → L1 : Récupérer les valeurs sources",
    },
    "ascent_dimension_title": {
        "en": "Vector",
        "fr": "Vecteur de données",
    },
    "ascent_dimension_desc": {
        "en": "L1 → L2: Define new categories",
        "fr": "L1 → L2 : Définir de nouvelles catégories",
    },
    "ascent_linkage_title": {
        "en": "Table",
        "fr": "Tableau de données",
    },
    "ascent_linkage_desc": {
        "en": "L2 → L3: Enrich with linkage keys",
        "fr": "L2 → L3 : Enrichir avec des clés de liaison",
    },
    "ascent_final_title": {
        "en": "Linkable data",
        "fr": "Données liables",
    },
    "ascent_final_desc": {
        "en": "Final verification",
        "fr": "Vérification finale",
    },

    # ============================================
    # MAIN APP - BUTTONS & MESSAGES
    # ============================================
    "reset_analyze": {
        "en": "Reset and Re-analyze",
        "fr": "Réinitialiser et ré-analyser",
    },
    "continue_arrow": {
        "en": "Continue →",
        "fr": "Continuer →",
    },
    "back_arrow": {
        "en": "← Back",
        "fr": "← Retour",
    },
    "configuration_complete": {
        "en": "Configuration complete! Moving to next step...",
        "fr": "Configuration terminée ! Passage à l'étape suivante...",
    },
    "could_not_analyze": {
        "en": "Could not automatically analyze your files. Please continue manually.",
        "fr": "Impossible d'analyser automatiquement vos fichiers. Veuillez continuer manuellement.",
    },
    "ai_structure_title": {
        "en": "AI-Assisted Item Definition",
        "fr": "Définition d'éléments assistée par IA",
    },
    "enter_openai_key": {
        "en": "Please enter your OpenAI API key",
        "fr": "Veuillez entrer votre clé API OpenAI",
    },
    "structure_generated": {
        "en": "Structure generated successfully!",
        "fr": "Structure générée avec succès !",
    },
    "ensure_ollama_running": {
        "en": "Make sure Ollama is running: `ollama serve`",
        "fr": "Assurez-vous qu'Ollama fonctionne : `ollama serve`",
    },
    "info_built_instantly": {
        "en": "Your connected information will be built instantly. No setup needed.",
        "fr": "Vos informations connectées seront construites instantanément. Aucune configuration requise.",
    },
    "how_to_connect_files": {
        "en": "How to connect your files:",
        "fr": "Comment connecter vos fichiers :",
    },
    "exact_matching_info": {
        "en": "Exact matching connects items that have the same identifier values.",
        "fr": "La correspondance exacte connecte les éléments ayant les mêmes valeurs d'identifiant.",
    },
    "connected_info_built": {
        "en": "Connected information built successfully!",
        "fr": "Informations connectées construites avec succès !",
    },
    "no_connected_view": {
        "en": "No connected view available yet. Please complete the previous step first.",
        "fr": "Aucune vue connectée disponible. Veuillez d'abord terminer l'étape précédente.",
    },
    "no_items_found": {
        "en": "No items found. Please check the previous step.",
        "fr": "Aucun élément trouvé. Veuillez vérifier l'étape précédente.",
    },
    "no_suitable_columns": {
        "en": "No suitable columns found for categorization",
        "fr": "Aucune colonne appropriée trouvée pour la catégorisation",
    },
    "enter_categories": {
        "en": "Enter the categories you want to group by:",
        "fr": "Entrez les catégories par lesquelles grouper :",
    },
    "categorizing_data": {
        "en": "Categorizing data by domains...",
        "fr": "Catégorisation des données par domaines...",
    },
    "select_column_extract": {
        "en": "Select a column to extract from your categorized data.",
        "fr": "Sélectionnez une colonne à extraire de vos données catégorisées.",
    },
    "no_categorized_data": {
        "en": "No categorized data available. Please complete the previous step.",
        "fr": "Aucune donnée catégorisée disponible. Veuillez terminer l'étape précédente.",
    },
    "no_columns_available": {
        "en": "No columns available in domain tables.",
        "fr": "Aucune colonne disponible dans les tables de domaines.",
    },
    "choose_calculation": {
        "en": "Choose how to calculate a final result from your values.",
        "fr": "Choisissez comment calculer un résultat final à partir de vos valeurs.",
    },
    "descent_complete": {
        "en": "Descent complete! Here are your results:",
        "fr": "Descente terminée ! Voici vos résultats :",
    },
    "cannot_start_redesign": {
        "en": "Cannot start redesign - descent data incomplete",
        "fr": "Impossible de démarrer le redesign - données de descente incomplètes",
    },
    "items_label": {
        "en": "Items:",
        "fr": "Éléments :",
    },
    "connections_label": {
        "en": "Connections:",
        "fr": "Connexions :",
    },
    "config_complete_l3_ready": {
        "en": "Configuration complete! Your joined L3 dataset is ready.",
        "fr": "Configuration terminée ! Votre jeu de données L3 joint est prêt.",
    },
    "no_l3_available": {
        "en": "No L3 dataset available",
        "fr": "Aucun jeu de données L3 disponible",
    },
    "no_items_matched": {
        "en": "No items matched this domain",
        "fr": "Aucun élément ne correspond à ce domaine",
    },
    "no_atomic_metrics": {
        "en": "No atomic metrics computed yet",
        "fr": "Aucune métrique atomique calculée pour le moment",
    },
    "no_knowledge_graph": {
        "en": "No knowledge graph built yet",
        "fr": "Aucun graphe de connaissances construit pour le moment",
    },
    "graph_viz_info": {
        "en": "Graph visualization is available when data is stored as a knowledge graph.",
        "fr": "La visualisation de graphe est disponible lorsque les données sont stockées comme graphe de connaissances.",
    },
    "node_types": {
        "en": "Node Types:",
        "fr": "Types de nœuds :",
    },
    "graph_statistics": {
        "en": "Graph Statistics:",
        "fr": "Statistiques du graphe :",
    },
    "download_csv_info": {
        "en": "Download each level's data directly as CSV files.",
        "fr": "Téléchargez les données de chaque niveau directement en fichiers CSV.",
    },
    "no_session_to_save": {
        "en": "No active session data to save. Complete the workflow first.",
        "fr": "Aucune donnée de session active à sauvegarder. Terminez d'abord le workflow.",
    },
    "upload_data_first": {
        "en": "Please upload data first in the guided workflow",
        "fr": "Veuillez d'abord charger des données dans le workflow guidé",
    },
    "session_loaded_success": {
        "en": "Session graph loaded successfully!",
        "fr": "Graphe de session chargé avec succès !",
    },
    "loaded_session_summary": {
        "en": "Loaded Session Summary:",
        "fr": "Résumé de la session chargée :",
    },
    "navigation_path": {
        "en": "Navigation Path:",
        "fr": "Chemin de navigation :",
    },
    "ground_truth_l0": {
        "en": "Ground Truth L0:",
        "fr": "Vérité terrain L0 :",
    },
    "l1_not_found": {
        "en": "L1 data not found in session graph",
        "fr": "Données L1 non trouvées dans le graphe de session",
    },
    "session_graph_not_available": {
        "en": "Session graph not available",
        "fr": "Graphe de session non disponible",
    },
    "no_category_column": {
        "en": "No category column found in L2 data",
        "fr": "Aucune colonne de catégorie trouvée dans les données L2",
    },
    "current_l2_table": {
        "en": "Current L2 Table:",
        "fr": "Tableau L2 actuel :",
    },
    "select_linkage_columns": {
        "en": "Select Linkage Key Column(s)",
        "fr": "Sélectionnez les colonnes de clés de liaison",
    },
    "linkage_columns_detected": {
        "en": "These columns were auto-detected as potential identifiers. You can modify this selection.",
        "fr": "Ces colonnes ont été détectées automatiquement comme identifiants potentiels. Vous pouvez modifier cette sélection.",
    },
    "no_linkage_columns_detected": {
        "en": "No obvious linkage columns detected (postal, commune, code, UAI, etc.). You can select any column from the list below.",
        "fr": "Aucune colonne de liaison évidente détectée (postal, commune, code, UAI, etc.). Vous pouvez sélectionner n'importe quelle colonne ci-dessous.",
    },
    "l3_not_found_session": {
        "en": "L3 data not found in session graph",
        "fr": "Données L3 non trouvées dans le graphe de session",
    },
    "selected_linkage_keys": {
        "en": "Selected Linkage Keys:",
        "fr": "Clés de liaison sélectionnées :",
    },
    "demographic_keys_available": {
        "en": "Demographic Linkage Keys Available:",
        "fr": "Clés démographiques disponibles :",
    },
    "simplified_l3_view": {
        "en": "Simplified L3 View (L2 + linkage columns):",
        "fr": "Vue L3 simplifiée (colonnes L2 + liaison) :",
    },
    "showing_l2_plus_linkage": {
        "en": "Showing only L2 columns + selected linkage keys",
        "fr": "Affichage des colonnes L2 + clés de liaison sélectionnées uniquement",
    },
    "ascent_complete_msg": {
        "en": "You've completed the ascent! You can continue exploring with different dimensions or start fresh.",
        "fr": "Ascension terminée ! Vous pouvez continuer l'exploration avec d'autres dimensions ou recommencer.",
    },
    "try_different_dimension": {
        "en": "Try Different Dimension",
        "fr": "Essayer une autre dimension",
    },
    "try_different_linkage": {
        "en": "Try Different Linkage",
        "fr": "Essayer une autre liaison",
    },
    "start_new_ascent": {
        "en": "Start New Ascent",
        "fr": "Nouvelle ascension",
    },
    "define_entities_core": {
        "en": "Please define entities and select a core entity",
        "fr": "Veuillez définir les entités et sélectionner une entité principale",
    },
    "generated_data_model": {
        "en": "Generated Data Model",
        "fr": "Modèle de données généré",
    },
    "nodes_label": {
        "en": "Nodes:",
        "fr": "Nœuds :",
    },
    "relationships_label": {
        "en": "Relationships:",
        "fr": "Relations :",
    },
    "entity_mapping": {
        "en": "Entity Mapping",
        "fr": "Mappage d'entités",
    },
    "map_entity_to_csv": {
        "en": "Map each entity from the data model to a CSV file and columns",
        "fr": "Mappez chaque entité du modèle de données vers un fichier CSV et ses colonnes",
    },
    "relationship_mapping": {
        "en": "Relationship Mapping",
        "fr": "Mappage de relations",
    },
    "define_entity_connections": {
        "en": "Define how entities connect: Key Matching or Semantic Similarity",
        "fr": "Définissez comment les entités se connectent : correspondance de clés ou similarité sémantique",
    },
    "no_entity_nodes": {
        "en": "No entity nodes found in the graph.",
        "fr": "Aucun nœud d'entité trouvé dans le graphe.",
    },
    "define_domain_categories": {
        "en": "Define Domain Categories",
        "fr": "Définir les catégories de domaines",
    },
    "choose_entity_column": {
        "en": "Choose an entity type and column for domain categorization.",
        "fr": "Choisissez un type d'entité et une colonne pour la catégorisation par domaines.",
    },
    "source_data_expand": {
        "en": "Source Data (what you're expanding/enriching)",
        "fr": "Données sources (ce que vous enrichissez)",
    },
    "navigation_session_ended": {
        "en": "Navigation session ended!",
        "fr": "Session de navigation terminée !",
    },
    "could_not_resume": {
        "en": "Could not resume session",
        "fr": "Impossible de reprendre la session",
    },
    "continue_free_exploration": {
        "en": "Continue to Free Exploration",
        "fr": "Continuer vers l'exploration libre",
    },
    "save_session_graph": {
        "en": "Save Session Graph",
        "fr": "Sauvegarder le graphe de session",
    },
    "start_new_analysis": {
        "en": "Start New Analysis",
        "fr": "Nouvelle analyse",
    },
    "start_redesign_ascent": {
        "en": "Start Redesign (Ascent)",
        "fr": "Démarrer le redesign (Ascension)",
    },
    "recover_source_values": {
        "en": "Recover Source Values",
        "fr": "Récupérer les valeurs sources",
    },
    "back_to_step_9": {
        "en": "Back to Step 9",
        "fr": "Retour à l'étape 9",
    },
    "back_to_step_10": {
        "en": "Back to Step 10",
        "fr": "Retour à l'étape 10",
    },
    "back_to_step_11": {
        "en": "Back to Step 11",
        "fr": "Retour à l'étape 11",
    },
    "use_column_as_categories": {
        "en": "Use '{column}' values as categories",
        "fr": "Utiliser les valeurs de '{column}' comme catégories",
    },
    "apply_categorization": {
        "en": "Apply Categorization",
        "fr": "Appliquer la catégorisation",
    },
    "complete_enrichment": {
        "en": "Complete Enrichment",
        "fr": "Terminer l'enrichissement",
    },
    "build_connected_info": {
        "en": "Build Connected Information",
        "fr": "Construire les informations connectées",
    },
    "generate_structure": {
        "en": "Generate Structure",
        "fr": "Générer la structure",
    },
    "generate_structure_ai": {
        "en": "Generate Structure with AI",
        "fr": "Générer la structure avec l'IA",
    },
    "extract_values": {
        "en": "Extract Values",
        "fr": "Extraire les valeurs",
    },
    "categorize_data": {
        "en": "Categorize Data",
        "fr": "Catégoriser les données",
    },
    "compute_metrics": {
        "en": "Compute Metrics",
        "fr": "Calculer les métriques",
    },
    "view_results": {
        "en": "View Results",
        "fr": "Voir les résultats",
    },
    "select_aggregation": {
        "en": "Select aggregation method",
        "fr": "Sélectionner la méthode d'agrégation",
    },
    "select_column": {
        "en": "Select column",
        "fr": "Sélectionner une colonne",
    },
    "llm_provider": {
        "en": "LLM Provider",
        "fr": "Fournisseur LLM",
    },
    "model_name": {
        "en": "Model name",
        "fr": "Nom du modèle",
    },
    "openai_api_key": {
        "en": "OpenAI API Key",
        "fr": "Clé API OpenAI",
    },
    "entities_input_label": {
        "en": "Entity names",
        "fr": "Noms des entités",
    },
    "core_entity_label": {
        "en": "Core entity",
        "fr": "Entité principale",
    },
    "similarity_threshold": {
        "en": "Similarity threshold",
        "fr": "Seuil de similarité",
    },
    "upload_csv_files": {
        "en": "Upload your CSV files",
        "fr": "Chargez vos fichiers CSV",
    },
    "upload_session_graph": {
        "en": "Upload a saved session graph (.json)",
        "fr": "Charger un graphe de session sauvegardé (.json)",
    },

    # ============================================
    # ASCENT PHASE - STEP LABELS
    # ============================================
    "source_values_available": {
        "en": "Source Values Available",
        "fr": "Valeurs sources disponibles",
    },
    "preview_l1_data": {
        "en": "Preview L1 Data (first 10 rows)",
        "fr": "Aperçu des données L1 (10 premières lignes)",
    },
    "recover_source_values_btn": {
        "en": "Recover Source Values",
        "fr": "Récupérer les valeurs sources",
    },
    "l1_not_found_warning": {
        "en": "L1 data not found in session graph",
        "fr": "Données L1 non trouvées dans le graphe de session",
    },
    "session_graph_unavailable": {
        "en": "Session graph not available",
        "fr": "Graphe de session non disponible",
    },
    "step_8_add_dimension": {
        "en": "Step 8: Add new dimension (L1 → L2)",
        "fr": "Étape 8 : Ajouter une nouvelle dimension (L1 → L2)",
    },
    "current_l1_values": {
        "en": "Current L1 Values",
        "fr": "Valeurs L1 actuelles",
    },
    "ascent_categorize_info": {
        "en": "Define categories to organize your L1 values. You can:\n1. Enter custom categories manually, OR\n2. **Use unique values from a column** as categories directly",
        "fr": "Définissez des catégories pour organiser vos valeurs L1. Vous pouvez :\n1. Entrer des catégories manuellement, OU\n2. **Utiliser les valeurs uniques d'une colonne** comme catégories",
    },
    "step_9_linkage": {
        "en": "Step 9: Enrich with linkage keys (L2 → L3)",
        "fr": "Étape 9 : Enrichir avec des clés de liaison (L2 → L3)",
    },
    "step_10_verification": {
        "en": "Step 10: Final Verification",
        "fr": "Étape 10 : Vérification finale",
    },
    "ascent_completed": {
        "en": "Ascent completed! Your redesigned dataset is ready.",
        "fr": "Ascension terminée ! Votre jeu de données redesigné est prêt.",
    },
    "rows_label": {
        "en": "rows",
        "fr": "lignes",
    },
    "from_l3": {
        "en": "from L3",
        "fr": "depuis L3",
    },
    "columns_label": {
        "en": "columns",
        "fr": "colonnes",
    },
    "all_available_columns": {
        "en": "All Available Columns from L3 ({count})",
        "fr": "Toutes les colonnes disponibles de L3 ({count})",
    },
    "enriched_l3_success": {
        "en": "Enriched L3 table: {rows} rows, {cols} columns",
        "fr": "Tableau L3 enrichi : {rows} lignes, {cols} colonnes",
    },
    "row_count_mismatch": {
        "en": "Row count mismatch: L2={l2}, L3={l3}. Using L3 directly.",
        "fr": "Différence de lignes : L2={l2}, L3={l3}. Utilisation directe de L3.",
    },
    "l3_data_not_found": {
        "en": "L3 data not found in session graph",
        "fr": "Données L3 non trouvées dans le graphe de session",
    },
    "unique_values_count": {
        "en": "{count} unique values",
        "fr": "{count} valeurs uniques",
    },
    "enriched_l3_table": {
        "en": "Enriched L3 Table",
        "fr": "Tableau L3 enrichi",
    },
    "rows_metric": {
        "en": "Rows",
        "fr": "Lignes",
    },
    "columns_metric": {
        "en": "Columns",
        "fr": "Colonnes",
    },
    "new_dimension_label": {
        "en": "New Dimension: {column}",
        "fr": "Nouvelle dimension : {column}",
    },
    "selected_linkage_keys_label": {
        "en": "Selected Linkage Keys:",
        "fr": "Clés de liaison sélectionnées :",
    },
    "demographic_linkage_keys_label": {
        "en": "Demographic Linkage Keys Available:",
        "fr": "Clés démographiques disponibles :",
    },
    "simplified_l3_view": {
        "en": "Simplified L3 View (L2 + linkage columns):",
        "fr": "Vue L3 simplifiée (colonnes L2 + clés de liaison) :",
    },
    "simplified_l3_caption": {
        "en": "Showing only L2 columns + selected linkage keys",
        "fr": "Affichage des colonnes L2 + clés de liaison sélectionnées uniquement",
    },
    "showing_columns_of_total": {
        "en": "Showing {count} columns of {total} total",
        "fr": "Affichage de {count} colonnes sur {total}",
    },
    "full_l3_table": {
        "en": "Full L3 Table ({count} columns)",
        "fr": "Tableau L3 complet ({count} colonnes)",
    },
    "all_columns": {
        "en": "All Columns",
        "fr": "Toutes les colonnes",
    },
    "export_ascent_artifacts": {
        "en": "Export Ascent Artifacts",
        "fr": "Exporter les artefacts d'ascension",
    },
    "download_l1_values": {
        "en": "Download L1 Values",
        "fr": "Télécharger les valeurs L1",
    },
    "download_l2_categorized": {
        "en": "Download L2 Categorized",
        "fr": "Télécharger L2 catégorisé",
    },
    "download_l3_enriched": {
        "en": "Download L3 Enriched",
        "fr": "Télécharger L3 enrichi",
    },
    "continue_exploration": {
        "en": "Continue Exploration",
        "fr": "Continuer l'exploration",
    },
    "ascent_complete_info": {
        "en": "You've completed the ascent! You can continue exploring with different dimensions or start fresh.",
        "fr": "Ascension terminée ! Vous pouvez continuer l'exploration avec d'autres dimensions ou recommencer.",
    },
    "try_different_dimension": {
        "en": "Try Different Dimension",
        "fr": "Autre dimension",
    },
    "try_different_linkage": {
        "en": "Try Different Linkage",
        "fr": "Autre liaison",
    },
    "start_new_ascent": {
        "en": "Start New Ascent",
        "fr": "Nouvelle ascension",
    },
    "step_7_recover": {
        "en": "Step 7: Recover Source Values (L0 → L1)",
        "fr": "Étape 7 : Récupérer les valeurs sources (L0 → L1)",
    },
    "no_suitable_columns": {
        "en": "No suitable columns found for categorization",
        "fr": "Aucune colonne appropriée trouvée pour la catégorisation",
    },
    "enter_categories": {
        "en": "Enter the categories you want to group by:",
        "fr": "Entrez les catégories de regroupement :",
    },
    "what_are_linkage_keys": {
        "en": "What are linkage keys?",
        "fr": "Qu'est-ce qu'une clé de liaison ?",
    },
    "linkage_keys_description": {
        "en": "Linkage keys are columns that can connect your data to external datasets:",
        "fr": "Les clés de liaison sont des colonnes qui peuvent connecter vos données à des sources externes :",
    },
    "postal_codes_link": {
        "en": "**Postal codes** can link to demographic data",
        "fr": "**Codes postaux** pour lier aux données démographiques",
    },
    "commune_names_link": {
        "en": "**Commune names** can link to geographic/administrative data",
        "fr": "**Noms de communes** pour lier aux données administratives",
    },
    "uai_codes_link": {
        "en": "**UAI codes** (education) can link to official school databases",
        "fr": "**Codes UAI** pour lier aux bases officielles de l'éducation",
    },
    "available_columns_from_datasets": {
        "en": "Available columns from original datasets: {count} columns",
        "fr": "Colonnes disponibles depuis les données originales : {count} colonnes",
    },
    "select_linkage_key_columns": {
        "en": "Select Linkage Key Column(s):",
        "fr": "Sélectionner les colonnes de liaison :",
    },
    "auto_detected_caption": {
        "en": "These columns were auto-detected as potential identifiers. You can modify this selection.",
        "fr": "Ces colonnes ont été détectées comme identifiants potentiels. Vous pouvez modifier cette sélection.",
    },
    "select_columns_linkage": {
        "en": "Select columns to expose as linkage keys",
        "fr": "Colonnes à exposer comme clés de liaison",
    },
    "columns_for_joins_help": {
        "en": "These columns will be prominently displayed for future joins",
        "fr": "Ces colonnes seront mises en évidence pour des jointures futures",
    },
    "no_linkage_detected": {
        "en": "No obvious linkage columns detected (postal, commune, code, UAI, etc.). You can select any column from the list below.",
        "fr": "Aucune colonne de liaison évidente détectée. Vous pouvez sélectionner n'importe quelle colonne ci-dessous.",
    },
    "choose_unique_identifiers_help": {
        "en": "Choose columns that contain unique identifiers useful for your analysis",
        "fr": "Choisissez des colonnes contenant des identifiants uniques",
    },
    "descent_mode_label": {
        "en": "DESCENT",
        "fr": "DESCENTE",
    },
    "ascent_mode_label": {
        "en": "ASCENT",
        "fr": "ASCENSION",
    },
    "pdf_not_found": {
        "en": "PDF document not found. Please download it manually.",
        "fr": "Document PDF introuvable. Veuillez le télécharger manuellement.",
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
    # Initialize language in session state if not present
    if SESSION_KEY_LANGUAGE not in st.session_state:
        st.session_state[SESSION_KEY_LANGUAGE] = DEFAULT_LANGUAGE

    current_lang = st.session_state[SESSION_KEY_LANGUAGE]
    lang_keys = list(SUPPORTED_LANGUAGES.keys())

    selected = st.sidebar.radio(
        label="🌐",
        options=lang_keys,
        format_func=lambda x: SUPPORTED_LANGUAGES[x],
        index=lang_keys.index(current_lang) if current_lang in lang_keys else 0,
        horizontal=True,
        key="language_radio",
    )

    # If language changed, update and rerun to refresh entire page
    if selected != current_lang:
        st.session_state[SESSION_KEY_LANGUAGE] = selected
        st.rerun()
