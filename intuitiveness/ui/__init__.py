"""
UI Components for Data Redesign Method

This package contains UI components for:
- Decision-tree navigation and drag-and-drop relationship builder (002-ascent-functionality)
- Level-specific display components (003-level-dataviz-display)
- Ascent UI forms for L0→L1, L1→L2, L2→L3 transitions (004-ascent-precision)

Features: 002-ascent-functionality, 003-level-dataviz-display, 004-ascent-precision
"""

from .drag_drop import DragDropRelationshipBuilder, get_entities_from_dataframe
from .decision_tree import DecisionTreeComponent, render_simple_tree
from .json_visualizer import JsonVisualizer, render_navigation_export

# Level-specific display components (003-level-dataviz-display)
from .level_displays import (
    NavigationDirection,
    DisplayType,
    LevelDisplayConfig,
    LEVEL_DISPLAY_MAPPING,
    get_display_level,
    render_l4_file_list,
    render_l2_domain_table,
    render_l1_vector,
    render_l0_datum,
    render_navigation_direction_indicator,
)

# Entity/relationship tab components (003-level-dataviz-display)
from .entity_tabs import (
    EntityTabData,
    RelationshipTabData,
    CombinedTabData,
    extract_entity_tabs,
    extract_relationship_tabs,
    render_entity_relationship_tabs,
    create_combined_all_table,
    create_combined_entity_table,
    create_combined_relationship_table,
    get_graph_summary,
)

# Recovery banner (005-session-persistence)
from .recovery_banner import (
    RecoveryAction,
    render_recovery_banner,
    render_start_fresh_button,
    render_start_fresh_confirmation,
    format_time_ago,
)

# Internationalization (006-playwright-mcp-e2e)
from .i18n import (
    t,
    get_language,
    set_language,
    render_language_toggle,
    render_language_toggle_compact,
    TRANSLATIONS,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    SESSION_KEY_LANGUAGE,
)

# Ascent UI forms (004-ascent-precision)
from .ascent_forms import (
    # Form renderers
    render_l0_to_l1_unfold_form,
    render_l1_to_l2_domain_form,
    render_l2_to_l3_entity_form,
    # Shared components
    _render_domain_categorization_inputs,
    _parse_domains,
    _apply_domain_categorization,
    # Constants
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_UNMATCHED_LABEL,
    DEFAULT_DOMAINS,
    # Discovery wizard components (Step 2 simplification)
    render_wizard_step_1_columns,
    render_wizard_step_1_entities,
    render_wizard_step_2_connections,
    render_wizard_step_2_relationships,
    render_wizard_step_3_confirm,
    convert_suggestions_to_mappings,
    _get_wizard_step,
    _set_wizard_step,
    SESSION_KEY_WIZARD_STEP,
    SESSION_KEY_DISCOVERY_RESULTS,
    # Step 3 join helpers
    _build_semantic_mapping,
    _perform_table_join,
)

__all__ = [
    # Drag-and-drop relationship builder (L2→L3)
    'DragDropRelationshipBuilder',
    'get_entities_from_dataframe',
    # Decision tree sidebar component
    'DecisionTreeComponent',
    'render_simple_tree',
    # JSON visualization
    'JsonVisualizer',
    'render_navigation_export',
    # Level-specific displays (003-level-dataviz-display)
    'NavigationDirection',
    'DisplayType',
    'LevelDisplayConfig',
    'LEVEL_DISPLAY_MAPPING',
    'get_display_level',
    'render_l4_file_list',
    'render_l2_domain_table',
    'render_l1_vector',
    'render_l0_datum',
    'render_navigation_direction_indicator',
    # Entity/relationship tabs (003-level-dataviz-display)
    'EntityTabData',
    'RelationshipTabData',
    'CombinedTabData',
    'extract_entity_tabs',
    'extract_relationship_tabs',
    'render_entity_relationship_tabs',
    'create_combined_all_table',
    'create_combined_entity_table',
    'create_combined_relationship_table',
    'get_graph_summary',
    # Ascent UI forms (004-ascent-precision)
    'render_l0_to_l1_unfold_form',
    'render_l1_to_l2_domain_form',
    'render_l2_to_l3_entity_form',
    '_render_domain_categorization_inputs',
    '_parse_domains',
    '_apply_domain_categorization',
    'DEFAULT_SIMILARITY_THRESHOLD',
    'DEFAULT_UNMATCHED_LABEL',
    'DEFAULT_DOMAINS',
    # Discovery wizard components (Step 2 simplification)
    'render_wizard_step_1_columns',
    'render_wizard_step_1_entities',
    'render_wizard_step_2_connections',
    'render_wizard_step_2_relationships',
    'render_wizard_step_3_confirm',
    'convert_suggestions_to_mappings',
    '_get_wizard_step',
    '_set_wizard_step',
    'SESSION_KEY_WIZARD_STEP',
    'SESSION_KEY_DISCOVERY_RESULTS',
    # Step 3 join helpers
    '_build_semantic_mapping',
    '_perform_table_join',
    # Recovery banner (005-session-persistence)
    'RecoveryAction',
    'render_recovery_banner',
    'render_start_fresh_button',
    'render_start_fresh_confirmation',
    'format_time_ago',
    # Internationalization (006-playwright-mcp-e2e)
    't',
    'get_language',
    'set_language',
    'render_language_toggle',
    'render_language_toggle_compact',
    'TRANSLATIONS',
    'SUPPORTED_LANGUAGES',
    'DEFAULT_LANGUAGE',
    'SESSION_KEY_LANGUAGE',
]
