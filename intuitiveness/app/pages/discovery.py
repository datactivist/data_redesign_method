"""
Discovery Page - L4→L3 Entity and Domain Discovery

Implements Spec 011: Code Simplification
Extracted from streamlit_app.py (lines 1182-1598)

Handles:
- Entity discovery with AI assistance
- Relationship discovery (semantic matching)
- Domain categorization wizard
- L3 graph construction

Usage:
    from intuitiveness.app.pages import render_discovery_page
    render_discovery_page()
"""

import streamlit as st
import networkx as nx
from typing import Dict, List, Any, Optional

from intuitiveness.complexity import Level3Dataset, Level4Dataset
from intuitiveness.interactive import (
    DataModelGenerator,
    SemanticMatcher,
    run_discovery
)
from intuitiveness.ui import (
    render_page_header,
    render_section_header,
    render_wizard_step_2_connections,
    render_wizard_step_2_relationships,
    render_wizard_step_3_confirm,
    convert_suggestions_to_mappings,
    t,
)
from intuitiveness.utils import SessionStateKeys


def render_discovery_page():
    """
    Render L4→L3 discovery workflow page.

    Implements Spec 011: Discovery Page Extraction

    Features:
    - AI-powered entity suggestion
    - Semantic relationship discovery
    - Interactive graph preview
    - L3 dataset construction

    Workflow:
    1. AI suggests entities from raw data columns
    2. User confirms/modifies entity mappings
    3. System discovers relationships (key matching + semantic)
    4. User confirms relationships
    5. L3 graph constructed
    """
    render_page_header(
        title=t('discovery_title'),
        subtitle=t('discovery_subtitle')
    )

    # Get raw data
    raw_data = st.session_state.get(SessionStateKeys.RAW_DATA)
    if not raw_data:
        st.error(t('no_data_loaded'))
        return

    # Run discovery workflow
    wizard_step = st.session_state.get('discovery_wizard_step', 1)

    if wizard_step == 1:
        render_entity_discovery(raw_data)
    elif wizard_step == 2:
        render_relationship_discovery()
    elif wizard_step == 3:
        render_graph_confirmation()
    else:
        st.success(t('discovery_complete'))
        if st.button(t('continue_to_domains')):
            st.session_state[SessionStateKeys.CURRENT_STEP] = 2
            st.rerun()


def render_entity_discovery(raw_data: Dict[str, Any]):
    """
    Step 1: Entity discovery with AI assistance.

    Uses DataModelGenerator to suggest entities from CSV columns.
    """
    render_section_header(t('step_1_entities'))

    with st.spinner(t('analyzing_columns')):
        # Run AI entity suggestion
        generator = DataModelGenerator()
        suggestions = generator.suggest_entities(raw_data)

        st.session_state['entity_suggestions'] = suggestions

    # Display suggestions with confirmation
    st.write(t('suggested_entities_prompt'))

    confirmed_entities = {}
    for source_name, entities in suggestions.items():
        st.subheader(f"📄 {source_name}")
        for entity in entities:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{entity['name']}** - {entity['description']}")
                st.caption(f"Columns: {', '.join(entity['columns'])}")
            with col2:
                if st.checkbox(t('confirm'), value=True, key=f"entity_{source_name}_{entity['name']}"):
                    confirmed_entities[entity['name']] = entity

    if confirmed_entities:
        st.session_state['confirmed_entities'] = confirmed_entities

        if st.button(t('continue_to_relationships'), type="primary"):
            st.session_state['discovery_wizard_step'] = 2
            st.rerun()


def render_relationship_discovery():
    """
    Step 2: Relationship discovery with semantic matching.

    Uses SemanticMatcher to find relationships between entities.
    """
    render_section_header(t('step_2_relationships'))

    confirmed_entities = st.session_state.get('confirmed_entities', {})
    if not confirmed_entities:
        st.error(t('no_entities_confirmed'))
        return

    # Relationship discovery settings
    st.write(t('relationship_discovery_prompt'))

    col1, col2 = st.columns(2)
    with col1:
        relationship_mode = st.selectbox(
            t('matching_strategy'),
            ["key_matching", "semantic", "hybrid"],
            index=2
        )
    with col2:
        threshold = st.slider(
            t('similarity_threshold'),
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )

    if st.button(t('discover_relationships'), type="primary"):
        with st.spinner(t('discovering_relationships')):
            # Run relationship discovery
            raw_data = st.session_state[SessionStateKeys.RAW_DATA]
            discovery_result = run_discovery(
                raw_data,
                confirmed_entities,
                mode=relationship_mode,
                threshold=threshold
            )

            st.session_state['discovery_result'] = discovery_result
            st.session_state['discovery_wizard_step'] = 3
            st.rerun()


def render_graph_confirmation():
    """
    Step 3: Graph preview and confirmation.

    Shows discovered relationships and constructs L3 graph.
    """
    render_section_header(t('step_3_confirm_graph'))

    discovery_result = st.session_state.get('discovery_result')
    if not discovery_result:
        st.error(t('no_discovery_results'))
        return

    # Display relationship summary
    st.write(t('discovered_relationships_summary'))
    st.metric(
        t('relationships_found'),
        len(discovery_result.get('relationships', [])),
        help=t('relationships_help')
    )

    # Show relationship table
    if discovery_result.get('relationships'):
        import pandas as pd
        rels_df = pd.DataFrame(discovery_result['relationships'])
        st.dataframe(rels_df, use_container_width=True)

    # Construct L3 graph
    if st.button(t('build_graph'), type="primary"):
        with st.spinner(t('building_graph')):
            # Build NetworkX graph from discovery result
            G = build_graph_from_discovery(discovery_result)

            # Create L3 dataset
            l3_dataset = Level3Dataset(G)
            st.session_state['datasets']['l3'] = l3_dataset

            st.success(t('graph_built_successfully'))
            st.session_state['discovery_wizard_step'] = 4
            st.rerun()


def build_graph_from_discovery(discovery_result: Dict[str, Any]) -> nx.DiGraph:
    """
    Build NetworkX graph from discovery results.

    Parameters:
    -----------
    discovery_result : Dict[str, Any]
        Discovery results with entities and relationships

    Returns:
    --------
    nx.DiGraph
        Constructed knowledge graph
    """
    G = nx.DiGraph()

    # Add entity nodes
    for entity in discovery_result.get('entities', []):
        G.add_node(
            entity['name'],
            node_type='entity',
            description=entity.get('description', ''),
            columns=entity.get('columns', [])
        )

    # Add relationship edges
    for rel in discovery_result.get('relationships', []):
        G.add_edge(
            rel['source'],
            rel['target'],
            relationship=rel.get('type', 'related_to'),
            confidence=rel.get('confidence', 1.0),
            method=rel.get('method', 'key_matching')
        )

    return G
