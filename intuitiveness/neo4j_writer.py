"""
Neo4j Writer Module

This module provides functions to write data to Neo4j databases.
It's designed to work with the MCP Neo4j tools but can also generate
Cypher queries for manual execution.

Author: Intuitiveness Framework
"""

from typing import Dict, List, Any, Optional
import networkx as nx
from dataclasses import dataclass


@dataclass
class Neo4jWriteResult:
    """Result of a Neo4j write operation."""
    success: bool
    nodes_created: int = 0
    relationships_created: int = 0
    constraints_created: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def generate_constraint_queries(data_model: Dict[str, Any]) -> List[str]:
    """
    Generate Cypher constraint queries from a data model.

    Args:
        data_model: Data model dict with 'nodes' list

    Returns:
        List of Cypher constraint queries
    """
    queries = []

    for node in data_model.get('nodes', []):
        label = node['label']
        key_prop = node['key_property']['name']

        query = f"""CREATE CONSTRAINT {label}_key IF NOT EXISTS
FOR (n:{label}) REQUIRE n.{key_prop} IS UNIQUE"""
        queries.append(query)

    return queries


def generate_node_ingest_query(node: Dict[str, Any]) -> str:
    """
    Generate a Cypher query to ingest nodes.

    Args:
        node: Node definition with label, key_property, properties

    Returns:
        Cypher UNWIND query for batch ingestion
    """
    label = node['label']
    key_prop = node['key_property']['name']

    props = [key_prop]
    for prop in node.get('properties', []):
        if prop['name'] != key_prop:
            props.append(prop['name'])

    set_clause = ", ".join([f"{p}: record.{p}" for p in props])

    return f"""UNWIND $records AS record
MERGE (n:{label} {{{key_prop}: record.{key_prop}}})
SET n += {{{set_clause}}}
RETURN count(n) as created"""


def generate_relationship_ingest_query(
    rel_type: str,
    start_label: str,
    end_label: str,
    start_key: str,
    end_key: str
) -> str:
    """
    Generate a Cypher query to create relationships.

    Args:
        rel_type: Relationship type (e.g., "HAS_SOURCE")
        start_label: Start node label
        end_label: End node label
        start_key: Start node key property name
        end_key: End node key property name

    Returns:
        Cypher UNWIND query for batch relationship creation
    """
    return f"""UNWIND $records AS record
MATCH (a:{start_label} {{{start_key}: record.start_id}})
MATCH (b:{end_label} {{{end_key}: record.end_id}})
MERGE (a)-[r:{rel_type}]->(b)
RETURN count(r) as created"""


def graph_to_neo4j_records(
    graph: nx.Graph,
    data_model: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert a NetworkX graph to Neo4j-ready records.

    Args:
        graph: NetworkX graph with nodes and edges
        data_model: Data model defining the schema

    Returns:
        Dict with 'nodes' and 'relationships' record lists
    """
    result = {
        'nodes': {},
        'relationships': []
    }

    # Get node labels from data model
    node_labels = {n['label']: n for n in data_model.get('nodes', [])}

    # Convert nodes
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('type', 'Entity')

        # Map to data model label
        label = node_type if node_type in node_labels else list(node_labels.keys())[0]

        if label not in result['nodes']:
            result['nodes'][label] = []

        # Get key property name
        key_prop = node_labels[label]['key_property']['name']

        record = {
            key_prop: str(node_id),
            'name': attrs.get('name', str(node_id))
        }

        # Add other properties
        for key, value in attrs.items():
            if key not in ['type', 'name']:
                record[key] = str(value) if value is not None else ''

        result['nodes'][label].append(record)

    # Convert relationships
    for source, target, edge_attrs in graph.edges(data=True):
        source_attrs = graph.nodes.get(source, {})
        target_attrs = graph.nodes.get(target, {})

        source_type = source_attrs.get('type', 'Entity')
        target_type = target_attrs.get('type', 'Entity')

        # Determine relationship type
        rel_type = edge_attrs.get('relation', f"RELATED_TO")

        result['relationships'].append({
            'start_id': str(source),
            'end_id': str(target),
            'start_type': source_type,
            'end_type': target_type,
            'type': rel_type
        })

    return result


def generate_full_ingest_script(
    graph: nx.Graph,
    data_model: Dict[str, Any]
) -> str:
    """
    Generate a complete Cypher script to ingest all data.

    This can be copy-pasted into Neo4j Browser or run via bolt.

    Args:
        graph: NetworkX graph
        data_model: Data model schema

    Returns:
        Complete Cypher script as string
    """
    lines = [
        "// ============================================",
        "// Data Redesign Method - Neo4j Ingest Script",
        "// ============================================",
        "",
        "// Step 1: Create constraints"
    ]

    for query in generate_constraint_queries(data_model):
        lines.append(query + ";")

    lines.extend(["", "// Step 2: Create nodes"])

    records = graph_to_neo4j_records(graph, data_model)

    for label, node_records in records['nodes'].items():
        if not node_records:
            continue

        lines.append(f"\n// Create {label} nodes ({len(node_records)} records)")

        # Create individual MERGE statements (for small datasets)
        for record in node_records[:100]:  # Limit for readability
            props = ", ".join([f'{k}: "{v}"' for k, v in record.items()])
            lines.append(f"MERGE (n:{label} {{{props}}});")

        if len(node_records) > 100:
            lines.append(f"// ... and {len(node_records) - 100} more {label} nodes")

    lines.extend(["", "// Step 3: Create relationships"])

    for rel in records['relationships'][:50]:  # Limit for readability
        lines.append(
            f'MATCH (a {{{list(records["nodes"].keys())[0] if records["nodes"] else "Entity"}_id: "{rel["start_id"]}"}}) '
            f'MATCH (b {{{list(records["nodes"].keys())[0] if records["nodes"] else "Entity"}_id: "{rel["end_id"]}"}}) '
            f'MERGE (a)-[:{rel["type"]}]->(b);'
        )

    if len(records['relationships']) > 50:
        lines.append(f"// ... and {len(records['relationships']) - 50} more relationships")

    return "\n".join(lines)


class Neo4jMCPWriter:
    """
    Writer that uses Neo4j MCP tools to write data.

    This class is designed to be used with the MCP tools but the actual
    MCP calls happen in the Streamlit app where the tools are available.
    """

    def __init__(self, database: str = "intuitiveness"):
        """
        Initialize the writer.

        Args:
            database: Which neo4j database to use ("intuitiveness", "cypher", etc.)
        """
        self.database = database
        self.pending_queries = []

    def prepare_ingest(
        self,
        graph: nx.Graph,
        data_model: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prepare queries for ingestion.

        Returns a list of query specs that can be executed via MCP.

        Args:
            graph: NetworkX graph
            data_model: Data model schema

        Returns:
            List of {query: str, params: dict} specs
        """
        queries = []

        # 1. Constraints
        for constraint_query in generate_constraint_queries(data_model):
            queries.append({
                'query': constraint_query,
                'params': {},
                'type': 'constraint'
            })

        # 2. Node records
        records = graph_to_neo4j_records(graph, data_model)
        node_labels = {n['label']: n for n in data_model.get('nodes', [])}

        for label, node_records in records['nodes'].items():
            if not node_records:
                continue

            node_def = node_labels.get(label, data_model['nodes'][0])
            ingest_query = generate_node_ingest_query(node_def)

            queries.append({
                'query': ingest_query,
                'params': {'records': node_records},
                'type': 'nodes',
                'label': label,
                'count': len(node_records)
            })

        # 3. Relationships
        # Group by type
        rel_by_type = {}
        for rel in records['relationships']:
            rel_type = rel['type']
            if rel_type not in rel_by_type:
                rel_by_type[rel_type] = []
            rel_by_type[rel_type].append({
                'start_id': rel['start_id'],
                'end_id': rel['end_id']
            })

        for rel_type, rel_records in rel_by_type.items():
            # Simplified - assumes first node label for both ends
            first_label = list(node_labels.keys())[0] if node_labels else 'Entity'
            first_key = node_labels[first_label]['key_property']['name'] if node_labels else 'id'

            rel_query = generate_relationship_ingest_query(
                rel_type, first_label, first_label, first_key, first_key
            )

            queries.append({
                'query': rel_query,
                'params': {'records': rel_records},
                'type': 'relationships',
                'rel_type': rel_type,
                'count': len(rel_records)
            })

        self.pending_queries = queries
        return queries

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of pending operations."""
        if not self.pending_queries:
            return {'status': 'empty', 'queries': 0}

        return {
            'status': 'ready',
            'queries': len(self.pending_queries),
            'constraints': sum(1 for q in self.pending_queries if q['type'] == 'constraint'),
            'node_batches': sum(1 for q in self.pending_queries if q['type'] == 'nodes'),
            'relationship_batches': sum(1 for q in self.pending_queries if q['type'] == 'relationships'),
            'total_nodes': sum(q.get('count', 0) for q in self.pending_queries if q['type'] == 'nodes'),
            'total_relationships': sum(q.get('count', 0) for q in self.pending_queries if q['type'] == 'relationships')
        }
