import pandas as pd
import networkx as nx
from typing import List, Dict, Any

def load_csv_as_df(path: str, **kwargs) -> pd.DataFrame:
    """Helper to load CSV into DataFrame."""
    return pd.read_csv(path, **kwargs)

def graph_to_dataframe(graph: nx.Graph, node_type: str, properties: List[str]) -> pd.DataFrame:
    """
    Extracts nodes of a specific type from a NetworkX graph into a DataFrame.
    """
    data = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get('type') == node_type:
            row = {prop: attrs.get(prop) for prop in properties}
            row['id'] = node
            data.append(row)
    return pd.DataFrame(data)
