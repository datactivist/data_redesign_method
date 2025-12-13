"""
Data Sources Module
====================

External data source integrations for the Data Redesign Method.

Feature: 008-datagouv-mcp
"""

from intuitiveness.data_sources.mcp_client import MCPClient
from intuitiveness.data_sources.datagouv import DataGouvClient
from intuitiveness.data_sources.nl_query import NLQueryEngine, NLQueryResult, parse_french_query

__all__ = [
    'MCPClient',
    'DataGouvClient',
    'NLQueryEngine',
    'NLQueryResult',
    'parse_french_query',
]
