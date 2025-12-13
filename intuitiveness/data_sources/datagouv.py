"""
Data.gouv.fr MCP Client
========================

Client for querying French open data through the official data.gouv MCP server.
Provides natural language search and SQL queries via Hydra.

Feature: 008-datagouv-mcp
Reference: https://github.com/datagouv/datagouv-mcp
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from intuitiveness.data_sources.mcp_client import MCPClient, MCPResponse


# Official data.gouv MCP endpoint
DATAGOUV_MCP_ENDPOINT = "https://mcp.data.gouv.fr/mcp"


@dataclass
class DataGouvDataset:
    """Represents a data.gouv.fr dataset."""
    id: str
    title: str
    description: str = ""
    organization: str = ""
    created_at: str = ""
    last_modified: str = ""
    frequency: str = ""
    tags: List[str] = field(default_factory=list)
    resources_count: int = 0
    url: str = ""

    @property
    def short_description(self) -> str:
        """Get truncated description for display."""
        if len(self.description) > 200:
            return self.description[:200] + "..."
        return self.description


@dataclass
class DataGouvResource:
    """Represents a resource (file) within a dataset."""
    id: str
    title: str
    format: str = ""
    url: str = ""
    filesize: int = 0
    mime_type: str = ""
    created_at: str = ""
    last_modified: str = ""
    schema_url: Optional[str] = None

    @property
    def size_display(self) -> str:
        """Human-readable file size."""
        if self.filesize < 1024:
            return f"{self.filesize} B"
        elif self.filesize < 1024 * 1024:
            return f"{self.filesize / 1024:.1f} KB"
        elif self.filesize < 1024 * 1024 * 1024:
            return f"{self.filesize / (1024 * 1024):.1f} MB"
        return f"{self.filesize / (1024 * 1024 * 1024):.1f} GB"


class DataGouvClient:
    """
    Client for data.gouv.fr via MCP.

    Provides:
    - Dataset search
    - Resource listing
    - Data querying (via Tabular API / Hydra)
    - CSV/JSON download and parsing

    Example:
        client = DataGouvClient()
        datasets = client.search("collèges France")
        resources = client.list_resources(datasets[0].id)
        df = client.query_data(resources[0].id, limit=100)
    """

    def __init__(self, endpoint: str = DATAGOUV_MCP_ENDPOINT, timeout: int = 60):
        """
        Initialize data.gouv client.

        Args:
            endpoint: MCP server endpoint (defaults to official server)
            timeout: Request timeout in seconds
        """
        self.mcp = MCPClient(endpoint, timeout=timeout)
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Ensure MCP session is initialized."""
        if not self._initialized:
            response = self.mcp.initialize()
            self._initialized = response.success
        return self._initialized

    def search(
        self,
        query: str,
        page: int = 1,
        page_size: int = 10
    ) -> List[DataGouvDataset]:
        """
        Search for datasets using natural language query.

        Args:
            query: Search query (natural language, French supported)
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            List of matching DataGouvDataset objects
        """
        if not self._ensure_initialized():
            return []

        response = self.mcp.call_tool("search_datasets", {
            "query": query,
            "page": page,
            "page_size": page_size
        })

        if not response.success:
            print(f"Search error: {response.error}")
            return []

        datasets = []
        content = response.data.get("content", [])

        for item in content:
            if item.get("type") == "text":
                # Parse the text content which contains dataset info
                # The MCP returns structured data in content blocks
                text = item.get("text", "")
                # Try to extract datasets from the response
                datasets.extend(self._parse_search_results(text, response.data))

        return datasets

    def nl_search(
        self,
        natural_language_query: str,
        hf_token: Optional[str] = None,
        page: int = 1,
        page_size: int = 10
    ) -> Tuple[List[DataGouvDataset], 'NLQueryResult']:
        """
        Search for datasets using natural language query.

        Uses SmolLM3-3B to understand the query and extract keywords,
        then searches data.gouv with those keywords.

        Args:
            natural_language_query: Question in French (e.g., "Quels datasets
                contiennent les résultats scolaires des collèges?")
            hf_token: HuggingFace API token (or set HF_TOKEN env var)
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            Tuple of (list of datasets, NLQueryResult with parsed query info)

        Example:
            datasets, nl_result = client.nl_search(
                "Quelles sont les données sur l'éducation prioritaire?"
            )
            print(f"Keywords used: {nl_result.keywords}")
            for ds in datasets:
                print(f"  - {ds.title}")
        """
        from intuitiveness.data_sources.nl_query import NLQueryEngine, NLQueryResult

        # Parse the natural language query
        try:
            engine = NLQueryEngine(hf_token)
            nl_result = engine.parse_query(natural_language_query)
        except ValueError:
            # No token - use fallback
            engine = NLQueryEngine.__new__(NLQueryEngine)
            engine.hf_token = None
            nl_result = engine._fallback_parse(natural_language_query, "No HF token")

        # Try keywords individually and combine results (data.gouv is picky)
        all_datasets = []
        seen_ids = set()

        for keyword in nl_result.keywords[:4]:
            results = self.search(keyword, page=page, page_size=page_size)
            for ds in results:
                if ds.id not in seen_ids:
                    seen_ids.add(ds.id)
                    all_datasets.append(ds)

        datasets = all_datasets[:page_size]

        return datasets, nl_result

    def _parse_search_results(self, text: str, raw_data: Any) -> List[DataGouvDataset]:
        """Parse search results from MCP response text format."""
        datasets = []

        if not text:
            return datasets

        # Parse the structured text format returned by data.gouv MCP
        # Format:
        # 1. Title
        #    ID: xxx
        #    Organization: xxx
        #    Tags: xxx
        #    Resources: N
        #    URL: xxx

        import re

        # Split by numbered entries (1. , 2. , etc.)
        entries = re.split(r'\n\d+\.\s+', text)

        for entry in entries[1:]:  # Skip the header
            lines = entry.strip().split('\n')
            if not lines:
                continue

            title = lines[0].strip()
            dataset_id = ""
            organization = ""
            tags = []
            resources_count = 0
            url = ""

            for line in lines[1:]:
                line = line.strip()
                if line.startswith('ID:'):
                    dataset_id = line[3:].strip()
                elif line.startswith('Organization:'):
                    organization = line[13:].strip()
                elif line.startswith('Tags:'):
                    tags_str = line[5:].strip()
                    tags = [t.strip() for t in tags_str.split(',') if t.strip()]
                elif line.startswith('Resources:'):
                    try:
                        resources_count = int(line[10:].strip())
                    except:
                        pass
                elif line.startswith('URL:'):
                    url = line[4:].strip()

            if title and dataset_id:
                datasets.append(DataGouvDataset(
                    id=dataset_id,
                    title=title,
                    organization=organization,
                    tags=tags,
                    resources_count=resources_count,
                    url=url
                ))

        return datasets

    def _make_dataset(self, data: Dict) -> DataGouvDataset:
        """Create DataGouvDataset from API response data."""
        return DataGouvDataset(
            id=data.get("id", ""),
            title=data.get("title", "Unknown"),
            description=data.get("description", ""),
            organization=data.get("organization", {}).get("name", "") if isinstance(data.get("organization"), dict) else str(data.get("organization", "")),
            created_at=data.get("created_at", ""),
            last_modified=data.get("last_modified", ""),
            frequency=data.get("frequency", ""),
            tags=[t.get("name", t) if isinstance(t, dict) else str(t) for t in data.get("tags", [])],
            resources_count=len(data.get("resources", [])),
            url=data.get("page", f"https://www.data.gouv.fr/fr/datasets/{data.get('id', '')}/")
        )

    def get_dataset_info(self, dataset_id: str) -> Optional[DataGouvDataset]:
        """
        Get detailed information about a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DataGouvDataset with full details, or None if not found
        """
        if not self._ensure_initialized():
            return None

        response = self.mcp.call_tool("get_dataset_info", {
            "dataset_id": dataset_id
        })

        if not response.success:
            print(f"Get dataset error: {response.error}")
            return None

        # Parse response content
        content = response.data.get("content", [])
        for item in content:
            if item.get("type") == "text":
                # Try to parse JSON from text
                import json
                try:
                    data = json.loads(item.get("text", "{}"))
                    return self._make_dataset(data)
                except:
                    pass

        return None

    def list_resources(self, dataset_id: str) -> List[DataGouvResource]:
        """
        List resources (files) in a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            List of DataGouvResource objects
        """
        if not self._ensure_initialized():
            return []

        response = self.mcp.call_tool("list_dataset_resources", {
            "dataset_id": dataset_id
        })

        if not response.success:
            print(f"List resources error: {response.error}")
            return []

        resources = []
        content = response.data.get("content", [])

        for item in content:
            if item.get("type") == "text":
                # Parse resources from text/JSON
                import json
                try:
                    data = json.loads(item.get("text", "[]"))
                    if isinstance(data, list):
                        for res in data:
                            resources.append(self._make_resource(res))
                    elif isinstance(data, dict) and "resources" in data:
                        for res in data["resources"]:
                            resources.append(self._make_resource(res))
                except:
                    pass

        return resources

    def _make_resource(self, data: Dict) -> DataGouvResource:
        """Create DataGouvResource from API response data."""
        return DataGouvResource(
            id=data.get("id", ""),
            title=data.get("title", "Unknown"),
            format=data.get("format", "").upper(),
            url=data.get("url", ""),
            filesize=data.get("filesize", 0) or 0,
            mime_type=data.get("mime", ""),
            created_at=data.get("created_at", ""),
            last_modified=data.get("last_modified", ""),
            schema_url=data.get("schema", {}).get("url") if isinstance(data.get("schema"), dict) else None
        )

    def query_data(
        self,
        resource_id: str,
        sql_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Optional[pd.DataFrame]:
        """
        Query tabular data from a resource.

        Uses the Tabular API / Hydra for SQL-like queries.

        Args:
            resource_id: Resource identifier
            sql_query: Optional SQL query (if supported)
            limit: Maximum rows to return (max 200 per request)
            offset: Row offset for pagination

        Returns:
            pandas DataFrame with query results, or None on error
        """
        if not self._ensure_initialized():
            return None

        # Ensure limit doesn't exceed MCP maximum
        limit = min(limit, 200)

        params = {
            "resource_id": resource_id,
            "limit": limit,
            "offset": offset
        }

        if sql_query:
            params["sql"] = sql_query

        response = self.mcp.call_tool("query_resource_data", params)

        if not response.success:
            print(f"Query error: {response.error}")
            return None

        # Parse response into DataFrame
        content = response.data.get("content", [])

        for item in content:
            if item.get("type") == "text":
                import json
                try:
                    data = json.loads(item.get("text", "{}"))
                    if "data" in data:
                        return pd.DataFrame(data["data"])
                    elif isinstance(data, list):
                        return pd.DataFrame(data)
                except:
                    pass

        return None

    def download_resource(
        self,
        resource_id: str,
        format_hint: str = "csv"
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse a resource file.

        For non-Tabular API resources (direct CSV/JSON download).

        Args:
            resource_id: Resource identifier
            format_hint: Expected format (csv, json, xlsx)

        Returns:
            pandas DataFrame with parsed data, or None on error
        """
        if not self._ensure_initialized():
            return None

        response = self.mcp.call_tool("download_and_parse_resource", {
            "resource_id": resource_id,
            "format": format_hint
        })

        if not response.success:
            print(f"Download error: {response.error}")
            return None

        # Parse response into DataFrame
        content = response.data.get("content", [])

        for item in content:
            if item.get("type") == "text":
                import json
                try:
                    data = json.loads(item.get("text", "{}"))
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    elif "data" in data:
                        return pd.DataFrame(data["data"])
                except:
                    # Try parsing as CSV string
                    text = item.get("text", "")
                    if text and ',' in text:
                        from io import StringIO
                        try:
                            return pd.read_csv(StringIO(text))
                        except:
                            pass

        return None

    def close(self):
        """Close the MCP connection."""
        self.mcp.close()
        self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

def search_datagouv(query: str, limit: int = 10) -> List[DataGouvDataset]:
    """
    Quick search for datasets on data.gouv.fr.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of matching datasets
    """
    with DataGouvClient() as client:
        return client.search(query, page_size=limit)


def load_datagouv_resource(resource_id: str) -> Optional[pd.DataFrame]:
    """
    Quick load of a data.gouv resource into DataFrame.

    Args:
        resource_id: Resource identifier

    Returns:
        DataFrame with data, or None on error
    """
    with DataGouvClient() as client:
        # Try query first (faster for tabular data)
        df = client.query_data(resource_id, limit=200)
        if df is not None:
            return df
        # Fall back to download
        return client.download_resource(resource_id)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'DataGouvClient',
    'DataGouvDataset',
    'DataGouvResource',
    'DATAGOUV_MCP_ENDPOINT',
    'search_datagouv',
    'load_datagouv_resource',
]
