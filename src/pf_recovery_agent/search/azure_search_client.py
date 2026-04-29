"""
Azure AI Search client for PF Recovery Agent.

Provides two main capabilities:
1. Index management  – create / update the RCA, TSG and dependency indexes.
2. Semantic + vector search – hybrid queries that combine BM25 full-text search
   with Azure AI Search's built-in semantic ranker and optional vector search
   (requires an Azure OpenAI embeddings deployment).

Usage
-----
The module falls back gracefully: if Azure AI Search is not configured, callers
should use the local knowledge-base tools instead (see tools/rca_tool.py and
tools/tsg_tool.py).
"""

from __future__ import annotations

import logging
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from pf_recovery_agent.config import (
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_DEPENDENCY_INDEX,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_RCA_INDEX,
    AZURE_SEARCH_TSG_INDEX,
    AZURE_OPENAI_EMBEDDING_DIMENSIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VECTOR_FIELD = "content_vector"
_VECTOR_PROFILE = "pf-hnsw-profile"
_ALGO_CONFIG = "pf-hnsw"
_SEMANTIC_CONFIG = "pf-semantic"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _credential() -> AzureKeyCredential:
    return AzureKeyCredential(AZURE_SEARCH_API_KEY)


def _index_client() -> SearchIndexClient:
    return SearchIndexClient(AZURE_SEARCH_ENDPOINT, _credential())


def _search_client(index_name: str) -> SearchClient:
    return SearchClient(AZURE_SEARCH_ENDPOINT, index_name, _credential())


def _vector_search_config() -> VectorSearch:
    """HNSW vector search configuration for approximate nearest-neighbour lookup."""
    return VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=_ALGO_CONFIG)],
        profiles=[VectorSearchProfile(name=_VECTOR_PROFILE, algorithm_configuration_name=_ALGO_CONFIG)],
    )


# ---------------------------------------------------------------------------
# Index definitions
# ---------------------------------------------------------------------------

def _rca_index() -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="affected_services_str", type=SearchFieldDataType.String),
        SimpleField(name="date", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchableField(name="root_cause", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="mitigation_steps_str", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="time_to_mitigate_minutes", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchableField(name="tags_str", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="lessons_learned", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchField(
            name=_VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=AZURE_OPENAI_EMBEDDING_DIMENSIONS,
            vector_search_profile_name=_VECTOR_PROFILE,
        ),
    ]
    semantic = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=_SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[
                        SemanticField(field_name="root_cause"),
                        SemanticField(field_name="mitigation_steps_str"),
                        SemanticField(field_name="lessons_learned"),
                    ],
                    keywords_fields=[SemanticField(field_name="tags_str")],
                ),
            )
        ]
    )
    return SearchIndex(
        name=AZURE_SEARCH_RCA_INDEX,
        fields=fields,
        vector_search=_vector_search_config(),
        semantic_search=semantic,
    )


def _tsg_index() -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="applicable_services_str", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="symptoms_str", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="diagnostic_steps_str", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="mitigation_steps_str", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="escalation_path", type=SearchFieldDataType.String),
        SearchableField(name="tags_str", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name=_VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=AZURE_OPENAI_EMBEDDING_DIMENSIONS,
            vector_search_profile_name=_VECTOR_PROFILE,
        ),
    ]
    semantic = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=_SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[
                        SemanticField(field_name="symptoms_str"),
                        SemanticField(field_name="diagnostic_steps_str"),
                        SemanticField(field_name="mitigation_steps_str"),
                    ],
                    keywords_fields=[SemanticField(field_name="tags_str")],
                ),
            )
        ]
    )
    return SearchIndex(
        name=AZURE_SEARCH_TSG_INDEX,
        fields=fields,
        vector_search=_vector_search_config(),
        semantic_search=semantic,
    )


def _dependency_index() -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="service_name", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="cluster", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="tier", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchableField(name="description", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="upstream_dependencies_str", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="downstream_dependents_str", type=SearchFieldDataType.String, filterable=True),
    ]
    return SearchIndex(name=AZURE_SEARCH_DEPENDENCY_INDEX, fields=fields)


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def create_or_update_indexes() -> None:
    """Create (or update) all three PF knowledge-base indexes in Azure AI Search."""
    client = _index_client()
    for index_def in [_rca_index(), _tsg_index(), _dependency_index()]:
        client.create_or_update_index(index_def)
        logger.info("Index '%s' created/updated.", index_def.name)


def upload_documents(index_name: str, documents: list[dict[str, Any]]) -> int:
    """
    Upload (upsert) documents into the specified Azure AI Search index.

    Returns the number of documents successfully uploaded.
    """
    client = _search_client(index_name)
    result = client.upload_documents(documents=documents)
    succeeded = sum(1 for r in result if r.succeeded)
    failed = len(result) - succeeded
    if failed:
        logger.warning("%d documents failed to upload to index '%s'.", failed, index_name)
    logger.info("Uploaded %d documents to index '%s'.", succeeded, index_name)
    return succeeded


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _build_filter(field: str, values: list[str]) -> str | None:
    """Build an OData filter expression matching any of the given values."""
    if not values:
        return None
    clauses = [f"search.ismatch('{v}', '{field}')" for v in values]
    return " or ".join(clauses)


def search_rca_index(
    query: str,
    affected_services: list[str] | None = None,
    keywords: list[str] | None = None,
    max_results: int = 5,
    query_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Query the RCA index using hybrid search (BM25 + semantic reranking + optional vector).

    Args:
        query:             Natural language query string.
        affected_services: Optional list of service names to weight the search.
        keywords:          Optional keyword hints appended to the query.
        max_results:       Maximum number of results.
        query_vector:      Pre-computed embedding vector for vector search (optional).

    Returns:
        List of RCA document dicts, ordered by relevance.
    """
    client = _search_client(AZURE_SEARCH_RCA_INDEX)

    # Augment query with service names and keywords
    full_query = query
    if affected_services:
        full_query += " " + " ".join(affected_services)
    if keywords:
        full_query += " " + " ".join(keywords)

    vector_queries = None
    if query_vector:
        vector_queries = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=max_results,
                fields=_VECTOR_FIELD,
            )
        ]

    results = client.search(
        search_text=full_query,
        vector_queries=vector_queries,
        query_type="semantic",
        semantic_configuration_name=_SEMANTIC_CONFIG,
        query_caption="extractive",
        query_answer="extractive",
        top=max_results,
        select=[
            "id", "title", "affected_services_str", "date",
            "root_cause", "mitigation_steps_str",
            "time_to_mitigate_minutes", "tags_str", "lessons_learned",
        ],
    )

    docs = []
    for r in results:
        doc = dict(r)
        # Re-expand stringified list fields
        doc["affected_services"] = _split_field(doc.pop("affected_services_str", ""))
        doc["mitigation_steps"] = _split_field(doc.pop("mitigation_steps_str", ""), delimiter="|")
        doc["tags"] = _split_field(doc.pop("tags_str", ""))
        doc["relevance_score"] = r.get("@search.score", 0)
        doc["reranker_score"] = r.get("@search.reranker_score")
        docs.append(doc)
    return docs


def search_tsg_index(
    query: str,
    affected_services: list[str] | None = None,
    symptoms: list[str] | None = None,
    max_results: int = 5,
    query_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Query the TSG index using hybrid search.

    Args:
        query:             Natural language query string.
        affected_services: Optional list of service names.
        symptoms:          Optional symptom strings.
        max_results:       Maximum number of results.
        query_vector:      Pre-computed embedding vector (optional).

    Returns:
        List of TSG document dicts, ordered by relevance.
    """
    client = _search_client(AZURE_SEARCH_TSG_INDEX)

    full_query = query
    if affected_services:
        full_query += " " + " ".join(affected_services)
    if symptoms:
        full_query += " " + " ".join(symptoms)

    vector_queries = None
    if query_vector:
        vector_queries = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=max_results,
                fields=_VECTOR_FIELD,
            )
        ]

    results = client.search(
        search_text=full_query,
        vector_queries=vector_queries,
        query_type="semantic",
        semantic_configuration_name=_SEMANTIC_CONFIG,
        query_caption="extractive",
        query_answer="extractive",
        top=max_results,
        select=[
            "id", "title", "applicable_services_str", "symptoms_str",
            "diagnostic_steps_str", "mitigation_steps_str",
            "escalation_path", "tags_str",
        ],
    )

    docs = []
    for r in results:
        doc = dict(r)
        doc["applicable_services"] = _split_field(doc.pop("applicable_services_str", ""))
        doc["symptoms"] = _split_field(doc.pop("symptoms_str", ""), delimiter="|")
        doc["diagnostic_steps"] = _split_field(doc.pop("diagnostic_steps_str", ""), delimiter="|")
        doc["mitigation_steps"] = _split_field(doc.pop("mitigation_steps_str", ""), delimiter="|")
        doc["tags"] = _split_field(doc.pop("tags_str", ""))
        doc["relevance_score"] = r.get("@search.score", 0)
        doc["reranker_score"] = r.get("@search.reranker_score")
        docs.append(doc)
    return docs


def search_dependency_index(
    service_names: list[str],
) -> list[dict[str, Any]]:
    """
    Retrieve service dependency documents from the dependency index.

    Args:
        service_names: List of PF service names to look up.

    Returns:
        List of service dependency dicts.
    """
    client = _search_client(AZURE_SEARCH_DEPENDENCY_INDEX)
    filter_expr = " or ".join(f"service_name eq '{s}'" for s in service_names)
    results = client.search(
        search_text="*",
        filter=filter_expr,
        top=len(service_names) * 2,
    )
    docs = []
    for r in results:
        doc = dict(r)
        doc["upstream_dependencies"] = _split_field(doc.pop("upstream_dependencies_str", ""))
        doc["downstream_dependents"] = _split_field(doc.pop("downstream_dependents_str", ""))
        docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _split_field(value: str, delimiter: str = ",") -> list[str]:
    """Split a stored string field back into a list, filtering empty items."""
    if not value:
        return []
    return [item.strip() for item in value.split(delimiter) if item.strip()]
