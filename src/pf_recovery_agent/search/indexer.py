"""
Azure AI Search indexer for PF Recovery Agent.

Reads the JSON knowledge-base files, optionally generates Azure OpenAI vector
embeddings for each document, then creates/updates the three search indexes
(pf-rcas, pf-tsgs, pf-dependencies) and uploads all documents.

Run directly:
    python -m pf_recovery_agent.search.indexer

Or via the CLI:
    python main.py index-knowledge-base
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pf_recovery_agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_SEARCH_DEPENDENCY_INDEX,
    AZURE_SEARCH_RCA_INDEX,
    AZURE_SEARCH_TSG_INDEX,
    DATA_DIR,
    USE_AZURE,
)
from pf_recovery_agent.search.azure_search_client import (
    create_or_update_indexes,
    upload_documents,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _get_embedding_client():
    """Return an Azure OpenAI client configured for embeddings, or None."""
    if not (USE_AZURE and AZURE_OPENAI_EMBEDDING_DEPLOYMENT):
        return None, None
    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        return client, AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    except ImportError:
        logger.warning("openai package not available – skipping vector embeddings.")
        return None, None


def generate_embedding(text: str, client, deployment: str) -> list[float] | None:
    """
    Generate a vector embedding for *text* using Azure OpenAI.

    Returns the embedding as a list of floats, or None on failure.
    """
    try:
        response = client.embeddings.create(input=[text], model=deployment)
        return response.data[0].embedding
    except Exception as exc:
        logger.warning("Embedding generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Document preparation
# ---------------------------------------------------------------------------

def _rca_to_doc(rca: dict[str, Any], embed_client, embed_deployment: str | None) -> dict[str, Any]:
    """Convert a raw RCA dict into an Azure AI Search document."""
    content_text = (
        f"{rca['title']} {rca['root_cause']} "
        + " ".join(rca.get("mitigation_steps", []))
        + f" {rca.get('lessons_learned', '')} "
        + " ".join(rca.get("tags", []))
    )
    doc: dict[str, Any] = {
        "id": rca["id"],
        "title": rca["title"],
        "affected_services_str": ", ".join(rca.get("affected_services", [])),
        "date": rca.get("date", ""),
        "root_cause": rca["root_cause"],
        "mitigation_steps_str": " | ".join(rca.get("mitigation_steps", [])),
        "time_to_mitigate_minutes": rca.get("time_to_mitigate_minutes", 0),
        "tags_str": ", ".join(rca.get("tags", [])),
        "lessons_learned": rca.get("lessons_learned", ""),
    }
    if embed_client and embed_deployment:
        vector = generate_embedding(content_text, embed_client, embed_deployment)
        if vector:
            doc["content_vector"] = vector
    return doc


def _tsg_to_doc(tsg: dict[str, Any], embed_client, embed_deployment: str | None) -> dict[str, Any]:
    """Convert a raw TSG dict into an Azure AI Search document."""
    content_text = (
        f"{tsg['title']} "
        + " ".join(tsg.get("symptoms", []))
        + " ".join(tsg.get("diagnostic_steps", []))
        + " ".join(tsg.get("mitigation_steps", []))
        + " ".join(tsg.get("tags", []))
    )
    doc: dict[str, Any] = {
        "id": tsg["id"],
        "title": tsg["title"],
        "applicable_services_str": ", ".join(tsg.get("applicable_services", [])),
        "symptoms_str": " | ".join(tsg.get("symptoms", [])),
        "diagnostic_steps_str": " | ".join(tsg.get("diagnostic_steps", [])),
        "mitigation_steps_str": " | ".join(tsg.get("mitigation_steps", [])),
        "escalation_path": tsg.get("escalation_path", ""),
        "tags_str": ", ".join(tsg.get("tags", [])),
    }
    if embed_client and embed_deployment:
        vector = generate_embedding(content_text, embed_client, embed_deployment)
        if vector:
            doc["content_vector"] = vector
    return doc


def _dep_to_doc(dep: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw dependency dict into an Azure AI Search document."""
    doc_id = f"{dep['service_name']}_{dep.get('cluster', 'default')}".replace(" ", "_")
    return {
        "id": doc_id,
        "service_name": dep["service_name"],
        "cluster": dep.get("cluster", ""),
        "tier": dep.get("tier", 1),
        "description": dep.get("description", ""),
        "upstream_dependencies_str": ", ".join(dep.get("upstream_dependencies", [])),
        "downstream_dependents_str": ", ".join(dep.get("downstream_dependents", [])),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_indexer(data_dir: Path | None = None, skip_embeddings: bool = False) -> dict[str, int]:
    """
    Create/update all Azure AI Search indexes and populate them from JSON files.

    Args:
        data_dir:         Override the default knowledge-base data directory.
        skip_embeddings:  If True, skip vector embedding generation (faster but
                          disables vector search).

    Returns:
        Dict with counts: {"rcas": N, "tsgs": N, "dependencies": N}
    """
    src = data_dir or DATA_DIR

    logger.info("Loading knowledge-base data from %s", src)
    with open(src / "rca_database.json", encoding="utf-8") as fh:
        rcas: list[dict] = json.load(fh)
    with open(src / "tsg_database.json", encoding="utf-8") as fh:
        tsgs: list[dict] = json.load(fh)
    with open(src / "service_dependencies.json", encoding="utf-8") as fh:
        deps: list[dict] = json.load(fh)

    embed_client, embed_deployment = (None, None) if skip_embeddings else _get_embedding_client()
    if embed_client:
        logger.info("Vector embeddings enabled (deployment: %s).", embed_deployment)
    else:
        logger.info("Vector embeddings disabled – using text-only search.")

    logger.info("Creating / updating Azure AI Search indexes…")
    create_or_update_indexes()

    rca_docs = [_rca_to_doc(r, embed_client, embed_deployment) for r in rcas]
    tsg_docs = [_tsg_to_doc(t, embed_client, embed_deployment) for t in tsgs]
    dep_docs = [_dep_to_doc(d) for d in deps]

    # Deduplicate dependency docs by id (same service may appear in multiple clusters)
    seen_ids: set[str] = set()
    dep_docs_unique = []
    for d in dep_docs:
        if d["id"] not in seen_ids:
            seen_ids.add(d["id"])
            dep_docs_unique.append(d)

    rca_count = upload_documents(AZURE_SEARCH_RCA_INDEX, rca_docs)
    tsg_count = upload_documents(AZURE_SEARCH_TSG_INDEX, tsg_docs)
    dep_count = upload_documents(AZURE_SEARCH_DEPENDENCY_INDEX, dep_docs_unique)

    logger.info(
        "Indexing complete – RCAs: %d, TSGs: %d, Dependencies: %d",
        rca_count, tsg_count, dep_count,
    )
    return {"rcas": rca_count, "tsgs": tsg_count, "dependencies": dep_count}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_indexer()
