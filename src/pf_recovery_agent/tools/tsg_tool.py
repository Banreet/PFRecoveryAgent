"""Tool: Retrieve TSG (Troubleshooting Guide) entries relevant to the outage.

When Azure AI Search is configured (``AZURE_SEARCH_ENDPOINT`` + ``AZURE_SEARCH_API_KEY``
env vars set), this tool performs a hybrid semantic + optional vector search
against the ``pf-tsgs`` index for highly relevant results.

If Azure AI Search is *not* configured, the tool falls back to local keyword
matching over the bundled JSON knowledge base.
"""

from __future__ import annotations

import logging

from pf_recovery_agent.config import USE_AZURE_SEARCH
from pf_recovery_agent.knowledge_base.loader import load_tsgs
from pf_recovery_agent.models import TSG

logger = logging.getLogger(__name__)


def search_tsgs(
    affected_services: list[str],
    symptoms: list[str] | None = None,
    max_results: int = 5,
) -> list[dict]:
    """
    Find the most relevant TSG entries for the affected services and observed
    symptoms.

    Uses Azure AI Search (semantic + vector) when configured, otherwise falls
    back to local keyword matching.

    Args:
        affected_services: List of PF service names currently impacted.
        symptoms:          Optional list of symptom strings reported by engineers.
        max_results:       Maximum number of TSGs to return (default 5).

    Returns:
        List of matching TSG dicts ordered by relevance (highest first).
        Each dict includes id, title, diagnostic_steps, mitigation_steps,
        escalation_path, tags, and relevance_score.
    """
    if USE_AZURE_SEARCH:
        return _search_tsgs_azure(affected_services, symptoms, max_results)
    return _search_tsgs_local(affected_services, symptoms, max_results)


# ---------------------------------------------------------------------------
# Azure AI Search path
# ---------------------------------------------------------------------------

def _search_tsgs_azure(
    affected_services: list[str],
    symptoms: list[str] | None,
    max_results: int,
) -> list[dict]:
    from pf_recovery_agent.search.azure_search_client import search_tsg_index
    from pf_recovery_agent.search.embeddings import embed_query

    query = " ".join(affected_services)
    if symptoms:
        query += " " + " ".join(symptoms)

    query_vector = embed_query(query)

    try:
        return search_tsg_index(
            query=query,
            affected_services=affected_services,
            symptoms=symptoms,
            max_results=max_results,
            query_vector=query_vector,
        )
    except Exception as exc:
        logger.warning(
            "Azure AI Search TSG query failed (%s) – falling back to local search.", exc
        )
        return _search_tsgs_local(affected_services, symptoms, max_results)


# ---------------------------------------------------------------------------
# Local keyword-matching fallback
# ---------------------------------------------------------------------------

def _search_tsgs_local(
    affected_services: list[str],
    symptoms: list[str] | None,
    max_results: int,
) -> list[dict]:
    all_tsgs: list[TSG] = load_tsgs()
    service_set = {s.lower() for s in affected_services}
    symptom_text = " ".join(symptoms or []).lower()

    scored: list[tuple[int, TSG]] = []
    for tsg in all_tsgs:
        score = 0
        tsg_services = {s.lower() for s in tsg.applicable_services}
        score += len(service_set & tsg_services) * 10

        for tsg_symptom in tsg.symptoms:
            if any(word in tsg_symptom.lower() for word in symptom_text.split() if len(word) > 3):
                score += 3

        for tag in tsg.tags:
            if tag in symptom_text:
                score += 4

        if score > 0:
            scored.append((score, tsg))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, tsg in scored[:max_results]:
        results.append(
            {
                "id": tsg.id,
                "title": tsg.title,
                "applicable_services": tsg.applicable_services,
                "symptoms": tsg.symptoms,
                "diagnostic_steps": tsg.diagnostic_steps,
                "mitigation_steps": tsg.mitigation_steps,
                "escalation_path": tsg.escalation_path,
                "tags": tsg.tags,
                "relevance_score": score,
            }
        )
    return results
