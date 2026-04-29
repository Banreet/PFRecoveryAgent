"""Tool: Search the RCA database for incidents similar to the current outage.

When Azure AI Search is configured (``AZURE_SEARCH_ENDPOINT`` + ``AZURE_SEARCH_API_KEY``
env vars set), this tool performs a hybrid semantic + optional vector search
against the ``pf-rcas`` index for highly relevant results.

If Azure AI Search is *not* configured, the tool falls back to local keyword
matching over the bundled JSON knowledge base.
"""

from __future__ import annotations

import logging

from pf_recovery_agent.config import USE_AZURE_SEARCH
from pf_recovery_agent.knowledge_base.loader import load_rcas
from pf_recovery_agent.models import RCA

logger = logging.getLogger(__name__)


def search_rcas(
    affected_services: list[str],
    keywords: list[str] | None = None,
    max_results: int = 5,
) -> list[dict]:
    """
    Search historical RCA records for incidents that match the affected services
    and optional keyword tags (e.g. 'tls', 'oom', 'cosmosdb', '429').

    Uses Azure AI Search (semantic + vector) when configured, otherwise falls
    back to local keyword matching.

    Args:
        affected_services: List of PF service names currently impacted.
        keywords:          Optional symptom or tag keywords to narrow the search.
        max_results:       Maximum number of RCAs to return (default 5).

    Returns:
        List of matching RCA dicts ordered by relevance (highest first).
        Each dict includes id, title, root_cause, mitigation_steps,
        time_to_mitigate_minutes, lessons_learned, tags, and relevance_score.
    """
    if USE_AZURE_SEARCH:
        return _search_rcas_azure(affected_services, keywords, max_results)
    return _search_rcas_local(affected_services, keywords, max_results)


# ---------------------------------------------------------------------------
# Azure AI Search path
# ---------------------------------------------------------------------------

def _search_rcas_azure(
    affected_services: list[str],
    keywords: list[str] | None,
    max_results: int,
) -> list[dict]:
    from pf_recovery_agent.search.azure_search_client import search_rca_index
    from pf_recovery_agent.search.embeddings import embed_query

    query = " ".join(affected_services)
    if keywords:
        query += " " + " ".join(keywords)

    query_vector = embed_query(query)

    try:
        return search_rca_index(
            query=query,
            affected_services=affected_services,
            keywords=keywords,
            max_results=max_results,
            query_vector=query_vector,
        )
    except Exception as exc:
        logger.warning(
            "Azure AI Search RCA query failed (%s) – falling back to local search.", exc
        )
        return _search_rcas_local(affected_services, keywords, max_results)


# ---------------------------------------------------------------------------
# Local keyword-matching fallback
# ---------------------------------------------------------------------------

def _search_rcas_local(
    affected_services: list[str],
    keywords: list[str] | None,
    max_results: int,
) -> list[dict]:
    all_rcas: list[RCA] = load_rcas()
    keywords_lower = [k.lower() for k in (keywords or [])]
    service_set = {s.lower() for s in affected_services}

    scored: list[tuple[int, RCA]] = []
    for rca in all_rcas:
        score = 0
        rca_services = {s.lower() for s in rca.affected_services}
        score += len(service_set & rca_services) * 10

        rca_tags = {t.lower() for t in rca.tags}
        rca_text = (rca.title + " " + rca.root_cause + " " + rca.lessons_learned).lower()
        for kw in keywords_lower:
            if kw in rca_tags:
                score += 5
            if kw in rca_text:
                score += 2

        if score > 0:
            scored.append((score, rca))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, rca in scored[:max_results]:
        results.append(
            {
                "id": rca.id,
                "title": rca.title,
                "date": rca.date,
                "affected_services": rca.affected_services,
                "root_cause": rca.root_cause,
                "mitigation_steps": rca.mitigation_steps,
                "time_to_mitigate_minutes": rca.time_to_mitigate_minutes,
                "lessons_learned": rca.lessons_learned,
                "tags": rca.tags,
                "relevance_score": score,
            }
        )
    return results
