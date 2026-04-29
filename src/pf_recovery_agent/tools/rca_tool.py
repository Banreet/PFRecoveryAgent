"""Tool: Search the RCA database for incidents similar to the current outage."""

from __future__ import annotations

from pf_recovery_agent.knowledge_base.loader import load_rcas
from pf_recovery_agent.models import RCA


def search_rcas(
    affected_services: list[str],
    keywords: list[str] | None = None,
    max_results: int = 5,
) -> list[dict]:
    """
    Search historical RCA records for incidents that match the affected services
    and optional keyword tags (e.g. 'tls', 'oom', 'cosmosdb', '429').

    Args:
        affected_services: List of PF service names currently impacted.
        keywords:          Optional list of symptom/tag keywords to narrow search.
        max_results:       Maximum number of RCAs to return (default 5).

    Returns:
        List of matching RCA dicts ordered by relevance score (highest first).
        Each dict includes id, title, root_cause, mitigation_steps,
        time_to_mitigate_minutes, lessons_learned, and score.
    """
    all_rcas: list[RCA] = load_rcas()
    keywords_lower = [k.lower() for k in (keywords or [])]
    service_set = {s.lower() for s in affected_services}

    scored: list[tuple[int, RCA]] = []
    for rca in all_rcas:
        score = 0
        rca_services = {s.lower() for s in rca.affected_services}
        # Score by service overlap
        overlap = len(service_set & rca_services)
        score += overlap * 10

        # Score by keyword/tag match
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
