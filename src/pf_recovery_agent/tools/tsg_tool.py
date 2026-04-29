"""Tool: Retrieve TSG (Troubleshooting Guide) entries relevant to the outage."""

from __future__ import annotations

from pf_recovery_agent.knowledge_base.loader import load_tsgs
from pf_recovery_agent.models import TSG


def search_tsgs(
    affected_services: list[str],
    symptoms: list[str] | None = None,
    max_results: int = 5,
) -> list[dict]:
    """
    Find the most relevant TSG entries for the affected services and observed
    symptoms.

    Args:
        affected_services: List of PF service names currently impacted.
        symptoms:          Optional list of symptom strings reported by engineers.
        max_results:       Maximum number of TSGs to return (default 5).

    Returns:
        List of matching TSG dicts ordered by relevance score (highest first).
        Each dict includes id, title, diagnostic_steps, mitigation_steps,
        escalation_path, and score.
    """
    all_tsgs: list[TSG] = load_tsgs()
    service_set = {s.lower() for s in affected_services}
    symptom_text = " ".join(symptoms or []).lower()

    scored: list[tuple[int, TSG]] = []
    for tsg in all_tsgs:
        score = 0
        tsg_services = {s.lower() for s in tsg.applicable_services}
        overlap = len(service_set & tsg_services)
        score += overlap * 10

        # Score by symptom overlap
        for tsg_symptom in tsg.symptoms:
            if any(word in tsg_symptom.lower() for word in symptom_text.split() if len(word) > 3):
                score += 3

        # Score by tag match against symptom text
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
