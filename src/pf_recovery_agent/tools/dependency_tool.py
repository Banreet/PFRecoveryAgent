"""Tool: Look up cross-cluster runtime service dependencies for affected PF services."""

from __future__ import annotations

from pf_recovery_agent.knowledge_base.loader import load_service_dependencies
from pf_recovery_agent.models import ServiceDependency


def get_service_dependencies(service_names: list[str]) -> dict:
    """
    Return upstream dependencies and downstream dependents for each requested
    PF service, plus a ranked list of transitive blast-radius services.

    Args:
        service_names: One or more PF service names (e.g. ['PFGateway', 'PFOrchestrator']).

    Returns:
        A dict with keys:
          - ``direct``: mapping of service_name -> {upstream, downstream, cluster, tier}
          - ``blast_radius``: sorted list of transitive services that may be impacted
          - ``tier1_risks``: list of Tier-1 services in the blast radius (highest priority)
    """
    all_deps = load_service_dependencies()
    index: dict[str, ServiceDependency] = {}
    for dep in all_deps:
        key = f"{dep.service_name}:{dep.cluster}"
        index[key] = dep
    # Also index by name alone (first match wins for each name)
    name_index: dict[str, ServiceDependency] = {}
    for dep in all_deps:
        if dep.service_name not in name_index:
            name_index[dep.service_name] = dep

    direct: dict[str, dict] = {}
    blast_radius: set[str] = set()

    for svc in service_names:
        dep = name_index.get(svc)
        if dep is None:
            direct[svc] = {"error": f"Service '{svc}' not found in dependency graph"}
            continue
        direct[svc] = {
            "cluster": dep.cluster,
            "tier": dep.tier,
            "description": dep.description,
            "upstream_dependencies": dep.upstream_dependencies,
            "downstream_dependents": dep.downstream_dependents,
        }
        # Add downstream dependents to blast radius (they will be affected)
        for d in dep.downstream_dependents:
            blast_radius.add(d)
        # Also record upstream as potential root causes
        for u in dep.upstream_dependencies:
            blast_radius.add(u)

    # Determine tier-1 risks
    tier1_risks = [
        s for s in blast_radius if name_index.get(s) and name_index[s].tier == 1
    ]

    return {
        "direct": direct,
        "blast_radius": sorted(blast_radius - set(service_names)),
        "tier1_risks": sorted(tier1_risks),
    }
