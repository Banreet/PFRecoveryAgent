"""
Tool: Azure Service Health events for the PF subscription.

Queries the Azure Service Health API to surface any active platform-level
outages, service degradations, or planned maintenance events in Azure that
could be the root cause of a PF outage.

This is the first thing to check during an outage – if Azure itself is
reporting an issue in the affected regions or for the impacted service types
(AKS, CosmosDB, Service Bus, etc.), the root cause lies outside PF and the
mitigation path is to wait for Azure resolution and/or fail over to another
region.

Requires:
    AZURE_SUBSCRIPTION_ID  – Azure subscription

Authentication uses ``DefaultAzureCredential``.
"""

from __future__ import annotations

import logging
from typing import Any

from pf_recovery_agent.config import AZURE_SUBSCRIPTION_ID, USE_AZURE_RESOURCE_HEALTH

logger = logging.getLogger(__name__)

# Service Health event types
_EVENT_TYPES = {
    "ServiceIssue": "Active Outage",
    "PlannedMaintenance": "Planned Maintenance",
    "HealthAdvisory": "Health Advisory",
    "SecurityAdvisory": "Security Advisory",
}

# Azure service names relevant to PF infrastructure
_PF_AZURE_SERVICES = {
    "azure kubernetes service",
    "azure cosmos db",
    "azure service bus",
    "azure key vault",
    "azure storage",
    "azure app configuration",
    "azure event hubs",
    "azure load balancer",
    "azure traffic manager",
    "azure active directory",
    "azure dns",
    "azure monitor",
    "azure application insights",
    "log analytics",
}


def get_azure_service_health(
    regions: list[str] | None = None,
    service_types: list[str] | None = None,
    subscription_id: str | None = None,
    active_only: bool = True,
) -> dict[str, Any]:
    """
    Retrieve Azure Service Health events for the subscription.

    Filters events to the specified regions and service types (if provided),
    and further narrows to PF-relevant Azure services.

    Args:
        regions:         List of Azure region names to filter by
                         (e.g. ['eastus', 'westus2']). If empty, all regions.
        service_types:   Azure service type names to filter by. If empty,
                         uses the built-in PF-relevant list.
        subscription_id: Azure subscription ID. Defaults to config.
        active_only:     If True (default), only return active / ongoing events.

    Returns:
        Dict with keys:
          ``active_outages``      – list of active ServiceIssue events
          ``planned_maintenance`` – list of upcoming maintenance windows
          ``health_advisories``   – list of health advisories
          ``pf_impacted``         – True if any event affects PF-relevant services
          ``summary``             – human-readable summary string
          ``error``               – set if the API call failed
    """
    sub_id = subscription_id or AZURE_SUBSCRIPTION_ID

    if not USE_AZURE_RESOURCE_HEALTH and not subscription_id:
        return {
            "error": (
                "Azure Service Health is not configured. "
                "Set AZURE_SUBSCRIPTION_ID in your environment."
            )
        }

    try:
        from azure.mgmt.resourcehealth import ResourceHealthMgmtClient

        from pf_recovery_agent.credential import get_credential

        client = ResourceHealthMgmtClient(get_credential(), sub_id)
    except ImportError as exc:
        return {"error": f"azure-mgmt-resourcehealth not installed: {exc}"}

    regions_lower = {r.lower().replace(" ", "") for r in (regions or [])}
    services_lower = (
        {s.lower() for s in service_types}
        if service_types
        else _PF_AZURE_SERVICES
    )

    active_outages: list[dict] = []
    planned_maintenance: list[dict] = []
    health_advisories: list[dict] = []

    try:
        events = client.events.list_by_subscription_id()
    except Exception as exc:
        logger.warning("Service Health API call failed: %s", exc)
        return {"error": str(exc)}

    for event in events:
        try:
            event_type: str = getattr(event, "event_type", "") or ""
            event_level: str = getattr(event, "level", "") or ""
            title: str = getattr(event, "title", "") or ""
            description: str = getattr(event, "summary", "") or ""
            status: str = getattr(event, "status", "") or ""
            last_update = getattr(event, "last_update_time", None)
            start_time = getattr(event, "impact_start_time", None)
            tracking_id: str = getattr(event, "tracking_id", "") or ""

            # Filter by active status if requested
            if active_only and status.lower() not in ("active", "resolved"):
                continue

            # Extract impacted services and regions
            impacted_services: list[str] = []
            impacted_regions: list[str] = []

            impacts = getattr(event, "impact", None) or []
            for impact in impacts:
                svc_name: str = getattr(impact, "impacted_service", "") or ""
                if svc_name:
                    impacted_services.append(svc_name)
                regions_in_impact = getattr(impact, "impacted_regions", None) or []
                for region in regions_in_impact:
                    region_name: str = getattr(region, "impacted_region", "") or ""
                    if region_name:
                        impacted_regions.append(region_name)

            # Apply region filter
            if regions_lower:
                region_match = any(
                    r.lower().replace(" ", "") in regions_lower
                    for r in impacted_regions
                )
                if not region_match:
                    continue

            # Apply service type filter
            svc_match = any(
                s.lower() in services_lower
                for s in impacted_services
            )
            if not svc_match and services_lower:
                continue

            entry = {
                "tracking_id": tracking_id,
                "title": title,
                "event_type": _EVENT_TYPES.get(event_type, event_type),
                "level": event_level,
                "status": status,
                "impacted_services": impacted_services,
                "impacted_regions": impacted_regions,
                "summary": description[:500] if description else "",
                "last_update": str(last_update) if last_update else None,
                "start_time": str(start_time) if start_time else None,
            }

            if event_type == "ServiceIssue":
                active_outages.append(entry)
            elif event_type == "PlannedMaintenance":
                planned_maintenance.append(entry)
            else:
                health_advisories.append(entry)

        except Exception as exc:
            logger.warning("Error processing service health event: %s", exc)
            continue

    pf_impacted = bool(active_outages or (planned_maintenance and not active_only))

    return {
        "subscription_id": sub_id,
        "filter_regions": list(regions_lower) if regions_lower else ["all"],
        "active_outages": active_outages,
        "planned_maintenance": planned_maintenance,
        "health_advisories": health_advisories,
        "pf_impacted": pf_impacted,
        "summary": _summarise_service_health(active_outages, planned_maintenance),
    }


def _summarise_service_health(
    active_outages: list[dict], planned_maintenance: list[dict]
) -> str:
    if not active_outages and not planned_maintenance:
        return "No active Azure Service Health issues detected for PF-relevant services."
    parts = []
    if active_outages:
        svcs = set()
        for o in active_outages:
            svcs.update(o.get("impacted_services", []))
        parts.append(
            f"ACTIVE AZURE OUTAGE affecting: {', '.join(list(svcs)[:5])} "
            f"({len(active_outages)} event(s))"
        )
    if planned_maintenance:
        parts.append(f"{len(planned_maintenance)} planned maintenance window(s) upcoming")
    return " | ".join(parts)
