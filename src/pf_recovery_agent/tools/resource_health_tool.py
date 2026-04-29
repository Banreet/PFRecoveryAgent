"""
Tool: Azure Resource Health status for PF infrastructure resources.

Queries the Azure Resource Health API to determine whether the Azure
platform resources that PF services depend on (AKS clusters, CosmosDB
accounts, Service Bus namespaces, Key Vaults, Storage accounts, etc.) are
currently degraded or unavailable.

This surfaces Azure platform-level failures that would not appear in
application logs, such as:
  - AKS node-pool degradation
  - CosmosDB regional availability issues
  - Service Bus namespace throttling or platform failures
  - Key Vault throttling

Requires:
    AZURE_SUBSCRIPTION_ID  – Azure subscription
    AZURE_RESOURCE_GROUP   – Resource group containing PF resources (optional;
                              if omitted, all resources in the subscription
                              are checked)

Authentication uses ``DefaultAzureCredential``.
"""

from __future__ import annotations

import logging
from typing import Any

from pf_recovery_agent.config import (
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    USE_AZURE_RESOURCE_HEALTH,
)

logger = logging.getLogger(__name__)

# Resource types most relevant to PF outages
_PF_RESOURCE_TYPES = {
    "Microsoft.ContainerService/managedClusters",
    "Microsoft.DocumentDB/databaseAccounts",
    "Microsoft.ServiceBus/namespaces",
    "Microsoft.KeyVault/vaults",
    "Microsoft.Storage/storageAccounts",
    "Microsoft.AppConfiguration/configurationStores",
    "Microsoft.EventHub/namespaces",
    "Microsoft.Network/loadBalancers",
    "Microsoft.Network/trafficManagerProfiles",
}

# Health states that indicate a problem
_DEGRADED_STATES = {"Degraded", "Unavailable", "Unknown"}


def get_resource_health(
    subscription_id: str | None = None,
    resource_group: str | None = None,
    resource_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Retrieve health status for Azure resources in the PF environment.

    When ``resource_ids`` is provided, checks those specific resources.
    Otherwise, lists all resources in ``resource_group`` (or the full
    subscription) and filters to PF-relevant resource types.

    Args:
        subscription_id: Azure subscription ID. Defaults to config.
        resource_group:  Resource group name. Defaults to config.
        resource_ids:    Explicit list of Azure resource IDs to check.

    Returns:
        Dict with keys:
          ``healthy``   – list of healthy resources
          ``degraded``  – list of degraded/unavailable resources (highest priority)
          ``unknown``   – list of resources with unknown health
          ``summary``   – human-readable summary
          ``error``     – error message if the API call failed
    """
    sub_id = subscription_id or AZURE_SUBSCRIPTION_ID
    rg = resource_group or AZURE_RESOURCE_GROUP

    if not USE_AZURE_RESOURCE_HEALTH and not subscription_id:
        return {
            "error": (
                "Azure Resource Health is not configured. "
                "Set AZURE_SUBSCRIPTION_ID in your environment."
            )
        }

    try:
        from azure.mgmt.resourcehealth import ResourceHealthMgmtClient

        from pf_recovery_agent.credential import get_credential

        client = ResourceHealthMgmtClient(get_credential(), sub_id)
    except ImportError as exc:
        return {"error": f"azure-mgmt-resourcehealth not installed: {exc}"}

    healthy: list[dict] = []
    degraded: list[dict] = []
    unknown: list[dict] = []

    try:
        if resource_ids:
            statuses = [
                client.availability_statuses.get_by_resource(resource_uri=rid, expand="recommendedactions")
                for rid in resource_ids
            ]
        elif rg:
            statuses = list(
                client.availability_statuses.list_by_resource_group(
                    resource_group_name=rg, expand="recommendedactions"
                )
            )
        else:
            statuses = list(
                client.availability_statuses.list_by_subscription_id(
                    expand="recommendedactions"
                )
            )
    except Exception as exc:
        logger.warning("Resource Health API call failed: %s", exc)
        return {"error": str(exc)}

    for status in statuses:
        resource_id: str = getattr(status, "id", "") or ""
        resource_type = _extract_resource_type(resource_id)

        # Filter to PF-relevant resource types only
        if resource_ids is None and resource_type not in _PF_RESOURCE_TYPES:
            continue

        props = status.properties
        availability_state: str = getattr(props, "availability_state", "Unknown") or "Unknown"
        summary: str = getattr(props, "summary", "") or ""
        reason: str = getattr(props, "reason_type", "") or ""
        occurred_time = getattr(props, "occurred_time", None)

        rec_actions: list[str] = []
        if hasattr(props, "recommended_actions") and props.recommended_actions:
            rec_actions = [
                getattr(a, "action", "") for a in props.recommended_actions if getattr(a, "action", "")
            ]

        entry = {
            "resource_id": resource_id,
            "resource_type": resource_type,
            "resource_name": _extract_resource_name(resource_id),
            "availability_state": availability_state,
            "summary": summary,
            "reason": reason,
            "occurred_time": str(occurred_time) if occurred_time else None,
            "recommended_actions": rec_actions,
        }

        if availability_state == "Available":
            healthy.append(entry)
        elif availability_state in _DEGRADED_STATES:
            degraded.append(entry)
        else:
            unknown.append(entry)

    return {
        "subscription_id": sub_id,
        "resource_group": rg or "(all)",
        "healthy_count": len(healthy),
        "degraded": degraded,
        "unknown": unknown,
        "healthy": healthy,
        "summary": _summarise_health(degraded, unknown),
    }


def _extract_resource_type(resource_id: str) -> str:
    """Extract 'Provider/Type' from a full Azure resource ID."""
    parts = resource_id.split("/")
    try:
        provider_idx = next(i for i, p in enumerate(parts) if p.lower() == "providers")
        return f"{parts[provider_idx + 1]}/{parts[provider_idx + 2]}"
    except (StopIteration, IndexError):
        return ""


def _extract_resource_name(resource_id: str) -> str:
    """Extract the leaf resource name from a full Azure resource ID."""
    return resource_id.rstrip("/").split("/")[-1] if resource_id else ""


def _summarise_health(degraded: list[dict], unknown: list[dict]) -> str:
    if not degraded and not unknown:
        return "All monitored Azure resources are healthy."
    parts = []
    if degraded:
        names = ", ".join(r["resource_name"] for r in degraded[:5])
        parts.append(f"DEGRADED/UNAVAILABLE: {names}")
    if unknown:
        names = ", ".join(r["resource_name"] for r in unknown[:3])
        parts.append(f"UNKNOWN health: {names}")
    return " | ".join(parts)
