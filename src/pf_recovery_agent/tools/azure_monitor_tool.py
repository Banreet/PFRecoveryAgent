"""
Tool: Live telemetry from Azure Monitor Log Analytics.

Runs KQL queries against the Log Analytics workspace linked to the PF
environment to surface real-time errors, exceptions, and failed dependency
calls that occurred during the outage window.

Requires:
    AZURE_LOG_ANALYTICS_WORKSPACE_ID  – Log Analytics workspace ID
    AZURE_SUBSCRIPTION_ID             – Azure subscription (for auth)

Authentication is handled by ``DefaultAzureCredential`` (Managed Identity,
Service Principal env vars, or ``az login`` for local dev).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from pf_recovery_agent.config import (
    AZURE_LOG_ANALYTICS_WORKSPACE_ID,
    AZURE_MONITOR_DEFAULT_WINDOW_HOURS,
    USE_AZURE_MONITOR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KQL query templates
# ---------------------------------------------------------------------------

_KQL_FAILED_REQUESTS = """
requests
| where timestamp > ago({window}h)
| where success == false
{service_filter}
| summarize
    total_requests = count(),
    failed_requests = countif(success == false),
    error_rate_pct = round(100.0 * countif(success == false) / count(), 2),
    p99_duration_ms = percentile(duration, 99)
  by cloud_RoleName, resultCode
| where failed_requests > 0
| order by failed_requests desc
| take 20
"""

_KQL_TOP_EXCEPTIONS = """
exceptions
| where timestamp > ago({window}h)
{service_filter}
| summarize
    exception_count = count(),
    sample_message = any(outerMessage)
  by cloud_RoleName, type
| order by exception_count desc
| take 15
"""

_KQL_FAILED_DEPENDENCIES = """
dependencies
| where timestamp > ago({window}h)
| where success == false
{service_filter}
| summarize
    failure_count = count(),
    avg_duration_ms = round(avg(duration), 2)
  by cloud_RoleName, name, target, type
| order by failure_count desc
| take 15
"""

_KQL_AVAILABILITY = """
availabilityResults
| where timestamp > ago({window}h)
{service_filter}
| summarize
    total = count(),
    passed = countif(success == 1),
    failed = countif(success == 0),
    availability_pct = round(100.0 * countif(success == 1) / count(), 2)
  by name, location
| where failed > 0
| order by failed desc
| take 10
"""


def _service_filter_clause(service_names: list[str], field: str = "cloud_RoleName") -> str:
    """Build a KQL ``| where`` clause that filters by service name."""
    if not service_names:
        return ""
    names = ", ".join(f'"{s}"' for s in service_names)
    return f"| where {field} in ({names})"


def _rows_to_dicts(table) -> list[dict[str, Any]]:
    """Convert a LogsQueryResult table into a list of dicts."""
    if table is None:
        return []
    return [dict(zip(table.columns, row)) for row in table.rows]


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def get_live_telemetry(
    service_names: list[str],
    time_window_hours: int | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    Query Azure Monitor Log Analytics for live error telemetry during the
    outage window.

    Args:
        service_names:     PF service names to filter telemetry by
                           (matches ``cloud_RoleName`` in Application Insights).
        time_window_hours: How many hours back to query. Defaults to
                           ``AZURE_MONITOR_DEFAULT_WINDOW_HOURS`` (2h).
        workspace_id:      Log Analytics workspace ID. Defaults to
                           ``AZURE_LOG_ANALYTICS_WORKSPACE_ID`` env var.

    Returns:
        Dict with keys: ``time_window_hours``, ``services_analysed``,
        ``failed_requests``, ``top_exceptions``, ``failed_dependencies``,
        ``availability_issues``, ``error`` (if query failed).
    """
    ws_id = workspace_id or AZURE_LOG_ANALYTICS_WORKSPACE_ID
    window = time_window_hours or AZURE_MONITOR_DEFAULT_WINDOW_HOURS

    if not USE_AZURE_MONITOR and not workspace_id:
        return {
            "error": (
                "Azure Monitor is not configured. "
                "Set AZURE_LOG_ANALYTICS_WORKSPACE_ID and AZURE_SUBSCRIPTION_ID."
            ),
            "time_window_hours": window,
            "services_analysed": service_names,
        }

    try:
        from azure.monitor.query import LogsQueryClient, LogsQueryStatus

        from pf_recovery_agent.credential import get_credential

        client = LogsQueryClient(get_credential())
    except ImportError as exc:
        return {"error": f"azure-monitor-query not installed: {exc}"}

    svc_filter = _service_filter_clause(service_names)
    timespan = timedelta(hours=window)

    result: dict[str, Any] = {
        "time_window_hours": window,
        "services_analysed": service_names,
        "workspace_id": ws_id,
        "failed_requests": [],
        "top_exceptions": [],
        "failed_dependencies": [],
        "availability_issues": [],
    }

    queries = {
        "failed_requests": _KQL_FAILED_REQUESTS.format(window=window, service_filter=svc_filter),
        "top_exceptions": _KQL_TOP_EXCEPTIONS.format(window=window, service_filter=svc_filter),
        "failed_dependencies": _KQL_FAILED_DEPENDENCIES.format(window=window, service_filter=svc_filter),
        "availability_issues": _KQL_AVAILABILITY.format(window=window, service_filter=svc_filter),
    }

    errors: list[str] = []
    for key, kql in queries.items():
        try:
            response = client.query_workspace(
                workspace_id=ws_id,
                query=kql,
                timespan=timespan,
            )
            if response.status == LogsQueryStatus.SUCCESS and response.tables:
                result[key] = _rows_to_dicts(response.tables[0])
            elif response.status == LogsQueryStatus.PARTIAL:
                result[key] = _rows_to_dicts(response.partial_data[0]) if response.partial_data else []
                errors.append(f"{key}: partial result – {response.partial_error}")
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            logger.warning("Log Analytics query '%s' failed: %s", key, exc)

    if errors:
        result["warnings"] = errors

    # Summarise the severity of what was found
    result["summary"] = _summarise_telemetry(result)
    return result


def _summarise_telemetry(data: dict[str, Any]) -> str:
    """Return a short human-readable summary of the telemetry findings."""
    parts = []
    fr = data.get("failed_requests", [])
    if fr:
        top = fr[0]
        parts.append(
            f"Highest failure rate: {top.get('cloud_RoleName')} "
            f"(HTTP {top.get('resultCode')}, {top.get('error_rate_pct')}% errors)"
        )
    ex = data.get("top_exceptions", [])
    if ex:
        top = ex[0]
        parts.append(
            f"Most frequent exception: [{top.get('cloud_RoleName')}] "
            f"{top.get('type')} × {top.get('exception_count')}"
        )
    fd = data.get("failed_dependencies", [])
    if fd:
        top = fd[0]
        parts.append(
            f"Top failing dependency: {top.get('target')} "
            f"called by {top.get('cloud_RoleName')} ({top.get('failure_count')} failures)"
        )
    av = data.get("availability_issues", [])
    if av:
        parts.append(
            f"Availability degradation detected for: "
            + ", ".join(r.get("name", "") for r in av[:3])
        )
    if not parts:
        return "No anomalies detected in the specified time window."
    return " | ".join(parts)
