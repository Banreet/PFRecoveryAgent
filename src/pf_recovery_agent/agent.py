"""
PF Recovery Agent – main orchestrator.

Uses Azure OpenAI function-calling to iteratively invoke tools that query
both real-time Azure data sources and the knowledge base, then synthesises
a structured recovery plan for the on-call engineer.

Tool execution order (enforced by the system prompt):
  1. get_azure_service_health  – check for Azure platform outages first
  2. get_resource_health       – check health of PF Azure resources
  3. get_live_telemetry        – pull real-time errors / exceptions from Log Analytics
  4. get_service_dependencies  – map blast radius from dependency graph
  5. search_rcas               – find similar past incidents
  6. search_tsgs               – retrieve relevant troubleshooting guides
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pf_recovery_agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    MAX_TOOL_ITERATIONS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    USE_AZURE,
)
from pf_recovery_agent.models import AgentResponse, OutageEvent, RecoveryInsight
from pf_recovery_agent.tools.azure_monitor_tool import get_live_telemetry
from pf_recovery_agent.tools.dependency_tool import get_service_dependencies
from pf_recovery_agent.tools.rca_tool import search_rcas
from pf_recovery_agent.tools.resource_health_tool import get_resource_health
from pf_recovery_agent.tools.service_health_tool import get_azure_service_health
from pf_recovery_agent.tools.tsg_tool import search_tsgs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI tool / function schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    # ── Live data: Azure Service Health ──────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_azure_service_health",
            "description": (
                "Check Azure Service Health for active platform-level outages, "
                "service degradations, or planned maintenance events that could be "
                "the root cause of a PF outage. Always call this FIRST to rule out "
                "an Azure-wide issue before investigating PF-specific causes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "regions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Azure region names to filter by (e.g. ['eastus', 'westus2']).",
                    },
                    "service_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Azure service names to filter by. Leave empty to use the default PF-relevant list.",
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "Return only active/ongoing events (default true).",
                        "default": True,
                    },
                },
                "required": [],
            },
        },
    },
    # ── Live data: Azure Resource Health ─────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_resource_health",
            "description": (
                "Retrieve health status for Azure infrastructure resources in the PF "
                "environment (AKS clusters, CosmosDB, Service Bus, Key Vault, Storage, "
                "App Configuration). Surfaces Azure platform failures that don't appear "
                "in application logs. Call this early in the investigation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_group": {
                        "type": "string",
                        "description": "Azure resource group name. Defaults to AZURE_RESOURCE_GROUP env var.",
                    },
                    "resource_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific Azure resource IDs to check (optional – checks all resources in the RG if omitted).",
                    },
                },
                "required": [],
            },
        },
    },
    # ── Live data: Azure Monitor Log Analytics ────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_live_telemetry",
            "description": (
                "Query Azure Monitor Log Analytics for live error telemetry: "
                "failed requests, top exceptions, failed dependency calls, and "
                "availability test failures for the specified PF services during "
                "the outage window. Use this to identify the error pattern and "
                "narrow down which service or dependency is the root cause."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "PF service names to filter telemetry by (matches cloud_RoleName in App Insights).",
                    },
                    "time_window_hours": {
                        "type": "integer",
                        "description": "Hours back to query (default 2).",
                        "default": 2,
                    },
                },
                "required": ["service_names"],
            },
        },
    },
    # ── Knowledge base: Service dependencies ─────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_service_dependencies",
            "description": (
                "Look up cross-cluster runtime service dependencies for the given PF "
                "services. Returns upstream dependencies, downstream dependents, blast "
                "radius, and Tier-1 risks. Use this to understand the full scope of "
                "impact and identify potential root causes in upstream services."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of PF service names to analyse.",
                    }
                },
                "required": ["service_names"],
            },
        },
    },
    # ── Knowledge base: RCA search ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_rcas",
            "description": (
                "Search historical Root Cause Analysis (RCA) records for past incidents "
                "that match the current outage's affected services and symptoms. Returns "
                "ranked RCAs with root cause, proven mitigation steps, and lessons learned. "
                "Use the telemetry findings as keywords to improve relevance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "affected_services": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "PF service names currently impacted.",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Symptom or tag keywords from live telemetry (e.g. 'tls', 'oom', '429', 'cosmosdb', 'timeout').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of RCAs to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["affected_services"],
            },
        },
    },
    # ── Knowledge base: TSG search ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_tsgs",
            "description": (
                "Retrieve Troubleshooting Guide (TSG) entries for the affected PF services "
                "and observed symptoms. Returns ranked TSGs with step-by-step diagnostic "
                "steps, mitigation commands, and escalation paths. Use observed symptoms "
                "and telemetry findings to improve relevance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "affected_services": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "PF service names currently impacted.",
                    },
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Observed symptoms or error signals (from logs, telemetry, or engineer reports).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of TSGs to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["affected_services"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "get_azure_service_health": lambda args: get_azure_service_health(**args),
    "get_resource_health": lambda args: get_resource_health(**args),
    "get_live_telemetry": lambda args: get_live_telemetry(**args),
    "get_service_dependencies": lambda args: get_service_dependencies(**args),
    "search_rcas": lambda args: search_rcas(**args),
    "search_tsgs": lambda args: search_tsgs(**args),
}


def _dispatch_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return JSON-encoded result."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    args = json.loads(arguments)
    result = handler(args)
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# OpenAI client factory
# ---------------------------------------------------------------------------

def _build_client():
    """Return the appropriate OpenAI client based on configuration."""
    try:
        if USE_AZURE:
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION,
            ), AZURE_OPENAI_DEPLOYMENT
        else:
            from openai import OpenAI

            return OpenAI(api_key=OPENAI_API_KEY), OPENAI_MODEL
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required. Run: pip install openai"
        ) from exc


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are PFRecoveryAgent, an AI expert for Azure PF (Platform Foundation) service outages.

Your mission is to help on-call engineers achieve the fastest possible RTO (Recovery Time Objective) by combining real-time Azure data with historical knowledge.

## Investigation order – always follow this sequence:

1. **get_azure_service_health** – Check for Azure platform outages FIRST. If Azure itself is down in the affected regions, that is the root cause and failover is the mitigation.
2. **get_resource_health** – Check if specific Azure resources (AKS, CosmosDB, ServiceBus, KeyVault) are degraded. Azure platform failures won't appear in app logs.
3. **get_live_telemetry** – Pull real errors and exceptions from Log Analytics for the affected services. Use the error types and exception messages to form a root cause hypothesis.
4. **get_service_dependencies** – Map the blast radius. Cross-reference failing dependencies from telemetry with the dependency graph to identify root cause services.
5. **search_rcas** – Use the telemetry findings as keywords to find matching historical RCAs. Past mitigations are the fastest recovery path.
6. **search_tsgs** – Retrieve TSG steps specific to the symptoms observed. Include concrete CLI commands.

## Response format:

### Summary
One paragraph describing what is happening and the most likely root cause.

### Root Cause Hypothesis
Based on live telemetry and resource health, state the most probable root cause with evidence.

### Dependency Risk Analysis
List Tier-1 services in the blast radius and their risk level.

### Recommended Actions
Numbered, prioritised list of concrete steps. Include actual kubectl/az CLI commands.

### Relevant Historical RCAs
Reference past incidents that match. State how long they took to resolve.

### Relevant TSGs
Link to applicable TSG entries and highlight the most relevant steps.

### Estimated RTO
Based on historical data, estimate time to mitigation.

Always provide concrete, actionable commands. Never give vague advice."""


# ---------------------------------------------------------------------------
# Public agent entry point
# ---------------------------------------------------------------------------

def run_agent(outage: OutageEvent) -> AgentResponse:
    """
    Run the PF Recovery Agent for the given outage event.

    Performs an agentic loop using Azure OpenAI function-calling:
      1. Queries live Azure data (Service Health, Resource Health, Log Analytics)
      2. Maps service dependencies and blast radius
      3. Searches the knowledge base (Azure AI Search) for matching RCAs and TSGs
      4. Synthesises a structured, prioritised recovery plan

    Args:
        outage: Structured description of the active PF outage.

    Returns:
        ``AgentResponse`` with insights, recommended actions, RCAs and TSGs.

    Raises:
        RuntimeError: If the OpenAI API is not configured or the call fails.
    """
    client, model = _build_client()

    user_message = _build_user_message(outage)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info("Starting PF Recovery Agent for outage: %s", outage.id)

    # Agentic loop
    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.debug("Agent iteration %d", iteration + 1)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        message = choice.message

        messages.append(message.model_dump(exclude_unset=True))

        if choice.finish_reason == "tool_calls" and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                logger.info("Tool call: %s(%s)", tool_name, tool_args[:120])

                tool_result = _dispatch_tool(tool_name, tool_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
        else:
            final_text = message.content or ""
            logger.info("Agent produced final response after %d iterations", iteration + 1)
            return _parse_response(outage.id, final_text, messages)

    last_content = messages[-1].get("content", "Agent reached maximum iterations without a final answer.")
    return _parse_response(outage.id, str(last_content), messages)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_user_message(outage: OutageEvent) -> str:
    lines = [
        f"## Outage Report: {outage.id}",
        f"**Title:** {outage.title}",
        f"**Severity:** {outage.severity.value}",
        f"**Status:** {outage.status.value}",
        f"**Start Time:** {outage.start_time.isoformat()}",
        "",
        f"**Affected Services:** {', '.join(outage.affected_services)}",
    ]
    if outage.affected_regions:
        lines.append(f"**Affected Regions:** {', '.join(outage.affected_regions)}")
    if outage.affected_clusters:
        lines.append(f"**Affected Clusters:** {', '.join(outage.affected_clusters)}")
    if outage.symptoms:
        lines.append("")
        lines.append("**Observed Symptoms:**")
        for s in outage.symptoms:
            lines.append(f"- {s}")
    if outage.additional_context:
        lines.append("")
        lines.append(f"**Additional Context:** {outage.additional_context}")
    lines.append("")
    lines.append(
        "Please investigate this outage using the available tools. "
        "Start with Azure Service Health and Resource Health to rule out Azure platform issues, "
        "then pull live telemetry, map dependencies, and match against historical RCAs and TSGs."
    )
    return "\n".join(lines)


def _parse_response(outage_id: str, text: str, messages: list[dict]) -> AgentResponse:
    """
    Extract structured fields from the LLM's response and tool call results.
    """
    relevant_rcas: list[dict] = []
    relevant_tsgs: list[dict] = []
    dependency_risks: list[str] = []
    live_telemetry_summaries: list[str] = []

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        try:
            data = json.loads(msg["content"])
        except (json.JSONDecodeError, KeyError):
            continue

        if isinstance(data, list) and data:
            if "root_cause" in data[0]:
                relevant_rcas.extend(data)
            elif "diagnostic_steps" in data[0]:
                relevant_tsgs.extend(data)
        elif isinstance(data, dict):
            if "tier1_risks" in data:
                dependency_risks.extend(data.get("tier1_risks", []))
                dependency_risks.extend(data.get("blast_radius", []))
            # Collect live data summaries
            if "summary" in data and any(
                k in data for k in ("failed_requests", "degraded", "active_outages")
            ):
                summary_text = data.get("summary", "")
                if summary_text and "No " not in summary_text:
                    live_telemetry_summaries.append(summary_text)

    recommended_actions = _extract_numbered_list(text)
    insights = _extract_insights(text)

    # Prepend live data summaries as high-priority insights
    for summary in live_telemetry_summaries:
        insights.insert(
            0,
            RecoveryInsight(
                category="Live Azure Telemetry",
                detail=summary,
                confidence="high",
                source="Azure Monitor / Resource Health",
            ),
        )

    seen: set[str] = set()
    dep_risks_dedup: list[str] = []
    for r in dependency_risks:
        if r not in seen:
            seen.add(r)
            dep_risks_dedup.append(r)

    estimated_rto = None
    if relevant_rcas:
        times = [r.get("time_to_mitigate_minutes", 0) for r in relevant_rcas if r.get("time_to_mitigate_minutes")]
        if times:
            estimated_rto = max(times)

    return AgentResponse(
        outage_id=outage_id,
        summary=_extract_summary(text),
        insights=insights,
        recommended_actions=recommended_actions,
        relevant_rcas=relevant_rcas,
        relevant_tsgs=relevant_tsgs,
        dependency_risks=dep_risks_dedup,
        estimated_rto_minutes=estimated_rto,
    )


def _extract_summary(text: str) -> str:
    lines = text.strip().splitlines()
    summary_lines = []
    in_summary = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_summary and summary_lines:
                break
            continue
        lower = stripped.lower()
        if "summary" in lower or "overview" in lower:
            in_summary = True
            continue
        if in_summary or (not summary_lines and not stripped.startswith("#")):
            summary_lines.append(stripped)
            in_summary = True
            if len(summary_lines) >= 5:
                break
    return " ".join(summary_lines) if summary_lines else text[:500]


def _extract_numbered_list(text: str) -> list[str]:
    import re

    items = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+[\.\)]\s+(.+)$", line)
        if match:
            items.append(match.group(1).strip())
    return items


def _extract_insights(text: str) -> list[RecoveryInsight]:
    insights = []
    current_heading = None
    current_lines: list[str] = []

    section_map = {
        "root cause": "Root Cause Hypothesis",
        "dependency": "Dependency Risk",
        "mitigation": "Mitigation Step",
        "recommended": "Recommended Action",
        "insight": "Key Insight",
        "risk": "Risk",
        "telemetry": "Live Telemetry Finding",
        "resource health": "Resource Health",
        "service health": "Azure Service Health",
    }

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower().lstrip("#").strip()
        matched_category = None
        for keyword, category in section_map.items():
            if keyword in lower:
                matched_category = category
                break
        if matched_category:
            if current_heading and current_lines:
                insights.append(
                    RecoveryInsight(
                        category=current_heading,
                        detail=" ".join(current_lines),
                        source="LLM analysis",
                    )
                )
            current_heading = matched_category
            current_lines = []
        elif current_heading and stripped and not stripped.startswith("#"):
            current_lines.append(stripped)

    if current_heading and current_lines:
        insights.append(
            RecoveryInsight(
                category=current_heading,
                detail=" ".join(current_lines),
                source="LLM analysis",
            )
        )

    return insights
