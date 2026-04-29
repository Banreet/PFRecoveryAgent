"""
PF Recovery Agent – main orchestrator.

Uses OpenAI function-calling to iteratively invoke knowledge-base tools
(dependency lookup, RCA search, TSG retrieval) and synthesise a structured
recovery plan for the on-call engineer.
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
from pf_recovery_agent.tools.dependency_tool import get_service_dependencies
from pf_recovery_agent.tools.rca_tool import search_rcas
from pf_recovery_agent.tools.tsg_tool import search_tsgs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI tool / function schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_service_dependencies",
            "description": (
                "Look up cross-cluster runtime service dependencies for the given PF "
                "services. Returns upstream dependencies, downstream dependents, blast "
                "radius, and Tier-1 risks."
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
    {
        "type": "function",
        "function": {
            "name": "search_rcas",
            "description": (
                "Search historical Root Cause Analysis (RCA) records for past incidents "
                "that match the current outage's affected services and symptoms. Returns "
                "ranked RCAs with root cause, mitigation steps, and lessons learned."
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
                        "description": "Optional symptom or tag keywords (e.g. 'tls', 'oom', '429', 'cosmosdb').",
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
    {
        "type": "function",
        "function": {
            "name": "search_tsgs",
            "description": (
                "Retrieve Troubleshooting Guide (TSG) entries for the affected PF services "
                "and observed symptoms. Returns ranked TSGs with diagnostic steps, "
                "mitigation steps, and escalation paths."
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
                        "description": "Observed symptoms or error signals.",
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

Your mission is to help on-call engineers achieve faster RTO (Recovery Time Objective) by:
1. Analysing the affected PF services and their cross-cluster runtime dependencies.
2. Searching historical RCAs for similar incidents and proven mitigations.
3. Retrieving relevant TSG (Troubleshooting Guide) steps specific to the symptoms.
4. Synthesising a prioritised, actionable recovery plan.

Always:
- Use the available tools to query the knowledge base before drawing conclusions.
- Prioritise Tier-1 services in your analysis.
- Provide concrete CLI commands and steps, not vague advice.
- Estimate time-to-mitigate based on historical RCA data when possible.
- Call out upstream dependency risks that could be the root cause.

Respond in a structured manner with: Summary, Key Insights, Recommended Actions (numbered, prioritised), Relevant RCAs, and Relevant TSGs."""


# ---------------------------------------------------------------------------
# Public agent entry point
# ---------------------------------------------------------------------------

def run_agent(outage: OutageEvent) -> AgentResponse:
    """
    Run the PF Recovery Agent for the given outage event.

    This performs an agentic loop: the LLM decides which tools to call,
    tools are dispatched, results fed back, until the LLM produces a final
    answer. The answer is then parsed into a structured ``AgentResponse``.

    Args:
        outage: Structured description of the active PF outage.

    Returns:
        ``AgentResponse`` with insights, recommended actions, RCAs and TSGs.

    Raises:
        RuntimeError: If the OpenAI API is not configured or call fails.
    """
    client, model = _build_client()

    # Build the initial user message
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

        # Append assistant turn
        messages.append(message.model_dump(exclude_unset=True))

        if choice.finish_reason == "tool_calls" and message.tool_calls:
            # Execute all requested tool calls
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
            # LLM has produced its final answer
            final_text = message.content or ""
            logger.info("Agent produced final response after %d iterations", iteration + 1)
            return _parse_response(outage.id, final_text, messages)

    # Fallback if we hit the iteration limit
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
        "Please analyse this outage using the available tools and provide a comprehensive recovery plan."
    )
    return "\n".join(lines)


def _parse_response(outage_id: str, text: str, messages: list[dict]) -> AgentResponse:
    """
    Extract structured fields from the LLM's free-text response.
    Also pulls raw tool results from the message history for structured output.
    """
    relevant_rcas: list[dict] = []
    relevant_tsgs: list[dict] = []
    dependency_risks: list[str] = []

    for msg in messages:
        if msg.get("role") == "tool":
            try:
                data = json.loads(msg["content"])
            except (json.JSONDecodeError, KeyError):
                continue
            # Detect what kind of tool result this is
            if isinstance(data, list) and data and "root_cause" in data[0]:
                relevant_rcas.extend(data)
            elif isinstance(data, list) and data and "diagnostic_steps" in data[0]:
                relevant_tsgs.extend(data)
            elif isinstance(data, dict) and "tier1_risks" in data:
                dependency_risks.extend(data.get("tier1_risks", []))
                dependency_risks.extend(data.get("blast_radius", []))

    # Parse recommended actions from numbered list in the response text
    recommended_actions = _extract_numbered_list(text)

    # Build a condensed set of insights from section headers
    insights = _extract_insights(text)

    # Deduplicate dependency risks
    seen: set[str] = set()
    dep_risks_dedup = []
    for r in dependency_risks:
        if r not in seen:
            seen.add(r)
            dep_risks_dedup.append(r)

    # Estimate RTO from RCA data
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
    """Extract a one-paragraph summary from the LLM response."""
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
    """Extract numbered action items from the text."""
    import re

    items = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+[\.\)]\s+(.+)$", line)
        if match:
            items.append(match.group(1).strip())
    return items


def _extract_insights(text: str) -> list[RecoveryInsight]:
    """Build insight objects from key headings in the response."""
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
