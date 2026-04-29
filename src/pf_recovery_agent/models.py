"""Pydantic data models for PF Recovery Agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutageSeverity(str, Enum):
    SEV0 = "SEV0"
    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"


class OutageStatus(str, Enum):
    ACTIVE = "active"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


class OutageEvent(BaseModel):
    """Represents an Azure PF outage event."""

    id: str = Field(..., description="Unique outage identifier (e.g. ICM incident ID)")
    title: str = Field(..., description="Brief description of the outage")
    affected_services: list[str] = Field(
        ..., description="List of PF service names impacted"
    )
    affected_regions: list[str] = Field(
        default_factory=list, description="Azure regions affected"
    )
    affected_clusters: list[str] = Field(
        default_factory=list, description="Specific cluster names affected"
    )
    severity: OutageSeverity = Field(default=OutageSeverity.SEV1)
    status: OutageStatus = Field(default=OutageStatus.ACTIVE)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    symptoms: list[str] = Field(
        default_factory=list, description="Observed symptoms / error signals"
    )
    additional_context: str = Field(
        default="", description="Free-text context provided by the on-call engineer"
    )


class ServiceDependency(BaseModel):
    """Describes runtime dependencies for a PF service."""

    service_name: str
    upstream_dependencies: list[str] = Field(
        default_factory=list,
        description="Services this service calls",
    )
    downstream_dependents: list[str] = Field(
        default_factory=list,
        description="Services that call this service",
    )
    cluster: str = Field(default="", description="Owning cluster")
    tier: int = Field(default=1, description="Service tier (1=critical)")
    description: str = Field(default="")


class RCA(BaseModel):
    """Root Cause Analysis record."""

    id: str
    title: str
    affected_services: list[str]
    date: str
    root_cause: str
    mitigation_steps: list[str]
    time_to_mitigate_minutes: int
    tags: list[str] = Field(default_factory=list)
    lessons_learned: str = Field(default="")


class TSG(BaseModel):
    """Troubleshooting Guide entry."""

    id: str
    title: str
    applicable_services: list[str]
    symptoms: list[str]
    diagnostic_steps: list[str]
    mitigation_steps: list[str]
    escalation_path: str = Field(default="")
    tags: list[str] = Field(default_factory=list)


class RecoveryInsight(BaseModel):
    """A single insight produced by the agent."""

    category: str = Field(
        ...,
        description="E.g. 'Root Cause Hypothesis', 'Mitigation Step', 'Dependency Risk'",
    )
    detail: str
    confidence: str = Field(default="medium", description="high / medium / low")
    source: str = Field(default="", description="Knowledge source that produced this")


class AgentResponse(BaseModel):
    """Full response from the PF Recovery Agent."""

    outage_id: str
    summary: str
    insights: list[RecoveryInsight]
    recommended_actions: list[str]
    relevant_rcas: list[dict[str, Any]] = Field(default_factory=list)
    relevant_tsgs: list[dict[str, Any]] = Field(default_factory=list)
    dependency_risks: list[str] = Field(default_factory=list)
    estimated_rto_minutes: int | None = None
