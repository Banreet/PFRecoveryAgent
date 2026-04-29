"""Configuration for PF Recovery Agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Azure OpenAI – LLM (required for AI agent)
# ---------------------------------------------------------------------------
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Azure OpenAI embeddings deployment – enables vector search in Azure AI Search.
AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)
AZURE_OPENAI_EMBEDDING_DIMENSIONS: int = int(
    os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")
)

# Standard OpenAI fallback
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

USE_AZURE: bool = bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)

# ---------------------------------------------------------------------------
# Azure AI Search – semantic + vector knowledge base search
# ---------------------------------------------------------------------------
AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY", "")
AZURE_SEARCH_RCA_INDEX: str = os.getenv("AZURE_SEARCH_RCA_INDEX", "pf-rcas")
AZURE_SEARCH_TSG_INDEX: str = os.getenv("AZURE_SEARCH_TSG_INDEX", "pf-tsgs")
AZURE_SEARCH_DEPENDENCY_INDEX: str = os.getenv(
    "AZURE_SEARCH_DEPENDENCY_INDEX", "pf-dependencies"
)

USE_AZURE_SEARCH: bool = bool(AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY)

# ---------------------------------------------------------------------------
# Azure Subscription & Identity
# Used by live data tools: Monitor, Resource Health, Service Health.
# DefaultAzureCredential is used for auth – it supports Managed Identity,
# Service Principal (AZURE_CLIENT_ID/SECRET/TENANT_ID), and Azure CLI login.
# ---------------------------------------------------------------------------
AZURE_SUBSCRIPTION_ID: str = os.getenv("AZURE_SUBSCRIPTION_ID", "")
AZURE_TENANT_ID: str = os.getenv("AZURE_TENANT_ID", "")
AZURE_RESOURCE_GROUP: str = os.getenv("AZURE_RESOURCE_GROUP", "")

# ---------------------------------------------------------------------------
# Azure Monitor / Log Analytics – live telemetry queries
# ---------------------------------------------------------------------------
AZURE_LOG_ANALYTICS_WORKSPACE_ID: str = os.getenv(
    "AZURE_LOG_ANALYTICS_WORKSPACE_ID", ""
)
AZURE_MONITOR_DEFAULT_WINDOW_HOURS: int = int(
    os.getenv("AZURE_MONITOR_DEFAULT_WINDOW_HOURS", "2")
)

USE_AZURE_MONITOR: bool = bool(AZURE_LOG_ANALYTICS_WORKSPACE_ID and AZURE_SUBSCRIPTION_ID)
USE_AZURE_RESOURCE_HEALTH: bool = bool(AZURE_SUBSCRIPTION_ID)

# ---------------------------------------------------------------------------
# Knowledge base & agent settings
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(__file__).parent / "knowledge_base" / "data"
MAX_TOOL_ITERATIONS: int = 10
