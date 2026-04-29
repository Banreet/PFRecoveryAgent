"""Configuration for PF Recovery Agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Azure OpenAI (preferred) vs standard OpenAI (fallback)
# ---------------------------------------------------------------------------
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

USE_AZURE: bool = bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)

# Knowledge base data directory
DATA_DIR: Path = Path(__file__).parent / "knowledge_base" / "data"

# Agent settings
MAX_TOOL_ITERATIONS: int = 10
