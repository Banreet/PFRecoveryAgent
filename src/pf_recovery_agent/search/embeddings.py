"""
Embeddings helper for PF Recovery Agent.

Generates Azure OpenAI vector embeddings that are used at query time to
perform vector similarity search in Azure AI Search.  Returns None silently
when Azure OpenAI is not configured so that callers can fall back to
text-only search.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embed_client():
    """
    Return (client, deployment) for Azure OpenAI embeddings, or (None, None).

    Result is cached so the client is only initialised once per process.
    """
    from pf_recovery_agent.config import (
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        AZURE_OPENAI_ENDPOINT,
        USE_AZURE,
    )

    if not (USE_AZURE and AZURE_OPENAI_EMBEDDING_DEPLOYMENT):
        return None, None
    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        return client, AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    except ImportError:
        return None, None


def embed_query(text: str) -> list[float] | None:
    """
    Generate a vector embedding for *text* using Azure OpenAI.

    Returns a list of floats suitable for passing to ``VectorizedQuery``,
    or ``None`` if embeddings are not configured or the call fails.
    """
    client, deployment = _get_embed_client()
    if client is None:
        return None
    try:
        response = client.embeddings.create(input=[text], model=deployment)
        return response.data[0].embedding
    except Exception as exc:
        logger.warning("embed_query failed (%s) – falling back to text search.", exc)
        return None
