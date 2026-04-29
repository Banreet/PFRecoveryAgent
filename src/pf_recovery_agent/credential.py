"""
Shared Azure credential helper for PF Recovery Agent live data tools.

Uses ``DefaultAzureCredential`` which tries the following in order:
1. ``EnvironmentCredential`` – AZURE_CLIENT_ID + AZURE_CLIENT_SECRET + AZURE_TENANT_ID
2. ``WorkloadIdentityCredential`` – Azure Workload Identity (AKS pods)
3. ``ManagedIdentityCredential`` – Azure Managed Identity (VMs, App Service, AKS node)
4. ``AzureCliCredential`` – `az login` (local development)
5. ``AzurePowerShellCredential`` – `Connect-AzAccount` (PowerShell)

For local development, running `az login` is sufficient.
For production, configure a Managed Identity or Service Principal with:
    AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_credential():
    """
    Return a cached ``DefaultAzureCredential`` instance.

    Raises ``ImportError`` if ``azure-identity`` is not installed.
    """
    try:
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential()
    except ImportError as exc:
        raise ImportError(
            "azure-identity is required. Run: pip install azure-identity"
        ) from exc
