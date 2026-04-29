# PFRecoveryAgent

[PM AI Hack] AI Agent which runs when an Azure PF outage happens, and helps in faster RTO by providing insights based on PF specific knowledge, cross cluster run time service dependencies, RCAs, PF Service TSGs etc.

---

## Overview

**PFRecoveryAgent** is an Azure-native AI agent for on-call engineers responding to Azure PF (Platform Foundation) service outages. When triggered during an active incident, the agent:

1. **Queries live Azure data sources** – Azure Service Health, Resource Health, and Log Analytics – to ground its analysis in the actual state of the environment.
2. **Maps the blast radius** using a cross-cluster runtime service dependency graph.
3. **Searches historical knowledge** (RCAs, TSGs) stored in Azure AI Search with semantic + vector search for similar past incidents and proven mitigations.
4. **Synthesises a prioritised recovery plan** using Azure OpenAI (GPT-4o), with concrete `kubectl` and `az` CLI commands.

### Key capabilities

| Capability | Technology |
|---|---|
| LLM agent / function-calling | Azure OpenAI (GPT-4o) |
| Semantic + vector knowledge search | Azure AI Search (hybrid search + semantic ranker) |
| Vector embeddings | Azure OpenAI (`text-embedding-3-small`) |
| Live error telemetry (KQL) | Azure Monitor / Log Analytics |
| Azure resource degradation status | Azure Resource Health API |
| Azure platform outage detection | Azure Service Health API |
| Authentication | `DefaultAzureCredential` (MSI / SPN / `az login`) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         On-Call Engineer                        │
│                    python main.py run ...                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ OutageEvent
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PF Recovery Agent                            │
│              (Azure OpenAI GPT-4o function-calling)             │
│                                                                 │
│  Agentic loop – LLM decides which tools to call in sequence:    │
│                                                                 │
│  1. get_azure_service_health ──► Azure Service Health API       │
│  2. get_resource_health      ──► Azure Resource Health API      │
│  3. get_live_telemetry       ──► Log Analytics (KQL queries)    │
│  4. get_service_dependencies ──► Dependency graph (JSON/Search) │
│  5. search_rcas              ──► Azure AI Search (pf-rcas)      │
│  6. search_tsgs              ──► Azure AI Search (pf-tsgs)      │
│                                                                 │
│  Final output: AgentResponse with summary, recommended actions, │
│  estimated RTO, matching RCAs, TSGs, and dependency risks.      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
PFRecoveryAgent/
├── main.py                          # CLI entry point
├── requirements.txt
├── pyproject.toml
├── .env.example                     # Environment variable template
└── src/
    └── pf_recovery_agent/
        ├── agent.py                 # Agent orchestrator (agentic loop)
        ├── config.py                # All configuration / env vars
        ├── credential.py            # DefaultAzureCredential helper
        ├── models.py                # Pydantic data models
        ├── knowledge_base/
        │   ├── loader.py            # JSON knowledge base loader
        │   └── data/
        │       ├── rca_database.json         # Historical RCA records
        │       ├── tsg_database.json         # Troubleshooting guides
        │       └── service_dependencies.json # Cross-cluster dependency graph
        ├── search/
        │   ├── azure_search_client.py  # Azure AI Search index mgmt + queries
        │   ├── embeddings.py           # Azure OpenAI embeddings helper
        │   └── indexer.py              # Seed Azure AI Search from JSON files
        └── tools/
            ├── azure_monitor_tool.py    # Live KQL telemetry queries
            ├── resource_health_tool.py  # Azure Resource Health per-resource status
            ├── service_health_tool.py   # Azure Service Health platform events
            ├── dependency_tool.py       # Cross-cluster dependency lookup
            ├── rca_tool.py              # RCA search (Azure AI Search + local fallback)
            └── tsg_tool.py              # TSG search (Azure AI Search + local fallback)
tests/
    ├── test_agent.py
    ├── test_knowledge_base.py
    ├── test_live_data_tools.py     # Mocked Azure SDK tests
    ├── test_search_client.py
    └── test_tools.py
```

---

## Setup

### Prerequisites

- Python 3.10+
- Azure subscription with:
  - **Azure OpenAI** resource with `gpt-4o` deployment (for the agent)
  - **Azure OpenAI** resource with `text-embedding-3-small` or `ada-002` deployment (for vector search)
  - **Azure AI Search** resource (Basic tier or higher for semantic search)
  - **Log Analytics workspace** linked to Application Insights for PF services
  - **Azure Resource Group** containing PF infrastructure (AKS, CosmosDB, Service Bus, etc.)

### Install

```bash
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```ini
# Azure OpenAI – LLM
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure OpenAI – Embeddings (enables vector search)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=...

# Azure Subscription (for live data tools)
AZURE_SUBSCRIPTION_ID=...
AZURE_RESOURCE_GROUP=pf-resource-group

# Log Analytics workspace ID (for live telemetry)
AZURE_LOG_ANALYTICS_WORKSPACE_ID=...
```

**Authentication** for live data tools uses `DefaultAzureCredential`. For local development, `az login` is sufficient. In production, configure a Managed Identity or service principal:

```bash
az login                              # local development
# OR set: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID (service principal)
```

### Seed Azure AI Search (one-time setup)

```bash
# With vector embeddings (recommended – enables hybrid search):
python main.py index-knowledge-base

# Text-only search (no embeddings deployment needed):
python main.py index-knowledge-base --skip-embeddings
```

---

## Usage

### Analyse an active outage

```bash
# Interactive mode (prompts for details):
python main.py run

# Via flags:
python main.py run \
  --outage-id ICM-12345 \
  --title "PFGateway 502 errors in East US" \
  --services PFGateway PFOrchestrator \
  --regions eastus \
  --symptoms "HTTP 502" "health probe failing" "TLS handshake error" \
  --severity SEV1

# Via JSON file:
python main.py run --from-file outage.json

# Output raw JSON (for automation / ICM integration):
python main.py run --from-file outage.json --json
```

**Example `outage.json`:**
```json
{
  "id": "ICM-98765",
  "title": "PFStorageService CosmosDB latency spike",
  "affected_services": ["PFStorageService", "PFWorkerPool"],
  "affected_regions": ["eastus"],
  "severity": "SEV1",
  "symptoms": ["429 throttling errors", "job queue backup", "worker pool idle"],
  "additional_context": "Started after 14:30 UTC, coincides with data migration job"
}
```

### Demo mode (no API keys needed)

```bash
python main.py demo
```

### Other CLI commands

```bash
python main.py search-status         # Show configuration status
python main.py list-services         # List all PF services in the dependency graph
python main.py list-rcas             # List all RCA records
python main.py list-tsgs             # List all TSG entries
```

---

## How the agent works

When `run_agent(outage)` is called, an agentic loop begins:

```
User: "PFGateway is returning 502s in East US"
  │
  ▼
Agent → calls get_azure_service_health(regions=["eastus"])
  │       ← "No active Azure outages detected"
  │
  ▼
Agent → calls get_resource_health(resource_group="pf-rg")
  │       ← "CosmosDB account 'pf-cosmos-eastus' is Degraded"  ← 🚨
  │
  ▼
Agent → calls get_live_telemetry(["PFGateway", "PFStorageService"], time_window_hours=2)
  │       ← "PFStorageService: 94% error rate | Top exception: CosmosException 503"
  │
  ▼
Agent → calls get_service_dependencies(["PFStorageService"])
  │       ← "Downstream: PFWorkerPool, PFOrchestrator, PFResultCollector (all Tier-1 risk)"
  │
  ▼
Agent → calls search_rcas(["PFStorageService"], keywords=["cosmosdb", "429", "throttling"])
  │       ← RCA-2024-003: "CosmosDB throttling – mitigated in 30min"
  │
  ▼
Agent → calls search_tsgs(["PFStorageService"], symptoms=["429", "throttling"])
  │       ← TSG-003: "PFStorageService Failures" with diagnostic + mitigation steps
  │
  ▼
Agent produces final response:
  Summary | Root Cause Hypothesis | Recommended Actions | RCAs | TSGs | Estimated RTO: 30min
```

---

## Adding new knowledge

### RCAs
Edit `src/pf_recovery_agent/knowledge_base/data/rca_database.json` and add entries following the schema, then re-run `python main.py index-knowledge-base`.

### TSGs
Edit `src/pf_recovery_agent/knowledge_base/data/tsg_database.json` similarly.

### Service dependencies
Edit `src/pf_recovery_agent/knowledge_base/data/service_dependencies.json` to add new PF services or update dependency relationships.

---

## Running tests

```bash
pytest tests/ -v
```

All tests run without live Azure credentials (Azure SDK calls are mocked).

---

## Extending with custom Azure resources

The `resource_health_tool.py` module checks all resources in `AZURE_RESOURCE_GROUP` by default. To check specific resources by their resource IDs:

```python
from pf_recovery_agent.tools.resource_health_tool import get_resource_health

result = get_resource_health(
    resource_ids=[
        "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.DocumentDB/databaseAccounts/pf-cosmos",
        "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.ContainerService/managedClusters/pf-aks",
    ]
)
```

