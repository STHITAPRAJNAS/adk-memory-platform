# Agentic Composite Memory (`agentic-composite-memory`)

An enterprise-grade, **domain-agnostic**, multi-tenant **Composite Memory SDK, FastAPI Service & MCP Server** engineered for **Google Agent Development Kit (ADK)** and multi-agent A2A ecosystems.

[![PyPI version](https://img.shields.io/pypi/v/agentic-composite-memory.svg)](https://pypi.org/project/agentic-composite-memory/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agentic-composite-memory.svg)](https://pypi.org/project/agentic-composite-memory/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Documentation Hub (`docs/`)

- **[Documentation Index](./docs/README.md)**
- **[Architecture, 3-Layer Ontology & 3-Tier Context Graph Guide](./docs/architecture_and_ontology_guide.md)**
- **[Real-World Usage & Production Recipes (Multi-Agent, Cloud Auth, Bedrock/Gemini/Ollama, ADK)](./docs/real_world_usage_and_recipes.md)**
- **[Local Setup, Docker Stack & PyPI Testing Guide](./docs/local_setup_and_testing.md)**
- **[FastAPI REST, MCP Server & CLI Reference](./docs/api_and_mcp_reference.md)**
- **[PostgreSQL RLS + `pgvector` HNSW Partitioned DDL](./docs/sql/001_postgres_rls_and_partitions.sql)**

---

## Installation

```bash
# Core SDK, ADK Integration, FastAPI + MCP Server, Postgres/Redis/Neo4j adapters
pip install agentic-composite-memory

# With Cloud IAM Auth & Bedrock / GCP / Azure / Ontology extras
pip install "agentic-composite-memory[aws,gcp,azure,ontology]"

# Everything included
pip install "agentic-composite-memory[all]"
```

Both import names are supported out of the box:
```python
from agentic_composite_memory import CompositeMemoryService, RemoteCompositeMemoryService
# or
from agentic_memory_sdk import CompositeMemoryService, RemoteCompositeMemoryService
```

---

## Why `agentic-composite-memory`?

```text
+-----------------------------------------------------------------------------------+
| LAYER 0: PLATFORM UPPER ONTOLOGY (Mandatory for 100% of Agent Apps, Zero-Config)  |
|   Classes:    p:Actor, p:Object, p:Event, p:Concept, p:CandidateConcept,          |
|               p:UserPreference                                                    |
+-----------------------------------------+-----------------------------------------+
                                          | rdfs:subClassOf / Extends
          +-------------------------------+-------------------------------+
          v                                                               v
+-----------------------------------------+     +-----------------------------------+
| LAYER 1: FEDERATED DOMAIN PACKS         |     | LAYER 2: APP-LOCAL EXTENSIONS     |
| (Plug-and-Play Shared Business TBoxes)  |     | (Self-Service / Emergent Schema)  |
|   - sre_infra@v1.0 (JSON / YAML)        |     |   - Quarantined overflow in       |
|   - customer_crm@v2.0 (Pydantic)        |     |     `app:{tenant}:{app}`          |
|   - payments@v2.1 (.ttl OWL/SHACL)      |     |   - Auto-evolution to Domain TBox |
+-----------------------------------------+     +-----------------------------------+
```

1. **3-Layer Ontology-Grounded Memory**:
   - **Layer 0 (Platform Upper Ontology)**: Built-in meta-graph (`p:Actor`, `p:Object`, `p:Event`, `p:Concept`, `p:CandidateConcept`, `p:UserPreference`) requiring zero upfront configuration.
   - **Layer 1 (Federated Domain Packs)**: Register domain ontologies via **JSON/YAML**, **Python Pydantic models**, or **W3C OWL/SHACL Turtle (`.ttl`)** files using `DomainPackLoader`.
   - **Layer 2 (Two-Bucket Runtime Governance & Auto-Evolution)**: `TBoxRuntimeValidator` routes validated facts to `domain:{tenant}:{domain}` and quarantines novel/low-confidence extractions in `app:{tenant}:{app}`, with automated promotion via `evolve_app_ontology_from_overflow()`.

2. **3-Tier Context Graph + Procedural Playbooks**:
   - **Tier 1 (Episodic & Seed Store)**: Hybrid Reciprocal Rank Fusion (RRF) search (`pgvector` HNSW + BM25 GIN) with pinned cross-app user preferences (`user:{tenant}:{user_id}`) and `PIIScrubber`.
   - **Tier 2 (Bi-Temporal Semantic Knowledge Graph)**: Multi-hop reasoning (`max_hops=2`), automatic functional edge invalidation (`t_valid`, `t_invalid`), point-in-time historical queries (`as_of_iso`), and `EntityResolutionGuard` (preventing false merges of numbered/regional siblings like `payment-router-01` vs `payment-router-02`).
   - **Tier 3 (Procedural Reasoning Traces & Playbooks)**: Records agent tool-call trajectories and caches high-reward reusable workflows (`[MATCHED_PROCEDURAL_PLAYBOOKS]`).

3. **Hot-Path vs. Sleep-Time Batch Extraction (`ExtractionMode`)**:
   - `ExtractionMode.HOT_PATH`: Immediate LLM/rule ontology extraction on every turn.
   - `ExtractionMode.SLEEP_TIME`: Sub-10ms raw episodic write on the hot path, with background batch ontology extraction, session compaction (`[AUTO_SESSION_SUMMARY]`), and Layer-3 Entity Dossier synthesis (`[Dossier: ...]`) during `run_memory_maintenance()`.

4. **Independent Per-Port or Single-Stack Database Adapters + Cloud Auth**:
   - Run **100% on PostgreSQL** (`CompositeMemoryService.from_postgres_stack`), **100% on Redis** (`CompositeMemoryService.from_redis_stack`), or **Mix-and-Match** per port (`PostgreSQL + Neo4j/Neptune + Redis`).
   - Built-in **Enterprise Cloud DB Auth** (`AwsIamDatabaseAuthProvider` for RDS/Aurora/Neptune/ElastiCache + STS `AssumeRole`, `GcpCloudSqlIamAuthProvider`, `AzureEntraDatabaseAuthProvider`, `VaultOrSecretManagerAuthProvider`, and `EnterpriseTlsConfig` mTLS).

5. **Multi-Provider LLM & Embedding Adapters**:
   - Google **Gemini**, **Amazon Bedrock** (Converse API + Titan/Cohere Embeddings), **OpenAI / Azure OpenAI / vLLM**, **Ollama**, and local **SentenceTransformers**.

6. **Unified FastAPI + MCP Server in One Pod**:
   - `adk-memory serve-api` co-hosts both REST (`/v1/...`) and MCP (`/mcp/http`, `/mcp/sse`, `/v1/mcp/tools`, `/v1/mcp/call`) simultaneously.

---

## Real-World Quickstart

### 1. Single-Stack PostgreSQL, Single-Stack Redis, or Hybrid Stack

```python
from agentic_composite_memory import (
    AwsIamDatabaseAuthProvider,
    BedrockEmbeddingAdapter,
    BedrockOntologyExtractorAdapter,
    CompositeMemoryService,
)

# 100% PostgreSQL stack with AWS RDS IAM Auth + Amazon Bedrock
auth = AwsIamDatabaseAuthProvider(
    region="us-east-1",
    service="rds-db",
    db_user="memory_iam_user",
    host="mydb.cluster-xyz.us-east-1.rds.amazonaws.com",
)
service = CompositeMemoryService.from_postgres_stack(
    dsn="postgresql://mydb.cluster-xyz.us-east-1.rds.amazonaws.com:5432/memory",
    embedding_adapter=BedrockEmbeddingAdapter(region="us-east-1", target_dim=1536),
    extractor_adapter=BedrockOntologyExtractorAdapter(region="us-east-1"),
    auth_provider=auth,
)
```

### 2. Google ADK Agent Pod Integration (`RemoteCompositeMemoryService`)

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from agentic_composite_memory import (
    RemoteCompositeMemoryService,
    get_adk_memory_tools,
    non_blocking_memory_commit_callback,
)

remote_memory = RemoteCompositeMemoryService(base_url="http://memory-platform:8095")

agent = LlmAgent(
    name="sre_commander",
    model="gemini-2.5-flash",
    tools=get_adk_memory_tools(remote_memory, default_app_name="acme:sre_commander"),
    after_agent_callback=non_blocking_memory_commit_callback,
)

runner = Runner(app_name="acme:sre_commander", agent=agent, memory_service=remote_memory)
```

### 3. Running the Unified FastAPI + MCP Server

```bash
# Start unified REST (/v1/*) + MCP (/mcp/http, /mcp/sse, /v1/mcp/*) server on port 8095
adk-memory serve-api --host 0.0.0.0 --port 8095 --mode in_memory
```

---

## Local Setup & Testing the Package (Docker Postgres + Neo4j + Redis)

You can test the full multi-backend platform locally using the included [`docker-compose.yml`](./docker-compose.yml):

```bash
# 1. Start PostgreSQL 16 (pgvector + RLS), Neo4j 5, and Redis 7
docker compose up -d --wait

# 2. Run the standalone unit & platform test suite (11 tests)
uv run --extra dev pytest tests/test_composite_memory_platform.py -v

# 3. Run the live Docker + Multi-Agent Google ADK A2A E2E suite
uv run --extra dev pytest tests/test_a2a_docker_e2e.py -v -s

# 4. Run the 5-Agent Enterprise Stress & Maturity Benchmark against live Docker
uv run python examples/realistic_enterprise_stress_test.py

# 5. Test the published PyPI package in an isolated environment against live Docker
uv run --isolated --no-project --refresh --with "agentic-composite-memory[all]" \
  python examples/realistic_enterprise_stress_test.py
```

See **[docs/local_setup_and_testing.md](./docs/local_setup_and_testing.md)** for a complete breakdown of the 5-agent enterprise benchmark and measured latencies.

---

## License

Licensed under the **Apache License, Version 2.0** — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
