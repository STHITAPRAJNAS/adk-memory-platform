# Agentic Composite Memory (`agentic-composite-memory`)

An enterprise-grade, **domain-agnostic**, multi-tenant **Composite Memory SDK, FastAPI Service & Model Context Protocol (MCP) Server** engineered for **Google Agent Development Kit (ADK)** and multi-agent A2A ecosystems.

[![PyPI version](https://img.shields.io/pypi/v/agentic-composite-memory.svg)](https://pypi.org/project/agentic-composite-memory/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agentic-composite-memory.svg)](https://pypi.org/project/agentic-composite-memory/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Documentation Hub (`docs/`)

- **[Documentation Index](./docs/README.md)**
- **[Domain Pack Authoring & Ontology Engineering Guide (6 Creation Methods + Retail & Payment Processing Blueprints)](./docs/domain_pack_authoring_guide.md)**
- **[Enterprise Best Practices & Production Go-Live Guide](./docs/best_practices_guide.md)**
- **[Architecture, 3-Layer Ontology & 3-Tier Context Graph Guide](./docs/architecture_and_ontology_guide.md)**
- **[Real-World Usage, Retail/SRE Domain Suites & The 3 Agent Consumption Patterns (SDK, FastAPI, MCP)](./docs/real_world_usage_and_recipes.md)**
- **[FastAPI REST, MCP Protocol (`2026-07-28` / `2025-11-25`) & CLI Reference](./docs/api_and_mcp_reference.md)**
- **[Local Setup, Docker Stack & PyPI Testing Guide](./docs/local_setup_and_testing.md)**
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
|   - retail_omnichannel@1.0.0 (.json/.ttl)|    |   - Quarantined overflow in       |
|   - sre_infra@v1.2 (JSON / YAML)        |     |     `app:{tenant}:{app}`          |
|   - customer_crm@v2.0 (Pydantic)        |     |   - Auto-evolution to Domain TBox |
+-----------------------------------------+     +-----------------------------------+
```

1. **3-Layer Domain-Agnostic Ontology**:
   - **Layer 0 (Platform Upper Ontology)**: Built-in meta-graph (`p:Actor`, `p:Object`, `p:Event`, `p:Concept`, `p:CandidateConcept`, `p:UserPreference`) requiring zero upfront configuration.
   - **Layer 1 (Federated Domain Packs)**: Register domain ontologies for **Retail & Supply Chain**, **SRE & Cloud Ops**, **Healthcare**, **FinOps**, or **Customer CRM** via **JSON/YAML**, **Python Pydantic models**, or **W3C OWL/SHACL Turtle (`.ttl`)** files using `DomainPackLoader`.
   - **Layer 2 (Two-Bucket Runtime Governance & Auto-Evolution)**: `TBoxRuntimeValidator` routes validated facts to `domain:{tenant}:{domain}` and quarantines novel/low-confidence extractions in `app:{tenant}:{app}`, with automated promotion via `evolve_app_ontology_from_overflow()`.

2. **3-Tier Context Graph + Procedural Playbooks**:
   - **Tier 1 (Episodic & Seed Store)**: Hybrid Reciprocal Rank Fusion (RRF) search (`pgvector` HNSW + BM25 GIN) with pinned cross-app user preferences (`user:{tenant}:{user_id}`) and `PIIScrubber`.
   - **Tier 2 (Bi-Temporal Semantic Knowledge Graph)**: Multi-hop reasoning (`max_hops=2`), automatic functional edge invalidation (`t_valid`, `t_invalid`), point-in-time historical queries (`as_of_iso`), and `EntityResolutionGuard` (preventing false merges of numbered/regional siblings like `FC-Seattle-01` vs `FC-Seattle-02`).
   - **Tier 3 (Procedural Reasoning Traces & Playbooks)**: Records agent tool-call trajectories and caches high-reward reusable workflows (`[MATCHED_PROCEDURAL_PLAYBOOKS]`).

3. **Hot-Path vs. Sleep-Time Batch Extraction (`ExtractionMode`)**:
   - `ExtractionMode.HOT_PATH`: Immediate LLM/rule ontology extraction on every turn.
   - `ExtractionMode.SLEEP_TIME`: Sub-10ms raw episodic write on the hot path, with background batch ontology extraction, session compaction (`[AUTO_SESSION_SUMMARY]`), and Layer-3 Entity Dossier synthesis (`[Dossier: ...]`) during `run_memory_maintenance()`.

4. **Independent Per-Port or Single-Stack Database Adapters + Cloud Auth**:
   - Run **100% on PostgreSQL** (`CompositeMemoryService.from_postgres_stack`), **100% on Redis** (`CompositeMemoryService.from_redis_stack`), or **Mix-and-Match** per port (`PostgreSQL + Neo4j/Neptune + Redis`).
   - Built-in **Enterprise Cloud DB Auth** (`AwsIamDatabaseAuthProvider` for RDS/Aurora/Neptune/ElastiCache + STS `AssumeRole`, `GcpCloudSqlIamAuthProvider`, `AzureEntraDatabaseAuthProvider`, `VaultOrSecretManagerAuthProvider`, and `EnterpriseTlsConfig` mTLS).

5. **Multi-Provider LLM & Embedding Adapters**:
   - Google **Gemini**, **Amazon Bedrock** (Converse API + Titan/Cohere Embeddings), **OpenAI / Azure OpenAI / vLLM**, **Ollama**, and local **SentenceTransformers**.

6. **100% Data-Plane + Control-Plane Parity Across SDK, FastAPI & MCP (`2026-07-28` / `2025-11-25`)**:
   - `adk-memory serve-api` co-hosts both REST (`/v1/...`) and the latest **MCP `2026-07-28` Stateless Request/Response Core & `2025-11-25` Streamable HTTP** (`/mcp/http/`, `/mcp/sse`, `/v1/mcp/tools`, `/v1/mcp/call`) with **11 Annotated Tools**, **4 Live Resources**, and **2 Workflow Prompts**.

---

## The 3 Agent Consumption Patterns

### Pattern 1: Direct In-Process Python SDK (`CompositeMemoryService`)
```python
from agentic_composite_memory import (
    AwsIamDatabaseAuthProvider,
    BedrockEmbeddingAdapter,
    BedrockOntologyExtractorAdapter,
    CompositeMemoryService,
)

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

### Pattern 2: Co-Hosted FastAPI Service + `RemoteCompositeMemoryService` (Google ADK)
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
    name="order_concierge",
    model="gemini-2.5-flash",
    tools=get_adk_memory_tools(remote_memory, default_app_name="nordstrom_global:order_concierge"),
    after_agent_callback=non_blocking_memory_commit_callback,
)

runner = Runner(app_name="nordstrom_global:order_concierge", agent=agent, memory_service=remote_memory)
```

### Pattern 3: LangGraph & LangChain Integration (`BaseStore`, `StateGraph` Nodes & Tools)
```python
from agentic_composite_memory import (
    LangGraphCompositeMemoryStore,
    RemoteCompositeMemoryService,
    create_langgraph_memory_nodes,
    get_langgraph_memory_tools,
)

remote_memory = RemoteCompositeMemoryService(base_url="http://memory-platform:8095")

# A) Drop-in LangGraph BaseStore: builder.compile(store=lg_store)
lg_store = LangGraphCompositeMemoryStore(
    remote_memory,
    default_app_name="nordstrom_global:order_concierge",
    default_user_id="shopper_maya_lin",
)

# B) Pre-built StateGraph Nodes (recall_node, check_answerable_node, commit_node)
nodes = create_langgraph_memory_nodes(
    remote_memory,
    default_app_name="nordstrom_global:order_concierge",
    inject_system_message=True,
)

# C) LangGraph ReAct / ToolNode Tools (StructuredTool compatible)
tools = get_langgraph_memory_tools(
    remote_memory,
    default_app_name="nordstrom_global:order_concierge",
)
```

### Pattern 4: Model Context Protocol (MCP `2026-07-28` & `2025-11-25` Server)
```bash
# Start unified FastAPI REST (/v1/*) + MCP (/mcp/http/, /mcp/sse, /v1/mcp/*) server on port 8095
adk-memory serve-api --host 0.0.0.0 --port 8095 --mode in_memory

# Or run standalone stdio MCP server for Claude Desktop / Cursor / Windsurf
adk-memory serve-mcp --mode in_memory
```

**Claude Desktop / Cursor / Windsurf `mcpServers` Config (`Streamable HTTP` or `stdio`)**:
```json
{
  "mcpServers": {
    "agentic-composite-memory": {
      "command": "uvx",
      "args": ["--from", "agentic-composite-memory", "adk-memory", "serve-mcp", "--mode", "in_memory"]
    }
  }
}
```

---

## Complete Platform Parity Matrix (SDK vs. FastAPI vs. MCP)

| Platform Capability | In-Process SDK (`CompositeMemoryService` / `Registrar`) | FastAPI REST (`api.py`) & `RemoteCompositeMemoryService` | MCP Server Tool / Resource (`mcp_server.py`) |
| :--- | :--- | :--- | :--- |
| **Health & Governance Inspection** | `service.circuit_breaker`, `registry.resolve_context` | `GET /health`, `GET /healthz`, `GET /v1/apps/{tenant}/{app}` | Tool: `inspect_platform_governance`<br>Resource: `memory://governance/health` |
| **List / Register Domain Pack (JSON, `.ttl`, SQL DDL, Ossie YAML)** | `registrar.register_domain_pack`, `DomainPackLoader.from_sql_ddl` / `from_ossie_yaml` | `GET /v1/domains/packs`, `POST /v1/domains/packs`, `POST /v1/domains/packs/turtle`, `POST /v1/domains/packs/sql-ddl`, `POST /v1/domains/packs/ossie` | Tool: `register_domain_pack`<br>Resource: `memory://domains/packs` |
| **Cold-Start Autonomous Domain Induction** | `service.induce_domain_pack(...)` | `POST /v1/domains/packs/induce` | Tool: `induce_domain_pack` |
| **Certify Domain Pack (`certified=True`)** | `registrar.certify_domain_pack(pack_id)` | `POST /v1/domains/packs/certify` | Tool: `certify_domain_pack` |
| **Onboard / Inspect Agent App** | `registrar.register_adk_app(manifest)` | `POST /v1/apps/register`, `GET /v1/apps/{tenant}/{app}` | Tool: `register_app_manifest`<br>Resource: `memory://apps/{tenant_id}/{app_name}/tbox` |
| **Pre-Flight Answerability & Coverage** | `service.check_answerable(app_name, user_id, query)` | `POST /v1/memory/check-answerable` | Tool: `check_answerable` (`readOnlyHint=true`) |
| **Ingest Turns (Hot-Path or Sleep-Time)** | `service.add_events_to_memory(..., extraction_mode=...)` | `POST /v1/memory/events` | Tool: `commit_episode` / `ingest_memory` |
| **3-Tier Hybrid Recall + Point-in-Time** | `service.search_memory(..., as_of_iso=..., max_hops=2)` | `POST /v1/memory/search` | Tool: `recall_context`<br>Prompt: `memory_grounded_assistant` |
| **Record Tier-3 Reasoning Trace** | `service.record_decision_trace(...)` | `POST /v1/traces` | Tool: `record_decision_trace` |
| **Register / Reinforce Tier-3 Playbook** | `service.register_playbook(...)` | `POST /v1/playbooks/register` | Tool: `register_playbook` |
| **Bottom-Up Ontology Auto-Evolution** | `service.evolve_app_ontology_from_overflow(...)` | `POST /v1/ontology/evolve` | Tool: `evolve_app_ontology` |
| **Sleep-Time Maintenance & Dossiers** | `service.run_memory_maintenance(...)` | `POST /v1/maintenance/run` | Tool: `run_memory_maintenance` |
| **Ontology Audit Provenance Log** | `registrar.audit_log` | `GET /v1/audit/ontology` | Resource: `memory://audit/ontology` |
| **GDPR Right-to-be-Forgotten Purge** | `service.purge_user_lineage(...)` | `POST /v1/lineage/purge` | Tool: `purge_user_lineage` (`destructiveHint=true`) |

See **[docs/api_and_mcp_reference.md](./docs/api_and_mcp_reference.md)** for full request/response schemas and `curl` examples.

---

## Local Setup & Testing the Package (Docker Postgres + Neo4j + Redis)

You can test the full multi-backend platform locally using the included [`docker-compose.yml`](./docker-compose.yml):

```bash
# 1. Start PostgreSQL 16 (pgvector + RLS), Neo4j 5, and Redis 7
docker compose up -d --wait

# 2. Run the standalone unit & platform test suite (13 tests)
uv run --extra dev pytest tests/test_composite_memory_platform.py -v

# 3. Run the live Docker + Multi-Agent Google ADK A2A E2E suite
uv run --extra dev pytest tests/test_a2a_docker_e2e.py -v -s

# 4. Run the 4-Agent Omnichannel Retail Suite & 3 Consumption Patterns (SDK, FastAPI, MCP) Demo
uv run python examples/retail_omnichannel_suite.py
uv run python examples/three_consumption_patterns_demo.py

# 5. Run the 5-Agent Enterprise Stress & Maturity Benchmark against live Docker
uv run python examples/realistic_enterprise_stress_test.py

# 6. Test the published PyPI package in an isolated environment against live Docker
uv run --isolated --no-project --refresh --with "agentic-composite-memory[all]" \
  python examples/realistic_enterprise_stress_test.py
```

See **[docs/local_setup_and_testing.md](./docs/local_setup_and_testing.md)** for a complete breakdown of the 5-agent enterprise benchmark and measured latencies.

---

## License

Licensed under the **Apache License, Version 2.0** — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
