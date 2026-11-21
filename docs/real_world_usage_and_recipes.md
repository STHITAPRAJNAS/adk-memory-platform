# Real-World Usage, Domain Suites & The 3 Agent Consumption Patterns

This guide provides complete, copy-pasteable production recipes for **`agentic-composite-memory`** (`import agentic_composite_memory` or `import agentic_memory_sdk`), covering:
1. **The 3 Consumption Patterns for Agent Apps (SDK, FastAPI, and MCP)**
2. **Omnichannel Retail & Supply-Chain Multi-Agent Suite (`retail_omnichannel@1.0.0`)**
3. **Cloud SRE, FinOps & SecOps Multi-Agent Suite (`sre_infra@1.0.0`)**
4. **Database Topologies (100% PostgreSQL, 100% Redis, or Hybrid Multi-Backend)**
5. **Enterprise Cloud Database Authentication & mTLS (AWS IAM, GCP Cloud SQL, Azure Entra ID)**
6. **Multi-Provider LLM & Embedding Adapters (Gemini, Bedrock, OpenAI/vLLM, Ollama, Local HuggingFace)**

---

## 1. The 3 Consumption Patterns for Agent Apps

You can consume `agentic-composite-memory` in **three ways** depending on your deployment architecture. A runnable end-to-end script testing all 3 patterns side-by-side is included in [`examples/three_consumption_patterns_demo.py`](../examples/three_consumption_patterns_demo.py).

```mermaid
flowchart LR
    subgraph Pattern1["Pattern 1: In-Process Python SDK"]
        A1["ADK Runner / Python Agent"] -->|"Direct async calls"| S1["CompositeMemoryService"]
    end
    subgraph Pattern2["Pattern 2: FastAPI Microservice + Remote Client"]
        A2["Lightweight ADK Agent Pod\n(RemoteCompositeMemoryService)"] -->|"HTTP REST /v1/*"| API["FastAPI + MCP Pod\n(adk-memory serve-api :8095)"]
    end
    subgraph Pattern3["Pattern 3: Model Context Protocol (MCP)"]
        A3["ADK McpToolset / Claude / Cursor / LangGraph"] -->|"/mcp/http, /mcp/sse, stdio,\nor stateless /v1/mcp/call"| API
    end
```

---

### Pattern A: Direct In-Process Python SDK Usage (`CompositeMemoryService`)
Use this pattern when running a standalone agent service or local orchestration script where the process holds direct database connection pools:

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from agentic_composite_memory import (
    AppMemoryManifest,
    CompositeMemoryService,
    DomainPackLoader,
    Event,
    GovernanceTier,
    InMemoryUnifiedAdapter,
    PlatformOntologyRegistrar,
    DeterministicHashEmbeddingAdapter,
    RuleBasedMockExtractorAdapter,
    get_adk_memory_tools,
    non_blocking_memory_commit_callback,
    types,
)

# 1. Initialize the Service & Ontology Registrar
store = InMemoryUnifiedAdapter()  # Or PostgresUnifiedAdapter / Neo4jGraphAdapter / RedisUnifiedAdapter
service = CompositeMemoryService(
    registry_adapter=store,
    seed_store_adapter=store,
    graph_adapter=store,
    playbook_adapter=store,
    embedding_adapter=DeterministicHashEmbeddingAdapter(dim=64),
    extractor_adapter=RuleBasedMockExtractorAdapter(),
)
registrar = PlatformOntologyRegistrar(store)

# 2. Load & Register a Domain Pack (JSON, Pydantic, or W3C OWL .ttl)
retail_pack = DomainPackLoader.from_json_file("examples/domain_packs/retail_omnichannel.json")
registrar.register_domain_pack(retail_pack)

# 3. Onboard the Agent App to Tier 2 Shared Domain Governance
await registrar.onboard_app(
    AppMemoryManifest(
        tenant_id="aurora_retail",
        app_name="returns_concierge_agent",
        governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
        subscribed_domain_packs=["retail_omnichannel@1.0.0"],
    )
)

# 4. Attach directly to a Google ADK Agent & Runner
concierge_agent = LlmAgent(
    name="returns_concierge_agent",
    model="gemini-2.5-flash",
    instruction="You are a VIP Retail Returns Concierge. Always check composite memory first.",
    tools=get_adk_memory_tools(service, default_app_name="aurora_retail:returns_concierge_agent"),
    after_agent_callback=non_blocking_memory_commit_callback,
)

runner = Runner(
    app_name="aurora_retail:returns_concierge_agent",
    agent=concierge_agent,
    memory_service=service,
)
```

---

### Pattern B: FastAPI Microservice + Lightweight Remote Client (`RemoteCompositeMemoryService`)
Use this pattern in **Kubernetes or multi-container deployments** where 10s or 100s of stateless agent pods connect to a centralized Memory Platform Service (`adk-memory serve-api`) without holding database credentials or connection pools inside agent containers.

#### Step 1: Start the Centralized Memory Platform Server
```bash
# Co-hosts both FastAPI REST (/v1/*) and MCP (/mcp/*, /v1/mcp/*) on port 8095
adk-memory serve-api --host 0.0.0.0 --port 8095 --mode production
```

#### Step 2: Connect Lightweight Agent Pods via `RemoteCompositeMemoryService`
```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from agentic_composite_memory import (
    AppMemoryManifest,
    Event,
    GovernanceTier,
    ReasoningTraceStep,
    RemoteCompositeMemoryService,
    get_adk_memory_tools,
    non_blocking_memory_commit_callback,
    types,
)

# Zero DB drivers or credentials needed inside the Agent Pod!
remote_memory = RemoteCompositeMemoryService(
    base_url="http://memory-platform.default.svc.cluster.local:8095",
    api_key="optional-service-bearer-token",
)

# 1. Register App Manifest remotely (`POST /v1/apps/register`)
await remote_memory.register_manifest(
    AppMemoryManifest(
        tenant_id="aurora_retail",
        app_name="order_fulfillment_agent",
        governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
        subscribed_domain_packs=["retail_omnichannel@1.0.0"],
    )
)

# 2. Ingest Events remotely (`POST /v1/memory/events`)
await remote_memory.add_events_to_memory(
    app_name="aurora_retail:order_fulfillment_agent",
    user_id="maya_lin",
    session_id="order_sess_1",
    events=[
        Event(
            author="user",
            content=types.Content(
                role="user",
                parts=[types.Part.from_text(text="Order ORD-9001 re-routed to ship from WH-Dallas-02.")],
            ),
        )
    ],
)

# 3. Register a Reusable Tier-3 Playbook remotely (`POST /v1/playbooks/register`)
await remote_memory.register_playbook(
    app_name="aurora_retail:order_fulfillment_agent",
    user_id="maya_lin",
    trigger_intent="ORD-9001 VIP exchange and fulfillment warehouse status",
    workflow_steps=[
        "verify_ambassador_vip_tier",
        "dispatch_instant_replacement_from_wh_dallas_02",
        "issue_prepaid_return_label_no_restock_fee",
    ],
    resolution_summary="Instant no-fee VIP exchange workflow for Ambassador-VIP shoppers",
)

# 4. Search 3-Tier Composite Memory remotely (`POST /v1/memory/search`)
recall = await remote_memory.search_memory(
    app_name="aurora_retail:order_fulfillment_agent",
    user_id="maya_lin",
    query="ORD-9001 VIP exchange and fulfillment warehouse status",
    max_hops=2,
)
for m in recall.memories:
    print(m.content.parts[0].text)

# 5. Or wire directly into Google ADK Runner
agent = LlmAgent(
    name="order_fulfillment_agent",
    model="gemini-2.5-flash",
    tools=get_adk_memory_tools(remote_memory, default_app_name="aurora_retail:order_fulfillment_agent"),
    after_agent_callback=non_blocking_memory_commit_callback,
)
runner = Runner(app_name="aurora_retail:order_fulfillment_agent", agent=agent, memory_service=remote_memory)
```

---

### Pattern C: Model Context Protocol (MCP) Usage Pattern
The platform exposes a full **MCP Server** (`create_mcp_server`) supporting **3 MCP transports**:
1. **Streamable HTTP (`/mcp/http`) & SSE (`/mcp/sse`)** — Co-hosted automatically on the FastAPI server (`adk-memory serve-api`).
2. **Stateless REST-Bridge MCP (`GET /v1/mcp/tools`, `POST /v1/mcp/call`)** — 100% stateless JSON RPC over HTTP for round-robin Kubernetes Ingress load balancing without sticky sessions.
3. **Standard I/O (`adk-memory serve-mcp`)** — For local MCP hosts like Claude Desktop, Cursor, or Windsurf.

#### 1. Connecting Claude Desktop / Cursor (`mcpServers` Config)
```json
{
  "mcpServers": {
    "agentic-composite-memory": {
      "command": "adk-memory",
      "args": ["serve-mcp", "--mode", "production"],
      "env": {
        "POSTGRES_DSN": "postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform",
        "NEO4J_URI": "bolt://127.0.0.1:7687",
        "REDIS_URL": "redis://127.0.0.1:6379/0"
      }
    }
  }
}
```

#### 2. Calling Co-Hosted Stateless MCP Tools over HTTP (`/v1/mcp/call`)
Any agent framework (ADK, LangGraph, CrewAI, AutoGen, or custom microservices) can list and invoke MCP tools over standard stateless HTTP:

```python
import httpx

async with httpx.AsyncClient(base_url="http://localhost:8095") as client:
    # 1. Discover available MCP tools
    tools_resp = await client.get("/v1/mcp/tools")
    print("MCP Tools:", tools_resp.json()["tools"])
    # -> ['recall_context', 'ingest_memory', 'record_trace_step', 'register_domain_pack_json']

    # 2. Invoke `recall_context` via MCP
    call_resp = await client.post(
        "/v1/mcp/call",
        json={
            "tool_name": "recall_context",
            "arguments": {
                "app_name": "aurora_retail:returns_concierge_agent",
                "user_id": "maya_lin",
                "query": "ORD-9001 VIP exchange and fulfillment warehouse status",
            },
        },
    )
    print(call_resp.json()["result"])
```

---

## 2. Domain-Agnostic Design & The Retail Omnichannel Example Suite

The core SDK contains **zero industry-specific or payment-specific vocabulary** — Layer 0 only defines universal upper ontology primitives (`p:Actor`, `p:Object`, `p:Event`, `p:Concept`, `p:CandidateConcept`, `p:UserPreference`). All business vocabulary is loaded dynamically via **Domain Packs** in [`examples/domain_packs/`](../examples/domain_packs/):

| Included Domain Pack | File(s) | Example Entities & Functional Relations |
| :--- | :--- | :--- |
| **Omnichannel Retail & Supply Chain** | [`retail_omnichannel.json`](../examples/domain_packs/retail_omnichannel.json), [`retail_omnichannel_tbox.ttl`](../examples/domain_packs/retail_omnichannel_tbox.ttl) | `RetailCustomer`, `ProductSKU`, `FulfillmentWarehouse`, `RetailOrder`, `LoyaltyTier`; functional `ENROLLED_IN_TIER` & `SHIPPED_FROM_PRIMARY` |
| **Cloud SRE & Incident Ops** | [`sre_incident_ops.json`](../examples/domain_packs/sre_incident_ops.json) | `Microservice`, `DatabaseCluster`, `CloudRegion`; functional `DEPLOYED_IN_PRIMARY` |
| **Enterprise B2B CRM** | [`customer_crm.json`](../examples/domain_packs/customer_crm.json) | `CustomerAccount`, `SubscriptionPlan`, `AccountExecutive`; functional `SUBSCRIBED_TO_TIER` & `MANAGED_BY_AE` |
| **Healthcare Clinical Trials** | [`healthcare_clinical.json`](../examples/domain_packs/healthcare_clinical.json) | `ClinicalTrial`, `InvestigationalDrug`, `PatientCohort`; functional `CURRENT_PHASE` |
| **Financial Payments (OWL/SHACL)** | [`payments_tbox.ttl`](../examples/domain_packs/payments_tbox.ttl), [`payments_shapes.shacl.ttl`](../examples/domain_packs/payments_shapes.shacl.ttl) | W3C OWL/SHACL reference pack |

### Running the 4-Agent Omnichannel Retail Suite ([`examples/retail_omnichannel_suite.py`](../examples/retail_omnichannel_suite.py))

```bash
uv run python examples/retail_omnichannel_suite.py
```

What the Retail Omnichannel Suite demonstrates end-to-end:
1. **`personal_stylist_agent`**: Captures shopper `maya_lin`'s global fabric/sizing preference (`Organic-Linen-Size-M` in `user:aurora_retail:maya_lin`) and upgrades her loyalty tier from `Silver-Tier` to `Ambassador-VIP` (automatically closing `t_invalid` on `Silver-Tier` via functional relation `ENROLLED_IN_TIER`).
2. **`order_fulfillment_agent`**: Ingests order `ORD-9001` containing `SKU-LINEN-01` (kept distinct from sibling `SKU-LINEN-02` by `EntityResolutionGuard`), scrubs checkout credit card PAN `4532-0151-1283-0099`, and re-routes primary fulfillment from `WH-Newark-01` to `WH-Dallas-02` (supporting point-in-time historical queries via `as_of_iso`).
3. **`returns_concierge_agent`**: Registers a reusable VIP Instant Exchange Playbook (`pb_...`) and recalls the unified 3-Tier Context Window combining the VIP Exchange Playbook, `maya_lin`'s active `Ambassador-VIP` tier, 2-hop order/SKU/warehouse graph, and sizing preferences.
4. **`merchandising_trend_agent`**: Detects an emergent social commerce concept (`ViralTikTokTrend` / `FEATURED_IN_TREND`) and promotes it from app overflow into the shared retail ontology via `evolve_app_ontology_from_overflow()`.

---

## 3. Database Topologies: Single-Stack Postgres, Single-Stack Redis, or Hybrid

### Option A: 100% PostgreSQL Stack (`from_postgres_stack`)
```python
from agentic_composite_memory import (
    CompositeMemoryService,
    GeminiEmbeddingAdapter,
    GeminiOntologyExtractorAdapter,
)

service = CompositeMemoryService.from_postgres_stack(
    dsn="postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform",
    embedding_adapter=GeminiEmbeddingAdapter(output_dimensionality=1536),
    extractor_adapter=GeminiOntologyExtractorAdapter(model="gemini-2.5-flash"),
)
```

### Option B: 100% Redis Stack (`from_redis_stack`)
```python
from agentic_composite_memory import (
    CompositeMemoryService,
    GeminiEmbeddingAdapter,
    GeminiOntologyExtractorAdapter,
)

service = CompositeMemoryService.from_redis_stack(
    redis_url="redis://127.0.0.1:6379/0",
    embedding_adapter=GeminiEmbeddingAdapter(output_dimensionality=1536),
    extractor_adapter=GeminiOntologyExtractorAdapter(model="gemini-2.5-flash"),
)
```

### Option C: Hybrid Best-of-Breed Enterprise Stack (PostgreSQL + Neo4j/Neptune + Redis)
```python
from agentic_composite_memory import (
    CompositeMemoryService,
    Neo4jGraphAdapter,
    PostgresPgVectorAdapter,
    RedisPlaybookAndWorkingAdapter,
    GeminiEmbeddingAdapter,
    GeminiOntologyExtractorAdapter,
)

pg = PostgresPgVectorAdapter(dsn="postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform")
neo4j = Neo4jGraphAdapter(uri="bolt://127.0.0.1:7687", auth=("neo4j", "platform_secret"))
redis_pb = RedisPlaybookAndWorkingAdapter(redis_url="redis://127.0.0.1:6379/0")

service = CompositeMemoryService(
    registry_adapter=pg,
    seed_store_adapter=pg,
    graph_adapter=neo4j,
    playbook_adapter=redis_pb,
    embedding_adapter=GeminiEmbeddingAdapter(output_dimensionality=1536),
    extractor_adapter=GeminiOntologyExtractorAdapter(model="gemini-2.5-flash"),
)
```

---

## 4. Enterprise Cloud Database Authentication & mTLS (`agentic_composite_memory.auth`)

All database adapters accept `auth_provider: CloudAuthProvider` and `tls_config: EnterpriseTlsConfig`:

```python
from agentic_composite_memory import (
    AwsIamDatabaseAuthProvider,
    AzureEntraDatabaseAuthProvider,
    EnterpriseTlsConfig,
    GcpCloudSqlIamAuthProvider,
    PostgresPgVectorAdapter,
)

tls = EnterpriseTlsConfig(enabled=True, ca_cert_path="/etc/certs/aws-rds-global-bundle.pem")

# AWS RDS / Aurora IAM Auth (with optional cross-account STS AssumeRole)
rds_auth = AwsIamDatabaseAuthProvider(
    region="us-east-1",
    service="rds-db",
    db_user="memory_iam_user",
    host="aurora-cluster.cluster-xyz.us-east-1.rds.amazonaws.com",
    port=5432,
)
pg_adapter = PostgresPgVectorAdapter(
    dsn="postgresql://aurora-cluster.cluster-xyz.us-east-1.rds.amazonaws.com:5432/memory_db",
    auth_provider=rds_auth,
    tls_config=tls,
)
```

---

## 5. Multi-Provider LLM & Embedding Adapters (`agentic_composite_memory.llm_extractors`)

| Provider | Embedding Adapter | Ontology Extractor Adapter |
| :--- | :--- | :--- |
| **Google Gemini / Vertex AI** | `GeminiEmbeddingAdapter(model="text-embedding-004", output_dimensionality=1536)` | `GeminiOntologyExtractorAdapter(model="gemini-2.5-flash")` |
| **Amazon Bedrock** | `BedrockEmbeddingAdapter(model_id="amazon.titan-embed-text-v2:0", dimensions=1024, target_dim=1536)` | `BedrockOntologyExtractorAdapter(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")` |
| **OpenAI / Azure OpenAI / vLLM** | `OpenAICompatibleEmbeddingAdapter(base_url="https://api.openai.com/v1", model="text-embedding-3-small")` | `OpenAICompatibleOntologyExtractorAdapter(base_url="https://api.openai.com/v1", model="gpt-4o-mini")` |
| **Local Ollama** | `OllamaEmbeddingAdapter(model="nomic-embed-text", target_dim=1536)` | `OllamaOntologyExtractorAdapter(model="llama3.2")` |
| **Local SentenceTransformers** | `LocalSentenceTransformerEmbeddingAdapter(model_name="BAAI/bge-small-en-v1.5", target_dim=1536)` | *(Pair with any LLM extractor above)* |

---

## 6. LangGraph & LangChain Integration Recipes (`agentic_composite_memory.langgraph_integration`)

`agentic-composite-memory` ships with 4 first-class LangGraph and LangChain adapters that work with **either** in-process `CompositeMemoryService` or stateless `RemoteCompositeMemoryService` (zero hard dependency required; automatically subclasses `langgraph.store.base.BaseStore`, `langchain_core.tools.StructuredTool`, and `BaseChatMessageHistory` when installed):

### 6.1 LangGraph `BaseStore` Adapter (`LangGraphCompositeMemoryStore`)
Attach 3-Tier Ontology-Grounded Memory directly to a compiled LangGraph graph via `builder.compile(store=...)`:

```python
from agentic_composite_memory import LangGraphCompositeMemoryStore, RemoteCompositeMemoryService

memory_client = RemoteCompositeMemoryService(base_url="http://localhost:8095")
lg_store = LangGraphCompositeMemoryStore(
    memory_client,
    default_app_name="aurora_retail:order_fulfillment_agent",
    default_user_id="maya_lin",
)

# Inside any LangGraph node with `store: BaseStore`:
await lg_store.aput(
    ("aurora_retail:order_fulfillment_agent", "maya_lin", "memories"),
    "turn_1",
    {"text": "Order ORD-9001 re-routed to WH-Dallas-02."},
)

# 3-Tier Hybrid Recall (Playbooks + 2-Hop Bi-Temporal Graph + Episodic turns)
items = await lg_store.asearch(
    ("aurora_retail:order_fulfillment_agent", "maya_lin"),
    query="Where is ORD-9001 shipping from?",
)
for item in items:
    print(item.key, item.score, item.value["text"])
```

### 6.2 Pre-Built LangGraph `StateGraph` Nodes (`create_langgraph_memory_nodes`)
Wire automatic pre-LLM recall, pre-flight answerability checking, and post-LLM turn + tool-trace commit into a `StateGraph`:

```python
from agentic_composite_memory import create_langgraph_memory_nodes

nodes = create_langgraph_memory_nodes(
    memory_client,
    default_app_name="aurora_retail:returns_concierge_agent",
    default_user_id="maya_lin",
    inject_system_message=True,
)

# In your LangGraph StateGraph builder:
# builder.add_node("check_coverage", nodes.check_answerable_node)
# builder.add_node("recall_memory", nodes.recall_node)
# builder.add_node("agent", call_model_node)
# builder.add_node("commit_memory", nodes.commit_node)
```

### 6.3 LangGraph ReAct / `ToolNode` Tools (`get_langgraph_memory_tools`)
Bind all 5 memory tools (`recall_memory_context`, `check_memory_answerable`, `commit_memory_episode`, `register_memory_playbook`, `record_memory_decision_trace`) to `create_react_agent` or `ToolNode`:

```python
from agentic_composite_memory import get_langgraph_memory_tools

tools = get_langgraph_memory_tools(
    memory_client,
    default_app_name="aurora_retail:returns_concierge_agent",
    default_user_id="maya_lin",
)
# agent = create_react_agent(model, tools=tools)
```

### 6.4 LangChain `BaseChatMessageHistory` (`CompositeMemoryChatMessageHistory`)
Use with `RunnableWithMessageHistory`:

```python
from agentic_composite_memory import CompositeMemoryChatMessageHistory

history = CompositeMemoryChatMessageHistory(
    memory_client,
    app_name="aurora_retail:returns_concierge_agent",
    user_id="maya_lin",
    session_id="thread_101",
)
```

