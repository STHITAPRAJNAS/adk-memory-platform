# FastAPI REST, Model Context Protocol (MCP) & CLI Reference

Running `adk-memory serve-api` (or mounting `create_memory_platform_app` in Uvicorn/Gunicorn) starts a **Unified Enterprise Memory-as-a-Service Pod** that co-hosts both:
1. **The FastAPI REST Control & Data Plane (`/v1/...`)** — consumed by `RemoteCompositeMemoryService`, microservices, CI/CD pipelines, and admin dashboards.
2. **The Model Context Protocol (MCP) Server (`/mcp/http`, `/mcp/sse`, `/v1/mcp/...`)** — compliant with the latest **`2026-07-28`** Stateless Request/Response Core (with `Mcp-Method` / `Mcp-Name` header routing and `structuredContent`) as well as **`2025-11-25`** Streamable HTTP, SSE, and `stdio`.

---

## 1. Complete Platform Parity Matrix (SDK vs. FastAPI vs. MCP)

Every Data-Plane, Control-Plane, Playbook, and Maintenance capability is accessible across all three surfaces:

| Platform Capability | In-Process Python SDK (`CompositeMemoryService` / `Registrar`) | FastAPI REST (`api.py`) & `RemoteCompositeMemoryService` | MCP Server Tool / Resource (`mcp_server.py`) |
| :--- | :--- | :--- | :--- |
| **Health & Governance Inspection** | `service.circuit_breaker`, `registry.resolve_context` | `GET /health`, `GET /healthz`, `GET /v1/apps/{tenant}/{app}` | Tool: `inspect_platform_governance`<br>Resource: `memory://governance/health` |
| **List / Read Domain Packs** | `registrar._domain_packs` | `GET /v1/domains/packs` | Resource: `memory://domains/packs` |
| **Register Domain Pack (JSON/Dict)** | `registrar.register_domain_pack(pack)` | `POST /v1/domains/packs` (`/v1/domains/register`) | Tool: `register_domain_pack` |
| **Steward Review & Refine Domain Pack** | `registrar.review_and_refine_domain_pack(pack)` | `POST /v1/domains/packs/review` | Tool: `register_domain_pack(..., review_with_steward=True)` |
| **Cold-Start Autonomous Domain Induction** | `service.induce_domain_pack(domain_name, transcripts=..., app_name=...)` | `POST /v1/domains/packs/induce` | Tool: `induce_domain_pack` |
| **Register W3C OWL/SHACL Turtle (`.ttl`)** | `DomainPackLoader.from_turtle_file(...)` | `POST /v1/domains/packs/turtle` | *(Via `register_domain_pack` or REST)* |
| **Register SQL DDL (`CREATE TABLE`)** | `DomainPackLoader.from_sql_ddl(pack_id, sql_ddl)` | `POST /v1/domains/packs/sql-ddl` | Tool: `register_domain_pack(..., sql_ddl=...)` |
| **Register Apache Ossie Semantic YAML** | `DomainPackLoader.from_ossie_yaml(pack_id, ossie_yaml)` | `POST /v1/domains/packs/ossie` | Tool: `register_domain_pack(..., ossie_yaml=...)` |
| **Certify Domain Pack (`certified=True`)** | `registrar.certify_domain_pack(pack_id)` | `POST /v1/domains/packs/certify` | Tool: `certify_domain_pack` |
| **Pre-Flight Answerability & Coverage** | `service.check_answerable(app_name, user_id, query)` | `POST /v1/memory/check-answerable` | Tool: `check_answerable` |
| **Onboard / Register Agent App** | `registrar.register_adk_app(manifest)` | `POST /v1/apps/register` (`/v1/apps/onboard`) | Tool: `register_app_manifest` |
| **Inspect App Compiled TBox** | `registry.resolve_context(tenant, app, user)` | `GET /v1/apps/{tenant_id}/{app_name}` | Tool: `inspect_platform_governance(app_name=...)`<br>Resource: `memory://apps/{tenant_id}/{app_name}/tbox` |
| **Ingest Turns (Hot-Path or Sleep-Time)** | `service.add_events_to_memory(..., extraction_mode=...)` | `POST /v1/memory/events` | Tool: `commit_episode` / `ingest_memory` |
| **3-Tier Hybrid Recall + Point-in-Time** | `service.search_memory(..., as_of_iso=..., max_hops=2)` | `POST /v1/memory/search` | Tool: `recall_context`<br>Prompt: `memory_grounded_assistant` |
| **Record Tier-3 Reasoning Trace** | `service.record_decision_trace(...)` | `POST /v1/traces` (`/v1/traces/record`) | Tool: `record_decision_trace` |
| **Register / Reinforce Tier-3 Playbook** | `service.register_playbook(...)` | `POST /v1/playbooks/register` | Tool: `register_playbook` |
| **Bottom-Up Ontology Auto-Evolution** | `service.evolve_app_ontology_from_overflow(...)` | `POST /v1/ontology/evolve` (`/v1/domains/evolve/{app}`) | Tool: `evolve_app_ontology` |
| **Sleep-Time Maintenance & Dossiers** | `service.run_memory_maintenance(...)` | `POST /v1/maintenance/run` | Tool: `run_memory_maintenance` |
| **Flush Transactional Outbox** | `service.process_outbox_batch(...)` | `POST /v1/outbox/flush` | *(Automatic background worker + `run_memory_maintenance`)* |
| **Ontology Audit Provenance Log** | `registrar.audit_log` | `GET /v1/audit/ontology` | Resource: `memory://audit/ontology` |
| **GDPR Right-to-be-Forgotten Purge** | `service.purge_user_lineage(...)` | `POST /v1/lineage/purge` | Tool: `purge_user_lineage` |

---

## 2. CLI Commands & Environment Variables

### CLI (`adk-memory`)

```bash
# 1. Start the unified FastAPI REST + MCP Server (default port 8095, zero-config in-memory mode)
adk-memory serve-api --host 0.0.0.0 --port 8095 --mode in_memory

# 2. Start in production mode (reads ADK_* environment variables below)
adk-memory serve-api --host 0.0.0.0 --port 8095 --mode production

# 3. Start standalone MCP server over stdio (for Claude Desktop / Cursor / Windsurf / local MCP hosts)
adk-memory serve-mcp --mode in_memory

# 4. Register a Domain Pack (.json or .ttl) via CLI
adk-memory register-domain --tenant-id acme --file examples/domain_packs/sre_incident_ops.json
```

### Environment Variables (`agentic_memory_sdk/server/main.py`)

| Variable | Values / Default | Description |
| :--- | :--- | :--- |
| `ADK_MEMORY_MODE` | `in_memory` (default), `postgres_stack`, `redis_stack`, `production` | Selects overall storage stack mode |
| `ADK_REGISTRY_BACKEND` | `postgres`, `redis`, `in_memory` | Per-port override for `RegistryStoreAdapter` |
| `ADK_SEED_BACKEND` | `postgres`, `opensearch`, `redis`, `in_memory` | Per-port override for `EpisodicAndSeedStoreAdapter` |
| `ADK_GRAPH_BACKEND` | `neo4j`, `neptune`, `postgres`, `redis`, `in_memory` | Per-port override for `GraphStoreAdapter` |
| `ADK_PLAYBOOK_BACKEND` | `redis`, `postgres`, `in_memory` | Per-port override for `PlaybookStoreAdapter` |
| `ADK_LLM_PROVIDER` | `gemini`, `bedrock`, `openai`, `ollama`, `mock` | Selects `OntologyExtractorAdapter` |
| `ADK_EMBEDDING_PROVIDER` | `gemini`, `bedrock`, `openai`, `ollama`, `local`, `hash` | Selects `EmbeddingAdapter` |
| `ADK_EMBEDDING_DIM` | `1536` (or `64` in `in_memory`) | Target vector dimension |
| `POSTGRES_DSN` | `postgresql://user:pass@host:5432/db` | PostgreSQL connection string |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` | Neo4j Bolt connection settings |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |

---

## 3. Detailed FastAPI REST Endpoint Reference (`/v1/...`)

### 3.1 Health & Governance Inspection

#### `GET /health` & `GET /healthz`
Returns platform liveness, circuit-breaker state, and all registered Layer-1 Domain Pack IDs.
```json
{
  "status": "ok",
  "circuit_breaker_state": "CLOSED",
  "registered_domain_packs": ["retail_omnichannel@1.0.0", "sre_infra@v1.2"]
}
```

#### `GET /v1/domains/packs`
Lists all registered `DomainPackSpec` schemas in the platform catalog.

#### `GET /v1/apps/{tenant_id}/{app_name}`
Inspects a specific ADK app's compiled `CompiledTBox`, governance tier, and active read/write namespaces.
- **`RemoteCompositeMemoryService` method**: `await client.inspect_app_governance(tenant_id="nordstrom_global", app_name="order_concierge")`
```json
{
  "tenant_id": "nordstrom_global",
  "app_name": "order_concierge",
  "read_namespaces": [
    "user:nordstrom_global: inspector",
    "app:nordstrom_global:order_concierge",
    "domain:nordstrom_global:retail_omnichannel",
    "global:nordstrom_global:core"
  ],
  "write_namespace": "domain:nordstrom_global:retail_omnichannel",
  "tbox": {
    "version": "1.0.0",
    "classes": { "RetailCustomer": "Customer profile...", "CustomerOrder": "Order..." },
    "relations": { "PLACED_ORDER": { "predicate": "PLACED_ORDER", "domain": "RetailCustomer", "range": "CustomerOrder" } }
  }
}
```

#### `GET /v1/audit/ontology`
Returns the immutable `OntologyAuditLogEntry` trail for all top-down Steward reviews and bottom-up overflow promotions.

---

### 3.2 Control Plane: Domain Pack & App Onboarding

#### `POST /v1/domains/packs` (alias `POST /v1/domains/register`)
Registers a Layer-1 Federated Domain Pack (`DomainPackSpec`).
- **`RemoteCompositeMemoryService` method**: `await client.register_domain_pack(pack_spec, review_with_steward=False)`
```bash
curl -X POST http://localhost:8095/v1/domains/packs \
  -H "Content-Type: application/json" \
  -d '{
    "pack_id": "retail_omnichannel@1.0.0",
    "domain_name": "retail_omnichannel",
    "version": "1.0.0",
    "classes": {
      "RetailCustomer": "Omnichannel shopper profile",
      "LoyaltyTier": "Membership tier"
    },
    "relations": {
      "ENROLLED_IN_TIER": {
        "predicate": "ENROLLED_IN_TIER",
        "domain": "RetailCustomer",
        "range": "LoyaltyTier",
        "is_functional": true,
        "description": "Current loyalty tier (functional: new tier invalidates prior tier)"
      }
    }
  }'
```

#### `POST /v1/domains/packs/review?auto_register=true`
Passes a proposed `DomainPackSpec` through the **Ontology Intelligence Steward** (roots unanchored classes under Layer 0 Upper Ontology, normalizes predicates to `UPPER_SNAKE_CASE`, infers functional/transitive flags, and logs an `OntologyAuditLogEntry`).
- **`RemoteCompositeMemoryService` method**: `await client.register_domain_pack(pack_spec, review_with_steward=True)`

#### `POST /v1/domains/packs/induce`
**Cold-Start Autonomous Ontology Induction**: synthesizes a complete Layer-1 `DomainPackSpec` (canonicalized `PascalCase` singular classes anchored under Layer 0 Upper Ontology, `UPPER_SNAKE_CASE` domain/range relations, and empirical + lexical `is_functional` / `is_transitive` rules) directly from sample conversation transcripts or stored app episodes.
- **`RemoteCompositeMemoryService` method**: `await client.induce_domain_pack(domain_name="supply_chain", transcripts=[...], min_support_count=2, auto_register=True)`
```json
{
  "domain_name": "supply_chain",
  "transcripts": [
    "Order ORD-501 shipped from FC-Seattle-01 and includes SKU-100 and SKU-200.",
    "Order ORD-502 shipped from FC-Reno-02 and includes SKU-300."
  ],
  "min_support_count": 1,
  "min_confidence": 0.70,
  "auto_register": true
}
```

#### `POST /v1/domains/packs/turtle`
Compiles and registers a W3C OWL/RDFS/SHACL Turtle (`.ttl`) file into a Layer-1 `DomainPackSpec`.
```json
{
  "pack_id": "retail_omnichannel@1.0.0",
  "turtle_content": "@prefix : <http://example.org/retail#> ...\n:RetailCustomer a owl:Class ; rdfs:subClassOf p:Actor ."
}
```

#### `POST /v1/apps/register` (alias `POST /v1/apps/onboard`)
Registers an agent application's `AppMemoryManifest`, compiling its `CompiledTBox` from Layer 0 + subscribed Layer 1 Domain Packs + Layer 2 app-local extensions.
- **`RemoteCompositeMemoryService` method**: `await client.register_app_manifest(manifest)`
```json
{
  "tenant_id": "nordstrom_global",
  "app_name": "order_concierge",
  "governance_tier": "DOMAIN_SHARED",
  "subscribed_domain_packs": ["retail_omnichannel@1.0.0"]
}
```

---

### 3.3 Data Plane: Ingest, 3-Tier Recall, Traces, Playbooks & Maintenance

#### `POST /v1/memory/events`
Ingests conversation turns into Tier-1 Episodic Memory (`PIIScrubber` + `SmartMemoryClassifier`) and triggers either immediate (`"hot_path"`) or deferred (`"sleep_time"`) ontology extraction.
- **`RemoteCompositeMemoryService` method**: `await client.add_events_to_memory(app_name, user_id, events, session_id=..., extraction_mode="hot_path")`
```json
{
  "app_name": "nordstrom_global:order_concierge",
  "user_id": "shopper_maya_lin",
  "session_id": "chat_session_101",
  "extraction_mode": "hot_path",
  "events": [
    {
      "author": "user",
      "text": "Always ship my orders to my Seattle locker and never substitute organic items. Order ORD-88410 is delayed."
    }
  ]
}
```

#### `POST /v1/memory/search`
Executes 3-Tier Hybrid Recall:
1. Matches reusable **Tier-3 Procedural Playbooks** (`[MATCHED_PROCEDURAL_PLAYBOOKS]`).
2. Runs **Tier-1 Hybrid RRF Seed Search** (`pgvector` cosine + BM25 full-text + pinned `user:{tenant}:{user_id}` preferences).
3. Expands **Tier-2 Bi-Temporal Knowledge Graph** up to `max_hops` (`t_valid` / `t_invalid`, or historical `as_of_iso` time travel).
- **`RemoteCompositeMemoryService` method**: `await client.search_memory(app_name, user_id, query, as_of_iso=None, max_hops=2)`
```json
{
  "app_name": "nordstrom_global:order_concierge",
  "user_id": "shopper_maya_lin",
  "query": "Where is ORD-88410 shipping from and what are Maya's preferences?",
  "as_of_iso": null,
  "max_hops": 2
}
```

#### `POST /v1/traces` (alias `POST /v1/traces/record`)
Records Tier-3 Procedural Reasoning Traces (`AgentRun -> ToolCall -> Entity`) for auditability and explainability.
- **`RemoteCompositeMemoryService` method**: `await client.record_decision_trace(app_name, user_id, run_id, session_id, steps)`
```json
{
  "app_name": "nordstrom_global:order_concierge",
  "user_id": "shopper_maya_lin",
  "run_id": "run_ord_88410_reroute",
  "session_id": "chat_session_101",
  "steps": [
    {
      "step_index": 1,
      "tool_name": "reroute_fulfillment_center",
      "input_summary": "order=ORD-88410, target_fc=FC-Reno-02",
      "output_summary": "Rerouted via FedEx Priority Overnight",
      "referenced_entities": ["ORD-88410", "FC-Reno-02"]
    }
  ]
}
```

#### `POST /v1/playbooks/register`
Registers or reinforces a reusable Tier-3 Procedural Playbook (`trigger_intent -> workflow_steps`).
- **`RemoteCompositeMemoryService` method**: `await client.register_playbook(app_name, user_id, trigger_intent, workflow_steps, resolution_summary)`
```json
{
  "app_name": "nordstrom_global:order_concierge",
  "user_id": "shopper_maya_lin",
  "trigger_intent": "Inventory stockout at primary fulfillment center for Diamond VIP order",
  "workflow_steps": [
    "1. Query secondary regional fulfillment center inventory.",
    "2. Rebind SHIPPED_FROM_PRIMARY functional edge to backup FC.",
    "3. Upgrade carrier to Priority Overnight at zero charge."
  ],
  "resolution_summary": "Zero-cost VIP split-shipment failover SOP",
  "share_in_domain": true
}
```

#### `POST /v1/ontology/evolve` (alias `POST /v1/domains/evolve/{app_name}`)
Scans quarantined overflow in `app:{tenant}:{app}`, promotes recurring high-confidence candidate classes and predicates into the governed TBox and domain ABox, and records provenance back to originating `source_episode_ids`.
- **`RemoteCompositeMemoryService` method**: `await client.evolve_app_ontology(app_name, min_confidence_to_promote=0.75)`

#### `POST /v1/maintenance/run`
Triggers Sleep-Time Memory Maintenance:
1. Drains deferred `SLEEP_TIME` episodes and runs batch ontology extraction.
2. Compacts older raw episodic turns into `[AUTO_SESSION_SUMMARY]` records while preserving `[PINNED_PREFERENCE]` records untouched.
3. Prunes low-confidence overflow graph edges below `min_overflow_confidence`.
4. Synthesizes Layer-3 `[Dossier: ...]` entity summaries from active graph assertions.
- **`RemoteCompositeMemoryService` method**: `await client.run_memory_maintenance(min_episodes_to_compact=4, min_overflow_confidence=0.55)`

#### `POST /v1/lineage/purge`
Executes a cascading GDPR / CCPA Right-to-be-Forgotten lineage deletion across both Episodic and Bi-Temporal Graph tiers.
- **`RemoteCompositeMemoryService` method**: `await client.purge_user_lineage(app_name, user_id)`

---

## 4. Model Context Protocol (MCP) Reference (`2026-07-28` & `2025-11-25`)

Our MCP server (`create_mcp_server` in [`mcp_server.py`](../src/agentic_memory_sdk/server/mcp_server.py)) is built on `mcp >= 2.3.0` and implements the latest **MCP `2026-07-28`** specification as well as **`2025-11-25`**, **`2025-06-18`**, and **`2025-03-26`**:

1. **Stateless Request/Response Core (`2026-07-28`) & Header Routing**:
   - Supports zero-handshake stateless JSON-RPC 2.0 requests with `MCP-Protocol-Version: 2026-07-28`, `Mcp-Method`, and `Mcp-Name` headers (allowing API gateways and Kubernetes Ingress load balancers to route and authorize MCP calls without sticky sessions).
   - Also supports standard `2025-11-25` Streamable HTTP (`/mcp/http/`), SSE (`/mcp/sse`), `stdio` (`adk-memory serve-mcp`), and our ultra-lightweight REST-to-MCP bridge (`POST /v1/mcp/call`).
2. **Rich `ToolAnnotations` & `structuredContent` (`outputSchema`)**:
   - Every tool declares explicit `ToolAnnotations` (`title`, `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) and `structured_output=True` so MCP hosts receive both human-readable `TextContent` and machine-typed `structuredContent`.
3. **Full 3-Primitive Coverage**:
   - **11 Annotated MCP Tools** (`tools/list`, `tools/call`)
   - **4 Live MCP Resources & URI Templates** (`resources/list`, `resources/templates/list`, `resources/read`)
   - **2 MCP Workflow Prompts** (`prompts/list`, `prompts/get`)

---

### 4.1 All 11 MCP Tools & Their `ToolAnnotations`

| Tool Name | Title | `readOnlyHint` | `destructiveHint` | `idempotentHint` | Key Arguments | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `recall_context` | 3-Tier Hybrid Memory Recall | `true` | `false` | `true` | `app_name`, `user_id`, `query`, `as_of_timestamp=None`, `max_hops=2` | Recalls Playbooks + RRF Seeds + 2-hop Bi-Temporal Graph + Episodes |
| `check_answerable` | Pre-Flight Ontology & Memory Coverage Check | `true` | `false` | `true` | `app_name`, `user_id`, `query`, `as_of_timestamp=None`, `max_hops=2` | Evaluates ontology/memory concept coverage (`ANSWERABLE`, `PARTIAL`, `UNANSWERABLE`) |
| `commit_episode` | Commit Interaction Episode | `false` | `false` | `false` | `app_name`, `user_id`, `raw_payload`, `author="agent"`, `session_id="mcp_session"`, `extraction_mode=None` | Scrubs PII, stores episode, and runs hot-path or sleep-time extraction |
| `ingest_memory` | Ingest Memory Turn | `false` | `false` | `false` | `app_name`, `user_id`, `text`, `author="user"`, `session_id="mcp_session"`, `extraction_mode=None` | Convenience alias for `commit_episode` using `text` |
| `record_decision_trace` | Record Procedural Reasoning Trace | `false` | `false` | `true` | `app_name`, `user_id`, `run_id`, `session_id`, `steps` | Writes `AgentRun -> ToolCall -> Entity` procedural graph trace |
| `register_playbook` | Register or Reinforce Procedural Playbook | `false` | `false` | `true` | `app_name`, `user_id`, `trigger_intent`, `workflow_steps`, `resolution_summary`, `share_in_domain=True` | Codifies or increments `success_count` on a Tier-3 SOP Playbook |
| `register_domain_pack` | Register Layer-1 Federated Domain Pack | `false` | `false` | `true` | `pack_id`, `classes=None`, `relations=None`, `sql_ddl=None`, `ossie_yaml=None`, `review_with_steward=True` | Registers a Layer-1 Domain Pack from JSON, SQL DDL, or Apache Ossie YAML |
| `induce_domain_pack` | Cold-Start Autonomous Domain Pack Induction | `false` | `false` | `true` | `domain_name`, `transcripts=None`, `app_name=None`, `min_support_count=1`, `min_confidence=0.70`, `auto_register=True` | Induces & registers a Layer-1 Domain Pack from raw transcripts or stored app episodes |
| `certify_domain_pack` | Certify Layer-1 Domain Pack | `false` | `false` | `true` | `pack_id`, `certified_by="domain_steward"` | Marks a Domain Pack as steward-certified (`certified=True`) in the audit log |
| `register_app_manifest` | Register Agent App Governance Manifest | `false` | `false` | `true` | `tenant_id`, `app_name`, `governance_tier="DOMAIN_SHARED"`, `subscribed_domain_packs=None` | Onboards an agent app and compiles its `CompiledTBox` |
| `evolve_app_ontology` | Evolve App Ontology from Overflow | `false` | `false` | `true` | `app_name`, `min_confidence_to_promote=0.75`, `min_support_count=1` | Canonicalizes & promotes overflow candidates meeting confidence and support thresholds |
| `run_memory_maintenance` | Run Sleep-Time Memory Maintenance | `false` | `false` | `true` | `min_episodes_to_compact=4`, `min_overflow_confidence=0.55` | Runs batch extraction, session compaction, pruning & dossier synthesis |
| `inspect_platform_governance` | Inspect Platform & App Governance | `true` | `false` | `true` | `app_name=None` | Returns circuit breaker state, Domain Packs, audit count & app TBox |
| `purge_user_lineage` | GDPR Right-to-be-Forgotten Lineage Purge | `false` | **`true`** | `true` | `app_name`, `user_id` | Cascading deletion across Episodic & Graph tiers (`destructiveHint=true` triggers host confirmation) |

---

### 4.2 All 4 MCP Resources (`resources/list`, `resources/templates/list`, `resources/read`)

| Resource URI | Name | MIME Type | Description |
| :--- | :--- | :--- | :--- |
| `memory://governance/health` | `platform_governance_health` | `application/json` | Live health status, circuit-breaker state, registered Domain Packs, and audit log count |
| `memory://domains/packs` | `domain_packs_catalog` | `application/json` | Complete catalog of all registered Layer-1 Federated Domain Packs (classes & relations) |
| `memory://audit/ontology` | `ontology_audit_log` | `application/json` | Immutable provenance trail of all top-down Steward reviews and bottom-up overflow promotions |
| `memory://apps/{tenant_id}/{app_name}/tbox` | `app_compiled_tbox` | `application/json` | **Dynamic Resource Template**: inspects any tenant/app's compiled `CompiledTBox` and active read/write namespaces |

---

### 4.3 All 2 MCP Prompts (`prompts/list`, `prompts/get`)

| Prompt Name | Arguments | Description |
| :--- | :--- | :--- |
| `memory_grounded_assistant` | `app_name`, `user_id`, `query` | Executes 3-Tier hybrid recall (`recall_context`) and returns a ready-to-use system/user prompt pre-populated with `[MATCHED_PROCEDURAL_PLAYBOOKS]` and `[COMPOSITE_MEMORY_CONTEXT]` |
| `ontology_steward_domain_designer` | `domain_name`, `sample_transcript` | Guides an LLM Ontology Steward to synthesize a Layer-1 `DomainPackSpec` anchored under Layer 0 Upper Ontology classes and invoke `register_domain_pack` |

---

### 4.4 Protocol Examples (`2026-07-28` vs. `2025-11-25` vs. `/v1/mcp/call`)

#### Example A: Latest MCP `2026-07-28` Stateless Core over Streamable HTTP (`/mcp/http/`)
```bash
curl -X POST http://localhost:8095/mcp/http/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "MCP-Protocol-Version: 2026-07-28" \
  -H "Mcp-Method: tools/call" \
  -H "Mcp-Name: inspect_platform_governance" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "inspect_platform_governance",
      "arguments": { "app_name": "nordstrom_global:order_concierge" },
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  }'
```

#### Example B: MCP `2025-11-25` Streamable HTTP (`tools/list`, `resources/read`, `prompts/get`)
```bash
# Read live governance health resource via MCP JSON-RPC 2.0
curl -X POST http://localhost:8095/mcp/http/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "MCP-Protocol-Version: 2025-11-25" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "resources/read",
    "params": { "uri": "memory://governance/health" }
  }'
```

#### Example C: Lightweight Stateless REST-to-MCP RPC (`POST /v1/mcp/call`)
Ideal for microservices, webhook workers, or non-MCP HTTP clients that want to invoke any of the 11 MCP tools with plain JSON:
```bash
curl -X POST http://localhost:8095/v1/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "recall_context",
    "arguments": {
      "app_name": "nordstrom_global:order_concierge",
      "user_id": "shopper_maya_lin",
      "query": "Where is ORD-88410 shipping from?"
    }
  }'
```

---

## 5. Ready-to-Copy Client Adoption Configs

### 5.1 Claude Desktop / Cursor / Windsurf / Antigravity (`mcpServers` Config)

**Option 1: Local `stdio` Mode (Zero Infrastructure)**
```json
{
  "mcpServers": {
    "agentic-composite-memory": {
      "command": "uvx",
      "args": [
        "--from",
        "agentic-composite-memory",
        "adk-memory",
        "serve-mcp",
        "--mode",
        "in_memory"
      ]
    }
  }
}
```

**Option 2: Remote Kubernetes / Docker Streamable HTTP Mode**
```json
{
  "mcpServers": {
    "agentic-composite-memory-remote": {
      "type": "streamable-http",
      "url": "http://localhost:8095/mcp/http/"
    }
  }
}
```

### 5.2 Google ADK Agent (`McpToolset` or `get_adk_memory_tools`)

```python
from google.adk.agents import LlmAgent
from agentic_composite_memory import RemoteCompositeMemoryService, get_adk_memory_tools

remote_memory = RemoteCompositeMemoryService(base_url="http://localhost:8095")

agent = LlmAgent(
    name="retail_concierge",
    model="gemini-2.5-flash",
    tools=get_adk_memory_tools(
        remote_memory,
        default_app_name="nordstrom_global:order_concierge",
    ),
)
```
