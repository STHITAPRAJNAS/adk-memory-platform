# FastAPI REST, MCP Server & CLI Reference

Running `adk-memory serve-api` (or `create_memory_platform_app`) starts a **unified ASGI server** that co-hosts both the **FastAPI REST API** (`/v1/...`) and the **Model Context Protocol (MCP) Server** (`/mcp/...`, `/v1/mcp/...`) on a single port (`:8095` by default).

---

## 1. CLI Commands (`adk-memory`)

```bash
# 1. Start the unified FastAPI REST + MCP Server (default port 8095)
adk-memory serve-api --host 0.0.0.0 --port 8095 --mode in_memory

# 2. Start in production mode (reads ADK_* environment variables below)
adk-memory serve-api --host 0.0.0.0 --port 8095 --mode production

# 3. Start standalone MCP server over stdio (for Claude Desktop / Cursor / local MCP hosts)
adk-memory serve-mcp --mode in_memory

# 4. Register a Domain Pack (.json or .ttl) via CLI
adk-memory register-domain --tenant-id acme --file examples/domain_packs/sre_incident_ops.json
```

---

## 2. Environment Variables for Kubernetes / Container Deployments

Configured in [`agentic_memory_sdk/server/main.py`](../src/agentic_memory_sdk/server/main.py):

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

## 3. Co-Hosted FastAPI REST Endpoints (`/v1/...`)

| Method & Path | Request Body / Params | Description |
| :--- | :--- | :--- |
| `GET /healthz` | — | Kubernetes readiness/liveness probe; returns circuit-breaker state |
| `POST /v1/domains/register` | `{ "tenant_id": "...", "spec": { ... } }` | Registers a JSON `DomainPackSpec` |
| `POST /v1/domains/register-turtle` | `{ "tenant_id": "...", "domain_id": "...", "turtle_content": "..." }` | Registers a W3C OWL/SHACL `.ttl` Domain Pack |
| `POST /v1/apps/onboard` | `AppMemoryManifest` JSON | Onboards an agent app to Tier 0/1/2/3 governance |
| `POST /v1/memory/events` | `{ "app_name": "t:app", "user_id": "u1", "events": [...], "extraction_mode": "hot_path" \| "sleep_time" }` | Ingests conversation turns into Episodic + Bi-Temporal Graph |
| `POST /v1/memory/search` | `{ "app_name": "t:app", "user_id": "u1", "query": "...", "as_of_iso": null, "max_hops": 2 }` | Executes 3-Tier hybrid recall (Playbooks + 2-Hop Graph + Episodes) |
| `POST /v1/traces/record` | `{ "app_name": "t:app", "user_id": "u1", "run_id": "...", "steps": [...] }` | Records Tier-3 procedural reasoning traces |
| `POST /v1/playbooks/register` | `{ "app_name": "t:app", "user_id": "u1", "trigger_intent": "...", "workflow_steps": [...], "resolution_summary": "..." }` | Registers or reinforces a Tier-3 Procedural Playbook |
| `POST /v1/maintenance/run` | `{ "min_episodes_to_compact": 4, "min_overflow_confidence": 0.55 }` | Runs sleep-time batch extraction, session compaction, overflow pruning & dossier synthesis |
| `POST /v1/ontology/evolve` | `{ "app_name": "t:app", "min_confidence_to_promote": 0.75 }` | Promotes quarantined overflow classes/predicates into the governed TBox |
| `DELETE /v1/memory/users/{app_name}/{user_id}` | Path params | Cascading Right-to-be-Forgotten (GDPR) lineage purge |

---

## 4. Co-Hosted MCP Server Endpoints & Tools

When `mount_mcp=True` (the default in `create_memory_platform_app`), the same FastAPI server exposes:

1. **Streamable HTTP Transport**: `/mcp/http`
2. **SSE Transport**: `/mcp/sse`
3. **Stateless JSON-Over-HTTP MCP Bridge** (ideal for round-robin Kubernetes Ingress without sticky sessions):
   - `GET /v1/mcp/tools` — Lists all available MCP tools.
   - `POST /v1/mcp/call` — Invokes any MCP tool by name:
     ```json
     {
       "tool_name": "recall_context",
       "arguments": {
         "app_name": "acme:sre_commander",
         "user_id": "alice_vp",
         "query": "Where is payment-router-01 deployed?"
       }
     }
     ```

### Available MCP Tools
- `recall_context(app_name, user_id, query, as_of_iso=None)`
- `ingest_memory(app_name, user_id, text, author="user", session_id="mcp_session")`
- `record_trace_step(app_name, user_id, run_id, tool_name, tool_input, tool_output, touched_vertex_ids=None)`
- `register_domain_pack_json(tenant_id, domain_pack_json)`
