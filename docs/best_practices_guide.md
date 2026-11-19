# Enterprise Best Practices Guide — `agentic-composite-memory`

This guide distills architectural patterns, operational guardrails, and tuning recommendations for running **Agentic Composite Memory** (`agentic-composite-memory`) in production across multi-agent ecosystems.

---

## 1. Choosing the Right Agent Consumption Pattern

`agentic-composite-memory` provides **100% Data-Plane + Control-Plane parity** across three consumption surfaces. Choose the pattern that matches your deployment topology:

| Consumption Pattern | Best For | Why / Key Advantage |
| :--- | :--- | :--- |
| **1. Direct In-Process SDK** (`CompositeMemoryService`) | Single-service Python/ADK apps, batch ingestion workers, and latency-critical monoliths | Zero extra HTTP hop between the agent process and your database adapters (`Postgres`, `Neo4j`, `Redis`). |
| **2. Co-Hosted FastAPI + `RemoteCompositeMemoryService`** | Horizontally scaled Kubernetes agent pods (`10–500` replicas) | Agent pods stay stateless and hold **zero DB credentials or heavy DB connection pools**; a central Memory Platform deployment manages DB pools, IAM token rotation, and the transactional outbox worker. |
| **3. Model Context Protocol (`MCP 2026-07-28` / `2025-11-25`)** | Polyglot agent fleets (Claude Desktop, Cursor, Windsurf, Google ADK `McpToolset`, LangChain, TypeScript/Go agents) | Zero SDK coupling; agents discover all **11 annotated tools**, **4 live `memory://` resources**, and **2 prompts** dynamically over Streamable HTTP (`/mcp/http/`) or `stdio`. |

> **Best Practice — Hybrid Deployment**: Run `adk-memory serve-api --mode production` as a central Kubernetes Service. Python/ADK agents can attach `RemoteCompositeMemoryService` to `Runner(memory_service=...)`, while external MCP agents connect to `http://memory-platform:8095/mcp/http/` on the **exact same pod**.

---

## 2. Selecting & Tuning Your Storage Topology

Because every storage port (`RegistryStoreAdapter`, `EpisodicAndSeedStoreAdapter`, `GraphStoreAdapter`, `PlaybookStoreAdapter`) is decoupled, you can start with a single database and scale out individual tiers as graph complexity grows:

| Topology | Factory / Adapters | Recommended Workload |
| :--- | :--- | :--- |
| **Single-Stack PostgreSQL** | `CompositeMemoryService.from_postgres_stack(...)` | Teams standardizing on **Cloud SQL / RDS / Aurora PostgreSQL (`pgvector`)**. Handles RRF hybrid search, recursive CTE 2-hop graph traversal, playbooks, and RLS tenant isolation in one ACID database. |
| **Single-Stack Redis** | `CompositeMemoryService.from_redis_stack(...)` | Ultra-low-latency (`<5ms`) real-time voice or trading agents using **Redis Stack / ElastiCache / MemoryDB**. |
| **Polyglot Enterprise Stack** | `PostgresPgVectorAdapter` + `Neo4jGraphAdapter` (or `NeptuneOpenCypherAdapter`) + `RedisPlaybookAndWorkingAdapter` | Large multi-hop enterprise knowledge graphs (`10M+` edges) requiring native Cypher traversal, `pgvector` HNSW + BM25 RRF episodic search, and `<5ms` Redis playbook caching. |

### Database Indexing & Connection Pool Checklist
1. **PostgreSQL (`pgvector` + GIN + RLS)**:
   - Apply [`docs/sql/001_postgres_rls_and_partitions.sql`](./sql/001_postgres_rls_and_partitions.sql) so `episodic_memories` has both an **HNSW cosine index** (`m = 16, ef_construction = 64`) and a **BM25 GIN index** (`to_tsvector('english', raw_text)`).
   - Keep `ADK_EMBEDDING_DIM` consistent across your embedding model and DDL (e.g., `1536` for `text-embedding-3-small` / Titan v2, or `768` for `text-embedding-004`).
2. **Neo4j / Amazon Neptune**:
   - `Neo4jGraphAdapter.initialize()` automatically creates the composite index `CREATE INDEX entity_tenant_vid IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.vertex_id)`. Ensure this index is online before running bulk backfills so batched `UNWIND $edges` merges execute in `<15ms`.

---

## 3. Ontology Design & Governance Best Practices (Layers 0, 1, and 2)

### 3.1 Designing Layer-1 Federated Domain Packs
- **Always Anchor Classes to Layer 0**: Every domain class should conceptually specialize one of the built-in Layer 0 Upper Ontology roots (`Agent`, `PhysicalObject`, `ConceptualEntity`, `Event`, `TemporalEntity`, `Location`).
- **Run Top-Down Steward Review Before Registering**:
  - Pass new Domain Packs through `registrar.review_and_refine_domain_pack(spec)` (or `POST /v1/domains/packs/review` / MCP `register_domain_pack(..., review_with_steward=True)`).
  - The Steward automatically normalizes predicates to `UPPER_SNAKE_CASE`, anchors unrooted classes, infers `is_functional` / `is_transitive` flags, and writes an immutable `OntologyAuditLogEntry`.

### 3.2 When to Mark Relations `is_functional=True` vs. `False`
This is the most important modeling decision for **Bi-Temporal Conflict Resolution**:
- **Set `is_functional=True` for single-valued state relationships** where a new value supersedes the old value over time:
  - *Examples*: `DEPLOYED_IN` (a microservice moves from `us-east-1` to `us-west-2`), `ENROLLED_IN_TIER` (a customer upgrades from `Gold` to `Diamond_VIP`), `SHIPPED_FROM_PRIMARY` (an order is rerouted from `FC-Seattle-01` to `FC-Reno-02`), `HAS_STATUS`.
  - *Runtime behavior*: When a new fact `(ORD-88410, SHIPPED_FROM_PRIMARY, FC-Reno-02)` arrives, the graph adapter automatically sets `t_invalid = now` on the prior `(ORD-88410, SHIPPED_FROM_PRIMARY, FC-Seattle-01)` edge while preserving it for historical point-in-time queries (`as_of_iso`).
- **Set `is_functional=False` for multi-valued relationships** where an entity can legitimately hold multiple simultaneous targets:
  - *Examples*: `INCLUDES_SKU` (an order contains multiple SKUs), `DEPENDS_ON` (a service depends on multiple downstream APIs), `PURCHASED_BY`.

### 3.3 Choosing the Right `GovernanceTier` per Agent App
```text
+-------------------------------------------------------------------------+
| TIER 1: APP_PRIVATE     -> Writes to `app:{tenant}:{app}` only          |
|                            (Best for experimental or sandbox agents)    |
+-------------------------------------------------------------------------+
| TIER 2: DOMAIN_SHARED   -> Reads/Writes `domain:{tenant}:{domain}`      |
|                            (Recommended default for production squads)  |
+-------------------------------------------------------------------------+
| TIER 3: GLOBAL_CORE     -> Promotes canonical cross-domain entities     |
|                            into `global:{tenant}:core`                  |
+-------------------------------------------------------------------------+
```
- **Recommendation**: Onboard most production agents at **`GovernanceTier.TIER2_DOMAIN_SHARED`** with explicit `subscribed_domain_packs` (e.g., `["retail_omnichannel@1.0.0"]`). Validated TBox facts are shared across collaborating agents in that domain, while novel/unvalidated LLM extractions are safely quarantined in `app:{tenant}:{app}` (`Bucket B`).

### 3.4 Bottom-Up Ontology Evolution (`evolve_app_ontology_from_overflow`) & Cold-Start Induction (`induce_domain_pack`)
- **Preventing One-Off Noise with `min_support_count`**:
  - Do **not** let a single one-off LLM hallucination permanently alter your shared domain schema.
  - Let `TBoxRuntimeValidator` quarantine unknown classes/predicates in `app:{tenant}:{app}` (`[ProposedClass=...]`, `[ProposedPredicate=...]`).
  - Periodically invoke `evolve_app_ontology` with `min_support_count=2` (or `3`) and `min_confidence_to_promote=0.80`. The engine automatically canonicalizes plural/casing variants (e.g., `ColdChainShipments` and `cold_chain_shipment` merge into `ColdChainShipment`), pools their support counts, checks empirical `(source_id, target_id)` graph cardinality (`1:1` vs. `1:N`) to infer `is_functional`, and promotes only proven concepts into the governed TBox.
- **Cold-Start Bootstrapping (`induce_domain_pack`)**:
  - When onboarding a brand-new business vertical with **zero** existing ontology, pass 10–100 historical support tickets, incident logs, or chat transcripts to `await service.induce_domain_pack(domain_name="...", transcripts=[...], min_support_count=2)` (or `POST /v1/domains/packs/induce` / MCP `induce_domain_pack`).
  - The platform extracts candidate entities and relationships, canonicalizes and anchors every class under Layer 0 (`p:Actor`, `p:Object`, `p:Event`, `p:Concept`, `p:UserPreference`) using head-noun-aware semantic alignment, infers domain/range constraints and `is_functional` / `is_transitive` rules from empirical cardinality, and registers the Steward-verified `DomainPackSpec`.

---

## 4. Hot-Path vs. Sleep-Time Extraction & Maintenance Cadence

### 4.1 Balancing Turn Latency vs. Immediate Graph Freshness
| Mode | Turn Write Latency | When to Use |
| :--- | :---: | :--- |
| **`ExtractionMode.SLEEP_TIME`** (`defer_extraction=True`) | **`~5–12 ms`** | High-throughput user-facing chat turns, voice agents, or streaming conversations where sub-15ms response time is critical and graph extraction can happen asynchronously in the background. |
| **`ExtractionMode.HOT_PATH`** (`defer_extraction=False`) | **`~150–800 ms`** *(LLM dependent)* | Critical state-mutation turns (e.g., incident mitigation commands, order rerouting, tier upgrades) where the very next turn or a sibling A2A agent immediately needs the updated bi-temporal graph edge. |

> **Tip**: Even if your service defaults to `ExtractionMode.HOT_PATH`, any caller can override `extraction_mode="sleep_time"` (or `"hot_path"`) **per turn** in `add_events_to_memory()`, `POST /v1/memory/events`, or the MCP `commit_episode` tool.

### 4.2 Running `run_memory_maintenance()` in Production
Schedule `run_memory_maintenance(min_episodes_to_compact=4, min_overflow_confidence=0.55)` via a Kubernetes `CronJob` ( calling `POST /v1/maintenance/run` or MCP `run_memory_maintenance`) every **5–15 minutes**:
1. **Drains `SLEEP_TIME` episodes** and executes batched ontology extraction.
2. **Compacts raw session turns** into `[AUTO_SESSION_SUMMARY]` episodes (reducing vector index bloat while **never** compacting `[PINNED_PREFERENCE]` records).
3. **Prunes noisy overflow edges** with confidence below `min_overflow_confidence`.
4. **Synthesizes Layer-3 `[Dossier: ...]` summaries** on active entities so 1-hop recall immediately sees a consolidated profile of each entity.

---

## 5. Entity Resolution & Naming Conventions

`EntityResolutionGuard` protects your knowledge graph from the classic vector-similarity trap where numbered, regional, or versioned siblings (e.g., `payment-router-01` vs. `payment-router-02`, `us-east-1` vs. `us-east-2`, `ORD-88410` vs. `ORD-88411`) have `>0.97` embedding cosine similarity.

- **Best Practice for Canonical Names**:
  - Preserve alphanumeric identifiers, region codes, and version suffixes in `canonical_name` (e.g., `FC-Seattle-01`, `SKU-9901-ORG`, `aurora-prod-02`).
  - `EntityResolutionGuard.can_merge()` will automatically block false merges when numeric/regional/roman tokens differ, while still merging true surface variations (e.g., `"Checkout API"` and `"checkout-api"`).

---

## 6. Tier-3 Procedural Playbooks & Reasoning Traces

1. **Write Intent-Focused `trigger_intent` Strings**:
   - When calling `register_playbook`, phrase `trigger_intent` the way an incoming incident or user query will describe the problem (e.g., *"Inventory stockout at primary fulfillment center for Diamond VIP order"* rather than *"Playbook #42"*).
   - Why: `search_memory()` embeds the incoming `query` and matches it against stored playbook embeddings (`score >= 0.35`), injecting matching SOPs at the very top of the recalled context as `[MATCHED_PROCEDURAL_PLAYBOOKS]`.
2. **Reinforce Successful Playbooks**:
   - Calling `register_playbook` with the same `trigger_intent` is **idempotent and reinforcing**: it increments `success_count`, boosting that playbook's ranking during recall.
3. **Always Pass `referenced_entities` in `record_decision_trace`**:
   - Linking `ReasoningTraceStep.referenced_entities` (e.g., `["ORD-88410", "FC-Reno-02"]`) connects the procedural `AgentRun -> ToolCall` chain directly to the Tier-2 semantic `:Entity` nodes in Neo4j/Postgres.

---

## 7. Multi-Tenant Security, Cloud IAM & GDPR Compliance

1. **Always Prefix `app_name` with `tenant_id:` (`"{tenant_id}:{app_name}"`)**:
   - Every SDK method, REST endpoint, and MCP tool accepts `app_name="acme_corp:sre_commander"`. This guarantees strict partition key and namespace isolation (`user:acme_corp:...`, `app:acme_corp:...`, `domain:acme_corp:...`) and sets `app.current_tenant_id` for PostgreSQL Row-Level Security (RLS).
2. **Use Zero-Static-Secret Cloud IAM Auth in Production**:
   - Use `AwsIamDatabaseAuthProvider` (for RDS/Aurora/Neptune/ElastiCache with automatic <15-minute token caching + optional STS `AssumeRole`), `GcpCloudSqlIamAuthProvider`, `AzureEntraDatabaseAuthProvider`, or `VaultOrSecretManagerAuthProvider` combined with `EnterpriseTlsConfig(verify_mode="CERT_REQUIRED")`.
3. **PII Scrubbing & GDPR Right-to-be-Forgotten**:
   - Keep `PIIScrubber` enabled (default) so SSNs, credit cards, API keys, emails, and phone numbers are redacted (`[REDACTED_SSN]`, `[REDACTED_CARD]`, etc.) **before** any text reaches an embedding API, LLM extractor, or database disk.
   - Bind your user-deletion webhook to `POST /v1/lineage/purge` (or `service.purge_user_lineage(app_name, user_id)` / MCP `purge_user_lineage`). It cascades across both Tier-1 episodic records and Tier-2 graph edges via `source_episode_ids` lineage pointers.

---

## 8. Production Go-Live Checklist

- [ ] **Domain Packs Reviewed**: All Layer-1 Domain Packs registered via `review_and_refine_domain_pack` with `is_functional` flags verified on single-valued predicates.
- [ ] **Apps Onboarded**: Every agent registered via `register_adk_app` (`AppMemoryManifest`) with explicit `GovernanceTier` and `subscribed_domain_packs`.
- [ ] **Database Indexes Online**: PostgreSQL `pgvector` HNSW + BM25 GIN indexes and Neo4j `:Entity(tenant_id, vertex_id)` composite index created.
- [ ] **Outbox Worker Enabled**: `enable_outbox_worker=True` on `create_memory_platform_app` (or `await service.start_outbox_worker()`) so transient graph spikes drain asynchronously.
- [ ] **Maintenance Job Scheduled**: `run_memory_maintenance()` scheduled every 5–15 minutes to drain `SLEEP_TIME` episodes, compact raw sessions, prune low-confidence overflow, and synthesize `[Dossier: ...]` entity summaries.
- [ ] **Health & Governance Monitored**: Kubernetes liveness/readiness probes pointed at `GET /healthz` (monitoring `circuit_breaker_state == "CLOSED"`) or MCP `memory://governance/health`.
