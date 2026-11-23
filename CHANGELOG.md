# Changelog

All notable changes to `agentic-composite-memory` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-11-23

### Security (Critical & High Fixes)
- **Wired Auth into Packaged Server & Guarded Public Bind**: `server/main.py` now loads `AuthPolicyConfig.from_env()` (`MEMORY_AUTH_ENABLED`, `MEMORY_JWT_SECRET`, `MEMORY_ADMIN_API_KEY`, `MEMORY_AGENT_API_KEY`, `MEMORY_API_KEYS_JSON`), and `adk-memory serve-api` adds `--auth-enabled`, `--jwt-secret`, `--admin-api-key`, and refuses to bind to `0.0.0.0` without authentication unless `--allow-insecure-public-bind` is explicitly passed.
- **Closed Query-vs-Body Tenant Smuggling & Path/MCP Scope Bypasses**: `auth_and_rbac_middleware` now checks query params, path params (`/v1/apps/{tenant_id}/{app_name}`, `/v1/domains/evolve/{app_name}`), request body fields, and nested MCP tool `arguments` (`/v1/mcp/call` and `/mcp/http/` `tools/call`) independently, rejecting any conflicting query-vs-body parameters (`403`) and enforcing per-tool RBAC via `AuthPolicyConfig.required_role_for_mcp_tool()`.
- **Fixed Route-to-RBAC Mapping**: Aligned `required_role_for_route()` with actual FastAPI routes (`/v1/lineage/purge`, `/v1/traces`, `/v1/playbooks/register`, `/v1/apps/onboard`, `/v1/outbox/flush`) with a safe default requiring at least `AGENT` for all mutating endpoints.
- **Strict JWT Validation**: `verify_hs256_jwt` now requires the `exp` claim, and `AuthPolicyConfig.authenticate` requires a non-empty `tenant_id` claim instead of defaulting to `default_tenant`.
- **Complete GDPR Right-to-be-Forgotten Purge**: `purge_user_lineage()` now also erases the user's `working_memory_blocks` and `memory_revisions` (both in-memory and PostgreSQL).
- **Enforced PostgreSQL Row-Level Security by Default**: `server/main.py` now defaults to the non-superuser `platform_app_user` role (`NOBYPASSRLS`) instead of `platform_admin`, and `docs/sql/001_postgres_rls_and_partitions.sql` provisions `platform_app_user` login credentials plus RLS-protected durable `memory_revisions` and `working_memory_blocks` tables.
- **CI-Gated Releases**: `.github/workflows/release.yml` now runs a mandatory `verify` gate (`ruff check` + `pytest`) before `build` and `publish-pypi`.

### Fixed & Cleanup
- Relabeled `examples/locomo_longmemeval_harness.py` honestly as a lightweight 6-category synthetic evaluation harness (LoCoMo/LongMemEval-inspired categories, not the official external datasets).
- Added `langgraph`, `langchain-core`, and `sentence-transformers` to `[project.optional-dependencies].all` in `pyproject.toml`; FastAPI app version is now resolved dynamically from package metadata.
- Updated `README.md` test counts (`20` tests) and authenticated `serve-api` examples.

## [0.2.1] - 2026-11-23

### Added
- **Configurable, Multi-Replica-Safe Built-In Sleep-Time Maintenance Worker**:
  - Added `CompositeMemoryService.start_maintenance_worker(interval_sec=300.0, ...)` and `stop_maintenance_worker()` with in-process `asyncio.Lock` plus non-blocking PostgreSQL session advisory lock (`pg_try_advisory_lock`) so even multi-replica deployments never race on the same Sleep-Time extraction/compaction/TTL batch.
  - Wired into FastAPI `create_memory_platform_app(..., enable_maintenance_worker=False, maintenance_interval_sec=300.0)` and `/health` (`maintenance_worker_running`, `maintenance_last_stats`).
  - Added CLI flags (`adk-memory serve-api --enable-maintenance-worker --maintenance-interval-sec 300`) and environment variables (`ENABLE_MAINTENANCE_WORKER`, `MAINTENANCE_INTERVAL_SEC`, `MAINTENANCE_MIN_EPISODES_TO_COMPACT`, `MAINTENANCE_MIN_OVERFLOW_CONFIDENCE`).

## [0.2.0] - 2026-11-23

### Added
- **Free-Text Cognitive Reconciliation & Immutable Memory Revisions (`MemoryReconciliationEngine`, `MemoryRevisionRecord`)**:
  - Automatically reconciles non-functional cognitive memories (`p:UserPreference`, `p:ProspectiveGoal`, `p:WorkingHypothesis`, `p:EpisodicLesson`) via `ADD`, `UPDATE`, `RESOLVE`, `DELETE`, and `NOOP` decisions (Mem0 & Vertex AI Memory Bank parity), logging immutable `MemoryRevisionRecord` entries.
  - Added manual inspection, inline editing (with re-embedding + `MANUAL_EDIT` revision), and single-item deletion (`list_user_memories`, `edit_user_memory`, `delete_user_memory`, `get_memory_revisions`).
- **Configurable Per-App Retention / TTL Policy & Few-Shot Custom Topics (`MemoryRetentionPolicy`, `FewShotExtractionExample`)**:
  - Configurable `default_episode_ttl_days`, `hypothesis_ttl_days`, `completed_goal_ttl_days`, and Ebbinghaus recency decay (`compute_recency_decay`), enforced via an automatic TTL expiration sweep in `run_memory_maintenance()` (both InMemory and PostgreSQL).
  - Per-app `custom_extraction_topics` and `few_shot_examples` on `AppMemoryManifest` automatically injected into `build_tbox_system_instruction()` across all LLM extractors (Vertex AI Memory Bank parity).
- **Letta / MemGPT-Style Self-Editable Core Working Memory Blocks (`WorkingMemoryBlock`)**:
  - Added `upsert_working_memory_block` (`replace` / `append`, PII-scrubbed, character-bounded) and `get_working_memory_blocks`, automatically prepended to `search_memory` context as `[WORKING_MEMORY_BLOCKS]`.
  - Exposed across Python SDK, FastAPI REST (`/v1/memory/blocks`), `RemoteCompositeMemoryService`, and MCP Server (expanding MCP to **17 annotated tools**, **4 resources**, and **2 prompts**).
- **Enterprise API AuthN/AuthZ (HS256 JWT + API Key + Pluggable OIDC + RBAC + Tenant Isolation) & Observability (`security.py`)**:
  - Added `AuthPolicyConfig`, `AuthPrincipal`, `MemoryRole` (`READER`, `AGENT`, `ONTOLOGY_STEWARD`, `COMPLIANCE_ADMIN`, `PLATFORM_ADMIN`), `mint_hs256_jwt`, `verify_hs256_jwt`, and `MemoryObservability` (OpenTelemetry auto-detection + `GET /v1/metrics` JSON + `GET /metrics` Prometheus text format).
- **Bundled Visual Memory & Ontology Governance Console UI (`GET /ui`)**:
  - Self-contained interactive SPA served directly by FastAPI (`server/ui_console.py`) for inspecting/editing/deleting user memories, managing Working Memory Blocks, auditing Memory Revisions, certifying/evolving Domain Packs, and monitoring live metrics.
- **LoCoMo / LongMemEval-Style Standardized Evaluation Harness (`examples/locomo_longmemeval_harness.py`)**:
  - Runnable 6-category benchmark harness supporting `--mode deterministic`, `--mode gemini`, `--mode openai`, and `--mode ollama`.

## [0.1.7] - 2026-11-21

### Added
- **12-Class Layer-0 Cognitive Upper Ontology (`UpperConceptType` & Universal Relations)**:
  - Expanded Layer-0 from 6 to **12 cognitive memory primitives** and **22 universal relations** inspired by cognitive architectures (CoALA, ACT-R, Soar, Mem0, Letta/MemGPT) and enterprise upper ontologies (BFO, DOLCE, PROV-O):
    - `p:PolicyGuardrail` (`CONSTRAINED_BY`, `VIOLATES`) — Hard negative constraints & compliance rules (`pinned_cognitive` compaction immunity + `1.5x` salience boost).
    - `p:WorkingHypothesis` (`HYPOTHESIZES`, `SUPPORTED_BY`, `REFUTED_BY`) — Speculative beliefs with **Epistemic Collapse Protection** (never invalidate `CONFIRMED` functional edges in Neo4j/InMemory; rendered with `[UNCONFIRMED_HYPOTHESIS]` and `0.6x` salience penalty).
    - `p:EpisodicLesson` (`LEARNED_FROM`, `MITIGATED_BY`) — Post-mortem reflections & failure-avoidance lessons (`pinned_cognitive` compaction immunity + `1.25x` salience boost).
    - `p:ProspectiveGoal` (`HAS_GOAL`, `BLOCKED_BY`, `DEPENDS_ON_GOAL`) — Future intentions & reminders (`1.3x` salience boost).
    - `p:QuantitativeMetric` (`HAS_METRIC`, `BREACHES_THRESHOLD`) — Typed KPIs, SLAs, and thresholds for structured bi-temporal tracking.
    - `p:SpatialLocation` (`LOCATED_AT`) — Physical & logical places (warehouses, cloud regions, clinics, stores).
- **Live Docker 9-Scenario Enterprise Agent Lift Benchmark (`examples/enterprise_agent_lift_benchmark.py`)**:
  - Added **Scenario 9 (`9_layer0_cognitive_memory_contracts`)** verifying `p:PolicyGuardrail` and `p:EpisodicLesson` compaction immunity directly in live **PostgreSQL 16 + `pgvector`** and `p:WorkingHypothesis` Epistemic Collapse Protection directly in live **Neo4j 5** alongside **Redis 7**, **FastAPI**, **MCP `2026-07-28`**, **Google ADK**, and **LangGraph**.
  - Updated [`docs/enterprise_agent_lift_report.json`](docs/enterprise_agent_lift_report.json), [`docs/architecture_and_ontology_guide.md`](docs/architecture_and_ontology_guide.md), [`docs/domain_pack_authoring_guide.md`](docs/domain_pack_authoring_guide.md), and [`README.md`](README.md).

## [0.1.6] - 2026-11-21

### Added
- **LangGraph & LangChain First-Class Memory Adapters (`agentic_composite_memory.langgraph_integration`)**:
  - `LangGraphCompositeMemoryStore`: Drop-in `langgraph.store.base.BaseStore` implementation (`aput`, `aget`, `asearch`, `adelete`, `alist_namespaces`, `abatch`, `batch`) backed by either in-process `CompositeMemoryService` or stateless `RemoteCompositeMemoryService`.
  - `create_langgraph_memory_nodes`: Pre-built `StateGraph` nodes (`recall_node`, `check_answerable_node`, `commit_node` with automatic extraction of `AIMessage.tool_calls` into Tier-3 Procedural Reasoning Traces).
  - `get_langgraph_memory_tools`: 5 ready-to-bind LangGraph ReAct / `ToolNode` tools (`recall_memory_context`, `check_memory_answerable`, `commit_memory_episode`, `register_memory_playbook`, `record_memory_decision_trace`).
  - `CompositeMemoryChatMessageHistory`: LangChain `BaseChatMessageHistory` implementation for `RunnableWithMessageHistory`.
  - Added optional `[langgraph]` extra (`pip install "agentic-composite-memory[langgraph]"`).
- **Domain Pack Authoring & Ontology Engineering Guide (`docs/domain_pack_authoring_guide.md`)**:
  - Comprehensive guide covering all 6 ways to create a Layer-1 Domain Pack (JSON/YAML, Python Pydantic, W3C OWL 2 / SHACL `.ttl`, Relational SQL DDL, Apache Ossie YAML, and Cold-Start Autonomous Transcript Induction) with complete blueprints for **Omnichannel Retail & Supply Chain** and **Payment Transaction Processing, Clearing & Dispute Resolution**.

## [0.1.5] - 2026-11-20

### Added
- **Pre-Flight Ontology & 3-Tier Memory Answerability / Coverage Check (`check_answerable`)**:
  - Evaluates whether an app's governed `CompiledTBox` (classes, predicates, synonyms) and 3-Tier Memory (Bi-Temporal Graph ABox, Episodic Seed Store, and Procedural Playbooks) cover every domain concept and entity in a user query before the agent attempts an answer.
  - Returns `answerable` (`bool`), `status` (`"ANSWERABLE"` | `"PARTIAL"` | `"UNANSWERABLE"`), `coverage_score` (`0.0–1.0`), `covered_concepts`, `unmatched_concepts`, `matched_tbox_classes`, `matched_entities`, `evidence_counts`, and an actionable `clarification_hint`.
  - Exposed across Python SDK (`service.check_answerable`), FastAPI REST (`POST /v1/memory/check-answerable`, `RemoteCompositeMemoryService.check_answerable`), and MCP Server (`check_answerable` read-only tool).
- **Relational SQL DDL & Apache Ossie Semantic Model Loaders (`DomainPackLoader`)**:
  - `DomainPackLoader.from_sql_ddl` & `POST /v1/domains/packs/sql-ddl`: compiles `CREATE TABLE` schemas, inline/table/ALTER foreign keys, and many-to-many bridge/junction tables into a Layer-1 `DomainPackSpec`.
  - `DomainPackLoader.from_ossie_yaml` & `POST /v1/domains/packs/ossie`: compiles Apache Ossie / BI Semantic Model YAML or JSON (`terms`, `synonyms`, `relationships`, `sql_attributes`) into a Layer-1 `DomainPackSpec`.
- **Steward Domain Pack Certification (`certify_domain_pack`)**:
  - Added `certified` (`bool`), `certified_by`, and `synonyms` to `DomainPackSpec`, and exposed `registrar.certify_domain_pack`, `POST /v1/domains/packs/certify`, and MCP `certify_domain_pack` (expanding the MCP Server to **14 annotated tools**, **4 resources**, and **2 prompts**).

## [0.1.4] - 2026-11-20

### Added
- **Cold-Start Autonomous Ontology Induction Engine (`induce_domain_pack`)**:
  - Synthesizes a complete Layer-1 `DomainPackSpec` (canonicalized `PascalCase` singular classes anchored under Layer 0 Upper Ontology, `UPPER_SNAKE_CASE` domain/range relations, and empirical + lexical `is_functional` / `is_transitive` rules) from sample conversation transcripts or stored app episodes.
  - Exposed across Python SDK (`service.induce_domain_pack`, `registrar.induce_domain_pack_from_payloads`), FastAPI REST (`POST /v1/domains/packs/induce`, `RemoteCompositeMemoryService.induce_domain_pack`), and MCP Server (`induce_domain_pack` — 12th annotated MCP tool).
- **Multi-Occurrence Support Gating (`min_support_count`), Canonicalization & Empirical ABox Cardinality**:
  - Added `PlatformOntologyRegistrar.canonicalize_class_name` and `canonicalize_predicate_name` to merge plural/casing variants (`ColdChainShipments` / `cold_chain_shipment` -> `ColdChainShipment`) and pool support counts before overflow promotion.
  - Added `min_support_count` parameter to `evolve_app_ontology_from_overflow`, `POST /v1/ontology/evolve`, and MCP `evolve_app_ontology` to prevent single-occurrence hallucinations from mutating the governed TBox.
  - Upgraded `infer_upper_concept_parent` with head-noun-aware multi-domain semantic alignment (Retail, Supply Chain, Healthcare, Finance, SRE, SecOps, CRM) and upgraded `infer_relation_flags` with empirical `(source_id, target_id)` graph cardinality analysis.
  - Updated `build_tbox_system_instruction` across Gemini, Bedrock, OpenAI/vLLM, and Ollama extractors to instruct models to emit descriptive `PascalCase` classes and `UPPER_SNAKE_CASE` predicates when `allow_emergent_overflow=True`.
- **Enterprise Best Practices Guide**:
  - Added [`docs/best_practices_guide.md`](docs/best_practices_guide.md) covering consumption pattern selection, database topology tuning, functional vs. multi-valued relation modeling, Hot-Path vs. Sleep-Time extraction, and the Production Go-Live Checklist.

## [0.1.3] - 2026-11-14

### Added
- **100% Data-Plane + Control-Plane Parity Across SDK, FastAPI, `RemoteCompositeMemoryService` & MCP**:
  - Expanded MCP Server (`create_mcp_server`, `MemoryMcpServerTools`, `/v1/mcp/call`) to **11 tools** (`recall_context`, `commit_episode`, `ingest_memory`, `record_decision_trace`, `register_playbook`, `register_domain_pack`, `register_app_manifest`, `evolve_app_ontology`, `run_memory_maintenance`, `inspect_platform_governance`, `purge_user_lineage`).
  - Added FastAPI routes (`GET /healthz`, `GET /v1/domains/packs`, `GET /v1/apps/{tenant_id}/{app_name}`, `POST /v1/playbooks/register`, `POST /v1/ontology/evolve`) and `extraction_mode` / `defer_extraction` support on `POST /v1/memory/events`.
  - Expanded `RemoteCompositeMemoryService` with `register_domain_pack`, `inspect_app_governance`, `register_playbook`, `evolve_app_ontology`, and `run_memory_maintenance`.
- **Latest Model Context Protocol (`2026-07-28` & `2025-11-25`) Compliance**:
  - Full support for MCP `2026-07-28` Stateless Request/Response Core over Streamable HTTP (`/mcp/http/`) with `Mcp-Method` & `Mcp-Name` header routing and `structuredContent` / `outputSchema`.
  - Added explicit `ToolAnnotations` (`title`, `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) across all 11 MCP tools.
  - Added 4 MCP Resources (`memory://governance/health`, `memory://domains/packs`, `memory://audit/ontology`, and dynamic template `memory://apps/{tenant_id}/{app_name}/tbox`) and 2 MCP Prompts (`memory_grounded_assistant`, `ontology_steward_domain_designer`).
  - Wired `StreamableHTTPSessionManager` lifespan and auto-initializing ASGI wrapper for zero-friction Streamable HTTP mounting inside FastAPI.
- **Omnichannel Retail Domain Suite & 3 Consumption Patterns**:
  - Added `examples/domain_packs/retail_omnichannel.json`, `examples/domain_packs/retail_omnichannel_tbox.ttl`, `examples/retail_omnichannel_suite.py`, `examples/three_consumption_patterns_demo.py`, and comprehensive documentation in `docs/`.

## [0.1.2] - 2026-11-14

### Fixed & Optimized
- **Batched Cypher Edge Mutations & Neo4j Composite Index**: Replaced per-edge Cypher loops in `NeptuneOpenCypherAdapter` / `Neo4jGraphAdapter` with single-round-trip `UNWIND $edges` and `UNWIND $func_edges` batches, and auto-created `:Entity(tenant_id, vertex_id)` composite index on Neo4j initialization.
- **Complete Live PostgreSQL + Neo4j Maintenance Wiring**: Wired raw episode session compaction (`[AUTO_SESSION_SUMMARY]`), Cypher low-confidence overflow edge pruning, and Layer-3 Entity Dossier synthesis (`[Dossier: ...]`) in `run_memory_maintenance()` for production PostgreSQL and Neo4j/Neptune stores.
- **Parallel Entity Embedding & Playbook Context Injection**: Parallelized entity embedding generation during extraction via `asyncio.gather` and aligned `register_playbook` trigger intent embeddings so `search_memory()` automatically injects `[MATCHED_PROCEDURAL_PLAYBOOKS]` into the recalled context window.

## [0.1.1] - 2026-11-14

### Changed
- Added full Apache 2.0 `LICENSE` text and `NOTICE` attribution file (`Copyright 2026 Sthitaprajna Sahoo`).
- Added complete PyPI project metadata (`authors`, `maintainers`, `[project.urls]` for Homepage, Repository, Documentation, Changelog, and Issues, and Python 3.11–3.13 classifiers).

## [0.1.0] - 2026-11-13

### Added
- **3-Layer Ontology-Grounded Memory Architecture**:
  - Layer 0 built-in Upper Ontology (`p:Actor`, `p:Object`, `p:Event`, `p:Concept`, `p:CandidateConcept`).
  - Layer 1 Federated Domain Pack loader (`DomainPackLoader`) supporting JSON/YAML, Python Pydantic models, and W3C OWL/SHACL Turtle (`.ttl`) specifications.
  - Layer 2 App-Local Extensions & Two-Bucket Runtime Governance (`TBoxRuntimeValidator`) with automated overflow promotion (`evolve_app_ontology_from_overflow`).
- **3-Tier Context Graph & Procedural Playbooks**:
  - Tier 1 Episodic & Seed Store with Reciprocal Rank Fusion (RRF) hybrid search (`pgvector` HNSW + BM25 GIN) and pinned user preferences.
  - Tier 2 Bi-Temporal Semantic Subgraph (`t_valid`, `t_invalid`, `t_created`), 2-hop multi-hop reasoning, and `EntityResolutionGuard`.
  - Tier 3 Procedural Reasoning Traces and reusable Playbook caching (`PlaybookRecord`).
- **Hot-Path vs. Sleep-Time Batch Extraction**:
  - Configurable `ExtractionMode.HOT_PATH` and `ExtractionMode.SLEEP_TIME_BATCH` with `run_memory_maintenance()` background batch extraction.
- **Modular Per-Port & Single-Stack Database Adapters**:
  - Unified PostgreSQL stack (`PostgresUnifiedAdapter`, `PostgresPgVectorAdapter`, `PostgresGraphAdapter`, `PostgresPlaybookAdapter`).
  - Unified Redis stack (`RedisUnifiedAdapter`, `RedisRegistryAdapter`, `RedisEpisodicAndSeedAdapter`, `RedisGraphAdapter`, `RedisPlaybookAndWorkingAdapter`).
  - Dedicated Graph & Search adapters (`Neo4jGraphAdapter`, `NeptuneOpenCypherAdapter`, `OpenSearchSeedAdapter`, `InMemoryUnifiedAdapter`).
- **Enterprise Cloud Database Authentication (`agentic_memory_sdk.auth`)**:
  - `AwsIamDatabaseAuthProvider` (RDS/Aurora IAM, ElastiCache/MemoryDB IAM, STS AssumeRole), `GcpCloudSqlIamAuthProvider`, `AzureEntraDatabaseAuthProvider`, `VaultOrSecretManagerAuthProvider`, and `EnterpriseTlsConfig` (mTLS).
- **Multi-Provider LLM & Embedding Adapters (`agentic_memory_sdk.llm_extractors`)**:
  - Google Gemini, Amazon Bedrock (Converse API + Titan/Cohere embeddings), OpenAI / Azure OpenAI / vLLM, Ollama, and local SentenceTransformers.
- **Unified FastAPI + MCP Server & Google ADK Integration**:
  - Co-hosted FastAPI REST (`/v1/*`) and MCP Server (`/mcp/http`, `/mcp/sse`, `/v1/mcp/tools`, `/v1/mcp/call`) via `create_memory_platform_app` and `adk-memory` CLI.
  - Lightweight `RemoteCompositeMemoryService` client, `get_adk_memory_tools`, and `non_blocking_memory_commit_callback` for Google ADK agents.
