# Changelog

All notable changes to `agentic-composite-memory` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
