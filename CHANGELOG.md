# Changelog

All notable changes to `agentic-composite-memory` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
