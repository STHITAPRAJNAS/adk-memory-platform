# Documentation Index — `agentic-composite-memory`

Welcome to the documentation for **Agentic Composite Memory** (`agentic-composite-memory` on PyPI), an enterprise-grade, multi-tenant, ontology-grounded **Composite Memory SDK, FastAPI Service, and MCP Server** for **Google Agent Development Kit (ADK)** and multi-agent A2A ecosystems.

---

## Documentation Guides

| Guide | What You Will Learn |
| :--- | :--- |
| **[1. Architecture & Ontology Guide](./architecture_and_ontology_guide.md)** | First principles of the 3-Layer Stratified Ontology (`Layer 0` Upper Ontology, `Layer 1` Federated Domain Packs, `Layer 2` App Overflow), the 3-Tier Context Graph (Episodic, Bi-Temporal Semantic Graph, Procedural Playbooks), Hot-Path vs. Sleep-Time Batch Extraction, Bottom-Up Schema Evolution, and the 8-Point Production Edge-Case Defense Matrix. |
| **[2. Real-World Usage & Production Recipes](./real_world_usage_and_recipes.md)** | End-to-end Python recipes for multi-agent enterprise deployments (SRE + FinOps + SecOps), Single-Stack PostgreSQL (`from_postgres_stack`), Single-Stack Redis (`from_redis_stack`), Hybrid Multi-Backend setups, Cloud IAM DB Auth (AWS RDS/Neptune/ElastiCache, GCP Cloud SQL, Azure Entra ID, Vault, mTLS), Multi-Provider LLM/Embedding adapters (Gemini, Bedrock, OpenAI/vLLM, Ollama, SentenceTransformers), Google ADK integration, and writing custom adapters. |
| **[3. Local Setup, Docker Stack & PyPI Testing Guide](./local_setup_and_testing.md)** | Step-by-step instructions to spin up our local Docker stack (`PostgreSQL 16 + pgvector`, `Neo4j 5`, `Redis 7`), run unit and live A2A E2E test suites, verify the published PyPI package (`agentic-composite-memory`) in an isolated environment, and inspect the 5-Agent Enterprise Stress & Maturity Benchmark results. |
| **[4. FastAPI REST, MCP Server & CLI Reference](./api_and_mcp_reference.md)** | Complete reference for all co-hosted FastAPI REST endpoints (`/v1/...`), MCP Server transports (`/mcp/http`, `/mcp/sse`, `/v1/mcp/tools`, `/v1/mcp/call`), the `adk-memory` CLI, and Kubernetes environment variables. |
| **[5. PostgreSQL DDL & Row-Level Security Schema](./sql/001_postgres_rls_and_partitions.sql)** | Production SQL schema for `pgvector` HNSW indexes, GIN full-text search indexes, kernel-enforced tenant Row-Level Security (RLS), hash partitioning, and the ACID transactional outbox. |
| **[6. Live Maturity Benchmark Report (JSON)](./maturity_observation_report.json)** | Latest automated telemetry and verification output from `examples/realistic_enterprise_stress_test.py` executed against the live Docker PostgreSQL + Neo4j + Redis stack. |
