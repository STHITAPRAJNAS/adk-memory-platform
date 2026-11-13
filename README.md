# Agentic Memory SDK (`agentic-memory-sdk`)

An enterprise-grade, **domain-agnostic**, multi-tenant **Composite Memory SDK, FastAPI Service & MCP Server** engineered for **Google Agent Development Kit (ADK)** and multi-agent A2A ecosystems.

---

## Installation

```bash
# Core SDK, ADK Integration, FastAPI + MCP Server, Postgres/Redis/Neo4j adapters
pip install agentic-memory-sdk

# With Cloud IAM Auth & Bedrock / Cloud extras
pip install "agentic-memory-sdk[aws,gcp,azure,ontology]"

# Everything included
pip install "agentic-memory-sdk[all]"
```

---

## Key Capabilities

1. **3-Layer Ontology-Grounded Memory**:
   - **Layer 0 (Platform Upper Ontology)**: Built-in meta-graph (`p:Actor`, `p:Object`, `p:Event`, `p:Concept`, `p:CandidateConcept`) with zero configuration required.
   - **Layer 1 (Federated Domain Packs)**: Register domain ontologies via **JSON/YAML**, **Python Pydantic models**, or **W3C OWL/SHACL Turtle (`.ttl`)** files using `DomainPackLoader`.
   - **Layer 2 (App-Local Extensions & Emergent Overflow)**: Two-bucket runtime governance (`TBoxRuntimeValidator`) routes validated facts to `domain:{tenant}:{domain}` and novel concepts to `app:{tenant}:{app}` with automated ontology promotion (`evolve_app_ontology_from_overflow`).

2. **3-Tier Context Graph + Procedural Playbooks**:
   - **Tier 1 (Episodic & Seed Store)**: Hybrid RRF search (`pgvector` HNSW + BM25 GIN) with pinned user preferences and `PIIScrubber`.
   - **Tier 2 (Bi-Temporal Semantic Knowledge Graph)**: Multi-hop reasoning (`max_hops=2`), temporal edge invalidation (`t_valid`, `t_invalid`), and `EntityResolutionGuard`.
   - **Tier 3 (Procedural Reasoning Traces & Playbooks)**: Records agent tool-call trajectories and caches high-reward reusable playbooks.

3. **Hot-Path vs. Sleep-Time Batch Extraction**:
   - `ExtractionMode.HOT_PATH`: Immediate LLM/rule ontology extraction on every turn.
   - `ExtractionMode.SLEEP_TIME_BATCH`: Sub-10ms raw episodic write on the hot path, with background batch ontology extraction during `run_memory_maintenance()`.

4. **Independent Per-Port or Single-Stack Database Adapters**:
   - Run **100% on PostgreSQL** (`CompositeMemoryService.from_postgres_stack`), **100% on Redis** (`CompositeMemoryService.from_redis_stack`), or **Mix-and-Match** per port (e.g., PostgreSQL + Neo4j/Neptune + Redis).
   - Built-in **Enterprise Cloud DB Auth** (`AwsIamDatabaseAuthProvider`, `GcpCloudSqlIamAuthProvider`, `AzureEntraDatabaseAuthProvider`, `VaultOrSecretManagerAuthProvider`, and `EnterpriseTlsConfig` mTLS).

5. **Multi-Provider LLM & Embedding Adapters**:
   - Google **Gemini**, **Amazon Bedrock** (Claude/Nova + Titan/Cohere Embeddings), **OpenAI / Azure OpenAI / vLLM**, **Ollama**, and local **SentenceTransformers**.

6. **Unified FastAPI + MCP Server in One Pod**:
   - `adk-memory serve-api` serves both REST (`/v1/...`) and MCP (`/mcp/http`, `/mcp/sse`, `/v1/mcp/tools`, `/v1/mcp/call`) simultaneously.

---

## Quickstart

### 1. Single-Stack PostgreSQL or Redis Factory

```python
from agentic_memory_sdk import (
    CompositeMemoryService,
    BedrockEmbeddingAdapter,
    BedrockOntologyExtractorAdapter,
    AwsIamDatabaseAuthProvider,
)

# Example: 100% PostgreSQL stack with AWS RDS IAM Auth + Amazon Bedrock
auth = AwsIamDatabaseAuthProvider(region="us-east-1", service="rds-db")
service = CompositeMemoryService.from_postgres_stack(
    dsn="postgresql://iam_user@mydb.cluster.us-east-1.rds.amazonaws.com:5432/memory",
    embedding_adapter=BedrockEmbeddingAdapter(region="us-east-1", target_dim=16),
    extractor_adapter=BedrockOntologyExtractorAdapter(region="us-east-1"),
    auth_provider=auth,
)
```

### 2. Extending `agentic-memory-sdk` with Custom Adapters

Every boundary in `agentic-memory-sdk` is a structural `typing.Protocol` in `agentic_memory_sdk.ports` and `agentic_memory_sdk.auth`. You can extend any port without subclassing internal classes:

```python
from agentic_memory_sdk.ports import EmbeddingAdapter, PlaybookStoreAdapter
from agentic_memory_sdk.models import PlaybookRecord

class MyCustomEmbeddingAdapter:
    """Implements EmbeddingAdapter Protocol via structural subtyping."""
    async def embed(self, text: str) -> list[float]:
        # Call your custom internal embedding gateway
        return [0.1] * 16
```

### 3. Running the Unified FastAPI + MCP Server

```bash
# Start unified REST + MCP server on port 8095
adk-memory serve-api --host 0.0.0.0 --port 8095 --mode in_memory
```
