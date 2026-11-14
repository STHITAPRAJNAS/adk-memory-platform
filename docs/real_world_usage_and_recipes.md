# Real-World Usage & Production Recipes

This guide provides complete, copy-pasteable code patterns for deploying and integrating **`agentic-composite-memory`** (`import agentic_composite_memory` or `import agentic_memory_sdk`) across real-world enterprise scenarios.

---

## 1. End-to-End Multi-Agent Enterprise Scenario (SRE + FinOps + SecOps)

In a real enterprise platform, multiple specialized agents serve the same organization (`tenant_id="acme_corp"`) across different bounded contexts while sharing governed domain ontologies and user preferences:

- **Agent 1 (`sre_commander`)**: Manages cloud incidents, microservice dependencies, and regional failovers in `domain:acme_corp:sre_infra`.
- **Agent 2 (`finops_economist`)**: Tracks workload cost-center ownership (`OWNED_BY_PRIMARY`) in `domain:acme_corp:finops` and records executive user preferences (`user:acme_corp:alice_vp`).
- **Agent 3 (`secops_auditor`)**: Scans for zero-day vulnerabilities (`CVE-*`); emergent classes are quarantined in `app:acme_corp:secops_auditor` and promoted via bottom-up ontology evolution.

```python
import asyncio
from agentic_composite_memory import (
    AppMemoryManifest,
    CompositeMemoryService,
    DeterministicHashEmbeddingAdapter,
    DomainPackSpec,
    ExtractionMode,
    GovernanceTier,
    InMemoryUnifiedAdapter,
    PlatformOntologyRegistrar,
    RelationRule,
    RuleBasedMockExtractorAdapter,
    UpperConceptType,
)
from agentic_composite_memory.service import Event, types


def make_event(text: str, author: str = "user") -> Event:
    return Event(
        author=author,
        content=types.Content(role=author, parts=[types.Part.from_text(text=text)]),
    )


async def main() -> None:
    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()

    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        playbook_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(store)

    # 1. Register Shared SRE Domain Pack (Layer 1 TBox)
    await registrar.register_domain_pack(
        "acme_corp",
        DomainPackSpec(
            domain_id="sre_infra",
            version="1.0.0",
            description="Cloud infrastructure, microservices, and regional failover ontology",
            classes={
                "Microservice": UpperConceptType.OBJECT,
                "DatabaseCluster": UpperConceptType.OBJECT,
                "CloudRegion": UpperConceptType.OBJECT,
            },
            relations=[
                RelationRule(
                    predicate="DEPENDS_ON",
                    domain_class="Microservice",
                    range_class="Microservice",
                ),
                RelationRule(
                    predicate="READS_FROM",
                    domain_class="Microservice",
                    range_class="DatabaseCluster",
                ),
                RelationRule(
                    predicate="DEPLOYED_IN_PRIMARY",
                    domain_class="Microservice",
                    range_class="CloudRegion",
                    is_functional=True,  # New primary region automatically closes t_invalid on old region!
                ),
            ],
        ),
    )

    # 2. Onboard the SRE Commander App to Tier 2 (Shared Domain Governance)
    await registrar.onboard_app(
        AppMemoryManifest(
            tenant_id="acme_corp",
            app_name="sre_commander",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["sre_infra@1.0.0"],
        )
    )

    # 3. Register a Reusable Tier-3 Procedural Playbook
    await service.register_playbook(
        app_name="acme_corp:sre_commander",
        user_id="alice_vp",
        trigger_intent="payment-router-01 region failover and ledger check",
        workflow_steps=[
            "verify_aurora_ledger_01_replication",
            "drain_envoy_us_east_1",
            "shift_bgp_anycast_eu_central_1",
        ],
        resolution_summary="Zero-data-loss failover runbook for payment-router-01",
    )

    # 4. Ingest a Sleep-Time Turn (<10ms hot path, deferred extraction)
    await service.add_events_to_memory(
        app_name="acme_corp:sre_commander",
        user_id="alice_vp",
        session_id="inc_101",
        events=[make_event("Post-incident telemetry: p99 latency stabilized at 18ms.")],
        extraction_mode=ExtractionMode.SLEEP_TIME,
    )

    # 5. Run Background Maintenance (Batch extraction + Session Compaction + Entity Dossiers)
    stats = await service.run_memory_maintenance(min_episodes_to_compact=2)
    print("Maintenance stats:", stats)

    # 6. Unified 3-Tier Context Recall (Playbooks + 2-Hop Bi-Temporal Graph + Episodes)
    recall = await service.search_memory(
        app_name="acme_corp:sre_commander",
        user_id="alice_vp",
        query="payment-router-01 region failover and ledger check",
        max_hops=2,
    )
    for entry in recall.memories:
        print(entry.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 2. Database Topologies: Single-Stack Postgres, Single-Stack Redis, or Hybrid

Because each storage port (`RegistryStoreAdapter`, `EpisodicAndSeedStoreAdapter`, `GraphStoreAdapter`, `PlaybookStoreAdapter`) is an independent `typing.Protocol`, you can run the entire platform on **one database stack** or **mix and match** per port:

### Option A: 100% PostgreSQL Stack (`from_postgres_stack`)
Uses PostgreSQL + `pgvector` for Registry, Episodic/Seed RRF search, Bi-Temporal Graph tables (`platform_graph_vertices`, `platform_graph_edges`), and Playbooks (`platform_playbooks`) — zero Graph DB or Redis required:

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
Uses Redis across all 4 ports for ultra-low-latency working memory, episodic seeds, graph adjacency, and procedural playbooks:

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
- **PostgreSQL (`pgvector` + RLS)**: Registry, ACID Transactional Outbox, and Hybrid Vector+BM25 Seed/Episodic Store.
- **Neo4j or Amazon Neptune**: Multi-hop Bi-Temporal Knowledge Graph and Reasoning Traces.
- **Redis**: Sub-5ms Tier-3 Procedural Playbook cache.

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

## 3. Enterprise Cloud Database Authentication & mTLS (`agentic_composite_memory.auth`)

All database adapters accept `auth_provider: CloudAuthProvider` (with automatic concurrency-safe token caching, proactive pre-expiry rotation, and `invalidate()` on connection errors) and `tls_config: EnterpriseTlsConfig`.

### AWS RDS / Aurora IAM + Amazon Neptune SigV4 + ElastiCache IAM
```python
from agentic_composite_memory import (
    AwsIamDatabaseAuthProvider,
    EnterpriseTlsConfig,
    NeptuneOpenCypherAdapter,
    PostgresPgVectorAdapter,
    RedisPlaybookAndWorkingAdapter,
)

tls = EnterpriseTlsConfig(
    enabled=True,
    ca_cert_path="/etc/certs/aws-rds-global-bundle.pem",
    verify_hostname=True,
)

# 1. AWS RDS / Aurora PostgreSQL IAM Token Auth (15-min auto-rotated token)
rds_auth = AwsIamDatabaseAuthProvider(
    region="us-east-1",
    service="rds-db",
    db_user="memory_iam_user",
    host="aurora-cluster.cluster-xyz.us-east-1.rds.amazonaws.com",
    port=5432,
    assume_role_arn="arn:aws:iam::123456789012:role/CrossAccountMemoryRole",
)
pg_adapter = PostgresPgVectorAdapter(
    dsn="postgresql://aurora-cluster.cluster-xyz.us-east-1.rds.amazonaws.com:5432/memory_db",
    auth_provider=rds_auth,
    tls_config=tls,
)

# 2. Amazon ElastiCache / MemoryDB IAM Token Auth
redis_auth = AwsIamDatabaseAuthProvider(
    region="us-east-1",
    service="elasticache",
    db_user="memory-redis-user",
    host="master.memory-cache.use1.cache.amazonaws.com",
)
redis_adapter = RedisPlaybookAndWorkingAdapter(
    redis_url="rediss://master.memory-cache.use1.cache.amazonaws.com:6379/0",
    auth_provider=redis_auth,
    tls_config=tls,
)

# 3. Amazon Neptune openCypher with IAM / STS AssumeRole
neptune_adapter = NeptuneOpenCypherAdapter(
    endpoint_by_tenant={"default": "https://neptune-cluster.cluster-xyz.us-east-1.neptune.amazonaws.com:8182"},
    region="us-east-1",
    aws_auth_provider=rds_auth,
)
```

### GCP Cloud SQL IAM & Azure Entra ID (Managed Identity)
```python
from agentic_composite_memory import (
    AzureEntraDatabaseAuthProvider,
    GcpCloudSqlIamAuthProvider,
    PostgresPgVectorAdapter,
)

# GCP Cloud SQL IAM Auth (ADC or Service Account Impersonation)
gcp_auth = GcpCloudSqlIamAuthProvider(
    db_user="memory-sa@my-gcp-project.iam",
    target_service_account="memory-sa@my-gcp-project.iam.gserviceaccount.com",
)
gcp_pg = PostgresPgVectorAdapter(dsn="postgresql://10.1.2.3:5432/memory", auth_provider=gcp_auth)

# Azure Database for PostgreSQL / Azure Managed Redis via Entra ID
azure_auth = AzureEntraDatabaseAuthProvider(principal_name="memory-managed-identity")
azure_pg = PostgresPgVectorAdapter(dsn="postgresql://my-pg.postgres.database.azure.com:5432/memory", auth_provider=azure_auth)
```

---

## 4. Multi-Provider LLM & Embedding Adapters (`agentic_composite_memory.llm_extractors`)

Swap between Google Gemini, Amazon Bedrock, OpenAI / Azure OpenAI / vLLM, Ollama, or local HuggingFace models without changing a single line of service code:

| Provider | Embedding Adapter | Ontology Extractor Adapter |
| :--- | :--- | :--- |
| **Google Gemini / Vertex AI** | `GeminiEmbeddingAdapter(model="text-embedding-004", output_dimensionality=1536)` | `GeminiOntologyExtractorAdapter(model="gemini-2.5-flash")` |
| **Amazon Bedrock** | `BedrockEmbeddingAdapter(model_id="amazon.titan-embed-text-v2:0", dimensions=1024, target_dim=1536)` | `BedrockOntologyExtractorAdapter(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")` |
| **OpenAI / Azure OpenAI / vLLM** | `OpenAICompatibleEmbeddingAdapter(base_url="https://api.openai.com/v1", model="text-embedding-3-small")` | `OpenAICompatibleOntologyExtractorAdapter(base_url="https://api.openai.com/v1", model="gpt-4o-mini")` |
| **Local Ollama** | `OllamaEmbeddingAdapter(model="nomic-embed-text", target_dim=1536)` | `OllamaOntologyExtractorAdapter(model="llama3.2")` |
| **Local SentenceTransformers** | `LocalSentenceTransformerEmbeddingAdapter(model_name="BAAI/bge-small-en-v1.5", target_dim=1536)` | *(Pair with any LLM extractor above)* |

---

## 5. Google ADK Agent Integration (In-Process vs. Remote Microservice)

### Pattern A: Lightweight Remote Client inside ADK Agent Pods (Recommended for Kubernetes)
Agent pods only install `pip install agentic-composite-memory` (no DB pools held inside agent containers) and talk to the central Memory Platform Service over HTTP:

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from agentic_composite_memory import (
    RemoteCompositeMemoryService,
    get_adk_memory_tools,
    non_blocking_memory_commit_callback,
)

remote_memory = RemoteCompositeMemoryService(base_url="http://memory-platform.default.svc.cluster.local:8095")

sre_agent = LlmAgent(
    name="sre_specialist",
    model="gemini-2.5-flash",
    instruction="You are an SRE Specialist. Always recall context from composite memory before acting.",
    tools=get_adk_memory_tools(remote_memory, default_app_name="acme:sre_specialist"),
    after_agent_callback=non_blocking_memory_commit_callback,
)

runner = Runner(
    app_name="acme:sre_specialist",
    agent=sre_agent,
    memory_service=remote_memory,
)
```

---

## 6. Extending the SDK with Custom Adapters (`typing.Protocol`)

Every boundary in `agentic_composite_memory.ports` uses structural subtyping (`@runtime_checkable class ... (Protocol)`). You can plug in any custom vector database, graph database, or internal corporate LLM gateway without subclassing internal SDK classes:

```python
from agentic_composite_memory import EmbeddingAdapter, PlaybookRecord, PlaybookStoreAdapter


class CustomCorporateGatewayEmbedder:
    """Satisfies EmbeddingAdapter Protocol automatically."""

    async def embed(self, text: str) -> list[float]:
        # Call your internal corporate embedding gateway
        return [0.01] * 1536
```
