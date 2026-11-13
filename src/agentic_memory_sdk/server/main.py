from __future__ import annotations

import os
from pathlib import Path

from ..adapters import (
    InMemoryUnifiedAdapter,
    Neo4jGraphAdapter,
    PostgresGraphAdapter,
    PostgresPgVectorAdapter,
    PostgresPlaybookAdapter,
    PostgresUnifiedAdapter,
    RedisEpisodicAndSeedAdapter,
    RedisGraphAdapter,
    RedisPlaybookAndWorkingAdapter,
    RedisRegistryAdapter,
    RedisUnifiedAdapter,
)
from ..governance import PlatformOntologyRegistrar
from ..llm_extractors import (
    BedrockEmbeddingAdapter,
    BedrockOntologyExtractorAdapter,
    DeterministicHashEmbeddingAdapter,
    GeminiEmbeddingAdapter,
    GeminiOntologyExtractorAdapter,
    LocalSentenceTransformerEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    OllamaOntologyExtractorAdapter,
    OpenAICompatibleEmbeddingAdapter,
    OpenAICompatibleOntologyExtractorAdapter,
    RuleBasedMockExtractorAdapter,
)
from ..models import (
    ExtractedEntity,
    ExtractedFact,
    ExtractionPayload,
)
from ..ontology_loader import DomainPackLoader
from ..service import CompositeMemoryService
from .api import create_memory_platform_app


def build_default_rule_extractor() -> RuleBasedMockExtractorAdapter:
    """
    Configures domain-aware extraction rules for deterministic local & Docker testing
    when `GOOGLE_API_KEY` is not set or `MEMORY_FAKE_LLM=1`.
    """
    extractor = RuleBasedMockExtractorAdapter()

    # SRE Domain Rules
    extractor.add_keyword_rule(
        "checkout-api is deployed in us-east-1",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="checkout-api",
                    entity_class="Microservice",
                    summary="Core checkout payment microservice",
                    confidence=0.95,
                ),
                ExtractedEntity(
                    canonical_name="us-east-1",
                    entity_class="CloudRegion",
                    summary="AWS US East N. Virginia region",
                    confidence=0.95,
                ),
                ExtractedEntity(
                    canonical_name="aurora-payments-01",
                    entity_class="DatabaseCluster",
                    summary="Primary Aurora PostgreSQL cluster for checkout",
                    confidence=0.95,
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="checkout-api",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-east-1",
                    target_class="CloudRegion",
                    fact_summary="checkout-api primary region is us-east-1",
                    confidence=0.95,
                    valid_from="2026-01-15T00:00:00Z",
                ),
                ExtractedFact(
                    source_entity="checkout-api",
                    source_class="Microservice",
                    predicate="READS_FROM",
                    target_entity="aurora-payments-01",
                    target_class="DatabaseCluster",
                    fact_summary="checkout-api reads from aurora-payments-01",
                    confidence=0.95,
                    valid_from="2026-01-15T00:00:00Z",
                ),
            ],
        ),
    )

    extractor.add_keyword_rule(
        "failed checkout-api over to us-west-2",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="checkout-api",
                    entity_class="Microservice",
                    summary="Core checkout payment microservice (failed over to us-west-2)",
                    confidence=0.96,
                ),
                ExtractedEntity(
                    canonical_name="us-west-2",
                    entity_class="CloudRegion",
                    summary="AWS US West Oregon failover region",
                    confidence=0.96,
                ),
                ExtractedEntity(
                    canonical_name="experimental-chaos-mesh",
                    entity_class="UnregisteredChaosTool",
                    summary="Emergent chaos engineering tool used during failover",
                    confidence=0.90,
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="checkout-api",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-west-2",
                    target_class="CloudRegion",
                    fact_summary="checkout-api failed over to primary region us-west-2",
                    confidence=0.96,
                    valid_from="2026-11-01T00:00:00Z",
                ),
                ExtractedFact(
                    source_entity="checkout-api",
                    source_class="Microservice",
                    predicate="TESTED_WITH_UNREGISTERED_TOOL",
                    target_entity="experimental-chaos-mesh",
                    target_class="UnregisteredChaosTool",
                    fact_summary="checkout-api was tested with experimental-chaos-mesh",
                    confidence=0.90,
                    valid_from="2026-11-01T00:00:00Z",
                ),
            ],
        ),
    )

    # CRM Domain Rules
    extractor.add_keyword_rule(
        "acme-corp upgraded to enterprise-platinum",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="Acme-Corp",
                    entity_class="CustomerAccount",
                    summary="Strategic enterprise customer Acme-Corp",
                    confidence=0.95,
                ),
                ExtractedEntity(
                    canonical_name="Enterprise-Platinum",
                    entity_class="SubscriptionPlan",
                    summary="Highest-tier Enterprise Platinum subscription",
                    confidence=0.95,
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Acme-Corp",
                    source_class="CustomerAccount",
                    predicate="SUBSCRIBED_TO_TIER",
                    target_entity="Enterprise-Platinum",
                    target_class="SubscriptionPlan",
                    fact_summary="Acme-Corp is subscribed to Enterprise-Platinum tier",
                    confidence=0.95,
                    valid_from="2026-11-05T00:00:00Z",
                )
            ],
        ),
    )

    return extractor


def build_platform_service_from_env() -> tuple[CompositeMemoryService, PlatformOntologyRegistrar]:
    """
    Builds `CompositeMemoryService` from environment variables.
    Supports either a high-level stack preset (`MEMORY_BACKEND`):
      - `MEMORY_BACKEND=docker`        -> Postgres Seed + Neo4j Graph + Redis Playbook
      - `MEMORY_BACKEND=postgres_only` -> Pure PostgreSQL (`PostgresUnifiedAdapter`)
      - `MEMORY_BACKEND=redis_only`    -> Pure Redis (`RedisUnifiedAdapter`)
      - `MEMORY_BACKEND=memory`        -> Zero-dependency `InMemoryUnifiedAdapter`
    OR independent per-port backend selection:
      - `ADK_REGISTRY_BACKEND=postgres|redis|memory`
      - `ADK_SEED_BACKEND=postgres|redis|memory`
      - `ADK_GRAPH_BACKEND=neo4j|postgres|redis|memory`
      - `ADK_PLAYBOOK_BACKEND=redis|postgres|memory`
    """
    backend = os.environ.get("MEMORY_BACKEND", "docker").lower()
    use_live_gemini = os.environ.get("MEMORY_LIVE_GEMINI", "0") == "1" and bool(
        os.environ.get("GOOGLE_API_KEY")
    )
    emb_provider = os.environ.get(
        "ADK_EMBEDDING_PROVIDER",
        "gemini" if use_live_gemini else "deterministic",
    ).lower()
    llm_provider = os.environ.get(
        "ADK_LLM_PROVIDER",
        "gemini" if use_live_gemini else "rules",
    ).lower()

    if emb_provider == "gemini":
        embedder = GeminiEmbeddingAdapter(
            model=os.environ.get("ADK_EMBEDDING_MODEL", "text-embedding-004"),
            output_dimensionality=1536,
        )
    elif emb_provider == "bedrock":
        embedder = BedrockEmbeddingAdapter(
            model_id=os.environ.get(
                "ADK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
            ),
            region=os.environ.get("AWS_REGION", "us-east-1"),
            target_dim=1536,
        )
    elif emb_provider == "openai":
        embedder = OpenAICompatibleEmbeddingAdapter(
            model=os.environ.get("ADK_EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            dimensions=1536,
        )
    elif emb_provider == "ollama":
        embedder = OllamaEmbeddingAdapter(
            model=os.environ.get("ADK_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            target_dim=1536,
        )
    elif emb_provider == "local_st":
        embedder = LocalSentenceTransformerEmbeddingAdapter(
            model_name=os.environ.get(
                "ADK_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
            ),
            target_dim=1536,
        )
    else:
        embedder = DeterministicHashEmbeddingAdapter(dim=1536)

    if llm_provider == "gemini":
        extractor = GeminiOntologyExtractorAdapter(
            model=os.environ.get("ADK_LLM_MODEL", "gemini-2.5-flash")
        )
    elif llm_provider == "bedrock":
        extractor = BedrockOntologyExtractorAdapter(
            model_id=os.environ.get(
                "ADK_LLM_MODEL", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
            ),
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )
    elif llm_provider == "openai":
        extractor = OpenAICompatibleOntologyExtractorAdapter(
            model=os.environ.get("ADK_LLM_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    elif llm_provider == "ollama":
        extractor = OllamaOntologyExtractorAdapter(
            model=os.environ.get("ADK_LLM_MODEL", "qwen2.5:7b"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        )
    else:
        extractor = build_default_rule_extractor()

    pg_dsn = os.environ.get(
        "POSTGRES_DSN",
        "postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform",
    )
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "platform_secret")
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")

    has_per_port_override = any(
        k in os.environ
        for k in (
            "ADK_REGISTRY_BACKEND",
            "ADK_SEED_BACKEND",
            "ADK_GRAPH_BACKEND",
            "ADK_PLAYBOOK_BACKEND",
        )
    )

    if has_per_port_override:
        mem_fallback = InMemoryUnifiedAdapter()
        reg_type = os.environ.get("ADK_REGISTRY_BACKEND", "postgres").lower()
        seed_type = os.environ.get("ADK_SEED_BACKEND", "postgres").lower()
        graph_type = os.environ.get("ADK_GRAPH_BACKEND", "neo4j").lower()
        pb_type = os.environ.get("ADK_PLAYBOOK_BACKEND", "postgres").lower()

        registry_adapter = (
            PostgresPgVectorAdapter(dsn=os.environ.get("POSTGRES_REGISTRY_DSN", pg_dsn))
            if reg_type == "postgres"
            else RedisRegistryAdapter(redis_url=os.environ.get("REDIS_REGISTRY_URL", redis_url))
            if reg_type == "redis"
            else mem_fallback
        )
        seed_adapter = (
            PostgresPgVectorAdapter(dsn=os.environ.get("POSTGRES_SEED_DSN", pg_dsn))
            if seed_type == "postgres"
            else RedisEpisodicAndSeedAdapter(redis_url=os.environ.get("REDIS_SEED_URL", redis_url))
            if seed_type == "redis"
            else mem_fallback
        )
        graph_adapter = (
            Neo4jGraphAdapter(uri=neo4j_uri, auth=(neo4j_user, neo4j_password))
            if graph_type == "neo4j"
            else PostgresGraphAdapter(dsn=os.environ.get("POSTGRES_GRAPH_DSN", pg_dsn))
            if graph_type == "postgres"
            else RedisGraphAdapter(redis_url=os.environ.get("REDIS_GRAPH_URL", redis_url))
            if graph_type == "redis"
            else mem_fallback
        )
        playbook_adapter = (
            PostgresPlaybookAdapter(dsn=os.environ.get("POSTGRES_PLAYBOOK_DSN", pg_dsn))
            if pb_type == "postgres"
            else RedisPlaybookAndWorkingAdapter(redis_url=os.environ.get("REDIS_PLAYBOOK_URL", redis_url))
            if pb_type == "redis"
            else mem_fallback
        )
        service = CompositeMemoryService(
            registry_adapter=registry_adapter,
            seed_store_adapter=seed_adapter,
            graph_adapter=graph_adapter,
            embedding_adapter=embedder,
            extractor_adapter=extractor,
            playbook_adapter=playbook_adapter,
            sync_outbox_immediately=True,
        )
        registrar = PlatformOntologyRegistrar(registry_adapter)
    elif backend == "postgres_only":
        service = CompositeMemoryService.from_postgres_stack(
            dsn=pg_dsn, embedding_adapter=embedder, extractor_adapter=extractor
        )
        registrar = PlatformOntologyRegistrar(service.registry)
    elif backend == "redis_only":
        service = CompositeMemoryService.from_redis_stack(
            redis_url=redis_url, embedding_adapter=embedder, extractor_adapter=extractor
        )
        registrar = PlatformOntologyRegistrar(service.registry)
    elif backend == "docker":
        pg_adapter = PostgresPgVectorAdapter(dsn=pg_dsn, min_pool_size=2, max_pool_size=15)
        neo4j_adapter = Neo4jGraphAdapter(
            uri=neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        playbook_adapter = RedisPlaybookAndWorkingAdapter(redis_url=redis_url)
        service = CompositeMemoryService(
            registry_adapter=pg_adapter,
            seed_store_adapter=pg_adapter,
            graph_adapter=neo4j_adapter,
            embedding_adapter=embedder,
            extractor_adapter=extractor,
            playbook_adapter=playbook_adapter,
            sync_outbox_immediately=True,
        )
        registrar = PlatformOntologyRegistrar(pg_adapter)
    else:
        mem = InMemoryUnifiedAdapter()
        service = CompositeMemoryService(
            registry_adapter=mem,
            seed_store_adapter=mem,
            graph_adapter=mem,
            embedding_adapter=embedder,
            extractor_adapter=extractor,
            playbook_adapter=mem,
            sync_outbox_immediately=True,
        )
        registrar = PlatformOntologyRegistrar(mem)

    # Pre-load standard sample Domain Packs if present
    repo_root = Path(__file__).resolve().parents[3]
    packs_dir = repo_root / "examples" / "domain_packs"
    if packs_dir.exists():
        for json_file in packs_dir.glob("*.json"):
            pack = DomainPackLoader.from_json_file(json_file)
            registrar.register_domain_pack(pack)

    return service, registrar


service, registrar = build_platform_service_from_env()
app = create_memory_platform_app(service, registrar)
