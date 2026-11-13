from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .models import (
    CompiledTBox,
    EpisodicRecord,
    ExtractionPayload,
    GraphEdgeMutation,
    GraphFactReadResult,
    GraphVertexUpsert,
    OutboxOperation,
    PlatformTenantContext,
    PlaybookRecord,
    ReasoningTraceStep,
)


@runtime_checkable
class RegistryStoreAdapter(Protocol):
    """Port for Platform Entitlements & Ontology TBox Registry."""

    async def register_app(
        self,
        tenant_id: str,
        app_name: str,
        read_namespaces: list[str],
        write_namespace: str,
        tbox: CompiledTBox,
    ) -> None: ...

    async def resolve_context(
        self, tenant_id: str, app_name: str, user_id: str
    ) -> PlatformTenantContext: ...


@runtime_checkable
class EpisodicAndSeedStoreAdapter(Protocol):
    """
    Port for Tier 1 Episodic Storage, Tier 2 Hybrid RRF Seeding / Entity Resolution,
    and the Transactional Outbox.
    Implementations: PostgresPgVectorAdapter, OpenSearchSeedAdapter, InMemoryUnifiedAdapter.
    """

    async def commit_episodes_and_outbox(
        self,
        ctx: PlatformTenantContext,
        episodes: Sequence[EpisodicRecord],
        vertex_upserts: Sequence[GraphVertexUpsert],
        outbox_op: OutboxOperation | None,
    ) -> None: ...

    async def resolve_existing_vertex_id(
        self,
        ctx: PlatformTenantContext,
        entity_class: str,
        canonical_name: str,
        embedding: list[float],
        similarity_threshold: float = 0.91,
    ) -> str | None: ...

    async def hybrid_seed_search(
        self,
        ctx: PlatformTenantContext,
        query_text: str,
        query_embedding: list[float],
        limit: int = 8,
    ) -> list[GraphVertexUpsert]: ...

    async def search_episodes(
        self,
        ctx: PlatformTenantContext,
        query_embedding: list[float],
        limit: int = 4,
    ) -> list[EpisodicRecord]: ...

    async def fetch_pending_outbox(self, batch_size: int = 25) -> list[OutboxOperation]: ...

    async def mark_outbox_processed(self, outbox_id: str) -> None: ...

    async def mark_outbox_failed(self, outbox_id: str, error_msg: str) -> None: ...

    async def purge_user_data(self, ctx: PlatformTenantContext) -> list[str]:
        """Deletes user episodes and returns purged episode_ids for graph lineage cleanup."""
        ...


@runtime_checkable
class GraphStoreAdapter(Protocol):
    """
    Port for Tier 2 Bi-Temporal Semantic Graph & Tier 3 Procedural Reasoning Traces.
    Implementations: NeptuneOpenCypherAdapter, Neo4jGraphAdapter, InMemoryUnifiedAdapter.
    """

    async def upsert_vertices_and_bitemporal_edges(
        self,
        tenant_id: str,
        vertices: Sequence[GraphVertexUpsert],
        edges: Sequence[GraphEdgeMutation],
    ) -> None: ...

    async def expand_active_subgraph(
        self,
        ctx: PlatformTenantContext,
        seed_vertex_ids: Sequence[str],
        as_of_iso: str | None = None,
        max_hops: int = 2,
        limit: int = 25,
    ) -> list[GraphFactReadResult]: ...

    async def record_reasoning_trace(
        self,
        ctx: PlatformTenantContext,
        run_id: str,
        session_id: str,
        steps: Sequence[ReasoningTraceStep],
    ) -> None: ...

    async def purge_episode_lineage(
        self, tenant_id: str, episode_ids: Sequence[str], user_id: str
    ) -> None: ...


@runtime_checkable
class PlaybookStoreAdapter(Protocol):
    """
    Port for Tier 3 Procedural Playbook Caching & Semantic Workflow Recall.
    Implementations:
      - `RedisPlaybookAndWorkingAdapter` (Redis / ElastiCache / MemoryDB / Azure Managed Redis)
      - `PostgresPlaybookAdapter` (PostgreSQL + pgvector)
      - `InMemoryUnifiedAdapter` / `PostgresUnifiedAdapter` / `RedisUnifiedAdapter`
    """

    async def upsert_playbook(self, playbook: PlaybookRecord) -> PlaybookRecord: ...

    async def match_playbooks(
        self,
        tenant_id: str,
        read_namespaces: list[str],
        query_embedding: list[float],
        similarity_threshold: float = 0.85,
        limit: int = 2,
    ) -> list[PlaybookRecord]: ...


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Port for generating dense vector embeddings."""

    async def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class OntologyExtractorAdapter(Protocol):
    """Port for TBox-constrained LLM extraction."""

    async def extract(
        self, transcript: str, tbox: CompiledTBox
    ) -> ExtractionPayload: ...
