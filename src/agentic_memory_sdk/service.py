from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

# Graceful import fallback when running outside a Google ADK environment
try:
    from google.adk.events.event import Event
    from google.adk.memory.base_memory_service import (
        BaseMemoryService,
        SearchMemoryResponse,
    )
    from google.adk.memory.memory_entry import MemoryEntry
    from google.adk.sessions.session import Session
    from google.genai import types

    HAS_ADK = True
except ImportError:
    HAS_ADK = False

    @dataclass
    class _Part:
        text: str | None = None

        @classmethod
        def from_text(cls, *, text: str) -> "_Part":
            return cls(text=text)

    @dataclass
    class _Content:
        role: str = "user"
        parts: list[_Part] = field(default_factory=list)

    class _TypesNamespace:
        Part = _Part
        Content = _Content

    types = _TypesNamespace()  # type: ignore

    @dataclass
    class Event:  # type: ignore
        author: str = "user"
        content: Any = None

    @dataclass
    class MemoryEntry:  # type: ignore
        content: Any
        author: str | None = None
        timestamp: str | None = None

    @dataclass
    class SearchMemoryResponse:  # type: ignore
        memories: list[MemoryEntry] = field(default_factory=list)

    @dataclass
    class Session:  # type: ignore
        id: str
        app_name: str
        user_id: str
        events: list[Event] = field(default_factory=list)

    class BaseMemoryService:  # type: ignore
        pass


from .adapters.redis_playbook import RedisPlaybookAndWorkingAdapter
from .governance import (
    AsyncCircuitBreaker,
    PIIScrubber,
    PlatformOntologyRegistrar,
    SmartMemoryClassifier,
    TBoxRuntimeValidator,
    TenantRateLimiter,
    TokenBudgetSerializer,
)
from .models import (
    CompiledTBox,
    DomainPackSpec,
    EpisodicRecord,
    ExtractionMode,
    ExtractionPayload,
    GraphEdgeMutation,
    GraphFactReadResult,
    GraphVertexUpsert,
    OntologyAuditLogEntry,
    OutboxOperation,
    PlatformTenantContext,
    PlaybookRecord,
    ReasoningTraceStep,
    RelationRule,
)
from .ports import (
    EmbeddingAdapter,
    EpisodicAndSeedStoreAdapter,
    GraphStoreAdapter,
    OntologyExtractorAdapter,
    PlaybookStoreAdapter,
    RegistryStoreAdapter,
)

logger = logging.getLogger(__name__)


class CompositeMemoryService(BaseMemoryService):
    """
    Domain-Agnostic, Multi-Tenant Google ADK `BaseMemoryService` Implementation.
    Unifies:
      - Tier 1: Immutable Episodic Memory
      - Tier 2: Stratified Ontology-Grounded Bi-Temporal Semantic Graph (`t_valid`, `t_invalid`)
      - Tier 3: Procedural Reasoning Traces (`AgentRun -> ToolCall -> Entity`)
    Decouples ADK agents from underlying databases via Hexagonal Ports & Adapters.
    """

    def __init__(
        self,
        registry_adapter: RegistryStoreAdapter,
        seed_store_adapter: EpisodicAndSeedStoreAdapter,
        graph_adapter: GraphStoreAdapter,
        embedding_adapter: EmbeddingAdapter,
        extractor_adapter: OntologyExtractorAdapter,
        default_tenant_id: str = "t1",
        max_context_tokens: int = 1500,
        graph_timeout_sec: float = 1.5,
        sync_outbox_immediately: bool = False,
        default_extraction_mode: ExtractionMode | str = ExtractionMode.HOT,
        playbook_adapter: PlaybookStoreAdapter | None = None,
    ):
        self.registry = registry_adapter
        self.seed_store = seed_store_adapter
        self.graph = graph_adapter
        self.embedder = embedding_adapter
        self.extractor = extractor_adapter
        self.default_tenant_id = default_tenant_id
        self.max_context_tokens = max_context_tokens
        self.graph_timeout_sec = graph_timeout_sec
        self.sync_outbox_immediately = sync_outbox_immediately
        self.default_extraction_mode = ExtractionMode(default_extraction_mode)
        if playbook_adapter is not None:
            self.playbook_adapter: PlaybookStoreAdapter = playbook_adapter
        elif isinstance(seed_store_adapter, PlaybookStoreAdapter):
            self.playbook_adapter = seed_store_adapter
        else:
            self.playbook_adapter = RedisPlaybookAndWorkingAdapter()
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=3, recovery_timeout_sec=30.0
        )
        self.rate_limiter = TenantRateLimiter(window_sec=60.0)
        self._outbox_task: asyncio.Task[None] | None = None
        self._outbox_stop_event: asyncio.Event | None = None

    @classmethod
    def from_postgres_stack(
        cls,
        dsn: str,
        embedding_adapter: EmbeddingAdapter,
        extractor_adapter: OntologyExtractorAdapter,
        auth_provider: Any = None,
        tls_config: Any = None,
        default_tenant_id: str = "t1",
        default_extraction_mode: ExtractionMode | str = ExtractionMode.HOT,
    ) -> "CompositeMemoryService":
        """
        Creates a single-stack Pure PostgreSQL `CompositeMemoryService` where Registry,
        Tier 1 Episodic, Tier 2 Seed + Bi-Temporal Graph, and Tier 3 Playbooks all run
        on PostgreSQL (`PostgresUnifiedAdapter`) with optional enterprise cloud IAM/mTLS auth.
        """
        from .adapters.postgres_unified import PostgresUnifiedAdapter

        unified = PostgresUnifiedAdapter(
            dsn=dsn,
            auth_provider=auth_provider,
            tls_config=tls_config,
        )
        return cls(
            registry_adapter=unified,
            seed_store_adapter=unified,
            graph_adapter=unified,
            embedding_adapter=embedding_adapter,
            extractor_adapter=extractor_adapter,
            default_tenant_id=default_tenant_id,
            sync_outbox_immediately=True,
            default_extraction_mode=default_extraction_mode,
            playbook_adapter=unified,  # type: ignore[arg-type]
        )

    @classmethod
    def from_redis_stack(
        cls,
        redis_url: str,
        embedding_adapter: EmbeddingAdapter,
        extractor_adapter: OntologyExtractorAdapter,
        auth_provider: Any = None,
        tls_config: Any = None,
        key_prefix: str = "adk_memory",
        default_tenant_id: str = "t1",
        default_extraction_mode: ExtractionMode | str = ExtractionMode.HOT,
    ) -> "CompositeMemoryService":
        """
        Creates a single-stack Pure Redis `CompositeMemoryService` where Registry,
        Tier 1 Episodic, Tier 2 Seed + Bi-Temporal Graph, and Tier 3 Playbooks all run
        on Redis (`RedisUnifiedAdapter`) with optional enterprise cloud IAM/mTLS auth.
        """
        from .adapters.redis_unified import RedisUnifiedAdapter

        unified = RedisUnifiedAdapter(
            redis_url=redis_url,
            key_prefix=key_prefix,
            auth_provider=auth_provider,
            tls_config=tls_config,
        )
        return cls(
            registry_adapter=unified,
            seed_store_adapter=unified,
            graph_adapter=unified,
            embedding_adapter=embedding_adapter,
            extractor_adapter=extractor_adapter,
            default_tenant_id=default_tenant_id,
            sync_outbox_immediately=True,
            default_extraction_mode=default_extraction_mode,
            playbook_adapter=unified,
        )

    async def _extract_entities_and_edges(
        self,
        ctx: PlatformTenantContext,
        sanitized_transcript: list[str],
        source_ep_id: str,
        target_episodes: Sequence[EpisodicRecord] | None = None,
    ) -> tuple[list[GraphVertexUpsert], list[GraphEdgeMutation]]:
        """
        Shared extraction helper used by both Hot-Path (`add_events_to_memory`)
        and Batch Sleep-Time Mode (`run_memory_maintenance`).
        """
        vertex_upserts: list[GraphVertexUpsert] = []
        edge_mutations: list[GraphEdgeMutation] = []
        if not sanitized_transcript:
            return vertex_upserts, edge_mutations

        try:
            raw_extraction = await self.extractor.extract(
                "\n".join(sanitized_transcript), ctx.tbox
            )
            if (
                raw_extraction.is_ephemeral_chitchat
                and not raw_extraction.entities
                and not raw_extraction.facts
                and target_episodes
            ):
                for ep in target_episodes:
                    ep.metadata["ephemeral"] = True
                    if raw_extraction.classification_reason:
                        ep.metadata["ephemeral_reason"] = (
                            raw_extraction.classification_reason
                        )

            routed_entities, routed_facts = TBoxRuntimeValidator.validate_and_partition(
                ctx, raw_extraction
            )

            name_to_vertex_id: dict[str, str] = {}
            cleaned_entities = [
                (ent, target_ns, PIIScrubber.scrub(ent.summary))
                for ent, target_ns in routed_entities
            ]
            ent_vectors = await asyncio.gather(
                *(
                    self.embedder.embed(f"{ent.canonical_name}: {clean_summary}")
                    for ent, _target_ns, clean_summary in cleaned_entities
                )
            )
            for (ent, target_ns, clean_summary), ent_vec in zip(
                cleaned_entities, ent_vectors
            ):
                existing_vid = await self.seed_store.resolve_existing_vertex_id(
                    ctx, ent.entity_class, ent.canonical_name, ent_vec
                )
                vid = existing_vid or GraphVertexUpsert.deterministic_id(
                    ctx.tenant_id, target_ns, ent.entity_class, ent.canonical_name
                )
                name_to_vertex_id[ent.canonical_name] = vid

                vertex_upserts.append(
                    GraphVertexUpsert(
                        vertex_id=vid,
                        tenant_id=ctx.tenant_id,
                        namespace=target_ns,
                        entity_class=ent.entity_class,
                        upper_class=ctx.tbox.resolve_upper_class(ent.entity_class),
                        canonical_name=ent.canonical_name,
                        summary=clean_summary,
                        embedding=ent_vec,
                    )
                )

            for fact, target_ns, is_func in routed_facts:
                src_vid = name_to_vertex_id.get(fact.source_entity)
                tgt_vid = name_to_vertex_id.get(fact.target_entity)
                if not src_vid or not tgt_vid or src_vid == tgt_vid:
                    continue

                edge_mutations.append(
                    GraphEdgeMutation(
                        edge_id=f"e_{uuid.uuid4().hex[:16]}",
                        tenant_id=ctx.tenant_id,
                        namespace=target_ns,
                        source_vertex_id=src_vid,
                        target_vertex_id=tgt_vid,
                        predicate=fact.predicate,
                        fact_summary=PIIScrubber.scrub(fact.fact_summary),
                        is_functional=is_func,
                        confidence=fact.confidence,
                        source_episode_id=source_ep_id,
                        source_app=ctx.app_name,
                        user_id=ctx.user_id,
                        t_valid=fact.valid_from or GraphEdgeMutation.model_fields["t_valid"].default_factory(),  # type: ignore
                    )
                )
        except Exception as exc:
            logger.warning(
                "Ontology extraction failed; committing episodic memory only: %s", exc
            )
        return vertex_upserts, edge_mutations

    def _split_tenant_app(self, app_name: str) -> tuple[str, str]:
        if ":" in app_name:
            t_id, a_name = app_name.split(":", 1)
            return t_id, a_name
        return self.default_tenant_id, app_name

    # ========================================================================
    # ADK CONTRACT 1: Full Session Ingestion
    # ========================================================================
    async def add_session_to_memory(self, session: Session) -> None:
        await self.add_events_to_memory(
            app_name=session.app_name,
            user_id=session.user_id,
            events=session.events,
            session_id=session.id,
        )

    # ========================================================================
    # ADK CONTRACT 2: Delta Event Ingestion (Hot-Path OR Sleep-Time Batch Mode)
    # ========================================================================
    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        defer_extraction: bool | None = None,
        extraction_mode: ExtractionMode | str | None = None,
    ) -> None:
        tenant_id, clean_app = self._split_tenant_app(app_name)
        ctx = await self.registry.resolve_context(tenant_id, clean_app, user_id)

        if not self.rate_limiter.check_and_record(
            ctx.tenant_id, ctx.app_name, max_rpm=ctx.max_rpm
        ):
            raise RuntimeError(
                f"Rate limit exceeded for tenant='{ctx.tenant_id}', app='{ctx.app_name}' (max_rpm={ctx.max_rpm})"
            )

        if extraction_mode is not None:
            should_defer = ExtractionMode(extraction_mode) == ExtractionMode.SLEEP_TIME
        elif defer_extraction is not None:
            should_defer = defer_extraction
        else:
            should_defer = (
                ctx.tbox.extraction_mode == ExtractionMode.SLEEP_TIME
                or self.default_extraction_mode == ExtractionMode.SLEEP_TIME
            )

        episodic_records: list[EpisodicRecord] = []
        sanitized_transcript: list[str] = []

        # 1. Scrub PII/Secrets and embed raw episodic chunks
        prev_text: str | None = None
        for ev in events:
            if not getattr(ev, "content", None) or not getattr(ev.content, "parts", None):
                continue
            raw_text = "\n".join(
                p.text for p in ev.content.parts if getattr(p, "text", None)
            )
            if not raw_text.strip():
                continue

            clean_text = PIIScrubber.scrub(raw_text)
            vec = await self.embedder.embed(clean_text)
            author = getattr(ev, "author", None) or "agent"
            is_ephemeral = SmartMemoryClassifier.is_ephemeral_turn(
                clean_text, preceding_text=prev_text
            )
            prev_text = clean_text

            ep_meta = dict(custom_metadata or {})
            if is_ephemeral:
                ep_meta["ephemeral"] = True
            elif should_defer:
                ep_meta["extraction_pending"] = True

            ep = EpisodicRecord(
                tenant_id=ctx.tenant_id,
                app_name=ctx.app_name,
                user_id=ctx.user_id,
                session_id=session_id or "default_session",
                namespace=ctx.app_private_namespace,
                author=author,
                content_text=clean_text,
                embedding=vec,
                metadata=ep_meta,
            )
            episodic_records.append(ep)
            if not is_ephemeral:
                sanitized_transcript.append(f"{author}: {clean_text}")

        if not episodic_records:
            return

        # 2. Hot-Path Mode vs. Sleep-Time Deferred Mode
        vertex_upserts: list[GraphVertexUpsert] = []
        edge_mutations: list[GraphEdgeMutation] = []

        if sanitized_transcript and not should_defer:
            vertex_upserts, edge_mutations = await self._extract_entities_and_edges(
                ctx=ctx,
                sanitized_transcript=sanitized_transcript,
                source_ep_id=episodic_records[-1].episode_id,
                target_episodes=episodic_records,
            )

        # 3. Build Transactional Outbox Operation for Graph Store
        outbox_op = None
        if vertex_upserts or edge_mutations:
            outbox_op = OutboxOperation(
                tenant_id=ctx.tenant_id,
                operation_type="UPSERT_GRAPH_BATCH",
                payload={
                    "vertices": [
                        v.model_dump(exclude={"embedding"}) for v in vertex_upserts
                    ],
                    "edges": [e.model_dump() for e in edge_mutations],
                },
            )

        # 4. Atomic ACID Commit to Seed/Episodic Store + Outbox
        await self.seed_store.commit_episodes_and_outbox(
            ctx, episodic_records, vertex_upserts, outbox_op
        )

        if self.sync_outbox_immediately and outbox_op is not None:
            await self.process_outbox_batch(batch_size=10)

    # ========================================================================
    # ADK CONTRACT 3: Direct Memory Entry Ingestion
    # ========================================================================
    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: dict[str, Any] | None = None,
    ) -> None:
        synthetic_events = [
            Event(author=m.author or "user", content=m.content)
            for m in memories
            if getattr(m, "content", None)
        ]
        await self.add_events_to_memory(
            app_name=app_name,
            user_id=user_id,
            events=synthetic_events,
            custom_metadata=custom_metadata,
        )

    # ========================================================================
    # ADK CONTRACT 4: Hybrid Read Path with Circuit Breaker & Token Budgeter
    # ========================================================================
    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
        as_of_iso: str | None = None,
        max_hops: int = 2,
    ) -> SearchMemoryResponse:
        tenant_id, clean_app = self._split_tenant_app(app_name)
        ctx = await self.registry.resolve_context(tenant_id, clean_app, user_id)
        q_vec = await self.embedder.embed(query)

        # 1. Parallel Seed Search (RRF), Verbatim Episodic Search & Semantic Playbook Lookup
        seeds, episodes, matched_playbooks = await asyncio.gather(
            self.seed_store.hybrid_seed_search(ctx, query, q_vec, limit=8),
            self.seed_store.search_episodes(ctx, q_vec, limit=4),
            self.playbook_adapter.match_playbooks(
                ctx.tenant_id, ctx.read_namespaces, q_vec
            ),
        )

        graph_facts: list[GraphFactReadResult] = []
        degraded_mode = False

        # 2. Circuit-Breaker Protected Graph Subgraph Traversal
        if seeds and self.circuit_breaker.allow_request():
            try:
                seed_ids = [s.vertex_id for s in seeds]
                graph_facts = await asyncio.wait_for(
                    self.graph.expand_active_subgraph(
                        ctx, seed_ids, as_of_iso=as_of_iso, max_hops=max_hops, limit=25
                    ),
                    timeout=self.graph_timeout_sec,
                )
                self.circuit_breaker.record_success()
            except Exception as exc:
                logger.error(
                    "Graph adapter failed or timed out; degrading to seed summaries: %s",
                    exc,
                )
                self.circuit_breaker.record_failure()
                degraded_mode = True
        elif seeds:
            degraded_mode = True

        # Edge Case: If Graph is down, synthesize fallback facts from Seed Index summaries
        if degraded_mode and seeds:
            graph_facts = [
                GraphFactReadResult(
                    source_name=s.canonical_name,
                    source_class=s.entity_class,
                    predicate="HAS_SUMMARY",
                    target_name=s.entity_class,
                    target_class=s.upper_class,
                    fact_summary=s.summary,
                    t_valid="SEED_CACHE",
                    t_invalid="ACTIVE",
                    namespace=s.namespace,
                    confidence=0.8,
                )
                for s in seeds
            ]

        # 3. Token-Budgeted Serialization
        graph_block, bounded_episodes = TokenBudgetSerializer.serialize(
            graph_facts,
            episodes,
            max_tokens=self.max_context_tokens,
            degraded_mode=degraded_mode,
        )

        entries: list[MemoryEntry] = []
        if matched_playbooks:
            pb_lines = ["[MATCHED_PROCEDURAL_PLAYBOOKS]"]
            for pb in matched_playbooks:
                steps_str = " -> ".join(pb.workflow_steps)
                pb_lines.append(
                    f"- [{pb.namespace}] Intent='{pb.trigger_intent}' (verified={pb.success_count}x): "
                    f"Steps=[{steps_str}] | Resolution: {pb.resolution_summary}"
                )
            entries.append(
                MemoryEntry(
                    author="platform_playbook_cache",
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="\n".join(pb_lines))],
                    ),
                )
            )
        if graph_block:
            entries.append(
                MemoryEntry(
                    author="platform_knowledge_graph",
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=graph_block)],
                    ),
                )
            )

        for ep in bounded_episodes:
            entries.append(
                MemoryEntry(
                    author=ep.author,
                    timestamp=ep.created_at.isoformat(),
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=ep.content_text)],
                    ),
                )
            )

        return SearchMemoryResponse(memories=entries)

    # ========================================================================
    # PLATFORM EXTENSIONS: Reasoning Traces, Transactional Outbox & GDPR Purge
    # ========================================================================
    async def record_decision_trace(
        self,
        *,
        app_name: str,
        user_id: str,
        run_id: str,
        session_id: str,
        steps: Sequence[ReasoningTraceStep],
    ) -> None:
        """Records Tier 3 Procedural Reasoning Trace (`AgentRun -> ToolCall -> Entity`)."""
        tenant_id, clean_app = self._split_tenant_app(app_name)
        ctx = await self.registry.resolve_context(tenant_id, clean_app, user_id)
        await self.graph.record_reasoning_trace(ctx, run_id, session_id, steps)

    async def register_playbook(
        self,
        *,
        app_name: str,
        user_id: str,
        trigger_intent: str,
        workflow_steps: list[str],
        resolution_summary: str,
        share_in_domain: bool = True,
    ) -> PlaybookRecord:
        """
        Caches a proven Tier 3 Agent Playbook in `RedisPlaybookAndWorkingAdapter`
        so future matching intents across the app or domain resolve with sub-5ms recall.
        """
        tenant_id, clean_app = self._split_tenant_app(app_name)
        ctx = await self.registry.resolve_context(tenant_id, clean_app, user_id)
        target_ns = ctx.write_namespace if share_in_domain else ctx.app_private_namespace
        clean_intent = PIIScrubber.scrub(trigger_intent)
        clean_res = PIIScrubber.scrub(resolution_summary)
        vec = await self.embedder.embed(clean_intent)
        pb = PlaybookRecord(
            tenant_id=ctx.tenant_id,
            namespace=target_ns,
            app_name=ctx.app_name,
            trigger_intent=clean_intent,
            workflow_steps=list(workflow_steps),
            resolution_summary=clean_res,
            embedding=vec,
        )
        return await self.playbook_adapter.upsert_playbook(pb)

    async def process_outbox_batch(self, batch_size: int = 25) -> int:
        """Idempotent background relay syncing pending graph mutations to the Graph Adapter."""
        ops = await self.seed_store.fetch_pending_outbox(batch_size=batch_size)
        processed = 0
        for op in ops:
            try:
                if op.operation_type == "UPSERT_GRAPH_BATCH":
                    vertices = [
                        GraphVertexUpsert(**{**v, "embedding": []})
                        for v in op.payload.get("vertices", [])
                    ]
                    edges = [
                        GraphEdgeMutation(**e)
                        for e in op.payload.get("edges", [])
                    ]
                    await self.graph.upsert_vertices_and_bitemporal_edges(
                        op.tenant_id, vertices, edges
                    )
                await self.seed_store.mark_outbox_processed(op.outbox_id)
                processed += 1
            except Exception as exc:
                logger.error("Outbox op %s failed: %s", op.outbox_id, exc)
                await self.seed_store.mark_outbox_failed(op.outbox_id, str(exc))
        return processed

    async def start_outbox_worker(
        self, interval_sec: float = 2.0, batch_size: int = 25
    ) -> None:
        """Starts an asynchronous background worker task that continuously drains the transactional outbox."""
        if self._outbox_task is not None and not self._outbox_task.done():
            return
        self._outbox_stop_event = asyncio.Event()

        async def _worker_loop() -> None:
            while self._outbox_stop_event and not self._outbox_stop_event.is_set():
                try:
                    await self.process_outbox_batch(batch_size=batch_size)
                except Exception as exc:
                    logger.warning("Background outbox worker iteration error: %s", exc)
                try:
                    await asyncio.wait_for(
                        self._outbox_stop_event.wait(), timeout=interval_sec
                    )
                except asyncio.TimeoutError:
                    pass

        self._outbox_task = asyncio.create_task(_worker_loop())

    async def stop_outbox_worker(self) -> None:
        """Cleanly stops the background transactional outbox worker."""
        if self._outbox_stop_event is not None:
            self._outbox_stop_event.set()
        if self._outbox_task is not None:
            try:
                await asyncio.wait_for(self._outbox_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._outbox_task.cancel()
            self._outbox_task = None

    async def purge_user_lineage(self, app_name: str, user_id: str) -> list[str]:
        """Cascading Right-to-be-Forgotten deletion across Episodic, Seed, and Graph stores."""
        tenant_id, clean_app = self._split_tenant_app(app_name)
        ctx = await self.registry.resolve_context(tenant_id, clean_app, user_id)
        purged_episode_ids = await self.seed_store.purge_user_data(ctx)
        await self.graph.purge_episode_lineage(
            ctx.tenant_id, purged_episode_ids, user_id
        )
        return purged_episode_ids

    async def run_memory_maintenance(
        self,
        *,
        min_episodes_to_compact: int = 4,
        min_overflow_confidence: float = 0.55,
    ) -> dict[str, int]:
        """
        Behind-the-scenes Auto-Summarization & Auto-Cleanup janitor cycle:
          1. Prunes ephemeral chitchat (`metadata.ephemeral == True`) and expired TTL episodes.
          2. Compacts verbose multi-turn session episodes (`>= min_episodes_to_compact`) into
             dense `[AUTO_SESSION_SUMMARY]` episodic records with fresh embeddings.
          3. Cleans up low-confidence quarantined overflow edges (`confidence < min_overflow_confidence`).
        """
        pruned_ephemeral = 0
        compacted_episodes = 0
        summaries_created = 0
        pruned_overflow_edges = 0
        sleep_time_extracted_batches = 0
        sleep_time_extracted_episodes = 0

        # Supports InMemoryUnifiedAdapter directly and PostgresPgVectorAdapter via SQL maintenance
        if hasattr(self.seed_store, "episodes"):
            from collections import defaultdict

            episodes_list: list[EpisodicRecord] = getattr(self.seed_store, "episodes")

            # Step 0: Sleep-Time Batch Ontology Extraction for deferred hot-path episodes!
            pending_by_session: dict[tuple[str, str, str, str], list[EpisodicRecord]] = (
                defaultdict(list)
            )
            for ep in episodes_list:
                if ep.metadata.get("extraction_pending") is True and not ep.metadata.get(
                    "ephemeral"
                ):
                    pending_by_session[
                        (ep.tenant_id, ep.app_name, ep.user_id, ep.session_id)
                    ].append(ep)

            for (t_id, a_name, u_id, _s_id), pending_eps in pending_by_session.items():
                ctx = await self.registry.resolve_context(t_id, a_name, u_id)
                batch_transcript = [
                    f"{e.author}: {e.content_text}" for e in pending_eps
                ]
                v_upserts, e_mutations = await self._extract_entities_and_edges(
                    ctx=ctx,
                    sanitized_transcript=batch_transcript,
                    source_ep_id=pending_eps[-1].episode_id,
                    target_episodes=pending_eps,
                )
                outbox_op = None
                if v_upserts or e_mutations:
                    outbox_op = OutboxOperation(
                        tenant_id=ctx.tenant_id,
                        operation_type="UPSERT_GRAPH_BATCH",
                        payload={
                            "vertices": [
                                v.model_dump(exclude={"embedding"}) for v in v_upserts
                            ],
                            "edges": [e.model_dump() for e in e_mutations],
                        },
                    )
                await self.seed_store.commit_episodes_and_outbox(
                    ctx, [], v_upserts, outbox_op
                )
                if self.sync_outbox_immediately and outbox_op is not None:
                    await self.process_outbox_batch(batch_size=10)
                for ep in pending_eps:
                    ep.metadata.pop("extraction_pending", None)
                sleep_time_extracted_batches += 1
                sleep_time_extracted_episodes += len(pending_eps)

            kept_episodes: list[EpisodicRecord] = []
            for ep in episodes_list:
                if ep.metadata.get("ephemeral") is True:
                    pruned_ephemeral += 1
                else:
                    kept_episodes.append(ep)

            # Group non-summary episodes by (tenant_id, app_name, user_id, session_id)
            from collections import defaultdict

            groups: dict[tuple[str, str, str, str], list[EpisodicRecord]] = defaultdict(list)
            passthrough: list[EpisodicRecord] = []
            for ep in kept_episodes:
                if ep.metadata.get("is_summary"):
                    passthrough.append(ep)
                else:
                    groups[(ep.tenant_id, ep.app_name, ep.user_id, ep.session_id)].append(ep)

            final_episodes: list[EpisodicRecord] = list(passthrough)
            for (t_id, a_name, u_id, s_id), group in groups.items():
                if len(group) >= min_episodes_to_compact:
                    bullet_points = " | ".join(
                        f"{e.author}: {e.content_text.strip()}" for e in group
                    )
                    summary_text = f"[AUTO_SESSION_SUMMARY ({len(group)} turns)]: {bullet_points[:600]}"
                    sum_vec = await self.embedder.embed(summary_text)
                    summary_ep = EpisodicRecord(
                        tenant_id=t_id,
                        app_name=a_name,
                        user_id=u_id,
                        session_id=s_id,
                        namespace=group[0].namespace,
                        author="memory_compactor",
                        content_text=summary_text,
                        embedding=sum_vec,
                        metadata={
                            "is_summary": True,
                            "compacted_count": len(group),
                            "source_episode_ids": [e.episode_id for e in group],
                        },
                    )
                    final_episodes.append(summary_ep)
                    compacted_episodes += len(group)
                    summaries_created += 1
                else:
                    final_episodes.extend(group)

            setattr(self.seed_store, "episodes", final_episodes)
        elif hasattr(self.seed_store, "_get_pool"):
            from collections import defaultdict

            pool = await getattr(self.seed_store, "_get_pool")()
            async with pool.acquire() as conn:
                # 1. Prune ephemeral chitchat in PostgreSQL
                del_rows = await conn.fetch(
                    """
                    DELETE FROM episodic_memories
                    WHERE custom_metadata->>'ephemeral' = 'true'
                    RETURNING episode_id::text
                    """
                )
                pruned_ephemeral += len(del_rows)

                # 2. Fetch pending Sleep-Time episodes in PostgreSQL
                pending_rows = await conn.fetch(
                    """
                    SELECT episode_id::text, tenant_id, app_name, user_id, session_id,
                           namespace, author, content_text, custom_metadata
                    FROM episodic_memories
                    WHERE custom_metadata->>'extraction_pending' = 'true'
                    ORDER BY created_at ASC
                    """
                )
                pg_pending: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
                for r in pending_rows:
                    pg_pending[
                        (r["tenant_id"], r["app_name"], r["user_id"], r["session_id"])
                    ].append(r)

            for (t_id, a_name, u_id, _s_id), rows_group in pg_pending.items():
                ctx = await self.registry.resolve_context(t_id, a_name, u_id)
                batch_transcript = [
                    f"{r['author']}: {r['content_text']}" for r in rows_group
                ]
                v_upserts, e_mutations = await self._extract_entities_and_edges(
                    ctx=ctx,
                    sanitized_transcript=batch_transcript,
                    source_ep_id=rows_group[-1]["episode_id"],
                )
                outbox_op = None
                if v_upserts or e_mutations:
                    outbox_op = OutboxOperation(
                        tenant_id=ctx.tenant_id,
                        operation_type="UPSERT_GRAPH_BATCH",
                        payload={
                            "vertices": [
                                v.model_dump(exclude={"embedding"}) for v in v_upserts
                            ],
                            "edges": [e.model_dump() for e in e_mutations],
                        },
                    )
                await self.seed_store.commit_episodes_and_outbox(
                    ctx, [], v_upserts, outbox_op
                )
                if self.sync_outbox_immediately and outbox_op is not None:
                    await self.process_outbox_batch(batch_size=10)
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE episodic_memories
                        SET custom_metadata = custom_metadata - 'extraction_pending'
                        WHERE tenant_id = $1 AND episode_id = ANY($2::uuid[])
                        """,
                        t_id,
                        [r["episode_id"] for r in rows_group],
                    )
                sleep_time_extracted_batches += 1
                sleep_time_extracted_episodes += len(rows_group)

            # 3. Compact verbose multi-turn sessions in PostgreSQL
            async with pool.acquire() as conn:
                non_sum_rows = await conn.fetch(
                    """
                    SELECT episode_id::text, tenant_id, app_name, user_id, session_id,
                           namespace, author, content_text, custom_metadata
                    FROM episodic_memories
                    WHERE COALESCE(custom_metadata->>'is_summary', 'false') != 'true'
                    ORDER BY created_at ASC
                    """
                )
            pg_groups: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
            for r in non_sum_rows:
                pg_groups[
                    (r["tenant_id"], r["app_name"], r["user_id"], r["session_id"])
                ].append(r)

            for (t_id, a_name, u_id, s_id), group in pg_groups.items():
                if len(group) >= min_episodes_to_compact:
                    bullet_points = " | ".join(
                        f"{r['author']}: {str(r['content_text']).strip()}"
                        for r in group
                    )
                    summary_text = f"[AUTO_SESSION_SUMMARY ({len(group)} turns)]: {bullet_points[:600]}"
                    sum_vec = await self.embedder.embed(summary_text)
                    summary_ep = EpisodicRecord(
                        tenant_id=t_id,
                        app_name=a_name,
                        user_id=u_id,
                        session_id=s_id,
                        namespace=group[0]["namespace"],
                        author="memory_compactor",
                        content_text=summary_text,
                        embedding=sum_vec,
                        metadata={
                            "is_summary": True,
                            "compacted_count": len(group),
                            "source_episode_ids": [r["episode_id"] for r in group],
                        },
                    )
                    ctx = await self.registry.resolve_context(t_id, a_name, u_id)
                    await self.seed_store.commit_episodes_and_outbox(
                        ctx, [summary_ep], [], None
                    )
                    async with pool.acquire() as conn:
                        await conn.execute(
                            """
                            DELETE FROM episodic_memories
                            WHERE tenant_id = $1 AND episode_id = ANY($2::uuid[])
                            """,
                            t_id,
                            [r["episode_id"] for r in group],
                        )
                    compacted_episodes += len(group)
                    summaries_created += 1

        if hasattr(self.graph, "edges"):
            edges_list: list[dict[str, Any]] = getattr(self.graph, "edges")
            kept_edges: list[dict[str, Any]] = []
            for edge in edges_list:
                is_app_overflow = str(edge.get("namespace", "")).startswith("app:") and edge.get("predicate") == "RELATED_TO"
                if is_app_overflow and float(edge.get("confidence", 1.0)) < min_overflow_confidence:
                    pruned_overflow_edges += 1
                else:
                    kept_edges.append(edge)
            setattr(self.graph, "edges", kept_edges)
        elif hasattr(self.graph, "_run_cypher"):
            pruned_rows = await getattr(self.graph, "_run_cypher")(
                self.default_tenant_id,
                """
                MATCH ()-[r:RELATES]->()
                WHERE r.namespace STARTS WITH 'app:'
                  AND r.predicate = 'RELATED_TO'
                  AND coalesce(r.confidence, 1.0) < $min_conf
                WITH r
                DELETE r
                RETURN count(*) AS pruned
                """,
                {"min_conf": min_overflow_confidence},
            )
            if pruned_rows:
                pruned_overflow_edges += int(pruned_rows[0].get("pruned", 0))

        # Step 4 (Graphiti Gc / Letta Sleep-Time Compute): Synthesize Layer 3 Entity Dossiers
        # Enriches each seed vertex's summary with its active 1-hop relational neighborhood
        synthesized_dossiers = 0
        if hasattr(self.seed_store, "seeds") and hasattr(self.graph, "edges"):
            seeds_dict: dict[tuple[str, str], GraphVertexUpsert] = getattr(
                self.seed_store, "seeds"
            )
            edges_list = getattr(self.graph, "edges")
            from collections import defaultdict

            active_rels_by_vid: dict[tuple[str, str], list[str]] = defaultdict(list)
            for edge in edges_list:
                if edge.get("t_invalid") != "ACTIVE":
                    continue
                t_id = str(edge.get("tenant_id", ""))
                src_vid = str(edge.get("source_vertex_id", ""))
                tgt_vid = str(edge.get("target_vertex_id", ""))
                tgt_seed = seeds_dict.get((t_id, tgt_vid))
                if tgt_seed:
                    active_rels_by_vid[(t_id, src_vid)].append(
                        f"{edge['predicate']}->{tgt_seed.canonical_name}"
                    )

            for (t_id, vid), rel_strs in active_rels_by_vid.items():
                seed = seeds_dict.get((t_id, vid))
                if seed and rel_strs:
                    base_summary = seed.summary.split(" [Dossier:")[0].strip()
                    dossier_str = ", ".join(dict.fromkeys(rel_strs[:5]))
                    seed.summary = f"{base_summary} [Dossier: {dossier_str}]"
                    synthesized_dossiers += 1
        elif hasattr(self.seed_store, "_get_pool") and hasattr(self.graph, "_run_cypher"):
            from collections import defaultdict

            pool = await getattr(self.seed_store, "_get_pool")()
            cypher_rels = await getattr(self.graph, "_run_cypher")(
                self.default_tenant_id,
                """
                MATCH (s:Entity)-[r:RELATES {t_invalid: 'ACTIVE'}]->(t:Entity)
                RETURN s.tenant_id AS tenant_id,
                       s.`~id` AS src_vid,
                       r.predicate AS pred,
                       t.canonical_name AS tgt_name
                """,
                {},
            )
            pg_rels_by_vid: dict[tuple[str, str], list[str]] = defaultdict(list)
            for row in cypher_rels:
                if row.get("tenant_id") and row.get("src_vid") and row.get("tgt_name"):
                    pg_rels_by_vid[(str(row["tenant_id"]), str(row["src_vid"]))].append(
                        f"{row['pred']}->{row['tgt_name']}"
                    )
            if pg_rels_by_vid:
                async with pool.acquire() as conn:
                    for (t_id, vid), rel_strs in pg_rels_by_vid.items():
                        row = await conn.fetchrow(
                            """
                            SELECT summary_text FROM graph_seed_index
                            WHERE tenant_id = $1 AND neptune_vertex_id = $2
                            """,
                            t_id,
                            vid,
                        )
                        if row:
                            base_summary = str(row["summary_text"]).split(" [Dossier:")[0].strip()
                            dossier_str = ", ".join(dict.fromkeys(rel_strs[:5]))
                            new_summary = f"{base_summary} [Dossier: {dossier_str}]"
                            await conn.execute(
                                """
                                UPDATE graph_seed_index
                                SET summary_text = $3
                                WHERE tenant_id = $1 AND neptune_vertex_id = $2
                                """,
                                t_id,
                                vid,
                                new_summary,
                            )
                            synthesized_dossiers += 1

        return {
            "sleep_time_extracted_batches": sleep_time_extracted_batches,
            "sleep_time_extracted_episodes": sleep_time_extracted_episodes,
            "pruned_ephemeral_episodes": pruned_ephemeral,
            "compacted_raw_episodes": compacted_episodes,
            "summaries_created": summaries_created,
            "pruned_low_confidence_overflow_edges": pruned_overflow_edges,
            "synthesized_entity_dossiers": synthesized_dossiers,
        }

    async def evolve_app_ontology_from_overflow(
        self,
        app_name: str,
        *,
        min_confidence_to_promote: float = 0.75,
        min_support_count: int = 1,
        registrar: PlatformOntologyRegistrar | None = None,
    ) -> OntologyAuditLogEntry:
        """
        Bottom-Up Automated Schema Evolution:
          1. Scans quarantined overflow vertices (`p:CandidateConcept` with `[ProposedClass=...]`)
             and overflow edges (`RELATED_TO` with `[ProposedPredicate=...]`) in `app:{tenant}:{app}`.
          2. Canonicalizes candidate class names (`PascalCase` singular) and predicate names (`UPPER_SNAKE_CASE`)
             and aggregates support counts across occurrences (`min_support_count`).
          3. Evaluates candidate classes and predicates using `PlatformOntologyRegistrar` intelligence
             (anchoring classes to Layer 0 `UpperConceptType` and inferring functional/transitive flags
             from both lexical rules and empirical ABox graph cardinality).
          4. Modifies the app's live `CompiledTBox` in the registry (bumping `tbox.version`).
          5. Promotes the quarantined vertices and edges into `ctx.write_namespace` while keeping
             all original `source_episode_id`, `t_valid`, `t_created`, and `user_id` provenance intact!
          6. Appends an immutable `OntologyAuditLogEntry` with `source_episode_ids`.
        """
        from collections import defaultdict
        import re

        tenant_id, clean_app = self._split_tenant_app(app_name)
        ctx = await self.registry.resolve_context(tenant_id, clean_app, "ontology_steward")
        tbox = ctx.tbox.model_copy(deep=True)
        prev_ver = tbox.version

        promoted_classes: dict[str, str] = {}
        promoted_relations: dict[str, dict[str, Any]] = {}
        modifications: list[str] = []
        source_ep_ids: set[str] = set()

        class_support: dict[str, int] = defaultdict(int)
        pred_support: dict[str, int] = defaultdict(int)
        pred_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)

        # 1. Inspect quarantined candidate vertices (InMemory OR PostgreSQL + Neo4j)
        if hasattr(self.seed_store, "seeds"):
            seeds_dict: dict[tuple[str, str], GraphVertexUpsert] = getattr(
                self.seed_store, "seeds"
            )
            matching_seeds: list[tuple[GraphVertexUpsert, str, str]] = []
            for (t_id, _vid), seed in seeds_dict.items():
                if t_id != tenant_id or seed.namespace != ctx.app_private_namespace:
                    continue
                m = re.search(r"\[ProposedClass=([^\]]+)\]\s*(.*)", seed.summary)
                if m:
                    cand_cls = PlatformOntologyRegistrar.canonicalize_class_name(
                        m.group(1).strip()
                    )
                    clean_sum = m.group(2).strip()
                    class_support[cand_cls] += 1
                    matching_seeds.append((seed, cand_cls, clean_sum))

            # Also count class support from high-confidence overflow edges before promoting vertices
            if hasattr(self.graph, "edges"):
                for edge in getattr(self.graph, "edges"):
                    if (
                        edge.get("tenant_id") == tenant_id
                        and edge.get("namespace") == ctx.app_private_namespace
                        and float(edge.get("confidence", 0.0)) >= min_confidence_to_promote
                    ):
                        me = re.search(
                            r"\[ProposedPredicate=([^\]|]+)\|([^\->]+)->([^\]]+)\]",
                            str(edge.get("fact_summary", "")),
                        )
                        if me:
                            p_norm = PlatformOntologyRegistrar.canonicalize_predicate_name(
                                me.group(1)
                            )
                            d_cls = PlatformOntologyRegistrar.canonicalize_class_name(
                                me.group(2)
                            )
                            r_cls = PlatformOntologyRegistrar.canonicalize_class_name(
                                me.group(3)
                            )
                            pred_support[p_norm] += 1
                            pred_pairs[p_norm].append(
                                (
                                    str(edge.get("source_vertex_id", "")),
                                    str(edge.get("target_vertex_id", "")),
                                )
                            )
                            for c_ref in (d_cls, r_cls):
                                if c_ref and c_ref != "owl:Thing":
                                    class_support[c_ref] += 1

            for seed, cand_cls, clean_sum in matching_seeds:
                if class_support[cand_cls] < min_support_count:
                    continue
                if cand_cls not in tbox.classes:
                    parent_upper = PlatformOntologyRegistrar.infer_upper_concept_parent(
                        cand_cls
                    )
                    tbox.classes[cand_cls] = parent_upper
                    promoted_classes[cand_cls] = parent_upper
                    modifications.append(
                        f"Promoted emergent class '{cand_cls}' -> '{parent_upper}' from quarantined overflow (support={class_support[cand_cls]})"
                    )
                seed.entity_class = cand_cls
                seed.upper_class = tbox.resolve_upper_class(cand_cls)
                seed.summary = clean_sum
                seed.namespace = ctx.write_namespace
        elif hasattr(self.seed_store, "_get_pool"):
            pool = await getattr(self.seed_store, "_get_pool")()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT neptune_vertex_id, summary_text
                    FROM graph_seed_index
                    WHERE tenant_id = $1 AND namespace = $2 AND summary_text LIKE '[ProposedClass=%'
                    """,
                    tenant_id,
                    ctx.app_private_namespace,
                )
                pg_matches: list[tuple[Any, str, str]] = []
                for r in rows:
                    m = re.search(
                        r"\[ProposedClass=([^\]]+)\]\s*(.*)", r["summary_text"]
                    )
                    if m:
                        cand_cls = PlatformOntologyRegistrar.canonicalize_class_name(
                            m.group(1).strip()
                        )
                        clean_sum = m.group(2).strip()
                        class_support[cand_cls] += 1
                        pg_matches.append((r, cand_cls, clean_sum))

                for r, cand_cls, clean_sum in pg_matches:
                    if class_support[cand_cls] < min_support_count:
                        continue
                    if cand_cls not in tbox.classes:
                        parent_upper = PlatformOntologyRegistrar.infer_upper_concept_parent(
                            cand_cls
                        )
                        tbox.classes[cand_cls] = parent_upper
                        promoted_classes[cand_cls] = parent_upper
                        modifications.append(
                            f"Promoted emergent class '{cand_cls}' -> '{parent_upper}' from PostgreSQL quarantined overflow (support={class_support[cand_cls]})"
                        )
                    upper_c = tbox.resolve_upper_class(cand_cls)
                    await conn.execute(
                        """
                        UPDATE graph_seed_index
                        SET entity_class = $3, upper_class = $4, summary_text = $5, namespace = $6
                        WHERE tenant_id = $1 AND neptune_vertex_id = $2
                        """,
                        tenant_id,
                        r["neptune_vertex_id"],
                        cand_cls,
                        upper_c,
                        clean_sum,
                        ctx.write_namespace,
                    )
                    if hasattr(self.graph, "_run_cypher"):
                        await getattr(self.graph, "_run_cypher")(
                            tenant_id,
                            """
                            MATCH (n:Entity {`~id`: $vid, tenant_id: $tenant_id})
                            SET n.entity_class = $cls, n.upper_class = $ucls, n.summary = $sum, n.namespace = $ns
                            """,
                            {
                                "vid": r["neptune_vertex_id"],
                                "tenant_id": tenant_id,
                                "cls": cand_cls,
                                "ucls": upper_c,
                                "sum": clean_sum,
                                "ns": ctx.write_namespace,
                            },
                        )

        # 2. Inspect quarantined candidate edges (InMemory OR Neo4j Cypher)
        if hasattr(self.graph, "edges"):
            edges_list: list[dict[str, Any]] = getattr(self.graph, "edges")
            for edge in edges_list:
                if (
                    edge.get("tenant_id") != tenant_id
                    or edge.get("namespace") != ctx.app_private_namespace
                    or float(edge.get("confidence", 0.0)) < min_confidence_to_promote
                ):
                    continue
                m = re.search(
                    r"\[ProposedPredicate=([^\]|]+)\|([^\->]+)->([^\]]+)\]\s*(.*)",
                    str(edge.get("fact_summary", "")),
                )
                if m:
                    raw_pred, dom_raw, rng_raw, clean_fact = (
                        m.group(1).strip(),
                        m.group(2).strip(),
                        m.group(3).strip(),
                        m.group(4).strip(),
                    )
                    norm_pred = PlatformOntologyRegistrar.canonicalize_predicate_name(
                        raw_pred
                    )
                    if pred_support[norm_pred] < min_support_count:
                        continue
                    dom_cls = PlatformOntologyRegistrar.canonicalize_class_name(dom_raw)
                    rng_cls = PlatformOntologyRegistrar.canonicalize_class_name(rng_raw)
                    for c_name in (dom_cls, rng_cls):
                        if c_name not in tbox.classes and c_name != "owl:Thing":
                            p_up = PlatformOntologyRegistrar.infer_upper_concept_parent(
                                c_name
                            )
                            tbox.classes[c_name] = p_up
                            promoted_classes[c_name] = p_up

                    if norm_pred not in tbox.relations:
                        is_func, is_trans = PlatformOntologyRegistrar.infer_relation_flags(
                            norm_pred, observed_pairs=pred_pairs[norm_pred]
                        )
                        rule = RelationRule(
                            predicate=norm_pred,
                            domain_class=dom_cls if dom_cls in tbox.classes else "owl:Thing",
                            range_class=rng_cls if rng_cls in tbox.classes else "owl:Thing",
                            is_functional=is_func,
                            is_transitive=is_trans,
                            description="Auto-promoted from emergent runtime overflow",
                        )
                        tbox.relations[norm_pred] = rule
                        promoted_relations[norm_pred] = rule.model_dump()
                        modifications.append(
                            f"Promoted emergent relation '{norm_pred}' ({dom_cls} -> {rng_cls}, functional={is_func}, support={pred_support[norm_pred]})"
                        )

                    # Promote edge in-place while keeping source_episode_id, t_valid, t_created intact!
                    edge["predicate"] = norm_pred
                    edge["fact_summary"] = clean_fact
                    edge["namespace"] = ctx.write_namespace
                    if edge.get("source_episode_id"):
                        source_ep_ids.add(str(edge["source_episode_id"]))
        elif hasattr(self.graph, "_run_cypher"):
            cypher_rows = await getattr(self.graph, "_run_cypher")(
                tenant_id,
                """
                MATCH (s:Entity)-[r:RELATES {tenant_id: $tenant_id, namespace: $ns}]->(t:Entity)
                WHERE r.confidence >= $min_conf AND r.fact_summary STARTS WITH '[ProposedPredicate='
                RETURN r.edge_id AS edge_id, r.fact_summary AS fact_summary, r.source_episode_id AS ep_id,
                       coalesce(s.vertex_id, s.`~id`) AS src_id, coalesce(t.vertex_id, t.`~id`) AS tgt_id
                """,
                {
                    "tenant_id": tenant_id,
                    "ns": ctx.app_private_namespace,
                    "min_conf": min_confidence_to_promote,
                },
            )
            for r in cypher_rows:
                m = re.search(
                    r"\[ProposedPredicate=([^\]|]+)\|([^\->]+)->([^\]]+)\]\s*(.*)",
                    str(r.get("fact_summary", "")),
                )
                if m:
                    p_norm = PlatformOntologyRegistrar.canonicalize_predicate_name(
                        m.group(1)
                    )
                    pred_support[p_norm] += 1
                    pred_pairs[p_norm].append(
                        (str(r.get("src_id", "")), str(r.get("tgt_id", "")))
                    )

            for r in cypher_rows:
                m = re.search(
                    r"\[ProposedPredicate=([^\]|]+)\|([^\->]+)->([^\]]+)\]\s*(.*)",
                    str(r.get("fact_summary", "")),
                )
                if m:
                    raw_pred, dom_raw, rng_raw, clean_fact = (
                        m.group(1).strip(),
                        m.group(2).strip(),
                        m.group(3).strip(),
                        m.group(4).strip(),
                    )
                    norm_pred = PlatformOntologyRegistrar.canonicalize_predicate_name(
                        raw_pred
                    )
                    if pred_support[norm_pred] < min_support_count:
                        continue
                    dom_cls = PlatformOntologyRegistrar.canonicalize_class_name(dom_raw)
                    rng_cls = PlatformOntologyRegistrar.canonicalize_class_name(rng_raw)
                    for c_name in (dom_cls, rng_cls):
                        if c_name not in tbox.classes and c_name != "owl:Thing":
                            p_up = PlatformOntologyRegistrar.infer_upper_concept_parent(
                                c_name
                            )
                            tbox.classes[c_name] = p_up
                            promoted_classes[c_name] = p_up
                    if norm_pred not in tbox.relations:
                        is_func, is_trans = PlatformOntologyRegistrar.infer_relation_flags(
                            norm_pred, observed_pairs=pred_pairs[norm_pred]
                        )
                        rule = RelationRule(
                            predicate=norm_pred,
                            domain_class=dom_cls if dom_cls in tbox.classes else "owl:Thing",
                            range_class=rng_cls if rng_cls in tbox.classes else "owl:Thing",
                            is_functional=is_func,
                            is_transitive=is_trans,
                            description="Auto-promoted from Neo4j emergent runtime overflow",
                        )
                        tbox.relations[norm_pred] = rule
                        promoted_relations[norm_pred] = rule.model_dump()
                        modifications.append(
                            f"Promoted emergent relation '{norm_pred}' ({dom_cls} -> {rng_cls}, functional={is_func}, support={pred_support[norm_pred]})"
                        )
                    await getattr(self.graph, "_run_cypher")(
                        tenant_id,
                        """
                        MATCH ()-[r:RELATES {tenant_id: $tenant_id, edge_id: $edge_id}]->()
                        SET r.predicate = $pred, r.fact_summary = $sum, r.namespace = $ns
                        """,
                        {
                            "tenant_id": tenant_id,
                            "edge_id": r["edge_id"],
                            "pred": norm_pred,
                            "sum": clean_fact,
                            "ns": ctx.write_namespace,
                        },
                    )
                    if r.get("ep_id"):
                        source_ep_ids.add(str(r["ep_id"]))

        if promoted_classes or promoted_relations:
            parts = (prev_ver.split(".") + ["0", "0"])[:3]
            try:
                parts[1] = str(int(parts[1]) + 1)
                tbox.version = ".".join(parts)
            except ValueError:
                tbox.version = f"{prev_ver}-evolved"

            # Persist updated TBox to Registry
            clean_read_ns = [
                ns for ns in ctx.read_namespaces if not ns.startswith("user:")
            ]
            await self.registry.register_app(
                tenant_id=tenant_id,
                app_name=clean_app,
                read_namespaces=clean_read_ns,
                write_namespace=ctx.write_namespace,
                tbox=tbox,
            )

        audit_entry = OntologyAuditLogEntry(
            tenant_id=tenant_id,
            target_id=clean_app,
            action="AUTO_EVOLVE_FROM_OVERFLOW",
            previous_version=prev_ver,
            new_version=tbox.version,
            added_or_modified_classes=promoted_classes,
            added_or_modified_relations=promoted_relations,
            modifications_summary=modifications or ["No eligible overflow candidates met promotion threshold"],
            triggered_by="overflow_auto_evolution_steward",
            source_episode_ids=sorted(source_ep_ids),
        )
        if registrar is not None:
            registrar.audit_log.append(audit_entry)
        return audit_entry

    async def induce_domain_pack(
        self,
        domain_name: str,
        *,
        transcripts: list[str] | None = None,
        app_name: str | None = None,
        version: str = "1.0.0",
        min_support_count: int = 1,
        min_confidence: float = 0.70,
        auto_register: bool = True,
        registrar: PlatformOntologyRegistrar | None = None,
    ) -> tuple[DomainPackSpec, OntologyAuditLogEntry, dict[str, Any]]:
        """
        Cold-Start Autonomous Ontology Induction:
        Synthesizes a complete Layer-1 `DomainPackSpec` (classes anchored under Layer 0 Upper Ontology,
        typed domain/range relations, and empirical + lexical `is_functional` / `is_transitive` rules)
        from either:
          1. A provided list of raw `transcripts` (conversations, incident logs, tickets, or docs), OR
          2. Stored episodic memories for `app_name` when `transcripts` is omitted.
        """
        ont_registrar = registrar or PlatformOntologyRegistrar(self.registry)
        corpus: list[str] = list(transcripts or [])
        tenant_id = "platform"

        if not corpus and app_name:
            tenant_id, clean_app = self._split_tenant_app(app_name)
            if hasattr(self.seed_store, "episodes"):
                for ep in getattr(self.seed_store, "episodes"):
                    if ep.tenant_id == tenant_id and ep.app_name == clean_app:
                        corpus.append(ep.content_text)
            elif hasattr(self.seed_store, "_get_pool"):
                pool = await getattr(self.seed_store, "_get_pool")()
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT content_text FROM episodic_memories
                        WHERE tenant_id = $1 AND app_name = $2
                        ORDER BY created_at DESC LIMIT 100
                        """,
                        tenant_id,
                        clean_app,
                    )
                    corpus.extend(r["content_text"] for r in rows)

        discovery_tbox = CompiledTBox(allow_emergent_overflow=True)
        payloads: list[ExtractionPayload] = []
        for text in corpus:
            extracted = await self.extractor.extract(text, discovery_tbox)
            if not extracted.entities and not extracted.facts:
                extracted = PlatformOntologyRegistrar.heuristic_extract_from_transcript(
                    text
                )
            payloads.append(extracted)

        return ont_registrar.induce_domain_pack_from_payloads(
            domain_name=domain_name,
            payloads=payloads,
            version=version,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
            tenant_id=tenant_id,
            auto_register=auto_register,
        )

    async def check_answerable(
        self,
        app_name: str,
        user_id: str,
        query: str,
        *,
        as_of_iso: str | None = None,
        max_hops: int = 2,
        registrar: PlatformOntologyRegistrar | None = None,
    ) -> dict[str, Any]:
        """
        Pre-flight Ontology & 3-Tier Memory Answerability / Coverage Check:
        Evaluates whether the app's governed `CompiledTBox` (classes, predicates, synonyms)
        and 3-Tier Memory (Bi-Temporal Graph ABox, Episodic Seed Store, and Procedural Playbooks)
        cover every domain concept and entity in `query` before the agent attempts an answer.
        """
        import re

        tenant_id, clean_app = self._split_tenant_app(app_name)
        ctx = await self.registry.resolve_context(tenant_id, clean_app, user_id)
        tbox = ctx.tbox
        q_vec = await self.embedder.embed(query)

        seeds, episodes, matched_playbooks = await asyncio.gather(
            self.seed_store.hybrid_seed_search(ctx, query, q_vec, limit=8),
            self.seed_store.search_episodes(ctx, q_vec, limit=4),
            self.playbook_adapter.match_playbooks(
                ctx.tenant_id, ctx.read_namespaces, q_vec
            ),
        )
        graph_facts: list[GraphFactReadResult] = []
        if seeds and self.circuit_breaker.allow_request():
            try:
                graph_facts = await asyncio.wait_for(
                    self.graph.expand_active_subgraph(
                        ctx,
                        [s.vertex_id for s in seeds],
                        as_of_iso=as_of_iso,
                        max_hops=max_hops,
                        limit=25,
                    ),
                    timeout=self.graph_timeout_sec,
                )
            except Exception:
                graph_facts = []

        # Build vocabulary from TBox classes, predicates, DomainPack synonyms, and recalled ABox/Episodic evidence
        upper_defaults = {
            "p:Actor",
            "p:Object",
            "p:Event",
            "p:Concept",
            "p:CandidateConcept",
            "p:UserPreference",
            "owl:Thing",
        }
        domain_classes = [c for c in tbox.classes if c not in upper_defaults]

        tbox_token_to_class: dict[str, str] = {}
        for cls_name in domain_classes:
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", cls_name).lower()
            tbox_token_to_class[spaced] = cls_name
            for w in spaced.split():
                if len(w) >= 3:
                    tbox_token_to_class[w] = cls_name
                    if w.endswith("s") and len(w) > 4:
                        tbox_token_to_class[w[:-1]] = cls_name
                    else:
                        tbox_token_to_class[w + "s"] = cls_name

        if registrar is not None:
            for pack in registrar._domain_packs.values():
                for cls_name, syns in pack.synonyms.items():
                    for syn in syns:
                        syn_low = syn.lower().strip()
                        tbox_token_to_class[syn_low] = cls_name
                        for w in re.split(r"[\s_\-]+", syn_low):
                            if len(w) >= 3:
                                tbox_token_to_class[w] = cls_name

        tbox_token_to_pred: dict[str, str] = {}
        for pred_name in tbox.relations:
            for w in pred_name.lower().split("_"):
                if len(w) >= 4 and w not in ("from", "with", "into", "over", "primary"):
                    tbox_token_to_pred[w] = pred_name

        evidence_blob_parts: list[str] = []
        matched_entities: list[str] = []
        for s in seeds:
            evidence_blob_parts.append(f"{s.canonical_name} {s.entity_class} {s.summary}")
            if s.canonical_name.lower() in query.lower() and s.canonical_name not in matched_entities:
                matched_entities.append(s.canonical_name)
        for gf in graph_facts:
            evidence_blob_parts.append(
                f"{gf.source_name} {gf.predicate} {gf.target_name} {gf.fact_summary}"
            )
            for ename in (gf.source_name, gf.target_name):
                if ename.lower() in query.lower() and ename not in matched_entities:
                    matched_entities.append(ename)
        for ep in episodes:
            evidence_blob_parts.append(ep.content_text)
        for pb in matched_playbooks:
            evidence_blob_parts.append(
                f"{pb.trigger_intent} {' '.join(pb.workflow_steps)} {pb.resolution_summary}"
            )
        evidence_text_low = " ".join(evidence_blob_parts).lower()

        # Extract candidate concepts & entities from the query
        q_low = query.lower()
        extracted_concepts: list[str] = []
        consumed_spans: list[tuple[int, int]] = []

        # 1. Explicit typed identifiers (ORD-88410, payment-router-01, us-east-1, SKU-9901, etc.)
        id_regex = re.compile(
            r"\b(?:[A-Z]{2,5}-[A-Za-z0-9\-]+|(?:us|eu|ap)-(?:east|west|central|south|north)-\d|[a-z0-9]+-(?:service|api|router|gateway|db|cluster|worker)(?:-\d+)?)\b"
        )
        for m in id_regex.finditer(query):
            val = m.group(0)
            if val not in extracted_concepts:
                extracted_concepts.append(val)
            consumed_spans.append((m.start(), m.end()))

        # 2. Multi-word business / technical domain phrases (e.g., 'return on ad spend', 'ad spend', 'loyalty tier')
        phrase_patterns = [
            "return on ad spend",
            "advertising spend",
            "ad spend",
            "marketing attribution",
            "customer acquisition cost",
            "churn prediction",
            "payroll tax",
            "carbon footprint",
            "loyalty tier",
            "fulfillment center",
            "delivery preference",
            "shipping carrier",
        ]
        for ph in phrase_patterns:
            idx = q_low.find(ph)
            if idx != -1:
                if not any(
                    not (idx + len(ph) <= s or idx >= e) for s, e in consumed_spans
                ):
                    extracted_concepts.append(ph)
                    consumed_spans.append((idx, idx + len(ph)))

        # 3. Remaining content words (excluding stopwords, calendar months, and pure numbers)
        stopwords = {
            "what",
            "was",
            "were",
            "for",
            "the",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "from",
            "by",
            "with",
            "of",
            "is",
            "are",
            "be",
            "been",
            "how",
            "where",
            "when",
            "who",
            "which",
            "why",
            "did",
            "does",
            "do",
            "can",
            "could",
            "would",
            "should",
            "tell",
            "me",
            "about",
            "show",
            "find",
            "get",
            "give",
            "list",
            "check",
            "compute",
            "calculate",
            "current",
            "primary",
            "active",
            "status",
            "latest",
            "recent",
            "history",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "today",
            "yesterday",
        }
        for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b", query):
            if any(not (m.end() <= s or m.start() >= e) for s, e in consumed_spans):
                continue
            word = m.group(0)
            w_low = word.lower()
            if w_low in stopwords or w_low.isdigit():
                continue
            if w_low not in [c.lower() for c in extracted_concepts]:
                extracted_concepts.append(word)

        covered_concepts: list[str] = []
        unmatched_concepts: list[str] = []
        matched_tbox_classes: set[str] = set()
        matched_tbox_predicates: set[str] = set()

        for concept in extracted_concepts:
            c_low = concept.lower()
            in_tbox_cls = tbox_token_to_class.get(c_low)
            in_tbox_pred = tbox_token_to_pred.get(c_low)
            content_toks = [
                tok
                for tok in re.split(r"[\s_\-]+", c_low)
                if len(tok) >= 2 and tok not in stopwords
            ]
            if not in_tbox_cls and content_toks and all(
                tok in tbox_token_to_class for tok in content_toks
            ):
                in_tbox_cls = tbox_token_to_class[content_toks[-1]]
            if not in_tbox_pred and content_toks and all(
                tok in tbox_token_to_pred for tok in content_toks
            ):
                in_tbox_pred = tbox_token_to_pred[content_toks[-1]]

            in_evidence = c_low in evidence_text_low
            if in_tbox_cls:
                matched_tbox_classes.add(in_tbox_cls)
            if in_tbox_pred:
                matched_tbox_predicates.add(in_tbox_pred)

            if in_tbox_cls or in_tbox_pred or in_evidence:
                covered_concepts.append(concept)
            else:
                unmatched_concepts.append(concept)

        total_concepts = len(covered_concepts) + len(unmatched_concepts)
        has_any_evidence = bool(
            graph_facts or episodes or matched_playbooks or matched_entities
        )
        if total_concepts == 0:
            coverage_score = 1.0 if has_any_evidence else 0.0
        else:
            coverage_score = round(len(covered_concepts) / total_concepts, 2)

        if coverage_score >= 0.75 and not unmatched_concepts and has_any_evidence:
            status = "ANSWERABLE"
            answerable = True
            clarification_hint = (
                f"Query is fully covered by ontology and memory (coverage={coverage_score:.2f})."
            )
        elif coverage_score > 0.0 and (covered_concepts or has_any_evidence):
            status = "PARTIAL"
            answerable = False
            clarification_hint = (
                f"Partial coverage ({coverage_score:.2f}): ontology and memory cover {covered_concepts}, "
                f"but no ontology term or memory record covers {unmatched_concepts}. "
                "Ask the user to clarify or connect a Domain Pack for the missing concepts."
            )
        else:
            status = "UNANSWERABLE"
            answerable = False
            clarification_hint = (
                f"Unanswerable from current ontology and memory (coverage=0.00): "
                f"no ontology terms or stored memories match {unmatched_concepts or [query]}."
            )

        return {
            "query": query,
            "answerable": answerable,
            "status": status,
            "coverage_score": coverage_score,
            "covered_concepts": covered_concepts,
            "unmatched_concepts": unmatched_concepts,
            "matched_tbox_classes": sorted(matched_tbox_classes),
            "matched_tbox_predicates": sorted(matched_tbox_predicates),
            "matched_entities": matched_entities,
            "evidence_counts": {
                "graph_facts": len(graph_facts),
                "episodic_records": len(episodes),
                "playbooks": len(matched_playbooks),
            },
            "clarification_hint": clarification_hint,
        }
