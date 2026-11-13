from __future__ import annotations

import json
import re
from typing import Sequence

from ..governance import EntityResolutionGuard
from ..models import (
    CompiledTBox,
    EpisodicRecord,
    GraphEdgeMutation,
    GraphFactReadResult,
    GraphVertexUpsert,
    OutboxOperation,
    PlatformTenantContext,
    ReasoningTraceStep,
)
from ..ports import (
    EpisodicAndSeedStoreAdapter,
    GraphStoreAdapter,
    RegistryStoreAdapter,
)
from .redis_playbook import RedisPlaybookAndWorkingAdapter


class RedisRegistryAdapter(RedisPlaybookAndWorkingAdapter, RegistryStoreAdapter):
    """
    Standalone Redis implementation of `RegistryStoreAdapter`.
    Can be paired with any Seed, Graph, or Playbook adapter.
    """

    def _k_registry(self, tenant_id: str) -> str:
        return f"{self.key_prefix}:registry:{tenant_id}"

    async def register_app(
        self,
        tenant_id: str,
        app_name: str,
        read_namespaces: list[str],
        write_namespace: str,
        tbox: CompiledTBox,
    ) -> None:
        r = await self._get_redis()
        if r is None:
            raise RuntimeError("RedisRegistryAdapter requires a valid `redis_url`.")
        payload = {
            "tenant_id": tenant_id,
            "app_name": app_name,
            "read_namespaces": read_namespaces,
            "write_namespace": write_namespace,
            "tbox": tbox.model_dump(mode="json"),
        }
        await r.hset(self._k_registry(tenant_id), app_name, json.dumps(payload))

    async def resolve_context(
        self, tenant_id: str, app_name: str, user_id: str
    ) -> PlatformTenantContext:
        r = await self._get_redis()
        if r is None:
            raise RuntimeError("RedisRegistryAdapter requires a valid `redis_url`.")
        app_private_ns = f"app:{tenant_id}:{app_name}"
        user_private_ns = f"user:{tenant_id}:{user_id}"
        raw = await r.hget(self._k_registry(tenant_id), app_name)
        if not raw:
            return PlatformTenantContext(
                tenant_id=tenant_id,
                app_name=app_name,
                user_id=user_id,
                app_private_namespace=app_private_ns,
                write_namespace=app_private_ns,
                read_namespaces=[
                    app_private_ns,
                    user_private_ns,
                    f"global:{tenant_id}:core",
                ],
                tbox=CompiledTBox(),
            )
        data = json.loads(raw)
        merged_read_ns = list(
            dict.fromkeys([*list(data["read_namespaces"]), user_private_ns])
        )
        return PlatformTenantContext(
            tenant_id=tenant_id,
            app_name=app_name,
            user_id=user_id,
            app_private_namespace=app_private_ns,
            write_namespace=data["write_namespace"],
            read_namespaces=merged_read_ns,
            tbox=CompiledTBox.model_validate(data["tbox"]),
        )


class RedisEpisodicAndSeedAdapter(
    RedisPlaybookAndWorkingAdapter, EpisodicAndSeedStoreAdapter
):
    """
    Standalone Redis implementation of `EpisodicAndSeedStoreAdapter` (Tier 1 Episodic
    Storage, Tier 2 Hybrid RRF Seeding, Entity Resolution Guard, and Transactional Outbox).
    Can be paired with Neo4j, Neptune, PostgresGraphAdapter, or RedisGraphAdapter.
    """

    def _k_episodes(self, tenant_id: str) -> str:
        return f"{self.key_prefix}:episodes:{tenant_id}"

    def _k_vertices(self, tenant_id: str) -> str:
        return f"{self.key_prefix}:vertices:{tenant_id}"

    def _k_outbox(self) -> str:
        return f"{self.key_prefix}:outbox:global"

    async def resolve_existing_vertex_id(
        self,
        ctx: PlatformTenantContext,
        entity_class: str,
        canonical_name: str,
        embedding: list[float],
        similarity_threshold: float = 0.91,
    ) -> str | None:
        r = await self._get_redis()
        raw_vertices = await r.hvals(self._k_vertices(ctx.tenant_id))
        best_vid: str | None = None
        best_score: float = -1.0
        for raw in raw_vertices:
            v = GraphVertexUpsert.model_validate_json(raw)
            if v.namespace not in ctx.read_namespaces or v.entity_class != entity_class:
                continue
            sim = self._cosine(v.embedding, embedding)
            if EntityResolutionGuard.can_merge(
                v.canonical_name, canonical_name, sim, similarity_threshold
            ):
                if v.canonical_name.strip().lower() == canonical_name.strip().lower():
                    return v.vertex_id
                if sim > best_score:
                    best_score = sim
                    best_vid = v.vertex_id
        return best_vid

    async def commit_episodes_and_outbox(
        self,
        ctx: PlatformTenantContext,
        episodes: Sequence[EpisodicRecord],
        vertex_upserts: Sequence[GraphVertexUpsert],
        outbox_op: OutboxOperation | None,
    ) -> None:
        r = await self._get_redis()
        pipe = r.pipeline(transaction=True)
        for ep in episodes:
            pipe.hset(
                self._k_episodes(ctx.tenant_id),
                ep.episode_id,
                ep.model_dump_json(),
            )
        for v in vertex_upserts:
            pipe.hset(
                self._k_vertices(ctx.tenant_id),
                v.vertex_id,
                v.model_dump_json(),
            )
        if outbox_op is not None:
            pipe.hset(
                self._k_outbox(),
                outbox_op.outbox_id,
                outbox_op.model_dump_json(),
            )
        await pipe.execute()

    async def hybrid_seed_search(
        self,
        ctx: PlatformTenantContext,
        query_text: str,
        query_embedding: list[float],
        limit: int = 8,
    ) -> list[GraphVertexUpsert]:
        r = await self._get_redis()
        raw_vertices = await r.hvals(self._k_vertices(ctx.tenant_id))
        candidates: list[GraphVertexUpsert] = []
        pinned_user_seeds: list[GraphVertexUpsert] = []
        user_ns = f"user:{ctx.tenant_id}:{ctx.user_id}"

        for raw in raw_vertices:
            v = GraphVertexUpsert.model_validate_json(raw)
            if v.namespace not in ctx.read_namespaces:
                continue
            if v.namespace == user_ns:
                pinned_user_seeds.append(v)
            candidates.append(v)

        if not candidates:
            return []

        q_tokens = set(re.findall(r"[a-z0-9_]+", query_text.lower()))
        vec_ranked = sorted(
            candidates,
            key=lambda x: self._cosine(x.embedding, query_embedding),
            reverse=True,
        )
        lex_ranked = sorted(
            candidates,
            key=lambda x: len(
                q_tokens
                & set(
                    re.findall(
                        r"[a-z0-9_]+", f"{x.canonical_name} {x.summary}".lower()
                    )
                )
            ),
            reverse=True,
        )

        rrf_scores: dict[str, float] = {}
        by_id: dict[str, GraphVertexUpsert] = {}
        for rank, v in enumerate(vec_ranked, start=1):
            by_id[v.vertex_id] = v
            rrf_scores[v.vertex_id] = rrf_scores.get(v.vertex_id, 0.0) + 1.0 / (
                60 + rank
            )
        for rank, v in enumerate(lex_ranked, start=1):
            by_id[v.vertex_id] = v
            rrf_scores[v.vertex_id] = rrf_scores.get(v.vertex_id, 0.0) + 1.0 / (
                60 + rank
            )

        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda vid: rrf_scores[vid], reverse=True
        )
        top_seeds = [by_id[vid] for vid in sorted_ids[:limit]]
        seen_ids = {v.vertex_id for v in top_seeds}
        for pv in pinned_user_seeds:
            if pv.vertex_id not in seen_ids:
                top_seeds.append(pv)
                seen_ids.add(pv.vertex_id)
        return top_seeds

    async def search_episodes(
        self,
        ctx: PlatformTenantContext,
        query_embedding: list[float],
        limit: int = 4,
    ) -> list[EpisodicRecord]:
        r = await self._get_redis()
        raw_eps = await r.hvals(self._k_episodes(ctx.tenant_id))
        eps: list[EpisodicRecord] = []
        for raw in raw_eps:
            ep = EpisodicRecord.model_validate_json(raw)
            if ep.app_name == ctx.app_name and ep.user_id == ctx.user_id:
                eps.append(ep)
        eps.sort(key=lambda e: self._cosine(e.embedding, query_embedding), reverse=True)
        return eps[:limit]

    async def fetch_pending_outbox(self, batch_size: int = 25) -> list[OutboxOperation]:
        r = await self._get_redis()
        raw_ops = await r.hvals(self._k_outbox())
        pending: list[OutboxOperation] = []
        for raw in raw_ops:
            op = OutboxOperation.model_validate_json(raw)
            if op.status in ("PENDING", "RETRY") and op.attempts < 5:
                pending.append(op)
        return pending[:batch_size]

    async def mark_outbox_processed(self, outbox_id: str) -> None:
        r = await self._get_redis()
        raw = await r.hget(self._k_outbox(), outbox_id)
        if raw:
            op = OutboxOperation.model_validate_json(raw)
            op.status = "PROCESSED"
            await r.hset(self._k_outbox(), outbox_id, op.model_dump_json())

    async def mark_outbox_failed(self, outbox_id: str, error_msg: str) -> None:
        r = await self._get_redis()
        raw = await r.hget(self._k_outbox(), outbox_id)
        if raw:
            op = OutboxOperation.model_validate_json(raw)
            op.attempts += 1
            op.status = "DEAD_LETTER" if op.attempts >= 5 else "RETRY"
            await r.hset(self._k_outbox(), outbox_id, op.model_dump_json())

    async def purge_user_data(self, ctx: PlatformTenantContext) -> list[str]:
        r = await self._get_redis()
        raw_map = await r.hgetall(self._k_episodes(ctx.tenant_id))
        purged_ids: list[str] = []
        for ep_id, raw in raw_map.items():
            ep = EpisodicRecord.model_validate_json(raw)
            if ep.user_id == ctx.user_id:
                purged_ids.append(ep_id)
        if purged_ids:
            await r.hdel(self._k_episodes(ctx.tenant_id), *purged_ids)
        return purged_ids


class RedisGraphAdapter(RedisPlaybookAndWorkingAdapter, GraphStoreAdapter):
    """
    Standalone Redis implementation of `GraphStoreAdapter` (Bi-Temporal Semantic Graph
    with functional edge invalidation & Tier 3 Procedural Reasoning Traces).
    Can be paired with PostgresPgVectorAdapter, OpenSearchSeedAdapter, or RedisEpisodicAndSeedAdapter.
    """

    def _k_vertices(self, tenant_id: str) -> str:
        return f"{self.key_prefix}:vertices:{tenant_id}"

    def _k_edges(self, tenant_id: str) -> str:
        return f"{self.key_prefix}:edges:{tenant_id}"

    def _k_traces(self, tenant_id: str) -> str:
        return f"{self.key_prefix}:traces:{tenant_id}"

    async def upsert_vertices_and_bitemporal_edges(
        self,
        tenant_id: str,
        vertices: Sequence[GraphVertexUpsert],
        edges: Sequence[GraphEdgeMutation],
    ) -> None:
        r = await self._get_redis()
        for v in vertices:
            existing_raw = await r.hget(self._k_vertices(tenant_id), v.vertex_id)
            if existing_raw and not v.embedding:
                old_v = GraphVertexUpsert.model_validate_json(existing_raw)
                v.embedding = old_v.embedding
            await r.hset(self._k_vertices(tenant_id), v.vertex_id, v.model_dump_json())

        raw_edges = await r.hgetall(self._k_edges(tenant_id))
        existing_edges: dict[str, GraphEdgeMutation] = {
            eid: GraphEdgeMutation.model_validate_json(raw)
            for eid, raw in raw_edges.items()
        }

        for new_e in edges:
            if new_e.is_functional:
                for eid, cur_e in existing_edges.items():
                    if (
                        cur_e.source_vertex_id == new_e.source_vertex_id
                        and cur_e.predicate == new_e.predicate
                        and cur_e.namespace == new_e.namespace
                        and cur_e.t_invalid == "ACTIVE"
                        and cur_e.target_vertex_id != new_e.target_vertex_id
                    ):
                        cur_e.t_invalid = new_e.t_valid
                        await r.hset(
                            self._k_edges(tenant_id), eid, cur_e.model_dump_json()
                        )

            matched_dup = False
            for eid, cur_e in existing_edges.items():
                if (
                    cur_e.source_vertex_id == new_e.source_vertex_id
                    and cur_e.predicate == new_e.predicate
                    and cur_e.target_vertex_id == new_e.target_vertex_id
                    and cur_e.namespace == new_e.namespace
                    and cur_e.t_invalid == "ACTIVE"
                ):
                    cur_e.fact_summary = new_e.fact_summary
                    cur_e.reinforcement_count += 1
                    cur_e.confidence = min(1.0, cur_e.confidence + 0.03)
                    await r.hset(self._k_edges(tenant_id), eid, cur_e.model_dump_json())
                    matched_dup = True
                    break

            if not matched_dup:
                existing_edges[new_e.edge_id] = new_e
                await r.hset(
                    self._k_edges(tenant_id),
                    new_e.edge_id,
                    new_e.model_dump_json(),
                )

    async def expand_active_subgraph(
        self,
        ctx: PlatformTenantContext,
        seed_vertex_ids: Sequence[str],
        as_of_iso: str | None = None,
        max_hops: int = 2,
        limit: int = 25,
    ) -> list[GraphFactReadResult]:
        if not seed_vertex_ids:
            return []
        r = await self._get_redis()
        raw_vertices = await r.hgetall(self._k_vertices(ctx.tenant_id))
        vertices = {
            vid: GraphVertexUpsert.model_validate_json(raw)
            for vid, raw in raw_vertices.items()
        }
        raw_edges = await r.hvals(self._k_edges(ctx.tenant_id))
        all_edges = [GraphEdgeMutation.model_validate_json(raw) for raw in raw_edges]

        frontier = set(seed_vertex_ids)
        visited_vertices = set(seed_vertex_ids)
        seen_edge_ids: set[str] = set()
        results: list[GraphFactReadResult] = []

        for _hop in range(max(1, max_hops)):
            if not frontier or len(results) >= limit:
                break
            next_frontier: set[str] = set()
            for e in all_edges:
                if e.edge_id in seen_edge_ids:
                    continue
                if e.namespace not in ctx.read_namespaces:
                    continue
                if (
                    e.source_vertex_id not in frontier
                    and e.target_vertex_id not in frontier
                ):
                    continue
                if as_of_iso is None:
                    if e.t_invalid != "ACTIVE":
                        continue
                else:
                    if not (
                        e.t_valid <= as_of_iso
                        and (e.t_invalid == "ACTIVE" or e.t_invalid > as_of_iso)
                    ):
                        continue

                src_v = vertices.get(e.source_vertex_id)
                tgt_v = vertices.get(e.target_vertex_id)
                if not src_v or not tgt_v:
                    continue
                if (
                    src_v.namespace not in ctx.read_namespaces
                    or tgt_v.namespace not in ctx.read_namespaces
                ):
                    continue

                seen_edge_ids.add(e.edge_id)
                results.append(
                    GraphFactReadResult(
                        source_name=src_v.canonical_name,
                        source_class=src_v.entity_class,
                        predicate=e.predicate,
                        target_name=tgt_v.canonical_name,
                        target_class=tgt_v.entity_class,
                        fact_summary=e.fact_summary,
                        t_valid=e.t_valid,
                        t_invalid=e.t_invalid,
                        namespace=e.namespace,
                        source_episode_id=e.source_episode_id,
                        confidence=e.confidence,
                        reinforcement_count=e.reinforcement_count,
                    )
                )
                for vid in (e.source_vertex_id, e.target_vertex_id):
                    if vid not in visited_vertices:
                        visited_vertices.add(vid)
                        next_frontier.add(vid)
                if len(results) >= limit:
                    break
            frontier = next_frontier

        return results[:limit]

    async def record_reasoning_trace(
        self,
        ctx: PlatformTenantContext,
        run_id: str,
        session_id: str,
        steps: Sequence[ReasoningTraceStep],
    ) -> None:
        r = await self._get_redis()
        payload = {
            "run_id": run_id,
            "session_id": session_id,
            "app_name": ctx.app_name,
            "namespace": ctx.app_private_namespace,
            "steps": [s.model_dump(mode="json") for s in steps],
        }
        await r.hset(self._k_traces(ctx.tenant_id), run_id, json.dumps(payload))

    async def purge_episode_lineage(
        self, tenant_id: str, episode_ids: Sequence[str], user_id: str
    ) -> None:
        r = await self._get_redis()
        raw_edges = await r.hgetall(self._k_edges(tenant_id))
        to_delete: list[str] = []
        ep_set = set(episode_ids)
        for eid, raw in raw_edges.items():
            e = GraphEdgeMutation.model_validate_json(raw)
            if e.source_episode_id in ep_set or e.user_id == user_id:
                to_delete.append(eid)
        if to_delete:
            await r.hdel(self._k_edges(tenant_id), *to_delete)


class RedisUnifiedAdapter(
    RedisRegistryAdapter,
    RedisEpisodicAndSeedAdapter,
    RedisGraphAdapter,
):
    """
    Single-Stack Pure Redis Unified Adapter (`RegistryStoreAdapter` +
    `EpisodicAndSeedStoreAdapter` + `GraphStoreAdapter` + `PlaybookStoreAdapter`).
    """
