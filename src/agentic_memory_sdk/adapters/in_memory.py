from __future__ import annotations

import math
from typing import Any, Sequence

from ..governance import EntityResolutionGuard
from ..models import (
    CompiledTBox,
    EpisodicRecord,
    GraphEdgeMutation,
    GraphFactReadResult,
    GraphVertexUpsert,
    OutboxOperation,
    PlatformTenantContext,
    PlaybookRecord,
    ReasoningTraceStep,
)
from ..ports import (
    EpisodicAndSeedStoreAdapter,
    GraphStoreAdapter,
    PlaybookStoreAdapter,
    RegistryStoreAdapter,
)


class InMemoryUnifiedAdapter(
    RegistryStoreAdapter,
    EpisodicAndSeedStoreAdapter,
    GraphStoreAdapter,
    PlaybookStoreAdapter,
):
    """
    Zero-dependency in-memory reference adapter enforcing the exact same
    RLS namespace isolation, hybrid RRF search, bi-temporal functional edge invalidation,
    reasoning trace storage, transactional outbox, procedural playbooks, and GDPR lineage purge rules.
    """

    def __init__(self):
        self.registry: dict[tuple[str, str], tuple[list[str], str, CompiledTBox]] = {}
        self.episodes: list[EpisodicRecord] = []
        self.seeds: dict[tuple[str, str], GraphVertexUpsert] = {}
        self.edges: list[dict[str, Any]] = []
        self.traces: dict[tuple[str, str], dict[str, Any]] = {}
        self.outbox: dict[str, OutboxOperation] = {}
        self.playbooks: dict[tuple[str, str], PlaybookRecord] = {}

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if (na and nb) else 0.0

    async def register_app(
        self,
        tenant_id: str,
        app_name: str,
        read_namespaces: list[str],
        write_namespace: str,
        tbox: CompiledTBox,
    ) -> None:
        self.registry[(tenant_id, app_name)] = (
            list(read_namespaces),
            write_namespace,
            tbox,
        )

    async def resolve_context(
        self, tenant_id: str, app_name: str, user_id: str
    ) -> PlatformTenantContext:
        app_ns = f"app:{tenant_id}:{app_name}"
        user_ns = f"user:{tenant_id}:{user_id}"
        if (tenant_id, app_name) not in self.registry:
            # Zero-config fallback: auto-provision Tier 0 isolated context + cross-app user namespace
            return PlatformTenantContext(
                tenant_id=tenant_id,
                app_name=app_name,
                user_id=user_id,
                app_private_namespace=app_ns,
                write_namespace=app_ns,
                read_namespaces=[app_ns, user_ns, f"global:{tenant_id}:core"],
                tbox=CompiledTBox(),
            )
        read_ns, write_ns, tbox = self.registry[(tenant_id, app_name)]
        merged_read_ns = list(dict.fromkeys([*read_ns, user_ns]))
        return PlatformTenantContext(
            tenant_id=tenant_id,
            app_name=app_name,
            user_id=user_id,
            app_private_namespace=app_ns,
            write_namespace=write_ns,
            read_namespaces=merged_read_ns,
            tbox=tbox,
        )

    async def resolve_existing_vertex_id(
        self,
        ctx: PlatformTenantContext,
        entity_class: str,
        canonical_name: str,
        embedding: list[float],
        similarity_threshold: float = 0.91,
    ) -> str | None:
        best_vid: str | None = None
        best_sim: float = -1.0
        for (t_id, v_id), seed in self.seeds.items():
            if t_id != ctx.tenant_id or seed.namespace not in ctx.read_namespaces:
                continue
            if seed.entity_class == entity_class:
                sim = self._cosine(seed.embedding, embedding)
                if EntityResolutionGuard.can_merge(
                    seed.canonical_name, canonical_name, sim, similarity_threshold
                ):
                    if (
                        EntityResolutionGuard.normalize_name(seed.canonical_name)
                        == EntityResolutionGuard.normalize_name(canonical_name)
                    ):
                        return v_id
                    if sim > best_sim:
                        best_sim = sim
                        best_vid = v_id
        return best_vid

    async def commit_episodes_and_outbox(
        self,
        ctx: PlatformTenantContext,
        episodes: Sequence[EpisodicRecord],
        vertex_upserts: Sequence[GraphVertexUpsert],
        outbox_op: OutboxOperation | None,
    ) -> None:
        allowed_write_ns = {
            ctx.write_namespace,
            ctx.app_private_namespace,
            ctx.user_private_namespace,
        }
        for ep in episodes:
            if ep.tenant_id != ctx.tenant_id or ep.app_name != ctx.app_name:
                raise PermissionError("Episodic RLS violation: mismatched tenant_id or app_name")
            self.episodes.append(ep)

        for v in vertex_upserts:
            if v.tenant_id != ctx.tenant_id or v.namespace not in allowed_write_ns:
                raise PermissionError(
                    f"Seed write RLS violation: {v.namespace} not in {allowed_write_ns}"
                )
            self.seeds[(ctx.tenant_id, v.vertex_id)] = v

        if outbox_op is not None:
            self.outbox[outbox_op.outbox_id] = outbox_op

    async def hybrid_seed_search(
        self,
        ctx: PlatformTenantContext,
        query_text: str,
        query_embedding: list[float],
        limit: int = 8,
    ) -> list[GraphVertexUpsert]:
        candidates = [
            s
            for (t_id, _), s in self.seeds.items()
            if t_id == ctx.tenant_id and s.namespace in ctx.read_namespaces
        ]
        if not candidates:
            return []

        # Reciprocal Rank Fusion (RRF) combining vector cosine rank and lexical BM25-like overlap rank
        vec_ranked = sorted(
            candidates,
            key=lambda s: self._cosine(s.embedding, query_embedding),
            reverse=True,
        )
        q_terms = {
            w.strip(".,!?;:()[]\"'").lower()
            for w in query_text.split()
            if w.strip(".,!?;:()[]\"'")
        }
        lex_ranked = sorted(
            candidates,
            key=lambda s: len(
                q_terms.intersection(
                    {
                        w.strip(".,!?;:()[]\"'").lower()
                        for w in (s.canonical_name + " " + s.summary).split()
                    }
                )
            ),
            reverse=True,
        )

        rrf_scores: dict[str, float] = {}
        by_id: dict[str, GraphVertexUpsert] = {}
        for rank_idx, s in enumerate(vec_ranked, start=1):
            rrf_scores[s.vertex_id] = rrf_scores.get(s.vertex_id, 0.0) + 1.0 / (60 + rank_idx)
            by_id[s.vertex_id] = s
        for rank_idx, s in enumerate(lex_ranked, start=1):
            rrf_scores[s.vertex_id] = rrf_scores.get(s.vertex_id, 0.0) + 1.0 / (60 + rank_idx)
            by_id[s.vertex_id] = s

        sorted_ids = sorted(rrf_scores.keys(), key=lambda vid: rrf_scores[vid], reverse=True)
        return [by_id[vid] for vid in sorted_ids[:limit]]

    async def search_episodes(
        self,
        ctx: PlatformTenantContext,
        query_embedding: list[float],
        limit: int = 4,
    ) -> list[EpisodicRecord]:
        scoped = [
            e
            for e in self.episodes
            if e.tenant_id == ctx.tenant_id
            and e.app_name == ctx.app_name
            and e.user_id == ctx.user_id
        ]
        scoped.sort(key=lambda e: self._cosine(e.embedding, query_embedding), reverse=True)
        return scoped[:limit]

    async def fetch_pending_outbox(self, batch_size: int = 25) -> list[OutboxOperation]:
        pending = [op for op in self.outbox.values() if op.attempts < 5]
        return pending[:batch_size]

    async def mark_outbox_processed(self, outbox_id: str) -> None:
        self.outbox.pop(outbox_id, None)

    async def mark_outbox_failed(self, outbox_id: str, error_msg: str) -> None:
        if outbox_id in self.outbox:
            self.outbox[outbox_id].attempts += 1

    async def upsert_vertices_and_bitemporal_edges(
        self,
        tenant_id: str,
        vertices: Sequence[GraphVertexUpsert],
        edges: Sequence[GraphEdgeMutation],
    ) -> None:
        for v in vertices:
            existing = self.seeds.get((tenant_id, v.vertex_id))
            if existing is None:
                self.seeds[(tenant_id, v.vertex_id)] = v
            else:
                # Preserve existing embedding if outbox stripped it
                merged = v.model_copy()
                if not merged.embedding and existing.embedding:
                    merged.embedding = existing.embedding
                self.seeds[(tenant_id, v.vertex_id)] = merged

        for e in edges:
            if e.is_functional:
                for existing in self.edges:
                    if (
                        existing["tenant_id"] == tenant_id
                        and existing["source_vertex_id"] == e.source_vertex_id
                        and existing["predicate"] == e.predicate
                        and existing["namespace"] == e.namespace
                        and existing["t_invalid"] == "ACTIVE"
                        and existing["target_vertex_id"] != e.target_vertex_id
                    ):
                        existing["t_invalid"] = e.t_valid

            # Idempotent check on (source_vertex_id, predicate, target_vertex_id, namespace, ACTIVE)
            # HippoRAG-inspired Associative Reinforcement: re-confirming an active edge boosts salience!
            already_active = False
            for existing in self.edges:
                if (
                    existing["tenant_id"] == tenant_id
                    and existing["source_vertex_id"] == e.source_vertex_id
                    and existing["target_vertex_id"] == e.target_vertex_id
                    and existing["predicate"] == e.predicate
                    and existing["namespace"] == e.namespace
                    and existing["t_invalid"] == "ACTIVE"
                ):
                    existing["fact_summary"] = e.fact_summary
                    existing["reinforcement_count"] = int(existing.get("reinforcement_count", 1)) + 1
                    existing["confidence"] = min(
                        1.0, max(float(existing.get("confidence", 0.9)), e.confidence) + 0.03
                    )
                    already_active = True
                    break

            if not already_active:
                self.edges.append(
                    {**e.model_dump(), "t_invalid": "ACTIVE", "reinforcement_count": 1}
                )

    async def expand_active_subgraph(
        self,
        ctx: PlatformTenantContext,
        seed_vertex_ids: Sequence[str],
        as_of_iso: str | None = None,
        max_hops: int = 2,
        limit: int = 25,
    ) -> list[GraphFactReadResult]:
        # Letta-inspired Core Persona Fast-Path: automatically include the user's cross-app
        # `user:{tenant}:{user_id}` preference vertices in the initial frontier even if the
        # current query vector is orthogonal to their persona preferences!
        user_persona_vids = [
            vid
            for (t_id, vid), s in self.seeds.items()
            if t_id == ctx.tenant_id and s.namespace == ctx.user_private_namespace
        ]
        combined_seeds = list(dict.fromkeys([*seed_vertex_ids, *user_persona_vids]))
        if not combined_seeds:
            return []
        frontier = set(combined_seeds)
        visited_vertices = set(combined_seeds)
        seen_edge_keys: set[tuple[str, str, str, str, str]] = set()
        out: list[GraphFactReadResult] = []

        for _hop in range(max(1, max_hops)):
            if not frontier or len(out) >= limit:
                break
            next_frontier: set[str] = set()
            for r in self.edges:
                if r["tenant_id"] != ctx.tenant_id or r["namespace"] not in ctx.read_namespaces:
                    continue
                src_id = r["source_vertex_id"]
                tgt_id = r["target_vertex_id"]
                if src_id not in frontier and tgt_id not in frontier:
                    continue
                if as_of_iso is None:
                    if r["t_invalid"] != "ACTIVE":
                        continue
                else:
                    if not (
                        r["t_valid"] <= as_of_iso
                        and (r["t_invalid"] == "ACTIVE" or r["t_invalid"] > as_of_iso)
                    ):
                        continue

                edge_key = (src_id, r["predicate"], tgt_id, r["namespace"], r["t_valid"])
                if edge_key in seen_edge_keys:
                    continue

                src_node = self.seeds.get((ctx.tenant_id, src_id))
                tgt_node = self.seeds.get((ctx.tenant_id, tgt_id))
                if src_node and tgt_node:
                    if (
                        src_node.namespace not in ctx.read_namespaces
                        or tgt_node.namespace not in ctx.read_namespaces
                    ):
                        continue
                    seen_edge_keys.add(edge_key)
                    out.append(
                        GraphFactReadResult(
                            source_name=src_node.canonical_name,
                            source_class=src_node.entity_class,
                            predicate=r["predicate"],
                            target_name=tgt_node.canonical_name,
                            target_class=tgt_node.entity_class,
                            fact_summary=r["fact_summary"],
                            t_valid=r["t_valid"],
                            t_invalid=r["t_invalid"],
                            namespace=r["namespace"],
                            source_episode_id=r.get("source_episode_id", ""),
                            confidence=r["confidence"],
                            reinforcement_count=int(r.get("reinforcement_count", 1)),
                        )
                    )
                    if src_id not in visited_vertices:
                        next_frontier.add(src_id)
                    if tgt_id not in visited_vertices:
                        next_frontier.add(tgt_id)
                    if len(out) >= limit:
                        break
            visited_vertices.update(next_frontier)
            frontier = next_frontier

        return out[:limit]

    async def record_reasoning_trace(
        self,
        ctx: PlatformTenantContext,
        run_id: str,
        session_id: str,
        steps: Sequence[ReasoningTraceStep],
    ) -> None:
        self.traces[(ctx.tenant_id, run_id)] = {
            "tenant_id": ctx.tenant_id,
            "app_name": ctx.app_name,
            "session_id": session_id,
            "namespace": ctx.app_private_namespace,
            "steps": [s.model_dump() for s in steps],
        }

    async def purge_user_data(self, ctx: PlatformTenantContext) -> list[str]:
        purged = [
            e.episode_id
            for e in self.episodes
            if e.user_id == ctx.user_id and e.tenant_id == ctx.tenant_id
        ]
        purged_set = set(purged)
        self.episodes = [e for e in self.episodes if e.episode_id not in purged_set]
        return purged

    async def purge_episode_lineage(
        self, tenant_id: str, episode_ids: Sequence[str], user_id: str
    ) -> None:
        ep_set = set(episode_ids)
        self.edges = [
            r
            for r in self.edges
            if not (
                r["tenant_id"] == tenant_id
                and (r["source_episode_id"] in ep_set or r["user_id"] == user_id)
            )
        ]

    async def upsert_playbook(self, playbook: PlaybookRecord) -> PlaybookRecord:
        for (t_id, pb_id), existing in list(self.playbooks.items()):
            if (
                t_id == playbook.tenant_id
                and existing.namespace == playbook.namespace
                and existing.trigger_intent.strip().lower()
                == playbook.trigger_intent.strip().lower()
            ):
                playbook.playbook_id = pb_id
                playbook.success_count = existing.success_count + 1
                self.playbooks[(t_id, pb_id)] = playbook
                return playbook
        self.playbooks[(playbook.tenant_id, playbook.playbook_id)] = playbook
        return playbook

    async def match_playbooks(
        self,
        tenant_id: str,
        read_namespaces: list[str],
        query_embedding: list[float],
        similarity_threshold: float = 0.85,
        limit: int = 2,
    ) -> list[PlaybookRecord]:
        scored: list[tuple[float, PlaybookRecord]] = []
        for (t_id, _), pb in self.playbooks.items():
            if t_id == tenant_id and pb.namespace in read_namespaces:
                sim = self._cosine(pb.embedding, query_embedding)
                if sim >= similarity_threshold:
                    scored.append((sim, pb))
        scored.sort(key=lambda item: (item[0], item[1].success_count), reverse=True)
        return [pb for _, pb in scored[:limit]]
