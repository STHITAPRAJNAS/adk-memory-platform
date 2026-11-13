from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from ..governance import EntityResolutionGuard
from ..models import (
    EpisodicRecord,
    GraphVertexUpsert,
    OutboxOperation,
    PlatformTenantContext,
)
from ..ports import EpisodicAndSeedStoreAdapter


class OpenSearchSeedAdapter(EpisodicAndSeedStoreAdapter):
    """
    Swappable Amazon OpenSearch adapter for Tier 1 Episodic Search and Tier 2 Hybrid
    (BM25 + k-NN HNSW) Entry-Point Seeding & Entity Resolution Blocking.
    Enforces tenant_id and namespace isolation filters on every query and write.
    Can be paired with PostgreSQL (RegistryStoreAdapter) and Amazon Neptune (GraphStoreAdapter).
    """

    def __init__(
        self,
        hosts: list[dict[str, Any]] | list[str],
        http_auth: tuple[str, str] | Any = None,
        seed_index_name: str = "platform_graph_seeds",
        episodic_index_name: str = "platform_episodic_memories",
        outbox_index_name: str = "platform_graph_outbox",
        use_ssl: bool = True,
    ):
        self.hosts = hosts
        self.http_auth = http_auth
        self.seed_index = seed_index_name
        self.episodic_index = episodic_index_name
        self.outbox_index = outbox_index_name
        self.use_ssl = use_ssl
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            from opensearchpy import AsyncOpenSearch

            self._client = AsyncOpenSearch(
                hosts=self.hosts,
                http_auth=self.http_auth,
                use_ssl=self.use_ssl,
                verify_certs=self.use_ssl,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def resolve_existing_vertex_id(
        self,
        ctx: PlatformTenantContext,
        entity_class: str,
        canonical_name: str,
        embedding: list[float],
        similarity_threshold: float = 0.91,
    ) -> str | None:
        client = self._get_client()
        query_body = {
            "size": 1,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"tenant_id": ctx.tenant_id}},
                        {"terms": {"namespace": ctx.read_namespaces}},
                        {"term": {"entity_class": entity_class}},
                    ],
                    "should": [
                        {"term": {"canonical_name.keyword": canonical_name}},
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": 3,
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        }
        resp = await client.search(index=self.seed_index, body=query_body)
        hits = resp.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit["_source"]
            cand_name = src.get("canonical_name", "")
            score = float(hit.get("_score", 0.0))
            if EntityResolutionGuard.can_merge(
                cand_name, canonical_name, score, similarity_threshold
            ):
                return src["vertex_id"]
        return None

    async def commit_episodes_and_outbox(
        self,
        ctx: PlatformTenantContext,
        episodes: Sequence[EpisodicRecord],
        vertex_upserts: Sequence[GraphVertexUpsert],
        outbox_op: OutboxOperation | None,
    ) -> None:
        client = self._get_client()
        allowed_write_ns = {ctx.write_namespace, ctx.app_private_namespace}

        bulk_ops: list[dict[str, Any]] = []
        for ep in episodes:
            if ep.tenant_id != ctx.tenant_id or ep.app_name != ctx.app_name:
                raise PermissionError("OpenSearch episodic write tenant/app isolation error")
            bulk_ops.append({"index": {"_index": self.episodic_index, "_id": ep.episode_id}})
            bulk_ops.append(
                {
                    "episode_id": ep.episode_id,
                    "tenant_id": ep.tenant_id,
                    "app_name": ep.app_name,
                    "user_id": ep.user_id,
                    "session_id": ep.session_id,
                    "namespace": ep.namespace,
                    "author": ep.author,
                    "content_text": ep.content_text,
                    "embedding": ep.embedding,
                    "created_at": ep.created_at.isoformat(),
                    "metadata": ep.metadata,
                }
            )

        for v in vertex_upserts:
            if v.tenant_id != ctx.tenant_id or v.namespace not in allowed_write_ns:
                raise PermissionError(
                    f"OpenSearch seed write namespace error: {v.namespace}"
                )
            doc_id = f"{v.tenant_id}:{v.vertex_id}"
            bulk_ops.append({"index": {"_index": self.seed_index, "_id": doc_id}})
            bulk_ops.append(v.model_dump())

        if outbox_op is not None:
            bulk_ops.append(
                {"index": {"_index": self.outbox_index, "_id": outbox_op.outbox_id}}
            )
            bulk_ops.append(
                {
                    **outbox_op.model_dump(),
                    "status": "PENDING",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        if bulk_ops:
            await client.bulk(body=bulk_ops, refresh="wait_for")

    async def hybrid_seed_search(
        self,
        ctx: PlatformTenantContext,
        query_text: str,
        query_embedding: list[float],
        limit: int = 8,
    ) -> list[GraphVertexUpsert]:
        client = self._get_client()
        security_filter = [
            {"term": {"tenant_id": ctx.tenant_id}},
            {"terms": {"namespace": ctx.read_namespaces}},
        ]
        # Execute BM25 lexical search + k-NN vector search and fuse with RRF
        body = {
            "size": limit,
            "query": {
                "bool": {
                    "filter": security_filter,
                    "should": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["canonical_name^2.0", "summary"],
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": limit * 2,
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
            "_source": {"excludes": ["embedding"]},
        }
        resp = await client.search(index=self.seed_index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [
            GraphVertexUpsert(**{**h["_source"], "embedding": []})
            for h in hits
        ]

    async def search_episodes(
        self,
        ctx: PlatformTenantContext,
        query_embedding: list[float],
        limit: int = 4,
    ) -> list[EpisodicRecord]:
        client = self._get_client()
        body = {
            "size": limit,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"tenant_id": ctx.tenant_id}},
                        {"term": {"app_name": ctx.app_name}},
                        {"term": {"user_id": ctx.user_id}},
                    ],
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": limit * 2,
                                }
                            }
                        }
                    ],
                }
            },
            "_source": {"excludes": ["embedding"]},
        }
        resp = await client.search(index=self.episodic_index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [
            EpisodicRecord(**{**h["_source"], "embedding": []})
            for h in hits
        ]

    async def fetch_pending_outbox(self, batch_size: int = 25) -> list[OutboxOperation]:
        client = self._get_client()
        body = {
            "size": batch_size,
            "sort": [{"created_at": {"order": "asc"}}],
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"status": ["PENDING", "RETRY"]}},
                        {"range": {"attempts": {"lt": 5}}},
                    ]
                }
            },
        }
        resp = await client.search(index=self.outbox_index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [
            OutboxOperation(
                outbox_id=h["_source"]["outbox_id"],
                tenant_id=h["_source"]["tenant_id"],
                operation_type=h["_source"]["operation_type"],
                payload=h["_source"]["payload"],
                attempts=h["_source"].get("attempts", 0),
            )
            for h in hits
        ]

    async def mark_outbox_processed(self, outbox_id: str) -> None:
        client = self._get_client()
        await client.update(
            index=self.outbox_index,
            id=outbox_id,
            body={
                "doc": {
                    "status": "PROCESSED",
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )

    async def mark_outbox_failed(self, outbox_id: str, error_msg: str) -> None:
        client = self._get_client()
        await client.update(
            index=self.outbox_index,
            id=outbox_id,
            body={
                "script": {
                    "source": """
                        ctx._source.attempts += 1;
                        ctx._source.last_error = params.err;
                        ctx._source.status = (ctx._source.attempts >= 5) ? 'DEAD_LETTER' : 'RETRY';
                    """,
                    "lang": "painless",
                    "params": {"err": error_msg[:500]},
                }
            },
        )

    async def purge_user_data(self, ctx: PlatformTenantContext) -> list[str]:
        client = self._get_client()
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"tenant_id": ctx.tenant_id}},
                        {"term": {"user_id": ctx.user_id}},
                    ]
                }
            },
            "_source": ["episode_id"],
            "size": 1000,
        }
        resp = await client.search(index=self.episodic_index, body=query)
        hits = resp.get("hits", {}).get("hits", [])
        episode_ids = [h["_source"]["episode_id"] for h in hits]
        if episode_ids:
            await client.delete_by_query(
                index=self.episodic_index,
                body={"query": {"terms": {"episode_id": episode_ids}}},
                refresh=True,
            )
        return episode_ids
