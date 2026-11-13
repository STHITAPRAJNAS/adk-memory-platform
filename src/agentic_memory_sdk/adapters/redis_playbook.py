from __future__ import annotations

import math
from typing import Any

from ..auth import CloudAuthProvider, EnterpriseTlsConfig
from ..models import PlaybookRecord
from ..ports import PlaybookStoreAdapter


class RedisPlaybookAndWorkingAdapter(PlaybookStoreAdapter):
    """
    Redis Iris-inspired Semantic Playbook Cache & Working Memory TTL Buffer Adapter.
    - When `redis_url` is provided (and `redis.asyncio` is installed), persists playbooks
      and working buffers in Redis hashes/keys with configurable TTLs for sub-5ms lookup.
    - Supports Enterprise Cloud Auth (`auth_provider`: AWS ElastiCache/MemoryDB IAM,
      Azure Managed Redis Entra ID, GCP Memorystore IAM, or Vault secrets) and mTLS (`tls_config`).
    - Includes a built-in zero-dependency in-memory store fallback so the exact same
      Semantic Playbook caching API works in local/unit tests without a Redis server.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        key_prefix: str = "adk_memory:playbook",
        default_ttl_sec: int = 86400 * 30,
        auth_provider: CloudAuthProvider | None = None,
        tls_config: EnterpriseTlsConfig | None = None,
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl_sec = default_ttl_sec
        self.auth_provider = auth_provider
        self.tls_config = tls_config
        self._redis: Any = None
        self._local_playbooks: dict[tuple[str, str], PlaybookRecord] = {}

    async def _get_redis(self):
        if not self.redis_url:
            return None
        if self._redis is None:
            import redis.asyncio as aioredis  # type: ignore

            conn_kwargs: dict[str, Any] = {"decode_responses": True}
            if self.auth_provider is not None:
                user, pwd = await self.auth_provider.get_credentials()
                if user and user != "default":
                    conn_kwargs["username"] = user
                conn_kwargs["password"] = pwd
            if self.tls_config is not None and self.tls_config.enabled:
                conn_kwargs["ssl"] = True
                if self.tls_config.ca_file:
                    conn_kwargs["ssl_ca_certs"] = self.tls_config.ca_file
                if self.tls_config.client_cert_file:
                    conn_kwargs["ssl_certfile"] = self.tls_config.client_cert_file
                if self.tls_config.client_key_file:
                    conn_kwargs["ssl_keyfile"] = self.tls_config.client_key_file

            self._redis = aioredis.from_url(self.redis_url, **conn_kwargs)
        return self._redis

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if (na and nb) else 0.0

    async def upsert_playbook(self, playbook: PlaybookRecord) -> PlaybookRecord:
        """
        Caches or reinforces a proven Agent Playbook (`trigger_intent -> workflow_steps -> resolution_summary`).
        """
        r = await self._get_redis()
        if r is not None:
            rkey = f"{self.key_prefix}:{playbook.tenant_id}"
            existing_raw = await r.hget(rkey, playbook.playbook_id)
            if existing_raw:
                existing = PlaybookRecord.model_validate_json(existing_raw)
                playbook.success_count = existing.success_count + 1
            else:
                all_raw = await r.hgetall(rkey)
                for pb_id, raw_json in all_raw.items():
                    cand = PlaybookRecord.model_validate_json(raw_json)
                    if (
                        cand.namespace == playbook.namespace
                        and cand.trigger_intent.strip().lower()
                        == playbook.trigger_intent.strip().lower()
                    ):
                        playbook.playbook_id = pb_id
                        playbook.success_count = cand.success_count + 1
                        break
            await r.hset(rkey, playbook.playbook_id, playbook.model_dump_json())
            await r.expire(rkey, self.default_ttl_sec)
            return playbook

        # In-memory fallback with identical semantics
        for (t_id, pb_id), existing in list(self._local_playbooks.items()):
            if (
                t_id == playbook.tenant_id
                and existing.namespace == playbook.namespace
                and existing.trigger_intent.strip().lower()
                == playbook.trigger_intent.strip().lower()
            ):
                playbook.playbook_id = pb_id
                playbook.success_count = existing.success_count + 1
                self._local_playbooks[(t_id, pb_id)] = playbook
                return playbook

        self._local_playbooks[(playbook.tenant_id, playbook.playbook_id)] = playbook
        return playbook

    async def match_playbooks(
        self,
        tenant_id: str,
        read_namespaces: list[str],
        query_embedding: list[float],
        similarity_threshold: float = 0.85,
        limit: int = 2,
    ) -> list[PlaybookRecord]:
        """
        Sub-5ms semantic lookup for proven Agent Playbooks matching the user's intent vector.
        Strictly isolated by `tenant_id` and `read_namespaces`.
        """
        candidates: list[PlaybookRecord] = []
        r = await self._get_redis()
        if r is not None:
            rkey = f"{self.key_prefix}:{tenant_id}"
            raw_map = await r.hgetall(rkey)
            for raw_json in raw_map.values():
                pb = PlaybookRecord.model_validate_json(raw_json)
                if pb.namespace in read_namespaces:
                    candidates.append(pb)
        else:
            candidates = [
                pb
                for (t_id, _), pb in self._local_playbooks.items()
                if t_id == tenant_id and pb.namespace in read_namespaces
            ]

        scored: list[tuple[float, PlaybookRecord]] = []
        for pb in candidates:
            sim = self._cosine(pb.embedding, query_embedding)
            if sim >= similarity_threshold:
                scored.append((sim, pb))

        scored.sort(key=lambda item: (item[0], item[1].success_count), reverse=True)
        return [pb for _, pb in scored[:limit]]

    async def close(self) -> None:
        if self._redis is not None:
            if hasattr(self._redis, "aclose"):
                await self._redis.aclose()
            else:
                await self._redis.close()
            self._redis = None
