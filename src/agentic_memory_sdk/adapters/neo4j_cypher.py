from __future__ import annotations

from typing import Any

from ..auth import CloudAuthProvider, EnterpriseTlsConfig
from .neptune_cypher import NeptuneOpenCypherAdapter


class Neo4jGraphAdapter(NeptuneOpenCypherAdapter):
    """
    Drop-in Neo4j Bolt replacement for `NeptuneOpenCypherAdapter`.
    Reuses the exact openCypher queries while rewriting Neptune's ``~id`` token to
    standard ``vertex_id`` and executing over `AsyncGraphDatabase`.
    Supports Basic Auth, Neo4j Enterprise / Aura SSO Bearer Token (`bearer_token`),
    dynamic `CloudAuthProvider`, and `EnterpriseTlsConfig`.
    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str] | None = None,
        database: str = "neo4j",
        auth_provider: CloudAuthProvider | None = None,
        bearer_token: str | None = None,
        tls_config: EnterpriseTlsConfig | None = None,
    ):
        self.uri = uri
        self.auth = auth
        self.database = database
        self.auth_provider = auth_provider
        self.bearer_token = bearer_token
        self.tls_config = tls_config
        self._driver: Any = None

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def _run_cypher(
        self, tenant_id: str, query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if self._driver is None:
            import neo4j
            from neo4j import AsyncGraphDatabase

            resolved_auth: Any = self.auth
            if self.auth_provider is not None:
                user, pwd = await self.auth_provider.get_credentials()
                resolved_auth = (user, pwd)
            elif self.bearer_token is not None:
                resolved_auth = neo4j.bearer_auth(self.bearer_token)

            driver_kwargs: dict[str, Any] = {"auth": resolved_auth}
            if (
                self.tls_config is not None
                and self.tls_config.enabled
                and not self.uri.startswith(("neo4j+s://", "bolt+s://", "neo4j+ssc://", "bolt+ssc://"))
            ):
                driver_kwargs["encrypted"] = True
            self._driver = AsyncGraphDatabase.driver(self.uri, **driver_kwargs)
        neo_query = query.replace("`~id`", "vertex_id")
        async with self._driver.session(database=self.database) as session:
            result = await session.run(neo_query, parameters=params)
            return [dict(record) async for record in result]
