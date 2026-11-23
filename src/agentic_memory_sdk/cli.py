from __future__ import annotations

import argparse
import asyncio
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adk-memory",
        description="CLI for the Enterprise Multi-Tenant Composite Memory SDK & Platform Service",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. serve-api: Run centralized FastAPI Memory Platform Server
    p_api = sub.add_parser(
        "serve-api",
        help="Start the centralized FastAPI Memory Platform Server for 100s of ADK agents",
    )
    p_api.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_api.add_argument("--port", type=int, default=8095, help="Bind port (default: 8095)")
    p_api.add_argument(
        "--backend",
        choices=["docker", "memory"],
        default=None,
        help="Override MEMORY_BACKEND ('docker' for Postgres+Neo4j, 'memory' for InMemory)",
    )
    p_api.add_argument(
        "--enable-maintenance-worker",
        action="store_true",
        help="Enable built-in periodic Sleep-Time batch extraction & TTL maintenance scheduler",
    )
    p_api.add_argument(
        "--maintenance-interval-sec",
        type=float,
        default=None,
        help="Interval in seconds between background maintenance cycles (default: 300)",
    )
    p_api.add_argument(
        "--auth-enabled",
        action="store_true",
        help="Enforce JWT / API-Key AuthN + RBAC + tenant isolation (sets MEMORY_AUTH_ENABLED=true)",
    )
    p_api.add_argument(
        "--jwt-secret",
        default=None,
        help="HS256 JWT signing secret (sets MEMORY_JWT_SECRET and enables auth)",
    )
    p_api.add_argument(
        "--admin-api-key",
        default=None,
        help="Bootstrap PLATFORM_ADMIN API key (sets MEMORY_ADMIN_API_KEY and enables auth)",
    )
    p_api.add_argument(
        "--allow-insecure-public-bind",
        action="store_true",
        help="Allow binding to 0.0.0.0 without authentication (NOT recommended for production)",
    )

    # 2. serve-mcp: Run Model Context Protocol (MCP) Server
    p_mcp = sub.add_parser(
        "serve-mcp",
        help="Start the Model Context Protocol (MCP) Server exposing Composite Memory tools",
    )
    p_mcp.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    p_mcp.add_argument(
        "--backend",
        choices=["docker", "memory"],
        default=None,
        help="Override MEMORY_BACKEND ('docker' or 'memory')",
    )

    # 3. init-db: Initialize PostgreSQL + pgvector RLS schema and partitions
    p_init = sub.add_parser(
        "init-db",
        help="Initialize PostgreSQL + pgvector RLS tables, partitions, and HNSW indexes",
    )
    p_init.add_argument(
        "--dsn",
        default="postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform",
        help="PostgreSQL connection DSN",
    )
    p_init.add_argument(
        "--embedding-dim",
        type=int,
        default=1536,
        help="Vector embedding dimensionality (e.g. 768, 1536, 3072)",
    )
    p_init.add_argument(
        "--tenant",
        default="t1",
        help="Initial tenant ID to provision partitions for (default: t1)",
    )

    # 4. flush-outbox: Drain pending transactional outbox batch
    p_outbox = sub.add_parser(
        "flush-outbox",
        help="Drain pending graph mutations from the Transactional Outbox to the Graph Store",
    )
    p_outbox.add_argument("--batch-size", type=int, default=50, help="Batch size (default: 50)")

    args = parser.parse_args(argv)

    if args.command == "serve-api":
        import os
        import uvicorn

        if args.backend:
            os.environ["MEMORY_BACKEND"] = args.backend
        if args.enable_maintenance_worker:
            os.environ["ENABLE_MAINTENANCE_WORKER"] = "true"
        if args.maintenance_interval_sec is not None:
            os.environ["MAINTENANCE_INTERVAL_SEC"] = str(args.maintenance_interval_sec)
        if args.jwt_secret:
            os.environ["MEMORY_JWT_SECRET"] = args.jwt_secret
        if args.admin_api_key:
            os.environ["MEMORY_ADMIN_API_KEY"] = args.admin_api_key
        if args.auth_enabled or args.jwt_secret or args.admin_api_key:
            os.environ["MEMORY_AUTH_ENABLED"] = "true"
        from .security import AuthPolicyConfig

        resolved_auth = AuthPolicyConfig.from_env()
        if (
            args.host in ("0.0.0.0", "::")
            and not resolved_auth.enabled
            and not args.allow_insecure_public_bind
        ):
            print(
                "ERROR: Refusing to bind to public interface 0.0.0.0 without authentication enabled. "
                "Pass --jwt-secret / --admin-api-key (or set MEMORY_AUTH_ENABLED=true), "
                "bind to 127.0.0.1, or pass --allow-insecure-public-bind for local testing.",
                file=sys.stderr,
            )
            return 2
        uvicorn.run(
            "agentic_memory_sdk.server.main:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
        return 0

    if args.command == "serve-mcp":
        import os
        from .server.main import build_platform_service_from_env
        from .server.mcp_server import create_mcp_server

        if args.backend:
            os.environ["MEMORY_BACKEND"] = args.backend
        service, registrar = build_platform_service_from_env()
        mcp = create_mcp_server(service=service, registrar=registrar)
        mcp.run(transport=args.transport)
        return 0

    if args.command == "init-db":
        from .adapters.postgres_pgvector import PostgresPgVectorAdapter

        async def _run_init() -> None:
            adapter = PostgresPgVectorAdapter(dsn=args.dsn, min_pool_size=1, max_pool_size=2)
            try:
                await adapter.init_schema(
                    embedding_dim=args.embedding_dim, default_tenant=args.tenant
                )
                print(
                    f"Initialized PostgreSQL + pgvector RLS schema (dim={args.embedding_dim}, tenant={args.tenant})."
                )
            finally:
                await adapter.close()

        asyncio.run(_run_init())
        return 0

    if args.command == "flush-outbox":
        from .server.main import build_platform_service_from_env

        async def _run_flush() -> None:
            service, _ = build_platform_service_from_env()
            count = await service.process_outbox_batch(batch_size=args.batch_size)
            print(f"Flushed {count} outbox operations.")

        asyncio.run(_run_flush())
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
