from __future__ import annotations

from typing import Any

from ..governance import PlatformOntologyRegistrar
from ..models import AppMemoryManifest, GovernanceTier, ReasoningTraceStep
from ..service import CompositeMemoryService, Event, types


class MemoryMcpServerTools:
    """
    Exposes governed Context Graph & Memory operations as MCP-compatible tools
    that can be mounted into a FastMCP server or passed directly into Google ADK's `McpToolset` / `FunctionTool`.
    """

    def __init__(
        self,
        service: CompositeMemoryService,
        registrar: PlatformOntologyRegistrar | None = None,
    ):
        self.service = service
        self.registrar = registrar or PlatformOntologyRegistrar(service.registry)

    async def recall_context(
        self,
        app_name: str,
        user_id: str,
        query: str,
        as_of_timestamp: str | None = None,
        max_hops: int = 2,
    ) -> dict[str, Any]:
        """
        Runs hybrid RRF seed search + multi-hop bi-temporal graph expansion and returns a token-budgeted
        subgraph slice + verbatim episodic memories. Supports point-in-time historical queries via `as_of_timestamp`.
        """
        resp = await self.service.search_memory(
            app_name=app_name,
            user_id=user_id,
            query=query,
            as_of_iso=as_of_timestamp,
            max_hops=max_hops,
        )
        blocks = []
        for m in resp.memories:
            if getattr(m, "content", None) and getattr(m.content, "parts", None):
                text = "\n".join(p.text for p in m.content.parts if getattr(p, "text", None))
                blocks.append({"author": m.author, "timestamp": m.timestamp, "text": text})
        return {
            "query": query,
            "as_of": as_of_timestamp,
            "max_hops": max_hops,
            "memories": blocks,
        }

    async def commit_episode(
        self,
        app_name: str,
        user_id: str,
        raw_payload: str,
        author: str = "agent",
        session_id: str = "mcp_session",
    ) -> dict[str, Any]:
        """
        Triggers PII scrubbing, episodic vector indexing, TBox-constrained extraction,
        two-bucket validation, and bi-temporal edge updates.
        """
        ev = Event(
            author=author,
            content=types.Content(role="user", parts=[types.Part.from_text(text=raw_payload)]),
        )
        await self.service.add_events_to_memory(
            app_name=app_name,
            user_id=user_id,
            events=[ev],
            session_id=session_id,
        )
        return {"status": "committed", "app_name": app_name, "session_id": session_id}

    async def record_decision_trace(
        self,
        app_name: str,
        user_id: str,
        run_id: str,
        session_id: str,
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Writes an agent's tool-execution trace to the Tier 3 Procedural Reasoning subgraph."""
        parsed_steps = [ReasoningTraceStep.model_validate(s) for s in steps]
        await self.service.record_decision_trace(
            app_name=app_name,
            user_id=user_id,
            run_id=run_id,
            session_id=session_id,
            steps=parsed_steps,
        )
        return {"status": "recorded", "run_id": run_id, "step_count": len(parsed_steps)}

    async def register_app_manifest(
        self,
        tenant_id: str,
        app_name: str,
        governance_tier: str = "DOMAIN_SHARED",
        subscribed_domain_packs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Registers an ADK agent app's Governance Tier and Domain Pack subscriptions via MCP."""
        manifest = AppMemoryManifest(
            tenant_id=tenant_id,
            app_name=app_name,
            governance_tier=GovernanceTier(governance_tier),
            subscribed_domain_packs=subscribed_domain_packs or [],
        )
        tbox = await self.registrar.register_adk_app(manifest)
        return {
            "status": "registered",
            "tenant_id": tenant_id,
            "app_name": app_name,
            "governance_tier": governance_tier,
            "compiled_classes": list(tbox.classes.keys()),
            "compiled_predicates": list(tbox.relations.keys()),
        }

    async def purge_user_lineage(
        self,
        app_name: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Executes a cascading GDPR Right-to-be-Forgotten purge across Episodic and Graph tiers."""
        purged_ids = await self.service.purge_user_lineage(
            app_name=app_name, user_id=user_id
        )
        return {
            "status": "purged",
            "app_name": app_name,
            "user_id": user_id,
            "purged_episode_ids": purged_ids,
        }


def create_mcp_server(
    service: CompositeMemoryService,
    registrar: PlatformOntologyRegistrar | None = None,
    name: str = "adk-composite-memory",
) -> Any:
    """
    Creates a standalone Model Context Protocol (MCP) Server using `MCPServer` (mcp 2.x)
    or `FastMCP` (mcp 1.x), exposing the 5 governed Composite Memory tools over
    stdio, SSE, or Streamable HTTP.
    Can be consumed by Google ADK (`McpToolset`), Claude Desktop, Cursor, or any MCP client.
    """
    try:
        from mcp.server.mcpserver import MCPServer as _MCPServerClass
    except ImportError:
        from mcp.server.fastmcp import FastMCP as _MCPServerClass

    mcp = _MCPServerClass(name)
    tools = MemoryMcpServerTools(service=service, registrar=registrar)

    mcp.tool(
        name="recall_context",
        description=(
            "Hybrid RRF seed search + multi-hop bi-temporal ontology graph traversal + "
            "episodic memory retrieval with point-in-time (`as_of_timestamp`) support."
        ),
    )(tools.recall_context)

    mcp.tool(
        name="commit_episode",
        description=(
            "Ingest an agent/user interaction into Episodic Memory and extract TBox-validated "
            "entities and bi-temporal facts into the Knowledge Graph."
        ),
    )(tools.commit_episode)

    mcp.tool(
        name="record_decision_trace",
        description=(
            "Record Tier 3 Procedural Reasoning Traces (`AgentRun -> ToolCall -> Entity`) "
            "for explainability and auditability."
        ),
    )(tools.record_decision_trace)

    mcp.tool(
        name="register_app_manifest",
        description=(
            "Register an agent app's Governance Tier (`APP_PRIVATE`, `DOMAIN_SHARED`, `GLOBAL_CORE`) "
            "and subscribed Domain Packs."
        ),
    )(tools.register_app_manifest)

    mcp.tool(
        name="purge_user_lineage",
        description=(
            "Execute a cascading GDPR Right-to-be-Forgotten lineage deletion across "
            "Episodic and Bi-Temporal Graph stores."
        ),
    )(tools.purge_user_lineage)

    return mcp
