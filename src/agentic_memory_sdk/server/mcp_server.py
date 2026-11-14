from __future__ import annotations

from typing import Any

from ..governance import PlatformOntologyRegistrar
from ..models import (
    AppMemoryManifest,
    DomainPackSpec,
    GovernanceTier,
    ReasoningTraceStep,
    RelationRule,
)
from ..service import CompositeMemoryService, Event, types


class MemoryMcpServerTools:
    """
    Exposes the complete Data Plane + Control Plane of the Composite Memory Platform
    as MCP-compatible tools that can be mounted into a FastMCP/MCPServer instance,
    invoked via `/v1/mcp/call`, or passed to Google ADK's `McpToolset`.
    """

    TOOL_NAMES: list[str] = [
        "recall_context",
        "commit_episode",
        "ingest_memory",
        "record_decision_trace",
        "register_playbook",
        "register_domain_pack",
        "register_app_manifest",
        "evolve_app_ontology",
        "run_memory_maintenance",
        "inspect_platform_governance",
        "purge_user_lineage",
    ]

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
        Runs 3-Tier hybrid recall: Tier-3 Procedural Playbook matching + Tier-1 RRF seed search +
        Tier-2 multi-hop bi-temporal graph expansion + verbatim episodic memories.
        Supports point-in-time historical queries via `as_of_timestamp`.
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
        extraction_mode: str | None = None,
    ) -> dict[str, Any]:
        """
        Triggers PII scrubbing, episodic vector indexing, TBox-constrained extraction
        (or deferred Sleep-Time extraction when `extraction_mode='sleep_time'`),
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
            extraction_mode=extraction_mode,
        )
        return {
            "status": "committed",
            "app_name": app_name,
            "session_id": session_id,
            "extraction_mode": extraction_mode or "default",
        }

    async def ingest_memory(
        self,
        app_name: str,
        user_id: str,
        text: str,
        author: str = "user",
        session_id: str = "mcp_session",
        extraction_mode: str | None = None,
    ) -> dict[str, Any]:
        """Alias for `commit_episode` with `text` parameter."""
        return await self.commit_episode(
            app_name=app_name,
            user_id=user_id,
            raw_payload=text,
            author=author,
            session_id=session_id,
            extraction_mode=extraction_mode,
        )

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

    async def register_playbook(
        self,
        app_name: str,
        user_id: str,
        trigger_intent: str,
        workflow_steps: list[str],
        resolution_summary: str,
        share_in_domain: bool = True,
    ) -> dict[str, Any]:
        """Registers or reinforces a reusable Tier-3 Procedural Playbook via MCP."""
        pb = await self.service.register_playbook(
            app_name=app_name,
            user_id=user_id,
            trigger_intent=trigger_intent,
            workflow_steps=workflow_steps,
            resolution_summary=resolution_summary,
            share_in_domain=share_in_domain,
        )
        return {
            "status": "registered",
            "playbook_id": pb.playbook_id,
            "success_count": pb.success_count,
            "namespace": pb.namespace,
        }

    async def register_domain_pack(
        self,
        pack_id: str,
        classes: dict[str, str],
        relations: list[dict[str, Any]] | None = None,
        version: str = "1.0.0",
        review_with_steward: bool = True,
    ) -> dict[str, Any]:
        """Registers a Layer-1 Federated Domain Pack in the platform catalog via MCP."""
        rel_map = {}
        for r in relations or []:
            rule = RelationRule.model_validate(r)
            rel_map[rule.predicate] = rule
        spec = DomainPackSpec(
            pack_id=pack_id,
            domain_name=pack_id.split("@")[0],
            version=version,
            classes=classes,
            relations=rel_map,
        )
        if review_with_steward:
            refined, audit = self.registrar.review_and_refine_domain_pack(
                spec, auto_register=True
            )
            return {
                "status": "reviewed_and_registered",
                "pack_id": refined.pack_id,
                "classes": refined.classes,
                "audit_modifications": audit.modifications_summary,
            }
        self.registrar.register_domain_pack(spec)
        return {
            "status": "registered",
            "pack_id": spec.pack_id,
            "classes": spec.classes,
        }

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

    async def evolve_app_ontology(
        self,
        app_name: str,
        min_confidence_to_promote: float = 0.75,
    ) -> dict[str, Any]:
        """Promotes high-confidence quarantined overflow classes & predicates into the governed TBox."""
        audit = await self.service.evolve_app_ontology_from_overflow(
            app_name=app_name,
            min_confidence_to_promote=min_confidence_to_promote,
            registrar=self.registrar,
        )
        return {"status": "evolved", **audit.model_dump()}

    async def run_memory_maintenance(
        self,
        min_episodes_to_compact: int = 4,
        min_overflow_confidence: float = 0.55,
    ) -> dict[str, Any]:
        """Runs sleep-time batch extraction, session compaction, overflow edge pruning, and entity dossier synthesis."""
        stats = await self.service.run_memory_maintenance(
            min_episodes_to_compact=min_episodes_to_compact,
            min_overflow_confidence=min_overflow_confidence,
        )
        return {"status": "completed", **stats}

    async def inspect_platform_governance(
        self,
        app_name: str | None = None,
    ) -> dict[str, Any]:
        """Inspects registered Domain Packs, circuit-breaker health, audit log count, and optional app TBox."""
        out: dict[str, Any] = {
            "circuit_breaker_state": self.service.circuit_breaker.state,
            "registered_domain_packs": sorted(
                {p.pack_id for p in self.registrar._domain_packs.values()}
            ),
            "audit_log_count": len(self.registrar.audit_log),
        }
        if app_name:
            t_id, a_name = self.service._split_tenant_app(app_name)
            ctx = await self.service.registry.resolve_context(t_id, a_name, "inspector")
            out["app_governance"] = {
                "tenant_id": ctx.tenant_id,
                "app_name": ctx.app_name,
                "read_namespaces": ctx.read_namespaces,
                "write_namespace": ctx.write_namespace,
                "tbox_version": ctx.tbox.version,
                "classes": ctx.tbox.classes,
                "predicates": list(ctx.tbox.relations.keys()),
            }
        return out

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
    Creates a Model Context Protocol (MCP) Server using `MCPServer` (mcp 2.x)
    or `FastMCP` (mcp 1.x), exposing the full 11-tool Data Plane + Control Plane over
    stdio, SSE, or Streamable HTTP.
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
            "3-Tier hybrid recall: Procedural Playbooks + RRF seed search + multi-hop bi-temporal "
            "ontology graph traversal + episodic memory with point-in-time (`as_of_timestamp`) support."
        ),
    )(tools.recall_context)

    mcp.tool(
        name="commit_episode",
        description=(
            "Ingest an agent/user interaction into Episodic Memory and extract TBox-validated "
            "entities and bi-temporal facts (supports `extraction_mode='hot_path'|'sleep_time'`)."
        ),
    )(tools.commit_episode)

    mcp.tool(
        name="ingest_memory",
        description="Alias for `commit_episode` using `text` parameter.",
    )(tools.ingest_memory)

    mcp.tool(
        name="record_decision_trace",
        description=(
            "Record Tier 3 Procedural Reasoning Traces (`AgentRun -> ToolCall -> Entity`) "
            "for explainability and auditability."
        ),
    )(tools.record_decision_trace)

    mcp.tool(
        name="register_playbook",
        description="Register or reinforce a reusable Tier-3 Procedural Playbook (`trigger_intent -> workflow_steps`).",
    )(tools.register_playbook)

    mcp.tool(
        name="register_domain_pack",
        description="Register a Layer-1 Federated Domain Pack (`classes` & `relations`) in the platform catalog.",
    )(tools.register_domain_pack)

    mcp.tool(
        name="register_app_manifest",
        description=(
            "Register an agent app's Governance Tier (`APP_PRIVATE`, `DOMAIN_SHARED`, `GLOBAL_CORE`) "
            "and subscribed Domain Packs."
        ),
    )(tools.register_app_manifest)

    mcp.tool(
        name="evolve_app_ontology",
        description="Promote high-confidence quarantined overflow classes & predicates into the governed TBox.",
    )(tools.evolve_app_ontology)

    mcp.tool(
        name="run_memory_maintenance",
        description="Run sleep-time batch extraction, session compaction, overflow pruning, and entity dossier synthesis.",
    )(tools.run_memory_maintenance)

    mcp.tool(
        name="inspect_platform_governance",
        description="Inspect registered Domain Packs, circuit-breaker health, audit log count, and optional app TBox.",
    )(tools.inspect_platform_governance)

    mcp.tool(
        name="purge_user_lineage",
        description=(
            "Execute a cascading GDPR Right-to-be-Forgotten lineage deletion across "
            "Episodic and Bi-Temporal Graph stores."
        ),
    )(tools.purge_user_lineage)

    return mcp
