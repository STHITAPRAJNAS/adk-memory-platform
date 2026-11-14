from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..governance import PlatformOntologyRegistrar
from ..models import (
    AppMemoryManifest,
    DomainPackSpec,
    ReasoningTraceStep,
)
from ..ontology_loader import DomainPackLoader
from ..service import CompositeMemoryService, Event, types
from .mcp_server import MemoryMcpServerTools, create_mcp_server


class McpToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TurtleDomainPackRequest(BaseModel):
    pack_id: str
    turtle_content: str


class EventTurnPayload(BaseModel):
    author: str = "user"
    text: str


class IngestEventsRequest(BaseModel):
    app_name: str
    user_id: str
    session_id: str = "default_session"
    events: list[EventTurnPayload]
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    extraction_mode: str | None = None
    defer_extraction: bool | None = None


class SearchMemoryRequest(BaseModel):
    app_name: str
    user_id: str
    query: str
    as_of_iso: str | None = None
    max_hops: int = 2


class RecordTraceRequest(BaseModel):
    app_name: str
    user_id: str
    run_id: str
    session_id: str
    steps: list[ReasoningTraceStep]


class PurgeLineageRequest(BaseModel):
    app_name: str
    user_id: str


class RegisterPlaybookRequest(BaseModel):
    app_name: str
    user_id: str
    trigger_intent: str
    workflow_steps: list[str]
    resolution_summary: str
    share_in_domain: bool = True


class EvolveOntologyRequest(BaseModel):
    app_name: str
    min_confidence_to_promote: float = 0.75


def create_memory_platform_app(
    service: CompositeMemoryService,
    registrar: PlatformOntologyRegistrar | None = None,
    enable_outbox_worker: bool = True,
    outbox_interval_sec: float = 2.0,
    mount_mcp: bool = True,
) -> FastAPI:
    """
    Creates the Unified Multi-Tenant Memory-as-a-Service FastAPI + MCP application.
    Serves BOTH:
      1. FastAPI REST Control & Data Plane (`/v1/...`) for `RemoteCompositeMemoryService`
      2. Model Context Protocol (MCP) Server (`/mcp/sse`, `/mcp/http`, `/v1/mcp/tools`, `/v1/mcp/call`)
    from a single ASGI process / Kubernetes Pod.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if enable_outbox_worker:
            await service.start_outbox_worker(interval_sec=outbox_interval_sec)
        try:
            yield
        finally:
            if enable_outbox_worker:
                await service.stop_outbox_worker()

    app = FastAPI(
        title="ADK Enterprise Composite Memory Platform API",
        version="0.1.0",
        description=(
            "Domain-Agnostic Multi-Tenant Memory Service combining Episodic Memory, "
            "Stratified Ontology (TBox), Bi-Temporal Graph KB (ABox), and Reasoning Traces."
        ),
        lifespan=lifespan,
    )
    ont_registrar = registrar or PlatformOntologyRegistrar(service.registry)

    @app.get("/health")
    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "circuit_breaker_state": service.circuit_breaker.state,
            "registered_domain_packs": sorted(
                {p.pack_id for p in ont_registrar._domain_packs.values()}
            ),
        }

    # ------------------------------------------------------------------------
    # CONTROL PLANE: Domain Pack Catalog & ADK App Registration
    # ------------------------------------------------------------------------
    @app.get("/v1/domains/packs")
    async def list_domain_packs() -> dict[str, Any]:
        """Lists all registered Layer 1 Federated Domain Packs in the catalog."""
        unique_packs = {
            p.pack_id: p.model_dump() for p in ont_registrar._domain_packs.values()
        }
        return {"count": len(unique_packs), "packs": unique_packs}

    @app.post("/v1/domains/packs")
    @app.post("/v1/domains/register")
    async def register_domain_pack(pack: DomainPackSpec) -> dict[str, Any]:
        """Registers a plug-and-play Layer 1 Federated Domain Pack (JSON/Pydantic)."""
        ont_registrar.register_domain_pack(pack)
        return {
            "status": "registered",
            "pack_id": pack.pack_id,
            "domain_name": pack.domain_name,
            "classes_count": len(pack.classes),
            "relations_count": len(pack.relations),
        }

    @app.post("/v1/domains/packs/review")
    async def review_and_refine_domain_pack(
        pack: DomainPackSpec, auto_register: bool = True
    ) -> dict[str, Any]:
        """
        Reviews a proposed Domain Pack via Ontology Intelligence Steward:
        roots unanchored classes under Layer 0, normalizes predicates, infers functional/transitive
        flags, and records an immutable OntologyAuditLogEntry.
        """
        refined, audit = ont_registrar.review_and_refine_domain_pack(
            pack, auto_register=auto_register
        )
        return {
            "status": "reviewed_and_registered" if auto_register else "reviewed",
            "refined_pack": refined.model_dump(),
            "audit_entry": audit.model_dump(),
        }

    @app.post("/v1/domains/evolve/{app_name}")
    async def evolve_app_ontology(
        app_name: str, min_confidence: float = 0.75
    ) -> dict[str, Any]:
        """
        Bottom-Up Ontology Evolution: scans quarantined overflow in `app:{tenant}:{app}`,
        promotes recurring high-confidence candidate classes & predicates into the governed TBox
        and ABox, and records provenance back to the originating `source_episode_ids`.
        """
        audit = await service.evolve_app_ontology_from_overflow(
            app_name=app_name,
            min_confidence_to_promote=min_confidence,
            registrar=ont_registrar,
        )
        return {"status": "evolved", "audit_entry": audit.model_dump()}

    @app.get("/v1/audit/ontology")
    async def get_ontology_audit_log() -> dict[str, Any]:
        """Returns the immutable schema provenance audit log."""
        return {
            "audit_count": len(ont_registrar.audit_log),
            "entries": [e.model_dump() for e in ont_registrar.audit_log],
        }

    @app.post("/v1/domains/packs/turtle")
    async def register_domain_pack_from_turtle(
        req: TurtleDomainPackRequest,
    ) -> dict[str, Any]:
        """Compiles and registers a W3C OWL/RDFS Turtle (.ttl) ontology as a Domain Pack."""
        pack = DomainPackLoader.from_turtle(req.pack_id, req.turtle_content)
        ont_registrar.register_domain_pack(pack)
        return {
            "status": "registered",
            "pack_id": pack.pack_id,
            "domain_name": pack.domain_name,
            "classes": pack.classes,
            "relations": {k: v.model_dump() for k, v in pack.relations.items()},
        }

    @app.post("/v1/apps/register")
    @app.post("/v1/apps/onboard")
    async def register_adk_app(manifest: AppMemoryManifest) -> dict[str, Any]:
        """Registers an ADK agent app's Governance Tier, Domain Pack subscriptions, and custom types."""
        compiled_tbox = await ont_registrar.register_adk_app(manifest)
        return {
            "status": "registered",
            "tenant_id": manifest.tenant_id,
            "app_name": manifest.app_name,
            "governance_tier": manifest.governance_tier.value,
            "compiled_classes": list(compiled_tbox.classes.keys()),
            "compiled_predicates": list(compiled_tbox.relations.keys()),
        }

    @app.get("/v1/apps/{tenant_id}/{app_name}")
    async def inspect_app_governance(tenant_id: str, app_name: str) -> dict[str, Any]:
        """Inspects an ADK app's compiled TBox, governance tier, and active read/write namespaces."""
        ctx = await service.registry.resolve_context(tenant_id, app_name, "inspector")
        return {
            "tenant_id": ctx.tenant_id,
            "app_name": ctx.app_name,
            "read_namespaces": ctx.read_namespaces,
            "write_namespace": ctx.write_namespace,
            "tbox": ctx.tbox.model_dump(),
        }

    # ------------------------------------------------------------------------
    # DATA PLANE: Write, Read, Trace, Outbox & Lineage Purge
    # ------------------------------------------------------------------------
    @app.post("/v1/memory/events")
    async def ingest_events(req: IngestEventsRequest) -> dict[str, Any]:
        adk_events = [
            Event(
                author=e.author,
                content=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=e.text)],
                ),
            )
            for e in req.events
        ]
        try:
            await service.add_events_to_memory(
                app_name=req.app_name,
                user_id=req.user_id,
                events=adk_events,
                session_id=req.session_id,
                custom_metadata=req.custom_metadata,
                defer_extraction=req.defer_extraction,
                extraction_mode=req.extraction_mode,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return {"status": "committed", "events_ingested": len(adk_events)}

    @app.post("/v1/memory/search")
    async def search_memory(req: SearchMemoryRequest) -> dict[str, Any]:
        resp = await service.search_memory(
            app_name=req.app_name,
            user_id=req.user_id,
            query=req.query,
            as_of_iso=req.as_of_iso,
            max_hops=req.max_hops,
        )
        serialized = []
        for m in resp.memories:
            text_val = ""
            if getattr(m, "content", None) and getattr(m.content, "parts", None):
                text_val = "\n".join(
                    p.text for p in m.content.parts if getattr(p, "text", None)
                )
            serialized.append(
                {
                    "author": getattr(m, "author", "memory"),
                    "timestamp": getattr(m, "timestamp", None),
                    "text": text_val,
                }
            )
        return {"memories": serialized}

    @app.post("/v1/traces")
    @app.post("/v1/traces/record")
    async def record_trace(req: RecordTraceRequest) -> dict[str, Any]:
        await service.record_decision_trace(
            app_name=req.app_name,
            user_id=req.user_id,
            run_id=req.run_id,
            session_id=req.session_id,
            steps=req.steps,
        )
        return {"status": "recorded", "run_id": req.run_id, "steps": len(req.steps)}

    @app.post("/v1/playbooks/register")
    async def register_playbook(req: RegisterPlaybookRequest) -> dict[str, Any]:
        pb = await service.register_playbook(
            app_name=req.app_name,
            user_id=req.user_id,
            trigger_intent=req.trigger_intent,
            workflow_steps=req.workflow_steps,
            resolution_summary=req.resolution_summary,
            share_in_domain=req.share_in_domain,
        )
        return {
            "status": "registered",
            "playbook_id": pb.playbook_id,
            "success_count": pb.success_count,
            "namespace": pb.namespace,
        }

    @app.post("/v1/ontology/evolve")
    async def evolve_ontology(req: EvolveOntologyRequest) -> dict[str, Any]:
        audit = await service.evolve_app_ontology_from_overflow(
            req.app_name,
            min_confidence_to_promote=req.min_confidence_to_promote,
            registrar=ont_registrar,
        )
        return {"status": "evolved", **audit.model_dump()}

    @app.post("/v1/outbox/flush")
    async def flush_outbox(batch_size: int = 25) -> dict[str, Any]:
        processed = await service.process_outbox_batch(batch_size=batch_size)
        return {"status": "flushed", "processed_count": processed}

    @app.post("/v1/lineage/purge")
    async def purge_lineage(req: PurgeLineageRequest) -> dict[str, Any]:
        purged_ids = await service.purge_user_lineage(
            app_name=req.app_name, user_id=req.user_id
        )
        return {
            "status": "purged",
            "user_id": req.user_id,
            "purged_episode_ids": purged_ids,
        }

    @app.post("/v1/maintenance/run")
    async def run_maintenance(
        min_episodes_to_compact: int = 4,
        min_overflow_confidence: float = 0.55,
    ) -> dict[str, Any]:
        """Triggers behind-the-scenes Auto-Summarization (Episodic Compaction) and Auto-Cleanup."""
        stats = await service.run_memory_maintenance(
            min_episodes_to_compact=min_episodes_to_compact,
            min_overflow_confidence=min_overflow_confidence,
        )
        return {"status": "completed", **stats}

    # ------------------------------------------------------------------------
    # UNIFIED MCP LAYER: Co-hosted MCP SSE, Streamable HTTP & Stateless Tool RPC
    # ------------------------------------------------------------------------
    if mount_mcp:
        mcp_tools = MemoryMcpServerTools(service=service, registrar=ont_registrar)
        mcp_server = create_mcp_server(service=service, registrar=ont_registrar)

        @app.get("/v1/mcp/tools")
        async def list_mcp_tools() -> dict[str, Any]:
            """Lists all MCP tools exposed by this unified Memory Platform pod."""
            return {
                "server_name": "adk-composite-memory",
                "transports": {
                    "sse": "/mcp/sse",
                    "streamable_http": "/mcp/http",
                    "stateless_rpc": "/v1/mcp/call",
                },
                "tools": MemoryMcpServerTools.TOOL_NAMES,
            }

        @app.post("/v1/mcp/call")
        async def call_mcp_tool(req: McpToolCallRequest) -> dict[str, Any]:
            """Stateless Kubernetes-friendly HTTP endpoint to invoke any MCP memory tool."""
            fn = getattr(mcp_tools, req.tool_name, None)
            if fn is None or req.tool_name.startswith("_"):
                raise HTTPException(
                    status_code=404, detail=f"Unknown MCP tool: {req.tool_name}"
                )
            res = await fn(**req.arguments)
            return {"tool_name": req.tool_name, "result": res}

        if hasattr(mcp_server, "sse_app"):
            app.mount("/mcp/sse", mcp_server.sse_app())
        if hasattr(mcp_server, "streamable_http_app"):
            app.mount("/mcp/http", mcp_server.streamable_http_app())

    return app
