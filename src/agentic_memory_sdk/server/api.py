from __future__ import annotations

from contextlib import asynccontextmanager
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..governance import PlatformOntologyRegistrar
from ..models import (
    AppMemoryManifest,
    DomainPackSpec,
    ReasoningTraceStep,
)
from ..ontology_loader import DomainPackLoader
from ..security import AuthPolicyConfig
from ..service import CompositeMemoryService, Event, types
from .mcp_server import MemoryMcpServerTools, create_mcp_server
from .ui_console import CONSOLE_HTML


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
    min_support_count: int = 1


class InduceDomainPackRequest(BaseModel):
    domain_name: str
    transcripts: list[str] = Field(default_factory=list)
    app_name: str | None = None
    version: str = "1.0.0"
    min_support_count: int = 1
    min_confidence: float = 0.70
    auto_register: bool = True


class SqlDdlDomainPackRequest(BaseModel):
    pack_id: str
    sql_ddl: str
    review_with_steward: bool = True


class OssieDomainPackRequest(BaseModel):
    pack_id: str
    ossie_content: str | dict[str, Any]
    review_with_steward: bool = True


class CertifyDomainPackRequest(BaseModel):
    pack_id: str
    certified_by: str = "domain_steward"


class CheckAnswerableRequest(BaseModel):
    app_name: str
    user_id: str
    query: str
    as_of_iso: str | None = None
    max_hops: int = 2


class EditMemoryRequest(BaseModel):
    app_name: str
    user_id: str
    new_content: str
    actor: str = "operator"
    reason: str = "Manual edit via API"


class UpsertWorkingBlockRequest(BaseModel):
    app_name: str
    user_id: str
    label: str
    value: str
    mode: str = "replace"
    max_chars: int = 2000
    actor: str = "agent"


def create_memory_platform_app(
    service: CompositeMemoryService,
    registrar: PlatformOntologyRegistrar | None = None,
    enable_outbox_worker: bool = True,
    outbox_interval_sec: float = 2.0,
    enable_maintenance_worker: bool = False,
    maintenance_interval_sec: float = 300.0,
    maintenance_min_episodes_to_compact: int = 4,
    maintenance_min_overflow_confidence: float = 0.55,
    maintenance_run_immediately: bool = False,
    mount_mcp: bool = True,
    auth_policy: AuthPolicyConfig | None = None,
) -> FastAPI:
    """
    Creates the Unified Multi-Tenant Memory-as-a-Service FastAPI + MCP application.
    Serves BOTH:
      1. FastAPI REST Control & Data Plane (`/v1/...`) for `RemoteCompositeMemoryService`
      2. Model Context Protocol (MCP) Server (`/mcp/sse`, `/mcp/http`, `/v1/mcp/tools`, `/v1/mcp/call`)
    from a single ASGI process / Kubernetes Pod.
    """

    ont_registrar = registrar or PlatformOntologyRegistrar(service.registry)
    mcp_tools = (
        MemoryMcpServerTools(service=service, registrar=ont_registrar)
        if mount_mcp
        else None
    )
    mcp_server = (
        create_mcp_server(service=service, registrar=ont_registrar)
        if mount_mcp
        else None
    )
    mcp_http_app: Any = None
    if mount_mcp and mcp_server is not None and hasattr(mcp_server, "streamable_http_app"):
        try:
            from mcp.server.transport_security import TransportSecuritySettings

            sec = TransportSecuritySettings(enable_dns_rebinding_protection=False)
            mcp_http_app = mcp_server.streamable_http_app(
                streamable_http_path="/",
                stateless_http=True,
                json_response=True,
                transport_security=sec,
            )
        except TypeError:
            mcp_http_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if enable_outbox_worker:
            await service.start_outbox_worker(interval_sec=outbox_interval_sec)
        if enable_maintenance_worker:
            await service.start_maintenance_worker(
                interval_sec=maintenance_interval_sec,
                min_episodes_to_compact=maintenance_min_episodes_to_compact,
                min_overflow_confidence=maintenance_min_overflow_confidence,
                run_immediately=maintenance_run_immediately,
            )
        try:
            if (
                mcp_server is not None
                and hasattr(mcp_server, "session_manager")
                and getattr(mcp_server.session_manager, "_task_group", None) is None
            ):
                mcp_server.session_manager._has_started = False
                async with mcp_server.session_manager.run():
                    yield
            else:
                yield
        finally:
            if enable_maintenance_worker:
                await service.stop_maintenance_worker()
            if enable_outbox_worker:
                await service.stop_outbox_worker()

    app = FastAPI(
        title="ADK Enterprise Composite Memory Platform API",
        version="0.2.1",
        description=(
            "Domain-Agnostic Multi-Tenant Memory Service combining Episodic Memory, "
            "Stratified Ontology (TBox), Bi-Temporal Graph KB (ABox), and Reasoning Traces."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "circuit_breaker_state": service.circuit_breaker.state,
            "maintenance_worker_running": service.maintenance_worker_running,
            "maintenance_last_stats": service._maintenance_last_stats,
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

    @app.post("/v1/domains/packs/induce")
    async def induce_domain_pack(req: InduceDomainPackRequest) -> dict[str, Any]:
        """
        Cold-Start Autonomous Ontology Induction: synthesizes a Layer-1 `DomainPackSpec`
        from sample conversation transcripts or stored app episodes.
        """
        pack, audit, stats = await service.induce_domain_pack(
            domain_name=req.domain_name,
            transcripts=req.transcripts or None,
            app_name=req.app_name,
            version=req.version,
            min_support_count=req.min_support_count,
            min_confidence=req.min_confidence,
            auto_register=req.auto_register,
            registrar=ont_registrar,
        )
        return {
            "status": "induced_and_registered" if req.auto_register else "induced",
            "pack": pack.model_dump(),
            "audit_entry": audit.model_dump(),
            "stats": stats,
        }

    @app.post("/v1/domains/evolve/{app_name}")
    async def evolve_app_ontology(
        app_name: str,
        min_confidence: float = 0.75,
        min_support_count: int = 1,
    ) -> dict[str, Any]:
        """
        Bottom-Up Ontology Evolution: scans quarantined overflow in `app:{tenant}:{app}`,
        promotes recurring high-confidence candidate classes & predicates into the governed TBox
        and ABox, and records provenance back to the originating `source_episode_ids`.
        """
        audit = await service.evolve_app_ontology_from_overflow(
            app_name=app_name,
            min_confidence_to_promote=min_confidence,
            min_support_count=min_support_count,
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

    @app.post("/v1/domains/packs/sql-ddl")
    async def register_domain_pack_from_sql_ddl(
        req: SqlDdlDomainPackRequest,
    ) -> dict[str, Any]:
        """Compiles and registers a relational SQL DDL schema (`CREATE TABLE ...`) as a Layer-1 Domain Pack."""
        pack = DomainPackLoader.from_sql_ddl(req.pack_id, req.sql_ddl)
        if req.review_with_steward:
            pack, audit = ont_registrar.review_and_refine_domain_pack(
                pack, auto_register=True
            )
            return {
                "status": "reviewed_and_registered",
                "pack_id": pack.pack_id,
                "domain_name": pack.domain_name,
                "classes": pack.classes,
                "relations": {k: v.model_dump() for k, v in pack.relations.items()},
                "synonyms": pack.synonyms,
                "audit_entry": audit.model_dump(),
            }
        ont_registrar.register_domain_pack(pack)
        return {
            "status": "registered",
            "pack_id": pack.pack_id,
            "domain_name": pack.domain_name,
            "classes": pack.classes,
            "relations": {k: v.model_dump() for k, v in pack.relations.items()},
            "synonyms": pack.synonyms,
        }

    @app.post("/v1/domains/packs/ossie")
    async def register_domain_pack_from_ossie(
        req: OssieDomainPackRequest,
    ) -> dict[str, Any]:
        """Compiles and registers an Apache Ossie / BI Semantic Model (YAML or JSON) as a Layer-1 Domain Pack."""
        pack = DomainPackLoader.from_ossie_yaml(req.pack_id, req.ossie_content)
        if req.review_with_steward:
            pack, audit = ont_registrar.review_and_refine_domain_pack(
                pack, auto_register=True
            )
            return {
                "status": "reviewed_and_registered",
                "pack_id": pack.pack_id,
                "domain_name": pack.domain_name,
                "classes": pack.classes,
                "relations": {k: v.model_dump() for k, v in pack.relations.items()},
                "synonyms": pack.synonyms,
                "certified": pack.certified,
                "audit_entry": audit.model_dump(),
            }
        ont_registrar.register_domain_pack(pack)
        return {
            "status": "registered",
            "pack_id": pack.pack_id,
            "domain_name": pack.domain_name,
            "classes": pack.classes,
            "relations": {k: v.model_dump() for k, v in pack.relations.items()},
            "synonyms": pack.synonyms,
            "certified": pack.certified,
        }

    @app.post("/v1/domains/packs/certify")
    async def certify_domain_pack(
        req: CertifyDomainPackRequest,
    ) -> dict[str, Any]:
        """Marks a registered or auto-induced Domain Pack as steward-certified (`certified=True`)."""
        try:
            pack, audit = ont_registrar.certify_domain_pack(
                pack_id=req.pack_id, certified_by=req.certified_by
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "status": "certified",
            "pack_id": pack.pack_id,
            "certified": pack.certified,
            "certified_by": pack.certified_by,
            "audit_entry": audit.model_dump(),
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

    @app.post("/v1/memory/check-answerable")
    async def check_answerable(req: CheckAnswerableRequest) -> dict[str, Any]:
        """
        Pre-flight Ontology & 3-Tier Memory Answerability / Coverage Check:
        Evaluates whether the app's governed TBox and 3-Tier Memory cover the concepts in `req.query`.
        """
        return await service.check_answerable(
            app_name=req.app_name,
            user_id=req.user_id,
            query=req.query,
            as_of_iso=req.as_of_iso,
            max_hops=req.max_hops,
            registrar=ont_registrar,
        )

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
            min_support_count=req.min_support_count,
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
    # ENTERPRISE AUTHENTICATION, RBAC & TENANT-ISOLATION MIDDLEWARE
    # ------------------------------------------------------------------------
    if auth_policy is not None and auth_policy.enabled:

        @app.middleware("http")
        async def auth_and_rbac_middleware(request: Request, call_next: Any):
            path = request.url.path
            if any(path == p or path.startswith(p + "/") for p in auth_policy.public_paths):
                return await call_next(request)
            try:
                principal = await auth_policy.authenticate(
                    request.headers.get("authorization"),
                    request.headers.get("x-api-key"),
                )
            except PermissionError as exc:
                service.observability.inc("auth_denied")
                return JSONResponse(status_code=401, content={"detail": str(exc)})

            required_role = auth_policy.required_role_for_route(request.method, path)
            if not principal.has_role(required_role):
                service.observability.inc("rbac_denied")
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": f"Forbidden: requires role '{required_role.value}' "
                        f"(caller '{principal.subject}' has {[r.value for r in principal.roles]})"
                    },
                )

            # Enforce tenant & scope isolation on query params or JSON body
            target_tenant = request.query_params.get("tenant_id")
            target_app = request.query_params.get("app_name")
            target_user = request.query_params.get("user_id")
            if request.method in ("POST", "PUT", "PATCH"):
                body_bytes = await request.body()
                if body_bytes:
                    try:
                        parsed = json.loads(body_bytes.decode("utf-8"))
                        if isinstance(parsed, dict):
                            target_tenant = target_tenant or parsed.get("tenant_id")
                            target_app = target_app or parsed.get("app_name")
                            target_user = target_user or parsed.get("user_id")
                    except Exception:
                        pass
            try:
                auth_policy.authorize_tenant_and_scope(
                    principal,
                    target_tenant_id=target_tenant,
                    target_app_name=target_app,
                    target_user_id=target_user,
                )
            except PermissionError as exc:
                service.observability.inc("tenant_isolation_denied")
                return JSONResponse(status_code=403, content={"detail": str(exc)})

            request.state.principal = principal
            return await call_next(request)

    # ------------------------------------------------------------------------
    # MEMORY INSPECTOR, REVISIONS, WORKING BLOCKS, OBSERVABILITY & CONSOLE UI
    # ------------------------------------------------------------------------
    @app.get("/ui", response_class=HTMLResponse)
    async def governance_console_ui() -> HTMLResponse:
        """Serves the bundled Visual Memory & Ontology Governance Console UI."""
        return HTMLResponse(content=CONSOLE_HTML)

    @app.get("/v1/memory/items")
    async def list_memory_items(
        app_name: str,
        user_id: str,
        include_expired: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Lists stored episodic & cognitive memories for inspection/editing."""
        items = await service.list_user_memories(
            app_name, user_id, include_expired=include_expired, limit=limit
        )
        return {"count": len(items), "memories": items}

    @app.put("/v1/memory/items/{episode_id}")
    async def edit_memory_item(episode_id: str, req: EditMemoryRequest) -> dict[str, Any]:
        """Manually edits a memory, re-embeds it, and records an immutable revision."""
        try:
            return await service.edit_user_memory(
                app_name=req.app_name,
                user_id=req.user_id,
                episode_id=episode_id,
                new_content=req.new_content,
                actor=req.actor,
                reason=req.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/v1/memory/items/{episode_id}")
    async def delete_memory_item(
        episode_id: str,
        app_name: str,
        user_id: str,
        actor: str = "operator",
        reason: str = "Manual delete via API",
    ) -> dict[str, Any]:
        """Deletes a single memory record and logs an immutable DELETE revision."""
        try:
            return await service.delete_user_memory(
                app_name=app_name,
                user_id=user_id,
                episode_id=episode_id,
                actor=actor,
                reason=reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/memory/revisions")
    async def list_memory_revisions(
        app_name: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Lists immutable memory revision history (ADD/UPDATE/RESOLVE/EXPIRE/DELETE/MANUAL_EDIT)."""
        revs = service.get_memory_revisions(app_name=app_name, user_id=user_id, limit=limit)
        return {"count": len(revs), "revisions": [r.model_dump() for r in revs]}

    @app.get("/v1/memory/blocks")
    async def list_working_blocks(app_name: str, user_id: str) -> dict[str, Any]:
        """Lists Letta-style self-editable Core Working Memory Blocks for a user/app."""
        blocks = service.get_working_memory_blocks(app_name, user_id)
        return {"count": len(blocks), "blocks": [b.model_dump() for b in blocks]}

    @app.post("/v1/memory/blocks")
    async def upsert_working_block(req: UpsertWorkingBlockRequest) -> dict[str, Any]:
        """Creates, replaces, or appends to a Letta-style Core Working Memory Block."""
        blk = service.upsert_working_memory_block(
            app_name=req.app_name,
            user_id=req.user_id,
            label=req.label,
            value=req.value,
            mode=req.mode,
            max_chars=req.max_chars,
            actor=req.actor,
        )
        return {"status": "saved", "block": blk.model_dump()}

    @app.get("/v1/metrics")
    async def get_metrics_json() -> dict[str, Any]:
        """Returns OpenTelemetry-compatible observability counters & latency percentiles."""
        return service.observability.snapshot()

    @app.get("/metrics", response_class=PlainTextResponse)
    async def get_metrics_prometheus() -> PlainTextResponse:
        """Returns Prometheus text-format metrics for scraping."""
        return PlainTextResponse(content=service.observability.to_prometheus_text())

    # ------------------------------------------------------------------------
    # UNIFIED MCP LAYER: Co-hosted MCP SSE, Streamable HTTP & Stateless Tool RPC
    # ------------------------------------------------------------------------
    if mount_mcp and mcp_tools is not None and mcp_server is not None:

        @app.get("/v1/mcp/tools")
        async def list_mcp_tools() -> dict[str, Any]:
            """Lists all MCP tools, resources, prompts, and protocol versions exposed by this pod."""
            return {
                "server_name": "adk-composite-memory",
                "protocol_version": "2026-07-28",
                "supported_protocol_versions": [
                    "2026-07-28",
                    "2025-11-25",
                    "2025-06-18",
                    "2025-03-26",
                ],
                "transports": {
                    "sse": "/mcp/sse",
                    "streamable_http": "/mcp/http",
                    "stateless_rpc": "/v1/mcp/call",
                },
                "tools": MemoryMcpServerTools.TOOL_NAMES,
                "resources": MemoryMcpServerTools.RESOURCE_URIS,
                "prompts": MemoryMcpServerTools.PROMPT_NAMES,
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
        if mcp_http_app is not None:
            async def _auto_lifespan_mcp_http(scope: Any, receive: Any, send: Any) -> None:
                sm = getattr(mcp_server, "session_manager", None)
                if sm is not None and getattr(sm, "_task_group", None) is None:
                    sm._has_started = False
                    async with sm.run():
                        await mcp_http_app(scope, receive, send)
                    sm._has_started = False
                else:
                    await mcp_http_app(scope, receive, send)

            app.mount("/mcp/http", _auto_lifespan_mcp_http)

    return app
