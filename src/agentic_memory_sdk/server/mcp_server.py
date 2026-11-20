from __future__ import annotations

import json
from typing import Any

from ..governance import PlatformOntologyRegistrar
from ..models import (
    AppMemoryManifest,
    DomainPackSpec,
    GovernanceTier,
    ReasoningTraceStep,
    RelationRule,
)
from ..ontology_loader import DomainPackLoader
from ..service import CompositeMemoryService, Event, types


class MemoryMcpServerTools:
    """
    Exposes the complete Data Plane + Control Plane of the Composite Memory Platform
    as MCP-compatible tools, resources, and prompts that can be mounted into a
    `MCPServer` (mcp 2.x) / `FastMCP` (mcp 1.x) instance, served over Streamable HTTP
    (`2026-07-28` / `2025-11-25`), invoked via `/v1/mcp/call`, or passed to Google ADK's `McpToolset`.
    """

    TOOL_NAMES: list[str] = [
        "recall_context",
        "check_answerable",
        "commit_episode",
        "ingest_memory",
        "record_decision_trace",
        "register_playbook",
        "register_domain_pack",
        "induce_domain_pack",
        "certify_domain_pack",
        "register_app_manifest",
        "evolve_app_ontology",
        "run_memory_maintenance",
        "inspect_platform_governance",
        "purge_user_lineage",
    ]

    RESOURCE_URIS: list[str] = [
        "memory://governance/health",
        "memory://domains/packs",
        "memory://audit/ontology",
        "memory://apps/{tenant_id}/{app_name}/tbox",
    ]

    PROMPT_NAMES: list[str] = [
        "memory_grounded_assistant",
        "ontology_steward_domain_designer",
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

    async def check_answerable(
        self,
        app_name: str,
        user_id: str,
        query: str,
        as_of_timestamp: str | None = None,
        max_hops: int = 2,
    ) -> dict[str, Any]:
        """
        Pre-flight Ontology & 3-Tier Memory Answerability / Coverage Check:
        Evaluates whether the app's governed TBox (classes, predicates, synonyms) and
        3-Tier Memory (ABox Graph, Episodic Seeds, Playbooks) cover the concepts in `query`.
        """
        return await self.service.check_answerable(
            app_name=app_name,
            user_id=user_id,
            query=query,
            as_of_iso=as_of_timestamp,
            max_hops=max_hops,
            registrar=self.registrar,
        )

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
        classes: dict[str, str] | None = None,
        relations: list[dict[str, Any]] | None = None,
        version: str = "1.0.0",
        review_with_steward: bool = True,
        sql_ddl: str | None = None,
        ossie_yaml: str | None = None,
    ) -> dict[str, Any]:
        """
        Registers a Layer-1 Federated Domain Pack in the platform catalog via MCP
        from explicit `classes`/`relations`, relational `sql_ddl`, or Apache `ossie_yaml`.
        """
        if sql_ddl:
            spec = DomainPackLoader.from_sql_ddl(pack_id, sql_ddl)
        elif ossie_yaml:
            spec = DomainPackLoader.from_ossie_yaml(pack_id, ossie_yaml)
        else:
            rel_map = {}
            for r in relations or []:
                rule = RelationRule.model_validate(r)
                rel_map[rule.predicate] = rule
            spec = DomainPackSpec(
                pack_id=pack_id,
                domain_name=pack_id.split("@")[0],
                version=version,
                classes=classes or {},
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
                "relations": {k: v.model_dump() for k, v in refined.relations.items()},
                "synonyms": refined.synonyms,
                "certified": refined.certified,
                "audit_modifications": audit.modifications_summary,
            }
        self.registrar.register_domain_pack(spec)
        return {
            "status": "registered",
            "pack_id": spec.pack_id,
            "classes": spec.classes,
            "relations": {k: v.model_dump() for k, v in spec.relations.items()},
            "synonyms": spec.synonyms,
            "certified": spec.certified,
        }

    async def certify_domain_pack(
        self,
        pack_id: str,
        certified_by: str = "domain_steward",
    ) -> dict[str, Any]:
        """Marks a registered or auto-induced Domain Pack as steward-certified (`certified=True`)."""
        pack, audit = self.registrar.certify_domain_pack(
            pack_id=pack_id, certified_by=certified_by
        )
        return {
            "status": "certified",
            "pack_id": pack.pack_id,
            "certified": pack.certified,
            "certified_by": pack.certified_by,
            "audit_entry": audit.model_dump(),
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
        min_support_count: int = 1,
    ) -> dict[str, Any]:
        """Promotes high-confidence quarantined overflow classes & predicates into the governed TBox."""
        audit = await self.service.evolve_app_ontology_from_overflow(
            app_name=app_name,
            min_confidence_to_promote=min_confidence_to_promote,
            min_support_count=min_support_count,
            registrar=self.registrar,
        )
        return {"status": "evolved", **audit.model_dump()}

    async def induce_domain_pack(
        self,
        domain_name: str,
        transcripts: list[str] | None = None,
        app_name: str | None = None,
        version: str = "1.0.0",
        min_support_count: int = 1,
        min_confidence: float = 0.70,
        auto_register: bool = True,
    ) -> dict[str, Any]:
        """
        Cold-Start Autonomous Ontology Induction: synthesizes a Layer-1 `DomainPackSpec`
        from sample conversation transcripts or stored app episodes.
        """
        pack, audit, stats = await self.service.induce_domain_pack(
            domain_name=domain_name,
            transcripts=transcripts,
            app_name=app_name,
            version=version,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
            auto_register=auto_register,
            registrar=self.registrar,
        )
        return {
            "status": "induced_and_registered" if auto_register else "induced",
            "pack": pack.model_dump(),
            "audit_entry": audit.model_dump(),
            "stats": stats,
        }

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
    Creates a Model Context Protocol (MCP) Server following the latest MCP specification
    (`2026-07-28` / `2025-11-25`) via `MCPServer` (mcp 2.x) or `FastMCP` (mcp 1.x), exposing:
      - 11 annotated tools with `ToolAnnotations` (`readOnlyHint`, `destructiveHint`,
        `idempotentHint`, `openWorldHint`) and `structuredContent` / `outputSchema`
      - 4 MCP Resources (`memory://governance/health`, `memory://domains/packs`,
        `memory://audit/ontology`, and template `memory://apps/{tenant_id}/{app_name}/tbox`)
      - 2 MCP Prompts (`memory_grounded_assistant`, `ontology_steward_domain_designer`)
    """
    try:
        from mcp.server.mcpserver import MCPServer as _MCPServerClass
    except ImportError:
        from mcp.server.fastmcp import FastMCP as _MCPServerClass

    try:
        from mcp.types import ToolAnnotations
    except ImportError:  # pragma: no cover
        ToolAnnotations = None  # type: ignore[assignment]

    def _ann(
        title: str,
        *,
        read_only: bool = False,
        destructive: bool = False,
        idempotent: bool = False,
        open_world: bool = False,
    ) -> Any:
        if ToolAnnotations is None:
            return None
        return ToolAnnotations(
            title=title,
            read_only_hint=read_only,
            destructive_hint=destructive,
            idempotent_hint=idempotent,
            open_world_hint=open_world,
        )

    try:
        mcp = _MCPServerClass(
            name,
            title="Agentic Composite Memory Platform MCP Server",
            version="0.1.5",
            instructions=(
                "Enterprise multi-tenant 3-Tier Composite Memory & Stratified Ontology MCP Server. "
                "Use `recall_context` before answering user queries, `commit_episode` to store turns, "
                "`register_playbook` to codify reusable SOPs, and `memory://...` resources to inspect schemas."
            ),
            website_url="https://github.com/STHITAPRAJNAS/adk-memory-platform",
        )
    except TypeError:  # Fallback for older FastMCP 1.x signatures
        mcp = _MCPServerClass(name)

    tools = MemoryMcpServerTools(service=service, registrar=registrar)

    # -------------------------------------------------------------------------
    # 1. MCP TOOLS (11 Annotated Data-Plane + Control-Plane Tools)
    # -------------------------------------------------------------------------
    mcp.tool(
        name="recall_context",
        title="3-Tier Hybrid Memory Recall",
        description=(
            "3-Tier hybrid recall: Procedural Playbooks + RRF seed search + multi-hop bi-temporal "
            "ontology graph traversal + episodic memory with point-in-time (`as_of_timestamp`) support."
        ),
        annotations=_ann(
            "3-Tier Hybrid Memory Recall",
            read_only=True,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.recall_context)

    mcp.tool(
        name="check_answerable",
        title="Pre-Flight Ontology & Memory Coverage Check",
        description=(
            "Evaluate whether the governed ontology (TBox classes, predicates, synonyms) and "
            "3-Tier Memory (Graph ABox, Episodic Seeds, Playbooks) cover the concepts in `query` "
            "before attempting an answer."
        ),
        annotations=_ann(
            "Pre-Flight Ontology & Memory Coverage Check",
            read_only=True,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.check_answerable)

    mcp.tool(
        name="commit_episode",
        title="Commit Interaction Episode",
        description=(
            "Ingest an agent/user interaction into Episodic Memory and extract TBox-validated "
            "entities and bi-temporal facts (supports `extraction_mode='hot_path'|'sleep_time'`)."
        ),
        annotations=_ann(
            "Commit Interaction Episode",
            read_only=False,
            destructive=False,
            idempotent=False,
            open_world=False,
        ),
        structured_output=True,
    )(tools.commit_episode)

    mcp.tool(
        name="ingest_memory",
        title="Ingest Memory Turn",
        description="Alias for `commit_episode` using `text` parameter.",
        annotations=_ann(
            "Ingest Memory Turn",
            read_only=False,
            destructive=False,
            idempotent=False,
            open_world=False,
        ),
        structured_output=True,
    )(tools.ingest_memory)

    mcp.tool(
        name="record_decision_trace",
        title="Record Procedural Reasoning Trace",
        description=(
            "Record Tier 3 Procedural Reasoning Traces (`AgentRun -> ToolCall -> Entity`) "
            "for explainability and auditability."
        ),
        annotations=_ann(
            "Record Procedural Reasoning Trace",
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.record_decision_trace)

    mcp.tool(
        name="register_playbook",
        title="Register or Reinforce Procedural Playbook",
        description="Register or reinforce a reusable Tier-3 Procedural Playbook (`trigger_intent -> workflow_steps`).",
        annotations=_ann(
            "Register or Reinforce Procedural Playbook",
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.register_playbook)

    mcp.tool(
        name="register_domain_pack",
        title="Register Layer-1 Federated Domain Pack",
        description="Register a Layer-1 Federated Domain Pack (`classes` & `relations`) in the platform catalog.",
        annotations=_ann(
            "Register Layer-1 Federated Domain Pack",
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.register_domain_pack)

    mcp.tool(
        name="induce_domain_pack",
        title="Cold-Start Autonomous Domain Pack Induction",
        description=(
            "Automatically induce and register a Layer-1 Federated Domain Pack (classes anchored "
            "under Layer 0 Upper Ontology, domain/range relations, and empirical functional/transitive "
            "flags) from raw sample transcripts or stored app episodes."
        ),
        annotations=_ann(
            "Cold-Start Autonomous Domain Pack Induction",
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.induce_domain_pack)

    mcp.tool(
        name="certify_domain_pack",
        title="Certify Layer-1 Domain Pack",
        description="Mark a registered or auto-induced Domain Pack as steward-certified (`certified=True`).",
        annotations=_ann(
            "Certify Layer-1 Domain Pack",
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.certify_domain_pack)

    mcp.tool(
        name="register_app_manifest",
        title="Register Agent App Governance Manifest",
        description=(
            "Register an agent app's Governance Tier (`APP_PRIVATE`, `DOMAIN_SHARED`, `GLOBAL_CORE`) "
            "and subscribed Domain Packs."
        ),
        annotations=_ann(
            "Register Agent App Governance Manifest",
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.register_app_manifest)

    mcp.tool(
        name="evolve_app_ontology",
        title="Evolve App Ontology from Overflow",
        description="Promote high-confidence quarantined overflow classes & predicates into the governed TBox.",
        annotations=_ann(
            "Evolve App Ontology from Overflow",
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.evolve_app_ontology)

    mcp.tool(
        name="run_memory_maintenance",
        title="Run Sleep-Time Memory Maintenance",
        description="Run sleep-time batch extraction, session compaction, overflow pruning, and entity dossier synthesis.",
        annotations=_ann(
            "Run Sleep-Time Memory Maintenance",
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.run_memory_maintenance)

    mcp.tool(
        name="inspect_platform_governance",
        title="Inspect Platform & App Governance",
        description="Inspect registered Domain Packs, circuit-breaker health, audit log count, and optional app TBox.",
        annotations=_ann(
            "Inspect Platform & App Governance",
            read_only=True,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.inspect_platform_governance)

    mcp.tool(
        name="purge_user_lineage",
        title="GDPR Right-to-be-Forgotten Lineage Purge",
        description=(
            "Execute a cascading GDPR Right-to-be-Forgotten lineage deletion across "
            "Episodic and Bi-Temporal Graph stores."
        ),
        annotations=_ann(
            "GDPR Right-to-be-Forgotten Lineage Purge",
            read_only=False,
            destructive=True,
            idempotent=True,
            open_world=False,
        ),
        structured_output=True,
    )(tools.purge_user_lineage)

    # -------------------------------------------------------------------------
    # 2. MCP RESOURCES (`resources/list`, `resources/templates/list`, `resources/read`)
    # -------------------------------------------------------------------------
    if hasattr(mcp, "resource"):

        @mcp.resource(
            "memory://governance/health",
            name="platform_governance_health",
            title="Platform Governance & Circuit Breaker Health",
            description="Live health status, circuit-breaker state, and registered Domain Pack IDs.",
            mime_type="application/json",
        )
        async def resource_governance_health() -> str:
            data = await tools.inspect_platform_governance()
            return json.dumps(data, indent=2)

        @mcp.resource(
            "memory://domains/packs",
            name="domain_packs_catalog",
            title="Layer-1 Federated Domain Packs Catalog",
            description="Complete catalog of all registered Layer-1 Federated Domain Packs (classes and relations).",
            mime_type="application/json",
        )
        def resource_domain_packs() -> str:
            packs = {
                p.pack_id: p.model_dump() for p in tools.registrar._domain_packs.values()
            }
            return json.dumps({"count": len(packs), "packs": packs}, indent=2)

        @mcp.resource(
            "memory://audit/ontology",
            name="ontology_audit_log",
            title="Immutable Ontology Provenance Audit Log",
            description="Immutable audit trail of all top-down steward reviews and bottom-up overflow promotions.",
            mime_type="application/json",
        )
        def resource_ontology_audit_log() -> str:
            entries = [e.model_dump() for e in tools.registrar.audit_log]
            return json.dumps({"audit_count": len(entries), "entries": entries}, indent=2)

        @mcp.resource(
            "memory://apps/{tenant_id}/{app_name}/tbox",
            name="app_compiled_tbox",
            title="App Compiled TBox & Namespaces",
            description="Inspects a specific tenant/app's compiled ontology TBox and active read/write namespaces.",
            mime_type="application/json",
        )
        async def resource_app_tbox(tenant_id: str, app_name: str) -> str:
            data = await tools.inspect_platform_governance(
                app_name=f"{tenant_id}:{app_name}"
            )
            return json.dumps(data.get("app_governance", data), indent=2)

    # -------------------------------------------------------------------------
    # 3. MCP PROMPTS (`prompts/list`, `prompts/get`)
    # -------------------------------------------------------------------------
    if hasattr(mcp, "prompt"):

        @mcp.prompt(
            name="memory_grounded_assistant",
            title="3-Tier Memory-Grounded Assistant Prompt",
            description=(
                "Executes 3-Tier hybrid recall (`recall_context`) and builds a grounded system/user "
                "prompt hydrated with matched Procedural Playbooks, Bi-Temporal Knowledge Graph facts, "
                "and Episodic conversation history."
            ),
        )
        async def prompt_memory_grounded_assistant(
            app_name: str,
            user_id: str,
            query: str,
        ) -> str:
            recalled = await tools.recall_context(
                app_name=app_name, user_id=user_id, query=query
            )
            blocks = "\n\n".join(
                m["text"] for m in recalled.get("memories", []) if m.get("text")
            )
            return (
                "You are an enterprise agent grounded in the Agentic Composite Memory Platform.\n"
                "Follow any `[MATCHED_PROCEDURAL_PLAYBOOKS]` SOPs strictly and prioritize current "
                "`[COMPOSITE_MEMORY_CONTEXT]` graph assertions over older episodes.\n\n"
                f"=== RECALLED MEMORY CONTEXT ({app_name} / {user_id}) ===\n"
                f"{blocks or '(No prior memories found.)'}\n\n"
                f"=== CURRENT USER REQUEST ===\n{query}"
            )

        @mcp.prompt(
            name="ontology_steward_domain_designer",
            title="Ontology Steward Domain Pack Designer Prompt",
            description=(
                "Guides an LLM Ontology Steward to synthesize a W3C/Layer-1 `DomainPackSpec` "
                "(classes anchored under Layer 0 Upper Ontology and typed relations) from domain transcripts."
            ),
        )
        def prompt_ontology_steward_domain_designer(
            domain_name: str,
            sample_transcript: str,
        ) -> str:
            return (
                f"Design a Layer-1 Federated Domain Pack for domain `{domain_name}`.\n"
                "1. Anchor every class under a Layer-0 Upper Ontology parent "
                "(`Agent`, `PhysicalObject`, `ConceptualEntity`, `Event`, `TemporalEntity`, `Location`).\n"
                "2. Normalize all predicates to `UPPER_SNAKE_CASE` and specify `domain`, `range`, "
                "`is_functional`, and `is_transitive`.\n"
                "3. Call the `register_domain_pack` MCP tool with `review_with_steward=True`.\n\n"
                f"=== SAMPLE DOMAIN TRANSCRIPT ===\n{sample_transcript}"
            )

    return mcp
