from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field


class UpperConceptType(str, Enum):
    """Layer 0 Platform Upper Ontology classes mandatory for all apps."""

    ACTOR = "p:Actor"
    OBJECT = "p:Object"
    EVENT = "p:Event"
    CONCEPT = "p:Concept"
    PREFERENCE = "p:UserPreference"
    CANDIDATE = "p:CandidateConcept"


class MemoryScope(str, Enum):
    """
    Smart memory sharing & visibility scope determined during extraction/classification:
      - AUTO: Inferred automatically from TBox class, confidence, and app governance tier
      - EPHEMERAL: Short-lived chitchat/acknowledgment; skipped from graph & pruned in cleanup
      - USER_GLOBAL: Shared across ALL agent apps for the SAME `user_id` (`user:{tenant}:{user_id}`)
      - APP_PRIVATE: Isolated to a single agent app (`app:{tenant}:{app}`)
      - DOMAIN_SHARED: Shared across all apps and users subscribed to the same domain (`domain:{tenant}:{domain}`)
      - TENANT_GLOBAL: Shared across all apps and users in the tenant (`global:{tenant}:core`)
    """

    AUTO = "auto"
    EPHEMERAL = "ephemeral"
    USER_GLOBAL = "user_global"
    APP_PRIVATE = "app_private"
    DOMAIN_SHARED = "domain_shared"
    TENANT_GLOBAL = "tenant_global"


class GovernanceTier(str, Enum):
    """Progressive governance tiers scaling strictness with blast radius."""

    TIER0_AUTO = "tier0_auto"                  # Zero-config, app-private only
    TIER1_APP_TYPED = "tier1_app_typed"        # Custom app schema, app-private
    TIER2_DOMAIN_SHARED = "tier2_domain_shared"  # Shared domain pack + app overflow
    TIER3_ENTERPRISE_CORE = "tier3_enterprise_core"  # Read-only reference data for agents


class ExtractionMode(str, Enum):
    """
    Controls when LLM ontology extraction & graph mutation occur:
      - HOT: Extracts entities and bi-temporal facts immediately during `add_events_to_memory`.
      - SLEEP_TIME: Hot path (<10ms, zero LLM cost) only scrubs PII and embeds raw episodes
        with `extraction_pending=True`; `run_memory_maintenance()` batch-extracts the knowledge
        graph asynchronously every N turns or during idle sleep-time compute windows.
    """

    HOT = "hot"
    SLEEP_TIME = "sleep_time"


class RelationRule(BaseModel):
    """TBox relationship predicate definition with domain, range, and cardinality rules."""

    predicate: str
    domain_class: str = "owl:Thing"
    range_class: str = "owl:Thing"
    is_functional: bool = False  # True = at most 1 active target (triggers bi-temporal invalidation)
    is_transitive: bool = False  # True = transitive closure traversal supported
    description: str = ""


class DomainPackSpec(BaseModel):
    """Layer 1 Federated Domain Pack published by domain stewards (e.g., payments@v2.1)."""

    pack_id: str  # e.g., "payments@v2.1"
    domain_name: str  # e.g., "payments"
    version: str = "1.0.0"
    classes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of class_name -> parent_class_name (e.g., {'Microservice': 'p:Object'})",
    )
    disjoint_classes: list[tuple[str, str]] = Field(default_factory=list)
    relations: dict[str, RelationRule] = Field(default_factory=dict)
    synonyms: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Optional mapping of canonical class/predicate name -> business synonyms (e.g. from Apache Ossie)",
    )
    certified: bool = Field(
        default=False,
        description="Whether this Domain Pack has been certified by a domain expert or ontology steward",
    )
    certified_by: str | None = None


class AppMemoryManifest(BaseModel):
    """
    Developer-facing Ontology-as-Code registration manifest for any ADK agent app.
    Supports zero-config (Tier 0), app-scoped Pydantic types (Tier 1), and shared Domain Packs (Tier 2).
    """

    tenant_id: str
    app_name: str
    governance_tier: GovernanceTier = GovernanceTier.TIER0_AUTO
    subscribed_domain_packs: list[str] = Field(default_factory=list)
    custom_classes: dict[str, str] = Field(
        default_factory=dict,
        description="App-local classes mapped to UpperConceptType or parent class",
    )
    custom_relations: list[RelationRule] = Field(default_factory=list)
    allow_emergent_overflow: bool = True
    min_domain_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    extraction_mode: ExtractionMode = ExtractionMode.HOT
    max_rpm: int = 600
    max_active_vertices: int = 500_000


def _default_layer0_relations() -> dict[str, RelationRule]:
    """
    Battle-tested Layer 0 Universal Relations active out-of-the-box for every agent
    (even in Zero-Config TIER0_AUTO mode without any Domain Pack).
    """
    return {
        "RELATED_TO": RelationRule(
            predicate="RELATED_TO",
            domain_class="owl:Thing",
            range_class="owl:Thing",
            is_functional=False,
            description="General associative link or quarantined overflow relation",
        ),
        "PREFERS": RelationRule(
            predicate="PREFERS",
            domain_class="owl:Thing",
            range_class="owl:Thing",
            is_functional=False,
            description="User-scoped preference shared across apps for the same user",
        ),
        "HAS_STATUS": RelationRule(
            predicate="HAS_STATUS",
            domain_class="owl:Thing",
            range_class="p:Concept",
            is_functional=True,
            description="Current lifecycle status (functional: supersedes prior status)",
        ),
        "ASSIGNED_TO": RelationRule(
            predicate="ASSIGNED_TO",
            domain_class="owl:Thing",
            range_class="p:Actor",
            is_functional=True,
            description="Current primary assignee/owner (functional: supersedes prior assignee)",
        ),
        "OWNED_BY": RelationRule(
            predicate="OWNED_BY",
            domain_class="owl:Thing",
            range_class="p:Actor",
            is_functional=True,
            description="Primary owning team or actor (functional)",
        ),
        "LOCATED_IN": RelationRule(
            predicate="LOCATED_IN",
            domain_class="owl:Thing",
            range_class="p:Object",
            is_functional=True,
            description="Primary location or deployment environment (functional)",
        ),
        "DEPENDS_ON": RelationRule(
            predicate="DEPENDS_ON",
            domain_class="owl:Thing",
            range_class="owl:Thing",
            is_functional=False,
            is_transitive=True,
            description="Upstream dependency (transitive across multi-hop chains)",
        ),
        "PART_OF": RelationRule(
            predicate="PART_OF",
            domain_class="owl:Thing",
            range_class="owl:Thing",
            is_functional=False,
            is_transitive=True,
            description="Compositional hierarchy (transitive)",
        ),
        "CAUSED_BY": RelationRule(
            predicate="CAUSED_BY",
            domain_class="p:Event",
            range_class="owl:Thing",
            is_functional=False,
            description="Root-cause causal link from an Event to an Object/Event/Actor",
        ),
    }


class CompiledTBox(BaseModel):
    """Compiled Terminological Box governing an app's runtime extraction and SHACL-lite validation."""

    version: str = "1.0.0"
    governance_tier: GovernanceTier = GovernanceTier.TIER0_AUTO
    extraction_mode: ExtractionMode = ExtractionMode.HOT
    allow_emergent_overflow: bool = True
    min_domain_confidence: float = 0.85
    classes: dict[str, str] = Field(
        default_factory=lambda: {c.value: "owl:Thing" for c in UpperConceptType}
    )
    disjoint_classes: list[tuple[str, str]] = Field(default_factory=list)
    relations: dict[str, RelationRule] = Field(default_factory=_default_layer0_relations)

    def is_subclass_of(self, child_cls: str, target_parent_cls: str) -> bool:
        """Checks RDFS subClassOf hierarchy transitively (e.g. Microservice -> SoftwareAsset -> Asset -> p:Object)."""
        if target_parent_cls in ("owl:Thing", child_cls):
            return True
        visited: set[str] = set()
        curr = child_cls
        while curr in self.classes and curr not in visited:
            visited.add(curr)
            parent = self.classes[curr]
            if parent == target_parent_cls or parent == "owl:Thing" and target_parent_cls == "owl:Thing":
                return True
            curr = parent
        return False

    def resolve_upper_class(self, cls_name: str) -> str:
        """Resolves the root Layer 0 UpperConceptType for any class in the hierarchy."""
        upper_values = {c.value for c in UpperConceptType}
        if cls_name in upper_values:
            return cls_name
        visited: set[str] = set()
        curr = cls_name
        while curr in self.classes and curr not in visited:
            visited.add(curr)
            parent = self.classes[curr]
            if parent in upper_values:
                return parent
            curr = parent
        return UpperConceptType.CONCEPT.value


class PlatformTenantContext(BaseModel):
    """Resolved security, namespace, and ontology context for a single ADK invocation."""

    tenant_id: str
    app_name: str
    user_id: str
    app_private_namespace: str
    write_namespace: str
    read_namespaces: list[str]
    tbox: CompiledTBox
    max_rpm: int = 600
    max_active_vertices: int = 500_000

    @property
    def user_private_namespace(self) -> str:
        """Cross-app user namespace (`user:{tenant_id}:{user_id}`) shared across apps for this specific user."""
        return f"user:{self.tenant_id}:{self.user_id}"


class EpisodicRecord(BaseModel):
    """Tier 1 Immutable Episodic Memory record."""

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    app_name: str
    user_id: str
    session_id: str
    namespace: str
    author: str
    content_text: str
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedEntity(BaseModel):
    """Candidate entity extracted from conversation or tool output."""

    canonical_name: str = Field(description="Normalized canonical entity name")
    entity_class: str = Field(description="Target Ontology TBox class name")
    summary: str = Field(description="1-2 sentence atomic description of the entity")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    scope: MemoryScope = Field(
        default=MemoryScope.AUTO,
        description="Target sharing scope (auto, user_global, app_private, domain_shared)",
    )


class ExtractedFact(BaseModel):
    """Candidate relationship fact extracted from conversation or tool output."""

    source_entity: str
    source_class: str
    predicate: str
    target_entity: str
    target_class: str
    fact_summary: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    valid_from: str | None = Field(
        default=None,
        validation_alias=AliasChoices("valid_from", "valid_from_iso", "t_valid"),
    )
    scope: MemoryScope = Field(
        default=MemoryScope.AUTO,
        description="Target sharing scope (auto, user_global, app_private, domain_shared)",
    )


class ExtractionPayload(BaseModel):
    """
    Complete structured output from an OntologyExtractorAdapter.
    Combines LLM-based turn classification (`is_ephemeral_chitchat`) with
    scoped entity and bi-temporal fact extraction in a single LLM pass.
    """

    is_ephemeral_chitchat: bool = Field(
        default=False,
        description=(
            "True if the transcript is purely social pleasantries, greetings, or terminal sign-offs "
            "with zero durable factual, procedural, or preference value."
        ),
    )
    classification_reason: str = Field(
        default="",
        description="Brief explanation if classified as ephemeral chitchat or special scope",
    )
    entities: list[ExtractedEntity] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)


class GraphVertexUpsert(BaseModel):
    """Canonical vertex representation for both the Seed Index and Graph KB."""

    vertex_id: str
    tenant_id: str
    namespace: str
    entity_class: str
    upper_class: str
    canonical_name: str
    summary: str
    embedding: list[float] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def deterministic_id(
        tenant_id: str, namespace: str, entity_class: str, canonical_name: str
    ) -> str:
        """
        Deterministic UUIDv5/SHA-256 canonical vertex ID.
        Prevents duplicate vertex creation under concurrent multi-agent race conditions.
        """
        norm = f"{tenant_id}:{namespace}:{entity_class}:{canonical_name.strip().lower()}"
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
        return f"v_{tenant_id}_{digest}"


class GraphEdgeMutation(BaseModel):
    """Bi-temporal directed relationship edge mutation in the ABox."""

    edge_id: str
    tenant_id: str
    namespace: str
    source_vertex_id: str
    target_vertex_id: str
    predicate: str
    fact_summary: str
    is_functional: bool
    confidence: float
    source_episode_id: str
    source_app: str
    user_id: str
    t_valid: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    t_invalid: str = "ACTIVE"
    t_created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reinforcement_count: int = 1


class GraphFactReadResult(BaseModel):
    """Active or point-in-time historical edge returned during subgraph expansion."""

    source_name: str
    source_class: str = "p:Concept"
    predicate: str
    target_name: str
    target_class: str = "p:Concept"
    fact_summary: str
    t_valid: str
    t_invalid: str
    namespace: str
    source_episode_id: str = ""
    confidence: float = 1.0
    reinforcement_count: int = 1


class ReasoningTraceStep(BaseModel):
    """Tier 3 Procedural Memory step capturing an ADK agent's tool call and touched entities."""

    step_index: int
    tool_name: str
    tool_input_summary: str
    tool_output_summary: str
    touched_vertex_ids: list[str] = Field(default_factory=list)
    status: Literal["SUCCESS", "ERROR", "FALLBACK"] = "SUCCESS"


class PlaybookRecord(BaseModel):
    """
    Redis Iris-inspired Tier 3 Semantic Playbook Record.
    Caches proven, reusable multi-step resolution workflows and tool sequences
    so agents can resolve recurring intents with lower latency and fewer LLM tokens.
    """

    playbook_id: str = Field(default_factory=lambda: f"pb_{uuid.uuid4().hex[:12]}")
    tenant_id: str
    namespace: str
    app_name: str
    trigger_intent: str
    workflow_steps: list[str] = Field(default_factory=list)
    resolution_summary: str
    embedding: list[float] = Field(default_factory=list)
    success_count: int = 1
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OutboxOperation(BaseModel):
    """Transactional Outbox item ensuring eventual consistency between Vector/Seed DB and Graph DB."""

    outbox_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    operation_type: Literal["UPSERT_GRAPH_BATCH", "RECORD_TRACE", "PURGE_LINEAGE"]
    payload: dict[str, Any]
    status: str = "PENDING"
    attempts: int = 0


class OntologyAuditLogEntry(BaseModel):
    """
    Immutable Schema Provenance record tracking every ontology registration,
    LLM-assisted review modification, and bottom-up overflow evolution.
    """

    audit_id: str = Field(default_factory=lambda: f"aud_{uuid.uuid4().hex[:12]}")
    tenant_id: str
    target_id: str  # pack_id or app_name
    action: Literal[
        "REGISTER_DOMAIN_PACK",
        "REGISTER_APP_MANIFEST",
        "LLM_REVIEW_AND_REFINE",
        "AUTO_EVOLVE_FROM_OVERFLOW",
        "CERTIFY_DOMAIN_PACK",
        "INDUCE_DOMAIN_PACK",
    ]
    previous_version: str = "0.0.0"
    new_version: str = "1.0.0"
    added_or_modified_classes: dict[str, str] = Field(default_factory=dict)
    added_or_modified_relations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    modifications_summary: list[str] = Field(default_factory=list)
    triggered_by: str = "platform_steward"
    source_episode_ids: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
