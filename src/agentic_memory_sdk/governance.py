from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from typing import Tuple

from .models import (
    AppMemoryManifest,
    CompiledTBox,
    DomainPackSpec,
    EpisodicRecord,
    ExtractedEntity,
    ExtractedFact,
    ExtractionPayload,
    GovernanceTier,
    GraphFactReadResult,
    MemoryScope,
    OntologyAuditLogEntry,
    PlatformTenantContext,
    RelationRule,
    UpperConceptType,
    _default_layer0_relations,
)


class PIIScrubber:
    """Deterministic pre-embedding and pre-extraction DLP sanitizer."""

    _PATTERNS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_PAN]"),
        (
            re.compile(
                r"(?i)\bBearer\s+[A-Za-z0-9_\-\.=]{12,}"
            ),
            "[REDACTED_SECRET]",
        ),
        (
            re.compile(
                r"(?i)(?:api[_-]?key|secret|token|password|Bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{12,}['\"]?"
            ),
            "[REDACTED_SECRET]",
        ),
        (
            re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
            "[REDACTED_API_KEY]",
        ),
    ]

    @classmethod
    def scrub(cls, text: str) -> str:
        scrubbed = text
        for pattern, replacement in cls._PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)
        return scrubbed


class EntityResolutionGuard:
    """
    Prevents false-positive vector entity merges between distinct numbered, directional,
    or environment-scoped identifiers (e.g., `us-east-1` vs `us-east-2`, `aurora-01` vs `aurora-02`,
    `prod-db` vs `staging-db`) even when dense embedding cosine similarity is >= 0.91.
    """

    _DIRECTION_OR_ENV_TOKENS = {
        "east",
        "west",
        "north",
        "south",
        "central",
        "prod",
        "production",
        "stage",
        "staging",
        "dev",
        "qa",
        "primary",
        "replica",
    }

    @classmethod
    def normalize_name(cls, name: str) -> str:
        return re.sub(r"[\s_\-]+", "-", name.strip().lower())

    @classmethod
    def extract_distinguishing_tokens(cls, name: str) -> tuple[set[str], set[str]]:
        norm = cls.normalize_name(name)
        # Extract normalized numeric tokens (e.g. '01' -> '1', '2' -> '2')
        num_tokens = {str(int(m)) for m in re.findall(r"\d+", norm)}
        words = set(norm.split("-"))
        env_tokens = words.intersection(cls._DIRECTION_OR_ENV_TOKENS)
        return num_tokens, env_tokens

    @classmethod
    def can_merge(
        cls,
        existing_name: str,
        candidate_name: str,
        cosine_sim: float,
        threshold: float = 0.91,
    ) -> bool:
        if cls.normalize_name(existing_name) == cls.normalize_name(candidate_name):
            return True

        # Check for conflicting numbers or directional/environment qualifiers
        nums_a, envs_a = cls.extract_distinguishing_tokens(existing_name)
        nums_b, envs_b = cls.extract_distinguishing_tokens(candidate_name)

        if (nums_a or nums_b) and nums_a != nums_b:
            return False
        if (envs_a or envs_b) and envs_a != envs_b:
            return False

        return cosine_sim >= threshold


class SmartMemoryClassifier:
    """
    Distinguishes memory types automatically at ingestion time:
      1. Ephemeral Chitchat / Noise (`is_ephemeral_turn`) -> Skips graph extraction and flags for TTL cleanup.
      2. Cross-App User Preferences (`is_user_scoped_entity` / `is_user_scoped_fact`) -> Routes to
         `user:{tenant}:{user_id}` so preferences follow the user across all apps without leaking to other users.
      3. App-Private vs Domain-Shared vs Tenant-Global -> Routed by `TBoxRuntimeValidator`.
    """

    _EPHEMERAL_EXACT = {
        "ok",
        "okay",
        "k",
        "thanks",
        "thank you",
        "thx",
        "got it",
        "sounds good",
        "cool",
        "bye",
        "goodbye",
        "hello",
        "hi",
        "hey",
        "ack",
        "noted",
    }

    _EPHEMERAL_WORDS = {
        "ok",
        "okay",
        "k",
        "thanks",
        "thank",
        "you",
        "thx",
        "got",
        "it",
        "sounds",
        "good",
        "great",
        "cool",
        "bye",
        "goodbye",
        "hello",
        "hi",
        "hey",
        "ack",
        "noted",
        "sure",
        "yes",
        "yep",
    }

    _USER_PREFERENCE_PREDICATES = {
        "PREFERS",
        "HAS_PREFERENCE",
        "HAS_TIMEZONE",
        "COMMUNICATION_STYLE",
        "SPEAKS_LANGUAGE",
        "WORKS_IN_ROLE",
    }

    _CONFIRMATION_WORDS = {
        "yes",
        "yep",
        "no",
        "nope",
        "sure",
        "ok",
        "okay",
        "correct",
        "confirmed",
        "approved",
    }

    @classmethod
    def is_ephemeral_turn(
        cls, text: str, preceding_text: str | None = None
    ) -> bool:
        """
        Context-aware fast gate:
          - Never drops short confirmations/negations ('yes', 'no', 'ok') when following a preceding
            turn/question (`preceding_text`), so the LLM extractor sees the full Q&A context!
          - Only flags unambiguous standalone terminal closings ('thanks!', 'bye') on the fast path,
            while deferring nuanced or multi-sentence chitchat classification to the LLM (`ExtractionPayload.is_ephemeral_chitchat`).
        """
        cleaned = re.sub(r"[.!?,:;\"']+", "", text.strip().lower())
        if not cleaned:
            return True
        words = cleaned.split()
        # If there is preceding context (e.g. an agent question), do NOT drop confirmations like "yes" or "ok"!
        if preceding_text and any(w in cls._CONFIRMATION_WORDS for w in words):
            return False
        if cleaned in cls._EPHEMERAL_EXACT:
            return True
        return len(words) <= 4 and all(w in cls._EPHEMERAL_WORDS for w in words)

    @classmethod
    def is_user_scoped_entity(cls, ent: ExtractedEntity, tbox: CompiledTBox) -> bool:
        if ent.scope == MemoryScope.USER_GLOBAL:
            return True
        return tbox.is_subclass_of(
            ent.entity_class, UpperConceptType.PREFERENCE.value
        )

    @classmethod
    def is_user_scoped_fact(
        cls, fact: ExtractedFact, user_preference_entities: set[str]
    ) -> bool:
        if fact.scope == MemoryScope.USER_GLOBAL:
            return True
        if fact.predicate in cls._USER_PREFERENCE_PREDICATES:
            return True
        return fact.target_entity in user_preference_entities


class TBoxRuntimeValidator:
    """
    Closed-world SHACL-equivalent runtime validator implementing Multi-Scope & Two-Bucket Routing.
    - Scope 0 (User-Global `user:{tenant}:{user_id}`): User preferences & persona attributes
      shared across all apps for that specific user, isolated from other users.
    - Bucket 1 (Governed `domain:*` or `app:*`): High-confidence entities and facts conforming
      to TBox classes, RDFS subClassOf hierarchies, disjointness rules, and domain/range constraints.
    - Bucket 2 (Emergent Overflow `app:{tenant}:{app}`): Schema violations or low-confidence facts
      are downgraded to `p:CandidateConcept` / `RELATED_TO` inside the app's private namespace.
    """

    @staticmethod
    def validate_and_partition(
        ctx: PlatformTenantContext,
        payload: ExtractionPayload,
    ) -> Tuple[list[Tuple[ExtractedEntity, str]], list[Tuple[ExtractedFact, str, bool]]]:
        tbox: CompiledTBox = ctx.tbox
        routed_entities: list[Tuple[ExtractedEntity, str]] = []
        routed_facts: list[Tuple[ExtractedFact, str, bool]] = []

        # Track resolved classes per canonical entity name (and detect disjointness collisions)
        entity_class_map: dict[str, str] = {}
        user_scoped_entities: set[str] = set()
        disjoint_pairs: set[tuple[str, str]] = set()
        for c1, c2 in tbox.disjoint_classes:
            disjoint_pairs.add((c1, c2))
            disjoint_pairs.add((c2, c1))

        for ent in payload.entities:
            ent_copy = ent.model_copy()
            if ent_copy.scope == MemoryScope.EPHEMERAL:
                continue

            is_valid_class = ent_copy.entity_class in tbox.classes
            meets_confidence = ent_copy.confidence >= tbox.min_domain_confidence

            # Disjointness check if entity was seen earlier in the same batch with a disjoint class
            prior_cls = entity_class_map.get(ent_copy.canonical_name)
            disjoint_violation = (
                prior_cls is not None and (prior_cls, ent_copy.entity_class) in disjoint_pairs
            )

            if SmartMemoryClassifier.is_user_scoped_entity(ent_copy, tbox):
                if not is_valid_class:
                    ent_copy.entity_class = UpperConceptType.PREFERENCE.value
                target_ns = ctx.user_private_namespace
                if tbox.is_subclass_of(
                    ent_copy.entity_class, UpperConceptType.PREFERENCE.value
                ):
                    user_scoped_entities.add(ent_copy.canonical_name)
            elif ent_copy.scope == MemoryScope.APP_PRIVATE:
                if not is_valid_class:
                    ent_copy.entity_class = UpperConceptType.CANDIDATE.value
                target_ns = ctx.app_private_namespace
            elif is_valid_class and meets_confidence and not disjoint_violation:
                target_ns = ctx.write_namespace
            elif tbox.allow_emergent_overflow:
                orig_cls = ent_copy.entity_class
                ent_copy.entity_class = UpperConceptType.CANDIDATE.value
                if orig_cls and orig_cls != UpperConceptType.CANDIDATE.value and "[ProposedClass=" not in ent_copy.summary:
                    ent_copy.summary = f"[ProposedClass={orig_cls}] {ent_copy.summary}"
                target_ns = ctx.app_private_namespace
            else:
                continue

            entity_class_map[ent_copy.canonical_name] = ent_copy.entity_class
            routed_entities.append((ent_copy, target_ns))

        for fact in payload.facts:
            fact_copy = fact.model_copy()
            if fact_copy.scope == MemoryScope.EPHEMERAL:
                continue

            if SmartMemoryClassifier.is_user_scoped_fact(fact_copy, user_scoped_entities):
                rel_rule = tbox.relations.get(fact_copy.predicate)
                is_func = rel_rule.is_functional if rel_rule else False
                routed_facts.append((fact_copy, ctx.user_private_namespace, is_func))
                continue

            if fact_copy.scope == MemoryScope.APP_PRIVATE:
                rel_rule = tbox.relations.get(fact_copy.predicate)
                is_func = rel_rule.is_functional if rel_rule else False
                routed_facts.append((fact_copy, ctx.app_private_namespace, is_func))
                continue

            rel_rule = tbox.relations.get(fact_copy.predicate)
            src_cls = entity_class_map.get(fact_copy.source_entity, fact_copy.source_class)
            tgt_cls = entity_class_map.get(fact_copy.target_entity, fact_copy.target_class)

            domain_ok = (
                rel_rule is not None
                and tbox.is_subclass_of(src_cls, rel_rule.domain_class)
                and tbox.is_subclass_of(tgt_cls, rel_rule.range_class)
            )
            confidence_ok = fact_copy.confidence >= tbox.min_domain_confidence

            if domain_ok and confidence_ok and rel_rule is not None:
                routed_facts.append((fact_copy, ctx.write_namespace, rel_rule.is_functional))
            elif tbox.allow_emergent_overflow:
                # Quarantine to app-private namespace; overflow facts never invalidate functional edges
                orig_pred = fact_copy.predicate
                fact_copy.predicate = "RELATED_TO"
                if orig_pred and orig_pred != "RELATED_TO" and "[ProposedPredicate=" not in fact_copy.fact_summary:
                    fact_copy.fact_summary = (
                        f"[ProposedPredicate={orig_pred}|{fact_copy.source_class}->{fact_copy.target_class}] "
                        f"{fact_copy.fact_summary}"
                    )
                routed_facts.append((fact_copy, ctx.app_private_namespace, False))

        return routed_entities, routed_facts


class PlatformOntologyRegistrar:
    """
    Compiles developer-friendly AppMemoryManifest declarations + Federated Domain Packs
    into a CompiledTBox and registers the ADK app in the platform RegistryStoreAdapter.
    """

    def __init__(self, registry_adapter):
        self.registry = registry_adapter
        self._domain_packs: dict[str, DomainPackSpec] = {}
        self.audit_log: list[OntologyAuditLogEntry] = []

    @staticmethod
    def infer_upper_concept_parent(class_name: str) -> str:
        """
        Semantic heuristic + Upper Ontology aligner: maps any unrooted domain class
        to the appropriate Layer 0 `UpperConceptType` (`p:Actor`, `p:Object`, `p:Event`, `p:UserPreference`, `p:Concept`).
        """
        low = class_name.lower()
        if any(k in low for k in ("user", "actor", "agent", "customer", "person", "team", "owner", "vendor", "patient", "investigator")):
            return UpperConceptType.ACTOR.value
        if any(k in low for k in ("event", "incident", "alert", "outage", "trial", "deployment", "transaction", "dispute", "call", "session")):
            return UpperConceptType.EVENT.value
        if any(k in low for k in ("pref", "setting", "style", "persona")):
            return UpperConceptType.PREFERENCE.value
        if any(k in low for k in ("service", "cluster", "database", "db", "tool", "asset", "system", "region", "device", "drug", "compound", "server", "api", "app")):
            return UpperConceptType.OBJECT.value
        return UpperConceptType.CONCEPT.value

    @staticmethod
    def infer_relation_flags(predicate: str) -> tuple[bool, bool]:
        """
        Infers `(is_functional, is_transitive)` when a service proposes an under-specified relation rule.
        """
        up = re.sub(r"[\s\-]+", "_", predicate.strip().upper())
        is_func = any(
            tok in up
            for tok in (
                "PRIMARY",
                "CURRENT",
                "ACTIVE_TIER",
                "SUBSCRIBED_TO_TIER",
                "HAS_STATUS",
                "ASSIGNED_TO",
                "OWNED_BY",
                "MANAGED_BY",
                "DEPLOYED_IN_PRIMARY",
            )
        )
        is_trans = any(
            tok in up
            for tok in (
                "DEPENDS_ON",
                "PART_OF",
                "SUBCLASS_OF",
                "UPSTREAM_OF",
                "DOWNSTREAM_OF",
            )
        )
        return is_func, is_trans

    def review_and_refine_domain_pack(
        self,
        pack: DomainPackSpec,
        tenant_id: str = "platform",
        auto_register: bool = True,
    ) -> tuple[DomainPackSpec, OntologyAuditLogEntry]:
        """
        Behind-the-scenes Intelligent Ontology Steward:
        Reviews a proposed `DomainPackSpec` (submitted by an app team or LLM),
        automatically fixes unrooted classes, normalizes predicate names to UPPER_SNAKE_CASE,
        infers missing `is_functional` (bi-temporal conflict invalidation) and `is_transitive` flags,
        repairs broken domain/range class references, and records an immutable `OntologyAuditLogEntry`.
        """
        refined = pack.model_copy(deep=True)
        modifications: list[str] = []
        upper_set = {c.value for c in UpperConceptType} | {"owl:Thing"}

        # 1. Ensure all classes are rooted in Layer 0 UpperConceptType
        for cls_name, parent_cls in list(refined.classes.items()):
            if not parent_cls or parent_cls in ("Thing", "owl:Thing", "None", ""):
                inferred_parent = self.infer_upper_concept_parent(cls_name)
                refined.classes[cls_name] = inferred_parent
                modifications.append(
                    f"Rooted unanchored class '{cls_name}' under Layer 0 '{inferred_parent}'"
                )
            elif parent_cls not in upper_set and parent_cls not in refined.classes:
                inferred_parent = self.infer_upper_concept_parent(parent_cls)
                refined.classes[parent_cls] = inferred_parent
                modifications.append(
                    f"Added missing intermediate parent class '{parent_cls}' -> '{inferred_parent}'"
                )

        # 2. Review and normalize relations
        new_relations: dict[str, RelationRule] = {}
        for raw_key, rel in refined.relations.items():
            norm_pred = re.sub(r"[\s\-]+", "_", (rel.predicate or raw_key).strip().upper())
            rel_copy = rel.model_copy(deep=True)
            if rel_copy.predicate != norm_pred:
                modifications.append(
                    f"Normalized predicate '{rel_copy.predicate}' -> '{norm_pred}'"
                )
                rel_copy.predicate = norm_pred

            # Ensure domain_class and range_class exist in classes or Layer 0
            for attr in ("domain_class", "range_class"):
                cls_val = getattr(rel_copy, attr)
                if cls_val not in upper_set and cls_val not in refined.classes:
                    inferred = self.infer_upper_concept_parent(cls_val)
                    refined.classes[cls_val] = inferred
                    modifications.append(
                        f"Auto-registered missing {attr} '{cls_val}' -> '{inferred}' referenced by '{norm_pred}'"
                    )

            inferred_func, inferred_trans = self.infer_relation_flags(norm_pred)
            if inferred_func and not rel_copy.is_functional:
                rel_copy.is_functional = True
                modifications.append(
                    f"Enabled bi-temporal functional invalidation (is_functional=True) on singular-state predicate '{norm_pred}'"
                )
            if inferred_trans and not rel_copy.is_transitive:
                rel_copy.is_transitive = True
                modifications.append(
                    f"Enabled multi-hop transitive closure (is_transitive=True) on predicate '{norm_pred}'"
                )

            new_relations[norm_pred] = rel_copy

        refined.relations = new_relations

        # Bump patch version if steward modified the proposed schema
        prev_ver = pack.version
        if modifications:
            parts = (pack.version.split(".") + ["0", "0"])[:3]
            try:
                parts[2] = str(int(parts[2]) + 1)
                refined.version = ".".join(parts)
            except ValueError:
                refined.version = f"{pack.version}-reviewed"

        audit_entry = OntologyAuditLogEntry(
            tenant_id=tenant_id,
            target_id=refined.pack_id,
            action="LLM_REVIEW_AND_REFINE",
            previous_version=prev_ver,
            new_version=refined.version,
            added_or_modified_classes=dict(refined.classes),
            added_or_modified_relations={
                k: v.model_dump() for k, v in refined.relations.items()
            },
            modifications_summary=modifications or ["Verified clean: no schema modifications required"],
            triggered_by="ontology_intelligence_steward",
        )
        self.audit_log.append(audit_entry)

        if auto_register:
            self.register_domain_pack(refined, record_audit=False)

        return refined, audit_entry

    def register_domain_pack(
        self, pack: DomainPackSpec, record_audit: bool = True
    ) -> None:
        """Registers a Layer 1 Federated Domain Pack (e.g. 'payments@v2.1') in the platform catalog."""
        self._domain_packs[pack.pack_id] = pack
        self._domain_packs[pack.domain_name] = pack
        if record_audit:
            self.audit_log.append(
                OntologyAuditLogEntry(
                    tenant_id="platform",
                    target_id=pack.pack_id,
                    action="REGISTER_DOMAIN_PACK",
                    previous_version="0.0.0",
                    new_version=pack.version,
                    added_or_modified_classes=dict(pack.classes),
                    added_or_modified_relations={
                        k: v.model_dump() for k, v in pack.relations.items()
                    },
                    modifications_summary=[f"Registered domain pack '{pack.pack_id}'"],
                    triggered_by="domain_steward",
                )
            )

    def compile_manifest(self, manifest: AppMemoryManifest) -> tuple[list[str], str, CompiledTBox]:
        """Computes read/write namespaces and merges Layer 0 + Layer 1 + Layer 2 TBox rules."""
        app_ns = f"app:{manifest.tenant_id}:{manifest.app_name}"
        global_ns = f"global:{manifest.tenant_id}:core"
        domain_namespaces = [
            f"domain:{manifest.tenant_id}:{pack.split('@')[0]}"
            for pack in manifest.subscribed_domain_packs
        ]

        read_namespaces = [app_ns, global_ns] + domain_namespaces
        # Deduplicate while preserving order
        read_namespaces = list(dict.fromkeys(read_namespaces))

        if (
            manifest.governance_tier == GovernanceTier.TIER2_DOMAIN_SHARED
            and domain_namespaces
        ):
            write_namespace = domain_namespaces[0]
        else:
            write_namespace = app_ns

        # Layer 0 Base Classes & Universal Relations
        classes: dict[str, str] = {c.value: "owl:Thing" for c in UpperConceptType}
        disjoint_classes: list[tuple[str, str]] = []
        relations: dict[str, RelationRule] = _default_layer0_relations()

        # Layer 1 Subscribed Domain Packs
        for pack_key in manifest.subscribed_domain_packs:
            pack = self._domain_packs.get(pack_key) or self._domain_packs.get(
                pack_key.split("@")[0]
            )
            if pack:
                classes.update(pack.classes)
                disjoint_classes.extend(pack.disjoint_classes)
                relations.update(pack.relations)

        # Layer 2 App-Local Custom Extensions
        for cls_name, parent_cls in manifest.custom_classes.items():
            parent_str = (
                parent_cls.value if isinstance(parent_cls, UpperConceptType) else str(parent_cls)
            )
            classes[cls_name] = parent_str

        for rel in manifest.custom_relations:
            relations[rel.predicate] = rel

        tbox = CompiledTBox(
            governance_tier=manifest.governance_tier,
            extraction_mode=manifest.extraction_mode,
            allow_emergent_overflow=manifest.allow_emergent_overflow,
            min_domain_confidence=manifest.min_domain_confidence,
            classes=classes,
            disjoint_classes=disjoint_classes,
            relations=relations,
        )
        return read_namespaces, write_namespace, tbox

    async def register_adk_app(self, manifest: AppMemoryManifest) -> CompiledTBox:
        """Compiles and persists an ADK app's governance policy and TBox in the RegistryStoreAdapter."""
        read_namespaces, write_namespace, tbox = self.compile_manifest(manifest)
        await self.registry.register_app(
            tenant_id=manifest.tenant_id,
            app_name=manifest.app_name,
            read_namespaces=read_namespaces,
            write_namespace=write_namespace,
            tbox=tbox,
        )
        return tbox


class AsyncCircuitBreaker:
    """Protects the ADK read path from Graph DB outages or latency spikes."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout_sec: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def allow_request(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if (time.monotonic() - self.last_failure_time) >= self.recovery_timeout_sec:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.monotonic()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"


class TenantRateLimiter:
    """Per-app sliding window rate limiter preventing runaway agent loops from starving the platform."""

    def __init__(self, window_sec: float = 60.0):
        self.window_sec = window_sec
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def check_and_record(self, tenant_id: str, app_name: str, max_rpm: int = 600) -> bool:
        now = time.monotonic()
        key = (tenant_id, app_name)
        q = self._hits[key]
        while q and (now - q[0]) > self.window_sec:
            q.popleft()
        if len(q) >= max_rpm:
            return False
        q.append(now)
        return True


class TokenBudgetSerializer:
    """
    Packs graph facts (ranked by HippoRAG-inspired associative salience = confidence * reinforcement boost,
    with Letta-inspired pinned user persona facts) and episodic chunks into a bounded token budget (~4 chars/token).
    """

    @staticmethod
    def compute_salience(fact: GraphFactReadResult) -> float:
        import math

        is_user_persona = fact.namespace.startswith("user:")
        persona_boost = 0.25 if is_user_persona else 0.0
        reinforcement_boost = 1.0 + 0.15 * math.log1p(max(0, fact.reinforcement_count - 1))
        return (fact.confidence * reinforcement_boost) + persona_boost

    @staticmethod
    def serialize(
        facts: list[GraphFactReadResult],
        episodes: list[EpisodicRecord],
        max_tokens: int = 1500,
        degraded_mode: bool = False,
    ) -> Tuple[str, list[EpisodicRecord]]:
        max_chars = max_tokens * 4
        used_chars = 0
        fact_lines: list[str] = []

        if degraded_mode:
            header = "[GRAPH_DEGRADED_FALLBACK_MODE: Serving Verified Seed Summaries]\n"
        else:
            header = "[ACTIVE_ONTOLOGY_GRAPH_FACTS]\n"
        used_chars += len(header)

        sorted_facts = sorted(
            facts, key=TokenBudgetSerializer.compute_salience, reverse=True
        )
        for f in sorted_facts:
            reinf_tag = (
                f" [verified={f.reinforcement_count}x]"
                if f.reinforcement_count > 1
                else ""
            )
            line = (
                f"- [{f.namespace}] ({f.source_name}) -[{f.predicate}]-> ({f.target_name}): "
                f"{f.fact_summary} [valid_from={f.t_valid}]{reinf_tag}\n"
            )
            if used_chars + len(line) > int(max_chars * 0.65):
                break
            fact_lines.append(line)
            used_chars += len(line)

        graph_block = (header + "".join(fact_lines)) if fact_lines else ""

        kept_episodes: list[EpisodicRecord] = []
        for ep in episodes:
            if used_chars + len(ep.content_text) > max_chars:
                remaining = max_chars - used_chars
                if remaining > 120:
                    ep_copy = ep.model_copy()
                    ep_copy.content_text = ep.content_text[:remaining] + "...[truncated]"
                    kept_episodes.append(ep_copy)
                break
            kept_episodes.append(ep)
            used_chars += len(ep.content_text)

        return graph_block, kept_episodes
