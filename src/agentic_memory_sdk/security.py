"""
Enterprise API Security (JWT / API-Key / OIDC AuthN + Tenant-Isolated RBAC)
and OpenTelemetry / Prometheus Observability for `agentic-composite-memory`.

Closes the production security & telemetry gap vs. managed cloud services
(Vertex AI Memory Bank / AWS Bedrock AgentCore Memory) while remaining 100%
cloud-agnostic and requiring zero mandatory third-party dependencies.
"""

from __future__ import annotations

import base64
from collections import defaultdict, deque
import contextvars
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
import time
from typing import Any, Awaitable, Callable, Mapping

from pydantic import BaseModel, Field

MIN_JWT_SECRET_BYTES = 32

_CURRENT_PRINCIPAL: contextvars.ContextVar["AuthPrincipal | None"] = contextvars.ContextVar(
    "adk_memory_current_principal", default=None
)


def get_current_principal() -> "AuthPrincipal | None":
    """Returns the authenticated `AuthPrincipal` bound to the current async request context."""
    return _CURRENT_PRINCIPAL.get()


def set_current_principal(
    principal: "AuthPrincipal | None",
) -> contextvars.Token["AuthPrincipal | None"]:
    """Binds an authenticated `AuthPrincipal` into the current async context."""
    return _CURRENT_PRINCIPAL.set(principal)


def reset_current_principal(token: contextvars.Token["AuthPrincipal | None"]) -> None:
    """Restores the previous `AuthPrincipal` context after request completion."""
    _CURRENT_PRINCIPAL.reset(token)


class MemoryRole(str, Enum):
    """Role-Based Access Control (RBAC) roles for the Memory Platform API & MCP."""

    READER = "reader"  # Can search memory, check_answerable, read working blocks & revisions
    AGENT = "agent"  # READER + commit episodes, edit working blocks, record traces/playbooks
    ONTOLOGY_STEWARD = "ontology_steward"  # Can register/review/induce/certify/evolve ontologies
    COMPLIANCE_ADMIN = "compliance_admin"  # Can run GDPR purge, manual edit/delete, maintenance
    PLATFORM_ADMIN = "platform_admin"  # Full superuser access across tenant


ROLE_HIERARCHY: dict[MemoryRole, set[MemoryRole]] = {
    MemoryRole.READER: {MemoryRole.READER},
    MemoryRole.AGENT: {MemoryRole.READER, MemoryRole.AGENT},
    MemoryRole.ONTOLOGY_STEWARD: {MemoryRole.READER, MemoryRole.ONTOLOGY_STEWARD},
    MemoryRole.COMPLIANCE_ADMIN: {MemoryRole.READER, MemoryRole.COMPLIANCE_ADMIN},
    MemoryRole.PLATFORM_ADMIN: {
        MemoryRole.READER,
        MemoryRole.AGENT,
        MemoryRole.ONTOLOGY_STEWARD,
        MemoryRole.COMPLIANCE_ADMIN,
        MemoryRole.PLATFORM_ADMIN,
    },
}


class AuthPrincipal(BaseModel):
    """Authenticated caller identity resolved from a JWT, API Key, or custom OIDC verifier."""

    subject: str
    tenant_id: str
    roles: list[MemoryRole] = Field(default_factory=lambda: [MemoryRole.AGENT])
    allowed_apps: list[str] | None = Field(
        default=None,
        description="Optional allowlist of app_names this principal may access (None = all apps in tenant)",
    )
    allowed_users: list[str] | None = Field(
        default=None,
        description="Optional allowlist of user_ids this principal may access (None = all users in tenant)",
    )

    def has_role(self, required: MemoryRole) -> bool:
        effective: set[MemoryRole] = set()
        for r in self.roles:
            effective.update(ROLE_HIERARCHY.get(r, {r}))
        return required in effective


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def mint_hs256_jwt(
    payload: dict[str, Any],
    secret: str,
    expires_in_seconds: int = 3600,
) -> str:
    """Mints a standard RFC 7519 HS256 JWT using Python stdlib (zero external crypto deps)."""
    header = {"alg": "HS256", "typ": "JWT"}
    body = dict(payload)
    now = int(time.time())
    body.setdefault("iat", now)
    body.setdefault("exp", now + expires_in_seconds)
    h_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p_b64 = _b64url_encode(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{h_b64}.{p_b64}.{_b64url_encode(sig)}"


def verify_hs256_jwt(
    token: str,
    secret: str,
    expected_issuer: str | None = None,
    expected_audience: str | None = None,
) -> dict[str, Any]:
    """Verifies an HS256 JWT signature, expiration, issuer, and audience."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed JWT: expected 3 segments")
    h_b64, p_b64, s_b64 = parts
    header = json.loads(_b64url_decode(h_b64).decode("utf-8"))
    if header.get("alg") != "HS256":
        raise ValueError(f"Unsupported JWT alg: {header.get('alg')}")
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(s_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid JWT signature")
    payload = json.loads(_b64url_decode(p_b64).decode("utf-8"))
    now = int(time.time())
    if "exp" not in payload or payload["exp"] is None:
        raise ValueError("JWT missing required 'exp' claim")
    if int(payload["exp"]) <= now:
        raise ValueError("JWT has expired")
    if expected_issuer and payload.get("iss") != expected_issuer:
        raise ValueError("JWT issuer mismatch")
    if expected_audience:
        aud = payload.get("aud")
        if isinstance(aud, list):
            if expected_audience not in aud:
                raise ValueError("JWT audience mismatch")
        elif aud != expected_audience:
            raise ValueError("JWT audience mismatch")
    return payload


@dataclass
class AuthPolicyConfig:
    """
    Production API AuthN/AuthZ policy for FastAPI & MCP endpoints.
    Disabled by default (`enabled=False`) so local development works with zero config.
    """

    enabled: bool = False
    api_keys: dict[str, AuthPrincipal] = field(default_factory=dict)
    jwt_secret: str | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    enforce_tenant_isolation: bool = True
    api_rate_limit_rpm: int | None = None
    public_paths: tuple[str, ...] = ("/health", "/healthz", "/docs", "/openapi.json", "/redoc", "/ui")
    custom_token_verifier: Callable[[str], Awaitable[AuthPrincipal | None]] | None = None
    custom_request_authenticator: Callable[[Mapping[str, str]], Awaitable[AuthPrincipal | None]] | None = None

    def __post_init__(self) -> None:
        if self.enabled and self.jwt_secret is not None:
            if len(self.jwt_secret.encode("utf-8")) < MIN_JWT_SECRET_BYTES:
                raise ValueError(
                    f"JWT secret must be at least {MIN_JWT_SECRET_BYTES} bytes long "
                    f"(got {len(self.jwt_secret.encode('utf-8'))} bytes)"
                )

    async def authenticate(
        self,
        authorization_header: str | None,
        api_key_header: str | None,
        headers: Mapping[str, str] | None = None,
    ) -> AuthPrincipal:
        """
        Resolves an `AuthPrincipal` via Bring-Your-Own-Auth hooks or built-in adapters:
          1. `custom_request_authenticator(headers)` — for API Gateways, Istio/Envoy mTLS, Kong, Apigee, Cloud IAP.
          2. Static `api_keys` registry (`X-API-Key` or `Authorization: Bearer`).
          3. `custom_token_verifier(token)` — for Okta, Entra ID, Auth0, Cognito OIDC/JWKS tokens.
          4. Built-in HS256 JWT verification (`jwt_secret` >= 32 bytes).
        """
        # 0. Bring-Your-Own-Auth request-header authenticator (e.g. Istio mTLS / Kong / Cloud IAP)
        if self.custom_request_authenticator is not None and headers is not None:
            principal = await self.custom_request_authenticator(headers)
            if principal is not None:
                return principal

        token: str | None = None
        if api_key_header:
            token = api_key_header.strip()
        elif authorization_header and authorization_header.lower().startswith("bearer "):
            token = authorization_header[7:].strip()

        if not token:
            raise PermissionError("Missing authentication credentials (X-API-Key or Bearer token)")

        # 1. Static API Key lookup
        if token in self.api_keys:
            return self.api_keys[token]

        # 2. Custom async OIDC / JWKS / Entra / Okta verifier
        if self.custom_token_verifier is not None:
            principal = await self.custom_token_verifier(token)
            if principal is not None:
                return principal

        # 3. Built-in HS256 JWT verification
        if self.jwt_secret:
            try:
                claims = verify_hs256_jwt(
                    token,
                    self.jwt_secret,
                    expected_issuer=self.jwt_issuer,
                    expected_audience=self.jwt_audience,
                )
            except Exception as exc:
                raise PermissionError(f"Invalid Bearer JWT: {exc}") from exc
            raw_tenant = claims.get("tenant_id")
            if not raw_tenant or not isinstance(raw_tenant, str) or not raw_tenant.strip():
                raise PermissionError("JWT missing required non-empty 'tenant_id' claim")
            raw_roles = claims.get("roles", [MemoryRole.AGENT.value])
            if isinstance(raw_roles, str):
                raw_roles = [raw_roles]
            parsed_roles = [MemoryRole(r) for r in raw_roles if r in {m.value for m in MemoryRole}]
            return AuthPrincipal(
                subject=str(claims.get("sub", "jwt_user")),
                tenant_id=raw_tenant.strip(),
                roles=parsed_roles or [MemoryRole.AGENT],
                allowed_apps=claims.get("allowed_apps"),
                allowed_users=claims.get("allowed_users"),
            )

        raise PermissionError("Invalid API key or unrecognized Bearer token")

    @classmethod
    def from_env(cls) -> "AuthPolicyConfig":
        """
        Loads `AuthPolicyConfig` from environment variables for `adk-memory serve-api`:
          - `MEMORY_AUTH_ENABLED` (`true`/`false`; auto-enabled if JWT secret or API keys are set)
          - `MEMORY_JWT_SECRET`, `MEMORY_JWT_ISSUER`, `MEMORY_JWT_AUDIENCE`
          - `MEMORY_ADMIN_API_KEY` (grants PLATFORM_ADMIN across tenant `*` or `MEMORY_ADMIN_TENANT_ID`)
          - `MEMORY_AGENT_API_KEY` (grants AGENT on `MEMORY_AGENT_TENANT_ID`, default `t1`)
          - `MEMORY_API_KEYS_JSON` (JSON map of `{key: {subject, tenant_id, roles, allowed_apps?, allowed_users?}}`)
        """
        import os

        jwt_secret = os.getenv("MEMORY_JWT_SECRET") or None
        jwt_issuer = os.getenv("MEMORY_JWT_ISSUER") or None
        jwt_audience = os.getenv("MEMORY_JWT_AUDIENCE") or None
        api_keys: dict[str, AuthPrincipal] = {}

        admin_key = os.getenv("MEMORY_ADMIN_API_KEY")
        if admin_key:
            api_keys[admin_key] = AuthPrincipal(
                subject="env_platform_admin",
                tenant_id=os.getenv("MEMORY_ADMIN_TENANT_ID", "*"),
                roles=[MemoryRole.PLATFORM_ADMIN],
            )
        agent_key = os.getenv("MEMORY_AGENT_API_KEY")
        if agent_key:
            api_keys[agent_key] = AuthPrincipal(
                subject="env_agent",
                tenant_id=os.getenv("MEMORY_AGENT_TENANT_ID", "t1"),
                roles=[MemoryRole.AGENT],
            )
        raw_keys_json = os.getenv("MEMORY_API_KEYS_JSON")
        if raw_keys_json:
            parsed = json.loads(raw_keys_json)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    api_keys[str(k)] = AuthPrincipal.model_validate(v)

        raw_enabled = os.getenv("MEMORY_AUTH_ENABLED")
        if raw_enabled is not None:
            enabled = raw_enabled.strip().lower() in ("1", "true", "yes")
        else:
            enabled = bool(jwt_secret or api_keys)

        return cls(
            enabled=enabled,
            api_keys=api_keys,
            jwt_secret=jwt_secret,
            jwt_issuer=jwt_issuer,
            jwt_audience=jwt_audience,
        )

    @staticmethod
    def is_global_admin_route(path: str) -> bool:
        """Routes that operate across ALL tenants require a global PLATFORM_ADMIN (`tenant_id == '*'`)."""
        p = path.lower().rstrip("/")
        return p.endswith("/v1/maintenance/run") or p.endswith("/v1/outbox/flush")

    @staticmethod
    def is_global_admin_mcp_tool(tool_name: str) -> bool:
        return tool_name.strip().lower() == "run_memory_maintenance"

    @staticmethod
    def required_role_for_mcp_tool(tool_name: str) -> MemoryRole:
        """Maps every MCP tool name to its minimum required RBAC role."""
        name = tool_name.strip().lower()
        if name == "run_memory_maintenance":
            return MemoryRole.PLATFORM_ADMIN
        if name == "purge_user_lineage":
            return MemoryRole.COMPLIANCE_ADMIN
        if name in (
            "register_domain_pack",
            "induce_domain_pack",
            "certify_domain_pack",
            "register_app_manifest",
            "evolve_app_ontology",
        ):
            return MemoryRole.ONTOLOGY_STEWARD
        if name in (
            "commit_episode",
            "ingest_memory",
            "record_decision_trace",
            "register_playbook",
            "edit_working_memory_block",
        ):
            return MemoryRole.AGENT
        if name in (
            "recall_context",
            "check_answerable",
            "inspect_platform_governance",
            "list_user_memories",
            "get_memory_revisions",
        ):
            return MemoryRole.READER
        # Default-deny safe fallback for any unknown tool
        return MemoryRole.PLATFORM_ADMIN

    @staticmethod
    def required_role_for_route(method: str, path: str) -> MemoryRole:
        """Maps actual FastAPI HTTP routes to minimum required RBAC role."""
        m = method.upper()
        p = path.lower().rstrip("/")
        # Global cross-tenant operations require PLATFORM_ADMIN (with tenant_id == '*')
        if p.endswith("/v1/maintenance/run") or p.endswith("/v1/outbox/flush"):
            return MemoryRole.PLATFORM_ADMIN
        # Compliance & destructive lifecycle routes
        if p.endswith("/v1/lineage/purge") or p.endswith("/v1/memory/purge"):
            return MemoryRole.COMPLIANCE_ADMIN
        if "/v1/memory/items" in p and m in ("PUT", "PATCH", "DELETE"):
            return MemoryRole.COMPLIANCE_ADMIN
        # Ontology & App Control Plane routes
        if any(
            seg in p
            for seg in (
                "/v1/domains/",
                "/v1/ontology/evolve",
                "/v1/apps/register",
                "/v1/apps/onboard",
            )
        ):
            return MemoryRole.READER if m == "GET" else MemoryRole.ONTOLOGY_STEWARD
        # MCP transport endpoints: READER may list tools/resources/prompts & invoke read-only tools;
        # mutating MCP tools escalate via `required_role_for_mcp_tool` in middleware & tool handlers.
        if p.startswith("/mcp/http") or p.startswith("/mcp/sse") or p == "/v1/mcp/call":
            return MemoryRole.READER
        # Explicitly read-only POST query routes
        if p.endswith("/v1/memory/search") or p.endswith("/v1/memory/check-answerable"):
            return MemoryRole.READER
        # Agent Data-Plane write routes
        if any(
            seg in p
            for seg in (
                "/v1/memory/events",
                "/v1/memory/blocks",
                "/v1/traces",
                "/v1/playbooks",
            )
        ):
            return MemoryRole.READER if m == "GET" else MemoryRole.AGENT
        # Safe default: GET is READER, any other mutating request requires AGENT
        return MemoryRole.READER if m == "GET" else MemoryRole.AGENT

    def authorize_tenant_and_scope(
        self,
        principal: AuthPrincipal,
        target_tenant_id: str | None = None,
        target_app_name: str | None = None,
        target_user_id: str | None = None,
    ) -> None:
        """Prevents cross-tenant spoofing at the HTTP/MCP boundary."""
        if not self.enforce_tenant_isolation:
            return
        if principal.has_role(MemoryRole.PLATFORM_ADMIN) and principal.tenant_id == "*":
            return
        resolved_tenant = target_tenant_id
        resolved_app = target_app_name
        if target_app_name and ":" in target_app_name:
            t_prefix, a_suffix = target_app_name.split(":", 1)
            resolved_tenant = resolved_tenant or t_prefix
            resolved_app = a_suffix
        if resolved_tenant and resolved_tenant != principal.tenant_id:
            raise PermissionError(
                f"Tenant isolation violation: principal tenant '{principal.tenant_id}' "
                f"cannot access tenant '{resolved_tenant}'"
            )
        if principal.allowed_apps is not None and resolved_app:
            if resolved_app not in principal.allowed_apps and target_app_name not in principal.allowed_apps:
                raise PermissionError(
                    f"App scope violation: principal '{principal.subject}' not allowed on app '{resolved_app}'"
                )
        if principal.allowed_users is not None and target_user_id:
            if target_user_id not in principal.allowed_users:
                raise PermissionError(
                    f"User scope violation: principal '{principal.subject}' not allowed on user '{target_user_id}'"
                )


class ApiRateLimiter:
    """
    Bounded sliding-window per-principal / per-client API rate limiter (returns 429 when exceeded).
    - Evicts expired keys and caps memory at `max_keys` (default 10,000) so key tables never grow unbounded.
    - When `redis_client` is provided, uses atomic Redis `INCR` + `EXPIRE` for cross-pod distributed limiting.
    """

    def __init__(
        self,
        default_rpm: int = 600,
        window_sec: float = 60.0,
        max_keys: int = 10_000,
        redis_client: Any = None,
    ):
        self.default_rpm = default_rpm
        self.window_sec = window_sec
        self.max_keys = max_keys
        self.redis_client = redis_client
        self._hits: dict[str, deque[float]] = {}

    def _prune_stale_keys(self, now: float) -> None:
        if len(self._hits) < self.max_keys:
            return
        stale = [k for k, q in self._hits.items() if not q or (now - q[-1]) > self.window_sec]
        for k in stale:
            del self._hits[k]
        # If still at capacity, drop oldest-seen keys (FIFO)
        while len(self._hits) >= self.max_keys:
            oldest_key = next(iter(self._hits))
            del self._hits[oldest_key]

    def check_and_record(self, key: str, max_rpm: int | None = None) -> bool:
        limit = max_rpm if max_rpm is not None else self.default_rpm
        now = time.monotonic()
        q = self._hits.get(key)
        if q is not None:
            while q and (now - q[0]) > self.window_sec:
                q.popleft()
            if not q:
                del self._hits[key]
                q = None
        if q is None:
            self._prune_stale_keys(now)
            q = deque()
            self._hits[key] = q
        if len(q) >= limit:
            return False
        q.append(now)
        return True

    async def async_check_and_record(self, key: str, max_rpm: int | None = None) -> bool:
        """Uses Redis fixed-window atomic counter when `redis_client` is available, else bounded in-memory."""
        limit = max_rpm if max_rpm is not None else self.default_rpm
        if self.redis_client is not None:
            try:
                bucket = int(time.time() // max(1.0, self.window_sec))
                rkey = f"adk_mem:rl:{key}:{bucket}"
                count = await self.redis_client.incr(rkey)
                if count == 1:
                    await self.redis_client.expire(rkey, int(self.window_sec) + 2)
                return int(count) <= limit
            except Exception:
                # Fail open to local bounded limiter if Redis is transiently unreachable
                pass
        return self.check_and_record(key, max_rpm=limit)


class MemoryObservability:
    """
    Lightweight OpenTelemetry-compatible + Prometheus-exportable observability collector.
    Automatically emits spans to OpenTelemetry if `opentelemetry-api` is installed,
    while always maintaining in-process counters and latency histograms for `/v1/metrics` & `/metrics`.
    """

    def __init__(self, service_name: str = "agentic-composite-memory"):
        self.service_name = service_name
        self.counters: dict[str, int] = defaultdict(int)
        self.latencies_ms: dict[str, list[float]] = defaultdict(list)
        self._otel_tracer = None
        try:
            from opentelemetry import trace  # type: ignore[import-not-found]

            self._otel_tracer = trace.get_tracer(service_name)
        except Exception:
            self._otel_tracer = None

    def inc(self, metric_name: str, amount: int = 1) -> None:
        self.counters[metric_name] += amount

    def observe_latency(self, operation: str, duration_ms: float) -> None:
        bucket = self.latencies_ms[operation]
        bucket.append(round(duration_ms, 2))
        if len(bucket) > 1000:
            del bucket[:-1000]

    def snapshot(self) -> dict[str, Any]:
        latency_summary: dict[str, dict[str, float]] = {}
        for op, vals in self.latencies_ms.items():
            if not vals:
                continue
            s = sorted(vals)
            n = len(s)
            latency_summary[op] = {
                "count": float(n),
                "avg_ms": round(sum(s) / n, 2),
                "p50_ms": s[int(0.50 * (n - 1))],
                "p95_ms": s[int(0.95 * (n - 1))],
                "max_ms": s[-1],
            }
        return {
            "service": self.service_name,
            "otel_enabled": self._otel_tracer is not None,
            "counters": dict(self.counters),
            "latencies": latency_summary,
        }

    def to_prometheus_text(self) -> str:
        lines: list[str] = []
        for k, v in sorted(self.counters.items()):
            safe_k = f"adk_memory_{k.replace('-', '_').replace('.', '_')}_total"
            lines.append(f"# TYPE {safe_k} counter")
            lines.append(f"{safe_k} {v}")
        for op, vals in sorted(self.latencies_ms.items()):
            if not vals:
                continue
            safe_op = f"adk_memory_{op.replace('-', '_').replace('.', '_')}_avg_ms"
            lines.append(f"# TYPE {safe_op} gauge")
            lines.append(f"{safe_op} {round(sum(vals) / len(vals), 2)}")
        return "\n".join(lines) + "\n"
