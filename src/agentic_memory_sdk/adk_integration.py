from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

import httpx

from .models import AppMemoryManifest, ReasoningTraceStep
from .service import BaseMemoryService, Event, MemoryEntry, SearchMemoryResponse, Session, types

logger = logging.getLogger(__name__)


class RemoteCompositeMemoryService(BaseMemoryService):
    """
    Lightweight HTTP Client implementing Google ADK's `BaseMemoryService` contract.
    Designed for platforms hosting 100s of ADK agent apps:
      - Agents do NOT need direct Postgres/OpenSearch/Neptune credentials or connection pools.
      - Plugs directly into `google.adk.runners.Runner(memory_service=RemoteCompositeMemoryService(...))`.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout_sec: float = 10.0,
        client: httpx.AsyncClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self._external_client = client or http_client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request(self, method: str, path: str, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._external_client is not None:
            resp = await self._external_client.request(
                method, f"{self.base_url}{path}", json=json_body, headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json()

        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            resp = await client.request(
                method, f"{self.base_url}{path}", json=json_body, headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json()

    async def register_manifest(self, manifest: AppMemoryManifest) -> dict[str, Any]:
        """Registers an ADK app's ontology manifest with the remote Memory Platform Service."""
        return await self._request("POST", "/v1/apps/register", manifest.model_dump(mode="json"))

    async def add_session_to_memory(self, session: Session) -> None:
        await self.add_events_to_memory(
            app_name=session.app_name,
            user_id=session.user_id,
            events=session.events,
            session_id=session.id,
        )

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> None:
        serialized_events = []
        for ev in events:
            if not getattr(ev, "content", None) or not getattr(ev.content, "parts", None):
                continue
            text_parts = [p.text for p in ev.content.parts if getattr(p, "text", None)]
            if text_parts:
                serialized_events.append(
                    {
                        "author": getattr(ev, "author", None) or "agent",
                        "text": "\n".join(text_parts),
                    }
                )
        if not serialized_events:
            return

        await self._request(
            "POST",
            "/v1/memory/events",
            {
                "app_name": app_name,
                "user_id": user_id,
                "session_id": session_id or "default_session",
                "events": serialized_events,
                "custom_metadata": custom_metadata or {},
            },
        )

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: dict[str, Any] | None = None,
    ) -> None:
        synthetic_events = [
            Event(author=m.author or "user", content=m.content)
            for m in memories
            if getattr(m, "content", None)
        ]
        await self.add_events_to_memory(
            app_name=app_name,
            user_id=user_id,
            events=synthetic_events,
            custom_metadata=custom_metadata,
        )

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
        as_of_iso: str | None = None,
        max_hops: int = 2,
    ) -> SearchMemoryResponse:
        data = await self._request(
            "POST",
            "/v1/memory/search",
            {
                "app_name": app_name,
                "user_id": user_id,
                "query": query,
                "as_of_iso": as_of_iso,
                "max_hops": max_hops,
            },
        )
        entries: list[MemoryEntry] = []
        for item in data.get("memories", []):
            entries.append(
                MemoryEntry(
                    author=item.get("author", "memory"),
                    timestamp=item.get("timestamp"),
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=item.get("text", ""))],
                    ),
                )
            )
        return SearchMemoryResponse(memories=entries)

    async def record_decision_trace(
        self,
        *,
        app_name: str,
        user_id: str,
        run_id: str,
        session_id: str,
        steps: Sequence[ReasoningTraceStep],
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/traces",
            {
                "app_name": app_name,
                "user_id": user_id,
                "run_id": run_id,
                "session_id": session_id,
                "steps": [s.model_dump(mode="json") for s in steps],
            },
        )

    async def register_playbook(
        self,
        *,
        app_name: str,
        user_id: str,
        trigger_intent: str,
        workflow_steps: list[str],
        resolution_summary: str,
        share_in_domain: bool = True,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/playbooks/register",
            {
                "app_name": app_name,
                "user_id": user_id,
                "trigger_intent": trigger_intent,
                "workflow_steps": workflow_steps,
                "resolution_summary": resolution_summary,
                "share_in_domain": share_in_domain,
            },
        )


# ============================================================================
# DROP-IN GOOGLE ADK CALLBACKS & EXPLICIT AGENT TOOLS
# ============================================================================
async def non_blocking_memory_commit_callback(callback_context: Any) -> None:
    """
    Google ADK `after_agent_callback`.
    Commits the latest user + agent turn delta asynchronously off the critical user response path.
    """
    session = getattr(callback_context, "session", None)
    if not session or not getattr(session, "events", None):
        return None
    recent_events = session.events[-2:]
    if recent_events and hasattr(callback_context, "add_events_to_memory"):
        asyncio.create_task(callback_context.add_events_to_memory(events=recent_events))
    return None


def get_adk_memory_tools(
    memory_service: BaseMemoryService,
    default_app_name: str = "adk_agent",
    default_user_id: str = "default_user",
) -> list[Any]:
    """
    Returns ready-to-attach Google ADK tool functions (`recall_memory_context`,
    `commit_memory_episode`, `record_agent_decision_trace`) backed by either
    `RemoteCompositeMemoryService` (no DB credentials needed) or in-process `CompositeMemoryService`.
    Can be added alongside `PreloadMemoryTool()` in `LlmAgent(tools=[...])`.
    """

    async def recall_memory_context(
        query: str,
        as_of_timestamp: str | None = None,
        app_name: str = default_app_name,
        user_id: str = default_user_id,
    ) -> dict[str, Any]:
        """Search composite memory (Bi-Temporal Knowledge Graph + Episodic Memory), optionally at a historical point in time (`as_of_timestamp`)."""
        resp = await memory_service.search_memory(  # type: ignore[attr-defined]
            app_name=app_name,
            user_id=user_id,
            query=query,
            as_of_iso=as_of_timestamp,
        )
        blocks = []
        for m in resp.memories:
            if getattr(m, "content", None) and getattr(m.content, "parts", None):
                text = "\n".join(
                    p.text for p in m.content.parts if getattr(p, "text", None)
                )
                blocks.append({"author": m.author, "timestamp": m.timestamp, "text": text})
        return {"query": query, "as_of": as_of_timestamp, "memories": blocks}

    async def commit_memory_episode(
        fact_or_observation: str,
        app_name: str = default_app_name,
        user_id: str = default_user_id,
    ) -> dict[str, Any]:
        """Explicitly commit an important observation or domain fact into Episodic Memory and the Bi-Temporal Knowledge Graph."""
        ev = Event(
            author="agent",
            content=types.Content(
                role="user", parts=[types.Part.from_text(text=fact_or_observation)]
            ),
        )
        await memory_service.add_events_to_memory(  # type: ignore[attr-defined]
            app_name=app_name,
            user_id=user_id,
            events=[ev],
        )
        return {"status": "committed", "app_name": app_name}

    return [recall_memory_context, commit_memory_episode]
