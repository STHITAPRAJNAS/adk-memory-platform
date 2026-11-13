"""
SRE Specialist Agent (A2A Server on :8091).
Subscribes to the `sre_infra@v1.2` Domain Pack (`GovernanceTier.TIER2_DOMAIN_SHARED`)
and talks to the centralized Memory Platform Service over HTTP via `RemoteCompositeMemoryService`.
"""

from __future__ import annotations

import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.genai import types

from agentic_memory_sdk.models import ReasoningTraceStep
from agentic_memory_sdk.service import Event

from a2a_sample.config import get_a2a_settings, get_remote_memory_client
from a2a_sample.scripted_llm import build_sre_model

APP_KEY = "t1:sre_specialist"


async def remember_sre_event(raw_observation: str) -> dict[str, Any]:
    """
    Ingests an SRE infrastructure event or failover observation into the governed
    `domain:t1:sre_infra` bi-temporal knowledge graph and records a procedural trace.
    """
    memory_client = get_remote_memory_client()
    ev = Event(
        author="sre_specialist",
        content=types.Content(
            role="user", parts=[types.Part.from_text(text=raw_observation)]
        ),
    )
    await memory_client.add_events_to_memory(
        app_name=APP_KEY,
        user_id="sre_operator",
        events=[ev],
        session_id="sre_a2a_session",
    )
    await memory_client.record_decision_trace(
        app_name=APP_KEY,
        user_id="sre_operator",
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        session_id="sre_a2a_session",
        steps=[
            ReasoningTraceStep(
                step_index=1,
                tool_name="remember_sre_event",
                tool_input_summary=raw_observation[:120],
                tool_output_summary="Committed to domain:t1:sre_infra",
                status="SUCCESS",
            )
        ],
    )
    return {"status": "committed", "summary": raw_observation}


async def recall_sre_context(query: str, as_of_iso: str = "") -> dict[str, Any]:
    """
    Queries the governed `domain:t1:sre_infra` bi-temporal knowledge graph and
    private `app:t1:sre_specialist` overflow/episodic memory.
    Pass `as_of_iso` (e.g. '2026-06-01T00:00:00Z') for point-in-time historical state.
    """
    memory_client = get_remote_memory_client()
    res = await memory_client.search_memory(
        app_name=APP_KEY,
        user_id="sre_operator",
        query=query,
        as_of_iso=as_of_iso.strip() or None,
    )
    blocks = []
    for m in res.memories:
        if m.content and m.content.parts:
            blocks.append("\n".join(p.text for p in m.content.parts if p.text))
    return {
        "query": query,
        "as_of_iso": as_of_iso or "ACTIVE",
        "context": "\n---\n".join(blocks) if blocks else "NO_MATCHING_MEMORIES",
    }


def build_agent() -> LlmAgent:
    settings = get_a2a_settings()
    model = build_sre_model() if settings.use_fake_llm else settings.model_name
    return LlmAgent(
        name="sre_specialist",
        model=model,
        description=(
            "Remote SRE & Cloud Infrastructure specialist reached over A2A. "
            "Records and recalls bi-temporal microservice deployments, database dependencies, "
            "and regional failovers in the governed `domain:t1:sre_infra` ontology graph."
        ),
        instruction=(
            "You are the SRE Specialist agent. Use `remember_sre_event` to store infrastructure "
            "changes and `recall_sre_context` to answer questions about active or historical service state."
        ),
        tools=[remember_sre_event, recall_sre_context],
    )


root_agent = build_agent()
app = App(
    name="sre_specialist",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
