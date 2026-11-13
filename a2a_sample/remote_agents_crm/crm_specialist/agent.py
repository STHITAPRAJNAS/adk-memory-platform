"""
CRM Specialist Agent (A2A Server on :8092).
Subscribes to the `customer_crm@v2.0` Domain Pack (`GovernanceTier.TIER2_DOMAIN_SHARED`).
Strictly isolated at the PostgreSQL RLS and Neo4j/Neptune openCypher level from `domain:t1:sre_infra`.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.genai import types

from agentic_memory_sdk.service import Event

from a2a_sample.config import get_a2a_settings, get_remote_memory_client
from a2a_sample.scripted_llm import build_crm_model

APP_KEY = "t1:crm_specialist"


async def remember_crm_event(raw_observation: str) -> dict[str, Any]:
    """Ingests a Customer Account or Subscription tier change into `domain:t1:customer_crm`."""
    memory_client = get_remote_memory_client()
    ev = Event(
        author="crm_specialist",
        content=types.Content(
            role="user", parts=[types.Part.from_text(text=raw_observation)]
        ),
    )
    await memory_client.add_events_to_memory(
        app_name=APP_KEY,
        user_id="crm_account_exec",
        events=[ev],
        session_id="crm_a2a_session",
    )
    return {"status": "committed", "summary": raw_observation}


async def recall_crm_context(query: str) -> dict[str, Any]:
    """Queries `domain:t1:customer_crm` and `app:t1:crm_specialist` memory."""
    memory_client = get_remote_memory_client()
    res = await memory_client.search_memory(
        app_name=APP_KEY,
        user_id="crm_account_exec",
        query=query,
    )
    blocks = []
    for m in res.memories:
        if m.content and m.content.parts:
            blocks.append("\n".join(p.text for p in m.content.parts if p.text))
    return {
        "query": query,
        "context": "\n---\n".join(blocks) if blocks else "NO_MATCHING_MEMORIES",
    }


def build_agent() -> LlmAgent:
    settings = get_a2a_settings()
    model = build_crm_model() if settings.use_fake_llm else settings.model_name
    return LlmAgent(
        name="crm_specialist",
        model=model,
        description=(
            "Remote Customer CRM & Subscription specialist reached over A2A. "
            "Manages customer account tiers and tickets in the governed `domain:t1:customer_crm` graph."
        ),
        instruction=(
            "You are the CRM Specialist agent. Use `remember_crm_event` to record customer subscription "
            "updates and `recall_crm_context` to answer customer account queries."
        ),
        tools=[remember_crm_event, recall_crm_context],
    )


root_agent = build_agent()
app = App(
    name="crm_specialist",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
