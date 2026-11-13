"""
Deterministic ScriptedLlm for Google ADK A2A agents (`platform_concierge`,
`sre_specialist`, and `crm_specialist`).
Routes A2A delegations (`transfer_to_agent`) and invokes `RemoteCompositeMemoryService`
tools (`remember_sre_event`, `recall_sre_context`, `remember_crm_event`, `recall_crm_context`)
deterministically so the full A2A + Docker Postgres/Neo4j stack can be tested end-to-end.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Callable

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class ConversationView:
    def __init__(self, contents: list[types.Content]) -> None:
        self.contents = contents

    def last_user_text(self) -> str:
        for content in reversed(self.contents):
            if content.role == "user" and content.parts:
                clean_parts = [
                    p.text
                    for p in content.parts
                    if p.text and "<<<BEGIN_QUOTED_AGENT_CONTENT>>>" not in p.text
                ]
                text = " ".join(clean_parts)
                if text.strip():
                    return text.strip()
        return ""

    def function_responses(self) -> list[types.FunctionResponse]:
        out: list[types.FunctionResponse] = []
        for content in self.contents:
            for part in content.parts or []:
                if part.function_response is not None:
                    out.append(part.function_response)
        return out

    def latest_response_for(self, tool_name: str) -> dict | None:
        for fr in reversed(self.function_responses()):
            if fr.name == tool_name:
                resp = fr.response
                return resp if isinstance(resp, dict) else {"result": resp}
        return None


def _text_reply(message: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=message)])
    )


def _tool_call(name: str, **args: object) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=f"fc-{uuid.uuid4().hex[:12]}",
                        name=name,
                        args=dict(args),
                    )
                )
            ],
        )
    )


class ScriptedAdkLlm(BaseLlm):
    decider: Callable[[ConversationView], LlmResponse]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        conv = ConversationView(list(llm_request.contents or []))
        yield self.decider(conv)


def _concierge_decider(conv: ConversationView) -> LlmResponse:
    text = conv.last_user_text().lower()
    if any(
        w in text
        for w in (
            "sre",
            "checkout-api",
            "microservice",
            "region",
            "failover",
            "failed",
            "deployed",
            "aurora",
            "incident",
        )
    ):
        return _tool_call("transfer_to_agent", agent_name="sre_specialist")
    if any(
        w in text
        for w in (
            "crm",
            "customer",
            "acme-corp",
            "subscription",
            "enterprise-platinum",
            "account",
            "billing",
        )
    ):
        return _tool_call("transfer_to_agent", agent_name="crm_specialist")
    return _text_reply(
        "I am the Platform Concierge. Ask me about SRE infrastructure (delegated over A2A to `sre_specialist`) "
        "or Customer CRM accounts (delegated over A2A to `crm_specialist`)."
    )


def _sre_specialist_decider(conv: ConversationView) -> LlmResponse:
    remember_res = conv.latest_response_for("remember_sre_event")
    if remember_res is not None:
        return _text_reply(
            f"[SRE Specialist] Recorded in shared SRE memory (`domain:t1:sre_infra`): {remember_res.get('summary', 'committed')}"
        )

    recall_res = conv.latest_response_for("recall_sre_context")
    if recall_res is not None:
        return _text_reply(
            f"[SRE Specialist] Recalled from `domain:t1:sre_infra`:\n{recall_res.get('context', 'NONE')}"
        )

    user_text = conv.last_user_text()
    lower = user_text.lower()
    if lower.startswith("remember:") or "deployed in" in lower or "failed" in lower:
        return _tool_call("remember_sre_event", raw_observation=user_text)

    as_of = None
    if "as_of=" in user_text:
        as_of = user_text.split("as_of=", 1)[1].split()[0].strip()
    return _tool_call(
        "recall_sre_context",
        query=user_text,
        as_of_iso=as_of or "",
    )


def _crm_specialist_decider(conv: ConversationView) -> LlmResponse:
    remember_res = conv.latest_response_for("remember_crm_event")
    if remember_res is not None:
        return _text_reply(
            f"[CRM Specialist] Recorded in shared CRM memory (`domain:t1:customer_crm`): {remember_res.get('summary', 'committed')}"
        )

    recall_res = conv.latest_response_for("recall_crm_context")
    if recall_res is not None:
        return _text_reply(
            f"[CRM Specialist] Recalled from `domain:t1:customer_crm`:\n{recall_res.get('context', 'NONE')}"
        )

    user_text = conv.last_user_text()
    lower = user_text.lower()
    if lower.startswith("remember:") or "upgraded to" in lower:
        return _tool_call("remember_crm_event", raw_observation=user_text)
    return _tool_call("recall_crm_context", query=user_text)


def build_concierge_model() -> ScriptedAdkLlm:
    return ScriptedAdkLlm(
        model="scripted/platform_concierge", decider=_concierge_decider
    )


def build_sre_model() -> ScriptedAdkLlm:
    return ScriptedAdkLlm(
        model="scripted/sre_specialist", decider=_sre_specialist_decider
    )


def build_crm_model() -> ScriptedAdkLlm:
    return ScriptedAdkLlm(
        model="scripted/crm_specialist", decider=_crm_specialist_decider
    )
