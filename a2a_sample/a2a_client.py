from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

ADK_CUSTOM_METADATA_KEY = "adk_custom_metadata"
A2A_TASK_ID_KEY = "a2a:task_id"


def text_message(
    text: str, *, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "messageId": uuid.uuid4().hex,
        "kind": "message",
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
    }
    if task_id:
        msg["taskId"] = task_id
    if context_id:
        msg["contextId"] = context_id
    return msg


@dataclass
class A2ATurnResult:
    frames: list[dict[str, Any]] = field(default_factory=list)
    task_id: str | None = None
    context_id: str | None = None
    downstream_task_ids: set[str] = field(default_factory=set)
    error: dict[str, Any] | None = None
    http_status: int | None = None

    @property
    def texts(self) -> list[str]:
        out: list[str] = []
        for frame in self.frames:
            res = frame.get("result") or {}
            candidates = []
            if "status" in res and isinstance(res["status"], dict):
                msg = res["status"].get("message")
                if isinstance(msg, dict):
                    candidates.extend(msg.get("parts") or [])
            if "artifact" in res and isinstance(res["artifact"], dict):
                candidates.extend(res["artifact"].get("parts") or [])
            for art in res.get("artifacts") or []:
                if isinstance(art, dict):
                    candidates.extend(art.get("parts") or [])
            if res.get("kind") == "message":
                candidates.extend(res.get("parts") or [])

            for part in candidates:
                if isinstance(part, dict) and part.get("text"):
                    t = part["text"].strip()
                    if t and (not out or out[-1] != t):
                        out.append(t)
        return out

    @property
    def combined_text(self) -> str:
        return "\n".join(self.texts)


class A2AWireClient:
    """Minimal JSON-RPC client speaking A2A `message/stream` and fetching Agent Cards."""

    def __init__(self, http: httpx.AsyncClient, rpc_url: str, card_url: str):
        self.http = http
        self.rpc_url = rpc_url
        self.card_url = card_url

    async def fetch_card(self) -> dict[str, Any]:
        resp = await self.http.get(self.card_url)
        resp.raise_for_status()
        return resp.json()

    async def stream(self, message: dict[str, Any]) -> A2ATurnResult:
        out = A2ATurnResult()
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/stream",
            "params": {"message": message},
        }
        async with self.http.stream(
            "POST",
            self.rpc_url,
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            out.http_status = resp.status_code
            if resp.status_code >= 400:
                out.error = {
                    "http_status": resp.status_code,
                    "body": (await resp.aread()).decode(errors="replace"),
                }
                return out

            if "text/event-stream" not in resp.headers.get("content-type", ""):
                body = json.loads((await resp.aread()).decode())
                out.error = body.get("error") or body
                return out

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                frame = json.loads(raw)
                if "error" in frame:
                    out.error = frame["error"]
                    break
                result = frame.get("result") or {}
                for key in ("task", "statusUpdate", "artifactUpdate", "message"):
                    if isinstance(result.get(key), dict):
                        result = {**result[key], "kind": key}
                        break
                out.frames.append({"result": result})
                out.task_id = out.task_id or result.get("id") or result.get("taskId")
                out.context_id = out.context_id or result.get("contextId")

                metadata = result.get("metadata") or {}
                raw_meta = metadata.get(ADK_CUSTOM_METADATA_KEY)
                if isinstance(raw_meta, str):
                    try:
                        raw_meta = json.loads(raw_meta)
                    except json.JSONDecodeError:
                        raw_meta = None
                if isinstance(raw_meta, dict) and isinstance(
                    raw_meta.get(A2A_TASK_ID_KEY), str
                ):
                    out.downstream_task_ids.add(raw_meta[A2A_TASK_ID_KEY])
        return out
