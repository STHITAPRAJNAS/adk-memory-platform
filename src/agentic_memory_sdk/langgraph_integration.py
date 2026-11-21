from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Callable, Iterable, Sequence
import uuid

from .adk_integration import RemoteCompositeMemoryService
from .models import ExtractionMode, ReasoningTraceStep
from .service import CompositeMemoryService, Event, types

MemoryServiceLike = CompositeMemoryService | RemoteCompositeMemoryService

# Optional LangGraph BaseStore & Item imports (graceful fallback when langgraph is not installed)
try:
    from langgraph.store.base import BaseStore as _LGStoreBase  # type: ignore
    from langgraph.store.base import GetOp, Item, ListNamespacesOp, PutOp, SearchItem, SearchOp  # type: ignore

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

    class _LGStoreBase:  # type: ignore
        pass

    @dataclass
    class Item:  # type: ignore
        value: dict[str, Any]
        key: str
        namespace: tuple[str, ...]
        created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
        updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @dataclass
    class SearchItem(Item):  # type: ignore
        score: float | None = None

    GetOp = PutOp = SearchOp = ListNamespacesOp = Any  # type: ignore

# Optional LangChain Core imports (graceful fallback when langchain-core is not installed)
try:
    from langchain_core.chat_history import BaseChatMessageHistory as _LCChatHistoryBase  # type: ignore
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage  # type: ignore
    from langchain_core.tools import StructuredTool  # type: ignore

    HAS_LANGCHAIN_CORE = True
except ImportError:
    HAS_LANGCHAIN_CORE = False

    class _LCChatHistoryBase:  # type: ignore
        pass

    @dataclass
    class BaseMessage:  # type: ignore
        content: str
        type: str = "human"
        additional_kwargs: dict[str, Any] = field(default_factory=dict)

    class HumanMessage(BaseMessage):  # type: ignore
        def __init__(self, content: str, **kwargs: Any):
            super().__init__(content=content, type="human", additional_kwargs=kwargs)

    class AIMessage(BaseMessage):  # type: ignore
        def __init__(self, content: str, **kwargs: Any):
            super().__init__(content=content, type="ai", additional_kwargs=kwargs)

    class SystemMessage(BaseMessage):  # type: ignore
        def __init__(self, content: str, **kwargs: Any):
            super().__init__(content=content, type="system", additional_kwargs=kwargs)

    StructuredTool = None  # type: ignore


def _extract_message_role_and_text(msg: Any) -> tuple[str, str]:
    """Extracts `(role, text)` from a LangChain `BaseMessage`, OpenAI-style dict, or `(role, text)` tuple."""
    if isinstance(msg, tuple) and len(msg) == 2:
        return str(msg[0]), str(msg[1])
    if isinstance(msg, dict):
        role = str(msg.get("role") or msg.get("type") or msg.get("author") or "user")
        content = msg.get("content") or msg.get("text") or ""
        if isinstance(content, list):
            content = "\n".join(
                str(b.get("text", "")) if isinstance(b, dict) else str(b)
                for b in content
            )
        return role, str(content)
    role = str(getattr(msg, "type", None) or getattr(msg, "role", None) or "user")
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b)
            for b in content
        )
    return role, str(content)


# =============================================================================
# 1. LangGraph `BaseStore` Implementation (`LangGraphCompositeMemoryStore`)
# =============================================================================


class LangGraphCompositeMemoryStore(_LGStoreBase):
    """
    Drop-in LangGraph `BaseStore` adapter (`builder.compile(store=LangGraphCompositeMemoryStore(...))`)
    backed by either in-process `CompositeMemoryService` or stateless `RemoteCompositeMemoryService`.

    Namespace Mapping Convention:
      - `namespace = ("acme:sre_agent", "alice_vp", ...)` -> `app_name="acme:sre_agent"`, `user_id="alice_vp"`
      - `namespace = ("users", "alice_vp")` or `("alice_vp",)` -> `app_name=default_app_name`, `user_id="alice_vp"`
      - `aput` / `put`: commits structured facts or conversation turns into Tier-1 Episodic Memory +
        Tier-2 Bi-Temporal Knowledge Graph (with PII scrubbing & Two-Bucket TBox validation).
      - `asearch` / `search`: executes 3-Tier Hybrid Recall (Playbooks + 2-Hop Bi-Temporal Graph + Episodes)
        and returns LangGraph `SearchItem` records.
    """

    def __init__(
        self,
        memory_service: MemoryServiceLike,
        *,
        default_app_name: str = "default:langgraph_app",
        default_user_id: str = "default_user",
        extraction_mode: ExtractionMode | str | None = None,
    ):
        self.memory_service = memory_service
        self.default_app_name = default_app_name
        self.default_user_id = default_user_id
        self.extraction_mode = extraction_mode
        self._items: dict[tuple[tuple[str, ...], str], Item] = {}

    def _resolve_app_and_user(
        self, namespace: tuple[str, ...] | Sequence[str]
    ) -> tuple[str, str]:
        ns = tuple(namespace)
        if not ns:
            return self.default_app_name, self.default_user_id
        if len(ns) >= 2 and ":" in ns[0]:
            return ns[0], ns[1]
        if len(ns) >= 2 and ns[0] in ("users", "user", "memories", "threads"):
            return self.default_app_name, ns[1]
        if len(ns) >= 2:
            return ns[0], ns[1]
        if ":" in ns[0]:
            return ns[0], self.default_user_id
        return self.default_app_name, ns[0]

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: float | None = None,
    ) -> None:
        ns = tuple(namespace)
        now = datetime.now(timezone.utc)
        existing = self._items.get((ns, key))
        created = existing.created_at if existing else now
        self._items[(ns, key)] = Item(
            value=value,
            key=key,
            namespace=ns,
            created_at=created,
            updated_at=now,
        )

        if index is False:
            return

        app_name, user_id = self._resolve_app_and_user(ns)
        text = (
            value.get("text")
            or value.get("content")
            or value.get("memory")
            or value.get("fact")
            or json.dumps(value)
        )
        author = str(value.get("author") or value.get("role") or "user")
        session_id = str(value.get("session_id") or value.get("thread_id") or key)
        mode = value.get("extraction_mode") or self.extraction_mode

        ev = Event(
            author=author,
            content=types.Content(
                role="user", parts=[types.Part.from_text(text=str(text))]
            ),
        )
        await self.memory_service.add_events_to_memory(
            app_name=app_name,
            user_id=user_id,
            events=[ev],
            session_id=session_id,
            extraction_mode=mode,
        )

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return self._items.get((tuple(namespace), key))

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        self._items.pop((tuple(namespace), key), None)

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        ns_prefix = tuple(namespace_prefix)
        app_name, user_id = self._resolve_app_and_user(ns_prefix)
        if filter:
            app_name = str(filter.get("app_name", app_name))
            user_id = str(filter.get("user_id", user_id))
        as_of_iso = filter.get("as_of_iso") if filter else None
        max_hops = int(filter.get("max_hops", 2)) if filter else 2

        if query:
            resp = await self.memory_service.search_memory(
                app_name=app_name,
                user_id=user_id,
                query=query,
                as_of_iso=as_of_iso,
                max_hops=max_hops,
            )
            results: list[SearchItem] = []
            for idx, m in enumerate(resp.memories):
                text_val = ""
                if getattr(m, "content", None) and getattr(m.content, "parts", None):
                    text_val = "\n".join(
                        p.text for p in m.content.parts if getattr(p, "text", None)
                    )
                author = getattr(m, "author", None) or "memory"
                score = round(max(0.1, 1.0 - (idx * 0.08)), 3)
                results.append(
                    SearchItem(
                        namespace=ns_prefix or (app_name, user_id),
                        key=f"recalled_{idx}_{author}",
                        value={
                            "author": author,
                            "text": text_val,
                            "timestamp": getattr(m, "timestamp", None),
                            "as_of_iso": as_of_iso,
                        },
                        score=score,
                    )
                )
            return results[offset : offset + limit]

        # When query is None, return matching in-memory items under namespace_prefix
        matched: list[SearchItem] = []
        for (item_ns, key), item in self._items.items():
            if item_ns[: len(ns_prefix)] == ns_prefix:
                matched.append(
                    SearchItem(
                        namespace=item.namespace,
                        key=key,
                        value=item.value,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                        score=1.0,
                    )
                )
        return matched[offset : offset + limit]

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        seen: list[tuple[str, ...]] = []
        for ns, _ in self._items:
            if prefix and ns[: len(prefix)] != tuple(prefix):
                continue
            if suffix and ns[-len(suffix) :] != tuple(suffix):
                continue
            cand = ns[:max_depth] if max_depth is not None else ns
            if cand not in seen:
                seen.append(cand)
        return seen[offset : offset + limit]

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """Executes a batch of LangGraph `BaseStore` operations (`GetOp`, `PutOp`, `SearchOp`, `ListNamespacesOp`)."""
        results: list[Any] = []
        for op in ops:
            op_cls = type(op).__name__
            if op_cls == "GetOp":
                results.append(await self.aget(op.namespace, op.key))
            elif op_cls == "PutOp":
                if getattr(op, "value", None) is None:
                    await self.adelete(op.namespace, op.key)
                else:
                    await self.aput(
                        op.namespace,
                        op.key,
                        op.value,
                        index=getattr(op, "index", None),
                    )
                results.append(None)
            elif op_cls == "SearchOp":
                results.append(
                    await self.asearch(
                        op.namespace_prefix,
                        query=getattr(op, "query", None),
                        filter=getattr(op, "filter", None),
                        limit=getattr(op, "limit", 10),
                        offset=getattr(op, "offset", 0),
                    )
                )
            elif op_cls == "ListNamespacesOp":
                results.append(
                    await self.alist_namespaces(
                        max_depth=getattr(op, "max_depth", None),
                        limit=getattr(op, "limit", 100),
                        offset=getattr(op, "offset", 0),
                    )
                )
            else:
                results.append(None)
        return results

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError(
                "Use async methods (`aget`, `aput`, `asearch`, `abatch`) inside an active event loop."
            )
        return asyncio.run(self.abatch(list(ops)))


# =============================================================================
# 2. LangGraph StateGraph Nodes (`create_langgraph_memory_nodes`)
# =============================================================================


@dataclass
class LangGraphMemoryNodes:
    """
    Pre-built async LangGraph `StateGraph` nodes for automatic 3-Tier Memory Recall,
    Pre-Flight Answerability Checking, and Turn/Trace Commit:
      - `recall_node(state, config=None)`: Enriches state with `memory_context` (and optional
        `answerability` or prepended `SystemMessage`).
      - `commit_node(state, config=None)`: Commits the latest user + assistant turn to
        Episodic + Bi-Temporal Graph memory and records any tool calls as Tier-3 Reasoning Traces.
    """

    recall_node: Callable[..., Any]
    commit_node: Callable[..., Any]
    check_answerable_node: Callable[..., Any]


def create_langgraph_memory_nodes(
    memory_service: MemoryServiceLike,
    *,
    default_app_name: str = "default:langgraph_app",
    default_user_id: str = "default_user",
    inject_system_message: bool = False,
    extraction_mode: ExtractionMode | str | None = None,
) -> LangGraphMemoryNodes:
    """
    Builds ready-to-wire LangGraph `StateGraph` nodes (`recall_node`, `commit_node`,
    `check_answerable_node`) compatible with `MessagesState` or custom state dicts.
    """

    def _extract_cfg(state: dict[str, Any], config: dict[str, Any] | None) -> tuple[str, str, str, str | None]:
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
        app_name = str(
            state.get("app_name")
            or cfg.get("app_name")
            or default_app_name
        )
        user_id = str(
            state.get("user_id")
            or cfg.get("user_id")
            or default_user_id
        )
        session_id = str(
            state.get("session_id")
            or cfg.get("thread_id")
            or cfg.get("session_id")
            or "langgraph_thread"
        )
        as_of_iso = state.get("as_of_iso") or cfg.get("as_of_iso")
        return app_name, user_id, session_id, as_of_iso

    def _latest_user_query(state: dict[str, Any]) -> str:
        if state.get("query"):
            return str(state["query"])
        messages = state.get("messages") or []
        for msg in reversed(messages):
            role, text = _extract_message_role_and_text(msg)
            if role in ("user", "human") and text.strip():
                return text.strip()
        if messages:
            _, text = _extract_message_role_and_text(messages[-1])
            return text.strip()
        return ""

    async def recall_node(
        state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        app_name, user_id, _session_id, as_of_iso = _extract_cfg(state, config)
        query = _latest_user_query(state)
        if not query:
            return {"memory_context": "", "memory_blocks": []}

        resp = await memory_service.search_memory(
            app_name=app_name,
            user_id=user_id,
            query=query,
            as_of_iso=as_of_iso,
        )
        blocks: list[str] = []
        for m in resp.memories:
            if getattr(m, "content", None) and getattr(m.content, "parts", None):
                text = "\n".join(
                    p.text for p in m.content.parts if getattr(p, "text", None)
                )
                if text:
                    blocks.append(text)
        combined = "\n\n".join(blocks)
        updates: dict[str, Any] = {
            "memory_context": combined,
            "memory_blocks": blocks,
        }
        if inject_system_message and combined:
            sys_msg = SystemMessage(
                content=(
                    "Ground your response in the following 3-Tier Composite Memory context "
                    "(Procedural Playbooks + Bi-Temporal Knowledge Graph + Episodic Memory):\n\n"
                    f"{combined}"
                )
            )
            updates["messages"] = [sys_msg, *(state.get("messages") or [])]
        return updates

    async def check_answerable_node(
        state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        app_name, user_id, _session_id, as_of_iso = _extract_cfg(state, config)
        query = _latest_user_query(state)
        if not query or not hasattr(memory_service, "check_answerable"):
            return {"answerability": {"answerable": False, "status": "UNANSWERABLE", "coverage_score": 0.0}}
        res = await getattr(memory_service, "check_answerable")(
            app_name=app_name,
            user_id=user_id,
            query=query,
            as_of_iso=as_of_iso,
        )
        return {"answerability": res}

    async def commit_node(
        state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        app_name, user_id, session_id, _ = _extract_cfg(state, config)
        messages = state.get("messages") or []
        recent = messages[-2:] if len(messages) >= 2 else messages
        adk_events: list[Event] = []
        trace_steps: list[ReasoningTraceStep] = []

        for idx, msg in enumerate(recent, start=1):
            role, text = _extract_message_role_and_text(msg)
            if role == "system":
                continue
            author = "user" if role in ("user", "human") else "agent"
            if text.strip():
                adk_events.append(
                    Event(
                        author=author,
                        content=types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=text)],
                        ),
                    )
                )
            # Extract LangChain / LangGraph tool_calls if present on AIMessage
            tool_calls = getattr(msg, "tool_calls", None) or (
                msg.get("tool_calls") if isinstance(msg, dict) else None
            )
            for t_idx, tc in enumerate(tool_calls or [], start=1):
                t_name = tc.get("name", "tool") if isinstance(tc, dict) else getattr(tc, "name", "tool")
                t_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                trace_steps.append(
                    ReasoningTraceStep(
                        step_index=idx * 10 + t_idx,
                        tool_name=str(t_name),
                        tool_input_summary=json.dumps(t_args)[:300],
                        tool_output_summary="Executed via LangGraph node",
                        touched_vertex_ids=[],
                    )
                )

        if adk_events:
            await memory_service.add_events_to_memory(
                app_name=app_name,
                user_id=user_id,
                events=adk_events,
                session_id=session_id,
                extraction_mode=state.get("extraction_mode") or extraction_mode,
            )
        if trace_steps:
            await memory_service.record_decision_trace(
                app_name=app_name,
                user_id=user_id,
                run_id=f"lg_run_{uuid.uuid4().hex[:10]}",
                session_id=session_id,
                steps=trace_steps,
            )
        return {
            "memory_committed_events": len(adk_events),
            "memory_recorded_traces": len(trace_steps),
        }

    return LangGraphMemoryNodes(
        recall_node=recall_node,
        commit_node=commit_node,
        check_answerable_node=check_answerable_node,
    )


# =============================================================================
# 3. LangGraph / LangChain Toolset (`get_langgraph_memory_tools`)
# =============================================================================


class _CompatibleAsyncTool:
    """Lightweight tool wrapper matching LangChain's `BaseTool` / `StructuredTool` interface when `langchain-core` is absent."""

    def __init__(self, fn: Callable[..., Any], name: str, description: str):
        self.coroutine = fn
        self.func = fn
        self.name = name
        self.description = description
        self.__name__ = name
        self.__doc__ = description

    async def ainvoke(self, input_data: dict[str, Any] | str, **kwargs: Any) -> Any:
        if isinstance(input_data, dict):
            return await self.coroutine(**input_data)
        return await self.coroutine(input_data)

    def invoke(self, input_data: dict[str, Any] | str, **kwargs: Any) -> Any:
        return asyncio.run(self.ainvoke(input_data, **kwargs))

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.coroutine(*args, **kwargs)


def _wrap_langchain_tool(fn: Callable[..., Any], name: str, description: str) -> Any:
    if HAS_LANGCHAIN_CORE and StructuredTool is not None:
        return StructuredTool.from_function(
            coroutine=fn,
            name=name,
            description=description,
        )
    return _CompatibleAsyncTool(fn=fn, name=name, description=description)


def get_langgraph_memory_tools(
    memory_service: MemoryServiceLike,
    default_app_name: str = "default:langgraph_app",
    default_user_id: str = "default_user",
) -> list[Any]:
    """
    Returns ready-to-bind LangGraph / LangChain tools (`StructuredTool` when `langchain-core`
    is installed, or compatible callable tool objects with `.ainvoke()` / `.name` / `.description`):
      1. `recall_memory_context`
      2. `check_memory_answerable`
      3. `commit_memory_episode`
      4. `register_memory_playbook`
      5. `record_memory_decision_trace`
    """

    async def recall_memory_context(
        query: str,
        as_of_timestamp: str | None = None,
        app_name: str = default_app_name,
        user_id: str = default_user_id,
    ) -> dict[str, Any]:
        """Recall 3-Tier Composite Memory (Procedural Playbooks + Bi-Temporal Knowledge Graph + Episodic Memory)."""
        resp = await memory_service.search_memory(
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

    async def check_memory_answerable(
        query: str,
        as_of_timestamp: str | None = None,
        app_name: str = default_app_name,
        user_id: str = default_user_id,
    ) -> dict[str, Any]:
        """Pre-flight coverage check: verify whether the governed ontology and 3-Tier Memory cover the concepts in `query`."""
        return await memory_service.check_answerable(
            app_name=app_name,
            user_id=user_id,
            query=query,
            as_of_iso=as_of_timestamp,
        )

    async def commit_memory_episode(
        fact_or_observation: str,
        extraction_mode: str | None = None,
        app_name: str = default_app_name,
        user_id: str = default_user_id,
    ) -> dict[str, Any]:
        """Commit an interaction turn or domain observation into Episodic Memory and the Bi-Temporal Knowledge Graph."""
        ev = Event(
            author="agent",
            content=types.Content(
                role="user", parts=[types.Part.from_text(text=fact_or_observation)]
            ),
        )
        await memory_service.add_events_to_memory(
            app_name=app_name,
            user_id=user_id,
            events=[ev],
            extraction_mode=extraction_mode,
        )
        return {"status": "committed", "app_name": app_name}

    async def register_memory_playbook(
        trigger_intent: str,
        workflow_steps: list[str],
        resolution_summary: str,
        app_name: str = default_app_name,
        user_id: str = default_user_id,
    ) -> dict[str, Any]:
        """Register or reinforce a reusable Tier-3 Procedural Playbook (`trigger_intent -> workflow_steps`)."""
        res = await memory_service.register_playbook(
            app_name=app_name,
            user_id=user_id,
            trigger_intent=trigger_intent,
            workflow_steps=workflow_steps,
            resolution_summary=resolution_summary,
        )
        if isinstance(res, dict):
            return res
        return {
            "status": "registered",
            "playbook_id": res.playbook_id,
            "success_count": res.success_count,
            "namespace": res.namespace,
        }

    async def record_memory_decision_trace(
        tool_name: str,
        input_summary: str,
        output_summary: str,
        referenced_entities: list[str] | None = None,
        app_name: str = default_app_name,
        user_id: str = default_user_id,
    ) -> dict[str, Any]:
        """Record a Tier-3 Procedural Reasoning Trace step (`AgentRun -> ToolCall -> Entity`)."""
        run_id = f"lg_trace_{uuid.uuid4().hex[:10]}"
        step = ReasoningTraceStep(
            step_index=1,
            tool_name=tool_name,
            tool_input_summary=input_summary,
            tool_output_summary=output_summary,
            touched_vertex_ids=referenced_entities or [],
        )
        await memory_service.record_decision_trace(
            app_name=app_name,
            user_id=user_id,
            run_id=run_id,
            session_id="langgraph_tool_trace",
            steps=[step],
        )
        return {"status": "recorded", "run_id": run_id}

    return [
        _wrap_langchain_tool(
            recall_memory_context,
            "recall_memory_context",
            recall_memory_context.__doc__ or "",
        ),
        _wrap_langchain_tool(
            check_memory_answerable,
            "check_memory_answerable",
            check_memory_answerable.__doc__ or "",
        ),
        _wrap_langchain_tool(
            commit_memory_episode,
            "commit_memory_episode",
            commit_memory_episode.__doc__ or "",
        ),
        _wrap_langchain_tool(
            register_memory_playbook,
            "register_memory_playbook",
            register_memory_playbook.__doc__ or "",
        ),
        _wrap_langchain_tool(
            record_memory_decision_trace,
            "record_memory_decision_trace",
            record_memory_decision_trace.__doc__ or "",
        ),
    ]


# =============================================================================
# 4. LangChain `BaseChatMessageHistory` Adapter (`CompositeMemoryChatMessageHistory`)
# =============================================================================


class CompositeMemoryChatMessageHistory(_LCChatHistoryBase):
    """
    LangChain `BaseChatMessageHistory` implementation backed by `CompositeMemoryService`
    or `RemoteCompositeMemoryService` (usable with `RunnableWithMessageHistory`).
    """

    def __init__(
        self,
        memory_service: MemoryServiceLike,
        *,
        app_name: str,
        user_id: str,
        session_id: str = "default_session",
        extraction_mode: ExtractionMode | str | None = None,
    ):
        self.memory_service = memory_service
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id
        self.extraction_mode = extraction_mode
        self.messages: list[BaseMessage] = []

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        adk_events: list[Event] = []
        for m in messages:
            self.messages.append(m)
            role, text = _extract_message_role_and_text(m)
            if not text.strip():
                continue
            author = "user" if role in ("user", "human") else "agent"
            adk_events.append(
                Event(
                    author=author,
                    content=types.Content(
                        role="user", parts=[types.Part.from_text(text=text)]
                    ),
                )
            )
        if adk_events:
            await self.memory_service.add_events_to_memory(
                app_name=self.app_name,
                user_id=self.user_id,
                events=adk_events,
                session_id=self.session_id,
                extraction_mode=self.extraction_mode,
            )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        asyncio.run(self.aadd_messages(messages))

    async def aget_messages(self, query: str = "recent conversation history") -> list[BaseMessage]:
        if self.messages:
            return list(self.messages)
        resp = await self.memory_service.search_memory(
            app_name=self.app_name,
            user_id=self.user_id,
            query=query,
        )
        hydrated: list[BaseMessage] = []
        for m in resp.memories:
            text_val = ""
            if getattr(m, "content", None) and getattr(m.content, "parts", None):
                text_val = "\n".join(
                    p.text for p in m.content.parts if getattr(p, "text", None)
                )
            author = getattr(m, "author", None) or "user"
            if author in ("user", "human"):
                hydrated.append(HumanMessage(content=text_val))
            else:
                hydrated.append(AIMessage(content=text_val))
        return hydrated

    async def aclear(self) -> None:
        self.messages.clear()
        await self.memory_service.purge_user_lineage(
            app_name=self.app_name, user_id=self.user_id
        )

    def clear(self) -> None:
        asyncio.run(self.aclear())
