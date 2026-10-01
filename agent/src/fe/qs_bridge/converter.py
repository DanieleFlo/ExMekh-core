from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any
from xml.etree import ElementTree as ET

from qs.core.agents.node import Node
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import UserMessage
from qs.core.tool_calls.execution_history import ExecutionHistory
from qs.interfaces.message import Message

from fe.qs_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    SkillNode,
    ToolEvent,
)
from fe.qs_bridge.skill_history import inner_skill_history_messages, take_skill_for_call
from fe.qs_bridge.tool_calls import parse_tool_calls, tool_result_for_call

logger = logging.getLogger(__name__)


def raw_dict_to_qs_message(
    data: dict[str, Any],
    pending_tool_calls: list[ToolCall] | None = None,
) -> Message[Any]:
    """Converts a raw dictionary (from raw_history or dump_history) to a qs Message instance."""
    role = str(data.get("role", "user")).lower()
    content = data.get("content")
    qs_msg: Message[Any]

    if role == "system":
        qs_msg = SystemMessage(content=str(content or ""))

    elif role == "user":
        qs_msg = UserMessage(content=str(content or ""))

    elif role == "assistant":
        tool_calls = parse_tool_calls(data.get("tool_calls"))
        if tool_calls and pending_tool_calls is not None:
            pending_tool_calls.extend(tool_calls)
        content_str = str(content) if content is not None else None
        qs_msg = AssistantMessage(content=content_str, tool_calls=tool_calls)

    elif role in ("tool", "function"):
        tool_call_id = str(data.get("tool_call_id") or "")
        name = str(data.get("name") or "")

        if pending_tool_calls:
            matching_index = next(
                (
                    index
                    for index, call in enumerate(pending_tool_calls)
                    if call.get_id() == tool_call_id
                ),
                None,
            )
            if matching_index is None and not tool_call_id:
                matching_index = 0
            if matching_index is not None:
                matched_call = pending_tool_calls.pop(matching_index)
                tool_call_id = tool_call_id or matched_call.get_id()
                name = name or matched_call.get_name()

        if not tool_call_id:
            tool_call_id = "call_default"
        if not name:
            name = "tool_default"

        qs_msg = ToolMessage(
            content=str(content or ""), tool_call_id=tool_call_id, name=name
        )

    else:
        qs_msg = UserMessage(content=str(content or ""))

    if "_fe_chat_message_snapshot" in data:
        qs_msg._fe_chat_message_snapshot = data["_fe_chat_message_snapshot"]
    if "_fe_session_metadata" in data:
        qs_msg._fe_session_metadata = data["_fe_session_metadata"]
    if "_fe_steering_message" in data:
        qs_msg._fe_steering_message = data["_fe_steering_message"]

    return qs_msg


def qs_message_to_raw_dict(
    msg: Message[Any], include_snapshots: bool = False
) -> dict[str, Any]:
    """Converts a qs Message instance to a clean, standardized raw history dict."""
    role = msg.get_role_name()

    if isinstance(msg, SystemMessage):
        res: dict[str, Any] = {
            "role": "system",
            "content": str(msg.get_message() or ""),
        }

    elif isinstance(msg, UserMessage):
        res = {"role": "user", "content": str(msg.get_message() or "")}

    elif isinstance(msg, AssistantMessage):
        tool_calls = msg.get_tool_calls()
        tc_data = None
        if tool_calls:
            tc_data = [
                {
                    "id": tc.get_id(),
                    "type": "function",
                    "function": {
                        "name": tc.get_name(),
                        "arguments": tc.get_arguments(),
                    },
                }
                for tc in tool_calls
            ]
        res = {"role": "assistant", "content": msg.get_message()}
        if tc_data is not None:
            res["tool_calls"] = tc_data

    elif isinstance(msg, ToolMessage):
        res = {
            "role": "tool",
            "name": msg.get_name(),
            "tool_call_id": msg.get_tool_call_id(),
            "content": str(msg.get_message() or ""),
        }

    else:
        res = {"role": role, "content": str(msg.get_message() or "")}

    if include_snapshots:
        if hasattr(msg, "_fe_chat_message_snapshot"):
            res["_fe_chat_message_snapshot"] = msg._fe_chat_message_snapshot
        if hasattr(msg, "_fe_session_metadata"):
            res["_fe_session_metadata"] = msg._fe_session_metadata
        if hasattr(msg, "_fe_steering_message"):
            res["_fe_steering_message"] = msg._fe_steering_message

    return res


def _serialize_skill_node_to_tool_call(
    skill: SkillNode, index: int
) -> tuple[ToolCall, ToolMessage]:
    """Serializes a SkillNode (and its recursive sub_skills and tool_events) into a ToolCall and ToolMessage pair."""
    call_id = f"call_skill_{skill.name}_{index}"
    args = {
        "path": skill.path,
        "depth": skill.depth,
        "status": skill.status,
        "skill_node": skill.model_dump(),
    }
    tc = ToolCall(id=call_id, function=FunctionCall(name=skill.name, arguments=args))
    tm = ToolMessage(
        content=f'<DeepSkill name="{skill.name}" path="{skill.path}" depth="{skill.depth}" status="{skill.status}">Completed {skill.name}</DeepSkill>',
        tool_call_id=call_id,
        name=skill.name,
    )
    return tc, tm


def close_pending_tool_calls(
    messages: list[Message[Any]], pending_tool_calls: list[ToolCall]
) -> None:
    """Repair an interrupted tool batch before the next conversation message."""
    for tool_call in pending_tool_calls:
        messages.append(
            ToolMessage(
                content="[Execution interrupted]",
                tool_call_id=tool_call.get_id(),
                name=tool_call.get_name(),
            )
        )
    pending_tool_calls.clear()


def chat_message_to_qs_messages(msg: ChatMessage) -> list[Message[Any]]:
    """Converts a ChatMessage into a sequence of qs Message objects.

    Guarantees valid MessageHistory role transitions:
    - user ChatMessage -> [UserMessage]
    - system ChatMessage -> [SystemMessage]
    - assistant ChatMessage -> [AssistantMessage(tool_calls), ToolMessage, ..., AssistantMessage(final)]
    """
    if msg.role == "user":
        um = UserMessage(content=msg.content)
        um._fe_chat_message_snapshot = msg.model_dump()
        return [um]

    if msg.role == "system":
        sm = SystemMessage(content=msg.content)
        sm._fe_chat_message_snapshot = msg.model_dump()
        return [sm]

    qs_list: list[Message[Any]] = []

    if msg.raw_history:
        raw_slice = msg.turn_raw_history()
        pending_tool_calls: list[ToolCall] = []
        for item in raw_slice:
            qs_msg = raw_dict_to_qs_message(item, pending_tool_calls=pending_tool_calls)
            if isinstance(qs_msg, UserMessage):
                qs_msg._fe_steering_message = True
            # Ensure a ToolMessage is never the first message of an assistant turn without a preceding AssistantMessage
            if isinstance(qs_msg, ToolMessage) and not qs_list:
                synth_tc = ToolCall(
                    id=qs_msg.get_tool_call_id(),
                    function=FunctionCall(name=qs_msg.get_name(), arguments={}),
                )
                qs_list.append(AssistantMessage(content=None, tool_calls=[synth_tc]))
            qs_list.append(qs_msg)

        close_pending_tool_calls(qs_list, pending_tool_calls)

    if not qs_list:
        if msg.skills:
            tool_calls: list[ToolCall] = []
            tool_msgs: list[ToolMessage] = []
            for idx, sk in enumerate(msg.skills):
                tc, tm = _serialize_skill_node_to_tool_call(sk, idx + 1)
                tool_calls.append(tc)
                tool_msgs.append(tm)
            qs_list.append(AssistantMessage(content=None, tool_calls=tool_calls))
            qs_list.extend(tool_msgs)

        if msg.tool_events:
            tool_calls = []
            tool_msgs = []
            for idx, ev in enumerate(msg.tool_events):
                cid = ev.tool_call_id or f"call_tool_{ev.tool_name}_{idx + 1}"
                tc = ToolCall(
                    id=cid,
                    function=FunctionCall(
                        name=ev.tool_name,
                        arguments=ev.arguments
                        or {
                            "depth": ev.depth,
                            "skill_path": ev.skill_path,
                            "event_type": ev.event_type,
                        },
                    ),
                )
                tm = ToolMessage(
                    content=f'<Tool name="{ev.tool_name}" depth="{ev.depth}" path="{ev.skill_path}">Output of {ev.tool_name}</Tool>',
                    tool_call_id=cid,
                    name=ev.tool_name,
                )
                tool_calls.append(tc)
                tool_msgs.append(tm)
            qs_list.append(AssistantMessage(content=None, tool_calls=tool_calls))
            qs_list.extend(tool_msgs)

        qs_list.append(AssistantMessage(content=msg.content))

    # Ensure the sequence ends with an AssistantMessage so transition to next UserMessage is valid
    if not isinstance(qs_list[-1], AssistantMessage):
        qs_list.append(AssistantMessage(content=msg.content or "[Interrupted]"))

    # Attach lossless snapshot to the final AssistantMessage of the turn
    qs_list[-1]._fe_chat_message_snapshot = msg.model_dump()
    return qs_list


def qs_messages_to_chat_messages(
    qs_messages: Sequence[Message[Any]],
) -> list[ChatMessage]:
    """Converts a sequence of qs Message objects back into structured ChatMessage instances."""
    chat_messages: list[ChatMessage] = []
    current_assistant_qs: list[Message[Any]] = []

    def flush_assistant() -> None:
        nonlocal current_assistant_qs
        if not current_assistant_qs:
            return

        last_msg = current_assistant_qs[-1]
        snapshot = getattr(last_msg, "_fe_chat_message_snapshot", None)
        if isinstance(snapshot, dict):
            chat_messages.append(ChatMessage.model_validate(snapshot))
            current_assistant_qs = []
            return

        raw_hist = [qs_message_to_raw_dict(m) for m in current_assistant_qs]
        tool_events: list[ToolEvent] = []
        skills: list[SkillNode] = []
        final_content = ""

        for m in current_assistant_qs:
            if isinstance(m, AssistantMessage):
                if m.get_message() is not None:
                    final_content = str(m.get_message() or "")
                tcs = m.get_tool_calls()
                if tcs:
                    for tc in tcs:
                        fn_name = tc.get_name()
                        fn_args = tc.get_arguments() or {}
                        if isinstance(fn_args, dict) and "skill_node" in fn_args:
                            skills.append(
                                SkillNode.model_validate(fn_args["skill_node"])
                            )
                        elif "skill" in fn_name.lower() or (
                            isinstance(fn_args, dict) and "path" in fn_args
                        ):
                            path = str(fn_args.get("path") or f"/{fn_name}")
                            depth = int(fn_args.get("depth", 1))
                            skills.append(
                                SkillNode(
                                    name=fn_name,
                                    path=path,
                                    depth=depth,
                                    status="completed",
                                )
                            )
                        else:
                            depth = (
                                int(fn_args.get("depth", 0))
                                if isinstance(fn_args, dict)
                                else 0
                            )
                            skill_path = (
                                str(fn_args.get("skill_path", "/"))
                                if isinstance(fn_args, dict)
                                else "/"
                            )
                            ev_type = (
                                str(fn_args.get("event_type", "execution"))
                                if isinstance(fn_args, dict)
                                else "execution"
                            )
                            tool_events.append(
                                ToolEvent(
                                    event_type=ev_type,
                                    tool_name=fn_name,
                                    tool_call_id=tc.get_id(),
                                    arguments=fn_args
                                    if isinstance(fn_args, dict)
                                    else {},
                                    depth=depth,
                                    skill_path=skill_path,
                                )
                            )
            # Embedded steering stays in the turn until its final snapshot arrives.
            elif isinstance(m, ToolMessage):  # pragma: no branch
                tname = m.get_name()
                if not any(e.tool_name == tname for e in tool_events) and not any(
                    s.name == tname for s in skills
                ):
                    tool_events.append(
                        ToolEvent(tool_name=tname, depth=0, skill_path="/")
                    )

        chat_messages.append(
            ChatMessage(
                role="assistant",
                content=final_content,
                tool_events=tool_events,
                skills=skills,
                raw_history=raw_hist,
            )
        )
        current_assistant_qs = []

    for msg in qs_messages:
        if getattr(msg, "_fe_steering_message", False):
            current_assistant_qs.append(msg)
        elif isinstance(msg, SystemMessage):
            flush_assistant()
            snap = getattr(msg, "_fe_chat_message_snapshot", None)
            if isinstance(snap, dict):
                chat_messages.append(ChatMessage.model_validate(snap))
            else:
                chat_messages.append(
                    ChatMessage(role="system", content=str(msg.get_message() or ""))
                )
        elif isinstance(msg, UserMessage):
            flush_assistant()
            snap = getattr(msg, "_fe_chat_message_snapshot", None)
            if isinstance(snap, dict):
                chat_messages.append(ChatMessage.model_validate(snap))
            else:
                chat_messages.append(
                    ChatMessage(role="user", content=str(msg.get_message() or ""))
                )
        elif isinstance(msg, (AssistantMessage, ToolMessage)):
            current_assistant_qs.append(msg)
        else:
            flush_assistant()
            role = getattr(msg, "get_role_name", lambda: "user")()
            chat_messages.append(
                ChatMessage(role=role, content=str(msg.get_message() or ""))
            )

    flush_assistant()
    return chat_messages


def session_to_qs_messages(session: ChatSession) -> list[Message[Any]]:
    """Bidirectional conversion: ChatSession -> list[qs.interfaces.message.Message]."""
    qs_messages: list[Message[Any]] = []
    sys_prompt = (
        session.system_prompt
        or "You are a friendly, accurate, and professional AI assistant."
    )
    sys_msg = SystemMessage(content=sys_prompt)
    sys_msg._fe_session_metadata = {
        "id": session.id,
        "title": session.title,
        "project_id": session.project_id,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "model": session.model,
        "agent_mode": session.agent_mode,
        "system_prompt": session.system_prompt,
        "settings": session.settings.model_dump(),
        "queued_messages": [m.model_dump() for m in session.queued_messages],
    }
    qs_messages.append(sys_msg)

    for msg in session.messages:
        qs_messages.extend(chat_message_to_qs_messages(msg))

    return qs_messages


def qs_messages_to_session(
    qs_messages: Sequence[Message[Any]],
    session_id: str | None = None,
    title: str = "New Conversation",
    model: str = "Qwen3.8-27B-UD-Q4_K_S",
    agent_mode: str = "deep_agent",
) -> ChatSession:
    """Bidirectional conversion: list[qs.interfaces.message.Message] -> ChatSession."""
    sys_prompt = ""
    session_meta: dict[str, Any] | None = None
    filtered_qs: list[Message[Any]] = []

    for i, m in enumerate(qs_messages):
        if i == 0 and isinstance(m, SystemMessage):
            sys_prompt = str(m.get_message() or "")
            session_meta = getattr(m, "_fe_session_metadata", None)
        else:
            filtered_qs.append(m)

    messages = qs_messages_to_chat_messages(filtered_qs)

    if isinstance(session_meta, dict):
        raw_queued = session_meta.get("queued_messages", [])
        queued_messages = [
            QueuedMessage.model_validate(q) if isinstance(q, dict) else q
            for q in raw_queued
        ]
        return ChatSession(
            id=session_id or session_meta.get("id"),
            title=session_meta.get("title", title),
            project_id=session_meta.get("project_id"),
            created_at=session_meta.get("created_at") or time.time(),
            updated_at=session_meta.get("updated_at") or time.time(),
            model=session_meta.get("model", model),
            agent_mode=session_meta.get("agent_mode", agent_mode),
            system_prompt=session_meta.get("system_prompt", sys_prompt),
            settings=session_meta.get("settings", {}),
            messages=messages,
            queued_messages=queued_messages,
        )

    session = ChatSession(
        title=title,
        model=model,
        agent_mode=agent_mode
        if agent_mode in ("deep_agent", "quick_chat")
        else "deep_agent",
        system_prompt=sys_prompt,
        messages=messages,
    )
    if session_id:
        session.id = session_id
    return session


def _parse_xml_header_output(text: str) -> tuple[str, str, str, str] | None:
    """Parse an XmlHeader (<header>...</header>\\nbody) into (name, source_type, id, body)."""
    if not isinstance(text, str) or not text.lstrip().startswith("<header>"):
        return None
    end_tag = "</header>"
    end_idx = text.find(end_tag)
    if end_idx < 0:
        return None
    header_xml = text[: end_idx + len(end_tag)]
    body = text[end_idx + len(end_tag) :].lstrip("\r\n")
    try:
        root = ET.fromstring(header_xml)
        name = (root.findtext("name") or "").strip()
        source_type = (root.findtext("type") or "Tool").strip()
        output_id = (root.findtext("id") or "").strip()
        if not output_id:
            return None
        return name, source_type, output_id, body
    except Exception:
        return None


def reconstruct_execution_history_and_cache(
    messages: Sequence[ChatMessage],
    execution_history: ExecutionHistory,
    node_cache: NodeCache | None = None,
) -> None:
    """Reconstruct ExecutionHistory entries and NodeCache ToolCallRecords from prior ChatMessages.

    Walks each assistant turn's new raw_history segment and its skill_history_tree so that
    get_execution_history, search_tool_outputs, and read_tool_output see all
    tools and skills executed in prior turns (including nested skills and exit_agent calls).
    """

    def _walk_scope(
        raw_slice: list[dict[str, Any]],
        children_nodes: list[dict[str, Any]],
        skill_path: str,
    ) -> None:
        children_queue = list(children_nodes)
        i = 0
        n = len(raw_slice)
        while i < n:
            item = raw_slice[i]
            if item.get("role") != "assistant":
                i += 1
                continue

            tcs = parse_tool_calls(item.get("tool_calls")) or []
            following_tools: list[dict[str, Any]] = []
            j = i + 1
            while j < n and raw_slice[j].get("role") == "tool":
                following_tools.append(raw_slice[j])
                j += 1

            for idx, tc in enumerate(tcs):
                tc_name = tc.get_name()
                tc_id = tc.get_id()
                tc_args = tc.get_arguments()
                entry = execution_history.record(tc_name, tc_id)
                matched_child = take_skill_for_call(children_queue, tc)
                tool_result = tool_result_for_call(following_tools, tc_id, idx)

                if matched_child is not None:
                    child_raw = matched_child.get("history") or []
                    inner_slice = inner_skill_history_messages(child_raw)
                    child_children = matched_child.get("children") or []
                    child_path = (
                        f"/{tc_name}"
                        if skill_path == "/"
                        else f"{skill_path.rstrip('/')}/{tc_name}"
                    )
                    _walk_scope(inner_slice, child_children, child_path)

                    child_output = str(
                        matched_child.get("output")
                        or (
                            tool_result.get("content", "")
                            if tool_result is not None
                            else ""
                        )
                    )
                    parsed = _parse_xml_header_output(child_output)
                    if parsed is not None and node_cache is not None:
                        _, _, output_id, body = parsed
                        record = ToolCallRecord(
                            name=tc_name,
                            id=output_id,
                            tool_call_id=tc_id,
                            args=tc_args,
                            result={"text": body},
                            skill_path=skill_path,
                            thinking=None,
                            source="DeepSkill",
                        )
                        node_cache.store(record)
                        execution_history.attach_output(
                            tc_id, output_id, node_cache.session_dir
                        )
                else:
                    if tool_result is not None:
                        tm_content = str(tool_result.get("content", ""))
                        parsed = _parse_xml_header_output(tm_content)
                        if parsed is not None and node_cache is not None:
                            _, h_type, output_id, body = parsed
                            source_lit = (
                                "DeepSkill" if h_type == "DeepSkill" else "Tool"
                            )
                            res_val: Any = (
                                {"text": body} if source_lit == "DeepSkill" else body
                            )
                            record = ToolCallRecord(
                                name=tc_name,
                                id=output_id,
                                tool_call_id=tc_id,
                                args=tc_args,
                                result=res_val,
                                skill_path=skill_path,
                                thinking=None,
                                source=source_lit,
                            )
                            node_cache.store(record)
                            entry.attach_output(output_id, node_cache.session_dir)

            i = j

    prev_raw: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role != "assistant":
            continue
        if msg.raw_history:
            raw = msg.raw_history
            turn_slice = msg.turn_raw_history(previous_history=prev_raw)
            prev_raw = raw
            _walk_scope(turn_slice, list(msg.skill_history_tree or []), "/")
        elif msg.tool_events:
            seen_call_ids: set[str] = set()
            for idx, ev in enumerate(msg.tool_events):
                call_id = ev.tool_call_id or f"call_{msg.id[:8]}_{idx}"
                if call_id not in seen_call_ids:
                    execution_history.record(ev.tool_name, call_id)
                    seen_call_ids.add(call_id)


def restore_node_history(
    node: Node,
    history: Sequence[ChatMessage] | None,
    prior_qs: list[Message[Any]],
) -> None:
    """Convert saved FE state into the public QS node restoration API."""
    if not history:
        return

    def restore_execution_state(
        execution_history: ExecutionHistory, node_cache: NodeCache
    ) -> None:
        try:
            reconstruct_execution_history_and_cache(
                history, execution_history, node_cache
            )
        except Exception as error:  # noqa: BLE001 - legacy history is best-effort
            logger.warning("Failed to restore node execution history: %s", error)

    try:
        node.restore_history(prior_qs, restore_execution_state)
    except Exception as error:  # noqa: BLE001 - legacy history is best-effort
        logger.warning("Failed to restore node message history: %s", error)
