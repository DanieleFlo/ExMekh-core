from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.tool_calls.execution_history import ExecutionHistory
from exmekh_lib.interfaces.message import Message

from exmekh_agent.exmekh_bridge.execution_restore import ExecutionHistoryRestorer
from exmekh_agent.exmekh_bridge.image_input import (
    deserialize_user_content,
    serialize_user_content,
    user_content,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    SkillNode,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.tool_calls import parse_tool_calls

logger = logging.getLogger(__name__)
AGENT_METADATA_KEYS = (
    "_fe_chat_message_snapshot",
    "_fe_session_metadata",
    "_fe_steering_message",
)


def raw_dict_to_exmekh_message(
    data: dict[str, Any],
    pending_tool_calls: list[ToolCall] | None = None,
) -> Message[Any]:
    """Converts a raw dictionary (from raw_history or dump_history) to a exmekh_lib Message instance."""
    role = str(data.get("role", "user")).lower()
    content = data.get("content")
    exmekh_msg: Message[Any]

    if role == "system":
        exmekh_msg = SystemMessage(content=str(content or ""))

    elif role == "user":
        exmekh_msg = UserMessage(content=deserialize_user_content(content))

    elif role == "assistant":
        tool_calls = parse_tool_calls(data.get("tool_calls"))
        if tool_calls and pending_tool_calls is not None:
            pending_tool_calls.extend(tool_calls)
        content_str = str(content) if content is not None else None
        exmekh_msg = AssistantMessage(content=content_str, tool_calls=tool_calls)

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

        exmekh_msg = ToolMessage(
            content=str(content or ""), tool_call_id=tool_call_id, name=name
        )

    else:
        exmekh_msg = UserMessage(content=deserialize_user_content(content))

    for key in AGENT_METADATA_KEYS:
        if key in data:
            exmekh_msg.set_metadata(key, data[key])

    return exmekh_msg


def exmekh_message_to_raw_dict(
    msg: Message[Any], include_snapshots: bool = False
) -> dict[str, Any]:
    """Converts a exmekh_lib Message instance to a clean, standardized raw history dict."""
    role = msg.get_role_name()

    if isinstance(msg, SystemMessage):
        res: dict[str, Any] = {
            "role": "system",
            "content": str(msg.get_message() or ""),
        }

    elif isinstance(msg, UserMessage):
        res = {"role": "user", "content": serialize_user_content(msg)}

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
        metadata = msg.get_metadata()
        res.update(
            {key: metadata[key] for key in AGENT_METADATA_KEYS if key in metadata}
        )

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


def chat_message_to_exmekh_messages(msg: ChatMessage) -> list[Message[Any]]:
    """Converts a ChatMessage into a sequence of exmekh_lib Message objects.

    Guarantees valid MessageHistory role transitions:
    - user ChatMessage -> [UserMessage]
    - system ChatMessage -> [SystemMessage]
    - assistant ChatMessage -> [AssistantMessage(tool_calls), ToolMessage, ..., AssistantMessage(final)]
    """
    if msg.role == "user":
        um = UserMessage(content=user_content(msg.content, msg.image_urls))
        um.set_metadata("_fe_chat_message_snapshot", msg.model_dump())
        return [um]

    if msg.role == "system":
        sm = SystemMessage(content=msg.content)
        sm.set_metadata("_fe_chat_message_snapshot", msg.model_dump())
        return [sm]

    exmekh_list: list[Message[Any]] = []

    if msg.raw_history:
        raw_slice = msg.turn_raw_history()
        pending_tool_calls: list[ToolCall] = []
        for item in raw_slice:
            exmekh_msg = raw_dict_to_exmekh_message(
                item, pending_tool_calls=pending_tool_calls
            )
            if isinstance(exmekh_msg, UserMessage):
                exmekh_msg.set_metadata("_fe_steering_message", True)
            # Ensure a ToolMessage is never the first message of an assistant turn without a preceding AssistantMessage
            if isinstance(exmekh_msg, ToolMessage) and not exmekh_list:
                synth_tc = ToolCall(
                    id=exmekh_msg.get_tool_call_id(),
                    function=FunctionCall(name=exmekh_msg.get_name(), arguments={}),
                )
                exmekh_list.append(
                    AssistantMessage(content=None, tool_calls=[synth_tc])
                )
            exmekh_list.append(exmekh_msg)

        close_pending_tool_calls(exmekh_list, pending_tool_calls)

    if not exmekh_list:
        if msg.skills:
            tool_calls: list[ToolCall] = []
            tool_msgs: list[ToolMessage] = []
            for idx, sk in enumerate(msg.skills):
                tc, tm = _serialize_skill_node_to_tool_call(sk, idx + 1)
                tool_calls.append(tc)
                tool_msgs.append(tm)
            exmekh_list.append(AssistantMessage(content=None, tool_calls=tool_calls))
            exmekh_list.extend(tool_msgs)

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
            exmekh_list.append(AssistantMessage(content=None, tool_calls=tool_calls))
            exmekh_list.extend(tool_msgs)

        exmekh_list.append(AssistantMessage(content=msg.content))

    # Ensure the sequence ends with an AssistantMessage so transition to next UserMessage is valid
    if not isinstance(exmekh_list[-1], AssistantMessage):
        exmekh_list.append(AssistantMessage(content=msg.content or "[Interrupted]"))

    # Attach lossless snapshot to the final AssistantMessage of the turn
    exmekh_list[-1].set_metadata("_fe_chat_message_snapshot", msg.model_dump())
    return exmekh_list


def exmekh_messages_to_chat_messages(
    exmekh_messages: Sequence[Message[Any]],
) -> list[ChatMessage]:
    """Converts a sequence of exmekh_lib Message objects back into structured ChatMessage instances."""
    chat_messages: list[ChatMessage] = []
    current_assistant_exmekh: list[Message[Any]] = []

    def flush_assistant() -> None:
        nonlocal current_assistant_exmekh
        if not current_assistant_exmekh:
            return

        last_msg = current_assistant_exmekh[-1]
        snapshot = last_msg.get_metadata().get("_fe_chat_message_snapshot", None)
        if isinstance(snapshot, dict):
            chat_messages.append(ChatMessage.model_validate(snapshot))
            current_assistant_exmekh = []
            return

        raw_hist = [exmekh_message_to_raw_dict(m) for m in current_assistant_exmekh]
        tool_events: list[ToolEvent] = []
        skills: list[SkillNode] = []
        final_content = ""

        for m in current_assistant_exmekh:
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
        current_assistant_exmekh = []

    for msg in exmekh_messages:
        if msg.get_metadata().get("_fe_steering_message", False):
            current_assistant_exmekh.append(msg)
        elif isinstance(msg, SystemMessage):
            flush_assistant()
            snap = msg.get_metadata().get("_fe_chat_message_snapshot", None)
            if isinstance(snap, dict):
                chat_messages.append(ChatMessage.model_validate(snap))
            else:
                chat_messages.append(
                    ChatMessage(role="system", content=str(msg.get_message() or ""))
                )
        elif isinstance(msg, UserMessage):
            flush_assistant()
            snap = msg.get_metadata().get("_fe_chat_message_snapshot", None)
            if isinstance(snap, dict):
                chat_messages.append(ChatMessage.model_validate(snap))
            else:
                chat_messages.append(
                    ChatMessage(
                        role="user",
                        content="\n".join(
                            part
                            for part in msg.iter_content_parts()
                            if isinstance(part, str)
                        ),
                        image_urls=[
                            part["image_url"]["url"]
                            for part in serialize_user_content(msg)
                            if isinstance(part, dict) and part["type"] == "image_url"
                        ]
                        if not isinstance(msg.get_message(), str)
                        else [],
                    )
                )
        elif isinstance(msg, (AssistantMessage, ToolMessage)):
            current_assistant_exmekh.append(msg)
        else:
            flush_assistant()
            role = msg.get_role_name()
            chat_messages.append(
                ChatMessage(role=role, content=str(msg.get_message() or ""))
            )

    flush_assistant()
    return chat_messages


def session_to_exmekh_messages(session: ChatSession) -> list[Message[Any]]:
    """Bidirectional conversion: ChatSession -> list[exmekh_lib.interfaces.message.Message]."""
    exmekh_messages: list[Message[Any]] = []
    sys_prompt = (
        session.system_prompt
        or "You are a friendly, accurate, and professional AI assistant."
    )
    sys_msg = SystemMessage(content=sys_prompt)
    sys_msg.set_metadata(
        "_fe_session_metadata",
        {
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
        },
    )
    exmekh_messages.append(sys_msg)

    for msg in session.messages:
        exmekh_messages.extend(chat_message_to_exmekh_messages(msg))

    return exmekh_messages


def exmekh_messages_to_session(
    exmekh_messages: Sequence[Message[Any]],
    session_id: str | None = None,
    title: str = "New Conversation",
    model: str = "Qwen3.8-27B-UD-Q4_K_S",
    agent_mode: str = "deep_agent",
) -> ChatSession:
    """Bidirectional conversion: list[exmekh_lib.interfaces.message.Message] -> ChatSession."""
    sys_prompt = ""
    session_meta: dict[str, Any] | None = None
    filtered_exmekh: list[Message[Any]] = []

    for i, m in enumerate(exmekh_messages):
        if i == 0 and isinstance(m, SystemMessage):
            sys_prompt = str(m.get_message() or "")
            session_meta = m.get_metadata().get("_fe_session_metadata", None)
        else:
            filtered_exmekh.append(m)

    messages = exmekh_messages_to_chat_messages(filtered_exmekh)

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


def reconstruct_execution_history_and_cache(
    messages: Sequence[ChatMessage],
    execution_history: ExecutionHistory,
    node_cache: NodeCache | None = None,
) -> None:
    """Restore tool, skill and steering state from persisted frontend turns."""
    ExecutionHistoryRestorer(execution_history, node_cache).restore(messages)


def restore_node_history(
    node: Node,
    history: Sequence[ChatMessage] | None,
    prior_exmekh: list[Message[Any]],
) -> None:
    """Convert saved ExMekh Agent state into the public ExMekh-Lib node restoration API."""
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
        node.restore_history(prior_exmekh, restore_execution_state)
    except Exception as error:  # noqa: BLE001 - legacy history is best-effort
        logger.warning("Failed to restore node message history: %s", error)
