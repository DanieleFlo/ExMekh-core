from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field

from fe.qs_bridge.image_input import content_text
from fe.qs_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from fe.qs_bridge.skill_history import inner_skill_history_messages, take_skill_for_call
from fe.qs_bridge.system_prompt import load_real_hds_system_prompt
from fe.qs_bridge.tool_calls import parse_tool_calls, tool_result_for_call

GraphNodeKind = Literal[
    "system",
    "user",
    "steering",
    "tool_bundle",
    "skill_bundle",
    "sub_agent_bundle",
    "assistant",
    "error",
]

NODE_COLORS: dict[GraphNodeKind, str] = {
    "system": "#f472b6",
    "user": "#38bdf8",
    "steering": "#fb923c",
    "tool_bundle": "#fbbf24",
    "skill_bundle": "#a855f7",
    "sub_agent_bundle": "#22d3ee",
    "assistant": "#34d399",
    "error": "#f87171",
}


class GraphNode(BaseModel):
    """Represents a single interactive graph node along a history line or counterclockwise skill offset."""

    id: str
    history_id: str
    history_label: str
    step_index: int
    kind: GraphNodeKind
    label: str
    short_label: str
    color: str
    depth: int = 0
    skill_path: str = "/"
    skill_name: str | None = None
    sub_agent_target_id: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    x: float = 500.0
    y: float = 215.0
    base_x: float = 500.0
    base_y: float = 215.0


class GraphSkillBranch(BaseModel):
    """Represents a counterclockwise-offset branch path for an opened skill."""

    skill_name: str
    skill_path: str
    depth: int
    color: str
    points: list[tuple[float, float]] = Field(default_factory=list)
    label_x: float = 500.0
    label_y: float = 260.0


class GraphInteractionLink(BaseModel):
    """Represents an interaction intersection or secondary constellation chord between two histories."""

    from_history_id: str
    to_history_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = "#22d3ee"


class GraphHistoryLine(BaseModel):
    """Represents one straight history line in the constellation."""

    id: str
    label: str
    is_main: bool = False
    parent_history_id: str | None = None
    interaction_x: float = 500.0
    interaction_y: float = 215.0
    angle_rad: float = 0.0
    angle_deg: float = 0.0
    x1: float = 110.0
    y1: float = 215.0
    x2: float = 890.0
    y2: float = 215.0
    line_color: str = "#a1a1aa"
    nodes: list[GraphNode] = Field(default_factory=list)
    skill_branches: list[GraphSkillBranch] = Field(default_factory=list)


class SessionGraphData(BaseModel):
    """Complete constellation graph layout and grouped node events for one ChatSession."""

    session_id: str
    session_title: str
    width: float = 1000.0
    height: float = 500.0
    center_x: float = 500.0
    center_y: float = 215.0
    histories: list[GraphHistoryLine] = Field(default_factory=list)
    interaction_links: list[GraphInteractionLink] = Field(default_factory=list)

    def find_node(self, node_id: str | None) -> GraphNode | None:
        if not node_id:
            return None
        for history in self.histories:
            for node in history.nodes:
                if node.id == node_id:
                    return node
        return None

    @property
    def total_nodes(self) -> int:
        return sum(len(h.nodes) for h in self.histories)


def _truncate_label(text: str, max_len: int = 18) -> str:
    clean = text.strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1] + "…"


def _steering_message_ids(nodes: list[GraphNode]) -> set[str]:
    return {
        event["message_id"]
        for node in nodes
        if node.kind == "steering"
        for event in node.events
        if event.get("message_id")
    }


def _append_steering_node(
    nodes: list[GraphNode],
    *,
    content: str,
    history_id: str,
    history_label: str,
    queued_message: QueuedMessage | None = None,
    message_id: str | None = None,
    pending: bool = False,
    depth: int = 0,
    skill_path: str = "/",
    skill_name: str | None = None,
) -> None:
    event: dict[str, Any] = {
        "role": "user",
        "event_stage": "steering_message",
        "stage_label": "Steering Message",
        "content": content,
        "pending": pending,
    }
    if queued_message is not None:
        event["message_id"] = queued_message.id
        event["timestamp"] = queued_message.created_at
        if queued_message.display_content is not None:
            event["display_content"] = queued_message.display_content
    elif message_id:
        event["message_id"] = message_id
    step_index = len(nodes)
    nodes.append(
        GraphNode(
            id=f"{history_id}_node_{step_index}",
            history_id=history_id,
            history_label=history_label,
            step_index=step_index,
            kind="steering",
            label=f"Steering: {_truncate_label(content, 24)}",
            short_label="Pending steering" if pending else "Steering",
            color=NODE_COLORS["steering"],
            depth=depth,
            skill_path=skill_path,
            skill_name=skill_name,
            events=[event],
        )
    )


def _normalize_skill_path(skill_path: str | None) -> str:
    clean = (skill_path or "/").strip("/")
    return f"/{clean}" if clean else "/"


def _path_step_depth(skill_path: str, fallback_depth: int = 0) -> int:
    """Return a deterministic step count for a skill_path so all nodes in the same path share the exact same offset."""
    norm = _normalize_skill_path(skill_path)
    if norm == "/":
        return 0
    segments = [seg for seg in norm.strip("/").split("/") if seg]
    return max(1, len(segments), fallback_depth)


def _build_tool_triplet_events(
    *,
    tool_name: str,
    tool_call_id: str,
    arguments: Any,
    assistant_content: str,
    tool_event: ToolEvent | None,
    tool_response_content: str,
    skill_path: str,
    depth: int,
) -> list[dict[str, Any]]:
    """Build the 3 grouped events for a tool/skill node:
    1) tool call
    2) tool execution
    3) tool response
    """
    norm_path = _normalize_skill_path(skill_path)
    call_event: dict[str, Any] = {
        "role": "assistant",
        "event_stage": "tool_call",
        "stage_label": "1. Tool Call",
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "skill_path": norm_path,
        "depth": depth,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments if arguments is not None else {},
                },
            }
        ],
    }
    if assistant_content:
        call_event["content"] = assistant_content

    exec_status = tool_event.execution_status if tool_event is not None else "completed"
    exec_type = tool_event.event_type if tool_event is not None else "tool_end"
    exec_duration = (
        round(tool_event.elapsed_seconds(), 4) if tool_event is not None else 0.0
    )
    execution_event: dict[str, Any] = {
        "role": "execution",
        "event_stage": "tool_execution",
        "stage_label": "2. Tool Execution",
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "event_type": exec_type,
        "status": exec_status,
        "skill_path": norm_path,
        "depth": depth,
        "duration_seconds": exec_duration,
    }

    response_event: dict[str, Any] = {
        "role": "tool",
        "event_stage": "tool_response",
        "stage_label": "3. Tool Response",
        "name": tool_name,
        "tool_call_id": tool_call_id,
        "skill_path": norm_path,
        "depth": depth,
        "content": tool_response_content,
    }
    return [call_event, execution_event, response_event]


def _append_tool_node(
    event: ToolEvent,
    *,
    nodes: list[GraphNode],
    history_id: str,
    history_label: str,
    skill_path: str,
    depth: int,
    skill_name: str | None,
    kind: GraphNodeKind = "tool_bundle",
) -> None:
    step_index = len(nodes)
    call_id = event.tool_call_id or f"call_{event.tool_name}_{step_index}"
    events = _build_tool_triplet_events(
        tool_name=event.tool_name,
        tool_call_id=call_id,
        arguments=event.arguments or {"skill_path": skill_path, "depth": depth},
        assistant_content="",
        tool_event=event,
        tool_response_content=f"Tool '{event.tool_name}': {event.execution_status}",
        skill_path=skill_path,
        depth=depth,
    )
    nodes.append(
        GraphNode(
            id=f"{history_id}_node_{step_index}",
            history_id=history_id,
            history_label=history_label,
            step_index=step_index,
            kind=kind,
            label=f"Tool: {event.tool_name}",
            short_label=_truncate_label(event.tool_name, 14),
            color=NODE_COLORS[kind],
            depth=depth,
            skill_path=skill_path,
            skill_name=skill_name,
            events=events,
        )
    )


def _extract_nodes_from_skill_node(
    skill: SkillNode,
    *,
    history_id: str,
    history_label: str,
    nodes: list[GraphNode],
) -> None:
    """Recursively extract graph nodes from a structured SkillNode tree."""
    eff_path = _normalize_skill_path(skill.path or f"/{skill.name}")
    eff_depth = _path_step_depth(eff_path, max(1, skill.depth))
    call_id = skill.opening_tool_call_id or f"call_skill_{skill.name}_{len(nodes) + 1}"

    open_events = _build_tool_triplet_events(
        tool_name=skill.name,
        tool_call_id=call_id,
        arguments={"path": eff_path, "depth": eff_depth},
        assistant_content="",
        tool_event=ToolEvent(
            tool_call_id=call_id,
            event_type="tool_end" if skill.status != "active" else "tool_start",
            status="completed"
            if skill.status == "completed"
            else ("error" if skill.status == "error" else "running"),
            tool_name=skill.name,
            depth=eff_depth,
            skill_path=eff_path,
            duration_seconds=skill.elapsed_seconds(),
        ),
        tool_response_content=f"Skill '{skill.name}' ({eff_path}) - status: {skill.status}",
        skill_path=eff_path,
        depth=eff_depth,
    )
    step_idx = len(nodes)
    nodes.append(
        GraphNode(
            id=f"{history_id}_node_{step_idx}",
            history_id=history_id,
            history_label=history_label,
            step_index=step_idx,
            kind="skill_bundle",
            label=f"Skill: {skill.name}",
            short_label=_truncate_label(skill.name, 14),
            color="#a855f7" if eff_depth == 1 else "#818cf8",
            depth=eff_depth,
            skill_path=eff_path,
            skill_name=skill.name,
            events=open_events,
        )
    )

    for activity in skill.ordered_activity():
        if isinstance(activity, SkillNode):
            _extract_nodes_from_skill_node(
                activity,
                history_id=history_id,
                history_label=history_label,
                nodes=nodes,
            )
            continue
        tool_path = _normalize_skill_path(
            activity.skill_path if activity.skill_path != "/" else eff_path
        )
        _append_tool_node(
            activity,
            nodes=nodes,
            history_id=history_id,
            history_label=history_label,
            skill_path=tool_path,
            depth=_path_step_depth(tool_path, max(eff_depth, activity.depth)),
            skill_name=skill.name,
        )


def _extract_nodes_from_raw_scope(
    raw_slice: list[dict[str, Any]],
    *,
    skill_tree: list[dict[str, Any]],
    structured_skills: list[SkillNode],
    tool_events: list[ToolEvent],
    history_id: str,
    history_label: str,
    depth: int,
    skill_path: str,
    skill_name: str | None,
    nodes: list[GraphNode],
    message: ChatMessage | None = None,
) -> None:
    """Walk a raw_history scope and emit grouped nodes for tool calls, steering messages, and nested skills."""
    children_queue = list(skill_tree)
    structured_queue = list(structured_skills)
    used_tool_event_ids: set[str] = set()
    norm_scope_path = _normalize_skill_path(skill_path)
    scope_depth = _path_step_depth(norm_scope_path, depth)

    def _match_tool_event(tc_id: str, tc_name: str) -> ToolEvent | None:
        for ev in tool_events:
            if ev.id in used_tool_event_ids:
                continue
            if tc_id and ev.tool_call_id == tc_id:
                used_tool_event_ids.add(ev.id)
                return ev
        for ev in tool_events:
            if ev.id in used_tool_event_ids:
                continue
            if ev.tool_name == tc_name and not ev.tool_call_id:
                used_tool_event_ids.add(ev.id)
                return ev
        return None

    i = 0
    n = len(raw_slice)
    while i < n:
        item = raw_slice[i]
        role = str(item.get("role", "")).lower()
        if role == "user":
            user_content = content_text(item.get("content"))
            raw_message_id = str(item.get("message_id") or "") or None
            emitted_ids = _steering_message_ids(nodes)
            if raw_message_id in emitted_ids:
                i += 1
                continue
            matched_message = (
                message.find_steering_message(
                    user_content,
                    message_id=raw_message_id,
                    excluded_ids=emitted_ids,
                )
                if message is not None
                else None
            )
            confirmed_ids = (
                {
                    insertion.message_id
                    for insertion in message.confirmed_steering_insertions()
                }
                if message is not None
                else set()
            )
            _append_steering_node(
                nodes,
                content=user_content,
                history_id=history_id,
                history_label=history_label,
                queued_message=matched_message,
                message_id=raw_message_id,
                pending=matched_message is not None
                and matched_message.id not in confirmed_ids,
                depth=scope_depth,
                skill_path=norm_scope_path,
                skill_name=skill_name,
            )
            i += 1
            continue

        if role != "assistant":
            i += 1
            continue

        tcs = parse_tool_calls(item.get("tool_calls")) or []
        following_tools: list[dict[str, Any]] = []
        j = i + 1
        while j < n and str(raw_slice[j].get("role", "")).lower() == "tool":
            following_tools.append(raw_slice[j])
            j += 1

        if not tcs:
            i = j
            continue

        for idx, tc in enumerate(tcs):
            tc_name = tc.get_name()
            tc_id = tc.get_id()
            tc_args = tc.get_arguments()
            matched_ev = _match_tool_event(tc_id, tc_name)
            tool_msg = tool_result_for_call(following_tools, tc_id, idx)
            tool_resp_str = (
                str(tool_msg.get("content", "")) if tool_msg is not None else ""
            )

            matched_skill_dict = take_skill_for_call(children_queue, tc)

            matched_struct_skill: SkillNode | None = None
            candidate_path = f"{norm_scope_path.rstrip('/')}/{tc_name}"
            for s_idx, s_node in enumerate(structured_queue):
                if s_node.matches_invocation(candidate_path, tc_id):
                    matched_struct_skill = structured_queue.pop(s_idx)
                    break
            if matched_struct_skill is None:
                for s_idx, s_node in enumerate(structured_queue):
                    if s_node.name == tc_name and not s_node.opening_tool_call_id:
                        matched_struct_skill = structured_queue.pop(s_idx)
                        break

            if matched_skill_dict is not None:
                child_path = _normalize_skill_path(
                    f"/{tc_name}"
                    if norm_scope_path == "/"
                    else f"{norm_scope_path.rstrip('/')}/{tc_name}"
                )
                child_depth = _path_step_depth(child_path, scope_depth + 1)
                child_raw = list(matched_skill_dict.get("history") or [])
                catalog = tool_result_for_call(child_raw[1:], tc_id, 0)
                catalog_msg = (
                    str(catalog.get("content", ""))
                    if catalog is not None and catalog.get("role") == "tool"
                    else str(matched_skill_dict.get("output") or tool_resp_str)
                )
                open_triplet = _build_tool_triplet_events(
                    tool_name=tc_name,
                    tool_call_id=tc_id,
                    arguments=tc_args,
                    assistant_content=str(item.get("content") or ""),
                    tool_event=matched_ev,
                    tool_response_content=catalog_msg,
                    skill_path=child_path,
                    depth=child_depth,
                )
                step_idx = len(nodes)
                nodes.append(
                    GraphNode(
                        id=f"{history_id}_node_{step_idx}",
                        history_id=history_id,
                        history_label=history_label,
                        step_index=step_idx,
                        kind="skill_bundle",
                        label=f"Skill: {tc_name}",
                        short_label=_truncate_label(tc_name, 14),
                        color="#a855f7" if child_depth == 1 else "#818cf8",
                        depth=child_depth,
                        skill_path=child_path,
                        skill_name=tc_name,
                        events=open_triplet,
                    )
                )
                inner_slice = inner_skill_history_messages(child_raw)
                inner_tool_events = (
                    matched_struct_skill.tool_events
                    if matched_struct_skill is not None
                    else tool_events
                )
                inner_sub_skills = (
                    matched_struct_skill.sub_skills
                    if matched_struct_skill is not None
                    else []
                )
                _extract_nodes_from_raw_scope(
                    inner_slice,
                    skill_tree=list(matched_skill_dict.get("children") or []),
                    structured_skills=inner_sub_skills,
                    tool_events=inner_tool_events,
                    history_id=history_id,
                    history_label=history_label,
                    depth=child_depth,
                    skill_path=child_path,
                    skill_name=tc_name,
                    nodes=nodes,
                    message=message,
                )
            elif matched_struct_skill is not None:
                _extract_nodes_from_skill_node(
                    matched_struct_skill,
                    history_id=history_id,
                    history_label=history_label,
                    nodes=nodes,
                )
            else:
                is_sub_agent_call = tc_name == "call_sub_agent"
                kind: GraphNodeKind = (
                    "sub_agent_bundle"
                    if is_sub_agent_call
                    else ("skill_bundle" if tc_name == "exit_agent" else "tool_bundle")
                )
                color = NODE_COLORS[kind]
                sub_target = (
                    tc_args.get("sub_agent", tc_name)
                    if isinstance(tc_args, dict) and is_sub_agent_call
                    else tc_name
                )
                label_prefix = "Sub-Agent" if is_sub_agent_call else "Tool"
                triplet = _build_tool_triplet_events(
                    tool_name=tc_name,
                    tool_call_id=tc_id,
                    arguments=tc_args,
                    assistant_content=str(item.get("content") or ""),
                    tool_event=matched_ev,
                    tool_response_content=tool_resp_str,
                    skill_path=norm_scope_path,
                    depth=scope_depth,
                )
                step_idx = len(nodes)
                nodes.append(
                    GraphNode(
                        id=f"{history_id}_node_{step_idx}",
                        history_id=history_id,
                        history_label=history_label,
                        step_index=step_idx,
                        kind=kind,
                        label=f"{label_prefix}: {sub_target}",
                        short_label=_truncate_label(str(sub_target), 14),
                        color=color,
                        depth=scope_depth,
                        skill_path=norm_scope_path,
                        skill_name=skill_name,
                        events=triplet,
                    )
                )

        i = j


def _extract_real_system_prompt_event(
    messages: list[ChatMessage],
    *,
    fallback_raw_history: list[dict[str, Any]] | None = None,
    explicit_system_prompt: str = "",
    capabilities_path_str: str = "",
    name_hds: str = "main",
    is_sub_agent: bool = False,
) -> dict[str, Any] | None:
    """Extract the real System Prompt event from recorded raw_history or from the real qs Node HDS."""
    for msg in messages:
        for rm in msg.raw_history:
            role = str(rm.get("role", "")).lower()
            content = str(rm.get("content", "")).strip()
            if role in ("system", "developer") and content:
                ev = dict(rm)
                ev["role"] = role
                ev["event_stage"] = "system_prompt"
                ev["stage_label"] = "System Prompt"
                return ev

    if fallback_raw_history:
        for rm in fallback_raw_history:
            role = str(rm.get("role", "")).lower()
            content = str(rm.get("content", "")).strip()
            if role in ("system", "developer") and content:
                ev = dict(rm)
                ev["role"] = role
                ev["event_stage"] = "system_prompt"
                ev["stage_label"] = "System Prompt"
                return ev

    for msg in messages:
        if msg.role == "system" and msg.content.strip():
            return {
                "role": "system",
                "event_stage": "system_prompt",
                "stage_label": "System Prompt",
                "content": msg.content,
            }

    if explicit_system_prompt.strip():
        return {
            "role": "system",
            "event_stage": "system_prompt",
            "stage_label": "System Prompt",
            "content": explicit_system_prompt,
        }

    real_hds_prompt = load_real_hds_system_prompt(
        capabilities_path_str=capabilities_path_str,
        name_hds=name_hds,
        is_sub_agent=is_sub_agent,
    )
    if real_hds_prompt.strip():
        return {
            "role": "system",
            "event_stage": "system_prompt",
            "stage_label": "System Prompt",
            "name_hds": (name_hds or "main").strip("/").split("/")[-1] or "main",
            "content": real_hds_prompt,
        }

    return None


def _build_nodes_for_messages(
    messages: list[ChatMessage],
    *,
    history_id: str,
    history_label: str,
    system_prompt: str = "",
    capabilities_path_str: str = "",
    name_hds: str = "main",
    fallback_raw_history: list[dict[str, Any]] | None = None,
    is_sub_agent: bool = False,
) -> list[GraphNode]:
    """Convert a sequence of ChatMessages into sequential GraphNode items,
    with separate nodes for System Prompt (kind='system') and User (kind='user').
    """
    nodes: list[GraphNode] = []
    prev_raw: list[dict[str, Any]] = []

    has_any_activity = bool(messages or fallback_raw_history)
    if has_any_activity:
        sys_event = _extract_real_system_prompt_event(
            messages,
            fallback_raw_history=fallback_raw_history,
            explicit_system_prompt=system_prompt,
            capabilities_path_str=capabilities_path_str,
            name_hds=name_hds,
            is_sub_agent=is_sub_agent,
        )
        if sys_event is not None:
            hds_short = (name_hds or "main").strip("/").split("/")[-1] or "main"
            step_idx = len(nodes)
            nodes.append(
                GraphNode(
                    id=f"{history_id}_node_{step_idx}",
                    history_id=history_id,
                    history_label=history_label,
                    step_index=step_idx,
                    kind="system",
                    label=f"System Prompt ({hds_short})",
                    short_label="System",
                    color=NODE_COLORS["system"],
                    depth=0,
                    skill_path="/",
                    events=[sys_event],
                )
            )

    if is_sub_agent and not any(m.role == "user" for m in messages):
        raw_user_content = ""
        if fallback_raw_history:
            for rm in fallback_raw_history:
                if (
                    str(rm.get("role", "")).lower() == "user"
                    and str(rm.get("content", "")).strip()
                ):
                    raw_user_content = content_text(rm.get("content"))
                    break
        if not raw_user_content:
            raw_user_content = (
                messages[0].content if messages else f"Task for {history_label}"
            )
        step_idx = len(nodes)
        nodes.append(
            GraphNode(
                id=f"{history_id}_node_{step_idx}",
                history_id=history_id,
                history_label=history_label,
                step_index=step_idx,
                kind="user",
                label=f"User: {_truncate_label(raw_user_content, 24)}",
                short_label="User",
                color=NODE_COLORS["user"],
                depth=0,
                skill_path="/",
                events=[
                    {
                        "role": "user",
                        "event_stage": "user_message",
                        "stage_label": "User Message",
                        "content": raw_user_content,
                    }
                ],
            )
        )

    for msg_idx, msg in enumerate(messages):
        if msg.role == "user":
            step_idx = len(nodes)
            nodes.append(
                GraphNode(
                    id=f"{history_id}_node_{step_idx}",
                    history_id=history_id,
                    history_label=history_label,
                    step_index=step_idx,
                    kind="user",
                    label=f"User: {_truncate_label(msg.content, 24)}",
                    short_label="User",
                    color=NODE_COLORS["user"],
                    depth=0,
                    skill_path="/",
                    events=[
                        {
                            "role": "user",
                            "event_stage": "user_message",
                            "stage_label": "User Message",
                            "content": msg.content,
                        }
                    ],
                )
            )
            continue

        if msg.role != "assistant":
            continue

        effective_raw = list(msg.raw_history)
        if not effective_raw and fallback_raw_history and msg_idx == len(messages) - 1:
            effective_raw = list(fallback_raw_history)

        before_tools_count = len(nodes)
        if effective_raw:
            turn_slice = msg.turn_raw_history(
                previous_history=prev_raw, fallback_history=effective_raw
            )
            prev_raw = effective_raw

            _extract_nodes_from_raw_scope(
                turn_slice,
                skill_tree=list(msg.skill_history_tree or []),
                structured_skills=list(msg.skills or []),
                tool_events=list(msg.tool_events or []),
                history_id=history_id,
                history_label=history_label,
                depth=0,
                skill_path="/",
                skill_name=None,
                nodes=nodes,
                message=msg,
            )

        if not any(
            node.kind in ("tool_bundle", "skill_bundle", "sub_agent_bundle")
            for node in nodes[before_tools_count:]
        ):
            known_skill_paths = {
                _normalize_skill_path(event.skill_path)
                for event in msg.tool_events
                if _normalize_skill_path(event.skill_path) != "/"
            } | {_normalize_skill_path(skill.path) for skill in msg.skills}
            for activity in msg.ordered_activity():
                if isinstance(activity, SkillNode):
                    _extract_nodes_from_skill_node(
                        activity,
                        history_id=history_id,
                        history_label=history_label,
                        nodes=nodes,
                    )
                    continue
                event_path = _normalize_skill_path(activity.skill_path)
                candidate_skill_path = f"/{activity.tool_name}"
                if event_path == "/" and candidate_skill_path in known_skill_paths:
                    event_path = candidate_skill_path
                _append_tool_node(
                    activity,
                    nodes=nodes,
                    history_id=history_id,
                    history_label=history_label,
                    skill_path=event_path,
                    depth=_path_step_depth(event_path, max(0, activity.depth)),
                    skill_name=event_path.strip("/").split("/")[-1]
                    if event_path != "/"
                    else None,
                    kind="skill_bundle"
                    if candidate_skill_path in known_skill_paths
                    else "tool_bundle",
                )

        emitted_steering_ids = _steering_message_ids(nodes[before_tools_count:])
        confirmed_ids = {
            insertion.message_id for insertion in msg.confirmed_steering_insertions()
        }
        for guidance in msg.steering_messages:
            if guidance.id in emitted_steering_ids:
                continue
            _append_steering_node(
                nodes,
                content=guidance.text,
                queued_message=guidance,
                pending=guidance.id not in confirmed_ids,
                history_id=history_id,
                history_label=history_label,
            )
            emitted_steering_ids.add(guidance.id)

        # Final assistant response node back on the main history line (depth = 0)
        resp_events: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "event_stage": "final_response",
                "stage_label": "Final Assistant Response",
                "content": msg.content,
                **({"error": msg.error} if msg.error else {}),
            }
        ]
        resp_kind: GraphNodeKind = "error" if msg.error else "assistant"
        step_idx = len(nodes)
        nodes.append(
            GraphNode(
                id=f"{history_id}_node_{step_idx}",
                history_id=history_id,
                history_label=history_label,
                step_index=step_idx,
                kind=resp_kind,
                label="Execution Error"
                if msg.error
                else f"Assistant: {_truncate_label(msg.content or 'Response', 22)}",
                short_label="Error" if msg.error else "Response",
                color=NODE_COLORS[resp_kind],
                depth=0,
                skill_path="/",
                events=resp_events,
            )
        )

    return nodes


def _ensure_interaction_nodes_on_main(
    main_nodes: list[GraphNode],
    sub_conversations: list[SubAgentConversation],
) -> dict[str, GraphNode]:
    """Ensure every sub-agent conversation has an interaction node on its parent history
    where the two history lines intersect to form a constellation.
    """
    existing_sub_nodes = [p for p in main_nodes if p.kind == "sub_agent_bundle"]
    interaction_map: dict[str, GraphNode] = {}

    for idx, sub_conv in enumerate(sub_conversations):
        if idx < len(existing_sub_nodes):
            node = existing_sub_nodes[idx]
            node.sub_agent_target_id = sub_conv.id
            interaction_map[sub_conv.id] = node
            continue

        # Insert an interaction node right before the final assistant response node on the main line
        insert_pos = max(0, len(main_nodes) - 1) if main_nodes else 0
        sub_label = f"{sub_conv.path} #{sub_conv.order}"
        sub_summary = (
            sub_conv.messages[-1].content
            if sub_conv.messages
            else f"Interaction with {sub_label}"
        )
        call_id = f"call_sub_agent_{sub_conv.id}"
        triplet = _build_tool_triplet_events(
            tool_name="call_sub_agent",
            tool_call_id=call_id,
            arguments={"sub_agent": sub_conv.path, "order": sub_conv.order},
            assistant_content="",
            tool_event=ToolEvent(
                tool_call_id=call_id,
                event_type="tool_end" if sub_conv.status != "active" else "tool_start",
                status="completed"
                if sub_conv.status == "completed"
                else ("error" if sub_conv.status == "error" else "running"),
                tool_name="call_sub_agent",
                depth=0,
                skill_path="/",
            ),
            tool_response_content=sub_summary,
            skill_path="/",
            depth=0,
        )
        inter_node = GraphNode(
            id=f"main_inter_{sub_conv.id}",
            history_id="main",
            history_label="Main History",
            step_index=insert_pos,
            kind="sub_agent_bundle",
            label=f"Interaction: {sub_label}",
            short_label=_truncate_label(sub_conv.path.split("/")[-1], 12),
            color=NODE_COLORS["sub_agent_bundle"],
            depth=0,
            skill_path="/",
            sub_agent_target_id=sub_conv.id,
            events=triplet,
        )
        main_nodes.insert(insert_pos, inter_node)
        interaction_map[sub_conv.id] = inter_node

    # Re-index main_nodes step_index sequentially
    for idx, p in enumerate(main_nodes):
        p.step_index = idx

    return interaction_map


def _apply_skill_offsets_and_branches(
    history: GraphHistoryLine,
    *,
    ux: float,
    uy: float,
    ccw_x: float,
    ccw_y: float,
    offset_step: float,
) -> None:
    """Apply identical offset to all nodes sharing a skill_path and build branch connectors."""
    path_offset_map: dict[str, float] = {"/": 0.0}
    for node in history.nodes:
        norm_path = _normalize_skill_path(node.skill_path)
        node.skill_path = norm_path
        if norm_path not in path_offset_map:
            steps = _path_step_depth(norm_path, node.depth)
            path_offset_map[norm_path] = round(steps * offset_step, 2)
        node.depth = int(round(path_offset_map[norm_path] / offset_step))

    for node in history.nodes:
        ccw_offset = path_offset_map[node.skill_path]
        node.x = round(node.base_x + ccw_offset * ccw_x, 2)
        node.y = round(node.base_y + ccw_offset * ccw_y, 2)

    n = len(history.nodes)
    branches: list[GraphSkillBranch] = []
    i = 0
    while i < n:
        if history.nodes[i].depth > 0:
            j = i
            while j < n and history.nodes[j].depth > 0:
                j += 1
            segment_nodes = history.nodes[i:j]
            pts: list[tuple[float, float]] = []
            if i > 0:
                prev_node = history.nodes[i - 1]
                pts.append((prev_node.x, prev_node.y))
            else:
                pts.append((segment_nodes[0].base_x, segment_nodes[0].base_y))

            for sn in segment_nodes:
                pts.append((sn.x, sn.y))

            if j < n:
                next_node = history.nodes[j]
                pts.append((next_node.x, next_node.y))
            else:
                pts.append((segment_nodes[-1].base_x, segment_nodes[-1].base_y))

            first_skill_node = segment_nodes[0]
            mid_node = segment_nodes[len(segment_nodes) // 2]
            branches.append(
                GraphSkillBranch(
                    skill_name=first_skill_node.skill_name or "skill",
                    skill_path=first_skill_node.skill_path,
                    depth=max(sn.depth for sn in segment_nodes),
                    color=first_skill_node.color,
                    points=pts,
                    label_x=round(mid_node.x + 18.0 * ccw_x, 2),
                    label_y=round(mid_node.y + 18.0 * ccw_y, 2),
                )
            )
            i = j
        else:
            i += 1

    history.skill_branches = branches


FIXED_NODE_SPACING_X: float = 110.0
FIXED_SUB_AGENT_NODE_SPACING: float = 48.0
GRAPH_HORIZONTAL_PADDING: float = 120.0
LINE_HORIZONTAL_EXTENSION: float = 55.0
MIN_GRAPH_WIDTH: float = 360.0
MIN_GRAPH_HEIGHT: float = 500.0


def _segments_cross(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
    tolerance: float = 1e-4,
) -> bool:
    """Return True if line segment (p1, p2) and line segment (p3, p4) strictly cross each other,
    ignoring common attachment endpoints.
    """
    d13 = math.hypot(p1[0] - p3[0], p1[1] - p3[1])
    d14 = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
    d23 = math.hypot(p2[0] - p3[0], p2[1] - p3[1])
    d24 = math.hypot(p2[0] - p4[0], p2[1] - p4[1])
    if min(d13, d14, d23, d24) < 1.5:
        return False

    def ccw(
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
    ) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    c1 = ccw(p1, p2, p3)
    c2 = ccw(p1, p2, p4)
    c3 = ccw(p3, p4, p1)
    c4 = ccw(p3, p4, p2)

    return (
        (c1 > tolerance and c2 < -tolerance) or (c1 < -tolerance and c2 > tolerance)
    ) and ((c3 > tolerance and c4 < -tolerance) or (c3 < -tolerance and c4 > tolerance))


def check_sub_agent_branches_do_not_cross(
    histories: list[GraphHistoryLine],
) -> list[tuple[str, str]]:
    """Check all pairs of sub-agent history lines to verify they do not cross each other.
    Returns a list of (history_a_id, history_b_id) pairs that cross, or an empty list if none cross.
    """
    sub_lines = [h for h in histories if not h.is_main]
    crossings: list[tuple[str, str]] = []
    n_sub = len(sub_lines)
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            ha = sub_lines[i]
            hb = sub_lines[j]
            p1 = (ha.x1, ha.y1)
            p2 = (ha.x2, ha.y2)
            p3 = (hb.x1, hb.y1)
            p4 = (hb.x2, hb.y2)
            if _segments_cross(p1, p2, p3, p4):
                crossings.append((ha.id, hb.id))
    return crossings


def _ensure_sub_agent_branches_do_not_cross(
    histories: list[GraphHistoryLine],
    node_spacing: float = FIXED_SUB_AGENT_NODE_SPACING,
) -> None:
    """Validate and guarantee that no sub-agent history lines cross each other.
    If an intersection is detected, adjust ray angles so branches diverge cleanly.
    """
    sub_lines = [h for h in histories if not h.is_main]
    if len(sub_lines) <= 1:
        return

    for _ in range(3):
        crossings = check_sub_agent_branches_do_not_cross(histories)
        if not crossings:
            break
        for id_a, id_b in crossings:
            h_target = next(
                (
                    h
                    for h in sub_lines
                    if h.id in (id_a, id_b) and h.parent_history_id != "main"
                ),
                next(h for h in sub_lines if h.id == id_b),
            )
            if h_target.angle_deg >= 90.0:
                new_angle = min(155.0, h_target.angle_deg + 16.0)
            else:
                new_angle = max(25.0, h_target.angle_deg - 16.0)
            _layout_sub_agent_constellation_line(
                h_target,
                anchor_x=h_target.interaction_x,
                anchor_y=h_target.interaction_y,
                angle_deg=new_angle,
                node_spacing=node_spacing,
            )


def _layout_main_horizontal_history(
    history: GraphHistoryLine,
    *,
    center_x: float | None = None,
    main_y: float = 225.0,
    offset_step: float = 48.0,
    node_spacing: float = FIXED_NODE_SPACING_X,
) -> float:
    """Position the horizontal main history line with a fixed horizontal distance between nodes
    and a dynamic horizontal width that grows with the number of nodes.
    """
    history.angle_rad = 0.0
    history.angle_deg = 0.0
    history.interaction_y = main_y
    history.y1 = main_y
    history.y2 = main_y

    n = len(history.nodes)
    if n == 0:
        canvas_width = 520.0
        eff_center_x = (
            center_x if center_x is not None else round(canvas_width / 2.0, 2)
        )
        history.interaction_x = eff_center_x
        history.x1 = round(eff_center_x - 160.0, 2)
        history.x2 = round(eff_center_x + 160.0, 2)
        return canvas_width

    if n == 1:
        canvas_width = max(
            MIN_GRAPH_WIDTH, round(2.0 * GRAPH_HORIZONTAL_PADDING + 120.0, 2)
        )
        eff_center_x = center_x if center_x is not None else 500.0
        x_values = [eff_center_x]
        history.interaction_x = eff_center_x
        history.x1 = round(eff_center_x - LINE_HORIZONTAL_EXTENSION, 2)
        history.x2 = round(eff_center_x + LINE_HORIZONTAL_EXTENSION, 2)
    else:
        desired_span = (n - 1) * node_spacing
        canvas_width = max(
            MIN_GRAPH_WIDTH,
            round(desired_span + 2.0 * GRAPH_HORIZONTAL_PADDING, 2),
        )
        eff_center_x = (
            round(canvas_width / 2.0, 2)
            if center_x is None
            else max(center_x, round(canvas_width / 2.0, 2))
        )
        start_x = eff_center_x - desired_span / 2.0
        x_values = [start_x + i * node_spacing for i in range(n)]
        history.interaction_x = eff_center_x
        history.x1 = round(x_values[0] - LINE_HORIZONTAL_EXTENSION, 2)
        history.x2 = round(x_values[-1] + LINE_HORIZONTAL_EXTENSION, 2)

    for idx, node in enumerate(history.nodes):
        node.base_x = round(x_values[idx], 2)
        node.base_y = round(main_y, 2)

    # On the horizontal main history (angle = 0), counterclockwise offset goes 1 step IN BASSO (+y)
    _apply_skill_offsets_and_branches(
        history,
        ux=1.0,
        uy=0.0,
        ccw_x=0.0,
        ccw_y=1.0,
        offset_step=offset_step,
    )
    return canvas_width


def _layout_sub_agent_constellation_line(
    history: GraphHistoryLine,
    *,
    anchor_x: float,
    anchor_y: float,
    angle_deg: float,
    offset_step: float = 40.0,
    node_spacing: float = FIXED_SUB_AGENT_NODE_SPACING,
) -> None:
    """Position a sub-agent straight line so that it intersects its parent history
    directly at the interaction point (anchor_x, anchor_y), with fixed node spacing
    along the ray that scales dynamically with node count.
    """
    theta = math.radians(angle_deg)
    history.angle_rad = theta
    history.angle_deg = round(angle_deg, 1)
    history.interaction_x = round(anchor_x, 2)
    history.interaction_y = round(anchor_y, 2)

    ux = math.cos(theta)
    uy = -math.sin(theta)
    ccw_x = math.sin(theta)
    ccw_y = math.cos(theta)

    n = len(history.nodes)
    if n == 0:
        forward_len = 160.0
        history.x1 = round(anchor_x, 2)
        history.y1 = round(anchor_y, 2)
        history.x2 = round(anchor_x + forward_len * ux, 2)
        history.y2 = round(anchor_y + forward_len * uy, 2)
        return

    start_t = 48.0
    if n == 1:
        t_values = [start_t + 16.0]
        forward_len = t_values[0] + 36.0
    else:
        t_values = [start_t + i * node_spacing for i in range(n)]
        forward_len = t_values[-1] + 36.0

    history.x1 = round(anchor_x, 2)
    history.y1 = round(anchor_y, 2)
    history.x2 = round(anchor_x + forward_len * ux, 2)
    history.y2 = round(anchor_y + forward_len * uy, 2)

    for idx, node in enumerate(history.nodes):
        t = t_values[idx]
        node.base_x = round(anchor_x + t * ux, 2)
        node.base_y = round(anchor_y + t * uy, 2)

    _apply_skill_offsets_and_branches(
        history,
        ux=ux,
        uy=uy,
        ccw_x=ccw_x,
        ccw_y=ccw_y,
        offset_step=offset_step,
    )


def build_graph_data_from_session(
    session: ChatSession,
    live_message: ChatMessage | None = None,
) -> SessionGraphData:
    """Build the independent constellation graph for a ChatSession:
    - Main history is horizontal (y = 225) with fixed distance between consecutive nodes.
    - Total horizontal width expands dynamically as nodes are added.
    - Skills opened on the main history step downward (in basso, +y) with identical offset per skill_path.
    - Sub-agent histories intersect their parent history at their exact points of interaction.
    """
    main_messages = list(session.messages)
    if live_message is not None:
        main_messages.append(live_message)

    sub_conversations: list[SubAgentConversation] = list(
        session.sub_agent_conversations
    )

    # 1. Build Main History nodes & ensure an interaction node exists for each sub-agent
    main_nodes = _build_nodes_for_messages(
        main_messages,
        history_id="main",
        history_label="Main History",
        system_prompt=session.system_prompt,
        capabilities_path_str=session.settings.capabilities_path,
        name_hds=session.settings.name_hds or "main",
        is_sub_agent=False,
    )
    interaction_map = _ensure_interaction_nodes_on_main(main_nodes, sub_conversations)

    main_line = GraphHistoryLine(
        id="main",
        label="Main History",
        is_main=True,
        line_color="#a1a1aa",
        nodes=main_nodes,
    )
    canvas_width = _layout_main_horizontal_history(main_line, main_y=225.0)

    histories: list[GraphHistoryLine] = [main_line]
    interaction_links: list[GraphInteractionLink] = []

    # 2. Position Sub-Agent Histories intersecting at their exact interaction points (Constellation)
    # Determine each sub-agent's attachment point first, then assign monotonically decreasing angles
    # ordered by anchor_x from left to right (theta_left > theta_right => cot(theta_left) < cot(theta_right))
    # so sub-agent histories fan out and mathematically NEVER cross each other unless they exchange data.
    ray_palette = ["#06b6d4", "#6366f1", "#ec4899", "#14b8a6", "#f97316"]
    history_by_path: dict[str, GraphHistoryLine] = {"main": main_line}

    pending_subs: list[dict[str, Any]] = []
    for idx, sub_conv in enumerate(sub_conversations):
        sub_label = f"{sub_conv.path} #{sub_conv.order}"
        sub_nodes = _build_nodes_for_messages(
            list(sub_conv.messages),
            history_id=sub_conv.id,
            history_label=sub_label,
            capabilities_path_str=session.settings.capabilities_path,
            name_hds=sub_conv.path,
            fallback_raw_history=list(sub_conv.raw_history),
            is_sub_agent=True,
        )
        inter_node = interaction_map.get(sub_conv.id)
        anchor_x = inter_node.x if inter_node is not None else 500.0 + idx * 60.0
        anchor_y = inter_node.y if inter_node is not None else 225.0
        pending_subs.append(
            {
                "idx": idx,
                "sub_conv": sub_conv,
                "sub_label": sub_label,
                "sub_nodes": sub_nodes,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
            }
        )

    # Order root-attached sub-agents from left to right by anchor_x to assign non-crossing fan angles
    sorted_by_x = sorted(pending_subs, key=lambda item: (item["anchor_x"], item["idx"]))
    m_total = len(sorted_by_x)
    angle_by_sub_id: dict[str, float] = {}
    for rank, item in enumerate(sorted_by_x):
        if m_total == 1:
            angle_by_sub_id[item["sub_conv"].id] = 82.0
        else:
            # Strictly decreasing from 108° (leftmost, tilts up-left) to 58° (rightmost, tilts up-right)
            angle_by_sub_id[item["sub_conv"].id] = round(
                108.0 - rank * (50.0 / (m_total - 1)), 1
            )

    for item in pending_subs:
        idx = item["idx"]
        sub_conv = item["sub_conv"]
        sub_label = item["sub_label"]
        sub_nodes = item["sub_nodes"]
        anchor_x = item["anchor_x"]
        anchor_y = item["anchor_y"]
        parent_id = "main"
        angle_deg = angle_by_sub_id[sub_conv.id]

        path_parts = [p for p in sub_conv.path.strip("/").split("/") if p]
        if len(path_parts) > 2:
            parent_prefix = "/".join(path_parts[:-1])
            if parent_prefix in history_by_path:
                parent_line = history_by_path[parent_prefix]
                parent_id = parent_line.id
                pivot = parent_line.nodes[min(1, len(parent_line.nodes) - 1)]
                anchor_x, anchor_y = pivot.x, pivot.y
                angle_deg = (
                    min(142.0, parent_line.angle_deg + 22.0)
                    if parent_line.angle_deg >= 90.0
                    else max(38.0, parent_line.angle_deg - 22.0)
                )

        color = ray_palette[idx % len(ray_palette)]
        sub_line = GraphHistoryLine(
            id=sub_conv.id,
            label=sub_label,
            is_main=False,
            parent_history_id=parent_id,
            line_color=color,
            nodes=sub_nodes,
        )
        _layout_sub_agent_constellation_line(
            sub_line,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            angle_deg=angle_deg,
        )
        histories.append(sub_line)
        history_by_path[sub_conv.path] = sub_line

        first_sub_node = sub_line.nodes[0]
        interaction_links.append(
            GraphInteractionLink(
                from_history_id=parent_id,
                to_history_id=sub_conv.id,
                x1=anchor_x,
                y1=anchor_y,
                x2=first_sub_node.x,
                y2=first_sub_node.y,
                color=color,
            )
        )

    _ensure_sub_agent_branches_do_not_cross(histories)

    all_x_coords: list[float] = [main_line.x1, main_line.x2]
    for h in histories:
        all_x_coords.extend([h.x1, h.x2, h.interaction_x])
        for n_item in h.nodes:
            all_x_coords.extend([n_item.x - 45.0, n_item.x + 45.0])
        for br in h.skill_branches:
            all_x_coords.append(br.label_x + 110.0)
            for pt in br.points:
                all_x_coords.append(pt[0])

    max_x = max(all_x_coords)
    canvas_width = max(canvas_width, round(max_x + 65.0, 2))
    center_x = round((main_line.x1 + main_line.x2) / 2.0, 2)

    all_y_coords: list[float] = [main_line.y1, main_line.y2]
    for h in histories:
        all_y_coords.extend([h.y1, h.y2, h.interaction_y])
        for n_item in h.nodes:
            all_y_coords.extend([n_item.y - 30.0, n_item.y + 30.0])
        for br in h.skill_branches:
            all_y_coords.extend([br.label_y - 20.0, br.label_y + 30.0])
            for pt in br.points:
                all_y_coords.append(pt[1])

    min_y = min(all_y_coords)
    if min_y < 35.0:
        shift_y = round(35.0 - min_y, 2)
        for h in histories:
            h.y1 = round(h.y1 + shift_y, 2)
            h.y2 = round(h.y2 + shift_y, 2)
            h.interaction_y = round(h.interaction_y + shift_y, 2)
            for n_item in h.nodes:
                n_item.y = round(n_item.y + shift_y, 2)
                n_item.base_y = round(n_item.base_y + shift_y, 2)
            for br in h.skill_branches:
                br.label_y = round(br.label_y + shift_y, 2)
                br.points = [(px, round(py + shift_y, 2)) for px, py in br.points]
        for lnk in interaction_links:
            lnk.y1 = round(lnk.y1 + shift_y, 2)
            lnk.y2 = round(lnk.y2 + shift_y, 2)
        all_y_coords = [y + shift_y for y in all_y_coords]

    max_y = max(all_y_coords)
    canvas_height = max(MIN_GRAPH_HEIGHT, round(max_y + 45.0, 2))
    center_y = round((main_line.y1 + main_line.y2) / 2.0, 2)

    return SessionGraphData(
        session_id=session.id,
        session_title=session.title,
        width=canvas_width,
        height=canvas_height,
        center_x=center_x,
        center_y=center_y,
        histories=histories,
        interaction_links=interaction_links,
    )
