from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from exmekh_lib.tokenizers.qwen import QwenTokenizer

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.image_input import content_text
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    SkillNode,
    SubAgentConversation,
)
from exmekh_agent.exmekh_bridge.system_prompt import load_real_hds_system_prompt

logger = logging.getLogger(__name__)

_tokenizer_instance: QwenTokenizer | None = None


def get_qwen_tokenizer() -> QwenTokenizer:
    """Return the process-wide shared QwenTokenizer instance."""
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = QwenTokenizer()
    return _tokenizer_instance


@lru_cache(maxsize=4096)
def count_text_tokens(text: str) -> int:
    """Count tokens in a string using the QwenTokenizer exposed by the exmekh_lib package."""
    if not text:
        return 0
    try:
        return get_qwen_tokenizer().count_tokens(text)
    except Exception as error:  # noqa: BLE001 - safe fallback if tokenizer cache is offline
        logger.debug("Fallback token estimation due to tokenizer error: %s", error)
        return max(1, len(text) // 4)


def format_compact_tokens(count: int) -> str:
    """Format a token count using compact 'k' notation (e.g. 1.4k, 22.0k, 32.8k)."""
    safe_count = max(0, int(count))
    if safe_count == 0:
        return "0"
    if safe_count < 100:
        return str(safe_count)
    return f"{safe_count / 1000.0:.1f}k"


def format_context_percentage(used_tokens: int, context_length: int) -> str:
    """Format the percentage of context consumed (e.g. '4.2%')."""
    if context_length <= 0:
        return "0.0%"
    pct = max(0.0, (max(0, int(used_tokens)) / float(context_length)) * 100.0)
    return f"{pct:.1f}%"


def _serialize_tool_calls_for_counting(tool_calls: Any) -> str:
    """Serialize tool_calls field into text for token counting."""
    if tool_calls is None or tool_calls == "null" or tool_calls == "":
        return ""
    if isinstance(tool_calls, str):
        stripped = tool_calls.strip()
        return "" if stripped in ("", "null") else stripped
    return json.dumps(tool_calls, ensure_ascii=False)


def count_message_dict_tokens(message: dict[str, Any]) -> int:
    """Count tokens consumed by a single message dictionary."""
    content = content_text(message.get("content"))
    tool_calls_str = _serialize_tool_calls_for_counting(message.get("tool_calls"))
    reasoning_str = str(message.get("reasoning_content") or "")
    name_str = str(message.get("name") or "")

    parts = [p for p in (name_str, content, tool_calls_str, reasoning_str) if p]
    if not parts:
        return 0
    return sum(count_text_tokens(part) for part in parts)


def _is_skill_tree_entry_open(
    skill_entry: dict[str, Any],
    skill_nodes_by_name: dict[str, SkillNode] | None = None,
) -> bool:
    """Determine whether a skill execution entry in skill_history_tree is still open."""
    status = str(skill_entry.get("status") or "").lower()
    if status in ("completed", "error"):
        return False
    if status == "active":
        return True
    if (
        skill_entry.get("output") is not None
        or skill_entry.get("end_index") is not None
    ):
        return False
    skill_name = str(skill_entry.get("skill") or "")
    if skill_nodes_by_name and skill_name in skill_nodes_by_name:
        return skill_nodes_by_name[skill_name].status == "active"
    return True


def _collect_skill_nodes_by_name(nodes: list[SkillNode]) -> dict[str, SkillNode]:
    """Build a lookup map of SkillNode instances by name across all depths."""
    result: dict[str, SkillNode] = {}
    for node in nodes:
        result[node.name] = node
        result.update(_collect_skill_nodes_by_name(node.sub_skills))
    return result


def _build_messages_from_skill_node(node: SkillNode) -> list[dict[str, Any]]:
    """Build visible context messages for a SkillNode according to its open/closed state."""
    call_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{node.name}",
                "type": "function",
                "function": {
                    "name": node.name,
                    "arguments": {"path": node.path, "depth": node.depth},
                },
            }
        ],
    }

    if node.status != "active":
        return [
            call_msg,
            {
                "role": "tool",
                "name": node.name,
                "content": f'<DeepSkill name="{node.name}" path="{node.path}" status="{node.status}">Completed {node.name}</DeepSkill>',
            },
        ]

    inner_messages: list[dict[str, Any]] = [
        call_msg,
        {
            "role": "tool",
            "name": node.name,
            "content": f'<DeepSkillInstructions name="{node.name}" path="{node.path}" />',
        },
    ]
    for block in node.reasoning_blocks:
        if block.content:
            inner_messages.append({"role": "assistant", "content": block.content})
    for ev in node.tool_events:
        inner_messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": ev.tool_call_id or f"call_{ev.tool_name}",
                        "type": "function",
                        "function": {"name": ev.tool_name, "arguments": {}},
                    }
                ],
            }
        )
        if ev.status == "completed" or ev.event_type == "tool_end":
            inner_messages.append(
                {
                    "role": "tool",
                    "name": ev.tool_name,
                    "content": f'<Tool name="{ev.tool_name}">Output of {ev.tool_name}</Tool>',
                }
            )
    for sub in node.sub_skills:
        inner_messages.extend(_build_messages_from_skill_node(sub))
    return inner_messages


def find_skill_opening_index(
    messages: list[dict[str, Any]],
    skill_name: str,
    opening_call_id: str | None,
) -> int:
    """Find the index of the AssistantMessage in `messages` that called the given skill."""
    for idx, msg in enumerate(messages):
        if str(msg.get("role", "")).lower() != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if isinstance(tcs, str):
            try:
                tcs = json.loads(tcs)
            except Exception:
                tcs = []
        if isinstance(tcs, list):
            for tc in tcs:
                if isinstance(tc, dict):
                    cid = tc.get("id")
                    fn = tc.get("function")
                    fn_name = (
                        fn.get("name") if isinstance(fn, dict) else None
                    ) or tc.get("name")
                    matches_id = opening_call_id is not None and cid == opening_call_id
                    matches_name = (
                        opening_call_id is None and skill_name and fn_name == skill_name
                    )
                    if matches_id or matches_name:
                        return idx
    return -1


def _slice_already_contains_inner(
    messages: list[dict[str, Any]],
    open_idx: int,
    inner_history: list[dict[str, Any]],
) -> int:
    """If `messages` after `open_idx` already contains `inner_history`, return the length matched."""
    if not inner_history or open_idx < 0 or open_idx + 1 >= len(messages):
        return 0
    match_len = 0
    for m_item, in_item in zip(messages[open_idx + 1 :], inner_history, strict=False):
        if m_item.get("role") == in_item.get("role") and m_item.get(
            "content"
        ) == in_item.get("content"):
            match_len += 1
        else:
            break
    return match_len


def _apply_skill_compaction_to_scope(
    scope_messages: list[dict[str, Any]],
    skill_tree: list[dict[str, Any]],
    skill_nodes_by_name: dict[str, SkillNode] | None = None,
) -> list[dict[str, Any]]:
    """Apply skill compaction rules to a sequence of messages at a given scope."""
    if not skill_tree:
        return list(scope_messages)

    result = list(scope_messages)
    for sk in skill_tree:
        skill_name = str(sk.get("skill") or "")
        call_id = sk.get("opening_tool_call_id")
        is_open = _is_skill_tree_entry_open(sk, skill_nodes_by_name)
        raw_skill_hist = list(sk.get("history") or [])
        inner_raw = (
            raw_skill_hist[1:]
            if raw_skill_hist
            and str(raw_skill_hist[0].get("role", "")).lower() == "assistant"
            else raw_skill_hist
        )

        open_idx = find_skill_opening_index(result, skill_name, call_id)
        matched_len = _slice_already_contains_inner(result, open_idx, inner_raw)

        if not is_open:
            if matched_len > 0 and open_idx >= 0:
                compacted_tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "name": skill_name,
                    "content": str(sk.get("output") or f"Completed {skill_name}"),
                }
                if call_id:
                    compacted_tool_msg["tool_call_id"] = call_id
                result = (
                    result[: open_idx + 1]
                    + [compacted_tool_msg]
                    + result[open_idx + 1 + matched_len :]
                )
        else:
            expanded_inner = _apply_skill_compaction_to_scope(
                inner_raw,
                list(sk.get("children") or []),
                skill_nodes_by_name=skill_nodes_by_name,
            )
            if open_idx >= 0:
                result = (
                    result[: open_idx + 1]
                    + expanded_inner
                    + result[open_idx + 1 + matched_len :]
                )
            else:
                result.extend(expanded_inner)

    return result


def _build_turn_messages_for_assistant(
    msg: ChatMessage,
) -> list[dict[str, Any]]:
    """Build the compacted/expanded message sequence for one assistant ChatMessage turn."""
    skill_nodes_map = _collect_skill_nodes_by_name(msg.skills)

    if msg.raw_history:
        turn_slice = msg.turn_raw_history()
        compacted_slice = _apply_skill_compaction_to_scope(
            turn_slice,
            msg.skill_history_tree,
            skill_nodes_by_name=skill_nodes_map,
        )
        if msg.content:
            has_final_content = any(
                str(m.get("role", "")).lower() == "assistant"
                and str(m.get("content") or "") == msg.content
                for m in compacted_slice
            )
            if not has_final_content:
                compacted_slice.append({"role": "assistant", "content": msg.content})
        return compacted_slice

    synthesized: list[dict[str, Any]] = []
    for block in msg.reasoning_blocks:
        if (block.depth == 0 or block.skill_path == "/") and block.content:
            synthesized.append({"role": "assistant", "content": block.content})

    for ev in msg.tool_events:
        if ev.depth == 0 or ev.skill_path == "/":
            synthesized.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": ev.tool_call_id or f"call_{ev.tool_name}",
                            "type": "function",
                            "function": {"name": ev.tool_name, "arguments": {}},
                        }
                    ],
                }
            )
            if ev.status == "completed" or ev.event_type == "tool_end":
                synthesized.append(
                    {
                        "role": "tool",
                        "name": ev.tool_name,
                        "content": f'<Tool name="{ev.tool_name}">Output of {ev.tool_name}</Tool>',
                    }
                )

    if msg.skill_history_tree:
        synthesized = _apply_skill_compaction_to_scope(
            synthesized,
            msg.skill_history_tree,
            skill_nodes_by_name=skill_nodes_map,
        )
    else:
        for skill_node in msg.skills:
            synthesized.extend(_build_messages_from_skill_node(skill_node))

    if msg.content:
        synthesized.append({"role": "assistant", "content": msg.content})

    return synthesized


def _resolve_system_prompt_for_session(
    session: ChatSession,
    messages: list[ChatMessage],
) -> str:
    """Resolve the system prompt seen by the agent for the given session."""
    for msg in messages:
        if msg.role == "system" and msg.content:
            return msg.content
        for rm in msg.raw_history:
            if str(rm.get("role", "")).lower() == "system" and rm.get("content"):
                return str(rm["content"])

    if session.system_prompt:
        return session.system_prompt

    if session.agent_mode == "deep_agent":
        hds_prompt = load_real_hds_system_prompt(
            capabilities_path_str=session.settings.capabilities_path or "",
            name_hds=session.settings.name_hds or "main",
            is_sub_agent=False,
        )
        if hds_prompt:
            return hds_prompt

    return "You are a friendly, accurate, and professional AI assistant."


def build_compacted_context_messages(
    session: ChatSession,
    collector: TurnStreamCollector | None = None,
    sub_agent: SubAgentConversation | None = None,
) -> list[dict[str, Any]]:
    """Build the exact list of messages that the agent sees up to the last generated message."""
    if sub_agent is not None:
        chat_messages = list(sub_agent.messages)
    else:
        chat_messages = list(session.messages)
        if collector is not None:
            chat_messages.append(collector.snapshot_message())

    if not chat_messages:
        return []

    sys_prompt = _resolve_system_prompt_for_session(session, chat_messages)
    context_messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]

    for msg in chat_messages:
        if msg.role == "user":
            context_messages.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            context_messages.extend(_build_turn_messages_for_assistant(msg))

    return context_messages


@dataclass(frozen=True)
class ContextUsageSummary:
    model_name: str
    used_tokens: int
    context_length: int
    estimated: bool

    @property
    def percentage(self) -> float:
        if self.context_length <= 0:
            return 0.0
        return max(0.0, self.used_tokens / float(self.context_length) * 100.0)


def compute_session_context_usage(
    session: ChatSession,
    collector: TurnStreamCollector | None = None,
    sub_agent: SubAgentConversation | None = None,
    model_override: str | None = None,
) -> ContextUsageSummary:
    """Resolve the visible context and return all information needed to display it."""
    model_name = model_override or config.default_model or session.model
    context_length = config.get_context_length(model_name)
    tracked_messages = sub_agent.messages if sub_agent is not None else session.messages
    context = (
        collector.get_provider_context()
        if collector is not None and sub_agent is None
        else None
    )
    if context is None and tracked_messages:
        context = tracked_messages[-1].context_usage
    if context is not None:
        used_tokens = context.total_tokens(count_text_tokens)
    else:
        visible_messages = build_compacted_context_messages(
            session=session,
            collector=collector,
            sub_agent=sub_agent,
        )
        used_tokens = sum(count_message_dict_tokens(m) for m in visible_messages)
    return ContextUsageSummary(
        model_name=model_name,
        used_tokens=used_tokens,
        context_length=context_length,
        estimated=context is None or context.is_estimated,
    )


def _extract_inner_skill_messages(
    skill_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return inner skill history messages, stripping the leading invocation prompt if present."""
    raw_skill_hist = [
        m for m in (skill_data.get("history") or []) if isinstance(m, dict)
    ]
    return (
        raw_skill_hist[1:]
        if raw_skill_hist
        and str(raw_skill_hist[0].get("role", "")).lower() == "assistant"
        else raw_skill_hist
    )


def format_skill_closure_tokens(tokens_before: int, tokens_after: int) -> str:
    """Format before/after token counts for a closed skill (e.g. '4.8k → 1.2k token')."""
    return f"{format_compact_tokens(tokens_before)} → {format_compact_tokens(tokens_after)} token"


def compute_raw_skill_compaction_tokens(
    skill_data: dict[str, Any],
    prefix_messages: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """Compute (tokens_before_close, tokens_after_close) for a raw skill_history_tree node."""
    existing_before = skill_data.get("tokens_before_close")
    existing_after = skill_data.get("tokens_after_close")
    if (
        isinstance(existing_before, int)
        and isinstance(existing_after, int)
        and existing_before > 0
    ):
        return existing_before, existing_after

    skill_name = str(skill_data.get("skill") or "skill")
    call_id = skill_data.get("opening_tool_call_id")

    if prefix_messages:
        prefix_tokens = sum(count_message_dict_tokens(m) for m in prefix_messages)
    else:
        prefix_tokens = count_message_dict_tokens(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": skill_name, "arguments": "{}"}}],
            }
        )

    inner_raw = _extract_inner_skill_messages(skill_data)
    children = (
        skill_data.get("children")
        if isinstance(skill_data.get("children"), list)
        else []
    )
    compacted_inner = _apply_skill_compaction_to_scope(inner_raw, children)
    if not compacted_inner and isinstance(skill_data.get("history"), list):
        compacted_inner = [m for m in skill_data["history"] if isinstance(m, dict)]
    inner_tokens = sum(count_message_dict_tokens(m) for m in compacted_inner)

    output_str = str(
        skill_data.get("output")
        if skill_data.get("output") is not None
        else "completed"
    )
    compacted_tool_msg: dict[str, Any] = {
        "role": "tool",
        "content": output_str,
        "name": skill_name,
    }
    if call_id:
        compacted_tool_msg["tool_call_id"] = str(call_id)
    output_tokens = count_message_dict_tokens(compacted_tool_msg)

    before_tokens = prefix_tokens + max(inner_tokens, output_tokens)
    after_tokens = prefix_tokens + output_tokens
    skill_data["tokens_before_close"] = before_tokens
    skill_data["tokens_after_close"] = after_tokens
    return before_tokens, after_tokens


def compute_skill_node_closure_tokens(
    node: SkillNode,
    base_tokens: int = 0,
) -> tuple[int, int]:
    """Compute (tokens_before_close, tokens_after_close) for a closed SkillNode."""
    if node.has_closure_tokens:
        return node.tokens_before_close, node.tokens_after_close

    active_copy = node.model_copy(deep=True)
    active_copy.status = "active"
    open_msgs = _build_messages_from_skill_node(active_copy)

    closed_copy = node.model_copy(deep=True)
    closed_copy.status = "completed"
    closed_msgs = _build_messages_from_skill_node(closed_copy)

    open_tokens = sum(count_message_dict_tokens(m) for m in open_msgs)
    closed_tokens = sum(count_message_dict_tokens(m) for m in closed_msgs)

    before_tokens = base_tokens + max(open_tokens, closed_tokens)
    after_tokens = base_tokens + closed_tokens

    node.record_closure_tokens(before_tokens, after_tokens)

    sub_base = base_tokens + count_message_dict_tokens(open_msgs[0])
    for sub in node.sub_skills:
        if sub.status != "active":
            compute_skill_node_closure_tokens(sub, base_tokens=sub_base)

    return before_tokens, after_tokens


def _annotate_skill_tree_recursive(
    scope_messages: list[dict[str, Any]],
    skill_tree: list[dict[str, Any]],
    prior_messages: list[dict[str, Any]],
    matched_by_name: dict[str, tuple[int, int]],
) -> None:
    """Recursively annotate skill_history_tree dicts with before/after closure tokens."""
    running_prefix = list(prior_messages)
    last_idx = 0
    for sk in skill_tree:
        skill_name = str(sk.get("skill") or "")
        call_id = sk.get("opening_tool_call_id")
        open_idx = find_skill_opening_index(scope_messages, skill_name, call_id)
        if open_idx >= last_idx:
            running_prefix.extend(scope_messages[last_idx : open_idx + 1])
            last_idx = open_idx + 1
        elif not running_prefix:
            running_prefix.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": skill_name or "skill", "arguments": "{}"}}
                    ],
                }
            )

        children = sk.get("children") if isinstance(sk.get("children"), list) else []
        if children:
            inner_scope = _extract_inner_skill_messages(sk)
            _annotate_skill_tree_recursive(
                inner_scope, children, running_prefix, matched_by_name
            )

        if not _is_skill_tree_entry_open(sk) and not sk.get("tokens_pending"):
            before_tok, after_tok = compute_raw_skill_compaction_tokens(
                sk, prefix_messages=running_prefix
            )
            if skill_name:
                matched_by_name[skill_name] = (before_tok, after_tok)


def _apply_matched_tokens_to_skill_nodes(
    skills: list[SkillNode],
    matched_by_name: dict[str, tuple[int, int]],
    base_tokens: int,
) -> None:
    """Apply matched tree token counts or fallback node calculation to closed SkillNodes."""
    running_base = base_tokens
    for node in skills:
        tail_name = node.path.rstrip("/").split("/")[-1] if node.path else ""
        matched = matched_by_name.get(node.name) or matched_by_name.get(tail_name)
        if node.status != "active":
            if node.tokens_pending:
                continue
            if not node.has_closure_tokens:
                if matched is not None:
                    node.record_closure_tokens(*matched)
                else:
                    compute_skill_node_closure_tokens(node, base_tokens=running_base)
            running_base = int(node.tokens_after_close or running_base)
        if node.sub_skills:
            _apply_matched_tokens_to_skill_nodes(
                node.sub_skills, matched_by_name, running_base
            )


def _populate_single_turn(
    skills: list[SkillNode],
    turn_history: list[dict[str, Any]],
    skill_history_tree: list[dict[str, Any]],
    running_messages: list[dict[str, Any]],
) -> None:
    """Populate closure tokens for one assistant turn's skill_history_tree and SkillNodes."""
    matched_by_name: dict[str, tuple[int, int]] = {}
    _annotate_skill_tree_recursive(
        turn_history,
        skill_history_tree,
        running_messages,
        matched_by_name,
    )
    base_tok = sum(count_message_dict_tokens(m) for m in running_messages)
    _apply_matched_tokens_to_skill_nodes(skills, matched_by_name, base_tok)


def populate_skill_closure_tokens(
    session: ChatSession,
    collector: TurnStreamCollector | None = None,
    sub_agent: SubAgentConversation | None = None,
) -> None:
    """Populate tokens_before_close and tokens_after_close across all closed skills in session/collector."""
    target_messages = (
        list(sub_agent.messages) if sub_agent is not None else list(session.messages)
    )
    if not target_messages and collector is None:
        return

    sys_prompt = _resolve_system_prompt_for_session(session, target_messages)
    running_messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]

    for msg in target_messages:
        if msg.role == "assistant":
            _populate_single_turn(
                msg.skills,
                msg.turn_raw_history(),
                msg.skill_history_tree,
                running_messages,
            )
            running_messages.extend(_build_turn_messages_for_assistant(msg))
        else:
            running_messages.append({"role": msg.role, "content": msg.content})

    if collector is not None and sub_agent is None:
        _populate_single_turn(
            collector.skills,
            collector.turn_raw_history(),
            collector.skill_history_tree,
            running_messages,
        )
