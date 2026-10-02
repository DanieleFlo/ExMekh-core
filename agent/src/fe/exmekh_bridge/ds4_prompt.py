"""Render the ChatML and DSML prompt used by DS4 for Qwen models."""

from __future__ import annotations

import json
from html import escape
from typing import Any

from fe.exmekh_bridge.image_input import content_text

_TOOLS_PREFIX = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
_TOOLS_SUFFIX = (
    "\n</tools>\n\n"
    "If you choose to call a function, ONLY reply in the following DSML "
    "format with NO suffix:\n\n"
    '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="$TOOL_NAME">\n'
    '<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">'
    "$PARAMETER_VALUE</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST use the DSML format above; native <tool_call> "
    "syntax is not executable in this runtime\n"
    "- Required parameters MUST be specified with their exact schema names\n"
    '- String parameters use raw text and string="true"; numbers, booleans, '
    'arrays, and objects use JSON and string="false"\n'
    "- When an instruction requires a listed function, call it instead of "
    "answering with prose\n"
    "- If no function call is needed, answer normally and do not mention "
    "function calls\n</IMPORTANT>"
)
_SCOPE_GUIDANCE = (
    ". These lists are permissions, not an unconditional request "
    "to call a capability. Use DSML to invoke a permitted capability "
    "when needed for unfinished work. Tool and skill results already "
    "in the conversation represent completed work: do not repeat "
    "successful calls just because the original request mentions them.\n"
)
_RETURN_GUIDANCE = (
    "The invoked skill has already finished and EXITED. "
    "You are now back in its caller, outside that skill. "
    "Do not execute that completed skill's exit instructions "
    "again: its exit has already produced the returned result. "
    "Its internal conversation, tool calls and nested skills "
    "have been removed from this context and replaced by this "
    "result. Their absence is not evidence of unfinished work. "
    "This is the result of executing the skill, not an instruction "
    "to enter it. Resume the caller after that skill invocation; "
    "do not reopen it to replay steps hidden by compaction. "
    "If the result reports a failure or missing work, handle that "
    "explicitly; otherwise continue with the next caller step "
    "or answer the user when the requested work is complete.\n"
)


def apply_ds4_scope(messages: list[dict[str, Any]], agentic: dict[str, Any]) -> None:
    """Mirror DS4's scope instruction on the newest user or tool message."""
    scope = (
        "<agent_runtime>\nAgent runtime scope for this turn. Allowed ordinary tools: "
        + (", ".join(agentic["allowed_tools"]) or "(none)")
        + ". Allowed skills: "
        + (", ".join(agentic["allowed_skills"]) or "(none)")
        + _SCOPE_GUIDANCE
    )
    if agentic.get("operation") == "return":
        scope += _RETURN_GUIDANCE
    scope += "</agent_runtime>"
    for message in reversed(messages):
        if message["role"] in ("user", "tool"):
            content = content_text(message.get("content"))
            message["content"] = content + ("\n\n" if content else "") + scope
            return
    messages.append({"role": "system", "content": scope})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if isinstance(calls, str):
        calls = json.loads(calls)
    return calls


def _render_tool_calls(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return ""
    parts = ["\n\n<｜DSML｜tool_calls>\n"]
    for call in calls:
        function = call["function"]
        parts.append(
            f'<｜DSML｜invoke name="{escape(function["name"], quote=True)}">\n'
        )
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        for name, value in arguments.items():
            is_string = isinstance(value, str)
            text = value if is_string else _json(value)
            replacement = "&lt;" if is_string else "\\u003c"
            text = text.replace(
                "</｜DSML｜parameter>", replacement + "/｜DSML｜parameter>"
            )
            parts.append(
                f'<｜DSML｜parameter name="{escape(name, quote=True)}" '
                f'string="{str(is_string).lower()}">{text}</｜DSML｜parameter>\n'
            )
        parts.append("</｜DSML｜invoke>\n")
    parts.append("</｜DSML｜tool_calls>")
    return "".join(parts)


def render_ds4_qwen_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    thinking: bool = True,
) -> str:
    """Include tool schemas, role boundaries, tool responses and generation prefix."""
    system_parts = []
    if tools:
        schemas = "\n".join(_json(tool.get("function", tool)) for tool in tools)
        system_parts.append(_TOOLS_PREFIX + schemas + _TOOLS_SUFFIX)
    system_parts.extend(
        content_text(message.get("content")).strip()
        for message in messages
        if message["role"] in ("system", "developer")
    )
    parts = []
    if system_parts:
        parts.append(
            "<|im_start|>system\n" + "\n\n".join(system_parts) + "<|im_end|>\n"
        )
    last_user = max(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        default=-1,
    )
    tool_context = bool(tools) or any(
        message["role"] == "tool" or message.get("tool_calls") for message in messages
    )
    tool_group = False
    pending_assistant = False
    for index, message in enumerate(messages):
        role = message["role"]
        content = content_text(message.get("content")).strip()
        if role != "tool" and tool_group:
            parts.append("<|im_end|>\n")
            tool_group = False
        if role in ("system", "developer"):
            continue
        if role == "tool":
            if not tool_group:
                parts.append("<|im_start|>user")
                tool_group = True
            parts.append("\n<tool_response>\n" + content + "\n</tool_response>")
            pending_assistant = True
        else:
            parts.append(f"<|im_start|>{role}\n")
            if role == "assistant":
                if (
                    thinking
                    and (tool_context or index > last_user)
                    and not content.startswith(("<think>", "</think>"))
                ):
                    reasoning = str(message.get("reasoning_content") or "").strip()
                    parts.append("<think>\n" + reasoning + "\n</think>\n\n")
                parts.append(content + _render_tool_calls(_tool_calls(message)))
                pending_assistant = False
            else:
                parts.append(content)
                pending_assistant = True
            parts.append("<|im_end|>\n")
    if tool_group:
        parts.append("<|im_end|>\n")
    if pending_assistant:
        parts.append("<|im_start|>assistant\n")
        parts.append("<think>\n" if thinking else "<think>\n\n</think>\n\n")
    return "".join(parts)
