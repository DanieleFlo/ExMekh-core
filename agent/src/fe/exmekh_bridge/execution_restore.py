"""Restore execution state through public ExMekh-Lib history and cache interfaces."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.cache.user_message_codec import serialize_user_message
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.tool_calls.execution_history import (
    ExecutionHistory,
    ExecutionHistoryEntry,
)

from fe.exmekh_bridge.image_input import deserialize_user_content
from fe.exmekh_bridge.models import ChatMessage
from fe.exmekh_bridge.recorded_output import RecordedToolOutput
from fe.exmekh_bridge.skill_history import (
    inner_skill_history_messages,
    take_skill_for_call,
)
from fe.exmekh_bridge.steering_history import direct_steering_output_ids
from fe.exmekh_bridge.tool_calls import parse_tool_calls, tool_result_for_call


@dataclass(frozen=True, slots=True)
class RecordedInvocation:
    call: ToolCall
    result: dict[str, Any] | None
    skill: dict[str, Any] | None

    @property
    def output(self) -> RecordedToolOutput | None:
        text = self.skill.get("output") if self.skill is not None else None
        if not text and self.result is not None:
            text = self.result.get("content")
        return RecordedToolOutput.parse(text)


def scope_events(
    history: list[dict[str, Any]],
    children: list[dict[str, Any]],
) -> Iterator[RecordedInvocation | dict[str, Any]]:
    """Pair each invocation once, with identical matching for all output types."""
    children_queue = list(children)
    for index, item in enumerate(history):
        if item.get("role") == "user":
            yield item
        if item.get("role") != "assistant":
            continue
        results = []
        for following_index in range(index + 1, len(history)):
            following = history[following_index]
            if following.get("role") != "tool":
                break
            results.append(following)
        for position, call in enumerate(parse_tool_calls(item.get("tool_calls")) or []):
            yield RecordedInvocation(
                call,
                tool_result_for_call(results, call.get_id(), position),
                take_skill_for_call(children_queue, call),
            )


class ExecutionHistoryRestorer:
    """Own chronological restoration; conversion never accesses ExMekh-Lib internals."""

    def __init__(
        self, history: ExecutionHistory, cache: NodeCache | None = None
    ) -> None:
        self._history = history
        self._cache = cache

    def restore(self, messages: Sequence[ChatMessage]) -> None:
        previous_raw: list[dict[str, Any]] = []
        for message in messages:
            if message.role != "assistant":
                continue
            if message.raw_history:
                turn = message.turn_raw_history(previous_history=previous_raw)
                previous_raw = message.raw_history
                self._restore_scope(turn, message.skill_history_tree, "/")
            else:
                self._restore_legacy_events(message)

    def _restore_scope(
        self,
        history: list[dict[str, Any]],
        children: list[dict[str, Any]],
        path: str,
        output: RecordedToolOutput | None = None,
    ) -> None:
        events = list(scope_events(history, children))
        child_outputs = [
            event.output
            for event in events
            if isinstance(event, RecordedInvocation) and event.skill is not None
        ]
        steering_ids = iter(direct_steering_output_ids(output, child_outputs))
        for event in events:
            if isinstance(event, RecordedInvocation):
                self._restore_invocation(event, path)
            elif path != "/":
                output_id = next(steering_ids, None)
                if output_id is not None:
                    self._restore_steering(event, output_id, path)

    def _restore_invocation(self, invocation: RecordedInvocation, path: str) -> None:
        call = invocation.call
        entry = self._history.record(call.get_name(), call.get_id())
        output = invocation.output
        if invocation.skill is not None:
            child_path = f"{path.rstrip('/')}/{call.get_name()}"
            self._restore_scope(
                inner_skill_history_messages(invocation.skill.get("history") or []),
                invocation.skill.get("children") or [],
                child_path,
                output,
            )
        if output is not None:
            self._restore_output(invocation, output, entry, path)

    def _restore_output(
        self,
        invocation: RecordedInvocation,
        output: RecordedToolOutput,
        entry: ExecutionHistoryEntry,
        path: str,
    ) -> None:
        if self._cache is None:
            return
        call = invocation.call
        is_reference = invocation.skill is None and (
            output.source == "UserMessage"
            or (bool(output.name) and output.name != call.get_name())
        )
        if is_reference and self._cache.has(output.id):
            entry.attach_output(output.id, self._cache.session_dir)
            return
        # A transcript of a read cannot recreate missing attachment bytes.
        if output.source == "UserMessage":
            return
        source = (
            "DeepSkill"
            if invocation.skill is not None or output.source == "DeepSkill"
            else "Tool"
        )
        self._cache.store(
            ToolCallRecord(
                name=output.name if is_reference else call.get_name(),
                id=output.id,
                tool_call_id=None if is_reference else call.get_id(),
                args={} if is_reference else call.get_arguments(),
                result={"text": output.body} if source == "DeepSkill" else output.body,
                skill_path=path,
                source=source,
                used_tools=output.used_tools,
            )
        )
        entry.attach_output(output.id, self._cache.session_dir)

    def _restore_steering(
        self, item: dict[str, Any], output_id: str, path: str
    ) -> None:
        entry = self._history.record(RUNTIME.steering_output_name, output_id)
        if self._cache is None:
            return
        message = UserMessage(deserialize_user_content(item.get("content")))
        self._cache.store(
            ToolCallRecord(
                name=RUNTIME.steering_output_name,
                id=output_id,
                args={},
                result=message.message_dump()["content"],
                skill_path=path,
                source="UserMessage",
                user_message=serialize_user_message(message),
            )
        )
        entry.attach_output(output_id, self._cache.session_dir)

    def _restore_legacy_events(self, message: ChatMessage) -> None:
        seen_call_ids: set[str] = set()
        for index, event in enumerate(message.tool_events):
            call_id = event.tool_call_id or f"call_{message.id[:8]}_{index}"
            if call_id not in seen_call_ids:
                self._history.record(event.tool_name, call_id)
                seen_call_ids.add(call_id)
