"""One execution order for tools, skills, guidance, text and compaction."""

from __future__ import annotations

from pydantic import BaseModel

from exmekh_agent.exmekh_bridge.history.compaction import (
    CompactionActivity,
    RecordedCompaction,
)
from exmekh_agent.exmekh_bridge.models.execution import (
    ContentBlock,
    SkillNode,
    ToolEvent,
)

type Invocation = SkillNode | ToolEvent
type Activity = Invocation | SteeringInsertion | ContentBlock | CompactionActivity


class SteeringInsertion(BaseModel):
    """The execution boundary where ExMekh-Lib confirmed a queued message."""

    message_id: str
    after_tool_call_id: str | None
    tool_name: str | None
    skill_path: str
    timestamp: float | None = None


def opening_time(activity: Invocation | ContentBlock | CompactionActivity) -> float:
    return (
        activity.start_time if isinstance(activity, SkillNode) else activity.timestamp
    )


def call_id(activity: Invocation) -> str | None:
    return (
        activity.opening_tool_call_id
        if isinstance(activity, SkillNode)
        else activity.tool_call_id
    )


class ExecutionTimeline:
    """Place each confirmation once, retaining the identity of repeated skill calls."""

    def __init__(
        self,
        activities: list[Invocation],
        insertions: list[SteeringInsertion],
        *,
        blocks: list[ContentBlock] | None = None,
        compactions: list[RecordedCompaction] | None = None,
        active_compactions: list[CompactionActivity] | None = None,
    ) -> None:
        self._scopes: dict[str, list[Activity]] = {}
        self._paths: dict[str, str] = {"": "/"}
        self._skills: dict[str, SkillNode] = {}
        self._collect_scope("", activities)
        if blocks is not None:
            self._place_blocks(blocks)
            self._place_compactions(compactions or [], active_compactions or [])
            for scope in self._scopes.values():
                scope.sort(key=opening_time)
        seen: set[str] = set()
        for insertion in insertions:
            if insertion.message_id not in seen:
                self._place(insertion, chronological=blocks is not None)
                seen.add(insertion.message_id)

    def root_activity(self) -> list[Activity]:
        return list(self._scopes[""])

    def skill_activity(self, skill: SkillNode) -> list[Activity]:
        return list(self._scopes[skill.id])

    def _collect_scope(self, key: str, activities: list[Invocation]) -> None:
        self._scopes[key] = list(activities)
        for activity in activities:
            if isinstance(activity, SkillNode):
                self._paths[activity.id] = activity.path
                self._skills[activity.id] = activity
                self._collect_scope(activity.id, activity.ordered_activity())

    def _place_blocks(self, blocks: list[ContentBlock]) -> None:
        scopes = {
            block.id: key
            for key, skill in self._skills.items()
            for block in skill.reasoning_blocks
        }
        for block in blocks:
            key = scopes.get(block.id)
            if key is None:
                key = self._scope_for(block.skill_path, block.timestamp)
            self._scopes[key].append(block)

        recorded_ids = {block.id for block in blocks}
        for key, skill in self._skills.items():
            self._scopes[key].extend(
                block
                for block in skill.reasoning_blocks
                if block.id not in recorded_ids
            )

    def _place_compactions(
        self, recorded: list[RecordedCompaction], active: list[CompactionActivity]
    ) -> None:
        for record in recorded:
            key = self._scope_for(record.skill_path, record.timestamp)
            timestamp = record.timestamp
            if timestamp is None:
                timestamp = next(
                    (
                        opening_time(activity)
                        for activity in reversed(self._scopes[key])
                        if isinstance(activity, (SkillNode, ToolEvent))
                        and record.after_tool_call_id is not None
                        and call_id(activity) == record.after_tool_call_id
                    ),
                    float("-inf"),
                )
            self._scopes[key].append(CompactionActivity(record.id, timestamp, record))
        for activity in active:
            key = self._scope_for(activity.skill_path, activity.timestamp)
            self._scopes[key].append(activity)

    def _place(self, insertion: SteeringInsertion, *, chronological: bool) -> None:
        scope = self._scope_for(insertion.skill_path, insertion.timestamp)
        if chronological and insertion.timestamp is not None:
            self._place_at_time(insertion, scope)
            return

        anchors = [
            (key, index, activity)
            for key, activities in self._scopes.items()
            for index, activity in enumerate(activities)
            if isinstance(activity, (SkillNode, ToolEvent))
            and insertion.after_tool_call_id is not None
            and call_id(activity) == insertion.after_tool_call_id
            and (not chronological or self._paths[key] == insertion.skill_path)
        ]
        if anchors:
            scoped = [
                anchor
                for anchor in anchors
                if self._paths[anchor[0]] == insertion.skill_path
            ]
            candidates = scoped or anchors
            if insertion.timestamp is not None:
                preceding = [
                    anchor
                    for anchor in candidates
                    if opening_time(anchor[2]) <= insertion.timestamp
                ]
                candidates = preceding or candidates
            key, index, _ = max(candidates, key=lambda anchor: opening_time(anchor[2]))
            activities = self._scopes[key]
            index += 1
            # Multiple confirmations at one boundary retain their FIFO order.
            while index < len(activities) and isinstance(
                activities[index], SteeringInsertion
            ):
                index += 1
            activities.insert(index, insertion)
        else:
            self._place_at_time(insertion, scope)

    def _place_at_time(self, insertion: SteeringInsertion, scope: str) -> None:
        activities = self._scopes[scope]
        inserted_at = (
            insertion.timestamp if insertion.timestamp is not None else float("-inf")
        )
        for position, activity in enumerate(activities):
            if (
                not isinstance(activity, SteeringInsertion)
                and opening_time(activity) > inserted_at
            ):
                activities.insert(position, insertion)
                return
        activities.append(insertion)

    def _scope_for(self, path: str, timestamp: float | None) -> str:
        candidates = [
            key for key, scope_path in self._paths.items() if scope_path == path
        ]
        if not candidates:
            return ""
        if timestamp is None or path == "/":
            return candidates[0]
        preceding = [
            key for key in candidates if self._skills[key].start_time <= timestamp
        ]
        return (
            max(preceding, key=lambda key: self._skills[key].start_time)
            if preceding
            else candidates[0]
        )
