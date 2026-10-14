"""One execution order for tools, skill invocations and accepted guidance."""

from __future__ import annotations

from pydantic import BaseModel

from exmekh_agent.exmekh_bridge.models.execution import SkillNode, ToolEvent

type Invocation = SkillNode | ToolEvent
type Activity = Invocation | SteeringInsertion


class SteeringInsertion(BaseModel):
    """The execution boundary where ExMekh-Lib confirmed a queued message."""

    message_id: str
    after_tool_call_id: str | None
    tool_name: str | None
    skill_path: str
    timestamp: float | None = None


def opening_time(activity: Invocation) -> float:
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
        self, activities: list[Invocation], insertions: list[SteeringInsertion]
    ) -> None:
        self._scopes: dict[str, list[Activity]] = {}
        self._paths: dict[str, str] = {"": "/"}
        self._skills: dict[str, SkillNode] = {}
        self._collect_scope("", activities)
        seen: set[str] = set()
        for insertion in insertions:
            if insertion.message_id not in seen:
                self._place(insertion)
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

    def _place(self, insertion: SteeringInsertion) -> None:
        anchors = [
            (key, index, activity)
            for key, activities in self._scopes.items()
            for index, activity in enumerate(activities)
            if not isinstance(activity, SteeringInsertion)
            and insertion.after_tool_call_id is not None
            and call_id(activity) == insertion.after_tool_call_id
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
        else:
            key = self._scope_for(insertion)
            activities = self._scopes[key]
            inserted_at = (
                insertion.timestamp
                if insertion.timestamp is not None
                else float("-inf")
            )
            index = len(activities)
            for position, activity in enumerate(activities):
                if (
                    not isinstance(activity, SteeringInsertion)
                    and opening_time(activity) > inserted_at
                ):
                    index = position
                    break
        activities.insert(index, insertion)

    def _scope_for(self, insertion: SteeringInsertion) -> str:
        candidates = [
            key for key, path in self._paths.items() if path == insertion.skill_path
        ]
        if not candidates:
            return ""
        if insertion.timestamp is None or insertion.skill_path == "/":
            return candidates[0]
        preceding = [
            key
            for key in candidates
            if self._skills[key].start_time <= insertion.timestamp
        ]
        return (
            max(preceding, key=lambda key: self._skills[key].start_time)
            if preceding
            else candidates[0]
        )
