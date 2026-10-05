"""Full app demo with recorded tool/skill output and a deterministic provider."""

from __future__ import annotations

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.models import ChatMessage, SkillNode, ToolEvent
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from tests.run_message_edit_agent import provider


def output(name: str, identifier: str, body: str) -> str:
    return f"<header>\n<name>{name}</name>\n<type>Tool</type>\n<id>{identifier}</id>\n</header>\n{body}"


def seed() -> None:
    config.data_dir.joinpath("work").mkdir(parents=True, exist_ok=True)
    project = session_manager.create_project(config.work_root)
    session = session_manager.create_session(project_id=project.id)
    root_tool = ToolEvent(
        tool_name="bash", tool_call_id="first", arguments={"command": "first command"}
    )
    second_tool = ToolEvent(
        tool_name="bash",
        tool_call_id="second",
        arguments={"command": "second command"},
        timestamp=root_tool.timestamp + 1,
    )
    inner_tool = ToolEvent(
        tool_name="calculate", tool_call_id="inner", skill_path="/analysis", depth=1
    )
    skill = SkillNode(
        name="analysis",
        path="/analysis",
        depth=1,
        status="completed",
        opening_tool_call_id="skill-call",
        is_collapsed=True,
        tool_events=[inner_tool],
    )
    history = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Inspect tool results"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "first",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "first command"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "first",
            "name": "bash",
            "content": output(
                "bash",
                "first-output",
                "FIRST RESULT\n"
                + "\n".join(f"Log line {index:03d}: completed" for index in range(120)),
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "second",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "second command"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "second",
            "name": "bash",
            "content": output(
                "bash", "second-output", '{"result":"SECOND RESULT","rows":[1,2,3]}'
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "skill-call", "function": {"name": "analysis", "arguments": {}}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "skill-call",
            "name": "analysis",
            "content": output(
                "analysis", "analysis-output", "SKILL FINAL RESULT\nAnalysis finished."
            ),
        },
        {
            "role": "assistant",
            "content": "Results are ready. Click a tool row to inspect the output.",
        },
    ]
    inner_history = [
        history[-3],
        {"role": "user", "content": "Analyze"},
        {
            "role": "tool",
            "tool_call_id": "inner",
            "name": "calculate",
            "content": output(
                "calculate", "inner-output", '{"answer":42,"steps":["one","two"]}'
            ),
        },
        {"role": "assistant", "content": "Finished"},
    ]
    session.system_prompt = history[0]["content"]
    session.messages = [
        ChatMessage(role="user", content=history[1]["content"]),
        ChatMessage(
            role="assistant",
            content=history[-1]["content"],
            raw_history=history,
            tool_events=[root_tool, second_tool, inner_tool],
            skills=[skill],
            skill_history_tree=[
                {
                    "skill": "analysis",
                    "opening_tool_call_id": "skill-call",
                    "history": inner_history,
                    "children": [],
                }
            ],
        ),
    ]
    session_manager.save_session(session)


if __name__ == "__main__":
    assert provider
    seed()
    main()
