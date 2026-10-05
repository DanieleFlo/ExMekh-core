"""Deterministic editor and archived-output scenarios, using the normal app."""

from __future__ import annotations

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.models import ChatMessage, ToolEvent
from exmekh_agent.main import main
from exmekh_agent.state.agent_design import AgentDesign, DesignLink, DesignNode
from exmekh_agent.state.session_manager import session_manager
from tests.run_message_edit_agent import provider


def seed():
    config.data_dir.joinpath("work").mkdir(parents=True, exist_ok=True)
    project = session_manager.create_project(config.work_root)
    session = session_manager.create_session(project_id=project.id)
    nodes = [
        DesignNode(
            id="root",
            kind="hds",
            name="root",
            content="Original root instructions",
            x=80,
            y=200,
        ),
        DesignNode(
            id="worker",
            kind="hds",
            name="worker",
            content="Worker instructions",
            x=650,
            y=380,
        ),
        DesignNode(
            id="free-worker",
            kind="hds",
            name="free-worker",
            content="Free worker",
            x=880,
            y=600,
        ),
        DesignNode(id="thinking", kind="skill", name="deep-thinking", x=350, y=100),
        DesignNode(id="nested", kind="skill", name="nested", x=650, y=200),
        DesignNode(id="calculate", kind="tool", name="calculate", x=350, y=360),
        DesignNode(id="date-time", kind="tool", name="date-time", x=650, y=520),
    ]
    session.settings.agent_profile = "custom"
    session.settings.agent_override = True
    session.settings.agent_design = AgentDesign(
        nodes=nodes,
        root_id="root",
        skill_connections_explicit=True,
        links=[
            DesignLink(parent="root", child="thinking"),
            DesignLink(parent="thinking", child="calculate"),
            DesignLink(parent="root", child="worker"),
            DesignLink(parent="worker", child="nested"),
        ],
    )
    calls = [
        {
            "id": mode,
            "function": {
                "name": "read_tool_output",
                "arguments": {"output_id": "archived", "summary_mode": mode},
            },
        }
        for mode in ("none", "brief", "detailed")
    ]
    outputs = [
        {
            "role": "tool",
            "content": "<header><name>read-web-page</name><type>Tool</type><id>archived</id>"
            + (f"<summary-mode>{mode}</summary-mode>" if mode != "none" else "")
            + "</header>\n"
            + mode.upper()
            + " ARCHIVED RESULT",
        }
        for mode in ("none", "brief", "detailed")
    ]
    history = [
        {"role": "user", "content": "Read saved website outputs"},
        {"role": "assistant", "tool_calls": calls},
        *reversed(outputs),
        {"role": "assistant", "content": "Archived results are ready."},
    ]
    session.messages = [
        ChatMessage(role="user", content="Read saved website outputs"),
        ChatMessage(
            role="assistant",
            content="Archived results are ready.",
            raw_history=history,
            tool_events=[
                ToolEvent(
                    tool_name="read_tool_output",
                    tool_call_id=call["id"],
                    arguments=call["function"]["arguments"],
                )
                for call in calls
            ],
        ),
    ]
    session_manager.save_session(session)


if __name__ == "__main__":
    assert provider
    seed()
    main()
