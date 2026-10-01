"""Isolated browser server which records execution mode without a model service."""

from __future__ import annotations

from typing import Any

from nicegui import app

from fe.main import main
from fe.qs_bridge import agent_runner
from fe.state.session_manager import session_manager
from fe.ui import settings_drawer

calls: list[dict[str, Any]] = []


def execute_turn(**kwargs: Any) -> str:
    calls.append(
        {
            key: kwargs[key]
            for key in (
                "agent_mode",
                "work_root",
                "name_hds",
                "tool_approval_mode",
                "user_prompt",
            )
        }
    )
    text = f"Response in {kwargs['agent_mode']}"
    kwargs["on_token"](text)
    return text


async def health(*args: Any, **kwargs: Any) -> tuple[bool, list[str]]:
    return True, ["project-test-model"]


@app.get("/test/state")
def state() -> dict[str, Any]:
    return {
        "current_id": session_manager.current_session_id,
        "projects": [
            project.model_dump(mode="json")
            for project in session_manager.list_projects()
        ],
        "sessions": [
            session.model_dump(mode="json")
            for session in session_manager.list_sessions_sorted()
        ],
        "calls": calls,
    }


agent_runner.execute_agent_turn_sync = execute_turn
agent_runner.generate_chat_title_sync = lambda text, **kwargs: f"Chat: {text[:30]}"
settings_drawer.check_provider_health = health

if __name__ == "__main__":
    main()
