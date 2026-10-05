"""Isolated browser server which records execution mode without a model service."""

from __future__ import annotations

from threading import get_ident
from typing import Any

from nicegui import app

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.settings import capability_paths_editor, settings_drawer

calls: list[dict[str, Any]] = []
folder_selection: str | None = None
picker_calls: list[dict[str, Any]] = []


def pick_test_folder(initial_dir):
    picker_calls.append({"initial_dir": initial_dir, "thread": get_ident()})
    return folder_selection


@app.post("/test/folder-selection")
def select_folder(payload: dict[str, str | None]):
    global folder_selection
    folder_selection = payload.get("path")
    return {"selected": folder_selection}


capability_paths_editor.pick_folder_dialog = pick_test_folder


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
                "capabilities_paths",
                "agent_profile",
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
        "default_capabilities_path": config.capabilities_path,
        "picker_calls": picker_calls,
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
