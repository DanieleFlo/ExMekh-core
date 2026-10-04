"""Isolated UI using the real health probe against mutable DS4 model metadata."""

from __future__ import annotations

from typing import Any

from nicegui import app

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge import agent_runner
from exmekh_agent.main import main

loaded = {"model": "qwen3.8-flash-next", "context_length": 16384}
calls: list[dict[str, Any]] = []


@app.get("/provider/v1/models")
def models() -> dict[str, Any]:
    model = str(loaded["model"])
    aliases = [model]
    if model == "qwen3.8-flash-next":
        aliases += [model + "-chat", model + "-reasoner"]
    return {
        "object": "list",
        "data": [
            {
                "id": alias,
                "owned_by": "ds4.c",
                "context_length": loaded["context_length"],
            }
            for alias in aliases
        ],
    }


@app.post("/test/loaded-model")
def replace_loaded_model(payload: dict[str, Any]) -> None:
    loaded.update(payload)


@app.get("/test/model-calls")
def model_calls() -> list[dict[str, Any]]:
    return calls


def execute_turn(**kwargs: Any) -> str:
    calls.append(
        {"model": kwargs["model"], "context_length": config.get_context_length()}
    )
    text = "MODEL_OK"
    kwargs["on_token"](text)
    return text


if __name__ == "__main__":
    config.base_url = f"http://{config.host}:{config.port}/provider/v1"
    config.default_model = "Qwen3.8-27B-UD-Q4_K_S"
    config.context_length = 22000
    agent_runner.execute_agent_turn_sync = execute_turn
    agent_runner.generate_chat_title_sync = lambda text, **kwargs: (
        "Model discovery test"
    )
    main()
