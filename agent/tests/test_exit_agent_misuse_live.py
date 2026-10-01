"""Opt-in adversarial runs against a real local model and the real QS agent."""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for_process
from typing import Any
from uuid import uuid4

import pytest

from fe.config import config
from fe.qs_bridge.agent_runner import check_provider_health, run_agent_turn

pytestmark = pytest.mark.live

_CASE_TIMEOUT_SECONDS = 90


def _run_live_case(prompt: str, model: str, sender: Connection) -> None:
    raw_history: list[dict[str, Any]] = []
    skill_tree: list[dict[str, Any]] = []

    def capture(history: list[dict[str, Any]], tree: list[dict[str, Any]]) -> None:
        raw_history[:] = history
        skill_tree[:] = tree

    try:
        asyncio.run(
            run_agent_turn(
                user_prompt=prompt,
                model=model,
                on_token=lambda _: None,
                on_reasoning=lambda *_: None,
                on_tool_event=lambda _: None,
                agent_mode="deep_agent",
                agent_context_id=f"exit-agent-misuse-{uuid4()}",
                on_raw_history=capture,
                history=[],
            )
        )
        sender.send({"history": raw_history, "tree": skill_tree})
    except Exception:
        sender.send({"error": traceback.format_exc()})
    finally:
        sender.close()


@dataclass(slots=True)
class LiveCaseProcess:
    """Run one model case; return None after stopping it at the deadline."""

    prompt: str
    model: str
    timeout_seconds: int = _CASE_TIMEOUT_SECONDS

    def run(self) -> dict[str, Any] | None:
        context = get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_run_live_case, args=(self.prompt, self.model, sender)
        )
        started = False
        try:
            process.start()
            started = True
            sender.close()
            ready = wait_for_process(
                [receiver, process.sentinel], timeout=self.timeout_seconds
            )
            if receiver in ready:
                return receiver.recv()
            if ready:
                raise RuntimeError("The live process exited without an execution trace")
            return None
        finally:
            receiver.close()
            sender.close()
            if started:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                process.join()


@dataclass(frozen=True, slots=True)
class SkillExecutionTrace:
    """Read the public serialized history of one QS skill execution."""

    history: list[dict[str, Any]]
    output: str | None

    @classmethod
    def from_tree(
        cls, tree: list[dict[str, Any]], skill_name: str
    ) -> SkillExecutionTrace:
        node = next((item for item in tree if item.get("skill") == skill_name), None)
        if node is None:
            raise ValueError(f"{skill_name} assente: {tree!r}")
        return cls(history=node["history"], output=node.get("output"))

    def tool_calls(self, name: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for message in self.history:
            raw_calls = message.get("tool_calls") or []
            if isinstance(raw_calls, str):
                raw_calls = json.loads(raw_calls)
            for call in raw_calls:
                function = call.get("function") or {}
                if function.get("name") != name:
                    continue
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append(arguments)
        return calls

    def tool_errors(self, name: str) -> list[str]:
        prefix = f"Error {name}:"
        return [
            str(message.get("content", ""))
            for message in self.history
            if message.get("role") == "tool"
            and str(message.get("content", "")).startswith(prefix)
        ]

    def has_requested_misuse(self, case: str) -> bool:
        """Check that the model actually produced the payload under test."""
        for call in self.tool_calls("exit_agent"):
            items = call.get("items")
            if case == "missing_items" and "items" not in call:
                return True
            if case == "wrong_items_type" and isinstance(items, str):
                return True
            if (
                case == "unknown_output_id"
                and isinstance(items, list)
                and "id-inesistente-xyz" in items
            ):
                return True
            if (
                case == "duplicate_output_id"
                and isinstance(items, list)
                and len(items) != len(set(items))
            ):
                return True
            if (
                case == "placeholder_without_item"
                and "{{id-inesistente-xyz}}" in str(call.get("text", ""))
                and not items
            ):
                return True
        return False


@pytest.mark.parametrize(
    ("case", "instruction", "expected_error"),
    [
        (
            "missing_items",
            'Chiama prima exit_agent con solo text="tentativo", omettendo items.',
            "Invalid arguments",
        ),
        (
            "wrong_items_type",
            'Chiama prima exit_agent con text="tentativo" e items="abc" (stringa, non lista).',
            "items must be a list",
        ),
        (
            "unknown_output_id",
            'Chiama prima exit_agent con text="tentativo" e items=["id-inesistente-xyz"].',
            "not present in this node's history",
        ),
        (
            "duplicate_output_id",
            "Chiama mock-sub-tool per ottenere un output salvato. "
            'Poi chiama exit_agent con text="tentativo" e lo stesso saved-output-id '
            "due volte nella lista items.",
            "appears more than once",
        ),
        (
            "placeholder_without_item",
            'Chiama prima exit_agent con text="tentativo {{id-inesistente-xyz}}" e items=[].',
            "requires its ID in items",
        ),
    ],
)
def test_exit_agent_rejects_misuse_then_recovers_live(
    case: str, instruction: str, expected_error: str
) -> None:
    """Each invalid first call must yield feedback; the skill must then exit validly."""
    available, models = asyncio.run(check_provider_health())
    if not available or not models:
        pytest.skip("The local provider is unavailable for live tests")
    model = config.default_model if config.default_model in models else models[0]

    prompt = (
        "Test reale del comportamento di exit_agent. Entra in mock-skill. "
        f"Caso {case}: {instruction} "
        "Questa prima chiamata deve essere volutamente errata. Leggi il messaggio "
        "di errore restituito dal tool; poi correggi gli argomenti e chiama di nuovo "
        "exit_agent con text non vuoto e items=[]. Nel messaggio finale descrivi solo "
        "ciò che è avvenuto davvero, senza simulare chiamate."
    )
    result = LiveCaseProcess(prompt, model).run()
    if result is None:
        pytest.skip(
            f"Live case {case} timed out after {_CASE_TIMEOUT_SECONDS}s; recovery was not verified"
        )

    if "error" in result:
        pytest.fail(result["error"])

    assert result["history"], "Manca la traccia reale del turno"
    trace = SkillExecutionTrace.from_tree(result["tree"], "mock-skill")
    exits = trace.tool_calls("exit_agent")
    errors = trace.tool_errors("exit_agent")

    if not trace.has_requested_misuse(case):
        pytest.skip(
            f"The model did not produce the requested malformed payload for {case}; this case was not exercised"
        )
    assert len(exits) >= 2, f"Servono errore e correzione reali: {trace.history!r}"
    assert any(expected_error in error for error in errors), (
        f"Nessun feedback atteso per {case}: {errors!r}"
    )
    assert errors[0].endswith("Correct the error and call exit_agent again."), (
        f"The feedback does not explain how to recover: {errors[0]!r}"
    )
    last_arguments = exits[-1]
    assert isinstance(last_arguments.get("text"), str)
    assert last_arguments["text"].strip()
    assert last_arguments.get("items") == []
    assert trace.output, "The skill did not exit successfully"
