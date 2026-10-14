"""Opt-in reproduction of agent history prefill after steering or auxiliary calls.

Run against one idle server with DS4_STEERING_CACHE_BASE_URL set. Tests assert
usage.input_tokens_details.cached_tokens, including a normal boundary control.
They use synthetic tool replies and never execute tools or modify user files.
Set DS4_STEERING_CACHE_REPORT to retain exact wire requests and measurements.
DS4_STEERING_CACHE_STREAM=1 repeats the same fixtures over SSE;
DS4_STEERING_CACHE_ARCHIVE_LINES scales the conversation history (default 96).
DS4_STEERING_CACHE_REASONING selects the reasoning effort (default none).
DS4_STEERING_CACHE_INTERACTIVE=0 isolates ordinary tool/steering behavior.
"""

import copy
import json
import os
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from tests.test_agentic_api import call_output, function_calls, function_tool, user_item
from tests.test_interactive_api import control

BASE = os.getenv("DS4_STEERING_CACHE_BASE_URL", "").rstrip("/")
MODEL = os.getenv("DS4_STEERING_CACHE_MODEL", "Qwen3.8-27B-UD-Q4_K_S")
REPORT_PATH = os.getenv("DS4_STEERING_CACHE_REPORT", "")
STREAM = os.getenv("DS4_STEERING_CACHE_STREAM", "0") == "1"
ARCHIVE_LINES = int(os.getenv("DS4_STEERING_CACHE_ARCHIVE_LINES", "96"))
REASONING = os.getenv("DS4_STEERING_CACHE_REASONING", "none")
INTERACTIVE = os.getenv("DS4_STEERING_CACHE_INTERACTIVE", "1") != "0"
REGISTRY = [function_tool(name) for name in ("file-processing", "warm", "exit_agent")]
REGISTRY.append({
    "type": "function", "name": "patch", "description": "Edit existing text",
    "parameters": {"type": "object", "properties": {"edits": {"type": "string"}},
                   "required": ["edits"], "additionalProperties": False},
})


def cache_accounting(response):
    usage = response["usage"]
    total = usage["input_tokens"]
    cached = usage["input_tokens_details"]["cached_tokens"]
    if type(total) is not int or type(cached) is not int or not 0 <= cached <= total:
        raise AssertionError(f"Invalid cache counters: {usage}")
    return {"input_tokens": total, "cached_tokens": cached, "prefill_tokens": total - cached}


def replay(call):
    return {key: call[key] for key in ("type", "call_id", "name", "arguments")}


class CacheAccountingTests(unittest.TestCase):
    def test_partial_system_hit_is_not_a_complete_history_hit(self):
        observed = cache_accounting({"usage": {
            "input_tokens": 18737, "input_tokens_details": {"cached_tokens": 5240}}})
        self.assertEqual(observed["prefill_tokens"], 13497)
        self.assertLess(observed["cached_tokens"], 18927)

    def test_missing_or_inconsistent_counters_are_not_zero_hits(self):
        with self.assertRaises(KeyError):
            cache_accounting({"usage": {"input_tokens": 100}})
        with self.assertRaises(AssertionError):
            cache_accounting({"usage": {
                "input_tokens": 100, "input_tokens_details": {"cached_tokens": 101}}})


@unittest.skipUnless(BASE, "set DS4_STEERING_CACHE_BASE_URL to one idle server")
class SteeringCacheLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with urlopen(BASE + "/v1/agentic/capabilities", timeout=15) as response:
            capability = json.load(response)["interactive"]
        if INTERACTIVE and "edit_file.v3" not in capability["behaviors"]:
            raise AssertionError("These fixtures require literal edits v3")
        cls.report = {"model": MODEL, "base_url": BASE, "stream": STREAM,
                      "archive_lines": ARCHIVE_LINES, "reasoning": REASONING, "interactive": INTERACTIVE,
                      "measurements": [], "wire": []}

    @classmethod
    def tearDownClass(cls):
        if REPORT_PATH:
            Path(REPORT_PATH).write_text(json.dumps(cls.report, ensure_ascii=False, indent=2) + "\n")

    def send(self, body, label):
        started = time.monotonic()
        request = Request(BASE + "/v1/responses", json.dumps(body).encode(),
                          {"Content-Type": "application/json"})
        is_stream = False
        try:
            with urlopen(request, timeout=120) as response:
                is_stream = "text/event-stream" in response.headers.get("Content-Type", "")
                status, raw = response.status, response.read().decode()
        except HTTPError as error:
            status, raw = error.code, error.read().decode()
        events = []
        if is_stream:
            events = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith("data: ") and line[6:] != "[DONE]"]
            terminal = []
            for event in events:
                kind = event.get("type", "")
                if kind in {"response.completed", "response.incomplete", "response.failed"}:
                    terminal.append(event["response"])
                elif kind == "interactive.result":
                    terminal.append(event["response"])
                elif kind == "interactive.await_input" or kind.startswith("interactive.cancelled"):
                    terminal.append(event)
            self.assertTrue(terminal, (body, events))
            result = terminal[-1]
        else:
            result = json.loads(raw)
        self.report["wire"].append({"test": self.id(), "label": label,
                                    "request": copy.deepcopy(body), "status": status,
                                    "response": result, "events": events,
                                    "seconds": time.monotonic() - started})
        self.assertEqual(status, 200, (label, result))
        return result

    def payload(self, context, history, name, *, max_output_tokens=None):
        if max_output_tokens is None:
            max_output_tokens = 128 if REASONING == "none" else 1024
        body = {
            "model": MODEL, "input": copy.deepcopy(history), "tools": REGISTRY,
            "stream": STREAM, "temperature": 0,
            "reasoning": {"effort": REASONING, **({"summary": "auto"} if REASONING != "none" else {})},
            "max_output_tokens": max_output_tokens,
            "tool_choice": "required" if name else "none",
            "agentic": {
                "agent_context_id": context, "allowed_tools": ["warm", "patch", "exit_agent"],
                "allowed_skills": ["file-processing"],
                "interactive": {
                    "version": 2, "operation": "configure", "request_id": uuid4().hex,
                    "tools": [{"name": "patch", "behavior": "edit_file.v3",
                               "fields": {"document": "edits"}}],
                },
            },
        }
        if not INTERACTIVE:
            body["agentic"].pop("interactive")
        return body

    def one_call(self, response, name):
        self.assertEqual(response.get("status"), "completed", response)
        calls = function_calls(response)
        self.assertEqual(len(calls), 1, response)
        self.assertEqual(calls[0]["name"], name, response)
        return replay(calls[0])

    def warm_skill(self):
        context = "steering-cache:" + uuid4().hex
        archive = "\n".join(
            f"Archive record {index}: source contents KEEP; the work remains in this skill."
            for index in range(ARCHIVE_LINES)
        )
        history = [
            {"role": "system", "content":
             "You are running a synthetic protocol test. Call the selected function exactly once. "
             "Use src.py when patch is selected. Stop tool work when steering requests exit_agent."},
            user_item("Call only file-processing exactly once now. Wait for its result before calling warm. Preserve this archive as history:\n" + archive),
        ]
        opened = self.send(self.payload(context, history, "file-processing"), "open-skill")
        skill = self.one_call(opened, "file-processing")
        self.addCleanup(self.return_skill, context, skill)
        history.extend([skill, call_output(skill["call_id"], "Inside file-processing: call warm once.")])
        warm = self.send(self.payload(context, history, "warm"), "warm-skill")
        call = self.one_call(warm, "warm")
        history.extend([call, call_output(call["call_id"], "Warm finished; await the next instruction.")])
        baseline = cache_accounting(warm)["input_tokens"]
        self.assertGreater(baseline, 1000, "The fixture must contain substantial conversation history")
        return context, history, baseline

    def return_skill(self, context, skill):
        body = self.payload(context, [call_output(skill["call_id"], "Synthetic skill complete. Answer done.")],
                            None, max_output_tokens=160 if REASONING == "none" else 1024)
        scope = body["agentic"]
        body["tool_choice"] = "auto"
        scope.pop("interactive", None)
        scope.update(operation="return", skill_call_id=skill["call_id"])
        returned = self.send(body, "return-skill-cleanup")
        self.assertIn(returned.get("status"), {"completed", "incomplete"}, returned)

    def require_history_hit(self, response, baseline, scenario):
        measured = {"test": self.id(), "scenario": scenario,
                    "minimum_cached_tokens": baseline - 16, **cache_accounting(response)}
        self.report["measurements"].append(measured)
        print(json.dumps(measured), flush=True)
        self.assertGreaterEqual(measured["cached_tokens"], measured["minimum_cached_tokens"],
                                "Conversation history was replayed: " + json.dumps(measured))

    def exit_with_steering(self, context, history):
        history.append(user_item("Steering: stop editing and call exit_agent exactly once now."))
        response = self.send(self.payload(context, history, "exit_agent"), "steering-before-exit")
        self.one_call(response, "exit_agent")
        return response

    def exit_after_tool(self, context, history):
        history[-1] = call_output(history[-1]["call_id"], "Warm finished. Call exit_agent exactly once now.")
        response = self.send(self.payload(context, history, "exit_agent"), "exit-after-tool")
        self.one_call(response, "exit_agent")
        return response

    def test_full_replay_without_steering_keeps_history(self):
        context, history, baseline = self.warm_skill()
        self.require_history_hit(self.exit_after_tool(context, history), baseline,
                                 "full-replay-with-trailing-tool-output")

    def test_completed_tool_boundary_keeps_history_with_steering(self):
        context, history, baseline = self.warm_skill()
        self.require_history_hit(self.exit_with_steering(context, history), baseline,
                                 "steering-after-completed-tool")

    def cancelled_edit_pause(self):
        if not INTERACTIVE:
            self.skipTest("An edit pause requires the interactive contract")
        context, history, warm_baseline = self.warm_skill()
        history[-1] = call_output(history[-1]["call_id"],
                                 'Call patch once for src.py, selecting KEEP and replacing it with CHANGED. '
                                 'Start <exmekh_edit path="src.py" and wait for the server ID.')
        body = self.payload(context, history, "patch")
        paused = self.send(body, "pause-edit")
        self.assertEqual(paused.get("type"), "interactive.await_input", paused)
        self.assertEqual(paused["tool_name"], "patch")
        self.assertEqual(paused["request"]["values"], {"path": "src.py"})
        self.assertGreaterEqual(cache_accounting(paused)["cached_tokens"], warm_baseline - 16,
                                "History must still be warm before cancelling the edit")
        completed_frontier = cache_accounting(paused)["cached_tokens"]
        cancelled = self.send(control(body, paused, operation="cancel", request_id="steering-cancel"),
                              "cancel-for-steering")
        self.assertEqual(cancelled.get("type"), "interactive.cancelled", cancelled)
        self.assertEqual(cancelled.get("output"), [], cancelled)
        history.append({key: value for key, value in cancelled.items() if key not in {"output", "usage"}})
        history.append({
            "type": "interactive.cancelled", "version": 2,
            "continuation_id": paused["continuation_id"], "code": "steered",
            "message": "Steering invalidated this inference before dispatch",
        })
        # Cancellation discards the interrupted request and its unfinished call.
        # Reuse the last completed turn; the replaced tool reply and steering are
        # new input, so their small tail is correctly evaluated again.
        return context, history, completed_frontier

    def test_cancelled_edit_pause_keeps_history_with_steering(self):
        context, history, completed_frontier = self.cancelled_edit_pause()
        self.require_history_hit(self.exit_with_steering(context, history), completed_frontier,
                                 "steering-after-cancelled-edit-pause")

    def test_cancelled_edit_pause_survives_stateless_auxiliary(self):
        context, history, completed_frontier = self.cancelled_edit_pause()
        self.auxiliary_call(scoped=False)
        self.require_history_hit(self.exit_with_steering(context, history), completed_frontier,
                                 "steering-after-cancelled-edit-pause-and-auxiliary")

    def test_resumed_edit_keeps_history_with_steering(self):
        if not INTERACTIVE:
            self.skipTest("An edit pause requires the interactive contract")
        context, history, warm_baseline = self.warm_skill()
        history[-1] = call_output(history[-1]["call_id"],
                                 'Call patch once for src.py, selecting KEEP and replacing it with CHANGED. '
                                 'Start <exmekh_edit path="src.py" and wait for the server ID.')
        body = self.payload(context, history, "patch")
        paused = self.send(body, "pause-edit-to-resume")
        self.assertEqual(paused.get("type"), "interactive.await_input", paused)
        self.assertGreaterEqual(cache_accounting(paused)["cached_tokens"], warm_baseline - 16)
        resumed = self.send(control(body, paused, reply={
            "status": "provide", "text": "KEEP", "encoding": "utf-8", "line_count": 1,
        }), "resume-edit")
        call = self.one_call(resumed, "patch")
        self.assertIn("KEEP", json.loads(call["arguments"])["edits"])
        history.extend([call, call_output(call["call_id"], "Edit complete. Await further instructions.")])
        self.require_history_hit(self.exit_with_steering(context, history),
                                 cache_accounting(paused)["input_tokens"],
                                 "steering-after-resumed-edit")

    def auxiliary_call(self, *, scoped):
        body = {
            "model": MODEL, "input": [user_item("Return a command approval object with approved true.")],
            "temperature": 0, "reasoning": {"effort": "none"}, "max_output_tokens": 64,
            "text": {"format": {"type": "json_schema", "name": "approval", "strict": True,
                                 "schema": {"type": "object", "properties": {"approved": {"type": "boolean"}},
                                            "required": ["approved"], "additionalProperties": False}}},
        }
        if scoped:
            body["agentic"] = {"agent_context_id": "auxiliary:" + uuid4().hex,
                               "allowed_tools": [], "allowed_skills": []}
        result = self.send(body, "scoped-auxiliary" if scoped else "stateless-auxiliary")
        self.assertEqual(result.get("status"), "completed", result)

    def test_scoped_auxiliary_keeps_agent_history(self):
        context, history, baseline = self.warm_skill()
        self.auxiliary_call(scoped=True)
        self.require_history_hit(self.exit_after_tool(context, history), baseline,
                                 "auxiliary-with-independent-agent-context")

    def test_stateless_auxiliary_keeps_agent_history(self):
        context, history, baseline = self.warm_skill()
        self.auxiliary_call(scoped=False)
        self.require_history_hit(self.exit_after_tool(context, history), baseline,
                                 "auxiliary-without-agent-context")


if __name__ == "__main__":
    unittest.main()