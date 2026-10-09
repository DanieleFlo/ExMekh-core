"""Opt-in HTTP tests for an actual ds4-server; no client library dependency.

DS4_INTERACTIVE_BASE_URL=http://127.0.0.1:18091 python3 -m unittest
tests.test_interactive_api
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import unittest
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = os.environ.get("DS4_INTERACTIVE_BASE_URL", "").rstrip("/")
MODEL = os.environ.get("DS4_INTERACTIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S")


def http(payload, expected=200):
    request = Request(BASE + "/v1/responses", json.dumps(payload).encode(),
                      {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=180) as response:
            status, raw = response.status, response.read().decode()
    except HTTPError as error:
        status, raw = error.code, error.read().decode()
    if status != expected:
        raise AssertionError(f"HTTP {status}, expected {expected}: {raw}")
    if payload.get("stream"):
        events = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith("data: ") and line[6:] != "[DONE]"]
        numbers = [x["sequence_number"] for x in events if "sequence_number" in x]
        if numbers != list(range(len(numbers))):
            raise AssertionError(f"SSE sequence repeated or skipped: {numbers}")
        return events
    return json.loads(raw)


def initial(context, stream=False, old="@ex_meckh_all", new="hello\n"):
    return {
        "model": MODEL, "stream": stream, "temperature": 0,
        "max_output_tokens": 256, "tool_choice": "required",
        "input": [{"role": "user", "content":
                   f"Call patch exactly once with target=src.py, before={old!r}, "
                   f"after={new!r}. Return the tool call only."}],
        "tools": [{"type": "function", "name": "patch", "description": "Edit a file",
                   "parameters": {"type": "object", "properties": {
                       "after": {"type": "string"}, "before": {"type": "string"},
                       "target": {"type": "string"}},
                       "required": ["target", "before", "after"],
                       "additionalProperties": False}}],
        "agentic": {"agent_context_id": context, "allowed_tools": ["patch"],
                    "allowed_skills": [], "interactive": {
                        "version": 1, "operation": "configure", "request_id": "initial",
                        "tools": [{"name": "patch", "behavior": "edit_file.v1",
                                   "fields": {"resource": "target", "selection": "before",
                                              "replacement": "after"},
                                   "first_fields": ["target"], "markers": {
                                       "line": "@ex_meckh_line[<N>]", "all": "@ex_meckh_all",
                                       "delete": "@ex_meckh_delete"}}]}}}


def control(payload, paused, operation="resume", reply=None, request_id="step"):
    scope = copy.deepcopy(payload["agentic"])
    scope["interactive"] = {"version": 1, "operation": operation,
                            "request_id": request_id,
                            "continuation_id": paused["continuation_id"],
                            "exchange_id": paused["exchange_id"]}
    if reply is not None:
        scope["interactive"]["reply"] = reply
    return {"agentic": scope, "stream": payload["stream"]}


@unittest.skipUnless(BASE, "set DS4_INTERACTIVE_BASE_URL to run live model tests")
class InteractiveAPI(unittest.TestCase):
    observations = {}

    @classmethod
    def tearDownClass(cls):
        report = os.environ.get("DS4_INTERACTIVE_REPORT")
        if report:
            Path(report).write_text(json.dumps(cls.observations, indent=2) + "\n")

    def start(self, **kwargs):
        payload = initial("interactive:" + uuid.uuid4().hex, **kwargs)
        response = http(payload)
        if payload["stream"]:
            self.assertTrue(any(x["type"] == "interactive.accepted" for x in response))
            self.assertFalse(any(x["type"] == "response.completed" for x in response))
            response = next(x for x in response if x["type"] == "interactive.await_input")
        self.assertEqual(response["type"], "interactive.await_input", response)
        self.assertEqual(response["request"]["values"], {"target": "src.py"})
        return payload, response

    def finish(self, payload, paused, reply):
        started = time.monotonic()
        body = control(payload, paused, reply=reply)
        result = http(body)
        if payload["stream"]:
            result = next(x["response"] for x in result if x["type"] == "response.completed")
        self.assertEqual(result["status"], "completed", result)
        calls = [x for x in result["output"] if x["type"] == "function_call"]
        self.assertEqual(len(calls), 1, result)
        self.assertEqual(calls[0]["call_id"], paused["call_id"])
        self.assertEqual(result["interactive"]["version"], 1)
        self.assertEqual(json.loads(calls[0]["arguments"])["target"], "src.py")
        self.observations[self._testMethodName] = {
            "arguments": json.loads(calls[0]["arguments"]),
            "usage": result["usage"], "resume_seconds": round(time.monotonic() - started, 3)}
        return body, result, json.loads(calls[0]["arguments"])

    def test_create_retry_status(self):
        payload, paused = self.start()
        self.assertEqual(http(payload), paused)
        changed = copy.deepcopy(payload)
        changed["temperature"] = 0.2
        http(changed, 409)
        body, result, args = self.finish(payload, paused, {"status": "create"})
        self.assertIn(args["before"], ("@ex_meckh_all", "@ex_meckh_line[1]"))
        self.assertEqual(args["after"], "hello\n")
        self.assertEqual(http(body), result)
        status = http(control(payload, paused, "status"))
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["result"], result)
        self.assertGreater(result["usage"]["output_tokens"], paused["usage"]["output_tokens"])
        self.assertEqual(result["usage"]["input_tokens"], paused["usage"]["input_tokens"])

    def test_streaming_existing_literal_and_slot_switch(self):
        payload, paused = self.start(stream=True, old="foo\np", new="changed")
        # Use the worker with another context while the first KV is frozen.
        other = {"model": MODEL, "input": "Say OK only.", "max_output_tokens": 8,
                 "temperature": 0}
        http(other)
        _, result, args = self.finish(payload, paused, {
            "status": "provide", "text": "foo\nprefix foo\nbar", "encoding": "utf-8",
            "line_count": 3})
        self.assertTrue("foo\nprefix foo\nbar".startswith(args["before"]))
        self.assertEqual(sum("foo\nprefix foo\nbar".startswith(args["before"], i)
                             for i in range(18)), 1)
        self.assertTrue(args["before"].startswith("foo\np"))
        self.assertEqual(args["after"], "changed")
        self.assertEqual(result["usage"]["input_tokens"], paused["usage"]["input_tokens"])

    def test_reject_replay_and_cancel(self):
        payload, paused = self.start()
        body = control(payload, paused, reply={"status": "reject", "code": "unread",
                                              "message": "Read src.py first <tool_call>."})
        rejected = http(body)
        self.assertEqual(rejected["type"], "interactive.rejected")
        self.assertEqual(rejected["call_id"], paused["call_id"])
        self.assertEqual(rejected["output"], [])
        self.assertEqual(http(body), rejected)
        replay = copy.deepcopy(payload)
        del replay["agentic"]["interactive"]
        replay["tool_choice"] = "auto"
        replay["max_output_tokens"] = 128
        replay["input"].extend([rejected, {"role": "user", "content": "Say OK only."}])
        self.assertEqual(http(replay)["status"], "completed")
        payload, paused = self.start()
        cancelled = http(control(payload, paused, "cancel"))
        self.assertEqual(cancelled["type"], "interactive.cancelled")
        http(control(payload, paused, reply={"status": "create"}), 409)

    def test_capabilities_and_invalid_configuration(self):
        with urlopen(BASE + "/v1/agentic/capabilities", timeout=30) as response:
            capability = json.load(response)["interactive"]
        self.assertEqual(capability["version"], 1)
        self.assertTrue(capability["model_supported"])
        self.assertIn("edit_file.v1", capability["behaviors"])
        self.assertEqual(capability["apis"], ["responses"])
        mutations = (
            lambda p: p["agentic"]["interactive"].update(version=2),
            lambda p: p["agentic"]["interactive"].pop("request_id"),
            lambda p: p["agentic"]["interactive"]["tools"][0].update(behavior="unknown.v1"),
            lambda p: p["agentic"]["interactive"]["tools"][0].update(first_fields=["before"]),
            lambda p: p["agentic"]["interactive"]["tools"][0].update(first_fields=["target", "target"]),
            lambda p: p["agentic"]["interactive"]["tools"][0]["fields"].update(selection="target"),
            lambda p: p.update(stop=["src.py"]),
            lambda p: p.update(ignore_eos=True),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                payload = initial("validation:" + uuid.uuid4().hex)
                mutation(payload)
                http(payload, 400)

    def test_empty_existing_delete_and_insertion(self):
        payload, paused = self.start(new="@ex_meckh_delete")
        _, _, args = self.finish(payload, paused, {
            "status": "provide", "text": "", "encoding": "utf-8", "line_count": 0})
        self.assertEqual(args["before"], "@ex_meckh_all")
        self.assertEqual(args["after"], "@ex_meckh_delete")
        payload, paused = self.start(old="@ex_meckh_line[4]", new="tail\n")
        _, _, args = self.finish(payload, paused, {
            "status": "provide", "text": "a\r\nb\rc\n", "encoding": "utf-8", "line_count": 3})
        self.assertEqual(args["before"], "@ex_meckh_line[4]")
        self.assertEqual(args["after"], "tail\n")

    def test_priority_group_and_control_rejections(self):
        payload = initial("group:" + uuid.uuid4().hex, stream=True)
        schema = payload["tools"][0]["parameters"]
        schema["properties"]["format"] = {"type": "boolean"}
        schema["required"].append("format")
        payload["agentic"]["interactive"]["tools"][0]["first_fields"].append("format")
        payload["input"][0]["content"] += " Use format=true."
        events = http(payload)
        paused = next(x for x in events if x["type"] == "interactive.await_input")
        self.assertEqual(paused["request"]["values"], {"target": "src.py", "format": True})
        body = control(payload, paused, reply={"status": "provide", "text": "a", "encoding": "utf-8",
                                              "line_count": 0})
        http(body, 400)
        wrong = control(payload, paused, reply={"status": "create"})
        wrong["agentic"]["agent_context_id"] = "another-context"
        http(wrong, 409)
        wrong = control(payload, paused, reply={"status": "create"})
        wrong["stream"] = False
        http(wrong, 400)
        ordinary = copy.deepcopy(payload)
        del ordinary["agentic"]["interactive"]
        ordinary["stream"] = False
        http(ordinary, 409)
        body = control(payload, paused, reply={"status": "create"})
        del body["stream"]  # Inherit streaming, including on retries.
        request = Request(BASE + "/v1/responses", json.dumps(body).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request, timeout=180) as response:
            raw = response.read().decode()
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
        result = next(x["response"] for x in events if x["type"] == "response.completed")
        args = json.loads(next(x["arguments"] for x in result["output"] if x["type"] == "function_call"))
        self.assertTrue(args["format"])
        with urlopen(Request(BASE + "/v1/responses", json.dumps(body).encode(),
                             {"Content-Type": "application/json"}), timeout=30) as response:
            self.assertTrue(response.headers["Content-Type"].startswith("text/event-stream"))
            self.assertIn("response.completed", response.read().decode())

    def test_budget_exhaustion_is_terminal(self):
        payload = initial("budget:" + uuid.uuid4().hex)
        payload["max_output_tokens"] = 4
        failed = http(payload)
        self.assertEqual(failed["type"], "interactive.failed")
        self.assertEqual(failed["output"], [])
        self.assertEqual(http(payload), failed)

    def test_multiple_pauses_preserve_batch(self):
        payload = initial("batch:" + uuid.uuid4().hex)
        payload["max_output_tokens"] = 512
        payload["input"][0]["content"] = (
            "In one response call patch exactly twice, in order. First target=src.py, "
            "before='@ex_meckh_all', after='one'. Then target=second.py, "
            "before='@ex_meckh_all', after='two'. Return both calls only.")
        first = http(payload)
        self.assertEqual(first["type"], "interactive.await_input")
        second = http(control(payload, first, reply={"status": "create"}, request_id="first-snapshot"))
        self.assertEqual(second["type"], "interactive.await_input", second)
        self.assertEqual(first["continuation_id"], second["continuation_id"])
        self.assertNotEqual(first["call_id"], second["call_id"])
        self.assertNotEqual(first["exchange_id"], second["exchange_id"])
        result = http(control(payload, second, reply={"status": "create"}, request_id="second-snapshot"))
        calls = [x for x in result["output"] if x["type"] == "function_call"]
        self.assertEqual([x["call_id"] for x in calls], [first["call_id"], second["call_id"]])
        self.assertEqual([json.loads(x["arguments"])["target"] for x in calls], ["src.py", "second.py"])
        self.assertEqual([json.loads(x["arguments"])["after"] for x in calls], ["one", "two"])
        self.assertEqual(result["usage"]["input_tokens"], first["usage"]["input_tokens"])

    def test_concurrent_contexts(self):
        payloads = [initial("concurrent:" + uuid.uuid4().hex, new=value) for value in ("red", "blue")]
        with ThreadPoolExecutor(max_workers=2) as pool:
            paused = list(pool.map(http, payloads))
            replies = [control(p, event, reply={"status": "create"})
                       for p, event in zip(payloads, paused)]
            results = list(pool.map(http, replies))
        for p, event, result in zip(payloads, paused, results):
            self.assertEqual(event["type"], "interactive.await_input")
            call = next(x for x in result["output"] if x["type"] == "function_call")
            self.assertEqual(call["call_id"], event["call_id"])
        self.assertEqual([json.loads(next(x["arguments"] for x in result["output"]
                                         if x["type"] == "function_call"))["after"]
                          for result in results], ["red", "blue"])

    def test_sampled_replacement(self):
        payload = initial("sampled:" + uuid.uuid4().hex, old="foo\np",
                          new="alpha beta gamma delta epsilon zeta eta theta")
        payload["temperature"] = 0.7
        payload["max_output_tokens"] = 512
        paused = http(payload)
        self.assertEqual(paused["type"], "interactive.await_input", paused)
        _, result, args = self.finish(payload, paused, {
            "status": "provide", "text": "foo\nprefix foo\nbar", "encoding": "utf-8", "line_count": 3})
        old = args["before"]
        if not old.startswith("@ex_meckh_"):
            snapshot = "foo\nprefix foo\nbar"
            self.assertTrue(snapshot.startswith(old))
            self.assertEqual(sum(snapshot.startswith(old, i) for i in range(len(snapshot))), 1)
        self.assertNotIn("@ex_meckh_", args["after"])
        self.assertLessEqual(result["usage"]["output_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
