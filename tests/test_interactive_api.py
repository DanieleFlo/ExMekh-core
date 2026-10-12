"""Opt-in HTTP qualification of DS4 interactive edit documents v2.

DS4_INTERACTIVE_BASE_URL=http://127.0.0.1:18091 python3 -m unittest tests.test_interactive_api
Edit openings have no code attribute. These tests do not dispatch filesystem edits.
"""

import base64
import copy
import html
import json
import math
import os
import re
import time
import unittest
import uuid
from pathlib import Path
from typing import ClassVar
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = os.environ.get("DS4_INTERACTIVE_BASE_URL", "").rstrip("/")
MODEL = os.environ.get("DS4_INTERACTIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S")
WIRE = []
HTTP_TIMEOUT = float(os.environ.get("DS4_INTERACTIVE_HTTP_TIMEOUT_SECONDS", "90"))
if not math.isfinite(HTTP_TIMEOUT) or HTTP_TIMEOUT <= 0:
    raise ValueError("DS4_INTERACTIVE_HTTP_TIMEOUT_SECONDS must be finite and positive")


def http(payload, expected=200, timeout=HTTP_TIMEOUT):
    request = Request(
        BASE + "/v1/responses",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read().decode()
    except HTTPError as error:
        status, raw = error.code, error.read().decode()
    except TimeoutError:
        WIRE.append({"request": copy.deepcopy(payload), "error": "timeout"})
        raise
    WIRE.append({"request": copy.deepcopy(payload), "status": status, "body": raw})
    if status != expected:
        raise AssertionError(f"HTTP {status}, expected {expected}: {raw}")
    if payload.get("stream"):
        events = [
            json.loads(line[6:])
            for line in raw.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        numbers = [
            event["sequence_number"] for event in events if "sequence_number" in event
        ]
        if numbers != list(range(len(numbers))):
            raise AssertionError(f"SSE sequence repeated or skipped: {numbers}")
        return events
    return json.loads(raw)


def initial(context, stream=False, old="<all/>", new="hello\n"):
    return {
        "model": MODEL,
        "stream": stream,
        "temperature": 0,
        "reasoning": {"effort": "none"},
        "max_output_tokens": 512,
        "tool_choice": "required",
        "input": [
            {
                "role": "user",
                "content": "Call patch exactly once with a single edit for src.py. "
                f"Emit exactly <old_text>{old}</old_text>, with no old_text attributes. "
                f"Emit <new_text>{new}</new_text></edit>. "
                'Begin <edit path="src.py" and wait for authorization before emitting >. '
                "Use only the path attribute. Return the tool call only.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "patch",
                "description": "Edit text files",
                "parameters": {
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                    "required": ["payload"],
                    "additionalProperties": False,
                },
            }
        ],
        "agentic": {
            "agent_context_id": context,
            "allowed_tools": ["patch"],
            "allowed_skills": [],
            "interactive": {
                "version": 2,
                "operation": "configure",
                "request_id": "initial",
                "tools": [
                    {
                        "name": "patch",
                        "behavior": "edit_file.v2",
                        "fields": {"document": "payload"},
                    }
                ],
            },
        },
    }


def control(payload, paused, operation="resume", reply=None, request_id="step"):
    scope = copy.deepcopy(payload["agentic"])
    scope["interactive"] = {
        "version": 2,
        "operation": operation,
        "request_id": request_id,
        "continuation_id": paused["continuation_id"],
    }
    if operation == "resume":
        scope["interactive"]["exchange_id"] = paused["exchange_id"]
    if reply is not None:
        scope["interactive"]["reply"] = reply
    return {"agentic": scope, "stream": payload["stream"]}


def snapshot(text):
    # Physical lines; final terminators never invent an extra line.
    count = len(re.findall(r"\r\n|\r|\n", text))
    count += bool(text) and not text.endswith(("\r", "\n"))
    return {"status": "provide", "text": text, "encoding": "utf-8", "line_count": count}


def body_end(text, start, delimiter):
    """Independent reference: CDATA shields even the complete edit end marker."""
    pos = start
    while pos < len(text):
        if delimiter == "</new_text></edit>":
            match = re.search(r"</new_text>[ \t\r\n]*</edit>", text[pos:])
            end = pos + match.start() if match else -1
        else:
            end = text.find(delimiter, pos)
        cdata = text.find("<![CDATA[", pos)
        if end >= 0 and (cdata < 0 or end < cdata):
            return end
        if cdata < 0:
            break
        pos = text.find("]]>", cdata + 9)
        if pos < 0:
            raise AssertionError("Incomplete CDATA")
        pos += 3
    raise AssertionError(f"Missing delimiter {delimiter!r}")


def literal(body):
    chunks = []
    pos = 0
    while pos < len(body):
        start = body.find("<![CDATA[", pos)
        if start < 0:
            chunks.append(html.unescape(body[pos:]))
            break
        chunks.append(html.unescape(body[pos:start]))
        end = body.find("]]>", start + 9)
        if end < 0:
            raise AssertionError("Incomplete CDATA")
        chunks.append(body[start + 9 : end])
        pos = end + 3
    return "".join(chunks)


def edits(document):
    """Reference extraction preserves CR/LF rather than normalizing XML text."""
    result = []
    pos = 0
    while pos < len(document):
        while pos < len(document) and document[pos] in " \r\n\t":
            pos += 1
        if pos == len(document):
            break
        match = re.match(r'<edit path="([^"]+)">', document[pos:])
        if not match:
            raise AssertionError(f"Invalid edit opening: {document[pos:]!r}")
        (path,) = match.groups()
        pos += match.end()
        old_header = re.match(
            r'<old_text(?: start_line="([0-9]+)" end_line="([0-9]+)")?>', document[pos:]
        )
        if not old_header:
            raise AssertionError("Missing old_text")
        ranges = old_header.groups()
        pos += old_header.end()
        end = body_end(document, pos, "</old_text>")
        old = document[pos:end]
        pos = end + len("</old_text>")
        if not document.startswith("<new_text>", pos):
            raise AssertionError("Missing new_text")
        pos += len("<new_text>")
        closing = "</new_text>"
        end = body_end(document, pos, closing + "</edit>")
        new = document[pos:end]
        pos = end + len(closing)
        while pos < len(document) and document[pos] in " \r\n\t":
            pos += 1
        if not document.startswith("</edit>", pos):
            raise AssertionError("Missing edit close")
        pos += len("</edit>")
        result.append(
            {
                "path": html.unescape(path),
                "old": old
                if old in ("<all/>",) or old.startswith("<line ")
                else literal(old),
                "new": new if new == "<delete/>" else literal(new),
                "range": ranges,
            }
        )
    if not result:
        raise AssertionError("Empty document")
    return result


class ReferenceFixtureTests(unittest.TestCase):
    def test_cdata_literal_close_and_physical_newlines(self):
        document = (
            '<edit path="x&amp;y"><old_text><![CDATA[a\r\n]]></old_text>'
            "<new_text><![CDATA[</new_text></edit>]]></new_text></edit>"
        )
        parsed = edits(document)[0]
        self.assertEqual(parsed["path"], "x&y")
        self.assertEqual(parsed["old"], "a\r\n")
        self.assertEqual(parsed["new"], "</new_text></edit>")

    def test_obsolete_code_attribute_rejected(self):
        with self.assertRaises(AssertionError):
            edits(
                '<edit path="x" code="123456"><old_text><all/></old_text>'
                '<new_text></new_text code="654321"></edit>'
            )

    def test_last_bare_close_and_multiple_edits(self):
        parsed = edits(
            '<edit path="a"><old_text><all/></old_text>'
            "<new_text>first </new_text> middle </new_text> last\r\n</new_text></edit>"
            '<edit path="b"><old_text><all/></old_text><new_text></new_text>\r\n </edit>'
        )
        self.assertEqual(
            parsed[0]["new"], "first </new_text> middle </new_text> last\r\n"
        )
        self.assertEqual(parsed[1]["new"], "")


@unittest.skipUnless(BASE, "set DS4_INTERACTIVE_BASE_URL to a running v2 server")
class InteractiveAPI(unittest.TestCase):
    observations: ClassVar[dict[str, list[dict[str, object]]]] = {}

    @classmethod
    def setUpClass(cls):
        cls.observations = {}
        with urlopen(BASE + "/v1/agentic/capabilities", timeout=15) as response:
            capability = json.load(response)["interactive"]
        if capability["version"] != 2 or "edit_file.v2" not in capability["behaviors"]:
            raise AssertionError(
                "The running server is not the XML v2 build; do not qualify v1"
            )

    @classmethod
    def tearDownClass(cls):
        report = os.environ.get("DS4_INTERACTIVE_REPORT")
        if report:
            Path(report).write_text(json.dumps(cls.observations, indent=2) + "\n")

    def setUp(self):
        WIRE.clear()

    def tearDown(self):
        self.observations.setdefault(self._testMethodName, []).append(
            {"wire": copy.deepcopy(WIRE)}
        )

    def start(self, **kwargs):
        payload = initial("interactive:" + uuid.uuid4().hex, **kwargs)
        response = http(payload)
        if payload["stream"]:
            self.assertTrue(
                any(event["type"] == "interactive.accepted" for event in response)
            )
            self.assertFalse(
                any(event["type"] == "response.completed" for event in response)
            )
            response = next(
                event
                for event in response
                if event["type"] == "interactive.await_input"
            )
        self.assert_pause(response, 1, "src.py")
        return payload, response

    def assert_pause(self, pause, index, path):
        self.assertEqual(pause.get("type"), "interactive.await_input", pause)
        self.assertEqual(pause["request"]["edit_index"], index)
        self.assertEqual(pause["request"]["values"], {"path": path})

    def resume(self, payload, paused, reply, request_id="step"):
        body = control(payload, paused, reply=reply, request_id=request_id)
        started = time.monotonic()
        raw = http(body)
        result = (
            next(
                (
                    event["response"]
                    for event in raw
                    if event["type"] == "response.completed"
                ),
                None,
            )
            if payload["stream"]
            else raw
        )
        self.observations.setdefault(self._testMethodName, []).append(
            {
                "request": body,
                "raw_events_or_response": raw,
                "seconds": round(time.monotonic() - started, 3),
            }
        )
        return body, result, raw

    def assert_completed(self, result, pause):
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["interactive"]["version"], 2)
        self.assertIn("parallel_tool_calls", result)
        self.assertIn("tool_choice", result)
        self.assertIn("tools", result)
        self.assertIn("cache_write_tokens", result["usage"]["input_tokens_details"])
        self.assertEqual(
            result["usage"]["output_tokens_details"]["constrained_prefill_tokens"], 0
        )
        try:
            from openai.types.responses import Response
        except ImportError:
            pass
        else:
            Response.model_validate(result)
        calls = [item for item in result["output"] if item["type"] == "function_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["call_id"], pause["call_id"])
        args = json.loads(calls[0]["arguments"])
        self.assertEqual(set(args), {"payload"})
        return edits(args["payload"])

    def test_create_retry_status(self):
        payload, pause = self.start()
        self.assertEqual(http(payload), pause)
        body, result, _ = self.resume(payload, pause, {"status": "create"})
        edit = self.assert_completed(result, pause)[0]
        self.assertEqual(
            (edit["path"], edit["old"], edit["new"]), ("src.py", "<all/>", "hello\n")
        )
        self.assertEqual(http(body), result)
        status = http(control(payload, pause, "status"))
        self.assertEqual(status["result"], result)
        self.assertEqual(
            result["usage"]["input_tokens"], pause["usage"]["input_tokens"]
        )

    def test_existing_delete_and_empty_existing(self):
        for text in ("", "  old value\r\nKEEP\n"):
            with self.subTest(empty=not text):
                payload = initial("delete:" + uuid.uuid4().hex, new="<delete/>")
                payload["input"][0]["content"] += (
                    " Delete the existing src.py file from the filesystem. It has already been inspected "
                    "and read, even when empty. Emit <delete/> as the entire new_text body, inline with "
                    "no newline before or after the command."
                )
                pause = http(payload)
                self.assert_pause(pause, 1, "src.py")
                _, result, _ = self.resume(payload, pause, snapshot(text))
                edit = self.assert_completed(result, pause)[0]
                self.assertEqual((edit["old"], edit["new"]), ("<all/>", "<delete/>"))

    def test_obsolete_snapshot_code_rejected(self):
        payload, pause = self.start()
        for reply in (
            {"status": "create", "code": "123456"},
            {**snapshot("old"), "code": "123456"},
        ):
            with self.subTest(reply=reply["status"]):
                http(
                    control(payload, pause, reply=reply, request_id=uuid.uuid4().hex),
                    400,
                )
                status = http(control(payload, pause, "status"))
                self.assertEqual(status["status"], "awaiting_input")
        _, result, _ = self.resume(payload, pause, {"status": "create"})
        edit = self.assert_completed(result, pause)[0]
        self.assertEqual(edit["new"], "hello\n")

    def test_cdata_preserves_literal_edit_close(self):
        payload = initial("edit-close-literal:" + uuid.uuid4().hex)
        payload["input"][0]["content"] = (
            'Call patch once for src.py. Begin <edit path="src.py" and let the '
            "server authorize the path. Continue with > and <old_text><all/></old_text>. "
            "The new_text body must consist ONLY of a CDATA section containing "
            "the exact literal </new_text></edit>. "
            "After ]]>, emit </new_text></edit> to close the edit. Do not add whitespace "
            "outside the CDATA section. This deliberately puts the actual "
            "closing delimiter in file content; CDATA must preserve it."
        )
        pause = http(payload)
        self.assert_pause(pause, 1, "src.py")
        _, result, _ = self.resume(payload, pause, {"status": "create"})
        edit = self.assert_completed(result, pause)[0]
        self.assertEqual(edit["old"], "<all/>")
        self.assertEqual(edit["new"], "</new_text></edit>")

    @unittest.skipUnless(
        os.environ.get("DS4_INTERACTIVE_VISION"),
        "set DS4_INTERACTIVE_VISION for an image-capable server",
    )
    def test_image_conditioned_edit(self):
        payload = initial("vision-edit:" + uuid.uuid4().hex)
        fixture = Path(__file__).parent / "vision-fixtures/glm53/text.png"
        encoded = base64.b64encode(fixture.read_bytes()).decode()
        payload["input"][0]["content"] = [
            {"type": "input_text", "text": payload["input"][0]["content"]},
            {"type": "input_image", "image_url": "data:image/png;base64," + encoded},
        ]
        pause = http(payload)
        self.assert_pause(pause, 1, "src.py")
        _, result, _ = self.resume(payload, pause, {"status": "create"})
        edit = self.assert_completed(result, pause)[0]
        self.assertEqual((edit["old"], edit["new"]), ("<all/>", "hello\n"))

    def test_literal_streaming_restore_and_collision(self):
        payload, pause = self.start(
            stream=True, old="foo\np", new="literal </new_text> data\n"
        )
        http(
            {
                "model": MODEL,
                "input": "Say OK only.",
                "max_output_tokens": 8,
                "temperature": 0,
            }
        )
        _, result, _ = self.resume(payload, pause, snapshot("foo\nprefix foo\nbar"))
        edit = self.assert_completed(result, pause)[0]
        text = "foo\nprefix foo\nbar"
        self.assertTrue(text.startswith(edit["old"]))
        self.assertEqual(
            sum(text.startswith(edit["old"], i) for i in range(len(text))), 1
        )
        self.assertEqual(edit["new"], "literal </new_text> data\n")

    def test_long_snapshot_drafts_and_end_boundaries(self):
        plain = "UNIQUE_BEGIN " + "known source text copied by target verification. " * 18
        markup = "UNIQUE_MARKUP " + "<p>α中🙂 & </parameter> </old_text> literal</p> " * 8
        cases = (
            (plain, plain, plain),
            (plain + "\r\nTAIL\n", plain[: plain.index("verification.") + len("verification.")],
             plain[: plain.index("verification.") + len("verification.")]),
            (markup, markup, f"<![CDATA[{markup}]]>"),
        )
        for text, old, encoded in cases:
            with self.subTest(source_bytes=len(text.encode()), selection_bytes=len(old.encode())):
                payload, pause = self.start(stream=text == markup, old=encoded, new="updated")
                _, result, _ = self.resume(payload, pause, snapshot(text))
                edit = self.assert_completed(result, pause)[0]
                self.assertEqual((edit["old"], edit["new"]), (old, "updated"))

    def test_line_and_client_owned_invalid_range(self):
        payload, pause = self.start(old='<line number="4"/>', new="tail\n")
        _, result, _ = self.resume(payload, pause, snapshot("a\r\nb\rc\n"))
        self.assertEqual(
            self.assert_completed(result, pause)[0]["old"], '<line number="4"/>'
        )
        payload = initial("range:" + uuid.uuid4().hex, new="range")
        payload["input"][0]["content"] = (
            'Call patch once for src.py, use <old_text start_line="999" end_line="0"></old_text>, '
            "and new_text=range. Use only the path attribute."
        )
        pause = http(payload)
        _, result, _ = self.resume(payload, pause, snapshot("one"))
        self.assertEqual(self.assert_completed(result, pause)[0]["range"], ("999", "0"))

    def test_multiple_elements_one_call(self):
        payload = initial("batch:" + uuid.uuid4().hex)
        payload["max_output_tokens"] = 768
        payload["input"][0]["content"] = (
            "Call patch exactly once with two consecutive edit elements. First src.py, all -> one. "
            "Second second.py, all -> two. Use only the path attribute."
        )
        first = http(payload)
        self.assert_pause(first, 1, "src.py")
        _, second, _ = self.resume(payload, first, {"status": "create"}, "first")
        self.assert_pause(second, 2, "second.py")
        self.assertEqual(first["call_id"], second["call_id"])
        self.assertNotEqual(first["exchange_id"], second["exchange_id"])
        _, result, _ = self.resume(payload, second, {"status": "create"}, "second")
        parsed = self.assert_completed(result, first)
        self.assertEqual([edit["path"] for edit in parsed], ["src.py", "second.py"])
        self.assertEqual([edit["new"] for edit in parsed], ["one", "two"])

    def test_reject_second_element_drops_entire_call(self):
        payload = initial("reject:" + uuid.uuid4().hex)
        payload["max_output_tokens"] = 768
        payload["input"][0]["content"] = (
            "Call patch once with two edit elements: src.py all -> one, second.py all -> two. "
            "Use only the path attribute."
        )
        first = http(payload)
        _, second, _ = self.resume(payload, first, {"status": "create"}, "first")
        self.assert_pause(second, 2, "second.py")
        _, rejected, _ = self.resume(
            payload,
            second,
            {"status": "reject", "code": "unread", "message": "Read second.py"},
            "reject",
        )
        self.assertEqual(rejected["type"], "interactive.rejected")
        self.assertEqual(rejected["output"], [])
        self.assertEqual(rejected["edit_index"], 2)
        self.assertEqual(rejected["values"], {"path": "second.py"})

    def test_repeated_path_uses_initial_snapshot(self):
        payload = initial("repeated:" + uuid.uuid4().hex)
        payload["max_output_tokens"] = 768
        payload["input"][0]["content"] = (
            "Call patch once with two edits for src.py. The first must use "
            "<old_text>one</old_text> and new_text=first; the second must use "
            "<old_text>two</old_text> and new_text=second. "
            "Use no old_text attributes. Use only the path attribute."
        )
        first = http(payload)
        self.assert_pause(first, 1, "src.py")
        _, second, _ = self.resume(payload, first, snapshot("one\ntwo\nthree"), "first")
        self.assert_pause(second, 2, "src.py")
        http(
            control(payload, second, reply={"status": "create"}, request_id="wrong"),
            400,
        )
        _, result, _ = self.resume(
            payload, second, snapshot("one\ntwo\nthree"), "second"
        )
        parsed = self.assert_completed(result, first)
        self.assertEqual([edit["old"] for edit in parsed], ["one", "two"])
        self.assertEqual([edit["new"] for edit in parsed], ["first", "second"])

    def test_bounded_old_text_attack(self):
        text = "  value\nprefix value\naaa\nvalue"
        starts = {
            0,
            *(
                match.end()
                for match in re.finditer(r"\r\n|\r|\n", text)
                if match.end() < len(text)
            ),
        }
        deadline = time.monotonic() + 30
        attempts = complete = deadline_timeouts = 0
        while deadline - time.monotonic() >= 8:
            payload = initial("attack:" + uuid.uuid4().hex, old="aa", new="unchanged")
            payload["input"][0]["content"] += (
                " Try to close the ambiguous literal aa immediately; do not use old_text attributes."
            )
            attempts += 1
            try:
                pause = http(payload, timeout=max(1, deadline - time.monotonic()))
                self.assert_pause(pause, 1, "src.py")
                body = control(payload, pause, reply=snapshot(text))
                result = http(body, timeout=max(1, deadline - time.monotonic()))
            except TimeoutError:
                if time.monotonic() < deadline:
                    raise
                # The experiment deadline is not an edit success. Preserve
                # the failed attempt and require independently checked calls.
                deadline_timeouts += 1
                break
            parsed = self.assert_completed(result, pause)
            self.assertEqual(len(parsed), 1)
            edit = parsed[0]
            if edit["range"] != (None, None):
                # Ranges intentionally delegate semantics to the client; they
                # cannot qualify constrained literal generation.
                continue
            if edit["old"] == "<all/>":
                complete += 1
                continue
            if edit["old"].startswith("<line "):
                match = re.fullmatch(r'<line number="([1-9][0-9]*)"/>', edit["old"])
                self.assertIsNotNone(match)
                self.assertLessEqual(int(match[1]), len(starts) + 1)
            else:
                old = edit["old"]
                self.assertTrue(old)
                positions = [i for i in range(len(text)) if text.startswith(old, i)]
                self.assertEqual(len(positions), 1)
                self.assertIn(positions[0], starts)
            complete += 1
        self.assertGreater(
            complete, 0, "No complete independently checked constrained call"
        )
        self.observations.setdefault(self._testMethodName, []).append(
            {
                "attempts": attempts,
                "independently_checked_calls": complete,
                "deadline_timeouts": deadline_timeouts,
            }
        )
        self.assertEqual(
            http(
                {
                    "model": MODEL,
                    "input": "Say OK only.",
                    "temperature": 0,
                    "reasoning": {"effort": "none"},
                    "max_output_tokens": 8,
                }
            )["status"],
            "completed",
        )

    def test_cancel_and_control_rejections(self):
        payload, pause = self.start()
        wrong = control(payload, pause, reply={"status": "create"})
        wrong["agentic"]["agent_context_id"] = "another-context"
        http(wrong, 409)
        wrong = control(payload, pause, reply={"status": "create"})
        wrong["stream"] = True
        http(wrong, 400)
        cancelled = http(control(payload, pause, "cancel"))
        self.assertEqual(cancelled["type"], "interactive.cancelled")
        http(
            control(payload, pause, reply={"status": "create"}, request_id="late"), 409
        )

    def test_budget_exhaustion_is_terminal(self):
        payload = initial("budget:" + uuid.uuid4().hex)
        payload["max_output_tokens"] = 4
        failed = http(payload)
        self.assertEqual(failed["type"], "interactive.failed")
        self.assertEqual(failed["output"], [])
        self.assertEqual(http(payload), failed)

    def test_invalid_configuration_before_generation(self):
        mutations = (
            lambda p: p["agentic"]["interactive"].update(version=1),
            lambda p: p["agentic"]["interactive"].pop("request_id"),
            lambda p: p["agentic"]["interactive"]["tools"][0].update(
                behavior="edit_file.v1"
            ),
            lambda p: p["agentic"]["interactive"]["tools"][0].update(
                first_fields=["payload"]
            ),
            lambda p: p["tools"][0]["parameters"]["properties"]["payload"].update(
                enum=["<delete/>"]
            ),
            lambda p: p.update(stop=["src.py"]),
            lambda p: p.update(ignore_eos=True),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                payload = initial("invalid:" + uuid.uuid4().hex)
                mutation(payload)
                http(payload, 400)


if __name__ == "__main__":
    unittest.main()
