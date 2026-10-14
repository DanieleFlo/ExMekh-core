"""Opt-in HTTP qualification of literal edits (behavior v3, transport v2)."""

import copy
import html
import json
import os
import re
import time
import unittest
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = os.environ.get("DS4_INTERACTIVE_BASE_URL", "").rstrip("/")
MODEL = os.environ.get("DS4_INTERACTIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S")
WIRE = []


def http(payload, expected=200):
    request = Request(BASE + "/v1/responses", json.dumps(payload).encode(),
                      {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=90) as response:
            status, raw = response.status, response.read().decode()
    except HTTPError as error:
        status, raw = error.code, error.read().decode()
    WIRE.append({"request": copy.deepcopy(payload), "status": status, "body": raw})
    if status != expected:
        raise AssertionError(f"HTTP {status}, expected {expected}: {raw}")
    if payload.get("stream"):
        events = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith("data: ") and line[6:] != "[DONE]"]
        numbers = [item["sequence_number"] for item in events if "sequence_number" in item]
        if numbers != list(range(len(numbers))):
            raise AssertionError(f"SSE sequence repeated or skipped: {numbers}")
        return events
    return json.loads(raw)


def initial(context, stream=False, old="KEEP", new="hello"):
    return {
        "model": MODEL, "stream": stream, "temperature": 0,
        "reasoning": {"effort": "none"}, "max_output_tokens": 768,
        "tool_choice": "required",
        "input": [{"role": "user", "content":
            "Call patch exactly once for src.py. Begin <exmekh_edit path=\"src.py\" "
            "and let the server supply the opening ID. "
            f"The exact old body is {json.dumps(old)} and new body is {json.dumps(new)}. "
            "Use only exmekh_edit and exmekh_new_text framing. Bodies are literal; "
            "do not add CDATA, entities, closing edit tags, or trailing newlines. "
            "Copy the supplied ID into the separator. Return only the tool call."}],
        "tools": [{"type": "function", "name": "patch", "description": "Edit existing text files",
                   "parameters": {"type": "object", "properties": {"payload": {"type": "string"}},
                                  "required": ["payload"], "additionalProperties": False}}],
        "agentic": {"agent_context_id": context, "allowed_tools": ["patch"], "allowed_skills": [],
                    "interactive": {"version": 2, "operation": "configure", "request_id": "initial",
                                    "tools": [{"name": "patch", "behavior": "edit_file.v3",
                                               "fields": {"document": "payload"}}]}}
    }


def control(payload, paused, operation="resume", reply=None, request_id="step"):
    scope = copy.deepcopy(payload["agentic"])
    scope["interactive"] = {"version": 2, "operation": operation, "request_id": request_id,
                            "continuation_id": paused["continuation_id"]}
    if operation == "resume":
        scope["interactive"]["exchange_id"] = paused["exchange_id"]
    if reply is not None:
        scope["interactive"]["reply"] = reply
    return {"agentic": scope, "stream": payload["stream"]}


def snapshot(text):
    count = len(re.findall(r"\r\n|\r|\n", text))
    count += bool(text) and not text.endswith(("\r", "\n"))
    return {"status": "provide", "text": text, "encoding": "utf-8", "line_count": count}


def edits(document):
    """Independent reference parser; bodies are not XML-decoded."""
    opening = re.compile(r'<exmekh_edit path="([^"<>]+)"(?: id="([0-9]{3})")?\s*/>')
    separator = re.compile(r'<exmekh_new_text(?: id="([0-9]{3})")?\s*/>')
    headers = list(opening.finditer(document))
    if not headers or document[:headers[0].start()].strip():
        raise AssertionError("Missing edit opening")
    used = set()
    result = []
    for index, header in enumerate(headers):
        identifier = header[2]
        if identifier is not None:
            if identifier in used:
                raise AssertionError("Duplicate edit ID")
            used.add(identifier)
        end = headers[index + 1].start() if index + 1 < len(headers) else len(document)
        markers = [match for match in separator.finditer(document, header.end(), end)
                   if match[1] is None or identifier is None or match[1] == identifier]
        explicit = [match for match in markers if identifier is not None and match[1] == identifier]
        if explicit:
            markers = explicit
        if len(markers) != 1:
            raise AssertionError("Missing or ambiguous separator")
        marker = markers[0]
        result.append({"path": html.unescape(header[1]), "id": identifier,
                       "old": document[header.end():marker.start()],
                       "new": document[marker.end():end]})
    return result


class ReferenceFixtureTests(unittest.TestCase):
    def test_literal_markup_and_whitespace(self):
        parsed = edits('<exmekh_edit path="x&amp;y" id="427" />a\r\n'
                       '<exmekh_new_text id="427"/><![CDATA[<all/>]]>&lt;\r\n ')[0]
        self.assertEqual(parsed["path"], "x&y")
        self.assertEqual(parsed["old"], "a\r\n")
        self.assertEqual(parsed["new"], "<![CDATA[<all/>]]>&lt;\r\n ")

    def test_missing_ids_and_multiple_edits(self):
        parsed = edits('<exmekh_edit path="a" />A<exmekh_new_text/>FIRST\r\n'
                       '<exmekh_edit path="b" />B<exmekh_new_text/>SECOND')
        self.assertEqual([item["new"] for item in parsed], ["FIRST\r\n", "SECOND"])

    def test_ambiguous_separator_and_obsolete_header(self):
        for source in ('<edit path="x"><all/></edit>',
                       '<exmekh_edit path="x" />A<exmekh_new_text/>B<exmekh_new_text/>C'):
            with self.assertRaises(AssertionError):
                edits(source)


@unittest.skipUnless(BASE, "set DS4_INTERACTIVE_BASE_URL to a running v3 server")
class InteractiveAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with urlopen(BASE + "/v1/agentic/capabilities", timeout=15) as response:
            capability = json.load(response)["interactive"]
        if capability["version"] != 2 or "edit_file.v3" not in capability["behaviors"]:
            raise AssertionError("The running server does not support literal edits v3")

    def start(self, **kwargs):
        payload = initial("literal:" + uuid.uuid4().hex, **kwargs)
        raw = http(payload)
        paused = next(item for item in raw if item["type"] == "interactive.await_input") if payload["stream"] else raw
        self.assertEqual(paused["type"], "interactive.await_input", paused)
        self.assertEqual(paused["request"]["values"], {"path": "src.py"})
        return payload, paused

    def resume(self, payload, paused, reply, request_id="step"):
        body = control(payload, paused, reply=reply, request_id=request_id)
        raw = http(body)
        if payload["stream"]:
            final = next(item["response"] for item in raw if item["type"] == "response.completed")
        else:
            final = raw
        return body, final

    def completed(self, final):
        calls = [item for item in final["output"] if item["type"] == "function_call"]
        self.assertEqual(len(calls), 1, final)
        self.assertEqual(calls[0]["name"], "patch")
        arguments = json.loads(calls[0]["arguments"])
        self.assertEqual(set(arguments), {"payload"})
        return edits(arguments["payload"])

    def test_literal_streaming_and_nonstreaming(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                replacement = '<all/><delete/>&lt;<![CDATA[raw]]>'
                payload, paused = self.start(stream=streaming, old="  old", new=replacement)
                _, final = self.resume(payload, paused, snapshot("  old\r\nKEEP\n"))
                edit = self.completed(final)[0]
                self.assertEqual(edit["old"], "  old")
                self.assertEqual(edit["new"], replacement)
                self.assertRegex(edit["id"], r"^[0-9]{3}$")

    def test_existing_empty_file_and_retry_receipt(self):
        payload, paused = self.start(old="", new="FILLED")
        body, final = self.resume(payload, paused, snapshot(""))
        self.assertEqual(self.completed(final)[0]["new"], "FILLED")
        self.assertEqual(http(body), final)

    def test_missing_separator_id(self):
        payload = initial("missing-id:" + uuid.uuid4().hex)
        payload["input"][0]["content"] = (
            'Call patch once for src.py. Let the server supply the opening ID. '
            'Select exactly KEEP and replace it with exactly hello. '
            'For this test the separator MUST be exactly <exmekh_new_text/> with NO id attribute. '
            'The optional separator ID is deliberately omitted. Do not copy the opening ID into it. '
            'Do not add whitespace or closing tags. Return only the tool call.'
        )
        paused = http(payload)
        _, final = self.resume(payload, paused, snapshot("KEEP"))
        self.assertEqual(self.completed(final)[0]["new"], "hello")
        self.assertIn("<exmekh_new_text/>", json.loads(final["output"][0]["arguments"])["payload"])

    def test_creation_reply_is_rejected(self):
        payload, paused = self.start()
        http(control(payload, paused, reply={"status": "create"}), expected=400)
        http(control(payload, paused, operation="cancel"))

    def test_repeated_path_uses_initial_snapshot_and_distinct_ids(self):
        payload = initial("multiple:" + uuid.uuid4().hex)
        payload["input"][0]["content"] = (
            'Call patch exactly once. Its payload must contain this entire two-block template: '
            '<exmekh_edit path="src.py" />ONE<exmekh_new_text/>FIRST'
            '<exmekh_edit path="src.py" />TWO<exmekh_new_text/>SECOND'
            '. The server adds an opening ID to each block; you may copy it into its separator. '
            'The first replacement is only FIRST. Do not close the payload after FIRST: '
            'immediately emit the second exmekh_edit opening inside the same payload. '
            'Both selections refer to the original ONE newline TWO snapshot. '
            'Do not emit a second patch call. Do not add whitespace or final newline.'
        )
        first = http(payload)
        _, second = self.resume(payload, first, snapshot("ONE\nTWO"))
        self.assertEqual(second.get("type"), "interactive.await_input", second)
        self.assertEqual(second["request"]["edit_index"], 2)
        _, final = self.resume(payload, second, snapshot("ONE\nTWO"), "second")
        result = self.completed(final)
        self.assertEqual([(item["old"], item["new"]) for item in result],
                         [("ONE", "FIRST"), ("TWO", "SECOND")])
        self.assertNotEqual(result[0]["id"], result[1]["id"])

    def test_second_resource_rejection_drops_entire_call(self):
        payload = initial("reject:" + uuid.uuid4().hex)
        payload["input"][0]["content"] = (
            "Return ONE patch call with two consecutive edit blocks. "
            "First src.py old ONE new FIRST; then other.py old TWO new SECOND. "
            "Use the literal two-marker framing and supplied IDs."
        )
        first = http(payload)
        _, second = self.resume(payload, first, snapshot("ONE"))
        self.assertEqual(second.get("type"), "interactive.await_input", second)
        _, final = self.resume(payload, second, {"status": "reject", "code": "unread", "message": "read first"}, "reject")
        self.assertEqual(final["type"], "interactive.rejected")
        self.assertEqual(final["output"], [])


if __name__ == "__main__":
    unittest.main()