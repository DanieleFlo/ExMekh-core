#!/usr/bin/env python3
"""Live regression: scoring a replay must not inherit the agent's tool frontier.

Reuses an already running server; does not load a second model or execute tools.
The same scoring prompt must have identical token usage with and without a live
binding for its last tool call. Run explicitly with --url and optionally --output.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(url, model, payload):
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/responses",
        data=json.dumps({"model": model, **payload}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def scoring_request(call):
    return {
        "instructions": "Rate the importance of the tool output. Return the requested scores.",
        "input": [
            {"role": "user", "content": "Call mock-tool once."},
            {key: call[key] for key in ("type", "call_id", "name", "arguments")},
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": "Useful evidence.",
            },
            {
                "role": "user",
                "content": "Score entry_3 as 30. Return only the JSON object.",
            },
        ],
        "reasoning": {"effort": "none"},
        "tool_choice": "none",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "CompactionScores",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "entry_3": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["entry_3"],
                    "additionalProperties": False,
                },
            }
        },
        "max_output_tokens": 128,
    }


def run(url):
    started = time.time()
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    prime = post(
        url,
        model,
        {
            "instructions": "Use the supplied mock tool. "
            + "Agent-only context must stay in its own request. " * 120,
            "input": [{"role": "user", "content": "Call mock-tool once."}],
            "tools": [
                {
                    "type": "function",
                    "name": "mock-tool",
                    "description": "Return evidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            ],
            "tool_choice": "required",
            "reasoning": {"effort": "none"},
            "max_output_tokens": 128,
        },
    )
    call = next(item for item in prime["output"] if item["type"] == "function_call")
    payload = scoring_request(call)
    bound = post(url, model, payload)
    # Replace the active tool frontier, then replay the identical scoring input.
    post(
        url,
        model,
        {
            "input": [{"role": "user", "content": "Reply OK."}],
            "reasoning": {"effort": "none"},
            "max_output_tokens": 16,
        },
    )
    isolated = post(url, model, payload)
    return {
        "seconds": time.time() - started,
        "prime_usage": prime["usage"],
        "bound": bound,
        "isolated": isolated,
        "same_prompt_tokens": bound["usage"]["input_tokens"]
        == isolated["usage"]["input_tokens"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.url)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("bound", "isolated")
            },
            indent=2,
        )
    )
    assert report["same_prompt_tokens"], (
        "Scoring inherited the agent frontier instead of using its own prompt"
    )
    for response in (report["bound"], report["isolated"]):
        assert response["status"] == "completed", response
        text = "".join(
            part["text"]
            for item in response["output"]
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )
        assert json.loads(text) == {"entry_3": 30}, text


if __name__ == "__main__":
    main()
