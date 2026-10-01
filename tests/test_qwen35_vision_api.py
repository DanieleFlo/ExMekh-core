#!/usr/bin/env python3
"""Opt-in Qwen27 vision API gate against an idle CUDA server with --vision.

Uses generated solid-color PNGs, so no image package or downloaded fixture is
needed. Run once with target-only and once with --mtp; checkpoint parity is
covered separately by test_server_agentic_payload.
"""
import argparse
import base64
import json
from pathlib import Path
import struct
import urllib.request
import zlib


def image(rgb, api="chat"):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    width, height = 320, 256
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress((b"\0" + bytes(rgb) * width) * height)) + chunk(b"IEND", b""))
    data = base64.b64encode(png).decode()
    if api == "anthropic":
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen3.8-27B-UD-Q4_K_S")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = []

    def post(body, path="/v1/chat/completions", stream=False):
        request = urllib.request.Request(args.url.rstrip("/") + path, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            if not stream:
                return json.load(response)
            text, finished = "", False
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    finished = True
                    break
                for choice in json.loads(data).get("choices", []):
                    text += choice.get("delta", {}).get("content", "") or ""
            assert finished, "SSE ended without [DONE]"
            return text

    def body(messages, **kwargs):
        return dict(model=args.model, messages=messages, temperature=0, max_tokens=64, thinking=False, **kwargs)

    def check(name, run):
        try:
            detail = run()
            results.append(dict(case=name, status="PASS", detail=detail))
        except Exception as exc:
            results.append(dict(case=name, status="FAIL", error=str(exc)))
        print(name, results[-1]["status"], flush=True)
        if args.report:
            args.report.write_text(json.dumps(results, indent=2) + "\n")

    red, blue = image((230, 20, 20)), image((20, 40, 230))
    prompt = "Name the dominant color of the image. Reply with one color word only."
    history = [{"role": "user", "content": [red, {"type": "text", "text": prompt}]}]

    def color(parts, words):
        reply = post(body([{"role": "user", "content": parts}]))
        text = reply["choices"][0]["message"]["content"].lower()
        positions = [text.find(word) for word in words]
        assert all(p >= 0 for p in positions) and positions == sorted(positions), text
        return text

    check("single-red", lambda: color(history[0]["content"], ["red"]))

    def continuation():
        reply = post(body(history))
        answer = reply["choices"][0]["message"]
        follow = post(body(history + [answer, {"role": "user", "content": "What was that color? One word."}]))
        text = follow["choices"][0]["message"]["content"].lower()
        cached = follow["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert "red" in text and cached > 0, (text, cached)
        return dict(content=text, cached_tokens=cached)

    check("cached-image-continuation", continuation)
    def append_image():
        reply = post(body(history))
        answer = reply["choices"][0]["message"]
        messages = history + [answer, {"role": "user", "content": [
            blue, {"type": "text", "text": "Name the dominant color of the NEW image. One word."}]}]
        follow = post(body(messages))
        text = follow["choices"][0]["message"]["content"].lower()
        cached = follow["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert "blue" in text and cached > 0, (text, cached)
        return dict(content=text, cached_tokens=cached)

    check("append-image-to-live-context", append_image)

    check("two-images-order", lambda: color([red, blue, {"type": "text", "text":
          "Name the dominant color of EACH image, in image order. Reply with both colors only."}], ["red", "blue"]))
    check("changed-pixels-blue", lambda: color([blue, {"type": "text", "text": prompt}], ["blue"]))

    def thinking():
        request = body(history)
        request.update(thinking=True, max_tokens=512)
        message = post(request)["choices"][0]["message"]
        assert message.get("reasoning_content") and "red" in message["content"].lower(), message
        return dict(content=message["content"], reasoning_chars=len(message["reasoning_content"]))

    check("thinking", thinking)
    color_schema = {"type": "object", "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
                    "required": ["color"], "additionalProperties": False}

    def structured():
        request = body([{"role": "user", "content": [blue, {"type": "text", "text":
                       "Identify the dominant color and return the color in JSON."}]}])
        request["response_format"] = {"type": "json_schema", "json_schema":
                                      {"name": "color", "strict": True, "schema": color_schema}}
        value = json.loads(post(request)["choices"][0]["message"]["content"])
        assert value == {"color": "blue"}, value
        return value

    check("json-preserves-image", structured)

    def tools():
        messages = [{"role": "user", "content": [red, {"type": "text", "text":
                    "Call record_color with the dominant color of this image."}]}]
        request = body(messages)
        request.update(tools=[{"type": "function", "function": {"name": "record_color",
                       "description": "Record the observed color", "parameters": color_schema}}],
                       tool_choice="required", max_tokens=128)
        message = post(request)["choices"][0]["message"]
        call = message["tool_calls"][0]
        assert call["function"]["name"] == "record_color", call
        assert json.loads(call["function"]["arguments"])["color"] == "red", call
        follow = messages + [message, {"role": "tool", "tool_call_id": call["id"], "content": "Recorded."},
                             {"role": "user", "content": "Which color did you observe? One word."}]
        reply = post(body(follow))
        text = reply["choices"][0]["message"]["content"].lower()
        assert "red" in text, text
        return dict(arguments=call["function"]["arguments"], continuation=text)

    check("tool-and-image-continuation", tools)
    check("sse-blue", lambda: stream_color(blue, prompt, body, post))

    def anthropic():
        reply = post({"model": args.model, "max_tokens": 64, "temperature": 0, "thinking": False,
                      "messages": [{"role": "user", "content": [image((20, 40, 230), "anthropic"),
                                    {"type": "text", "text": prompt}]}]}, "/v1/messages")
        text = "".join(part.get("text", "") for part in reply["content"]).lower()
        assert "blue" in text, text
        return text

    check("anthropic-blue", anthropic)
    return int(any(result["status"] != "PASS" for result in results))


def stream_color(blue, prompt, body, post):
    request = body([{"role": "user", "content": [blue, {"type": "text", "text": prompt}]}])
    request["stream"] = True
    text = post(request, stream=True)
    assert "blue" in text.lower(), text
    return text


if __name__ == "__main__":
    raise SystemExit(main())