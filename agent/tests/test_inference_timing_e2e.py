import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

from playwright.async_api import async_playwright, expect

from tests.browser_support import BrowserTestServer, send_command, wait_turn_completion


async def test_response_timer_shows_request_timing_only_while_inference_is_streaming(
    tmp_path,
):
    advance = Event()
    measure_tool = Event()
    decode = Event()
    finish = Event()

    class Provider(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/v1/models":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "timing-model"}]}).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions"
            if not body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(self.chunk({"content": "Timing test"}, "stop")).encode()
                )
                return
            assert self.headers.get("X-DS4-Stream-Timing") == "1"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": ds4-timing prefill 150 500\n\n")
            self.send_chunk({"role": "assistant"})
            self.send_chunk({"reasoning_content": "Consider using a tool"})
            assert advance.wait(25)
            self.wfile.write(b": ds4-timing tool 0 0\n\n")
            assert measure_tool.wait(25)
            self.wfile.write(b": ds4-timing tool 20 1000\n\n")
            assert decode.wait(25)
            self.wfile.write(b": ds4-timing decode 75 2500\n\n")
            self.send_chunk({"content": "Timing verified"})
            assert finish.wait(25)
            self.send_chunk({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")

        def send_chunk(self, delta, reason=None):
            self.wfile.write(
                b"data: " + json.dumps(self.chunk(delta, reason)).encode() + b"\n\n"
            )

        @staticmethod
        def chunk(delta, finish_reason):
            return {
                "id": "timing-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "timing-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "message": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }

    with ThreadingHTTPServer(("127.0.0.1", 0), Provider) as server:
        worker = Thread(target=server.serve_forever)
        worker.start()
        environment = {
            "BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            "DEFAULT_MODEL": "timing-model",
            "API_TYPE": "chat_completions",
            "API_KEY": "",
        }
        try:
            with BrowserTestServer(tmp_path, environment=environment) as url:
                async with async_playwright() as playwright:
                    browser = await playwright.chromium.launch()
                    try:
                        page = await browser.new_page()
                        await page.goto(url)
                        await send_command(page, "Check live throughput")
                        timer = page.locator(".msg-assistant .live-turn-timing")
                        prefill = timer.filter(
                            has_text=re.compile(r"^Prefill 300\.0 tok/s · \d+s$")
                        )
                        tool = timer.filter(
                            has_text=re.compile(r"^Tool arguments 20\.0 tok/s · \d+s$")
                        )
                        tool_preparation = page.locator(
                            ".msg-assistant .tool-preparation-header"
                        )
                        live_reasoning = page.locator(
                            ".msg-assistant .reasoning-live-header"
                        )
                        await expect(prefill).to_be_visible()
                        await expect(live_reasoning).to_be_visible()
                        await expect(tool_preparation).to_have_count(0)
                        await page.reload()
                        await expect(prefill).to_be_visible()
                        advance.set()
                        await expect(tool_preparation).to_have_text(
                            "Preparing tool call…"
                        )
                        await expect(live_reasoning).to_have_count(0)
                        await expect(
                            page.locator(".reasoning-master-header")
                        ).to_be_visible()
                        measure_tool.set()
                        await expect(tool).to_be_visible()
                        await expect(prefill).to_be_hidden()
                        await page.reload()
                        await expect(tool_preparation).to_be_visible()
                        decode.set()
                        decode_rate = timer.filter(
                            has_text=re.compile(r"^Decode 30\.0 tok/s · \d+s$")
                        )
                        await expect(decode_rate).to_be_visible()
                        await expect(tool_preparation).to_have_count(0)
                        finish.set()
                        await wait_turn_completion(page)
                        await expect(decode_rate).to_be_hidden()
                        await expect(
                            page.get_by_text("Timing verified", exact=True)
                        ).to_be_visible()
                    finally:
                        advance.set()
                        measure_tool.set()
                        decode.set()
                        finish.set()
                        await browser.close()
        finally:
            server.shutdown()
            worker.join()
