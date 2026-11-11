"""The management HTTP contract and a byte-preserving native inference relay."""

import contextlib
import hmac
import json
import logging
import select
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from managed_assets import Problem

LOG = logging.getLogger("exmekh.core.http")
MAX_REQUEST = 64 * 1024
CHUNK = 64 * 1024


class ManagedHTTP(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, core, token):
        super().__init__(address, Handler)
        self.core = core
        self.token = token


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def do_GET(self):
        self.route()

    def do_POST(self):
        self.route()

    def do_DELETE(self):
        self.route()

    def do_OPTIONS(self):
        self.route()

    def json_response(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def body_length(self):
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) > 1:
            raise Problem(
                "REQUEST_FRAMING", "Use a single Content-Length for request bodies."
            )
        length = lengths[0] if lengths else "0"
        if not length.isascii() or not length.isdecimal() or len(length) > 12:
            raise Problem("REQUEST_FRAMING", "Invalid Content-Length.")
        return int(length)

    def json_payload(self):
        length = self.body_length()
        if length > MAX_REQUEST:
            raise Problem(
                "PAYLOAD_TOO_LARGE", "The management payload is too large.", 413
            )
        if self.headers.get_content_type() != "application/json":
            raise Problem("MEDIA_TYPE", "Send an application/json payload.", 415)
        try:
            data = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeError) as exc:
            raise Problem("JSON_INVALID", "Send a valid JSON object.") from exc
        if not isinstance(data, dict):
            raise Problem("JSON_INVALID", "Send a JSON object.")
        return data

    def route(self):
        self.close_connection = True
        try:
            path = urlsplit(self.path).path
            if path.startswith("/v1/manage/"):
                self.manage(path.removeprefix("/v1/manage/"))
            elif path.startswith("/v1/") and self.path.startswith("/"):
                self.proxy()
            else:
                raise Problem("NOT_FOUND", "Endpoint not found.", 404)
        except Problem as exc:
            with contextlib.suppress(OSError):
                self.json_response(
                    exc.status, {"error": {"code": exc.code, "message": str(exc)}}
                )
        except (OSError, ValueError):
            LOG.debug("HTTP request ended", exc_info=True)

    def manage(self, path):
        # Reject browser cross-origin requests and DNS-rebinding hosts, including
        # when the loopback management API uses its local trust default.
        host = self.headers.get("Host", "").partition(":")[0].lower()
        if host not in {"127.0.0.1", "localhost"} or self.headers.get("Origin"):
            raise Problem(
                "MANAGEMENT_ORIGIN", "Use the local management endpoint.", 403
            )
        token = self.server.token
        supplied = self.headers.get("Authorization", "")
        if token and not hmac.compare_digest(
            supplied.encode(), ("Bearer " + token).encode()
        ):
            raise Problem("MANAGEMENT_AUTH", "Provide the Core management token.", 401)
        core = self.server.core
        if self.command == "GET":
            if path == "status":
                self.json_response(200, core.status())
                return
            if path == "hardware":
                self.json_response(200, core.host())
                return
            if path == "catalog":
                self.json_response(200, core.catalog())
                return
            if path == "jobs" or path.startswith("jobs/"):
                key = path.removeprefix("jobs/") if path != "jobs" else None
                self.json_response(200, core.get_jobs(key))
                return
        if self.command == "POST":
            if path == "plans":
                self.json_response(200, core.plan(self.json_payload()))
                return
            if path == "jobs":
                key = self.headers.get("Idempotency-Key")
                if key and len(key) > 128:
                    raise Problem(
                        "IDEMPOTENCY_INVALID", "The idempotency key is too long."
                    )
                self.json_response(202, core.submit(self.json_payload(), key))
                return
            if path.startswith("jobs/") and path.endswith("/actions"):
                key = path.split("/")[1]
                self.json_response(
                    200, core.action(key, self.json_payload().get("action"))
                )
                return
        raise Problem("NOT_FOUND", "Management endpoint not found.", 404)

    def proxy(self):
        length = self.body_length()
        with self.server.core.open_inference() as worker:
            omit = {"connection", "proxy-connection", "keep-alive", "expect", "upgrade"}
            omit.update(
                part.strip().lower()
                for part in self.headers.get("Connection", "").split(",")
            )
            if omit & {"host", "content-length"}:
                raise Problem(
                    "REQUEST_FRAMING",
                    "Connection cannot remove Host or Content-Length.",
                )
            lines = [f"{self.command} {self.path} HTTP/1.1", "Connection: close"]
            lines.extend(
                f"{name}: {value}"
                for name, value in self.headers.raw_items()
                if name.lower() not in omit
            )
            worker.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin1"))
            while length:
                data = self.rfile.read(min(CHUNK, length))
                if not data:
                    return
                worker.sendall(data)
                length -= len(data)
            # Relay raw native responses, including SSE and error framing. Monitor
            # client EOF even while the worker is silent so native cancellation fires.
            while not self.server.core.is_closing():
                readable, _, _ = select.select([worker, self.connection], [], [], 0.5)
                if self.connection in readable:
                    return
                if worker in readable:
                    data = worker.recv(CHUNK)
                    if not data:
                        return
                    self.connection.sendall(data)
