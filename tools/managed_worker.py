"""Own the native worker process, private listener and startup protocol."""

import contextlib
import http.client
import json
import os
import select
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from managed_assets import ROOT, Problem, atomic_json

STARTUP_PHASES = ("opening_model", "initializing_runtime", "ready")


class Stopped(Exception):
    """The supervisor has been asked to stop its current operation."""


@dataclass(frozen=True)
class NativeLaunch:
    argv: tuple[str, ...]
    suite_config: dict | None = None


@dataclass(frozen=True)
class NativeModel:
    name: str
    owner: str
    ready_operations: frozenset[str]


class NativeWorker:
    def __init__(self):
        self._directory = None
        self._control = None
        self._process = None

    def is_running(self):
        return self._process is not None and self._process.poll() is None

    def connect(self, timeout=30):
        """Open a connection owned by the caller."""
        if self._directory is None:
            raise OSError("native worker has no listener")
        connection = socket.socket(socket.AF_UNIX)
        try:
            connection.settimeout(timeout)
            connection.connect(str(Path(self._directory.name) / "inference.sock"))
        except BaseException:
            connection.close()
            raise
        return connection

    def start(self, launch, progress, check):
        if self._process is not None:
            raise RuntimeError("native worker already started")
        try:
            self._spawn(launch)
            self._wait_for_ready(progress, check)
            return self._discover_model()
        except BaseException:
            self.stop()
            raise

    def _spawn(self, launch):
        # Keep the listener and its configuration private, with a short Unix path.
        self._directory = tempfile.TemporaryDirectory(prefix="exmekh-")
        directory = Path(self._directory.name)
        argv = list(launch.argv)
        if launch.suite_config:
            catalog = directory / "suite.json"
            atomic_json(catalog, launch.suite_config)
            argv.extend(("--suite-config", str(catalog)))
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(directory / "inference.sock"))
            listener.listen(128)
            self._control, child = socket.socketpair()
            with child:
                environment = dict(
                    os.environ,
                    DS4_MANAGED_LISTENER_FD=str(listener.fileno()),
                    DS4_MANAGED_CONTROL_FD=str(child.fileno()),
                )
                self._process = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    env=environment,
                    pass_fds=(listener.fileno(), child.fileno()),
                )

    def _wait_for_ready(self, progress, check):
        started = time.monotonic()
        pending = b""
        announced = False
        while (elapsed := time.monotonic() - started) < 1800:
            check()
            if not announced and elapsed > 15:
                raise Problem(
                    "WORKER_PROTOCOL",
                    "The native worker did not announce managed startup. Rebuild Core and retry.",
                )
            progress(None, elapsed)
            if not select.select([self._control], [], [], 0.25)[0]:
                continue
            data = self._control.recv(1024)
            if not data:
                break
            pending += data
            if len(pending) > 4096:
                raise Problem(
                    "WORKER_PROTOCOL",
                    "The native worker returned an invalid startup event.",
                )
            while b"\n" in pending:
                event, pending = pending.split(b"\n", 1)
                phase = event.decode("ascii", "replace")
                if phase not in STARTUP_PHASES:
                    continue
                announced = True
                if phase == "ready":
                    return
                progress(phase, elapsed)
        raise Problem(
            "WORKER_START_FAILED",
            "Core could not start the model. Check the Core terminal, then retry or change the profile.",
        )

    def _discover_model(self):
        try:
            with self.connect(timeout=10) as connection:
                connection.sendall(
                    b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
                )
                with http.client.HTTPResponse(connection) as response:
                    response.begin()
                    if response.status != 200:
                        raise ValueError("unexpected HTTP status")
                    models = json.loads(response.read())["data"]
            chat = next(
                model
                for model in models
                if "chat" in model.get("ds4", {}).get("operations", ["chat"])
            )
            if not isinstance(chat["id"], str) or not chat["id"]:
                raise ValueError("invalid chat model ID")
            operations = frozenset(
                operation
                for model in models
                if (metadata := model.get("ds4", {})).get("ready") is True
                for operation in metadata.get("operations", ())
            )
            return NativeModel(chat["id"], chat.get("owned_by", ""), operations)
        except (
            OSError,
            http.client.HTTPException,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            StopIteration,
        ) as exc:
            raise Problem(
                "WORKER_DISCOVERY_FAILED",
                "The native worker did not report a usable chat model.",
            ) from exc

    def stop(self):
        if self._control is not None:
            self._control.close()
            self._control = None
        if self._process is not None:
            if self.is_running():
                with contextlib.suppress(ProcessLookupError):
                    self._process.terminate()
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None


def run_command(argv, closing):
    """Run a native build or asset compiler, including its subprocesses."""
    try:
        with subprocess.Popen(argv, cwd=ROOT, start_new_session=True) as process:
            while process.poll() is None:
                if closing.wait(0.25):
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    raise Stopped
            if process.returncode:
                raise Problem(
                    "WORKER_BUILD_FAILED",
                    "A native build or asset preparation failed. See the Core terminal.",
                )
    except FileNotFoundError as exc:
        raise Problem(
            "BUILD_TOOL_MISSING",
            "Install the native build tools listed in the README.",
        ) from exc
