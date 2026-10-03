#!/usr/bin/env python3
"""Run the outstanding Qwen27 CUDA vision gates serially on one GPU.

Uses the real photo fixture, the EXMEKH-AGENT-installed ExMekh-Lib wheel and existing native
checkpoint gates. No model download or frontend conversation state is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gguf/Qwen3.8-27B-UD-Q4_K_S.gguf"
VISION = "gguf/mmproj-Qwen3.8-27B-F16.gguf"
PHOTO = ROOT / "tests/vision-fixtures/glm53/earth.jpg"


class Acceptance:
    def __init__(self, directory: Path, port: int) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.port = port
        if (directory / "results.json").exists():
            raise FileExistsError("Use a fresh --out-dir to preserve previous results")
        self.results: list[dict] = []
        self.environment = dict(os.environ, DS4_VISION_TIMING="1")
        self.uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")

    def save(self) -> None:
        (self.directory / "results.json").write_text(
            json.dumps(self.results, indent=2) + "\n"
        )

    def command(
        self,
        name: str,
        command: list[str],
        *,
        cwd: Path = ROOT,
        stdin: str | None = None,
        environment: dict | None = None,
    ) -> dict:
        started = time.monotonic()
        with (self.directory / f"{name}.log").open("w") as log:
            process = subprocess.run(
                command,
                cwd=cwd,
                env=environment or self.environment,
                input=stdin,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        result = {
            "case": name,
            "status": "PASS" if process.returncode == 0 else "FAIL",
            "exit_code": process.returncode,
            "seconds": time.monotonic() - started,
            "command": command,
        }
        self.results.append(result)
        self.save()
        print(name, result["status"], flush=True)
        return result

    def native_cli(self, mtp: bool) -> None:
        tag = "mtp" if mtp else "target"
        # A mirrored real photo supplies different pixels, without answer-bearing paths.
        subprocess.run(
            [
                str(ROOT / "agent/.venv/bin/python"),
                "-c",
                "from PIL import Image,ImageOps; import sys; "
                "im=Image.open(sys.argv[1]); ImageOps.mirror(im).save(sys.argv[2])",
                str(PHOTO),
                str(self.directory / "photo-b.png"),
            ],
            check=True,
        )
        photo_a = self.directory / "photo-a.jpg"
        photo_a.write_bytes(PHOTO.read_bytes())
        secret = uuid4().hex[:12]
        lines = (
            f"Remember this confidential secret: {secret}. Reply READY only.\n"
            f"/read {photo_a}\n"
            "Name the main object in the photo, and recall the confidential secret. "
            "Reply with object|secret only.\n"
            f"/read {self.directory / 'photo-b.png'}\n"
            "Without rereading files, recall the main object in the FIRST and SECOND "
            "photos and the confidential secret from before either image. "
            "Reply with first_object|second_object|secret only.\n/quit\n"
        )
        command = [
            "./ds4",
            "--cuda",
            "-m",
            MODEL,
            "--vision",
            VISION,
            "--ctx",
            "8192",
            "--prefill-chunk",
            "256",
            "--temp",
            "0",
            "--nothink",
            "-n",
            "96",
        ]
        if mtp:
            command.append("--mtp")
        result = self.command(f"native-cli-photo-memory-{tag}", command, stdin=lines)
        log = (self.directory / f"native-cli-photo-memory-{tag}.log").read_text()
        result["secret"] = secret
        result["encoded_images"] = len(
            re.findall(r"ds4: image \d+x\d+, \d+ image tokens", log)
        )
        result["recall_exact"] = bool(
            re.search(r"earth\s*\|\s*earth\s*\|\s*" + secret, log, re.I)
        )
        result["vision_parking_cycles"] = log.count("VISION_MEMORY ")
        errors = re.findall(
            r"^.*(?:failed|outside the vocabulary).*$", log, re.M | re.I
        )
        result["errors"] = errors
        if result["encoded_images"] != 2 or not result["recall_exact"] or errors:
            result["status"] = "FAIL"
        self.save()
        print(result["case"], "semantic", result["status"], flush=True)

    def server(self, mtp: bool) -> None:
        tag = "mtp" if mtp else "target"
        launcher = "agent/start-ds4-server.sh" if mtp else "start-ds4-server.sh"
        command = [
            "bash",
            launcher,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--prefill-chunk",
            "256",
            "--kv-disk-dir",
            str(self.directory / f"kv-{tag}"),
        ]
        with (self.directory / f"server-{tag}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=self.environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 180
                while True:
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{self.port}/v1/models", timeout=2
                        ) as reply:
                            json.load(reply)
                        break
                    except Exception:
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(
                                f"{tag} launcher did not start; see server log"
                            )
                        time.sleep(0.5)
                self.command(
                    f"http-vision-{tag}",
                    [
                        "python3",
                        "tests/test_qwen35_vision_api.py",
                        "--url",
                        f"http://127.0.0.1:{self.port}",
                        "--report",
                        str(self.directory / f"http-{tag}.json"),
                    ],
                )
                self.command(
                    f"exmekh-photo-tools-{tag}",
                    [
                        self.uv,
                        "run",
                        "--frozen",
                        "python",
                        "../tests/exmekh_vision_live.py",
                        "--url",
                        f"http://127.0.0.1:{self.port}/v1",
                        "--photo",
                        str(PHOTO),
                        "--report",
                        str(self.directory / f"exmekh-{tag}.json"),
                    ],
                    cwd=ROOT / "agent",
                )
                environment = dict(
                    self.environment,
                    BASE_URL=f"http://127.0.0.1:{self.port}/v1",
                    API_KEY="local-test",
                    EXMEKH_LIB_LIVE_IMAGE_PATH=str(PHOTO),
                )
                self.command(
                    f"exmekh-upstream-mixed-live-{tag}",
                    [
                        str(ROOT / "exmekh_lib/.venv/bin/python"),
                        "-m",
                        "pytest",
                        "-n",
                        "0",
                        "--run-live",
                        "tests/live/test_mixed_user_inputs_live.py",
                        "-k",
                        "not native_file",
                        "-q",
                        "--tb=short",
                    ],
                    cwd=ROOT / "exmekh_lib",
                    environment=environment,
                )
                self.command(
                    f"exmekh_agent-agentic-live-{tag}",
                    [
                        self.uv,
                        "run",
                        "--frozen",
                        "pytest",
                        "-n",
                        "0",
                        "--run-live",
                        "--no-cov",
                        "tests/test_live_e2e.py::test_live_e2e_multi_turn_execution_history_preserves_prior_tools",
                        "-q",
                        "--tb=short",
                    ],
                    cwd=ROOT / "agent",
                    environment=environment,
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        text = (self.directory / f"server-{tag}.log").read_text()
        memory = [
            {"free_before": int(a), "free_parked": int(b), "free_restored": int(c)}
            for a, b, c in re.findall(
                r"VISION_MEMORY free_before=(\d+) free_parked=(\d+) free_restored=(\d+)",
                text,
            )
        ]
        self.results.append(
            {
                "case": f"vision-vram-release-{tag}",
                "status": "PASS"
                if memory
                and all(row["free_parked"] > row["free_before"] for row in memory)
                else "FAIL",
                "cycles": memory,
                "launcher": launcher,
                "configured_context": 28000 if mtp else 22593,
            }
        )
        self.save()

    def checkpoint(self, mtp: bool) -> None:
        tag = "mtp" if mtp else "target"
        environment = dict(
            self.environment, DS4_TEST_CTX="22593", DS4_TEST_PREFIX_TOKENS="20000"
        )
        command = [
            "./tests/test_server_agentic_payload",
            MODEL,
            "--vision",
            VISION,
            "--image",
            str(PHOTO),
        ]
        if mtp:
            command.append("--mtp")
        self.command(
            f"photo-20k-skill-checkpoint-{tag}", command, environment=environment
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    acceptance = Acceptance(args.out_dir.resolve(), args.port)
    for mtp in (False, True):
        acceptance.native_cli(mtp)
        acceptance.server(mtp)
        acceptance.checkpoint(mtp)
    return int(any(result["status"] != "PASS" for result in acceptance.results))


if __name__ == "__main__":
    raise SystemExit(main())
