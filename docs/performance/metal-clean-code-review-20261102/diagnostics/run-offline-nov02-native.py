"""Run reduced native fixtures sequentially; no GGUF or live server."""
import hashlib
import json
import subprocess
from pathlib import Path

commands = [
    ["tests/test_qwen35_metal_graph"],
    ["tests/test_qwen35_metal_features"],
    ["tests/test_flash_metal_stream_graph"],
    ["tests/test_qwen_metal_lifecycle", "qwen27"],
    ["tests/test_qwen_metal_lifecycle", "qwen27-stream"],
    ["tests/test_qwen_metal_lifecycle", "flash"],
]
results = []
for command in commands:
    print("RUN", *command, flush=True)
    process = subprocess.run(command, check=False, timeout=90)
    results.append({"command": command, "exit_code": process.returncode,
                    "executable_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()})
    if process.returncode:
        break
Path(__file__).with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
raise SystemExit(0 if len(results) == len(commands) and all(r["exit_code"] == 0 for r in results) else 1)
