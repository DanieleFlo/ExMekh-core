"""Measure the real client against one already-running DS4 server.

Run with exmekh_lib/.venv/bin/python. Each case keeps its messages and exchanges;
an intent failure is retained and rejects that sample, never treated as speedup.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


def edit_decode_measurements(log_text):
    """Count new decode tokens from progress spans, excluding restored output."""
    records = []
    new_tokens, phase, drafts = 0, None, None
    for line in log_text.splitlines():
        if "chat ctx=" in line and "prompt start" in line:
            new_tokens, phase, drafts = 0, None, None
        progress = re.search(r"chat ctx=\d+\.\.\d+:(\d+) gen=\d+ .* decoding ", line)
        if progress:
            new_tokens += int(progress[1])
        if "old_text drafts cycles=" in line:
            drafts = {name: int(value) for name, value in
                      re.findall(r"(cycles|proposed|accepted)=(\d+)", line)}
        profile = re.search(r"phase profile .* gen=(\d+) .* wall=([\d.]+)ms", line)
        if profile:
            phase = {"total_output_tokens": int(profile[1]),
                     "decode_seconds": float(profile[2]) / 1000}
        if "tool calls ctx=" in line and "names=[edit-file]" in line and phase:
            if not new_tokens or phase["decode_seconds"] <= 0:
                raise ValueError("Edit decode log has no measured tokens or duration")
            records.append({**phase, "new_decode_tokens": new_tokens,
                            "restored_output_tokens": phase["total_output_tokens"] - new_tokens,
                            "decode_tokens_per_second": new_tokens / phase["decode_seconds"],
                            "snapshot_drafts": drafts})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("html", "unicode", "old_text", "old_text_long", "old_text_cdata"), default="html")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--server-log", type=Path,
                        help="Server log with DS4_SERVER_PHASE_PROFILE=1 to measure edit decoding separately")
    parser.add_argument("--numeric-entities", action="store_true",
                        help="Spell non-ASCII characters and CR/LF as XML numeric entities in the prompt")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    os.environ["BASE_URL"] = args.base_url
    os.environ.setdefault("EXMEKH_LIB_LIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S")
    os.environ.setdefault("EXMEKH_LIB_LIVE_REQUEST_TIMEOUT_SECONDS", "300")
    # Reuse the qualified client flow, including inspect/read evidence and the
    # real interactive transport, rather than implement a second HTTP client.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exmekh_lib"))
    from tests.interactive_support import edit_fragment
    from tests.live.test_file_editing_live import EditingLiveProvider, editing_node

    html = "<html><body>\n" + "<p>Hello world</p>\n" * 14 + "</body></html>\n"
    long_line = "UNIQUE_LINE_BEGIN " + "known source text copied exactly with a target verifier. " * 24
    cdata_line = "UNIQUE_MARKUP_BEGIN " + "<p>α中🙂 & literal markup</p> " * 16
    original, old, replacement = {
        "html": (None, "<all/>", html),
        "unicode": (None, "<all/>", "α中🙂 & <literal/> </new_text></edit>\r\n"),
        "old_text": ("same α中🙂 & < ]\r\nKEEP\n", "same α中🙂 & < ]", "updated α中🙂"),
        "old_text_long": (long_line + "\r\nKEEP\n", long_line, "updated"),
        "old_text_cdata": (cdata_line + "\r\nKEEP\n", cdata_line, "updated"),
    }[args.case]
    expected = (replacement + "\r\nKEEP\n" if original is not None else replacement).encode()
    document = edit_fragment("page.html", old, replacement)
    if args.numeric_entities:
        document = "".join(f"&#{ord(character)};" if ord(character) >= 128 or character in "\r\n"
                           else character for character in document)
    records = []
    for repeat in range(args.repeats + 1):
        directory = args.work_dir / f"case-{repeat}"
        directory.mkdir()
        if original is not None:
            (directory / "page.html").write_bytes(original.encode())
        provider = EditingLiveProvider(directory)
        record = {"repeat": repeat, "warmup": repeat == 0, "case": args.case,
                  "numeric_entities": args.numeric_entities}
        started = time.monotonic()
        log_offset = args.server_log.stat().st_size if args.server_log else 0
        try:
            node, _ = editing_node(provider, directory)
            started = time.monotonic()
            prerequisites = (
                "First inspect page.html and read both physical lines in separate turns. "
                if original is not None else "The file is missing; do not inspect or read it. "
            )
            action = "Create page.html containing exactly the requested HTML. " if args.case == "html" else \
                "Apply the requested edit to page.html, preserving every byte. "
            if args.numeric_entities:
                action += "Copy the XML numeric entities exactly; they specify the requested Unicode and CR/LF bytes. "
            node.run(
                "Enter file-processing. " + action
                + prerequisites
                + "Then call edit-file exactly once with this JSON argument object, preserving every character: "
                + json.dumps({"edits": document})
                + ". After success exit_agent(text='done',items=[]), then answer done without more tools."
            )
            actual = (directory / "page.html").read_bytes()
            record.update(exact=actual == expected, bytes=len(actual),
                          output_sha256=hashlib.sha256(actual).hexdigest())
        except Exception as error:
            record.update(exact=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            if args.server_log:
                try:
                    with args.server_log.open("rb") as log:
                        log.seek(log_offset)
                        measurements = edit_decode_measurements(log.read().decode(errors="replace"))
                    if len(measurements) != 1:
                        raise ValueError(f"Expected one measured edit, got {len(measurements)}")
                    record.update(measurements[0])
                except (OSError, ValueError) as error:
                    record["measurement_error"] = f"{type(error).__name__}: {error}"
            records.append(record)
            (args.work_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            provider.get_session().close()
            provider.client.close()
            print(json.dumps(record), flush=True)
        if not record["exact"] or record.get("measurement_error"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
