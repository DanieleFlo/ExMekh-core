"""Measure the real client against one already-running DS4 server.

Run with exmekh_lib/.venv/bin/python. Each case keeps its messages and exchanges;
an intent failure is retained and rejects that sample, never treated as speedup.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("html", "unicode", "old_text"), default="html")
    parser.add_argument("--repeats", type=int, default=5)
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
    original, old, replacement = {
        "html": (None, "<all/>", html),
        "unicode": (None, "<all/>", "α中🙂 & <literal/> </new_text></edit>\r\n"),
        "old_text": ("same α中🙂 & < ]\r\nKEEP\n", "same α中🙂 & < ]", "updated α中🙂"),
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
            records.append(record)
            (args.work_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            provider.get_session().close()
            provider.client.close()
            print(json.dumps(record), flush=True)
        if not record["exact"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
