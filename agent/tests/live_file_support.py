"""Check frontend effects against actual calls, independently of model intent."""

from pathlib import Path
from xml.etree import ElementTree as ET

from exmekh_lib.core.models.edit_document import EditDocument

from exmekh_agent.exmekh_bridge.history.recorded_output import RecordedToolOutput
from exmekh_agent.exmekh_bridge.models import ChatMessage, ToolEvent


def _deletion_outcomes(
    event: ToolEvent, history: ChatMessage
) -> list[tuple[str, bool]]:
    output = history.output_for_call(event.tool_call_id)
    assert output is not None, "IMPLEMENTATION: missing deletion receipt"
    envelope = RecordedToolOutput.split_header(output)
    root = ET.fromstring(envelope[1] if envelope else output)
    outcomes = root.find("results")
    if outcomes is None:
        assert root.findtext("ok") == "false", (
            "IMPLEMENTATION: missing per-path outcomes"
        )
        return []
    observed = [
        (item.findtext("path"), item.findtext("ok") == "true") for item in outcomes
    ]
    assert [path for path, _ in observed] == event.arguments["paths"], (
        "IMPLEMENTATION: deletion receipt order differs from arguments"
    )
    assert (root.findtext("ok") == "true") == all(ok for _, ok in observed)
    return observed


def verify_file_effects(
    events: list[ToolEvent],
    root: Path,
    initial: dict[Path, bytes | None],
    *,
    history: ChatMessage | None = None,
) -> list[ToolEvent]:
    expected = {path.resolve(): content for path, content in initial.items()}
    terminal = {}
    for event in events:
        if event.is_running:
            continue
        if event.tool_name in {"new-file", "edit-file", "delete-file"}:
            terminal[event.invocation_key] = event
    calls = []
    for event in terminal.values():
        if event.execution_status == "error" and event.tool_name in {
            "new-file",
            "delete-file",
        }:
            args = event.arguments
            if isinstance(args.get("path"), str):
                path = (root / args["path"]).resolve()
                if path in expected:
                    valid = (
                        expected[path] is not None
                        if event.tool_name == "delete-file"
                        else expected[path] is None
                        and isinstance(args.get("content"), str)
                        and len(args["content"]) >= 2
                        and "\x00" not in args["content"]
                    )
                    assert not valid, (
                        f"IMPLEMENTATION: valid {event.tool_name} call failed"
                    )
        if event.execution_status != "completed":
            continue
        args = event.arguments
        if event.tool_name == "delete-file":
            assert history is not None, (
                "IMPLEMENTATION: deletion verification requires recorded outcomes"
            )
            outcomes = _deletion_outcomes(event, history)
            for resource, ok in outcomes:
                path = (root / resource).resolve()
                affected = [
                    target
                    for target in expected
                    if target == path or path in target.parents
                ]
                if ok:
                    expected.update({target: None for target in affected})
                else:
                    assert not any(
                        expected[target] is not None for target in affected
                    ), "IMPLEMENTATION: valid delete-file call failed"
            if outcomes and all(ok for _, ok in outcomes):
                calls.append(event)
            continue
        calls.append(event)
        if event.tool_name == "edit-file":
            document = EditDocument.parse(args["edits"])
            for path in {(root / item.path).resolve() for item in document.elements}:
                old = expected[path]
                assert old is not None, (
                    "IMPLEMENTATION: edit succeeded for a missing file"
                )
                text = old.decode()
                changes = []
                for item in document.elements:
                    if (root / item.path).resolve() != path:
                        continue
                    assert (
                        text.count(item.old_text) == 1 if item.old_text else not text
                    ), "IMPLEMENTATION: accepted selection is absent or ambiguous"
                    start = text.index(item.old_text)
                    changes.append(
                        (start, start + len(item.old_text), item.replacement)
                    )
                for start, end, replacement in sorted(changes, reverse=True):
                    text = text[:start] + replacement + text[end:]
                expected[path] = text.encode()
        else:
            expected[(root / args["path"]).resolve()] = args["content"].encode()
    for path, content in expected.items():
        if content is None:
            assert not path.exists(), (
                "IMPLEMENTATION: delete differs from its successful call"
            )
        else:
            assert path.exists() and path.read_bytes() == content, (
                "IMPLEMENTATION: file bytes differ from successful call arguments"
            )
    return calls
