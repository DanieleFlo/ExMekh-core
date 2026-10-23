"""Check frontend effects against actual calls, independently of model intent."""

from pathlib import Path
from xml.etree import ElementTree as ET

from exmekh_lib.core.models.creation_document import CreationDocument, CreationElement
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.utils.tool_errors import ToolInputError

from exmekh_agent.exmekh_bridge.history.recorded_output import RecordedToolOutput
from exmekh_agent.exmekh_bridge.models import ChatMessage, ToolEvent


def _operation_outcomes(event: ToolEvent, history: ChatMessage) -> list[ET.Element]:
    output = history.output_for_call(event.tool_call_id)
    assert output is not None, "IMPLEMENTATION: missing file operation receipt"
    envelope = RecordedToolOutput.split_header(output)
    root = ET.fromstring(envelope[1] if envelope else output)
    outcomes = root.find("results")
    if outcomes is None:
        assert root.findtext("ok") == "false", (
            "IMPLEMENTATION: missing per-resource outcomes"
        )
        return []
    assert (root.findtext("ok") == "true") == all(
        item.findtext("ok") == "true" for item in outcomes
    )
    return list(outcomes)


def _deletion_outcomes(
    event: ToolEvent, history: ChatMessage
) -> list[tuple[str, bool]]:
    observed = [
        (item.findtext("path"), item.findtext("ok") == "true")
        for item in _operation_outcomes(event, history)
    ]
    if observed:
        assert [path for path, _ in observed] == event.arguments["paths"], (
            "IMPLEMENTATION: deletion receipt order differs from arguments"
        )
    return observed


def _verify_creation(
    event: ToolEvent,
    history: ChatMessage,
    root: Path,
    expected: dict[Path, bytes | None],
) -> bool:
    outcomes = _operation_outcomes(event, history)
    try:
        elements = CreationDocument.parse(event.arguments["document"]).elements
    except ToolInputError:
        assert not outcomes, "IMPLEMENTATION: invalid document has resource outcomes"
        return False
    assert outcomes, "IMPLEMENTATION: valid document has no resource outcomes"
    assert [item.findtext("index") for item in outcomes] == [
        str(index) for index in range(1, len(elements) + 1)
    ], "IMPLEMENTATION: creation receipt order differs from document"
    for element, outcome in zip(elements, outcomes, strict=True):
        ok = outcome.findtext("ok") == "true"
        if not isinstance(element, CreationElement):
            assert not ok, "IMPLEMENTATION: malformed creation block succeeded"
            continue
        assert outcome.findtext("path") == element.path
        path = (root / element.path).resolve()
        if ok:
            if element.resource_type == "directory":
                assert path.is_dir(), "IMPLEMENTATION: successful directory is missing"
            else:
                assert expected.get(path) is None, (
                    "IMPLEMENTATION: creation overwrote an existing file"
                )
                expected[path] = element.content.encode()
        elif element.resource_type == "file" and path in expected:
            valid = (
                expected[path] is None
                and path.parent.is_dir()
                and len(element.content) >= 2
                and "\x00" not in element.content
            )
            assert not valid, "IMPLEMENTATION: valid new-file call failed"
    return all(item.findtext("ok") == "true" for item in outcomes)


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
        if event.tool_name == "new-file":
            assert history is not None, (
                "IMPLEMENTATION: creation verification requires recorded outcomes"
            )
            if _verify_creation(event, history, root, expected):
                calls.append(event)
            continue
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
