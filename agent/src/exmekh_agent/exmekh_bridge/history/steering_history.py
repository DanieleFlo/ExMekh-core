"""Recover canonical steering references from compacted skill outputs."""

from __future__ import annotations

from collections.abc import Sequence

from exmekh_lib.config.runtime import RUNTIME

from exmekh_agent.exmekh_bridge.history.recorded_output import RecordedToolOutput


def direct_steering_output_ids(
    output: RecordedToolOutput | None,
    child_outputs: Sequence[RecordedToolOutput | None],
) -> list[str]:
    """Child headers already include descendants; keep only this scope's IDs."""
    descendant_ids = {
        item.id
        for child in child_outputs
        if child is not None
        for item in child.used_tools or ()
    }
    references = (output.used_tools or ()) if output is not None else ()
    return list(
        dict.fromkeys(
            item.id
            for item in references
            if item.name == RUNTIME.steering_output_name
            and item.id not in descendant_ids
        )
    )
