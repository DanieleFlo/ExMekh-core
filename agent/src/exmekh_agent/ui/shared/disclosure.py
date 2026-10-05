"""Keyboard and mouse interaction shared by expandable execution rows."""

import json
from collections.abc import Callable

from nicegui import ui


class DisclosureState:
    """Keep user expansion choices outside execution models and stream snapshots."""

    def __init__(self) -> None:
        self._expanded: set[str] = set()

    def is_expanded(self, key: str) -> bool:
        return key in self._expanded

    def set_expanded(self, key: str, expanded: bool) -> None:
        if expanded:
            self._expanded.add(key)
        else:
            self._expanded.discard(key)


class Disclosure:
    def __init__(
        self,
        header: ui.element,
        panel: ui.element,
        *,
        label: str,
        expanded: bool,
        on_change: Callable[[bool], None],
        chevron: ui.icon | None = None,
    ) -> None:
        self._header = header
        self._panel = panel
        self._on_change = on_change
        self._chevron = chevron
        self._expanded = expanded
        header.props(f"role=button tabindex=0 aria-label={json.dumps(label)}")
        header.on("click", self.toggle)
        header.on("keydown.enter", self.toggle)
        header.on(
            "keydown.space",
            self.toggle,
            js_handler="event => { event.preventDefault(); emit(); }",
        )
        self._render()

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._on_change(self._expanded)
        self._render()

    def _render(self) -> None:
        self._panel.set_visibility(self._expanded)
        self._header.props(f"aria-expanded={str(self._expanded).lower()}")
        if self._chevron:
            self._chevron.name = "expand_less" if self._expanded else "expand_more"
