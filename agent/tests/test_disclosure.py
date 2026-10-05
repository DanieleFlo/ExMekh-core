from __future__ import annotations

import pytest
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.ui.shared.disclosure import Disclosure


@pytest.mark.parametrize("with_chevron", [False, True])
def test_disclosure_updates_visibility_and_notifies_its_owner(with_chevron: bool):
    client = Client(page("/disclosure-test"))
    try:
        with client:
            header = ui.row()
            panel = ui.column()
            chevron = ui.icon("expand_more") if with_chevron else None
            changes = []
            disclosure = Disclosure(
                header,
                panel,
                label="Inspect result",
                expanded=False,
                on_change=changes.append,
                chevron=chevron,
            )
            assert not panel.visible
            disclosure.toggle()
            assert panel.visible and changes == [True]
            if chevron:
                assert chevron.name == "expand_less"
            disclosure.toggle()
            assert not panel.visible and changes == [True, False]
            if chevron:
                assert chevron.name == "expand_more"
    finally:
        client.delete()
