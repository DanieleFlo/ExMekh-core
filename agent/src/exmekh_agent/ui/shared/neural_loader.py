"""A client-owned canvas animation with vocabulary loading off the UI thread."""

from __future__ import annotations

import asyncio
import logging

from nicegui import ui

from exmekh_agent.exmekh_bridge.providers.loader_tokens import sample_loader_tokens
from exmekh_agent.ui.resources import ui_text

logger = logging.getLogger(__name__)


class NeuralLoader(ui.element, component="../resources/js/neural_loader.js"):
    def __init__(self) -> None:
        super().__init__()
        self._props.update(
            active=False, tokens=[], label=ui_text("chat.input_bar.generating")
        )
        ui.timer(0, self._load_tokens, once=True)

    async def _load_tokens(self) -> None:
        try:
            tokens = await asyncio.get_running_loop().run_in_executor(
                None, sample_loader_tokens
            )
        except Exception:
            logger.debug("Unable to load the neural loader vocabulary", exc_info=True)
            return
        if not self.is_deleted:
            with self.client:
                self._props["tokens"] = list(tokens)
                self.update()

    def set_active(self, active: bool) -> None:
        self._props["active"] = active
        self.update()
