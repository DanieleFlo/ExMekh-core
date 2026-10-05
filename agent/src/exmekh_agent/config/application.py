from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from dotenv import load_dotenv
from exmekh_lib.core.agents.node import DefaultCapabilities

from exmekh_agent.config.agents import DEFAULTS as AGENT_DEFAULTS
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.environment import ENVIRONMENT_FIELDS, PERSISTED_FIELDS
from exmekh_agent.config.paths import DEFAULT_DATA_DIR, DEFAULT_ENV_PATH
from exmekh_agent.config.provider import DEFAULTS as PROVIDER_DEFAULTS
from exmekh_agent.config.server import DEFAULTS as SERVER_DEFAULTS
from exmekh_agent.utils.files import atomic_write_text

# Load .env if present in the current working directory or ExMekh Agent root
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    # Server / LLM Provider settings
    base_url: str = field(
        default_factory=lambda: os.getenv(
            ENVIRONMENT_FIELDS["base_url"], PROVIDER_DEFAULTS["base_url"]
        )
    )
    api_key: str = field(
        default_factory=lambda: os.getenv(
            ENVIRONMENT_FIELDS["api_key"], PROVIDER_DEFAULTS["api_key"]
        )
    )
    default_model: str = field(
        default_factory=lambda: os.getenv(
            ENVIRONMENT_FIELDS["default_model"], PROVIDER_DEFAULTS["default_model"]
        )
    )
    context_length: int = field(
        default_factory=lambda: int(
            os.getenv(
                ENVIRONMENT_FIELDS["context_length"],
                str(PROVIDER_DEFAULTS["context_length"]),
            )
        )
    )
    model_context_lengths: dict[str, int] = field(default_factory=dict)
    model_owners: dict[tuple[str, str], str] = field(default_factory=dict)
    api_type: str = field(
        default_factory=lambda: os.getenv(
            ENVIRONMENT_FIELDS["api_type"], PROVIDER_DEFAULTS["api_type"]
        )
    )  # "chat_completions" or "responses"
    reasoning_effort: str = field(
        default_factory=lambda: os.getenv(
            ENVIRONMENT_FIELDS["reasoning_effort"],
            PROVIDER_DEFAULTS["reasoning_effort"],
        )
    )  # none, low, medium, xhigh

    # Agent settings
    name_hds: str = AGENT_DEFAULTS["name_hds"]
    capabilities_path: str = field(
        default_factory=lambda: str(DefaultCapabilities().capabilities_path)
    )
    work_root: str = field(default_factory=lambda: str(Path.cwd()))
    agent_mode: str = AGENT_DEFAULTS["agent_mode"]  # "deep_agent" or "quick_chat"
    coding_agent_persistent_history: bool = field(
        default_factory=lambda: (
            os.getenv(
                ENVIRONMENT_FIELDS["coding_agent_persistent_history"],
                str(AGENT_DEFAULTS["coding_agent_persistent_history"]).lower(),
            ).lower()
            in {"1", "true", "yes"}
        )
    )

    # Web UI settings
    host: str = field(
        default_factory=lambda: os.getenv(
            ENVIRONMENT_FIELDS["host"], SERVER_DEFAULTS["host"]
        )
    )
    port: int = field(
        default_factory=lambda: int(
            os.getenv(ENVIRONMENT_FIELDS["port"], str(SERVER_DEFAULTS["port"]))
        )
    )
    app_title: str = field(
        default_factory=lambda: os.getenv(
            ENVIRONMENT_FIELDS["app_title"], SERVER_DEFAULTS["app_title"]
        )
    )
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)

    def save_to_env(self, env_path: Path | None = None) -> bool:
        """Persist settings atomically, reporting failure without changing the old file."""
        env_path = env_path or DEFAULT_ENV_PATH
        lines: list[str] = []
        if env_path.exists():
            try:
                lines = env_path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                logger.warning(app_text("configuration.read_error"), env_path, error)
                return False

        env_vars = {
            ENVIRONMENT_FIELDS[name]: (
                str(getattr(self, name)).lower()
                if isinstance(getattr(self, name), bool)
                else str(getattr(self, name))
            )
            for name in PERSISTED_FIELDS
        }

        updated_keys: set[str] = set()
        new_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k = stripped.split("=", 1)[0].strip()
                if k in env_vars:
                    new_lines.append(f"{k}={env_vars[k]}")
                    updated_keys.add(k)
                    continue
            new_lines.append(line)

        for k, v in env_vars.items():
            if k not in updated_keys:
                new_lines.append(f"{k}={v}")

        try:
            atomic_write_text(env_path, "\n".join(new_lines) + "\n")
        except OSError as error:
            logger.warning(app_text("configuration.persist_error"), env_path, error)
            return False
        return True

    def set_model_context_length(self, model: str, length: int) -> None:
        if model and isinstance(length, int) and length > 0:
            self.model_context_lengths[model] = length
            if model == self.default_model:
                self.context_length = length

    @staticmethod
    def _model_endpoint_key(model: str, base_url: str) -> tuple[str, str]:
        return base_url.rstrip("/"), model

    def set_model_owner(self, model: str, owner: str, base_url: str) -> None:
        self.model_owners[self._model_endpoint_key(model, base_url)] = owner

    def uses_ds4_qwen_prompt(self, model: str, base_url: str) -> bool:
        return (
            self.model_owners.get(self._model_endpoint_key(model, base_url))
            == PROVIDER_DEFAULTS["ds4_owner"]
            and PROVIDER_DEFAULTS["qwen_name_fragment"] in model.lower()
        )

    def get_context_length(self, model: str | None = None) -> int:
        target_model = model or self.default_model
        if target_model and target_model in self.model_context_lengths:
            return self.model_context_lengths[target_model]
        return self.context_length

    def select_available_model(self, models: Sequence[str]) -> None:
        """Follow the loaded model in memory; only explicit settings saves persist it."""
        if not models:
            return
        if self.default_model not in models:
            self.default_model = models[0]
        self.context_length = self.get_context_length()

    def _apply_updates(self, values: dict[str, object]) -> None:
        allowed_fields = {item.name for item in fields(self)}
        for name, value in values.items():
            if name in allowed_fields:
                setattr(self, name, value)
        if self.default_model in self.model_context_lengths:
            self.context_length = self.model_context_lengths[self.default_model]

    def update(self, **kwargs: object) -> None:
        """Apply runtime settings and persist synchronously for non-UI callers."""
        self._apply_updates(kwargs)
        self.save_to_env()

    async def update_async(self, **kwargs: object) -> bool:
        """Apply settings on the caller loop; write an independent snapshot in a worker."""
        self._apply_updates(kwargs)
        snapshot = replace(self)
        return await asyncio.get_running_loop().run_in_executor(
            None, snapshot.save_to_env
        )


config = AppConfig()
