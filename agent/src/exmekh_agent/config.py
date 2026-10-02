from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from exmekh_lib.core.agents.node import DefaultCapabilities

# Load .env if present in the current working directory or ExMekh Agent root
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    # Server / LLM Provider settings
    base_url: str = field(
        default_factory=lambda: os.getenv("BASE_URL", "http://localhost:8080/v1")
    )
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", "prova"))
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "Qwen3.8-27B-UD-Q4_K_S")
    )
    context_length: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_LENGTH", "22000"))
    )
    model_context_lengths: dict[str, int] = field(default_factory=dict)
    model_owners: dict[tuple[str, str], str] = field(default_factory=dict)
    api_type: str = field(
        default_factory=lambda: os.getenv("API_TYPE", "responses")
    )  # "chat_completions" or "responses"
    reasoning_effort: str = field(
        default_factory=lambda: os.getenv("REASONING_EFFORT", "xhigh")
    )  # none, low, medium, xhigh

    # Agent settings
    name_hds: str = "main"
    capabilities_path: str = field(
        default_factory=lambda: str(DefaultCapabilities().capabilities_path)
    )
    work_root: str = field(default_factory=lambda: str(Path.cwd()))
    agent_mode: str = "deep_agent"  # "deep_agent" or "quick_chat"
    coding_agent_persistent_history: bool = field(
        default_factory=lambda: (
            os.getenv("CODING_AGENT_PERSISTENT_HISTORY", "true").lower()
            in {"1", "true", "yes"}
        )
    )

    # Web UI settings
    host: str = field(
        default_factory=lambda: os.getenv("EXMEKH_AGENT_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("EXMEKH_AGENT_PORT", "8000"))
    )
    app_title: str = field(
        default_factory=lambda: os.getenv("EXMEKH_AGENT_TITLE", "ExMekh Agent")
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data"
    )

    def save_to_env(self, env_path: Path | None = None) -> None:
        """Persist current configuration to .env file in ExMekh Agent directory."""
        env_path = env_path or Path(__file__).resolve().parents[2] / ".env"
        lines: list[str] = []
        if env_path.exists():
            try:
                lines = env_path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                logger.warning(
                    "Could not read existing settings from %s: %s", env_path, error
                )
                lines = []

        env_vars = {
            "BASE_URL": self.base_url,
            "API_KEY": self.api_key,
            "DEFAULT_MODEL": self.default_model,
            "REASONING_EFFORT": str(self.reasoning_effort),
            "API_TYPE": self.api_type,
            "EXMEKH_AGENT_HOST": self.host,
            "EXMEKH_AGENT_PORT": str(self.port),
            "EXMEKH_AGENT_TITLE": self.app_title,
            "CODING_AGENT_PERSISTENT_HISTORY": str(
                self.coding_agent_persistent_history
            ).lower(),
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

        temporary_path = env_path.with_suffix(".env.tmp")
        try:
            temporary_path.write_text(
                "\n".join(new_lines) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(env_path)
        except OSError as error:
            logger.warning("Could not persist settings to %s: %s", env_path, error)
            temporary_path.unlink(missing_ok=True)

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
            self.model_owners.get(self._model_endpoint_key(model, base_url)) == "ds4.c"
            and "qwen" in model.lower()
        )

    def get_context_length(self, model: str | None = None) -> int:
        target_model = model or self.default_model
        if target_model and target_model in self.model_context_lengths:
            return self.model_context_lengths[target_model]
        if self.model_context_lengths:
            return next(iter(self.model_context_lengths.values()))
        return self.context_length

    def update(self, **kwargs: object) -> None:
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
        if self.default_model in self.model_context_lengths:
            self.context_length = self.model_context_lengths[self.default_model]
        self.save_to_env()


config = AppConfig()
