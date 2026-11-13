"""Agent-side HTTP boundary for Core management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import httpx
from pydantic import AwareDatetime, BaseModel, Field, TypeAdapter, ValidationError

from exmekh_agent.config import config
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.providers.provider_health import check_provider_health


class CoreProblem(Exception):
    def __init__(
        self, message: str, status: int = 0, *, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class ActiveProfile(BaseModel):
    model_id: str
    mode_id: str
    context_tokens: int
    features: list[str]
    api_model: str
    owned_by: str
    available_features: list[str]
    runtime_id: str


class CoreStatus(BaseModel):
    version: Literal[1]
    boot_id: str
    revision: int
    state: Literal["empty", "starting", "error", "ready"]
    chat_ready: bool
    active_requests: int = 0
    stopping_requests: int = 0
    active_profile: ActiveProfile | None = None
    saved_profile: dict | None = None
    job_id: str | None = None
    activation_job: CoreJob | None = None
    error: dict | None = None
    feature_errors: dict[str, str] = Field(default_factory=dict)

    @property
    def ready_profile(self) -> ActiveProfile | None:
        return self.active_profile if self.chat_ready else None


class JobProgress(BaseModel):
    completed: int | None = None
    total: int | None = None
    unit: Literal["bytes", "milestones"] = "bytes"
    percent: float | None = None
    eta_seconds: int | None = None
    bytes_per_second: int | None = None
    elapsed_seconds: int | None = None


class JobFile(BaseModel):
    name: str
    feature: str | None = None
    size_bytes: int
    state: str
    progress: JobProgress | None = None


class CoreJob(BaseModel):
    id: str
    kind: str
    model_id: str
    state: str
    phase: str
    created_at: AwareDatetime | None = None
    elapsed_seconds: float = 0
    file: str = ""
    model_name: str = ""
    files: list[JobFile] = Field(default_factory=list)
    profile: dict | None = None
    progress: JobProgress | None = None
    error: dict[str, str] | None = None
    allowed_actions: list[Literal["pause", "resume", "cancel"]] = Field(
        default_factory=list
    )

    @property
    def active(self) -> bool:
        return self.state in {"queued", "running", "paused", "pausing", "cancelling"}

    @property
    def preparing_files(self) -> bool:
        return self.phase in {"downloading", "verifying"} or any(
            file.state in {"queued", "downloading", "verifying"} for file in self.files
        )

    @property
    def retryable(self) -> bool:
        return self.kind in {"download", "import"} and self.state in {
            "failed",
            "interrupted",
            "cancelled",
        }


class ActivationPlan(BaseModel):
    id: str
    missing_bytes: int


_JOB_LIST = TypeAdapter(list[CoreJob])


@dataclass(frozen=True)
class Connection:
    kind: Literal["managed", "manual", "unavailable"]
    status: CoreStatus | None = None
    detail: str = ""
    models: tuple[str, ...] = ()

    @property
    def ready_profile(self) -> ActiveProfile | None:
        return self.status.ready_profile if self.status else None


def validated_url(value: str) -> str:
    try:
        url = urlsplit(value.strip())
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.port == 0
            or any(c.isspace() for c in value.strip())
        ):
            raise ValueError
    except ValueError as exc:
        raise CoreProblem(app_text("errors_provider.core.invalid_url")) from exc
    return url.geturl().rstrip("/")


class CoreClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self._url = validated_url(config.base_url if base_url is None else base_url)
        self._key = config.api_key if api_key is None else api_key

    async def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        try:
            async with httpx.AsyncClient(timeout=12, headers=headers) as client:
                response = await client.request(
                    method, f"{self._url}/manage/{path}", json=data
                )
        except httpx.HTTPError as exc:
            raise CoreProblem(app_text("errors_provider.core.unreachable")) from exc
        try:
            result = response.json()
        except ValueError:
            result = None
        if response.status_code >= 400:
            error = result.get("error") if isinstance(result, dict) else None
            details = error if isinstance(error, dict) else {}
            message = details.get("message")
            if response.status_code in {401, 403}:
                message = app_text("errors_provider.core.access_denied")
            raise CoreProblem(
                message or app_text("errors_provider.core.request_rejected"),
                response.status_code,
                code=details.get("code"),
            )
        if result is None:
            raise CoreProblem(app_text("errors_provider.core.incompatible_response"))
        return result

    async def _read(self, method, path, schema, data=None, *, idempotency_key=None):
        try:
            payload = await self._request(
                method, path, data, idempotency_key=idempotency_key
            )
            validator = (
                schema.validate_python
                if isinstance(schema, TypeAdapter)
                else schema.model_validate
            )
            return validator(payload)
        except ValidationError as exc:
            raise CoreProblem(
                app_text("errors_provider.core.incompatible_protocol")
            ) from exc

    async def status(self) -> CoreStatus:
        return await self._read("GET", "status", CoreStatus)

    async def stop_inference(self) -> None:
        await self._request("POST", "inference/stop")

    async def hardware(self) -> dict:
        return await self._request("GET", "hardware")

    async def catalog(self, *, refresh: bool = False) -> dict:
        return await self._request("GET", "catalog?refresh=1" if refresh else "catalog")

    async def plan(self, profile: dict) -> ActivationPlan:
        return await self._read("POST", "plans", ActivationPlan, profile)

    async def jobs(self) -> list[CoreJob]:
        return await self._read("GET", "jobs", _JOB_LIST)

    async def job(self, job_id: str) -> CoreJob:
        return await self._read("GET", f"jobs/{quote(job_id, safe='')}", CoreJob)

    async def activation(
        self, status: CoreStatus, previous_job: str = ""
    ) -> CoreJob | None:
        """Use the current snapshot or look up the operation that just ended."""
        if status.activation_job is not None:
            return status.activation_job
        job_id = status.job_id or previous_job
        if not job_id:
            return None
        try:
            return await self.job(job_id)
        except CoreProblem as error:
            if error.status != 404:
                raise
            return None

    async def start_job(
        self, kind: str, *, idempotency_key: str | None = None, **options
    ) -> CoreJob:
        return await self._read(
            "POST",
            "jobs",
            CoreJob,
            {"kind": kind, **options},
            idempotency_key=idempotency_key,
        )

    async def change_job(self, job_id: str, action: str) -> CoreJob:
        return await self._read(
            "POST",
            f"jobs/{quote(job_id, safe='')}/actions",
            CoreJob,
            {"action": action},
        )

    async def connect(self) -> Connection:
        try:
            return Connection("managed", await self.status())
        except CoreProblem as error:
            if error.status != 404:
                return Connection("unavailable", detail=str(error))
        online, models = await check_provider_health(self._url, self._key)
        if online and models:
            return Connection(
                "manual",
                detail=app_text("errors_provider.core.manual_connection"),
                models=tuple(models),
            )
        if online:
            return Connection(
                "unavailable",
                detail=app_text("errors_provider.core.no_chat_model"),
            )
        return Connection(
            "unavailable",
            detail=app_text("errors_provider.core.model_discovery_failed"),
        )

    def follow(self, connection: Connection) -> None:
        """Apply discovery only while this endpoint is still selected."""
        if self._url != config.base_url.rstrip("/") or self._key != config.api_key:
            return
        profile = connection.ready_profile
        if profile:
            config.set_model_context_length(
                profile.api_model, profile.context_tokens, base_url=self._url
            )
            config.set_model_owner(profile.api_model, profile.owned_by, self._url)
            config.select_available_model([profile.api_model])
        elif connection.kind == "manual":
            config.select_available_model(connection.models)


async def connect_core() -> Connection:
    try:
        client = CoreClient()
        connection = await client.connect()
        client.follow(connection)
        return connection
    except CoreProblem as error:
        return Connection("unavailable", detail=str(error))
