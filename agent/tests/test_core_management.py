"""Exercise the public Core client against real HTTP request and response objects."""

from collections import deque

import httpx
import pytest

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.providers.core_management import (
    ActiveProfile,
    CoreClient,
    CoreJob,
    CoreProblem,
    CoreStatus,
    connect_core,
    validated_url,
)
from exmekh_agent.ui.settings.core_navigation import (
    chat_destination,
    feature_matches_mode,
    gate,
    profile_link,
    route,
    settings_group,
)

BASE = "http://example.test/v1"
PROFILE = {
    "model_id": "qwen27",
    "mode_id": "vram",
    "context_tokens": 8192,
    "features": [],
    "api_model": "qwen",
    "owned_by": "ds4.c",
    "available_features": [],
    "runtime_id": "runtime",
}
STATUS = {
    "version": 1,
    "boot_id": "boot",
    "revision": 1,
    "state": "ready",
    "chat_ready": True,
    "active_profile": PROFILE,
}
JOB = {
    "id": "job-1",
    "kind": "download",
    "model_id": "qwen27",
    "state": "running",
    "phase": "downloading",
    "allowed_actions": ["pause", "cancel"],
}


@pytest.mark.parametrize(
    ("section", "group"),
    [
        ("server", "server"),
        ("connection", "server"),
        ("models", "server"),
        ("features", "server"),
        ("activity", "server"),
        ("conversations", "client"),
        ("storage", "client"),
    ],
)
def test_settings_sections_belong_to_their_group(section, group):
    assert settings_group(section) == group


@pytest.mark.parametrize(
    ("phase", "file_state", "expected"),
    [
        ("downloading", None, True),
        ("checking_profile", "queued", True),
        ("opening_model", "verified", False),
        ("opening_model", None, False),
    ],
)
def test_job_identifies_file_preparation(phase, file_state, expected):
    files = (
        [{"name": "model.gguf", "size_bytes": 1024, "state": file_state}]
        if file_state
        else []
    )
    assert (
        CoreJob.model_validate({**JOB, "phase": phase, "files": files}).preparing_files
        is expected
    )


@pytest.fixture
def server(monkeypatch):
    client_type = httpx.AsyncClient
    requests = []
    responses = deque()

    def respond(request):
        requests.append(request)
        response = responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def client(**kwargs):
        return client_type(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(config, "base_url", BASE)
    monkeypatch.setattr(config, "api_key", "key")
    monkeypatch.setattr(config, "default_model", "previous")
    monkeypatch.setattr(config, "context_length", 4096)
    monkeypatch.setattr(config, "model_context_lengths", {})
    monkeypatch.setattr(config, "server_context_lengths", {})
    monkeypatch.setattr(config, "model_owners", {})
    return requests, responses


async def test_public_operations_and_profile(server):
    requests, responses = server
    values = [
        STATUS,
        {"devices": []},
        {"models": []},
        {"models": []},
        {"id": "plan", "missing_bytes": 5},
        [{**JOB, "created_at": 1794564000.5}],
        JOB,
        JOB,
        {**JOB, "state": "failed"},
        {"requested": 1},
    ]
    responses.extend(httpx.Response(200, json=value) for value in values)
    connection = await connect_core()
    assert gate(connection, "/") is None
    assert config.default_model == "qwen"
    assert config.get_context_length() == 8192
    client = CoreClient()
    assert await client.hardware() == {"devices": []}
    assert await client.catalog() == {"models": []}
    assert await client.catalog(refresh=True) == {"models": []}
    assert requests[-1].url.params == httpx.QueryParams({"refresh": "1"})
    assert (await client.plan(PROFILE)).missing_bytes == 5
    (job,) = await client.jobs()
    assert job.active and not job.retryable
    assert job.created_at is not None and job.created_at.timestamp() == 1794564000.5
    legacy_job = await client.job(job.id)
    assert legacy_job.created_at is None
    assert legacy_job.allowed_actions == ["pause", "cancel"]
    assert (
        await client.start_job("download", model_id="qwen27", idempotency_key="same")
    ).id == job.id
    assert requests[-1].headers["Idempotency-Key"] == "same"
    failed = await client.change_job(job.id, "pause")
    assert not failed.active and failed.retryable
    assert requests[-1].url.path == "/v1/manage/jobs/job-1/actions"
    await client.stop_inference()
    assert requests[-1].url.path == "/v1/manage/inference/stop"
    assert requests[-1].method == "POST"
    assert requests[0].headers["Authorization"] == "Bearer key"


async def test_activation_prefers_the_status_snapshot_and_recovers_completed_jobs(
    server,
):
    requests, responses = server
    client = CoreClient()
    status = CoreStatus.model_validate(
        {**STATUS, "job_id": JOB["id"], "activation_job": JOB}
    )
    assert (await client.activation(status, "older")).id == JOB["id"]
    assert requests == []

    status = status.model_copy(update={"activation_job": None})
    responses.append(httpx.Response(200, json={**JOB, "state": "completed"}))
    assert (await client.activation(status, "older")).state == "completed"
    assert requests[-1].url.path == f"/v1/manage/jobs/{JOB['id']}"

    status = status.model_copy(update={"job_id": None})
    assert await client.activation(status) is None
    responses.extend([httpx.Response(404, json={}), httpx.Response(503, json={})])
    assert await client.activation(status, JOB["id"]) is None
    with pytest.raises(CoreProblem) as unavailable:
        await client.activation(status, JOB["id"])
    assert unavailable.value.status == 503


async def test_management_error_exposes_the_server_code(server):
    _, responses = server
    responses.append(
        httpx.Response(
            409,
            json={"error": {"message": "Review the plan again.", "code": "PLAN_STALE"}},
        )
    )
    with pytest.raises(CoreProblem, match="Review the plan again") as stale:
        await CoreClient().start_job("apply", plan_id="expired")
    assert stale.value.code == "PLAN_STALE"


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (httpx.Response(200, json={"version": 2}), "incompatible management protocol"),
        (httpx.Response(200, text="not JSON"), "incompatible management response"),
        (httpx.Response(401, json={"error": {"message": "hidden"}}), "Access denied"),
        (httpx.Response(409, json={"error": {"message": "Try again"}}), "Try again"),
        (httpx.Response(500, text="failure"), "did not accept"),
        (httpx.ConnectError("closed"), "unreachable"),
    ],
)
async def test_management_failures_do_not_fall_back(server, response, detail):
    requests, responses = server
    responses.append(response)
    connection = await connect_core()
    assert connection.kind == "unavailable" and detail in connection.detail
    assert gate(connection, "/") == "/settings/connection?setup=1"
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("status", "models", "kind", "detail"),
    [
        (200, [], "unavailable", "no chat model"),
        (503, [], "unavailable", "Could not discover"),
        (200, [{"id": "qwen", "context_length": 1024}], "manual", "reachable"),
    ],
)
async def test_legacy_discovery_requires_chat_model(
    server, monkeypatch, status, models, kind, detail
):
    requests, responses = server
    responses.extend(
        [httpx.Response(404, json={}), httpx.Response(status, json={"data": models})]
    )
    client = CoreClient(api_key="")
    connection = await client.connect()
    assert connection.kind == kind and detail in connection.detail
    assert requests[1].url.path == "/v1/models"
    assert "Authorization" not in requests[0].headers
    client.follow(connection)
    assert config.default_model == "previous"
    monkeypatch.setattr(config, "api_key", "")
    client.follow(connection)
    if kind == "manual":
        assert gate(connection, "/") is None
        assert config.default_model == "qwen"


@pytest.mark.parametrize(
    ("state", "destination"),
    [
        ("empty", "/setup/model"),
        ("starting", "/startup"),
        ("error", "/startup"),
    ],
)
async def test_not_ready_routes(server, state, destination):
    _, responses = server
    responses.append(
        httpx.Response(200, json={**STATUS, "chat_ready": False, "state": state})
    )
    assert gate(await connect_core(), "/") == destination
    assert config.default_model == "previous"


async def test_invalid_config_reports_unavailable(monkeypatch):
    monkeypatch.setattr(config, "base_url", "invalid")
    assert (await connect_core()).kind == "unavailable"
    with pytest.raises(CoreProblem):
        validated_url("http://host:invalid")
    assert validated_url(" http://host/v1/ ") == "http://host/v1"
    assert chat_destination("/chat/a/b") == "/chat/a/b"
    assert chat_destination("//untrusted.test") == "/"
    assert (
        route("/startup", "/chat/a/b", job="job-1")
        == "/startup?job=job-1&return_to=%2Fchat%2Fa%2Fb"
    )


def test_reconfigure_link_carries_the_selected_profile():
    assert profile_link(
        {**PROFILE, "features": ["vision"]}, "/chat/a/b", "features"
    ) == (
        "/setup/features?model=qwen27&mode=vram&tokens=8192&features=vision&return_to=%2Fchat%2Fa%2Fb"
    )


def test_active_profile_exports_only_configuration():
    profile = ActiveProfile.model_validate(
        {**PROFILE, "features": ["vision", "images"]}
    )
    assert profile.configuration() == {
        "model_id": "qwen27",
        "mode_id": "vram",
        "context_tokens": 8192,
        "features": ["images", "vision"],
    }
    assert profile.configuration(features=[])["features"] == []
    assert profile.features == ["vision", "images"]


async def test_model_capabilities_are_scoped_to_the_active_model(server):
    _, responses = server
    image = {"id": "images", "available": True, "available_modes": ["tiered"]}
    catalog = {
        "models": [
            {"id": "other", "features": []},
            {"id": "flash", "features": [image]},
        ]
    }
    responses.extend(httpx.Response(200, json=catalog) for _ in range(2))
    client = CoreClient()
    assert await client.model_capabilities("flash") == {"images": image}
    assert await client.model_capabilities("missing") == {}


def test_feature_compatibility_uses_the_catalog_mode_and_legacy_fallback():
    assert feature_matches_mode({"available_modes": ["tiered"]}, "tiered")
    assert not feature_matches_mode({"available": True, "available_modes": []}, "vram")
    assert feature_matches_mode({"available": True}, "vram")
    assert not feature_matches_mode({"available": True}, "tiered")
    assert feature_matches_mode({"available": True}, "ssd", legacy=True)
    assert not feature_matches_mode({"available": False}, "vram")
