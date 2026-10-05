"""Capability folders stay additive across models, runtime and UI lifetimes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event, get_ident
from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.core.agents.node import DefaultCapabilities
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.agents.agent_runner import execute_agent_turn_sync
from exmekh_agent.exmekh_bridge.agents.system_prompt import load_real_hds_system_prompt
from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.exmekh_bridge.models import ChatSettings, Project, ProjectSettings
from exmekh_agent.ui.settings.capability_paths_editor import CapabilityPathsEditor
from exmekh_agent.ui.shared.folder_picker import pick_folder_dialog


def test_folder_resolution_migrates_legacy_and_preserves_default_and_project_snapshots(
    tmp_path,
):
    default = tmp_path / "default"
    extra = tmp_path / "extra"
    settings = ProjectSettings(
        name_hds="legacy-name",
        capabilities_path=str(extra),
        additional_capabilities_paths=[
            str(default),
            str(extra),
            "",
            str(extra / ".." / "extra"),
        ],
    )
    assert settings.capability_paths(default) == [str(default), str(extra)]
    migrated = settings.with_additional_capability_paths(
        settings.capability_paths(default), default
    )
    assert migrated.capabilities_path is None
    assert migrated.name_hds == "legacy-name"
    assert migrated.additional_capabilities_paths == [str(extra)]
    assert settings.capabilities_path == str(extra)
    restored = ProjectSettings.model_validate_json(migrated.model_dump_json())
    assert restored == migrated
    project = Project(work_root=str(tmp_path), settings=migrated)
    chat = ChatSettings().with_project(project)
    assert chat.capability_paths(default) == [str(default), str(extra)]
    chat.additional_capabilities_paths.append("chat-only")
    assert project.settings.additional_capabilities_paths == [str(extra)]
    assert ProjectSettings().capability_paths(default) == [str(default)]


def test_real_node_and_prompt_load_additional_folder_without_replacing_defaults(
    tmp_path,
):
    default = str(DefaultCapabilities().capabilities_path)
    extra = tmp_path / "extra" / "capabilities"
    (extra / "hds").mkdir(parents=True)
    (extra / "hds" / "extra-root.md").write_text(
        "---\ndescription: Additional root\nexit-tool: exit_agent\n---\nEXTRA_CAPABILITY_MARKER",
        encoding="utf-8",
    )
    response = MagicMock()
    response.get_message.return_value = "Completed"
    with (
        patch("exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agents.agent_runner.DeepAgent") as agent,
    ):
        agent.return_value.run.return_value = response
        assert (
            execute_agent_turn_sync(
                user_prompt="Use folders",
                model="mock",
                base_url="http://example.test/v1",
                api_key="",
                api_type="responses",
                reasoning_effort_str="none",
                agent_mode="deep_agent",
                capabilities_path_str=default,
                capabilities_paths=[default, str(extra)],
                name_hds="main",
                on_token=lambda _: None,
                on_reasoning=lambda _: None,
                on_tool_event=lambda _: None,
            )
            == "Completed"
        )
        root = agent.call_args.kwargs["node"]
        loaded = root.load_capabilities(work_root=tmp_path)
        assert loaded.get_hds("main").get_name() == "main"
        assert "EXTRA_CAPABILITY_MARKER" in loaded.get_hds("extra-root").get_hds()
    load_real_hds_system_prompt.cache_clear()
    assert "EXTRA_CAPABILITY_MARKER" in load_real_hds_system_prompt(
        name_hds="extra-root", capability_paths=(default, str(extra))
    )
    assert "EXTRA_CAPABILITY_MARKER" not in load_real_hds_system_prompt(
        name_hds="extra-root", capability_paths=(default,)
    )
    load_real_hds_system_prompt.cache_clear()


@pytest.mark.parametrize("initial", ["existing", "child", "missing"])
def test_native_picker_uses_nearest_existing_parent_and_releases_window(
    tmp_path, initial
):
    selected = tmp_path if initial == "existing" else tmp_path / "child"
    if initial == "missing":
        selected = tmp_path / "missing" / "child"
    with (
        patch("exmekh_agent.ui.shared.folder_picker.tkinter.Tk") as window,
        patch(
            "exmekh_agent.ui.shared.folder_picker.filedialog.askdirectory",
            return_value="chosen",
        ) as picker,
    ):
        assert pick_folder_dialog(selected) == "chosen"
    expected = str(tmp_path) if initial != "missing" else None
    assert picker.call_args.kwargs["initialdir"] == expected
    window.return_value.destroy.assert_called_once()


def test_native_picker_releases_window_after_dialog_error():
    with (
        patch("exmekh_agent.ui.shared.folder_picker.tkinter.Tk") as window,
        patch(
            "exmekh_agent.ui.shared.folder_picker.filedialog.askdirectory",
            side_effect=OSError("unavailable"),
        ),
    ):
        with pytest.raises(OSError):
            pick_folder_dialog()
    window.return_value.destroy.assert_called_once()


@pytest.fixture
def path_editor(tmp_path):
    client = Client(page("/capability-paths"))
    editor = CapabilityPathsEditor(str(tmp_path))
    editor.set_paths([str(tmp_path)])
    with client:
        editor.build()
    try:
        yield client, editor
    finally:
        if not client.is_deleted:
            client.delete()


async def call_in_client(client, operation):
    with client:
        return await operation()


async def test_paths_editor_add_remove_deduplicate_and_cancellation(
    path_editor, tmp_path
):
    client, editor = path_editor
    extra = str(tmp_path / "extra" / "capabilities")
    Path(extra).mkdir(parents=True)
    worker_threads = []
    caller = get_ident()

    def pick(initial):
        worker_threads.append(get_ident())
        return extra

    with (
        client,
        patch(
            "exmekh_agent.ui.settings.capability_paths_editor.pick_folder_dialog",
            side_effect=pick,
        ),
    ):
        await editor.browse()
        await editor.browse()
    assert editor.all_paths() == [str(tmp_path), extra]
    assert all(worker != caller for worker in worker_threads)
    owned = editor.additional_paths()
    owned.clear()
    assert editor.additional_paths() == [extra]
    with (
        client,
        patch(
            "exmekh_agent.ui.settings.capability_paths_editor.pick_folder_dialog",
            return_value=None,
        ),
    ):
        await editor.browse()
        editor._remove(extra)
    assert editor.all_paths() == [str(tmp_path)]


async def test_picker_failure_keeps_paths_and_allows_retry(path_editor):
    client, editor = path_editor
    with (
        client,
        patch(
            "exmekh_agent.ui.settings.capability_paths_editor.pick_folder_dialog",
            side_effect=OSError("failed"),
        ),
        patch("exmekh_agent.ui.settings.capability_paths_editor.ui.notify") as notify,
    ):
        await editor.browse()
        assert notify.call_args.kwargs["type"] == "negative"
    assert not editor._browsing
    assert editor.additional_paths() == []


@pytest.mark.parametrize("failure", [False, True])
async def test_picker_is_exclusive_and_does_not_touch_deleted_client(
    path_editor, tmp_path, failure
):
    client, editor = path_editor
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = Event()
    selected = tmp_path / "capabilities"
    selected.mkdir()

    def pick(initial):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        if failure:
            raise OSError("closed")
        return str(selected)

    with (
        patch(
            "exmekh_agent.ui.settings.capability_paths_editor.pick_folder_dialog",
            side_effect=pick,
        ) as picker,
        patch("exmekh_agent.ui.settings.capability_paths_editor.ui.notify") as notify,
    ):
        task = asyncio.create_task(call_in_client(client, editor.browse))
        try:
            await asyncio.wait_for(started.wait(), 2)
            await call_in_client(client, editor.browse)
            assert picker.call_count == 1
            client.delete()
        finally:
            release.set()
            await task
    assert not editor._browsing
    assert editor.additional_paths() == []
    notify.assert_not_called()


@pytest.mark.parametrize("select_parent", [False, True])
def test_capability_folder_reference_accepts_root_and_containing_project(
    tmp_path, select_parent
):
    root = tmp_path / "capabilities"
    root.mkdir()
    selected = tmp_path if select_parent else root
    assert CapabilityRepository.folder_reference(str(selected)) == str(root)


async def test_invalid_folder_reference_notifies_without_adding_path(
    path_editor, tmp_path
):
    client, editor = path_editor
    with (
        client,
        patch(
            "exmekh_agent.ui.settings.capability_paths_editor.pick_folder_dialog",
            return_value=str(tmp_path / "missing"),
        ),
        patch("exmekh_agent.ui.settings.capability_paths_editor.ui.notify") as notify,
    ):
        await editor.browse()
    assert notify.call_args.kwargs["type"] == "negative"
    assert editor.additional_paths() == []
    assert editor.all_paths() == [str(tmp_path)]
