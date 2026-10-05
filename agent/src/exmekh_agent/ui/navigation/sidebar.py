from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from nicegui import ui

from exmekh_agent.exmekh_bridge.models import ChatSession, Project
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.resources import ui_text


class SidebarComponent:
    def __init__(
        self,
        on_session_switched: Callable[[str], None],
        on_new_chat: Callable[[], None],
        on_sub_agent_selected: Callable[[str], None] | None = None,
        on_new_project: Callable[[], None] | None = None,
        on_project_chat: Callable[[str], None] | None = None,
        on_project_settings: Callable[[str], None] | None = None,
    ) -> None:
        self.on_session_switched = on_session_switched
        self.on_new_chat = on_new_chat
        self.on_sub_agent_selected = on_sub_agent_selected
        self.on_new_project = on_new_project
        self.on_project_chat = on_project_chat
        self.on_project_settings = on_project_settings
        self.selected_sub_agent_id: str | None = None
        self.container: ui.column | None = None
        self.drawer: ui.left_drawer | None = None
        self._project_expansion: dict[str, bool] = {}

    def build(self) -> ui.left_drawer:
        with (
            ui.left_drawer(value=False)
            .props("show-if-above breakpoint=900")
            .classes(
                "lm-sidebar p-3 flex flex-col justify-between w-64 no-wrap"
            ) as drawer
        ):
            self.drawer = drawer
            ui.button(icon="close", on_click=drawer.hide).props(
                "flat round dense aria-label='Close projects menu'"
            ).classes("sidebar-close-button text-zinc-300")
            with ui.column().classes(
                "w-full h-full flex-grow min-h-0 flex flex-col no-wrap overflow-hidden"
            ):
                self.container = ui.column().classes(
                    "w-full h-full flex-grow min-h-0 flex flex-col no-wrap overflow-hidden max-h-[calc(100vh-110px)]"
                )
                self.render_sessions()

            # Bottom info
            with ui.column().classes(
                "w-full pt-3 border-t border-zinc-800 text-[11px] text-zinc-400 gap-1 flex-shrink-0"
            ):
                with ui.row().classes("items-center justify-between w-full"):
                    ui.label(
                        ui_text("navigation.sidebar.exmekh_lib_framework")
                    ).classes("font-semibold text-zinc-300")
                    ui.badge("v0.1.0").classes(
                        "font-mono text-[10px] border border-zinc-700 bg-zinc-800 text-zinc-400"
                    )
                ui.label(
                    ui_text("navigation.sidebar.local_ai_pair_programmer")
                ).classes("text-[10px] text-zinc-500")

        return drawer

    def _handle_new_chat(self) -> None:
        self.on_new_chat()

    def render_sessions(self) -> None:
        if not self.container:
            return

        self.container.clear()
        projects = session_manager.list_projects()

        with self.container:
            # Projects section with scrollable list
            with ui.column().classes(
                "sidebar-section-projects w-full flex flex-col min-h-0 flex-shrink-0 no-wrap"
            ):
                with ui.row().classes(
                    "sidebar-section-header items-center justify-between w-full px-2 pt-1 flex-shrink-0 no-wrap"
                ):
                    ui.label(ui_text("navigation.sidebar.projects")).classes(
                        "text-[11px] font-bold text-zinc-400 tracking-wider"
                    )
                    ui.button(icon="add", on_click=self.on_new_project).props(
                        "flat round dense size=xs aria-label='New project'"
                    ).classes("sidebar-section-btn").tooltip(
                        ui_text("navigation.sidebar.new_project")
                    )

                with ui.column().classes(
                    "sidebar-list-scroll w-full overflow-y-auto flex-grow min-h-0 gap-1 pr-1 no-wrap"
                ):
                    for project in projects:
                        self._render_project(
                            project, session_manager.list_sessions(project.id)
                        )
                    if not projects:
                        ui.label(ui_text("navigation.sidebar.no_projects_yet")).classes(
                            "text-xs text-zinc-500 italic px-2"
                        )

            # Generous spacing between PROJECTS and QUICK CHATS
            ui.element("div").classes("h-6 flex-shrink-0")

            # Quick chats section with independent scrollable list
            with ui.column().classes(
                "sidebar-section-quickchats w-full flex flex-col min-h-0 flex-grow overflow-hidden no-wrap"
            ):
                with ui.row().classes(
                    "sidebar-section-header items-center justify-between w-full px-2 pt-1 flex-shrink-0 no-wrap"
                ):
                    ui.label(ui_text("navigation.sidebar.quick_chats")).classes(
                        "text-[11px] font-bold text-zinc-400 tracking-wider"
                    )
                    ui.button(icon="add", on_click=self._handle_new_chat).props(
                        "flat round dense size=xs aria-label='New Quick Chat'"
                    ).classes("sidebar-section-btn").tooltip(
                        ui_text("navigation.sidebar.new_quick_chat")
                    )

                with ui.column().classes(
                    "sidebar-list-scroll w-full overflow-y-auto flex-grow min-h-0 gap-1 pr-1 no-wrap"
                ):
                    for session in session_manager.list_sessions(None):
                        self._render_session(session)

    def _render_project(self, project: Project, sessions: list[ChatSession]) -> None:
        is_expanded = self.is_project_expanded(project.id)
        if is_expanded:
            self.expand_project(project.id)
        with (
            ui.expansion(project.name, value=is_expanded)
            .props("dense expand-icon-class=hidden")
            .classes("sidebar-project w-full text-zinc-200")
            .tooltip(project.work_root) as expansion
        ):
            expansion.on_value_change(
                lambda e, pid=project.id: self._on_project_toggle(pid, bool(e.value))
            )
            with expansion.add_slot("header"):
                with ui.row().classes("w-full items-center gap-1.5 no-wrap"):
                    ui.icon("folder_open", size="xs").classes(
                        "sidebar-folder-icon sidebar-folder-open text-zinc-400"
                    )
                    ui.icon("folder", size="xs").classes(
                        "sidebar-folder-icon sidebar-folder-closed text-zinc-400"
                    )
                    ui.label(project.name).classes(
                        "text-xs font-medium truncate flex-grow min-w-0"
                    )
                    self._action(
                        "add",
                        "New project chat",
                        lambda _, pid=project.id: self._new_project_chat(pid),
                    )
                    with (
                        ui.button(icon="more_vert")
                        .props("flat round dense size=xs aria-label='Project actions'")
                        .classes(
                            "sidebar-project-menu-btn text-zinc-500 hover:text-zinc-200"
                        )
                    ) as menu_btn:
                        menu_btn.on(
                            "click", js_handler="(event) => event.stopPropagation()"
                        )
                        with (
                            ui.menu()
                            .props(
                                'anchor="top right" self="top left" auto-close :offset="[4, 0]"'
                            )
                            .classes(
                                "sidebar-project-menu bg-zinc-900 border border-zinc-800 text-zinc-200 shadow-xl py-1 rounded-md"
                            )
                        ):
                            with (
                                ui.menu_item(
                                    on_click=lambda _, pid=project.id: (
                                        self._open_project_settings(pid)
                                    )
                                )
                                .props("v-close-popup aria-label='Project settings'")
                                .classes(
                                    "sidebar-project-menu-item flex flex-row items-center gap-2.5 px-3 py-1.5 min-h-0 text-xs hover:bg-zinc-800 cursor-pointer"
                                )
                            ):
                                ui.icon("settings", size="16px").classes(
                                    "sidebar-project-menu-icon text-zinc-400 shrink-0"
                                )
                                ui.label(
                                    ui_text("navigation.sidebar.project_settings")
                                ).classes("self-center leading-none text-zinc-200")

                            with (
                                ui.menu_item(
                                    on_click=lambda _, pid=project.id: (
                                        self._confirm_delete_project(pid)
                                    )
                                )
                                .props("v-close-popup aria-label='Delete project'")
                                .classes(
                                    "sidebar-project-menu-item flex flex-row items-center gap-2.5 px-3 py-1.5 min-h-0 text-xs hover:bg-zinc-800 cursor-pointer text-rose-400"
                                )
                            ):
                                ui.icon("delete_outline", size="16px").classes(
                                    "sidebar-project-menu-icon text-rose-400 shrink-0"
                                )
                                ui.label(
                                    ui_text("navigation.sidebar.delete_project")
                                ).classes("self-center leading-none text-rose-400")
            with ui.column().classes("w-full gap-1 pl-2"):
                for session in sessions:
                    self._render_session(session)
                if not sessions:
                    ui.label(
                        ui_text("navigation.sidebar.no_conversations_yet")
                    ).classes("text-xs text-zinc-500 italic")

    @staticmethod
    def _action(
        icon: str, label: str, callback: Callable, *, extra_classes: str = ""
    ) -> None:
        classes = f"sidebar-action-btn text-zinc-400 hover:text-zinc-100 {extra_classes}".strip()
        ui.button(icon=icon, on_click=callback).props(
            f"flat round dense size=xs color=grey-5 aria-label='{label}'"
        ).classes(classes).tooltip(label).on(
            "click", js_handler="(event) => event.stopPropagation()"
        )

    def _render_session(self, session: ChatSession) -> None:
        is_active = session_manager.is_current_session(session.id)
        selected = (
            "sidebar-item-selected"
            if is_active and self.selected_sub_agent_id is None
            else ""
        )
        with (
            ui.card()
            .classes(f"sidebar-item w-full p-2 rounded-lg cursor-pointer {selected}")
            .props("flat tabindex=0 role=button")
            .on("click", lambda _, sid=session.id: self._select_session(sid))
            .on("keydown.enter", lambda _, sid=session.id: self._select_session(sid))
        ):
            with ui.row().classes("items-center w-full no-wrap gap-1"):
                with ui.column().classes("gap-0 min-w-0 flex-grow"):
                    ui.label(session.title).classes(
                        f"text-xs truncate w-full {'text-zinc-100 font-medium' if is_active else 'text-zinc-400'}"
                    ).tooltip(session.title)
                    date = datetime.fromtimestamp(session.updated_at).strftime(
                        "%d/%m %H:%M"
                    )
                    ui.label(
                        ui_text(
                            "navigation.sidebar.msg",
                            date=date,
                            len_session_messages=len(session.messages),
                        )
                    ).classes("text-[10px] text-zinc-500 font-mono")
                self._action(
                    "delete_outline",
                    "Delete chat",
                    lambda _, sid=session.id: self._confirm_delete(sid),
                    extra_classes="sidebar-chat-delete-btn",
                )
        if is_active:
            self._render_sub_agents(session)

    def _render_sub_agents(self, session: ChatSession) -> None:
        for child in session.sub_agent_conversations:
            color = (
                "border-sky-700/70 bg-sky-950/35"
                if child.persistent_history
                else "border-amber-700/70 bg-amber-950/30"
            )
            selected = (
                "sidebar-item-selected sub-agent-card-selected"
                if child.id == self.selected_sub_agent_id
                else ""
            )
            with (
                ui.card()
                .classes(
                    f"sidebar-item ml-3 w-[calc(100%-0.75rem)] p-1.5 rounded-md border gap-1 cursor-pointer sub-agent-card {color} {selected}"
                )
                .props("tabindex=0 role=button")
                .on(
                    "click",
                    lambda _, cid=child.id: self._select_sub_agent(cid),
                )
                .on(
                    "keydown.enter",
                    lambda _, cid=child.id: self._select_sub_agent(cid),
                )
            ):
                ui.label(f"{child.path} #{child.order}").classes(
                    "text-[10px] font-mono text-zinc-200 truncate w-full"
                )
                ui.label(
                    (
                        "History retained"
                        if child.persistent_history
                        else "History cleared"
                    )
                    + (" · running" if child.status == "active" else "")
                ).classes("text-[9px] text-zinc-400")

    def _on_project_toggle(self, project_id: str, is_expanded: bool) -> None:
        self._project_expansion[project_id] = is_expanded

    def is_project_expanded(self, project_id: str) -> bool:
        """Keep explicit user choices; expand the selected project by default."""
        current = session_manager.get_current_session()
        return self._project_expansion.get(project_id, current.project_id == project_id)

    def expand_project(self, project_id: str) -> None:
        self._project_expansion[project_id] = True

    def _select_session(self, session_id: str) -> None:
        session = session_manager.get_session(session_id)
        if session and session.project_id:
            self.expand_project(session.project_id)
        session_manager.switch_session(session_id)
        self.clear_sub_agent_selection()
        self.render_sessions()
        self.on_session_switched(session_id)

    def toggle(self) -> None:
        if self.drawer:
            self.drawer.toggle()

    def clear_sub_agent_selection(self) -> None:
        self.selected_sub_agent_id = None

    def _select_sub_agent(self, conversation_id: str) -> None:
        self.selected_sub_agent_id = conversation_id
        self.render_sessions()
        if self.on_sub_agent_selected:
            self.on_sub_agent_selected(conversation_id)

    def _delete_session(self, session_id: str) -> None:
        try:
            session_manager.delete_session(session_id)
        except (OSError, ValueError) as error:
            ui.notify(
                ui_text("navigation.sidebar.unable_to_delete_chat", error=error),
                type="negative",
            )
            return
        self.clear_sub_agent_selection()
        self.render_sessions()
        self.on_session_switched(session_manager.get_current_session().id)

    def _new_project_chat(self, project_id: str) -> None:
        self.expand_project(project_id)
        if self.on_project_chat:
            self.on_project_chat(project_id)

    def _open_project_settings(self, project_id: str) -> None:
        if self.on_project_settings:
            self.on_project_settings(project_id)

    async def _confirm_delete_project(self, project_id: str) -> None:
        project = session_manager.get_project(project_id)
        if project is None:
            return
        count = len(session_manager.list_sessions(project_id))
        if await self._confirm(
            "Delete this project?",
            f"{project.name}: all {count} conversations will be deleted.",
        ):
            try:
                session_manager.delete_project(project_id)
            except (OSError, ValueError) as error:
                ui.notify(
                    ui_text("navigation.sidebar.unable_to_delete_project", error=error),
                    type="negative",
                )
            else:
                self._project_expansion.pop(project_id, None)
            self.clear_sub_agent_selection()
            self.render_sessions()
            self.on_session_switched(session_manager.get_current_session().id)

    async def _confirm_delete(self, session_id: str) -> None:
        session = session_manager.get_session(session_id)
        if not session:
            return
        if await self._confirm("Delete this chat?", session.title):
            self._delete_session(session_id)

    @staticmethod
    async def _confirm(title: str, detail: str) -> bool:
        with ui.dialog() as dialog, ui.card().classes("bg-zinc-900 text-zinc-100"):
            ui.label(title).classes("font-semibold")
            ui.label(detail).classes("text-sm text-zinc-400")
            with ui.row().classes("justify-end w-full"):
                ui.button(
                    ui_text("navigation.sidebar.cancel"),
                    on_click=lambda: dialog.submit(False),
                ).props("flat no-caps")
                ui.button(
                    ui_text("navigation.sidebar.yes_delete"),
                    on_click=lambda: dialog.submit(True),
                ).props("flat no-caps color=negative")
        return bool(await dialog)
