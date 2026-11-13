from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from nicegui import ui
from nicegui.events import GenericEventArguments

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.formatting import DEFAULTS as FORMAT_DEFAULTS
from exmekh_agent.exmekh_bridge.models import ChatSession, Project
from exmekh_agent.state.session_manager import SessionManager, session_manager
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from exmekh_agent.ui.resources import presentation_config, render_html, ui_text

SELECT_CHAT_JS = """(event) => {
    if (event.type === 'keydown' && event.target !== event.currentTarget) return;
    if (window.exmekhChatNavigation) {
        window.exmekhChatNavigation.select(event.currentTarget.dataset.chatPath);
    } else {
        emit();
    }
}"""

TRACK_PROJECT_SCROLL_JS = render_html("shared/track_project_scroll.js")

UI_PRESENTATION = presentation_config("controls/sidebar")
PROJECT_CHAT_PAGE_SIZE = 5


class SidebarComponent:
    def __init__(
        self,
        on_session_switched: Callable[[str], None],
        on_new_chat: Callable[[], None],
        on_sub_agent_selected: Callable[[str], None] | None = None,
        on_new_project: Callable[[], None] | None = None,
        on_project_chat: Callable[[str], None] | None = None,
        on_project_settings: Callable[[str], None] | None = None,
        sessions: SessionManager | None = None,
    ) -> None:
        self._sessions = sessions
        self.on_session_switched = on_session_switched
        self.on_new_chat = on_new_chat
        self.on_sub_agent_selected = on_sub_agent_selected
        self.on_new_project = on_new_project
        self.on_project_chat = on_project_chat
        self.on_project_settings = on_project_settings
        self.selected_sub_agent_id: str | None = None
        self.container: ui.column | None = None
        self.drawer: ui.left_drawer | None = None
        self._project_list: ui.column | None = None
        self._project_rail: ui.element | None = None
        self._quick_chat_list: ui.column | None = None
        self._project_expansion: dict[str, bool] = {}
        self._project_chat_limits: dict[str, int] = {}
        self._pending_project_jump: ui.expansion | None = None
        self._project_dots: dict[str, ui.element] = {}
        self._selected_project_id: str | None = None
        self._selected_session_id: str | None = None

    @property
    def sessions(self) -> SessionManager:
        return self._sessions if self._sessions is not None else session_manager

    def build(self) -> ui.left_drawer:
        with (
            ui.left_drawer(value=False)
            .props(UI_PRESENTATION["build.props.show_if_above_breakpoint_900"])
            .classes(
                UI_PRESENTATION[
                    "build.classes.lm_sidebar_p_3_flex_flex_col_justify_between_"
                ]
            ) as drawer
        ):
            self.drawer = drawer
            ui.button(
                icon=UI_PRESENTATION["build.icon.close"], on_click=drawer.hide
            ).props(
                UI_PRESENTATION[
                    "build.props.flat_round_dense_aria_label_close_projects_me"
                ]
            ).classes(
                UI_PRESENTATION["build.classes.sidebar_close_button_text_zinc_300"]
            )
            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.w_full_h_full_flex_grow_min_h_0_flex_flex_col"
                ]
            ):
                self.container = ui.column().classes(
                    UI_PRESENTATION[
                        "build.classes.w_full_h_full_flex_grow_min_h_0_flex_flex_col_2"
                    ]
                )
                self.render_sessions()

            # Bottom info
            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.w_full_pt_3_border_t_border_zinc_800_text_11p"
                ]
            ):
                with ui.row().classes(
                    UI_PRESENTATION["build.classes.items_center_justify_between_w_full"]
                ):
                    ui.label(
                        ui_text("navigation.sidebar.exmekh_lib_framework")
                    ).classes(
                        UI_PRESENTATION["build.classes.font_semibold_text_zinc_300"]
                    )
                    ui.badge(ui_text("navigation.sidebar.v0_1_0")).classes(
                        UI_PRESENTATION[
                            "build.classes.font_mono_text_10px_border_border_zinc_700_bg"
                        ]
                    )
                ui.label(
                    ui_text("navigation.sidebar.local_ai_pair_programmer")
                ).classes(UI_PRESENTATION["build.classes.text_10px_text_zinc_500"])

        return drawer

    def _handle_new_chat(self) -> None:
        self.on_new_chat()

    def render_sessions(self) -> None:
        if not self.container:
            return

        if self._project_list is None or self._quick_chat_list is None:
            self._build_session_lists()
        assert self._project_list is not None and self._quick_chat_list is not None
        assert self._project_rail is not None

        # Keep the scroll containers mounted when selection or history changes.
        projects = self.sessions.list_projects()
        current = self.sessions.get_current_session()
        if current.id != self._selected_session_id:
            self._selected_session_id = current.id
            if current.project_id is not None:
                self._selected_project_id = current.project_id
        if self._selected_project_id not in {project.id for project in projects}:
            self._selected_project_id = current.project_id
        self._pending_project_jump = None
        self._project_list.clear()
        self._project_list.props["data-visible-project"] = (
            self._selected_project_id or ""
        )
        self._project_rail.set_visibility(len(projects) > 1)
        self._project_rail.clear()
        self._project_dots.clear()
        with self._project_list:
            for project in projects:
                expansion = self._render_project(
                    project, self.sessions.list_sessions(project.id)
                )
                if len(projects) > 1:
                    with self._project_rail:
                        self._render_project_dot(project, expansion)
            if not projects:
                ui.label(ui_text("navigation.sidebar.no_projects_yet")).classes(
                    UI_PRESENTATION[
                        "render_sessions.classes.text_xs_text_zinc_500_italic_px_2"
                    ]
                )

        self._quick_chat_list.clear()
        with self._quick_chat_list:
            for session in self.sessions.list_sessions(None):
                self._render_session(session)

    def _build_session_lists(self) -> None:
        with self.container:
            # Projects section with scrollable list
            with ui.column().classes(
                UI_PRESENTATION[
                    "render_sessions.classes.sidebar_section_projects_w_full_flex_flex_col"
                ]
            ):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_sessions.classes.sidebar_section_header_items_center_justify_b"
                    ]
                ):
                    ui.label(ui_text("navigation.sidebar.projects")).classes(
                        UI_PRESENTATION[
                            "render_sessions.classes.text_11px_font_bold_text_zinc_400_tracking_wi"
                        ]
                    )
                    ui.button(
                        icon=UI_PRESENTATION["render_sessions.icon.add"],
                        on_click=self.on_new_project,
                    ).props(
                        UI_PRESENTATION[
                            "render_sessions.props.flat_round_dense_size_xs_aria_label_new_proje"
                        ]
                    ).classes(
                        UI_PRESENTATION["render_sessions.classes.sidebar_section_btn"]
                    ).tooltip(ui_text("navigation.sidebar.new_project"))

                with ui.row().classes(UI_PRESENTATION["projects.classes.content"]):
                    self._project_list = ui.column().classes(
                        UI_PRESENTATION[
                            "render_sessions.classes.sidebar_list_scroll_w_full_overflow_y_auto_fl"
                        ]
                    )
                    self._project_list.on(
                        "scroll",
                        self._on_project_scroll,
                        js_handler=TRACK_PROJECT_SCROLL_JS,
                    )
                    self._project_rail = ui.element("nav").classes(
                        UI_PRESENTATION["projects.classes.rail"]
                    )
                    self._project_rail.props["aria-label"] = ui_text(
                        "navigation.sidebar.project_navigation"
                    )

            # Generous spacing between PROJECTS and QUICK CHATS
            ui.element("div").classes(
                UI_PRESENTATION["render_sessions.classes.h_6_flex_shrink_0"]
            )

            # Quick chats section with independent scrollable list
            with ui.column().classes(
                UI_PRESENTATION[
                    "render_sessions.classes.sidebar_section_quickchats_w_full_flex_flex_c"
                ]
            ):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_sessions.classes.sidebar_section_header_items_center_justify_b"
                    ]
                ):
                    ui.label(ui_text("navigation.sidebar.quick_chats")).classes(
                        UI_PRESENTATION[
                            "render_sessions.classes.text_11px_font_bold_text_zinc_400_tracking_wi"
                        ]
                    )
                    ui.button(
                        icon=UI_PRESENTATION["render_sessions.icon.add"],
                        on_click=self._handle_new_chat,
                    ).props(
                        UI_PRESENTATION[
                            "render_sessions.props.flat_round_dense_size_xs_aria_label_new_quick"
                        ]
                    ).classes(
                        UI_PRESENTATION["render_sessions.classes.sidebar_section_btn"]
                    ).tooltip(ui_text("navigation.sidebar.new_quick_chat"))

                self._quick_chat_list = ui.column().classes(
                    UI_PRESENTATION[
                        "render_sessions.classes.sidebar_list_scroll_w_full_overflow_y_auto_fl"
                    ]
                )

    def _render_project_dot(self, project: Project, expansion: ui.expansion) -> None:
        label = ui_text("navigation.sidebar.jump_to_project", name=project.name)
        dot = (
            ui.element("button")
            .classes(UI_PRESENTATION["projects.classes.dot"])
            .props(UI_PRESENTATION["projects.props.dot"])
        )
        dot.props["aria-label"] = label
        dot.props["aria-controls"] = expansion.html_id
        self._project_dots[project.id] = dot
        dot.props["aria-current"] = str(project.id == self._selected_project_id).lower()
        dot.tooltip(project.name)
        dot.on("click", lambda: self._jump_to_project(project.id, expansion))

    def _select_project_dot(self, project_id: str) -> None:
        if self._selected_project_id == project_id:
            return
        previous = self._project_dots.get(self._selected_project_id or "")
        if previous is not None:
            previous.props["aria-current"] = "false"
        self._selected_project_id = project_id
        self._project_dots[project_id].props["aria-current"] = "true"
        if self._project_list is not None:
            self._project_list.props["data-visible-project"] = project_id

    def _on_project_scroll(self, event: GenericEventArguments) -> None:
        if isinstance(event.args, str) and event.args in self._project_dots:
            self._select_project_dot(event.args)

    def _jump_to_project(self, project_id: str, expansion: ui.expansion) -> None:
        if self._pending_project_jump is expansion:
            return
        self._select_project_dot(project_id)
        self.expand_project(project_id)
        if expansion.value:
            self._pending_project_jump = None
            self._scroll_to_project(expansion)
        else:
            self._pending_project_jump = expansion
            expansion.open()

    def _on_project_opened(self, expansion: ui.expansion) -> None:
        if self._pending_project_jump is expansion:
            self._pending_project_jump = None
            self._scroll_to_project(expansion)

    def _scroll_to_project(self, expansion: ui.expansion) -> None:
        if self._project_list is None:
            return
        ui.run_javascript(
            render_html(
                "shared/scroll_to_project.js",
                list_id=self._project_list.html_id,
                project_id=expansion.html_id,
            )
        )

    def _render_project(
        self, project: Project, sessions: list[ChatSession]
    ) -> ui.expansion:
        is_expanded = self.is_project_expanded(project.id)
        if is_expanded:
            self.expand_project(project.id)
        with (
            ui.expansion(project.name, value=is_expanded)
            .props(
                UI_PRESENTATION["render_project.props.dense_expand_icon_class_hidden"]
            )
            .classes(
                UI_PRESENTATION[
                    "render_project.classes.sidebar_project_w_full_text_zinc_200"
                ]
            )
            .tooltip(project.work_root) as expansion
        ):
            expansion.props["data-project-id"] = project.id
            expansion.on_value_change(
                lambda e, pid=project.id: self._on_project_toggle(pid, bool(e.value))
            )
            expansion.on("after-show", lambda: self._on_project_opened(expansion))
            with expansion.add_slot("header"):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_project.classes.w_full_items_center_gap_1_5_no_wrap"
                    ]
                ):
                    ui.icon(
                        UI_PRESENTATION["render_project.icon.folder_open"],
                        size=UI_PRESENTATION["render_project.size.xs"],
                    ).classes(
                        UI_PRESENTATION[
                            "render_project.classes.sidebar_folder_icon_sidebar_folder_open_text_"
                        ]
                    )
                    ui.icon(
                        UI_PRESENTATION["render_project.icon.folder"],
                        size=UI_PRESENTATION["render_project.size.xs"],
                    ).classes(
                        UI_PRESENTATION[
                            "render_project.classes.sidebar_folder_icon_sidebar_folder_closed_tex"
                        ]
                    )
                    ui.label(project.name).classes(
                        UI_PRESENTATION[
                            "render_project.classes.text_xs_font_medium_truncate_flex_grow_min_w_"
                        ]
                    )
                    self._action(
                        "add",
                        app_text("navigation.sidebar.new_project_chat"),
                        lambda _, pid=project.id: self._new_project_chat(pid),
                    )
                    with (
                        ui.button(icon=UI_PRESENTATION["render_project.icon.more_vert"])
                        .props(
                            UI_PRESENTATION[
                                "render_project.props.flat_round_dense_size_xs_aria_label_project_a"
                            ]
                        )
                        .classes(
                            UI_PRESENTATION[
                                "render_project.classes.sidebar_project_menu_btn_text_zinc_500_hover_"
                            ]
                        )
                    ) as menu_btn:
                        menu_btn.on(
                            "click", js_handler="(event) => event.stopPropagation()"
                        )
                        with (
                            ui.menu()
                            .props(
                                UI_PRESENTATION[
                                    "render_project.props.anchor_top_right_self_top_left_auto_close_off"
                                ]
                            )
                            .classes(
                                UI_PRESENTATION[
                                    "render_project.classes.sidebar_project_menu_bg_zinc_900_border_borde"
                                ]
                            )
                        ):
                            with (
                                ui.menu_item(
                                    on_click=lambda _, pid=project.id: (
                                        self._open_project_settings(pid)
                                    )
                                )
                                .props(
                                    UI_PRESENTATION[
                                        "render_project.props.v_close_popup_aria_label_project_settings"
                                    ]
                                )
                                .classes(
                                    UI_PRESENTATION[
                                        "render_project.classes.sidebar_project_menu_item_flex_flex_row_items"
                                    ]
                                )
                            ):
                                ui.icon(
                                    UI_PRESENTATION["render_project.icon.settings"],
                                    size=UI_PRESENTATION["render_project.size.16px"],
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_project.classes.sidebar_project_menu_icon_text_zinc_400_shrin"
                                    ]
                                )
                                ui.label(
                                    ui_text("navigation.sidebar.project_settings")
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_project.classes.self_center_leading_none_text_zinc_200"
                                    ]
                                )

                            with (
                                ui.menu_item(
                                    on_click=lambda _, pid=project.id: (
                                        self._confirm_delete_project(pid)
                                    )
                                )
                                .props(
                                    UI_PRESENTATION[
                                        "render_project.props.v_close_popup_aria_label_delete_project"
                                    ]
                                )
                                .classes(
                                    UI_PRESENTATION[
                                        "render_project.classes.sidebar_project_menu_item_flex_flex_row_items_2"
                                    ]
                                )
                            ):
                                ui.icon(
                                    UI_PRESENTATION[
                                        "render_project.icon.delete_outline"
                                    ],
                                    size=UI_PRESENTATION["render_project.size.16px"],
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_project.classes.sidebar_project_menu_icon_text_rose_400_shrin"
                                    ]
                                )
                                ui.label(
                                    ui_text("navigation.sidebar.delete_project")
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_project.classes.self_center_leading_none_text_rose_400"
                                    ]
                                )
            self._render_project_chats(project.id, sessions)
        return expansion

    def _render_project_chats(
        self, project_id: str, sessions: list[ChatSession]
    ) -> None:
        with ui.column().classes(
            UI_PRESENTATION["render_project.classes.w_full_gap_1_pl_2"]
        ) as chat_list:
            if not sessions:
                ui.label(ui_text("navigation.sidebar.no_conversations_yet")).classes(
                    UI_PRESENTATION[
                        "render_project.classes.text_xs_text_zinc_500_italic"
                    ]
                )
                return

        def render() -> None:
            limit = self._project_chat_limits.get(project_id, PROJECT_CHAT_PAGE_SIZE)
            current = self.sessions.get_current_session()
            visible = sessions[:limit]
            if current.project_id == project_id and all(
                chat.id != current.id for chat in visible
            ):
                visible = [*visible[:-1], current]
            remaining = len(sessions) - len(visible)
            chat_list.clear()
            with chat_list:
                for session in visible:
                    self._render_session(session)
                if remaining or limit > PROJECT_CHAT_PAGE_SIZE:
                    with ui.row().classes(
                        UI_PRESENTATION["project_chats.classes.actions"]
                    ):
                        if remaining:
                            button(
                                "show_more_chats",
                                limit + PROJECT_CHAT_PAGE_SIZE,
                                remaining=remaining,
                            )
                        if limit > PROJECT_CHAT_PAGE_SIZE:
                            button("show_fewer_chats", PROJECT_CHAT_PAGE_SIZE)

        def change_limit(limit: int) -> None:
            self._project_chat_limits[project_id] = limit
            render()

        def button(key: str, limit: int, **values) -> None:
            ui.button(
                ui_text(f"navigation.sidebar.{key}", **values),
                on_click=lambda: change_limit(limit),
            ).props(UI_PRESENTATION["project_chats.props.button"]).classes(
                UI_PRESENTATION["project_chats.classes.button"]
            )

        render()

    @staticmethod
    def _action(
        icon: str, label: str, callback: Callable, *, extra_classes: str = ""
    ) -> None:
        classes = (
            UI_PRESENTATION["action.classes"]
            .format(extra_classes=extra_classes)
            .strip()
        )
        ui.button(icon=icon, on_click=callback).props(
            UI_PRESENTATION["action.props_template"].format(label=label)
        ).classes(classes).tooltip(label).on(
            "click", js_handler="(event) => event.stopPropagation()"
        )

    def _render_session(self, session: ChatSession) -> None:
        is_active = self.sessions.is_current_session(session.id)
        selected = (
            UI_PRESENTATION["render_session.state_classes"]
            if is_active and self.selected_sub_agent_id is None
            else ""
        )
        with (
            ui.card()
            .classes(
                UI_PRESENTATION["render_session.classes_template_2"].format(
                    selected=selected
                )
            )
            .props(UI_PRESENTATION["render_session.props.flat_tabindex_0_role_button"])
            .props(f"data-chat-path={ChatRoutes(self.sessions).path_for(session)}")
            .on(
                "click",
                lambda _, sid=session.id: self._select_session(sid),
                js_handler=SELECT_CHAT_JS,
            )
            .on(
                "keydown.enter",
                lambda _, sid=session.id: self._select_session(sid),
                js_handler=SELECT_CHAT_JS,
            )
        ):
            with ui.row().classes(
                UI_PRESENTATION[
                    "render_session.classes.items_center_w_full_no_wrap_gap_1"
                ]
            ):
                with ui.column().classes(
                    UI_PRESENTATION["render_session.classes.gap_0_min_w_0_flex_grow"]
                ):
                    ui.label(session.title).classes(
                        UI_PRESENTATION["render_session.classes_template"].format(
                            state_1=UI_PRESENTATION["render_session.state_classes_3"]
                            if is_active
                            else UI_PRESENTATION["render_session.state_classes_4"]
                        )
                    ).tooltip(session.title)
                    date = datetime.fromtimestamp(session.updated_at).strftime(
                        FORMAT_DEFAULTS["sidebar_date_template"]
                    )
                    ui.label(
                        ui_text(
                            "navigation.sidebar.msg",
                            date=date,
                            len_session_messages=len(session.messages),
                        )
                    ).classes(
                        UI_PRESENTATION[
                            "render_session.classes.text_10px_text_zinc_500_font_mono"
                        ]
                    )
                self._action(
                    "delete_outline",
                    app_text("navigation.sidebar.delete_chat"),
                    lambda _, sid=session.id: self._confirm_delete(sid),
                    extra_classes=UI_PRESENTATION["render_session.state_classes_2"],
                )
        if is_active:
            self._render_sub_agents(session)

    def _render_sub_agents(self, session: ChatSession) -> None:
        for child in session.sub_agent_conversations:
            color = (
                UI_PRESENTATION["render_sub_agents.state_classes"]
                if child.persistent_history
                else UI_PRESENTATION["render_sub_agents.state_classes_2"]
            )
            selected = (
                UI_PRESENTATION["render_sub_agents.state_classes_3"]
                if child.id == self.selected_sub_agent_id
                else ""
            )
            with (
                ui.card()
                .classes(
                    UI_PRESENTATION["render_sub_agents.classes_template"].format(
                        color=color, selected=selected
                    )
                )
                .props(
                    UI_PRESENTATION["render_sub_agents.props.tabindex_0_role_button"]
                )
                .on(
                    "click",
                    lambda _, cid=child.id: self._select_sub_agent(cid),
                )
                .on(
                    "keydown.enter",
                    lambda _, cid=child.id: self._select_sub_agent(cid),
                )
            ):
                ui.label(
                    ui_text(
                        "navigation.sidebar.value",
                        value_1=child.path,
                        value_2=child.order,
                    )
                ).classes(
                    UI_PRESENTATION[
                        "render_sub_agents.classes.text_10px_font_mono_text_zinc_200_truncate_w_"
                    ]
                )
                ui.label(
                    (
                        app_text("navigation.sidebar.history_retained")
                        if child.persistent_history
                        else app_text("navigation.sidebar.history_cleared")
                    )
                    + (" · running" if child.status == "active" else "")
                ).classes(
                    UI_PRESENTATION["render_sub_agents.classes.text_9px_text_zinc_400"]
                )

    def _on_project_toggle(self, project_id: str, is_expanded: bool) -> None:
        self._project_expansion[project_id] = is_expanded

    def is_project_expanded(self, project_id: str) -> bool:
        """Keep explicit user choices; expand the selected project by default."""
        current = self.sessions.get_current_session()
        return self._project_expansion.get(project_id, current.project_id == project_id)

    def expand_project(self, project_id: str) -> None:
        self._project_expansion[project_id] = True

    def show_session(self, session: ChatSession) -> None:
        """Show the selected chat and reveal its project in the sidebar."""
        if session.project_id:
            self.expand_project(session.project_id)
        self.show_main_selection()

    def show_main_selection(self) -> None:
        self.clear_sub_agent_selection()
        self.render_sessions()

    def _select_session(self, session_id: str) -> None:
        session = self.sessions.switch_session(session_id)
        if session is not None:
            self.show_session(session)
            self.on_session_switched(session.id)

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
            self.sessions.delete_session(session_id)
        except (OSError, ValueError) as error:
            ui.notify(
                ui_text("navigation.sidebar.unable_to_delete_chat", error=error),
                type="negative",
            )
            return
        self.clear_sub_agent_selection()
        self.render_sessions()
        self.on_session_switched(self.sessions.get_current_session().id)

    def _new_project_chat(self, project_id: str) -> None:
        self.expand_project(project_id)
        if self.on_project_chat:
            self.on_project_chat(project_id)

    def _open_project_settings(self, project_id: str) -> None:
        if self.on_project_settings:
            self.on_project_settings(project_id)

    async def _confirm_delete_project(self, project_id: str) -> None:
        project = self.sessions.get_project(project_id)
        if project is None:
            return
        count = len(self.sessions.list_sessions(project_id))
        if await self._confirm(
            app_text("navigation.sidebar.delete_this_project"),
            app_text(
                "navigation.sidebar.delete_project_description",
                name=project.name,
                count=count,
            ),
        ):
            try:
                self.sessions.delete_project(project_id)
            except (OSError, ValueError) as error:
                ui.notify(
                    ui_text("navigation.sidebar.unable_to_delete_project", error=error),
                    type="negative",
                )
            else:
                self._project_expansion.pop(project_id, None)
                self._project_chat_limits.pop(project_id, None)
            self.clear_sub_agent_selection()
            self.render_sessions()
            self.on_session_switched(self.sessions.get_current_session().id)

    async def _confirm_delete(self, session_id: str) -> None:
        session = self.sessions.get_session(session_id)
        if not session:
            return
        if await self._confirm(
            app_text("navigation.sidebar.delete_this_chat"), session.title
        ):
            self._delete_session(session_id)

    @staticmethod
    async def _confirm(title: str, detail: str) -> bool:
        with (
            ui.dialog() as dialog,
            ui.card().classes(
                UI_PRESENTATION["confirm.classes.bg_zinc_900_text_zinc_100"]
            ),
        ):
            ui.label(title).classes(UI_PRESENTATION["confirm.classes.font_semibold"])
            ui.label(detail).classes(
                UI_PRESENTATION["confirm.classes.text_sm_text_zinc_400"]
            )
            with ui.row().classes(
                UI_PRESENTATION["confirm.classes.justify_end_w_full"]
            ):
                ui.button(
                    ui_text("navigation.sidebar.cancel"),
                    on_click=lambda: dialog.submit(False),
                ).props(UI_PRESENTATION["confirm.props.flat_no_caps"])
                ui.button(
                    ui_text("navigation.sidebar.yes_delete"),
                    on_click=lambda: dialog.submit(True),
                ).props(UI_PRESENTATION["confirm.props.flat_no_caps_color_negative"])
        return bool(await dialog)
