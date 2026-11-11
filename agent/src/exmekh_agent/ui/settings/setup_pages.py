"""Configure a Core profile and follow activation across page reloads."""

from uuid import uuid4

from nicegui import ui

from exmekh_agent.exmekh_bridge.providers.core_management import (
    CoreClient,
    CoreProblem,
    connect_core,
)
from exmekh_agent.ui.resources import ui_text
from exmekh_agent.ui.settings.core_navigation import (
    FEATURES,
    STEPS,
    chat_destination,
    gate,
    profile_link,
    route,
)
from exmekh_agent.ui.settings.core_views import (
    STYLES,
    block,
    fact,
    hardware_summary,
    job_controls,
    job_poll,
    notify_error,
    page,
    progress,
    row_for_model,
    show_connection,
)
from exmekh_agent.utils.formatting import format_file_size


async def setup_page(
    step: str,
    return_to: str,
    model: str = "",
    mode: str = "",
    tokens: int = 0,
    features: str = "",
) -> None:
    destination = chat_destination(return_to)
    connection = await connect_core()
    if not connection.status:
        ui.navigate.to(route("/settings/connection", destination, setup=1))
        return
    client = CoreClient()
    try:
        catalog = await client.catalog()
    except CoreProblem as error:
        with page(
            ui_text("system_settings.configure_core"),
            ui_text("system_settings.catalog_unavailable"),
            destination,
        ):
            ui.label(str(error))
            ui.link(
                ui_text("system_settings.retry"), route("/setup/model", destination)
            )
        return
    models = {item["id"]: item for item in catalog["models"]}
    if step != "model" and model not in models:
        ui.navigate.to(route("/setup/model", destination))
        return
    host = catalog["hardware"]
    selected = models.get(model)
    params = {
        "model": model,
        "mode": mode,
        "tokens": tokens or None,
        "features": features,
    }
    step_order = tuple(STEPS)
    prior_steps = step_order[: step_order.index(step)]
    with page(
        ui_text("system_settings.choose_chat_model")
        if step == "model"
        else STEPS[step],
        hardware_summary(host) if step == "model" else selected["name"],
        destination,
        setup_step=step,
        setup_links={
            key: route(f"/setup/{key}", destination, **params) for key in prior_steps
        },
        connected=True,
    ):
        if step == "model":
            selected_model = model

            def advance() -> None:
                options = (
                    params if selected_model == model else {"model": selected_model}
                )
                ui.navigate.to(route("/setup/runtime", destination, **options))

            def render(jobs, data):
                nonlocal selected_model
                items = data["models"]
                available = [
                    item["id"]
                    for item in items
                    if any(mode["available"] for mode in item["modes"])
                ]
                if selected_model not in available:
                    selected_model = available[0] if available else ""
                configure.set_enabled(bool(available))
                with ui.column().classes(STYLES["classes.surface"]):
                    for item in items:

                        def choose(item_id=item["id"]) -> None:
                            nonlocal selected_model
                            selected_model = item_id

                        row_for_model(
                            client,
                            item,
                            jobs,
                            choose=choose,
                            selected=selected_model == item["id"],
                        )

            job_poll(client, render, catalog=True)
            with ui.row().classes(STYLES["classes.action_row"]):
                configure = (
                    ui.button(
                        ui_text("system_settings.configure_memory_and_context"),
                        on_click=advance,
                    )
                    .props(STYLES["props.unelevated_no_caps"])
                    .classes(STYLES["classes.primary"])
                )
                configure.disable()
            return
        choices = {item["mode_id"]: item for item in selected["modes"]}
        eligible = {
            key: item["label"] for key, item in choices.items() if item["available"]
        }
        if not eligible:
            ui.label(
                ui_text(
                    "system_settings.this_server_has_no_qualified_memory_mode_for_the_selected_model"
                )
            ).classes(STYLES["classes.text_negative"])
            ui.link(
                ui_text("system_settings.choose_another_model"),
                route("/setup/model", destination),
            )
            return
        mode = mode if mode in eligible else next(iter(eligible))
        if step == "runtime":
            with ui.column().classes(STYLES["classes.surface"]):
                for item in choices.values():
                    with ui.column().classes(STYLES["classes.item"]):
                        limit = item["estimated_max_context_tokens"]
                        block(
                            item["label"],
                            ui_text(
                                "system_settings.context_estimates",
                                limit=limit,
                                recommended=item["recommended_context_tokens"],
                            )
                            if limit
                            else item["reason"],
                        )
            selection = ui.select(
                eligible, value=mode, label=ui_text("system_settings.memory_mode")
            ).classes(STYLES["classes.w_full_max_w_xl"])
            number = ui.number(
                ui_text("system_settings.context_tokens"),
                value=tokens or choices[mode]["recommended_context_tokens"],
                min=1,
                step=1,
            )

            def recommend() -> None:
                number.value = choices[selection.value]["recommended_context_tokens"]

            selection.on_value_change(lambda _: recommend())
            with ui.row():
                ui.button(
                    ui_text("system_settings.use_recommended"), on_click=recommend
                ).props(STYLES["props.outline_no_caps"])
                ui.link(
                    ui_text("system_settings.refresh_estimate"),
                    route("/setup/runtime", destination, **params),
                ).classes(STYLES["classes.self_center"])

            def next_step() -> None:
                value = number.value
                limit = choices[selection.value]["estimated_max_context_tokens"]
                if value is None or value != int(value) or not 1 <= value <= limit:
                    notify_error(
                        ValueError(
                            ui_text("system_settings.invalid_context", limit=limit)
                        )
                    )
                    return
                ui.navigate.to(
                    route(
                        "/setup/features",
                        destination,
                        model=model,
                        mode=selection.value,
                        tokens=int(value),
                        features=features,
                    )
                )

            with ui.row().classes(STYLES["classes.action_row"]):
                ui.button(
                    ui_text("system_settings.continue"), on_click=next_step
                ).props(STYLES["props.unelevated_no_caps"]).classes(
                    STYLES["classes.primary"]
                )
            return
        tokens = tokens or choices[mode]["recommended_context_tokens"]
        toggles = {}
        if step == "features":
            with ui.column().classes(STYLES["classes.surface"]):
                for item in selected["features"]:
                    key = item["id"]
                    compatible = item["available"] and (
                        key == "vision" or mode == "vram"
                    )
                    with ui.column().classes(STYLES["classes.item"]):
                        toggle = ui.checkbox(
                            FEATURES[key],
                            value=compatible and key in features.split(","),
                        )
                        toggle.set_enabled(compatible)
                        toggles[key] = toggle
                        if not compatible:
                            caption = item["reason"] or ui_text(
                                "system_settings.unavailable_for_memory_mode"
                            )
                        elif item.get("size_bytes"):
                            caption = (
                                ui_text("system_settings.already_downloaded")
                                if item["installed"]
                                else format_file_size(item["size_bytes"])
                            )
                        else:
                            caption = ""
                        if caption:
                            ui.label(caption).classes(STYLES["classes.item_detail"])

            def review(skip: bool) -> None:
                chosen = (
                    ""
                    if skip
                    else ",".join(
                        key for key, toggle in toggles.items() if toggle.value
                    )
                )
                ui.navigate.to(
                    route(
                        "/setup/review",
                        destination,
                        model=model,
                        mode=mode,
                        tokens=tokens,
                        features=chosen,
                    )
                )

            with ui.row().classes(STYLES["classes.action_row"]):
                ui.button(
                    ui_text("system_settings.continue_with_chat_only"),
                    on_click=lambda: review(True),
                ).props(STYLES["props.outline_no_caps"])
                ui.button(
                    ui_text("system_settings.review_profile"),
                    on_click=lambda: review(False),
                ).props(STYLES["props.unelevated_no_caps"]).classes(
                    STYLES["classes.primary"]
                )
            return
        profile = {
            "model_id": model,
            "mode_id": mode,
            "context_tokens": tokens,
            "features": [key for key in features.split(",") if key in FEATURES],
        }
        fact(ui_text("system_settings.memory_mode"), choices[mode]["label"])
        fact(ui_text("system_settings.context"), f"{tokens:,} tokens")
        fact(
            ui_text("system_settings.optional_features"),
            ", ".join(FEATURES[key] for key in profile["features"])
            or ui_text("system_settings.chat_only"),
        )
        try:
            plan = await client.plan(profile)
        except CoreProblem as error:
            ui.label(str(error)).classes(STYLES["classes.text_negative"])
            ui.link(
                ui_text("system_settings.adjust_memory_and_context"),
                route("/setup/runtime", destination, **params),
            )
            return
        missing = plan.missing_bytes
        fact(ui_text("system_settings.remaining_download"), format_file_size(missing))
        block(
            ui_text("system_settings.next_startup"),
            ui_text(
                "system_settings.core_will_restore_this_profile_after_it_starts_successfully"
            ),
        )
        application_key = uuid4().hex

        async def apply() -> None:
            action.disable()
            try:
                job = await client.start_job(
                    "apply", plan_id=plan.id, idempotency_key=application_key
                )
                ui.navigate.to(route("/startup", destination, job=job.id))
            except CoreProblem as error:
                notify_error(error)
                if error.code == "PLAN_STALE":
                    action.set_visibility(False)
                    review_again.set_visibility(True)
                else:
                    action.enable()

        label_key = (
            "apply_profile"
            if connection.status.chat_ready
            else "download_and_start"
            if missing
            else "start_model"
        )
        with ui.row().classes(STYLES["classes.action_row"]):
            action = (
                ui.button(ui_text(f"system_settings.{label_key}"), on_click=apply)
                .props(STYLES["props.unelevated_no_caps"])
                .classes(STYLES["classes.primary"])
            )
            review_again = (
                ui.button(
                    ui_text("system_settings.review_updated_profile"),
                    on_click=ui.navigate.reload,
                )
                .props(STYLES["props.unelevated_no_caps"])
                .classes(STYLES["classes.primary"])
            )
            review_again.set_visibility(False)


async def startup_page(return_to: str, job: str = "") -> None:
    destination = chat_destination(return_to)
    with page(
        ui_text("system_settings.core_startup"),
        ui_text(
            "system_settings.chat_is_available_when_core_has_loaded_a_verified_language_model"
        ),
        destination,
        setup_step="startup",
        connected=False,
    ) as badge:
        view = ui.column().classes(STYLES["classes.w_full_gap_4"])
        previous = None
        tracked_job = job

        async def refresh() -> None:
            nonlocal previous, tracked_job
            try:
                client = CoreClient()
                connection = await client.connect()
                if connection.kind == "manual":
                    client.follow(connection)
                    ui.navigate.to(destination)
                    return
                status = connection.status
                if status is None:
                    raise CoreProblem(connection.detail)
                show_connection(badge, True)
                activity = await client.activation(status, tracked_job)
                tracked_job = status.job_id or tracked_job
                if status.state == "empty" and activity is None:
                    ui.navigate.to(gate(connection, destination) or destination)
                    return
                snapshot = (status, activity)
                if snapshot == previous:
                    return
                previous = snapshot
                if status.ready_profile:
                    client.follow(connection)
                    if not status.error and (
                        activity is None or activity.state == "completed"
                    ):
                        ui.navigate.to(destination)
                        return
                view.clear()
                with view:
                    if activity:
                        with ui.column().classes(STYLES["classes.progress_card"]):
                            progress(activity, details=True)
                            job_controls(client, activity)
                    if status.error and not activity:
                        ui.label(status.error["message"]).classes(
                            STYLES["classes.text_negative"]
                        )
                    for feature, error in status.feature_errors.items():
                        block(FEATURES.get(feature, feature), error)
                    if status.ready_profile and (
                        activity is None or not activity.active
                    ):
                        ui.button(
                            ui_text("system_settings.open_chat"),
                            on_click=lambda: ui.navigate.to(destination),
                        ).props(STYLES["props.unelevated_no_caps"])
                    if (
                        status.saved_profile
                        and not status.job_id
                        and not status.chat_ready
                    ):

                        async def restore() -> None:
                            try:
                                restore_job = await client.start_job("restore")
                                ui.navigate.to(
                                    route("/startup", destination, job=restore_job.id)
                                )
                            except CoreProblem as error:
                                notify_error(error)

                        ui.button(
                            ui_text("system_settings.retry_saved_profile"),
                            on_click=restore,
                        ).props(STYLES["props.outline_no_caps"])
                    if not activity or not activity.active:
                        retry_profile = (
                            (activity.profile if activity else None)
                            or status.saved_profile
                            or {}
                        )
                        ui.link(
                            ui_text("system_settings.configure_startup"),
                            profile_link(retry_profile, destination),
                        )
                    ui.link(
                        ui_text("system_settings.activity_details"),
                        route("/settings/activity", destination),
                    )
            except CoreProblem as error:
                previous = None
                show_connection(badge, False)
                view.clear()
                with view:
                    block(ui_text("system_settings.reconnecting_to_core"), str(error))
                    ui.link(
                        ui_text("system_settings.connection_settings"),
                        route("/settings/connection", destination, setup=1),
                    )

        await refresh()
        ui.timer(1.5, refresh)
