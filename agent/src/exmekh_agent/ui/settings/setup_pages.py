"""Configure a Core profile and follow activation across page reloads."""

from uuid import uuid4

from nicegui import ui

from exmekh_agent.exmekh_bridge.providers.core_management import (
    Connection,
    CoreClient,
    CoreProblem,
    connect_core,
)
from exmekh_agent.ui.resources import ui_text
from exmekh_agent.ui.settings.core_navigation import (
    FEATURES,
    STEPS,
    chat_destination,
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
    with page(
        STEPS[step],
        hardware_summary(host) if step == "model" else selected["name"],
        destination,
    ):
        with ui.row().classes(
            STYLES["classes.w_full_items_center_gap_2_text_sm_opacity_80"]
        ):
            for key, title in STEPS.items():
                if list(STEPS).index(key) < list(STEPS).index(step):
                    ui.link(title, route(f"/setup/{key}", destination, **params))
                else:
                    ui.label(title).classes(
                        "font-semibold" if key == step else "opacity-60"
                    )
        if step == "model":

            def render(jobs, data):
                for item in data["models"]:

                    def choose(item=item) -> None:
                        options = (
                            params if item["id"] == model else {"model": item["id"]}
                        )
                        ui.navigate.to(route("/setup/runtime", destination, **options))

                    row_for_model(client, item, jobs, choose=choose)

            job_poll(client, render, catalog=True)
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
            for key, item in choices.items():
                with ui.card().classes(STYLES["classes.w_full"]):
                    limit = item["estimated_max_context_tokens"]
                    block(
                        item["label"],
                        f"Estimated maximum · {limit:,} tokens · "
                        f"Recommended {item['recommended_context_tokens']:,} tokens"
                        if limit
                        else item["reason"],
                    )
            selection = ui.select(eligible, value=mode, label="Memory mode").classes(
                STYLES["classes.w_full_max_w_xl"]
            )
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
                        ValueError(f"Enter a whole number from 1 to {limit:,} tokens.")
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

            ui.button(ui_text("system_settings.continue"), on_click=next_step).props(
                STYLES["props.unelevated_no_caps"]
            )
            return
        tokens = tokens or choices[mode]["recommended_context_tokens"]
        toggles = {}
        if step == "features":
            for item in selected["features"]:
                key = item["id"]
                compatible = item["available"] and (key == "vision" or mode == "vram")
                with ui.card().classes(STYLES["classes.w_full"]):
                    toggle = ui.checkbox(
                        FEATURES[key], value=compatible and key in features.split(",")
                    )
                    toggle.set_enabled(compatible)
                    toggles[key] = toggle
                    if not compatible:
                        ui.label(
                            item["reason"]
                            or "Unavailable for the selected memory mode."
                        ).classes(STYLES["classes.text_sm_opacity_75"])
                    elif item.get("size_bytes"):
                        caption = (
                            ui_text("system_settings.already_downloaded")
                            if item["installed"]
                            else format_file_size(item["size_bytes"])
                        )
                        ui.label(caption).classes(STYLES["classes.text_sm_opacity_75"])

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

            with ui.row():
                ui.button(
                    ui_text("system_settings.continue_with_chat_only"),
                    on_click=lambda: review(True),
                ).props(STYLES["props.outline_no_caps"])
                ui.button(
                    ui_text("system_settings.review_profile"),
                    on_click=lambda: review(False),
                ).props(STYLES["props.unelevated_no_caps"])
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
            ", ".join(FEATURES[key] for key in profile["features"]) or "Chat only",
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
                action.enable()

        label = (
            "Apply profile"
            if connection.status.chat_ready
            else "Download and start"
            if missing
            else "Start model"
        )
        action = ui.button(label, on_click=apply).props(
            STYLES["props.unelevated_no_caps"]
        )


async def startup_page(return_to: str, job: str = "", auto: int = 0) -> None:
    destination = chat_destination(return_to)
    with page(
        ui_text("system_settings.core_startup"),
        ui_text(
            "system_settings.chat_is_available_when_core_has_loaded_a_verified_language_model"
        ),
        destination,
    ):
        view = ui.column().classes(STYLES["classes.w_full_gap_4"])
        previous = None

        async def refresh() -> None:
            nonlocal previous
            try:
                client = CoreClient()
                status = await client.status()
                key = job or status.job_id
                activity = None
                if key:
                    try:
                        activity = await client.job(key)
                    except CoreProblem as error:
                        if error.status != 404:
                            raise
                        if status.job_id and status.job_id != key:
                            activity = await client.job(status.job_id)
                snapshot = (status, activity)
                if snapshot == previous:
                    return
                previous = snapshot
                if status.ready_profile:
                    client.follow(Connection("managed", status))
                    if auto and (not activity or not activity.active):
                        ui.navigate.to(destination)
                        return
                view.clear()
                with view:
                    if activity:
                        with ui.card().classes(STYLES["classes.w_full"]):
                            progress(activity, details=True)
                            job_controls(client, activity)
                    if status.error and not activity:
                        ui.label(status.error["message"]).classes(
                            STYLES["classes.text_negative"]
                        )
                    for feature, error in status.feature_errors.items():
                        block(FEATURES.get(feature, feature), error)
                    if status.ready_profile:
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
                                await client.start_job("restore")
                                ui.navigate.to(route("/startup", destination, auto=1))
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
                view.clear()
                with view:
                    block(ui_text("system_settings.reconnecting_to_core"), str(error))
                    ui.link(
                        ui_text("system_settings.connection_settings"),
                        route("/settings/connection", destination, setup=1),
                    )

        await refresh()
        ui.timer(1.5, refresh)
