"""Configure a Core profile and follow activation across page reloads."""

from functools import partial
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
    action_button,
    action_link,
    activation_view,
    block,
    core_frame,
    fact,
    hardware_summary,
    inference_activity,
    job_controls,
    job_poll,
    notify_error,
    page,
    row_for_model,
    show_connection,
    whole_number,
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
    status = connection.status
    saved = status.saved_profile or (
        status.active_profile.model_dump() if status.active_profile else {}
    )
    if saved and not model:
        model = saved["model_id"]
        mode = saved["mode_id"]
        tokens = saved["context_tokens"]
        features = ",".join(saved["features"])
    client = CoreClient()
    try:
        catalog = await client.catalog(refresh=step != "model")
    except CoreProblem as error:
        with page(
            ui_text("system_settings.configure_core"),
            ui_text("system_settings.catalog_unavailable"),
            destination,
        ):
            ui.label(str(error))
            action_link(
                ui_text("system_settings.retry"),
                route("/setup/model", destination),
                tone="accent",
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
    available_steps = step_order if saved else step_order[: step_order.index(step)]
    step_targets = {
        key: route(f"/setup/{key}", destination, **params) for key in available_steps
    }
    with page(
        ui_text("system_settings.choose_chat_model")
        if step == "model"
        else STEPS[step],
        hardware_summary(host) if step == "model" else selected["name"],
        destination,
        setup_step=step,
        setup_links=step_targets,
        configured_setup=bool(saved),
        show_back=not saved or status.chat_ready,
        connected=True,
    ) as links:

        def update_step_links(options: dict, *, blocked: tuple[str, ...] = ()) -> None:
            for key, link in links.items():
                if key in blocked:
                    step_targets.pop(key, None)
                    link.props(STYLES["props.aria_disabled"], remove="href")
                else:
                    target = route(f"/setup/{key}", destination, **options)
                    step_targets[key] = target
                    link.props(f'href="{target}"', remove="aria-disabled")

        def follow_step(key: str) -> None:
            if target := step_targets.get(key):
                ui.navigate.to(target)

        # Use the same event queue as the draft controls when following a step.
        for key, link in links.items():
            link.on(
                "click",
                lambda _, key=key: follow_step(key),
                js_handler=(
                    "event => {"
                    "  if (event.button || event.ctrlKey || event.metaKey ||"
                    "      event.shiftKey || event.altKey) return;"
                    "  event.preventDefault(); emit();"
                    "}"
                ),
            )

        if step == "model":
            selected_model = model

            def selected_options() -> dict:
                return params if selected_model == model else {"model": selected_model}

            def advance() -> None:
                ui.navigate.to(
                    route("/setup/runtime", destination, **selected_options())
                )

            def render(jobs, inventory):
                nonlocal selected_model
                items = inventory["models"]
                available = [
                    item["id"]
                    for item in items
                    if any(mode["available"] for mode in item["modes"])
                ]
                if selected_model not in available:
                    selected_model = available[0] if available else ""
                configure.set_enabled(bool(available))
                update_step_links(
                    selected_options(), blocked=() if available else tuple(links)
                )
                with ui.column().classes(STYLES["classes.surface"]):
                    for item in items:

                        def choose(item_id=item["id"]) -> None:
                            nonlocal selected_model
                            selected_model = item_id
                            update_step_links(selected_options())

                        row_for_model(
                            client,
                            item,
                            jobs,
                            choose=choose,
                            selected=selected_model == item["id"],
                        )

            job_poll(client, render, load_inventory=client.catalog)
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
            action_link(
                ui_text("system_settings.choose_another_model"),
                route("/setup/model", destination),
            )
            return
        if mode and mode not in eligible and step != "runtime":
            ui.navigate.to(route("/setup/runtime", destination, **params))
            return
        mode = mode if mode in eligible else next(iter(eligible))

        def context_limit(mode_id: str, with_vision: bool = False) -> int:
            option = choices[mode_id]
            base = option["estimated_max_context_tokens"]
            return (
                option.get("estimated_max_context_tokens_with_vision", base)
                if with_vision
                else base
            ) or 0

        if step == "runtime":
            with_vision = "vision" in features.split(",")
            if model_limit := selected.get("max_context_tokens"):
                fact(
                    ui_text("system_settings.model_context_limit"),
                    f"{model_limit:,} tokens",
                )
            with ui.column().classes(STYLES["classes.surface"]):
                for item in choices.values():
                    with ui.column().classes(STYLES["classes.item"]):
                        limit = item["estimated_max_context_tokens"]
                        block(
                            item["label"],
                            ui_text(
                                "system_settings.estimated_context_limit"
                                if limit == item["recommended_context_tokens"]
                                else "system_settings.context_estimates",
                                limit=limit,
                                recommended=item["recommended_context_tokens"],
                            )
                            if limit
                            else item["reason"],
                        )
                        if vision_limit := item.get(
                            "estimated_max_context_tokens_with_vision"
                        ):
                            ui.label(
                                ui_text(
                                    "system_settings.vision_context_limit",
                                    limit=vision_limit,
                                )
                            ).classes(STYLES["classes.item_detail"])
            selection = ui.select(
                eligible, value=mode, label=ui_text("system_settings.memory_mode")
            ).classes(STYLES["classes.w_full_max_w_xl"])

            def recommended(mode_id: str) -> int:
                tokens = choices[mode_id]["recommended_context_tokens"]
                vision_limit = context_limit(mode_id, True) if with_vision else 0
                return min(tokens, vision_limit) if vision_limit else tokens

            presets: dict[int, ui.button] = {}
            with ui.column().classes(STYLES["classes.w_full_gap_1"]):
                number = ui.input(
                    ui_text("system_settings.context_tokens"),
                    value=str(tokens or recommended(mode)),
                ).props(STYLES["props.positive_integer"])
                max_limit = max(context_limit(key, with_vision) for key in eligible)
                with (
                    ui.row()
                    .classes(STYLES["classes.gap_2"])
                    .props(
                        f'role=group aria-label="{ui_text("system_settings.context_presets")}"'
                    ) as preset_row
                ):
                    count = 4096
                    while count <= max_limit:
                        label = ui_text(
                            "system_settings.use_context_limit", limit=count
                        )
                        button = action_button(
                            f"{count // 1024}K", partial(number.set_value, str(count))
                        ).classes(STYLES["classes.context_preset"])
                        button.props(f'aria-label="{label}"').tooltip(label)
                        presets[count] = button
                        count *= 2

            def recommend() -> None:
                number.value = str(recommended(selection.value))

            def current_options() -> dict:
                value = whole_number(number.value)
                return {
                    **params,
                    "mode": selection.value,
                    "tokens": value if value is not None and value > 0 else None,
                }

            def refresh_estimate() -> None:
                ui.navigate.to(
                    route("/setup/runtime", destination, **current_options())
                )

            def sync_context_controls() -> None:
                options = current_options()
                count = options["tokens"]
                limit = context_limit(selection.value, with_vision)
                preset_row.set_visibility(any(value <= limit for value in presets))
                for value, button in presets.items():
                    button.set_visibility(value <= limit)
                    button.props(f"aria-pressed={str(value == count).lower()}")
                blocked = (
                    ("features", "review")
                    if count is None
                    else ("review",)
                    if count > limit
                    else ()
                )
                update_step_links(options, blocked=blocked)

            previous_mode = mode

            def change_mode() -> None:
                nonlocal previous_mode
                if whole_number(number.value) == recommended(previous_mode):
                    recommend()
                previous_mode = selection.value
                sync_context_controls()

            selection.on_value_change(lambda _: change_mode())
            number.on_value_change(lambda _: sync_context_controls())
            sync_context_controls()
            with ui.row():
                action_button(ui_text("system_settings.use_recommended"), recommend)
                action_button(
                    ui_text("system_settings.refresh_estimate"), refresh_estimate
                )

            def next_step() -> None:
                options = current_options()
                value = options["tokens"]
                limit = (
                    context_limit(selection.value, True) if with_vision else 0
                ) or context_limit(selection.value)
                if value is None or value > limit:
                    notify_error(
                        ValueError(
                            ui_text("system_settings.invalid_context", limit=limit)
                        )
                    )
                    return
                ui.navigate.to(route("/setup/features", destination, **options))

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
            optional_models = {
                item["id"]: item for item in catalog.get("optional_models", ())
            }
            with ui.column().classes(STYLES["classes.surface"]):
                for item in selected["features"]:
                    key = item["id"]
                    if key not in FEATURES:
                        continue
                    available_modes = item.get("available_modes")
                    compatible = (
                        mode in available_modes
                        if available_modes is not None
                        else item["available"]
                        and (
                            context_limit(mode, True) > 0
                            if key == "vision"
                            else mode == "vram"
                        )
                    )
                    with ui.column().classes(STYLES["classes.item"]):
                        toggle = ui.checkbox(
                            FEATURES[key],
                            value=compatible and key in features.split(","),
                        )
                        toggle.set_enabled(compatible)
                        toggles[key] = toggle
                        if optional := optional_models.get(key):
                            ui.label(optional["name"]).classes(
                                STYLES["classes.item_detail"]
                            )
                        if not compatible:
                            caption = item["reason"] or ui_text(
                                "system_settings.unavailable_for_memory_mode"
                            )
                        elif item.get("size_bytes"):
                            missing = item.get("missing_bytes", item["size_bytes"])
                            if item["installed"]:
                                caption = ui_text("system_settings.already_downloaded")
                            elif missing == 0:
                                caption = ui_text(
                                    "system_settings.available_for_verification"
                                )
                            else:
                                caption = ui_text(
                                    "system_settings.feature_download_remaining",
                                    size=format_file_size(missing),
                                )
                        else:
                            caption = ""
                        if (
                            key == "vision"
                            and compatible
                            and "estimated_max_context_tokens_with_vision"
                            in choices[mode]
                        ):
                            maximum = ui_text(
                                "system_settings.vision_context_limit",
                                limit=context_limit(mode, True),
                            )
                            caption = " · ".join(
                                part for part in (caption, maximum) if part
                            )
                        if caption:
                            ui.label(caption).classes(STYLES["classes.item_detail"])

            def selected_limit() -> int:
                vision = toggles.get("vision")
                return context_limit(mode, bool(vision and vision.value))

            def selected_features() -> str:
                return ",".join(key for key, toggle in toggles.items() if toggle.value)

            def update_limit() -> None:
                limit = selected_limit()
                needs_adjustment = tokens > limit
                limit_message.set_text(
                    ui_text("system_settings.selected_context_limit", limit=limit)
                )
                limit_message.set_visibility(needs_adjustment)
                adjust.set_text(
                    ui_text("system_settings.use_context_limit", limit=limit)
                )
                adjust.set_visibility(needs_adjustment)
                review_button.set_enabled(not needs_adjustment)
                update_step_links(
                    {
                        **params,
                        "mode": mode,
                        "tokens": tokens,
                        "features": selected_features(),
                    },
                    blocked=("review",) if needs_adjustment else (),
                )

            def adjust_context() -> None:
                nonlocal tokens
                tokens = selected_limit()
                update_limit()

            def review(skip: bool) -> None:
                limit = context_limit(mode) if skip else selected_limit()
                if tokens > limit:
                    notify_error(
                        ValueError(
                            ui_text(
                                "system_settings.selected_context_limit", limit=limit
                            )
                        )
                    )
                    return
                chosen = "" if skip else selected_features()
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

            limit_message = ui.label("").props(STYLES["props.role_status"])
            adjust = action_button("", adjust_context, tone="caution")
            with ui.row().classes(STYLES["classes.action_row"]):
                action_button(
                    ui_text("system_settings.continue_with_chat_only"),
                    lambda: review(True),
                )
                review_button = (
                    ui.button(
                        ui_text("system_settings.review_profile"),
                        on_click=lambda: review(False),
                    )
                    .props(STYLES["props.unelevated_no_caps"])
                    .classes(STYLES["classes.primary"])
                )
            for toggle in toggles.values():
                toggle.on_value_change(lambda _: update_limit())
            update_limit()
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
            action_link(
                ui_text("system_settings.adjust_memory_and_context"),
                route("/setup/runtime", destination, **params),
                tone="caution",
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
        submitting = False
        expired = False
        busy = status.active_requests > 0

        def update_apply(active_requests: bool) -> None:
            nonlocal busy
            busy = active_requests
            action.set_enabled(not (busy or submitting or expired))

        async def apply() -> None:
            nonlocal submitting, expired
            if submitting or expired or busy:
                return
            submitting = True
            update_apply(busy)
            try:
                job = await client.start_job(
                    "apply", plan_id=plan.id, idempotency_key=application_key
                )
                ui.navigate.to(route("/startup", destination, job=job.id))
            except CoreProblem as error:
                notify_error(error)
                if error.code == "PLAN_STALE":
                    expired = True
                    action.set_visibility(False)
                    review_again.set_visibility(True)
                submitting = False
                update_apply(busy or error.code == "INFERENCE_BUSY")

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
        inference_activity(client, status, on_busy=update_apply)


async def startup_page(return_to: str, job: str = "") -> None:
    destination = chat_destination(return_to)
    with core_frame(
        destination, connected=False, extra_classes=STYLES["classes.activation_console"]
    ) as badge:
        view = ui.column().classes(STYLES["classes.activation_body"])
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
                    activation_view(activity, status.error)
                    if activity:
                        job_controls(client, activity)
                    for feature, error in status.feature_errors.items():
                        block(FEATURES.get(feature, feature), error)
                    if status.ready_profile and (
                        activity is None or not activity.active
                    ):
                        ui.button(
                            ui_text("system_settings.open_chat"),
                            on_click=lambda: ui.navigate.to(destination),
                        ).props(STYLES["props.unelevated_no_caps"]).classes(
                            STYLES["classes.primary"]
                        )
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

                        action_button(
                            ui_text("system_settings.retry_saved_profile"),
                            restore,
                            tone="accent",
                        )
                    if not activity or not activity.active:
                        retry_profile = (
                            (activity.profile if activity else None)
                            or status.saved_profile
                            or {}
                        )
                        action_link(
                            ui_text("system_settings.configure_startup"),
                            profile_link(retry_profile, destination),
                            tone="accent",
                        )
                    action_link(
                        ui_text("system_settings.activity_details"),
                        route("/settings/activity", destination),
                    )
            except CoreProblem as error:
                previous = None
                show_connection(badge, False)
                view.clear()
                with view:
                    block(ui_text("system_settings.reconnecting_to_core"), str(error))
                    action_link(
                        ui_text("system_settings.connection_settings"),
                        route("/settings/connection", destination, setup=1),
                        tone="accent",
                    )

        await refresh()
        ui.timer(1.5, refresh)
