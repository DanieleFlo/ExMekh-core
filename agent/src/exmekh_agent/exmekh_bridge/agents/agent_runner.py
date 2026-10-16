from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.reasoning_effort import (
    ReasoningEffort,
    ReasoningEffortLMH,
    ReasoningEffortLMX,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider

from exmekh_agent.config import config
from exmekh_agent.config.agents import DEFAULTS as AGENT_DEFAULTS
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.execution import TITLE_GENERATION
from exmekh_agent.exmekh_bridge.agents.auto_compaction import AutoCompactionPolicy
from exmekh_agent.exmekh_bridge.agents.cancellation import (
    CancellableProvider,
    raise_if_cancelled,
)
from exmekh_agent.exmekh_bridge.agents.event_handlers import (
    DeepAgentTurnEventHandler,
    QuickChatEventHandler,
    publish_raw_history,
)
from exmekh_agent.exmekh_bridge.agents.steering import TurnSteering
from exmekh_agent.exmekh_bridge.agents.sub_agents import (
    SubAgentEvent,
    SubAgentEventRouter,
)
from exmekh_agent.exmekh_bridge.agents.turn_request import AgentTurnRequest
from exmekh_agent.exmekh_bridge.capabilities.agent_design import (
    build_designed_agent,
    build_generic_agent,
)
from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.exmekh_bridge.history.compaction import (
    HistoryArchive,
    ModelContextCheckpoint,
    latest_model_context,
)
from exmekh_agent.exmekh_bridge.history.converter import (
    convert_to_exmekh_messages,
    exmekh_message_to_raw_dict,
    restore_node_history,
    serialize_skill_execution,
)
from exmekh_agent.exmekh_bridge.history.image_input import (
    restore_image_history,
    serialize_user_content,
    user_content,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatSettings,
    ChatTitle,
    ProviderContextUsage,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.providers.provider_context import (
    ContextTrackingProvider,
)
from exmekh_agent.exmekh_bridge.providers.provider_health import (
    check_provider_health as check_provider_health,
)
from exmekh_agent.exmekh_bridge.providers.provider_health import (
    extract_model_context_length as extract_model_context_length,
)
from exmekh_agent.exmekh_bridge.resources import load_prompt
from exmekh_agent.exmekh_bridge.tools.tool_approval import (
    DEFAULT_TOOL_APPROVAL_MODE,
    CommandApprovalHook,
    CommandApprovedCallback,
)
from exmekh_agent.state.agent_design import AgentDesign

logger = logging.getLogger(__name__)

# ds4 currently keeps one live agentic checkpoint frontier per server process.
# Serializing complete requests prevents one conversation from restoring another
# conversation's skill checkpoint.  A threading lock also covers worker threads
# used by title generation and by separate NiceGUI clients.
_provider_request_lock = threading.RLock()


def execute_agent_turn_sync(
    user_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    api_type: str,
    reasoning_effort_str: str,
    agent_mode: str,
    capabilities_path_str: str,
    name_hds: str,
    on_token: Callable[[str], None],
    on_reasoning: Callable[..., None],
    on_tool_event: Callable[[ToolEvent], None],
    on_skill_event: Callable[[SkillNode, str], None] | None = None,
    on_raw_history: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None]
    | None = None,
    history: Any | None = None,
    agent_context_id: str | None = None,
    on_sub_agent_event: Callable[[SubAgentEvent], None] | None = None,
    sub_agent_history: list[SubAgentConversation] | None = None,
    work_root: str | None = None,
    coding_agent_persistent_history: bool | None = None,
    on_context: Callable[[ProviderContextUsage], None] | None = None,
    on_request_manual_approval: Callable[[str], bool] | None = None,
    on_command_approved: CommandApprovedCallback | None = None,
    tool_approval_mode: str | None = None,
    get_tool_approval_mode: Callable[[], str] | None = None,
    approved_commands_history: list[str] | None = None,
    tool_approval_raw_history: list[dict[str, Any]] | None = None,
    cancellation_event: threading.Event | None = None,
    steering: TurnSteering | None = None,
    on_steering_added: Callable[[], None] | None = None,
    image_urls: list[str] | None = None,
    agent_profile: str = AGENT_DEFAULTS["default_profile"],
    agent_design: AgentDesign | None = None,
    capabilities_paths: list[str] | None = None,
    on_model_context: Callable[[ModelContextCheckpoint], None] | None = None,
) -> str:
    """Run one complete provider request without overlapping another request."""
    with _provider_request_lock:
        if cancellation_event is not None:
            raise_if_cancelled(cancellation_event)
        return _execute_agent_turn_sync_unlocked(
            request=AgentTurnRequest(
                user_prompt=user_prompt,
                model=model,
                base_url=base_url,
                api_key=api_key,
                api_type=api_type,
                reasoning_effort_str=reasoning_effort_str,
                agent_mode=agent_mode,
                capabilities_path_str=capabilities_path_str,
                name_hds=name_hds,
                on_token=on_token,
                on_reasoning=on_reasoning,
                on_tool_event=on_tool_event,
                on_skill_event=on_skill_event,
                on_raw_history=on_raw_history,
                history=history,
                agent_context_id=agent_context_id,
                on_sub_agent_event=on_sub_agent_event,
                sub_agent_history=sub_agent_history,
                work_root=work_root,
                coding_agent_persistent_history=coding_agent_persistent_history,
                on_context=on_context,
                on_request_manual_approval=on_request_manual_approval,
                on_command_approved=on_command_approved,
                tool_approval_mode=tool_approval_mode,
                get_tool_approval_mode=get_tool_approval_mode,
                approved_commands_history=approved_commands_history,
                tool_approval_raw_history=tool_approval_raw_history,
                cancellation_event=cancellation_event,
                steering=steering,
                on_steering_added=on_steering_added,
                image_urls=image_urls,
                agent_profile=agent_profile,
                agent_design=agent_design,
                capabilities_paths=capabilities_paths,
                on_model_context=on_model_context,
            )
        )


def _execute_agent_turn_sync_unlocked(request: AgentTurnRequest) -> str:
    """Synchronous execution of a turn using exmekh_lib, dispatched in background thread."""
    content = user_content(request.user_prompt, request.image_urls or [])
    reasoning_effort: ReasoningEffort | None = None
    if request.reasoning_effort_str:
        clean_effort = request.reasoning_effort_str.strip().lower()
        try:
            reasoning_effort = ReasoningEffortLMX(clean_effort)
        except ValueError:
            try:
                reasoning_effort = ReasoningEffortLMH(clean_effort)
            except ValueError:
                reasoning_effort = None

    provider = LocalProvider(
        model=request.model,
        api="responses" if request.api_type == "responses" else "chat_completions",
        reasoning_effort=reasoning_effort,
        base_url=request.base_url,
        api_key=request.api_key,
    )
    context_provider = None
    if (
        (
            request.agent_mode == "deep_agent"
            and config.get_server_context_length(request.model, request.base_url)
            is not None
        )
        or request.on_context
        or (request.steering is not None and request.agent_mode == "deep_agent")
    ):
        context_provider = ContextTrackingProvider(
            provider,
            on_context=request.on_context,
            context_id=request.agent_context_id
            or request.name_hds
            or AGENT_DEFAULTS["name_hds"],
            agentic_only=request.agent_mode == "deep_agent",
            ds4_qwen=config.uses_ds4_qwen_prompt(request.model, request.base_url),
            thinking=request.reasoning_effort_str.strip().lower() != "none",
            api_type=request.api_type,
        )
        provider = context_provider

    if request.cancellation_event is not None:
        provider = CancellableProvider(provider, request.cancellation_event)

    if request.agent_mode == "deep_agent":
        effective_approval_mode = (
            request.tool_approval_mode
            if request.tool_approval_mode is not None
            else DEFAULT_TOOL_APPROVAL_MODE
        )
        approval_hook = CommandApprovalHook(
            mode=effective_approval_mode,
            get_mode=request.get_tool_approval_mode,
            history=request.tool_approval_raw_history
            or request.approved_commands_history
            or [],
            request_manual_approval=request.on_request_manual_approval,
            on_command_approved=request.on_command_approved,
            model=request.model,
            base_url=request.base_url,
            api_key=request.api_key,
            api_type=request.api_type,
        )
        approval_tool_hooks = {
            AGENT_DEFAULTS["shell_tool"]: ToolHooks(before=approval_hook)
        }
        cap_path = Path(request.capabilities_path_str or config.capabilities_path)
        cap_paths = (
            [Path(path) for path in request.capabilities_paths]
            if request.capabilities_paths
            else [cap_path]
        )
        event_router = SubAgentEventRouter(request.on_sub_agent_event)
        if (
            request.agent_design is not None
            or request.agent_profile != AGENT_DEFAULTS["generic_profile"]
        ):
            design = request.agent_design or CapabilityRepository().default_design(
                request.agent_profile, [str(path) for path in cap_paths]
            )
            root_node, children = build_designed_agent(
                design, event_router, approval_tool_hooks, request.agent_context_id
            )
        else:
            root_node, children = build_generic_agent(
                cap_paths,
                event_router,
                approval_tool_hooks,
                root_name=request.name_hds or AGENT_DEFAULTS["name_hds"],
                coding_persistent_history=config.coding_agent_persistent_history
                if request.coding_agent_persistent_history is None
                else request.coding_agent_persistent_history,
                agent_context_id=request.agent_context_id,
            )

        prior_exmekh = convert_to_exmekh_messages(
            request.history, current_user_prompt=request.user_prompt
        )
        content_by_dump = {
            message.message_dump()["content"]: serialize_user_content(message)
            for message in [*prior_exmekh, UserMessage(content)]
            if isinstance(message, UserMessage)
            and not isinstance(message.get_message(), str)
        }

        archive = HistoryArchive(latest_model_context(request.history))
        archive.begin_turn([exmekh_message_to_raw_dict(m) for m in prior_exmekh])

        def publish_multimodal_history(raw_history, skill_tree):
            restored_history, restored_tree = restore_image_history(
                raw_history, skill_tree, content_by_dump
            )
            expanded_history = archive.update(restored_history)
            expanded_history, expanded_tree = restore_image_history(
                expanded_history, archive.expand_tree(restored_tree), content_by_dump
            )
            checkpoint = archive.checkpoint()
            if checkpoint is not None and request.on_model_context:
                try:
                    request.on_model_context(checkpoint)
                except Exception as error:
                    logger.warning(
                        app_text(
                            "errors_agents.event_handlers.unable_to_publish_model_context_s"
                        ),
                        error,
                    )
            if request.on_raw_history:
                publish_raw_history(
                    request.on_raw_history,
                    expanded_history,
                    expanded_tree,
                )

        def on_compaction(event):
            if context_provider is not None:
                context_provider.invalidate_context(event.agent_context_id)
            if event.agent_context_id != root_node.agent_context_id:
                return
            archive.record_compaction(event.replaced_history)
            publish_multimodal_history(event.history, event.history_tree)

        event_handler = DeepAgentTurnEventHandler(
            event_router,
            request.on_token,
            request.on_reasoning,
            request.on_tool_event,
            request.on_skill_event,
            publish_multimodal_history
            if request.on_raw_history or request.on_model_context
            else None,
            request.steering,
            request.on_steering_added,
            on_compaction,
        )
        if context_provider is not None:
            context_provider.set_request_observer(event_handler.begin_request)

        agent = DeepAgent(
            provider=provider,
            node=root_node,
            work_root=Path(request.work_root or config.work_root),
            on_stream=event_handler,
        )
        if context_provider is not None:
            for node in agent.capabilities_by_node:
                node.set_history_compaction_policy(
                    AutoCompactionPolicy(
                        context_provider, model=request.model, base_url=request.base_url
                    )
                )

        for child in children:
            if not child.persistent_history or not request.sub_agent_history:
                continue
            previous = next(
                (
                    item
                    for item in reversed(request.sub_agent_history)
                    if item.path == child.path and item.persistent_history
                ),
                None,
            )
            if previous:
                child.restore_conversation(
                    previous,
                    prior_exmekh=convert_to_exmekh_messages(previous.replay_history()),
                )

        if request.history:
            restore_node_history(root_node, request.history, prior_exmekh=prior_exmekh)
            if context_provider is not None:
                for message in reversed(request.history):
                    usage = getattr(message, "context_usage", None)
                    if usage is not None:
                        context_provider.restore_context_usage(
                            root_node.agent_context_id, usage
                        )
                        break

        try:
            if request.steering is not None:
                # Restored messages follow one system message and precede the
                # initial user message. Skill compaction preserves that prefix.
                archived_prefix = archive.expand(
                    [exmekh_message_to_raw_dict(m) for m in prior_exmekh]
                )
                request.steering.bind(
                    agent.steer, history_start=len(archived_prefix) + 2
                )
            response = agent.run(user_message=content)
        except BaseException:
            event_handler.finish("error")
            raise
        else:
            event_handler.finish()
        finally:
            if request.steering is not None:
                request.steering.close()

        if request.on_raw_history or request.on_model_context:
            publish_multimodal_history(
                response.history,
                [serialize_skill_execution(s) for s in response.skill_executions],
            )

        return str(response.get_message() or "")

    else:
        # Quick chat using Client
        client = Client(
            provider=provider,
            system_message=SystemMessage(load_prompt("quick_chat")),
        )
        prior_exmekh = convert_to_exmekh_messages(
            request.history, current_user_prompt=request.user_prompt
        )
        if prior_exmekh:
            try:
                client.replace_history_from(1, prior_exmekh)
            except Exception as e:  # noqa: BLE001 - legacy history is best-effort
                logger.warning(
                    app_text(
                        "errors_agents.agent_runner.failed_to_seed_history_into_client",
                        value_1=e,
                    )
                )

        event_handler = QuickChatEventHandler(request.on_token, request.on_reasoning)

        response = client.call(
            ClientRequest(
                UserMessage(content),
                streaming=True,
                on_event=event_handler,
            )
        )
        final_text = event_handler.finish(str(response.get_message() or ""))

        if request.on_raw_history:
            publish_raw_history(
                request.on_raw_history,
                [exmekh_message_to_raw_dict(item) for item in client.get_history()],
                [],
            )

        return final_text


async def run_agent_turn(
    user_prompt: str,
    model: str,
    on_token: Callable[[str], None],
    on_reasoning: Callable[..., None],
    on_tool_event: Callable[[ToolEvent], None],
    agent_mode: str | None = None,
    on_skill_event: Callable[[SkillNode, str], None] | None = None,
    on_raw_history: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None]
    | None = None,
    history: Any | None = None,
    agent_context_id: str | None = None,
    on_sub_agent_event: Callable[[SubAgentEvent], None] | None = None,
    sub_agent_history: list[SubAgentConversation] | None = None,
    settings: ChatSettings | None = None,
    on_context: Callable[[ProviderContextUsage], None] | None = None,
    on_request_manual_approval: Callable[[str], bool] | None = None,
    on_command_approved: CommandApprovedCallback | None = None,
    steering: TurnSteering | None = None,
    on_steering_added: Callable[[], None] | None = None,
    image_urls: list[str] | None = None,
    get_tool_approval_mode: Callable[[], str] | None = None,
    on_model_context: Callable[[ModelContextCheckpoint], None] | None = None,
) -> str:
    """Asynchronously runs the agent turn in a background worker thread."""
    loop = asyncio.get_running_loop()

    mode = agent_mode or config.agent_mode
    settings = settings or ChatSettings()
    cancellation_event = threading.Event()

    def request_manual_approval(callback: Callable[[str], bool], command: str) -> bool:
        raise_if_cancelled(cancellation_event)
        try:
            approved = callback(command)
        except asyncio.CancelledError:
            cancellation_event.set()
            raise
        raise_if_cancelled(cancellation_event)
        return approved

    def _call_sync() -> str:
        return execute_agent_turn_sync(
            user_prompt=user_prompt,
            model=model or config.default_model,
            base_url=config.base_url,
            api_key=config.api_key,
            api_type=config.api_type,
            reasoning_effort_str=config.reasoning_effort,
            agent_mode=mode,
            capabilities_path_str=settings.capabilities_path
            or config.capabilities_path,
            name_hds=settings.name_hds or config.name_hds,
            agent_profile=settings.agent_profile,
            agent_design=settings.agent_design,
            capabilities_paths=settings.capability_paths(config.capabilities_path),
            on_token=on_token,
            on_reasoning=on_reasoning,
            on_tool_event=on_tool_event,
            on_skill_event=on_skill_event,
            on_raw_history=on_raw_history,
            history=history,
            agent_context_id=agent_context_id,
            on_sub_agent_event=on_sub_agent_event,
            sub_agent_history=sub_agent_history,
            work_root=settings.work_root or config.work_root,
            coding_agent_persistent_history=(
                config.coding_agent_persistent_history
                if settings.coding_agent_persistent_history is None
                else settings.coding_agent_persistent_history
            ),
            on_context=on_context,
            on_model_context=on_model_context,
            on_request_manual_approval=partial(
                request_manual_approval, on_request_manual_approval
            )
            if on_request_manual_approval is not None
            else None,
            on_command_approved=on_command_approved,
            tool_approval_mode=settings.tool_approval_mode,
            get_tool_approval_mode=get_tool_approval_mode,
            approved_commands_history=list(settings.approved_commands_history),
            tool_approval_raw_history=list(settings.tool_approval_raw_history),
            cancellation_event=cancellation_event,
            steering=steering,
            on_steering_added=on_steering_added,
            image_urls=image_urls,
        )

    try:
        return await loop.run_in_executor(None, _call_sync)
    except asyncio.CancelledError:
        cancellation_event.set()
        raise
    finally:
        if steering is not None:
            steering.close()


def generate_chat_title_sync(
    user_prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
) -> str:
    """Generate a title without overlapping another local-provider request."""
    with _provider_request_lock:
        return _generate_chat_title_sync_unlocked(
            user_prompt=user_prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_type=api_type,
        )


def _generate_chat_title_sync_unlocked(
    user_prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
) -> str:
    """Generate a concise chat title from the user's first prompt."""
    if not user_prompt or not user_prompt.strip():
        return app_text("execution.agent_runner.new_conversation")

    fallback_title = TITLE_GENERATION.fallback(user_prompt)

    effective_model = model or config.default_model
    effective_api = api_type or config.api_type

    try:
        provider = LocalProvider(
            model=effective_model,
            api="responses" if effective_api == "responses" else "chat_completions",
            base_url=base_url or config.base_url,
            api_key=api_key or config.api_key,
        )

        client = Client(
            provider=provider,
            system_message=SystemMessage(load_prompt("chat_title")),
        )

        request = ClientRequest(
            message=UserMessage(
                load_prompt("chat_title_input").format(user_prompt=user_prompt)
            ),
            extra=TITLE_GENERATION.request_extra(effective_api),
        )

        response = client.call_structured(request, ChatTitle, strict=True)

        title = TITLE_GENERATION.normalize(response.get_message().title)

        if title:
            return title
    except Exception as e:  # noqa: BLE001 - title generation intentionally falls back
        logger.warning(
            app_text(
                "errors_agents.agent_runner.error_during_structured_title_generation_via_agentic_client",
                value_1=e,
            )
        )

    return fallback_title


async def generate_chat_title(
    user_prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
) -> str:
    """Asynchronously generates a chat title using the agentic Client in a worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        generate_chat_title_sync,
        user_prompt,
        model or config.default_model,
        base_url or config.base_url,
        api_key or config.api_key,
        api_type or config.api_type,
    )
