from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.history import MessageHistory
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.tool_calls.execution_history import ExecutionHistory
from exmekh_lib.core.tool_calls.harness_memory_tools import HarnessMemoryTools

from fe.exmekh_bridge.agent_runner import (
    convert_to_exmekh_messages,
    execute_agent_turn_sync,
    run_agent_turn,
)
from fe.exmekh_bridge.converter import (
    exmekh_message_to_raw_dict,
    exmekh_messages_to_chat_messages,
    exmekh_messages_to_session,
    raw_dict_to_exmekh_message,
    reconstruct_execution_history_and_cache,
    session_to_exmekh_messages,
)
from fe.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ChatSettings,
    ReasoningBlock,
    SkillNode,
    ToolEvent,
)
from tests.mock_client import MockProvider


def test_chat_options_survive_exmekh_history_roundtrip() -> None:
    session = ChatSession(
        settings=ChatSettings(
            name_hds="chat-hds",
            capabilities_path="chat-capabilities",
            work_root="chat-workspace",
            coding_agent_persistent_history=False,
        )
    )

    messages = session_to_exmekh_messages(session)
    restored = exmekh_messages_to_session(messages)
    assert restored.settings == session.settings

    serialized = [
        exmekh_message_to_raw_dict(message, include_snapshots=True)
        for message in messages
    ]
    restored_from_json = exmekh_messages_to_session(
        [raw_dict_to_exmekh_message(item) for item in serialized]
    )
    assert restored_from_json.settings == session.settings


def test_history_preserved_in_quick_chat_agent_turn() -> None:
    """Verifies that on turn 2, prior conversation history is passed to the provider

    rather than being cleared / reset.
    """
    mock_provider = MockProvider()

    prior_history = [
        ChatMessage(role="user", content="Mi chiamo Mario"),
        ChatMessage(role="assistant", content="Piacere Mario!"),
    ]

    with patch(
        "fe.exmekh_bridge.agent_runner.LocalProvider", return_value=mock_provider
    ):
        execute_agent_turn_sync(
            user_prompt="Come mi chiamo?",
            model="test-model",
            base_url="http://localhost:8080/v1",
            api_key="k",
            api_type="chat_completions",
            reasoning_effort_str="none",
            agent_mode="quick_chat",
            capabilities_path_str="",
            name_hds="main",
            on_token=lambda t: None,
            on_reasoning=lambda *args: None,
            on_tool_event=lambda ev: None,
            history=prior_history,
        )

    assert mock_provider.last_request is not None
    messages = mock_provider.last_request.get_messages()
    roles = [m.get_role_name() for m in messages]
    contents = [str(m.get_message() or "") for m in messages]

    assert roles == ["system", "user", "assistant", "user"]
    assert contents[1] == "Mi chiamo Mario"
    assert contents[2] == "Piacere Mario!"
    assert contents[3] == "Come mi chiamo?"


def test_history_preserved_in_deep_agent_turn() -> None:
    """Verifies that DeepAgent mode also preserves prior conversation history."""
    mock_provider = MockProvider()

    prior_history = [
        ChatMessage(role="user", content="Primo messaggio"),
        ChatMessage(role="assistant", content="Prima risposta"),
    ]

    with patch(
        "fe.exmekh_bridge.agent_runner.LocalProvider", return_value=mock_provider
    ):
        execute_agent_turn_sync(
            user_prompt="Secondo messaggio",
            model="test-model",
            base_url="http://localhost:8080/v1",
            api_key="k",
            api_type="responses",
            reasoning_effort_str="none",
            agent_mode="deep_agent",
            capabilities_path_str="",
            name_hds="main",
            on_token=lambda t: None,
            on_reasoning=lambda *args: None,
            on_tool_event=lambda ev: None,
            history=prior_history,
        )

    assert mock_provider.last_request is not None
    messages = mock_provider.last_request.get_messages()
    roles = [m.get_role_name() for m in messages]
    contents = [str(m.get_message() or "") for m in messages]

    assert roles == ["system", "user", "assistant", "user"]
    assert contents[1] == "Primo messaggio"
    assert contents[2] == "Prima risposta"
    assert contents[3] == "Secondo messaggio"


@pytest.mark.asyncio
async def test_run_agent_turn_passes_history_to_model() -> None:
    """Verifies that run_agent_turn forwards history across consecutive turns in the same chat."""
    mock_provider = MockProvider()
    session = ChatSession(title="Test Multi-turn", agent_mode="quick_chat")

    with patch(
        "fe.exmekh_bridge.agent_runner.LocalProvider", return_value=mock_provider
    ):
        # Turn 1
        user1 = ChatMessage(role="user", content="Il mio numero fortunato è 77")
        session.messages.append(user1)
        resp1 = await run_agent_turn(
            user_prompt=user1.content,
            model=session.model,
            on_token=lambda t: None,
            on_reasoning=lambda *args: None,
            on_tool_event=lambda ev: None,
            agent_mode=session.agent_mode,
            history=session.messages[:-1],
        )
        session.messages.append(ChatMessage(role="assistant", content=resp1))

        # Turn 2 in the SAME chat
        user2 = ChatMessage(role="user", content="Qual è il mio numero fortunato?")
        session.messages.append(user2)
        await run_agent_turn(
            user_prompt=user2.content,
            model=session.model,
            on_token=lambda t: None,
            on_reasoning=lambda *args: None,
            on_tool_event=lambda ev: None,
            agent_mode=session.agent_mode,
            history=session.messages[:-1],
        )

    assert mock_provider.last_request is not None
    messages = mock_provider.last_request.get_messages()
    roles = [m.get_role_name() for m in messages]
    contents = [str(m.get_message() or "") for m in messages]

    assert roles == ["system", "user", "assistant", "user"]
    assert contents[1] == "Il mio numero fortunato è 77"
    assert contents[2] == resp1
    assert contents[3] == "Qual è il mio numero fortunato?"


# =============================================================================
# 10 REALISTIC COMPLEX CHATS FOR BIDIRECTIONAL CONVERSION & ROUND-TRIP TESTING
# =============================================================================


def build_10_diverse_realistic_chats() -> list[ChatSession]:
    """Constructs 10 highly varied, realistic ChatSessions covering:

    1. Multi-turn Q&A with LaTeX & code blocks
    2. Root tools + Depth-1 Skill with inner tools
    3. 3-level nested Matryoshka skills (Skill -> Sub-Skill -> Sub-Sub-Skill)
    4. Parallel Sub-Agents delegation (Researcher + Coder sub-agents)
    5. Multi-turn tool chain with memory recall across turns
    6. Error recovery & fallback nested sub-skill
    7. Complex JSON arguments & XML-wrapped tool outputs in raw_history
    8. Multi-branch sibling skills & multi-path reasoning blocks
    9. Full Sub-Agent hierarchy with nested DeepSkills and AST tools
    10. 5-turn full-lifecycle stress test mixing quick turns, deep skills, sub-agents, and tools
    """
    chats: list[ChatSession] = []

    # -------------------------------------------------------------------------
    # CHAT 1: Multi-turn Q&A with LaTeX formulas & code blocks
    # -------------------------------------------------------------------------
    chat1 = ChatSession(
        id="chat-01-latex-qa",
        title="Analisi Matematica e Python",
        model="Qwen3.8-27B-UD-Q4_K_S",
        agent_mode="quick_chat",
        system_prompt="Sei un professore di matematica e informatica.",
        messages=[
            ChatMessage(role="user", content="Spiegami l'integrale di Gauss."),
            ChatMessage(
                role="assistant",
                content="L'integrale di Gauss è $\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$.",
                reasoning_content="Richiamo la formula in coordinate polari.",
                reasoning_blocks=[
                    ReasoningBlock(
                        skill_path="/",
                        depth=0,
                        generation_index=1,
                        content="Richiamo la formula in coordinate polari.",
                    )
                ],
                raw_history=[
                    {
                        "role": "assistant",
                        "content": "L'integrale di Gauss è $\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$.",
                    }
                ],
            ),
            ChatMessage(
                role="user", content="Come lo approssimo in Python con Monte Carlo?"
            ),
            ChatMessage(
                role="assistant",
                content="Ecco un esempio usando `numpy.random.normal` per stimare il valore.",
                reasoning_content="Scrivo uno snippet compatto con numpy.",
                reasoning_blocks=[
                    ReasoningBlock(
                        skill_path="/",
                        depth=0,
                        generation_index=1,
                        content="Scrivo uno snippet compatto con numpy.",
                    )
                ],
                raw_history=[
                    {
                        "role": "assistant",
                        "content": "Ecco un esempio usando `numpy.random.normal` per stimare il valore.",
                    }
                ],
            ),
        ],
    )
    chats.append(chat1)

    # -------------------------------------------------------------------------
    # CHAT 2: Root tools + Depth-1 Skill with inner tools
    # -------------------------------------------------------------------------
    skill_code_analyzer = SkillNode(
        id="sk-code-analyzer",
        name="code-analyzer-skill",
        path="/code-analyzer-skill",
        depth=1,
        status="completed",
        is_collapsed=True,
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/code-analyzer-skill",
                depth=1,
                generation_index=2,
                content="Analizzo la complessità ciclomatica.",
            )
        ],
        tool_events=[
            ToolEvent(
                tool_name="ast_parse", depth=1, skill_path="/code-analyzer-skill"
            ),
            ToolEvent(
                tool_name="complexity_metric",
                depth=1,
                skill_path="/code-analyzer-skill",
            ),
        ],
    )
    chat2 = ChatSession(
        id="chat-02-root-tools-and-skill",
        title="Analisi Static Codebase",
        agent_mode="deep_agent",
        messages=[
            ChatMessage(
                role="user",
                content="Controlla i file in src/ e analizza la complessità.",
            ),
            ChatMessage(
                role="assistant",
                content="Ho analizzato `src/`: complessità ciclomatica media pari a 3.2 (ottima).",
                reasoning_content="Prima elenchiamo i file, poi apriamo code-analyzer-skill.",
                reasoning_blocks=[
                    ReasoningBlock(
                        skill_path="/",
                        depth=0,
                        generation_index=1,
                        content="Prima elenchiamo i file.",
                    ),
                    ReasoningBlock(
                        skill_path="/code-analyzer-skill",
                        depth=1,
                        generation_index=2,
                        content="Analizzo la complessità ciclomatica.",
                    ),
                ],
                tool_events=[
                    ToolEvent(tool_name="list_dir", depth=0, skill_path="/"),
                ],
                skills=[skill_code_analyzer],
            ),
        ],
    )
    chats.append(chat2)

    # -------------------------------------------------------------------------
    # CHAT 3: 3-Level Nested Matryoshka Skills (Depth 1 -> Depth 2 -> Depth 3)
    # -------------------------------------------------------------------------
    depth3_skill = SkillNode(
        id="sk-d3-sql",
        name="sql-schema-validator",
        path="/arch-planner/db-migrator/sql-schema-validator",
        depth=3,
        status="completed",
        is_collapsed=True,
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/arch-planner/db-migrator/sql-schema-validator",
                depth=3,
                generation_index=3,
                content="Verifico vincoli foreign key.",
            )
        ],
        tool_events=[
            ToolEvent(
                tool_name="check_fk_constraints",
                depth=3,
                skill_path="/arch-planner/db-migrator/sql-schema-validator",
            ),
            ToolEvent(
                tool_name="explain_query_plan",
                depth=3,
                skill_path="/arch-planner/db-migrator/sql-schema-validator",
            ),
        ],
    )
    depth2_skill = SkillNode(
        id="sk-d2-db",
        name="db-migrator",
        path="/arch-planner/db-migrator",
        depth=2,
        status="completed",
        is_collapsed=True,
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/arch-planner/db-migrator",
                depth=2,
                generation_index=2,
                content="Genero script Alembic e delego la validazione SQL.",
            )
        ],
        tool_events=[
            ToolEvent(
                tool_name="generate_alembic_revision",
                depth=2,
                skill_path="/arch-planner/db-migrator",
            ),
        ],
        sub_skills=[depth3_skill],
    )
    depth1_skill = SkillNode(
        id="sk-d1-arch",
        name="arch-planner",
        path="/arch-planner",
        depth=1,
        status="completed",
        is_collapsed=True,
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/arch-planner",
                depth=1,
                generation_index=1,
                content="Pianifico l'architettura microservizi.",
            )
        ],
        tool_events=[
            ToolEvent(
                tool_name="read_architecture_doc", depth=1, skill_path="/arch-planner"
            ),
        ],
        sub_skills=[depth2_skill],
    )
    chat3 = ChatSession(
        id="chat-03-matryoshka-depth-3",
        title="Migrazione Database Gerarchica",
        agent_mode="deep_agent",
        messages=[
            ChatMessage(
                role="user",
                content="Pianifica l'architettura e valida la migrazione DB a 3 livelli.",
            ),
            ChatMessage(
                role="assistant",
                content="Migrazione pianificata e validata fino al livello 3 (`sql-schema-validator`).",
                skills=[depth1_skill],
                skill_history_tree=[
                    {
                        "skill": "arch-planner",
                        "opening_tool_call_id": "call_arch_1",
                        "output": "Architettura completata",
                        "history": [
                            {"role": "system", "content": "HDS arch-planner"},
                            {"role": "assistant", "content": "Avvio db-migrator"},
                        ],
                        "children": [
                            {
                                "skill": "db-migrator",
                                "opening_tool_call_id": "call_db_2",
                                "output": "DB migrato",
                                "history": [
                                    {"role": "system", "content": "HDS db-migrator"},
                                    {
                                        "role": "assistant",
                                        "content": "Avvio sql-schema-validator",
                                    },
                                ],
                                "children": [
                                    {
                                        "skill": "sql-schema-validator",
                                        "opening_tool_call_id": "call_sql_3",
                                        "output": "Schema SQL valido",
                                        "history": [
                                            {
                                                "role": "system",
                                                "content": "HDS sql-schema-validator",
                                            },
                                            {
                                                "role": "assistant",
                                                "content": "FK verificate",
                                            },
                                        ],
                                        "children": [],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            ),
        ],
    )
    chats.append(chat3)

    # -------------------------------------------------------------------------
    # CHAT 4: Parallel Sub-Agents Delegation (Researcher + Coder Sub-Agents)
    # -------------------------------------------------------------------------
    sub_agent_research = SkillNode(
        id="subagent-researcher",
        name="sub_agent_researcher",
        path="/sub_agent_researcher",
        depth=1,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="arxiv_search", depth=1, skill_path="/sub_agent_researcher"
            ),
            ToolEvent(
                tool_name="extract_citations",
                depth=1,
                skill_path="/sub_agent_researcher",
            ),
        ],
    )
    sub_agent_coder = SkillNode(
        id="subagent-coder",
        name="sub_agent_coder",
        path="/sub_agent_coder",
        depth=1,
        status="completed",
        tool_events=[
            ToolEvent(tool_name="write_module", depth=1, skill_path="/sub_agent_coder"),
            ToolEvent(tool_name="run_pytest", depth=1, skill_path="/sub_agent_coder"),
        ],
    )
    chat4 = ChatSession(
        id="chat-04-parallel-sub-agents",
        title="Delega Sub-Agenti Paralleli",
        agent_mode="deep_agent",
        messages=[
            ChatMessage(
                role="user",
                content="Avvia il sub-agente ricercatore e il sub-agente sviluppatore in parallelo.",
            ),
            ChatMessage(
                role="assistant",
                content="Entrambi i sub-agenti (`sub_agent_researcher` e `sub_agent_coder`) hanno terminato.",
                skills=[sub_agent_research, sub_agent_coder],
                raw_history=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_sub_res_1",
                                "type": "function",
                                "function": {
                                    "name": "sub_agent_researcher",
                                    "arguments": {
                                        "task": "Ricerca paper Transformer",
                                        "depth": 1,
                                        "path": "/sub_agent_researcher",
                                    },
                                },
                            },
                            {
                                "id": "call_sub_cod_2",
                                "type": "function",
                                "function": {
                                    "name": "sub_agent_coder",
                                    "arguments": {
                                        "task": "Implementa MultiHeadAttention",
                                        "depth": 1,
                                        "path": "/sub_agent_coder",
                                    },
                                },
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "sub_agent_researcher",
                        "tool_call_id": "call_sub_res_1",
                        "content": '<SubAgent name="sub_agent_researcher">Paper Vaswani et al. analizzato</SubAgent>',
                    },
                    {
                        "role": "tool",
                        "name": "sub_agent_coder",
                        "tool_call_id": "call_sub_cod_2",
                        "content": '<SubAgent name="sub_agent_coder">Modulo attention.py implementato e testato</SubAgent>',
                    },
                    {
                        "role": "assistant",
                        "content": "Entrambi i sub-agenti (`sub_agent_researcher` e `sub_agent_coder`) hanno terminato.",
                    },
                ],
            ),
        ],
    )
    chats.append(chat4)

    # -------------------------------------------------------------------------
    # CHAT 5: 3-Turn Tool Chain & Memory Recall Across Turns
    # -------------------------------------------------------------------------
    chat5 = ChatSession(
        id="chat-05-memory-recall-chain",
        title="Pipeline Estrazione e Recall Memoria",
        agent_mode="deep_agent",
        messages=[
            ChatMessage(
                role="user", content="Turno 1: Estrai i log dal server di produzione."
            ),
            ChatMessage(
                role="assistant",
                content="Log estratti e salvati in cache con ID `log_batch_99`.",
                tool_events=[
                    ToolEvent(tool_name="fetch_prod_logs", depth=0, skill_path="/")
                ],
                raw_history=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_log_1",
                                "type": "function",
                                "function": {
                                    "name": "fetch_prod_logs",
                                    "arguments": {"env": "prod"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "fetch_prod_logs",
                        "tool_call_id": "call_log_1",
                        "content": '<Tool name="fetch_prod_logs" id="log_batch_99">500 lines fetched</Tool>',
                    },
                    {
                        "role": "assistant",
                        "content": "Log estratti e salvati in cache con ID `log_batch_99`.",
                    },
                ],
            ),
            ChatMessage(
                role="user",
                content="Turno 2: Usa il sub-agente per filtrare gli errori 502 dal batch `log_batch_99`.",
            ),
            ChatMessage(
                role="assistant",
                content="Il sub-agente ha individuato 3 errori 502 Bad Gateway tra le 14:00 e le 14:05.",
                skills=[
                    SkillNode(
                        name="log-filter-skill",
                        path="/log-filter-skill",
                        depth=1,
                        status="completed",
                        tool_events=[
                            ToolEvent(
                                tool_name="read_cached_output",
                                depth=1,
                                skill_path="/log-filter-skill",
                            )
                        ],
                    )
                ],
            ),
            ChatMessage(
                role="user",
                content="Turno 3: Riassumi la causa radice basandoti sui due turni precedenti.",
            ),
            ChatMessage(
                role="assistant",
                content="La causa radice dei 3 errori 502 nel batch `log_batch_99` è il timeout di upstream Redis.",
            ),
        ],
    )
    chats.append(chat5)

    # -------------------------------------------------------------------------
    # CHAT 6: Error Recovery & Fallback Nested Sub-Skill
    # -------------------------------------------------------------------------
    fallback_sub = SkillNode(
        name="oauth2-token-refresh-skill",
        path="/api-connector-skill/oauth2-token-refresh-skill",
        depth=2,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="refresh_jwt",
                depth=2,
                skill_path="/api-connector-skill/oauth2-token-refresh-skill",
            )
        ],
    )
    api_skill = SkillNode(
        name="api-connector-skill",
        path="/api-connector-skill",
        depth=1,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="http_get_endpoint",
                depth=1,
                skill_path="/api-connector-skill",
            ),
            ToolEvent(
                tool_name="http_get_endpoint_retry",
                depth=1,
                skill_path="/api-connector-skill",
            ),
        ],
        sub_skills=[fallback_sub],
    )
    chat6 = ChatSession(
        id="chat-06-error-recovery",
        title="Recovery Automatico OAuth2",
        agent_mode="deep_agent",
        messages=[
            ChatMessage(
                role="user", content="Sincronizza i dati dall'endpoint protetto."
            ),
            ChatMessage(
                role="assistant",
                content="Dopo un 401 iniziale, ho rigenerato il token tramite `oauth2-token-refresh-skill` e completato la sincronizzazione.",
                skills=[api_skill],
            ),
            ChatMessage(role="user", content="Il nuovo token quanto dura?"),
            ChatMessage(
                role="assistant",
                content="Il token JWT appena rigenerato ha validità di 3600 secondi (1 ora).",
            ),
        ],
    )
    chats.append(chat6)

    # -------------------------------------------------------------------------
    # CHAT 7: Complex JSON Arguments & Dump-Style Stringified Tool Calls
    # -------------------------------------------------------------------------
    chat7 = ChatSession(
        id="chat-07-complex-json-args",
        title="Query Strutturate e JSON Annidati",
        agent_mode="deep_agent",
        messages=[
            ChatMessage(
                role="user",
                content="Esegui la pipeline di aggregazione con filtri JSON complessi.",
            ),
            ChatMessage(
                role="assistant",
                content="Aggregazione completata su 12.450 record.",
                raw_history=[
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": json.dumps(
                            [
                                {
                                    "id": "call_agg_900",
                                    "type": "function",
                                    "function": {
                                        "name": "run_aggregation_pipeline",
                                        "arguments": {
                                            "filters": {
                                                "status": ["ACTIVE", "PENDING"],
                                                "min_score": 0.85,
                                                "verified": True,
                                            },
                                            "group_by": ["region", "tier"],
                                            "limit": 100,
                                        },
                                    },
                                }
                            ]
                        ),
                    },
                    {
                        "role": "tool",
                        "content": '<Tool name="run_aggregation_pipeline" id="900">{"total": 12450, "buckets": 8}</Tool>',
                    },
                    {
                        "role": "assistant",
                        "content": "Aggregazione completata su 12.450 record.",
                        "tool_calls": "null",
                    },
                ],
            ),
        ],
    )
    chats.append(chat7)

    # -------------------------------------------------------------------------
    # CHAT 8: Multi-Branch Sibling Skills & Multi-Path Reasoning
    # -------------------------------------------------------------------------
    pw_sub_skill = SkillNode(
        name="playwright-e2e-runner",
        path="/e2e-verifier/playwright-e2e-runner",
        depth=2,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="browser_snapshot",
                depth=2,
                skill_path="/e2e-verifier/playwright-e2e-runner",
            )
        ],
    )
    sibling_1 = SkillNode(
        name="frontend-builder",
        path="/frontend-builder",
        depth=1,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="compile_tailwind", depth=1, skill_path="/frontend-builder"
            )
        ],
    )
    sibling_2 = SkillNode(
        name="backend-builder",
        path="/backend-builder",
        depth=1,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="build_fastapi_routes", depth=1, skill_path="/backend-builder"
            )
        ],
    )
    sibling_3 = SkillNode(
        name="e2e-verifier",
        path="/e2e-verifier",
        depth=1,
        status="completed",
        sub_skills=[pw_sub_skill],
    )
    chat8 = ChatSession(
        id="chat-08-sibling-skills",
        title="Build Full-Stack con 3 Skill Fratelli",
        agent_mode="deep_agent",
        messages=[
            ChatMessage(
                role="user", content="Costruisci FE, BE e lancia la verifica E2E."
            ),
            ChatMessage(
                role="assistant",
                content="Full-stack build e test Playwright superati al 100%.",
                reasoning_blocks=[
                    ReasoningBlock(
                        skill_path="/",
                        depth=0,
                        generation_index=1,
                        content="Orchestro le 3 skill principali.",
                    ),
                    ReasoningBlock(
                        skill_path="/frontend-builder",
                        depth=1,
                        generation_index=2,
                        content="Build CSS e componenti.",
                    ),
                    ReasoningBlock(
                        skill_path="/backend-builder",
                        depth=1,
                        generation_index=3,
                        content="Generazione endpoint API.",
                    ),
                    ReasoningBlock(
                        skill_path="/e2e-verifier/playwright-e2e-runner",
                        depth=2,
                        generation_index=4,
                        content="Verifica headless.",
                    ),
                ],
                skills=[sibling_1, sibling_2, sibling_3],
            ),
        ],
    )
    chats.append(chat8)

    # -------------------------------------------------------------------------
    # CHAT 9: Sub-Agent Tree with Nested DeepSkills and AST Rewriter Tools
    # -------------------------------------------------------------------------
    patch_skill = SkillNode(
        name="auto-patch-skill",
        path="/security-sub-agent/vuln-scanner-skill/auto-patch-skill",
        depth=3,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="apply_ast_diff",
                depth=3,
                skill_path="/security-sub-agent/vuln-scanner-skill/auto-patch-skill",
            )
        ],
    )
    vuln_skill = SkillNode(
        name="vuln-scanner-skill",
        path="/security-sub-agent/vuln-scanner-skill",
        depth=2,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="cve_database_query",
                depth=2,
                skill_path="/security-sub-agent/vuln-scanner-skill",
            )
        ],
        sub_skills=[patch_skill],
    )
    sec_sub_agent = SkillNode(
        name="security-sub-agent",
        path="/security-sub-agent",
        depth=1,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="sbom_generator", depth=1, skill_path="/security-sub-agent"
            )
        ],
        sub_skills=[vuln_skill],
    )
    chat9 = ChatSession(
        id="chat-09-security-subagent-tree",
        title="Audit di Sicurezza e Auto-Patching",
        agent_mode="deep_agent",
        messages=[
            ChatMessage(
                role="user",
                content="Esegui un audit di sicurezza completo e applica le patch.",
            ),
            ChatMessage(
                role="assistant",
                content="Individuata e corretta 1 vulnerabilità critica tramite `auto-patch-skill`.",
                skills=[sec_sub_agent],
            ),
            ChatMessage(role="user", content="Mostrami i file toccati dalla patch."),
            ChatMessage(
                role="assistant",
                content="Il file aggiornato da `apply_ast_diff` è `src/auth/token_verifier.py`.",
            ),
        ],
    )
    chats.append(chat9)

    # -------------------------------------------------------------------------
    # CHAT 10: 5-Turn Full-Lifecycle Stress Test
    # -------------------------------------------------------------------------
    chat10 = ChatSession(
        id="chat-10-full-lifecycle-stress",
        title="Stress Test 5 Turni Misti",
        model="Qwen3.8-27B-UD-Q4_K_S",
        agent_mode="deep_agent",
        system_prompt="System prompt custom per stress test multi-turno.",
        messages=[
            ChatMessage(
                role="user", content="Turno 1: Definisci il progetto 'QuantumBridge'."
            ),
            ChatMessage(
                role="assistant",
                content="Progetto `QuantumBridge` inizializzato nel workspace.",
            ),
            ChatMessage(
                role="user",
                content="Turno 2: Usa i tool di ricerca per mappare i moduli.",
            ),
            ChatMessage(
                role="assistant",
                content="Mappati 4 moduli core.",
                tool_events=[
                    ToolEvent(tool_name="find_by_name", depth=0, skill_path="/"),
                    ToolEvent(tool_name="grep_search", depth=0, skill_path="/"),
                ],
            ),
            ChatMessage(
                role="user", content="Turno 3: Apri la skill di refactoring annidata."
            ),
            ChatMessage(
                role="assistant",
                content="Refactoring completato su tutti i 4 moduli.",
                skills=[
                    SkillNode(
                        name="refactor-orchestrator",
                        path="/refactor-orchestrator",
                        depth=1,
                        status="completed",
                        sub_skills=[
                            SkillNode(
                                name="type-checker-subskill",
                                path="/refactor-orchestrator/type-checker-subskill",
                                depth=2,
                                status="completed",
                                tool_events=[
                                    ToolEvent(
                                        tool_name="mypy_strict_check",
                                        depth=2,
                                        skill_path="/refactor-orchestrator/type-checker-subskill",
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
            ChatMessage(
                role="user", content="Turno 4: Invoca il sub-agente di benchmark."
            ),
            ChatMessage(
                role="assistant",
                content="Benchmark eseguito: latenza ridotta del 42%.",
                skills=[
                    SkillNode(
                        name="benchmark_sub_agent",
                        path="/benchmark_sub_agent",
                        depth=1,
                        status="completed",
                        tool_events=[
                            ToolEvent(
                                tool_name="run_locust_load_test",
                                depth=1,
                                skill_path="/benchmark_sub_agent",
                            )
                        ],
                    )
                ],
            ),
            ChatMessage(
                role="user",
                content="Turno 5: Conferma il nome del progetto definito al Turno 1 e il risultato del Turno 4.",
            ),
            ChatMessage(
                role="assistant",
                content="Il progetto è `QuantumBridge` (Turno 1) e la riduzione di latenza ottenuta è del 42% (Turno 4).",
            ),
        ],
    )
    chats.append(chat10)

    return chats


@pytest.mark.parametrize(
    "chat_index",
    list(range(10)),
    ids=[
        "01_multi_turn_latex_qa",
        "02_root_tools_and_depth1_skill",
        "03_matryoshka_skills_depth_3",
        "04_parallel_sub_agents",
        "05_multi_turn_tool_chain_memory",
        "06_error_recovery_nested_skill",
        "07_complex_json_args_and_xml",
        "08_sibling_skills_multipath_reasoning",
        "09_security_subagent_tree_depth_3",
        "10_five_turn_full_lifecycle_stress",
    ],
)
def test_10_realistic_chats_bidirectional_conversion_and_roundtrip(
    chat_index: int,
) -> None:
    """Tests each of the 10 realistic chats:

    1. Converts ChatSession -> exmekh_lib Message list (session_to_exmekh_messages).
    2. Validates that exmekh_lib MessageHistory accepts the entire sequence (check_history).
    3. Verifies Client can load the converted history and send all turns to Provider on a follow-up prompt.
    4. Converts exmekh_lib Message list -> ChatSession (exmekh_messages_to_session) and verifies exact equality.
    5. Serializes through raw dicts (exmekh_message_to_raw_dict -> raw_dict_to_exmekh_message) and back, verifying exact equality.
    6. Strips snapshots and verifies structural reconstruction from pure exmekh_lib Messages.
    """
    all_chats = build_10_diverse_realistic_chats()
    original_session = all_chats[chat_index]

    # 1. Forward conversion: ChatSession -> list[exmekh_lib.interfaces.message.Message]
    exmekh_messages = session_to_exmekh_messages(original_session)
    assert len(exmekh_messages) >= len(original_session.messages) + 1
    assert exmekh_messages[0].get_role_name() == "system"

    # 2. Validate against exmekh_lib MessageHistory rules (system first, valid role transitions, valid ToolCalls)
    msg_history = MessageHistory(history=exmekh_messages)
    msg_history.check_history()

    # 3. Verify that loading into a exmekh_lib Client and sending a new message passes the ENTIRE history to the model
    mock_provider = MockProvider()
    client = Client(
        provider=mock_provider,
        system_message=SystemMessage(original_session.system_prompt or "System"),
    )
    prior_exmekh = convert_to_exmekh_messages(original_session.messages)
    client.replace_history_from(1, prior_exmekh)
    client.call(ClientRequest(UserMessage("Messaggio di verifica finale")))

    assert mock_provider.last_request is not None
    sent_messages = mock_provider.last_request.get_messages()
    # Must contain System + all converted history messages + the new UserMessage
    assert len(sent_messages) == 1 + len(prior_exmekh) + 1
    # Verify every single user prompt from the original session is present in order in sent_messages
    original_user_prompts = [
        m.content for m in original_session.messages if m.role == "user"
    ]
    sent_user_prompts = [
        str(m.get_message()) for m in sent_messages if m.get_role_name() == "user"
    ]
    assert sent_user_prompts == [*original_user_prompts, "Messaggio di verifica finale"]

    # 4. Backward conversion (Round-trip 1): list[exmekh_lib.Message] -> ChatSession
    restored_session = exmekh_messages_to_session(exmekh_messages)
    assert restored_session.model_dump() == original_session.model_dump()
    assert restored_session == original_session

    # 5. Round-trip through JSON dicts: ChatSession -> exmekh_messages -> raw_dicts -> exmekh_messages -> ChatSession
    raw_dicts = [
        exmekh_message_to_raw_dict(m, include_snapshots=True) for m in exmekh_messages
    ]
    pending_tcs: list[Any] = []
    reconstructed_exmekh = [
        raw_dict_to_exmekh_message(d, pending_tool_calls=pending_tcs) for d in raw_dicts
    ]
    restored_from_dicts = exmekh_messages_to_session(reconstructed_exmekh)
    assert restored_from_dicts.model_dump() == original_session.model_dump()

    # 6. Pure exmekh_lib Messages reconstruction (without snapshots) preserves roles, contents, skills, and tools
    pure_exmekh_dicts = [
        exmekh_message_to_raw_dict(m, include_snapshots=False) for m in exmekh_messages
    ]
    pending_pure: list[Any] = []
    pure_exmekh_msgs = [
        raw_dict_to_exmekh_message(d, pending_tool_calls=pending_pure)
        for d in pure_exmekh_dicts
    ]
    pure_restored = exmekh_messages_to_chat_messages(pure_exmekh_msgs[1:])

    assert len(pure_restored) == len(original_session.messages)
    for orig_msg, pure_msg in zip(original_session.messages, pure_restored):
        assert pure_msg.role == orig_msg.role
        assert pure_msg.content == orig_msg.content
        if orig_msg.skills:
            assert len(pure_msg.skills) == len(orig_msg.skills)
            for orig_sk, pure_sk in zip(orig_msg.skills, pure_msg.skills):
                assert pure_sk.name == orig_sk.name
                assert pure_sk.path == orig_sk.path
                assert pure_sk.depth == orig_sk.depth
                assert len(pure_sk.sub_skills) == len(orig_sk.sub_skills)


def test_execution_history_tool_preserves_prior_tools_from_chat_109e3e20(
    tmp_path: Path,
) -> None:
    """Verifies that a persisted turn restores tool history and cached outputs.

    and invoking get_execution_history on Turn 2, it does NOT return <total>1</total>,
    but returns all 8 capability calls from Turn 1 plus the Turn 2 call (<total>9</total>)
    along with their <saved-output-id>s and recoverable NodeCache entries.
    """
    calls = [
        (
            "mock-skill",
            "call_1aa42f66bfbf46afd9df928e58baed9d",
            "0_0_5",
            "Demo dimostrativa completata",
        ),
        (
            "mock-sub-tool",
            "call_4adc8f5ba98d6e2476bc1074f2f70347",
            "0_0_0",
            "Mock sub-tool result: chiamata diretta dal livello mock-skill",
        ),
        (
            "shared-tool",
            "call_1f00e87e28ae524e9cf2481b98ff9a23",
            "0_0_1",
            "shared result",
        ),
        (
            "nested-skill",
            "call_0365fde7dd39d4594463726635aa9b19",
            "0_0_4",
            "nested result",
        ),
        (
            "nested-sub-tool",
            "call_1bf81064ebe820a501bf8f096382ece5",
            "0_0_2",
            "nested tool result",
        ),
        (
            "shared-tool",
            "call_65a86e04e7a5783111328452169e9534",
            "0_0_3",
            "nested shared result",
        ),
        ("exit_agent", "call_7f62bb716b25cb23cb7ea5f89de56801", None, "nested exit"),
        ("exit_agent", "call_52de5bdcf770b7f3f86600c080e63cb9", None, "root exit"),
    ]
    raw_history: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": {}},
                }
                for name, call_id, _, _ in calls
            ],
        }
    ]
    raw_history.extend(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": (
                f"<header><name>{name}</name><type>Tool</type>"
                f"<id>{output_id}</id></header>\n{body}"
                if output_id is not None
                else body
            ),
        }
        for name, call_id, output_id, body in calls
    )
    turn1_messages = [
        ChatMessage(role="user", content="Esegui la demo"),
        ChatMessage(
            role="assistant", content="Demo completata", raw_history=raw_history
        ),
    ]

    exec_history = ExecutionHistory()
    session_cache = SessionCache(tmp_path / "session_cache")
    node_cache = NodeCache("main-test", session_cache)

    reconstruct_execution_history_and_cache(turn1_messages, exec_history, node_cache)

    # Simulate Turn 2 calling get_execution_history (just as NodeRuntime._execute_tool_call does)
    exec_history.record(
        "get_execution_history", "call_dc99926b5d2b54fb4aebf35c101c2f7d"
    )
    memory_tools = HarnessMemoryTools(exec_history, lambda: node_cache)
    xml_result = memory_tools.get_execution_history(
        start_position=None, limit=50
    ).content

    assert "<total>1</total>" not in xml_result
    assert "<total>9</total>" in xml_result
    expected_calls = [
        ("1", "mock-skill", "call_1aa42f66bfbf46afd9df928e58baed9d", "0_0_5"),
        ("2", "mock-sub-tool", "call_4adc8f5ba98d6e2476bc1074f2f70347", "0_0_0"),
        ("3", "shared-tool", "call_1f00e87e28ae524e9cf2481b98ff9a23", "0_0_1"),
        ("4", "nested-skill", "call_0365fde7dd39d4594463726635aa9b19", "0_0_4"),
        ("5", "nested-sub-tool", "call_1bf81064ebe820a501bf8f096382ece5", "0_0_2"),
        ("6", "shared-tool", "call_65a86e04e7a5783111328452169e9534", "0_0_3"),
        ("7", "exit_agent", "call_7f62bb716b25cb23cb7ea5f89de56801", None),
        ("8", "exit_agent", "call_52de5bdcf770b7f3f86600c080e63cb9", None),
        ("9", "get_execution_history", "call_dc99926b5d2b54fb4aebf35c101c2f7d", None),
    ]
    for pos, name, call_id, out_id in expected_calls:
        assert f"<position>{pos}</position>" in xml_result
        assert f"<name>{name}</name>" in xml_result
        assert f"<id>{call_id}</id>" in xml_result
        if out_id is not None:
            assert f"<saved-output-id>{out_id}</saved-output-id>" in xml_result

    # Also verify that saved outputs are recoverable via read_tool_output
    saved_0 = memory_tools.read_tool_output("0_0_0").content
    assert "Mock sub-tool result: chiamata diretta dal livello mock-skill" in saved_0
    saved_5 = memory_tools.read_tool_output("0_0_5").content
    assert "Demo dimostrativa completata" in saved_5
