/* Model-facing agent scope instructions. Keep rendering separate from policy. */
#ifndef DS4_AGENTIC_SETTINGS_H
#define DS4_AGENTIC_SETTINGS_H

static const char agentic_scope_open[] =
    "<agent_runtime>\n";

static const char agentic_allowed_tools[] =
    "Agent runtime scope for this turn. Allowed ordinary tools: ";

static const char agentic_no_capabilities[] =
    "(none)";

static const char agentic_list_separator[] =
    ", ";

static const char agentic_allowed_skills[] =
    ". Allowed skills: ";

static const char agentic_capability_guidance[] =
    ". These lists are permissions, not an unconditional request "
    "to call a capability. Use DSML to invoke a permitted capability "
    "when needed for unfinished work. Tool and skill results already "
    "in the conversation represent completed work: do not repeat "
    "successful calls just because the original request mentions them.\n";

static const char agentic_skill_return_guidance[] =
    "The invoked skill has already finished and EXITED. "
    "You are now back in its caller, outside that skill. "
    "Do not execute that completed skill's exit instructions "
    "again: its exit has already produced the returned result. "
    "Its internal conversation, tool calls and nested skills "
    "have been removed from this context and replaced by this "
    "result. Their absence is not evidence of unfinished work. "
    "This is the result of executing the skill, not an instruction "
    "to enter it. Resume the caller after that skill invocation; "
    "do not reopen it to replay steps hidden by compaction. "
    "If the result reports a failure or missing work, handle that "
    "explicitly; otherwise continue with the next caller step "
    "or answer the user when the requested work is complete.\n";

static const char agentic_required_tool_guidance[] =
    "This turn requires the single available zero-argument tool. "
    "You MUST emit its valid DSML call before the output-token budget "
    "ends. Keep reasoning brief; do not substitute a prose plan, "
    "checklist, or promise for the actual call.\n";

static const char agentic_scope_close[] =
    "</agent_runtime>";

/* Agent context persistence diagnostics. */
static const char agentic_invalid_checkpoint_request[] = "invalid agent context checkpoint request";
static const char agentic_invalid_checkpoint_frontier[] = "invalid agent context frontier";
static const char agentic_missing_checkpoint_payload[] = "agent context has no portable session payload";
static const char agentic_checkpoint_create_error[] = "failed to create agent context checkpoint: %s";
static const char agentic_checkpoint_open_error[] = "failed to open agent context checkpoint: %s";
static const char agentic_checkpoint_flush_error[] = "failed to flush agent context checkpoint: %s";
static const char agentic_checkpoint_sync_error[] = "failed to sync agent context checkpoint: %s";
static const char agentic_checkpoint_close_error[] = "failed to close agent context checkpoint: %s";
static const char agentic_checkpoint_budget_error[] = "interactive checkpoint byte budget exceeded";
static const char agentic_checkpoint_missing_call[] = "agent context has no live tool call";
static const char agentic_checkpoint_unavailable[] = "agent context checkpoint is unavailable";

/* Product text for this domain; preserve formatting placeholders. */
#define DS4_AGENT_TEXT_AGENTIC_CAPABILITIES_CANNOT_BE_USED_WITH_TOOL_CHOICE_NONE "agentic capabilities cannot be used with tool_choice=none"
#define DS4_AGENT_TEXT_AGENTIC_CAPABILITY_IS_NOT_REGISTERED_S "agentic capability is not registered: %s"
#define DS4_AGENT_TEXT_AGENTIC_CAPABILITY_HAS_UNSUPPORTED_JSON_SCHEMA_S "agentic capability has unsupported JSON schema: %s"
#define DS4_AGENT_TEXT_AGENTIC_CAPABILITY_CANNOT_BE_BOTH_TOOL_AND_SKILL_S "agentic capability cannot be both tool and skill: %s"
#define DS4_AGENT_TEXT_AGENTIC_CAPABILITY_LIST_MUST_BE_AN_ARRAY "agentic capability list must be an array"
#define DS4_AGENT_TEXT_AGENTIC_CAPABILITY_NAMES_MUST_BE_NON_EMPTY_STRINGS "agentic capability names must be non-empty strings"
#define DS4_AGENT_TEXT_DUPLICATE_AGENTIC_CAPABILITY_S "duplicate agentic capability: %s"
#define DS4_AGENT_TEXT_INVALID_AGENTIC_CAPABILITY_ARRAY "invalid agentic capability array"
#define DS4_AGENT_TEXT_UNTERMINATED_AGENTIC_CAPABILITY_ARRAY "unterminated agentic capability array"
#define DS4_AGENT_TEXT_AGENTIC_MUST_BE_AN_OBJECT "agentic must be an object"
#define DS4_AGENT_TEXT_AGENTIC_OPERATION_MUST_BE_RETURN "agentic.operation must be return"
#define DS4_AGENT_TEXT_AGENTIC_SKILL_CALL_ID_MUST_BE_A_NON_EMPTY_STRING "agentic.skill_call_id must be a non-empty string"
#define DS4_AGENT_TEXT_AGENTIC_AGENT_CONTEXT_ID_MUST_BE_A_NON_EMPTY_STRING "agentic.agent_context_id must be a non-empty string"
#define DS4_AGENT_TEXT_INVALID_AGENTIC_INTERACTIVE_OBJECT "invalid agentic.interactive object"
#define DS4_AGENT_TEXT_UNSUPPORTED_AGENTIC_FIELD_S "unsupported agentic field: %s"
#define DS4_AGENT_TEXT_AGENTIC_REQUIRES_AGENT_CONTEXT_ID_ALLOWED_TOOLS_AND_ALLOWED_SKILLS "agentic requires agent_context_id, allowed_tools and allowed_skills"
#define DS4_AGENT_TEXT_AGENTIC_RETURN_REQUIRES_OPERATION_AND_SKILL_CALL_ID_TOGETHER "agentic return requires operation and skill_call_id together"
#define DS4_AGENT_TEXT_AGENTIC_IS_SUPPORTED_ONLY_BY_V1_RESPONSES "agentic is supported only by /v1/responses"
#define DS4_AGENT_TEXT_AGENTIC_RETURN_INPUT_MUST_CONTAIN_FUNCTION_CALL_OUTPUT_FOR_S "agentic return input must contain function_call_output for %s"
#define DS4_AGENT_TEXT_FAILED_TO_DELETE_SKILL_CHECKPOINT_PATH_S_ERROR_S "ds4-server: failed to delete skill checkpoint path=%s error=%s"
#define DS4_AGENT_TEXT_REMOVED_ORPHAN_AGENTIC_CHECKPOINTS_FILES_LLU_BYTES_LLU "ds4-server: removed orphan agentic checkpoints files=%llu bytes=%llu"
#define DS4_AGENT_TEXT_INVALID_SKILL_CHECKPOINT_REQUEST "invalid skill checkpoint request"
#define DS4_AGENT_TEXT_ACTIVE_SKILL_CHECKPOINTS_EXCEED_THE_QWEN_SESSION_CACHE_BUDGET "active skill checkpoints exceed the Qwen session cache budget"
#define DS4_AGENT_TEXT_FAILED_TO_RESERVE_QWEN_SESSION_CACHE_SPACE_FOR_SKILL "failed to reserve Qwen session cache space for skill checkpoint"
#define DS4_AGENT_TEXT_FAILED_TO_CREATE_SKILL_CHECKPOINT_S "failed to create skill checkpoint: %s"
#define DS4_AGENT_TEXT_FAILED_TO_OPEN_SKILL_CHECKPOINT_S "failed to open skill checkpoint: %s"
#define DS4_AGENT_TEXT_FAILED_TO_FLUSH_SKILL_CHECKPOINT_S "failed to flush skill checkpoint: %s"
#define DS4_AGENT_TEXT_FAILED_TO_SYNC_SKILL_CHECKPOINT_S "failed to sync skill checkpoint: %s"
#define DS4_AGENT_TEXT_FAILED_TO_CHECKSUM_SKILL_CHECKPOINT "failed to checksum skill checkpoint"
#define DS4_AGENT_TEXT_FAILED_TO_CLOSE_SKILL_CHECKPOINT_S "failed to close skill checkpoint: %s"
#define DS4_AGENT_TEXT_FAILED_TO_PUBLISH_SKILL_CHECKPOINT_S "failed to publish skill checkpoint: %s"
#define DS4_AGENT_TEXT_SKILL_CHECKPOINT_LOG "SKILL_CHECKPOINT call_id=%s skill=%s sequence=%d checkpoint_tokens=%d checkpoint_file=%s checkpoint_bytes=%llu checkpoint_stage_ms=%.3f checkpoint_write_ms=%.3f skill_checkpoint_count=%llu skill_checkpoint_live_bytes=%llu skill_checkpoint_peak_bytes=%llu"
#define DS4_AGENT_TEXT_INVALID_SKILL_RESTORE_REQUEST "invalid skill restore request"
#define DS4_AGENT_TEXT_CORRUPT_OR_TRUNCATED_SKILL_CHECKPOINT "corrupt or truncated skill checkpoint"
#define DS4_AGENT_TEXT_AGENT_CONTEXT_SYSTEM_PROMPT_OR_TOOL_REGISTRY_CHANGED "Agent context system prompt or tool registry changed"
#define DS4_AGENT_TEXT_AGENT_CONTEXT_CHECKPOINT_RESTORE_FAILED "Agent context checkpoint restore failed"
#define DS4_AGENT_TEXT_AGENTIC_TOOL_REGISTRY_CHANGED_WITHIN_AN_AGENT_CONTEXT "Agentic tool registry changed within an agent context"
#define DS4_AGENT_TEXT_AGENTIC_SKILL_CHECKPOINT_IS_NOT_AVAILABLE_FOR_THIS_CALL_ID "Agentic skill checkpoint is not available for this call_id"
#define DS4_AGENT_TEXT_AGENTIC_SKILL_CHECKPOINT_RESTORE_FAILED "Agentic skill checkpoint restore failed"
#define DS4_AGENT_TEXT_SKILL_RETURN_LOG "SKILL_RETURN call_id=%s restored_tokens=%d discarded_child_tokens=%d result_prefill_tokens=%d checkpoint_read_ms=%.3f checkpoint_restore_ms=%.3f checkpoint_deleted=%s skill_return_count=%llu skill_checkpoint_live_bytes=%llu"
#define DS4_AGENT_TEXT_FAILED_TO_CREATE_AGENTIC_CHECKPOINT_DIRECTORY_S "ds4-server: failed to create agentic checkpoint directory: %s"
#define DS4_AGENT_TEXT_AGENTIC_SKILL_CHECKPOINTS_DIR_S "ds4-server: agentic skill checkpoints dir=%s"

#define DS4_AGENT_TEXT_CHECKPOINT_SAVED_LOG "AGENT_CONTEXT_CHECKPOINT slot=%d context_id=%s checkpoint_tokens=%d system_tokens=%d checkpoint_file=%s checkpoint_bytes=%llu checkpoint_write_ms=%.3f agent_context_checkpoint_count=%llu"

#define DS4_AGENT_TEXT_CHECKPOINT_RESTORED_LOG "AGENT_CONTEXT_RESTORE slot=%d context_id=%s restored_tokens=%d discarded_context_tokens=%d result_prefill_tokens=%d checkpoint_read_ms=%.3f checkpoint_deleted=%s agent_context_restore_count=%llu"

#endif
