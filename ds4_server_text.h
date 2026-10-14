#ifndef DS4_SERVER_TEXT_H
#define DS4_SERVER_TEXT_H

/* Product text owned by the server. Macros preserve literal concatenation. */
#define DS4_SERVER_TEXT_OUT_OF_MEMORY "out of memory"
#define DS4_SERVER_TEXT_BUFFER_OVERFLOW "buffer overflow"
#define DS4_SERVER_TEXT_VSNPRINTF_FAILED "vsnprintf failed"
#define DS4_SERVER_TEXT_IGNORE_EOS_REQUIRES_AN_EXPLICIT_TEMPERATURE_OF "ignore_eos requires an explicit temperature of 0"
#define DS4_SERVER_TEXT_UNSUPPORTED_CONTENT_TYPE_ONLY_TEXT_AND_IMAGES_ARE_SUPPORTED \
    "Unsupported content type: only text and images are supported. " \
    "Convert documents or other attachments to text or images before sending."
#define DS4_SERVER_TEXT_TOO_MANY_IMAGES_AT_MOST_ARE_ALLOWED "too many images; at most 16 are allowed"
#define DS4_SERVER_TEXT_IMAGE_INPUT_REQUIRES_STARTING_DS4_SERVER_WITH_VISION_FILE "image input requires starting ds4-server with --vision FILE"
#define DS4_SERVER_TEXT_IMAGE_MARKER_WAS_LOST_WHILE_RENDERING_THE_REQUEST "image marker was lost while rendering the request"
#define DS4_SERVER_TEXT_RESPONSES_CONTINUATION_STATE_IS_NOT_AVAILABLE_FOR_CALL_ID_S "Responses continuation state is not available for call_id %s; retry by replaying the full input history"
#define DS4_SERVER_TEXT_ANTHROPIC_CONTINUATION_STATE_IS_NOT_AVAILABLE_FOR_TOOL_USE_ID_S "Anthropic continuation state is not available for tool_use_id %s; retry by replaying the full messages history"
#define DS4_SERVER_TEXT_INVALID_OR_UNSUPPORTED_JSON_SCHEMA_RESPONSE_FORMAT "invalid or unsupported json_schema response_format"
#define DS4_SERVER_TEXT_MISSING_MESSAGES "missing messages"
#define DS4_SERVER_TEXT_JSON_SCHEMA_RESPONSE_FORMAT_CANNOT_BE_COMBINED_WITH_TOOLS "json_schema response_format cannot be combined with tools"
#define DS4_SERVER_TEXT_INVALID_JSON_REQUEST "invalid JSON request"
#define DS4_SERVER_TEXT_TOOL_CHOICE_S_NOT_SUPPORTED "tool_choice=%s not supported"
#define DS4_SERVER_TEXT_FORCED_TOOL_CHOICE_NOT_SUPPORTED "forced tool_choice not supported"
#define DS4_SERVER_TEXT_INVALID_OR_UNSUPPORTED_JSON_SCHEMA_TEXT_FORMAT "invalid or unsupported json_schema text.format"
#define DS4_SERVER_TEXT_S_IS_NOT_SUPPORTED_REPLAY_FULL_INPUT_INSTEAD "%s is not supported; replay full input instead"
#define DS4_SERVER_TEXT_MISSING_INPUT "missing input"
#define DS4_SERVER_TEXT_TOOL_CHOICE_REQUIRED_NEEDS_AT_LEAST_ONE_TOOL "tool_choice=required needs at least one tool"
#define DS4_SERVER_TEXT_JSON_SCHEMA_TEXT_FORMAT_CANNOT_BE_COMBINED_WITH_TOOLS "json_schema text.format cannot be combined with tools"
#define DS4_SERVER_TEXT_MISSING_PROMPT "missing prompt"
#define DS4_SERVER_TEXT_THINKING_NOT_CLOSED_IGNORING_INCOMPLETE_DSML_IN_REASONING "ds4-server: thinking not closed, ignoring incomplete DSML in reasoning\n"
#define DS4_SERVER_TEXT_THINKING_NOT_CLOSED_IGNORING_INCOMPLETE_GLM_TOOL_CALLS_IN "ds4-server: thinking not closed, ignoring incomplete GLM tool calls in reasoning\n"
#define DS4_SERVER_TEXT_THINKING_NOT_CLOSED_IGNORING_INCOMPLETE_QWEN_TOOL_CALLS_IN "ds4-server: thinking not closed, ignoring incomplete Qwen tool calls in reasoning\n"
#define DS4_SERVER_TEXT_INVALID_TOOL_CALL "invalid tool call"
#define DS4_SERVER_TEXT_NO_CONTENT "No Content"
#define DS4_SERVER_TEXT_BAD_REQUEST "Bad Request"
#define DS4_SERVER_TEXT_NOT_FOUND "Not Found"
#define DS4_SERVER_TEXT_INTERNAL_SERVER_ERROR "Internal Server Error"
#define DS4_SERVER_TEXT_PROMPT_HAS_D_TOKENS_BUT_THE_CONFIGURED_CONTEXT_SIZE "Prompt has %d tokens, but the configured context size is %d tokens"
#define DS4_SERVER_TEXT_INTERNAL_SERVER_ERROR_SSE_ERROR_EVENT "internal server error"
#define DS4_SERVER_TEXT_ERROR_CODE_SERVER_ERROR_MESSAGE_GENERATION_FAILED \
    ",\"error\":{\"code\":\"server_error\"," \
    "\"message\":\"generation failed\"}"
#define DS4_SERVER_TEXT_DUPLICATE_SKILL_CALL_ID_S "duplicate skill call_id: %s"
#define DS4_SERVER_TEXT_VISION_EMBEDDING_CACHE_HIT "ds4-server: vision embedding cache hit"
#define DS4_SERVER_TEXT_VISION_EMBEDDING_ENCODED_CACHE_ZU_BYTES "ds4-server: vision embedding encoded; cache=%zu bytes"
#define DS4_SERVER_TEXT_INVALID_RESIDENT_KV_FRONTIER_SIZE "invalid resident KV frontier size"
#define DS4_SERVER_TEXT_OUT_OF_MEMORY_FOR_RESIDENT_KV_FRONTIER "out of memory for resident KV frontier"
#define DS4_SERVER_TEXT_FAILED_TO_OPEN_RESIDENT_KV_MEMORY_STREAM "failed to open resident KV memory stream"
#define DS4_SERVER_TEXT_FAILED_TO_FINALIZE_RESIDENT_KV_FRONTIER "failed to finalize resident KV frontier"
#define DS4_SERVER_TEXT_COMPACT_FRONTIER_LOST_ITS_LIVE_KV_PREFIX "compact frontier lost its live KV prefix"
#define DS4_SERVER_TEXT_FAILED_TO_OPEN_RESIDENT_KV_FRONTIER "failed to open resident KV frontier"
#define DS4_SERVER_TEXT_FAILED_TO_CLOSE_RESIDENT_KV_FRONTIER "failed to close resident KV frontier"
#define DS4_SERVER_TEXT_AGENT_SYSTEM_BOUNDARY_SYSTEM_TOKENS_D_PROMPT_TOKENS_D_SYNTAX_S "ds4-server: agent-system boundary system_tokens=%d prompt_tokens=%d syntax=%s"
#define DS4_SERVER_TEXT_KV_CACHE_DISCARDED_REASON_PREFILL_FAILED_FILE_S "ds4-server: kv cache discarded reason=prefill-failed file=%s"
#define DS4_SERVER_TEXT_KV_CACHE_FAILED_TO_DISCARD_PREFILL_FAILED_FILE_S "ds4-server: kv cache failed to discard prefill-failed file=%s: %s"
#define DS4_SERVER_TEXT_CACHE_DECISION_LOG \
    "\n--- cache decision ---\n" \
    "live_tokens_before: %d\n" \
    "prompt_tokens: %d\n" \
    "live_prompt_common: %d\n" \
    "memory_token_reusable: %d\n" \
    "memory_miss_reason: %s\n" \
    "tool_replay: mem=%d disk=%d canonical=%d missing_ids=%d\n" \
    "cache_source: %s\n" \
    "cached_tokens: %d\n" \
    "disk_cached_tokens: %d\n"
#define DS4_SERVER_TEXT_RAW_REQUEST_JSON "\n--- raw request json ---\n"
#define DS4_SERVER_TEXT_RENDERED_PROMPT "\n--- rendered prompt ---\n"
#define DS4_SERVER_TEXT_GENERATED_TEXT "\n--- generated text ---\n"
#define DS4_SERVER_TEXT_PARSED_MESSAGE_TRACE "\n\n--- parsed message ---\nfinish: %s\ngenerated_tokens: %d\ndsml_start: %d\ndsml_end: %d\nelapsed_sec: %.3f\n"
#define DS4_SERVER_TEXT_END_REQUEST_LLU "\n===== end request %llu =====\n"
#define DS4_SERVER_TEXT_S_CTX_S_GEN_D_S_S_DECODING_CHUNK "ds4-server: %s ctx=%s gen=%d%s%s decoding chunk=%.2f t/s avg=%.2f t/s %.3fs"
#define DS4_SERVER_TEXT_TOOL_ERROR_INVALID_DSML_TOOL_CALL "Tool error: invalid DSML tool call"
#define DS4_SERVER_TEXT_TOOL_ERROR_INVALID_GLM_TOOL_CALL "Tool error: invalid GLM tool call"
#define DS4_SERVER_TEXT_TOOL_ERROR_INVALID_QWEN_TOOL_CALL "Tool error: invalid Qwen tool call"
#define DS4_SERVER_TEXT_PREFILL_MADE_NO_PROGRESS "prefill made no progress"
#define DS4_SERVER_TEXT_RESIDENT_SYSTEM_FRONTIER_CAPTURED_TOKENS_D_BYTES_F_MIB "ds4-server: resident system frontier captured tokens=%d bytes=%.2f MiB location=%s %.3fms"
#define DS4_SERVER_TEXT_LIVE_SESSION_IS_UNAVAILABLE "live session is unavailable"
#define DS4_SERVER_TEXT_CONSTRAINT_DIVERGENCE_LOG \
    "ds4-server: constraint divergence kind=%s token=%d " \
    "analyzed_allowed=%s oracle_allowed=%s piece_len=%zu piece_hex=%s " \
    "raw_len=%zu dsml_mode=%d decode=%d tracker_pos=%zu " \
    "inside_invoke=%s active_prop=%d invoke_body_start=%zu " \
    "param_value_start=%zu saw_invoke=%s"
#define DS4_SERVER_TEXT_TOOL_CALLS_CTX_S_S_S_N_D_RAW_TOOL_TEXT "ds4-server: tool calls ctx=%s%s%s n=%d raw_tool_text=%d ids=[%s] names=[%s]"
#define DS4_SERVER_TEXT_S_CTX_S_S_S_PREFILL_FAILED_AFTER_STREAM "ds4-server: %s ctx=%s%s%s prefill failed after stream closed: %s"
#define DS4_SERVER_TEXT_S_CTX_S_S_S_PREFILL_SSE_ERROR_FAILED "ds4-server: %s ctx=%s%s%s prefill SSE error failed: %s"
#define DS4_SERVER_TEXT_THINKING_LIVE_CHECKPOINT_REMEMBERED_CTX_S_LIVE_D_VISIBLE "ds4-server: thinking live checkpoint remembered ctx=%s live=%d visible=%zu"
#define DS4_SERVER_TEXT_THINKING_LIVE_CHECKPOINT_REMEMBERED_LIVE_D_VISIBLE_ZU "thinking live checkpoint remembered: live=%d visible=%zu"
#define DS4_SERVER_TEXT_QWEN_TOOL_TURN_VISIBLE_CHECKPOINT_REMEMBERED_CTX_S_LIVE "ds4-server: qwen tool-turn visible checkpoint remembered ctx=%s live=%d visible=%zu"
#define DS4_SERVER_TEXT_POST_RESPONSE_CHECKPOINT_REBUILT_SOURCE_MEMORY_REQUEST_CACHED_D "ds4-server: post-response checkpoint rebuilt source=memory-request cached=%d response_prefill=%d target=%d location=%s %.3fms"
#define DS4_SERVER_TEXT_POST_RESPONSE_CHECKPOINT_REBUILT_VIA_REQUEST_FRONTIER_CACHED_D "post-response checkpoint rebuilt via request frontier: cached=%d response_prefill=%d target=%d"
#define DS4_SERVER_TEXT_RESIDENT_REQUEST_FRONTIER_RESTORE_FAILED_CTX_S_ERROR_S "ds4-server: resident request frontier restore failed ctx=%s error=\"%s\"; falling back"
#define DS4_SERVER_TEXT_UNKNOWN_ERROR "unknown error"
#define DS4_SERVER_TEXT_MULTIMODAL_TOOL_CHECKPOINT_CANONICALIZATION_SKIPPED_CTX_S_COMMON_D "ds4-server: multimodal tool checkpoint canonicalization skipped ctx=%s common=%d live=%d canonical=%d reason=preserve-image-state"
#define DS4_SERVER_TEXT_MULTIMODAL_TOOL_CHECKPOINT_CANONICALIZATION_SKIPPED_COMMON_D_LIVE_D "multimodal tool checkpoint canonicalization skipped: common=%d live=%d canonical=%d"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZATION_SKIPPED_COMMON_D_PROMPT_D_LIVE "tool checkpoint canonicalization skipped: common=%d prompt=%d live=%d canonical=%d"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZATION_CROSSES_BPE_SEAM_COMMON_D_PROMPT "tool checkpoint canonicalization crosses BPE seam: common=%d prompt=%d"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZED_CTX_S_COMMON_D_LIVE_D "ds4-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZED_COMMON_D_LIVE_D_CANONICAL_D "tool checkpoint canonicalized: common=%d live=%d canonical=%d"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZATION_NEEDS_D_TOKENS_REBUILD_CTX_S "ds4-server: tool checkpoint canonicalization needs %d tokens rebuild ctx=%s request_ctx=%s reason=canonical-tail-rewrite tail=%d discard=%d common=%d live=%d target=%d cached=%d source=%s%s%s"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_REBUILD "tool checkpoint rebuild"
#define DS4_SERVER_TEXT_CHECKPOINT_FROM_MEMORY_LOG "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=memory-system cached=%d replay=%d target=%d %.3fs"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZED_VIA_RESIDENT_SYSTEM_COMMON_D_LIVE "tool checkpoint canonicalized via resident system: common=%d live=%d canonical=%d cached=%d"
#define DS4_SERVER_TEXT_CHECKPOINT_FROM_DISK_LOG "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=disk cached=%d replay=%d target=%d %.3fs"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZED_VIA_DISK_COMMON_D_LIVE_D "tool checkpoint canonicalized via disk: common=%d live=%d canonical=%d cached=%d file=%s"
#define DS4_SERVER_TEXT_CHECKPOINT_FROM_REPLAY_LOG "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=full cached=0 replay=%d target=%d %.3fs"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZED_VIA_REBUILD_COMMON_D_LIVE_D "tool checkpoint canonicalized via rebuild: common=%d live=%d canonical=%d reason=%s"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_REBUILD_FAILED_CTX_S_REQUEST_CTX_S_SOURCE "ds4-server: tool checkpoint rebuild failed ctx=%s request_ctx=%s source=%s cached=%d replay=%d target=%d error=\"%s\""
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZATION_FAILED_AFTER_REBUILD_REQUEST_S "tool checkpoint canonicalization failed after rebuild request: %s"
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZATION_FAILED_CTX_S_COMMON_D_LIVE "ds4-server: tool checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\""
#define DS4_SERVER_TEXT_TOOL_CHECKPOINT_CANONICALIZATION_FAILED_S "tool checkpoint canonicalization failed: %s"
#define DS4_SERVER_TEXT_CLIENT_DISCONNECTED "client disconnected"
#define DS4_SERVER_TEXT_SHUTDOWN_REQUESTED "shutdown requested"
#define DS4_SERVER_TEXT_SESSION_ALREADY_HAS_A_DECODE_IN_FLIGHT "session already has a decode in flight"
#define DS4_SERVER_TEXT_DECODE_INTERRUPTED "decode interrupted"
#define DS4_SERVER_TEXT_DECODE_BATCH_COUNT_D_ELAPSED_F_MS_STATUS_S "ds4-server: decode batch count=%d elapsed=%.3f ms status=%s"
#define DS4_SERVER_TEXT_BATCHED_DECODE_FAILED "batched decode failed"
#define DS4_SERVER_TEXT_RESPONSES_CONTINUATION_STATE_IS_NOT_AVAILABLE_RETRY_BY_REPLAYING "Responses continuation state is not available; retry by replaying the full input history"
#define DS4_SERVER_TEXT_ANTHROPIC_CONTINUATION_STATE_IS_NOT_AVAILABLE_RETRY_BY_REPLAYING "Anthropic continuation state is not available; retry by replaying the full messages history"
#define DS4_SERVER_TEXT_REWOUND_GLM_LIVE_PREFIX_FROM_D_TO_D_FINAL "ds4-server: rewound GLM live prefix from %d to %d; final prompt token will be reevaluated"
#define DS4_SERVER_TEXT_GLM_LIVE_PREFIX_REWIND_FROM_D_TO_D_REQUIRES "ds4-server: GLM live prefix rewind from %d to %d requires rebuild"
#define DS4_SERVER_TEXT_RESIDENT_SYSTEM_FRONTIER_RESTORED_TOKENS_D_FMS "ds4-server: resident system frontier restored tokens=%d %.3fms"
#define DS4_SERVER_TEXT_RESIDENT_SYSTEM_FRONTIER_RESTORE_FAILED_S "ds4-server: resident system frontier restore failed: %s"
#define DS4_SERVER_TEXT_LIVE_KV_CACHE_MISS_S_LIVE_D_PROMPT_D "ds4-server: live kv cache miss%s live=%d prompt=%d common=%d vision=%s reason=%s"
#define DS4_SERVER_TEXT_MULTIMODAL_LIVE_KV_HIT_IMAGES_ZU_CACHED_D_PROMPT "ds4-server: multimodal live kv hit images=%zu cached=%d prompt=%d identity=fingerprint-match"
#define DS4_SERVER_TEXT_RESPONSES_LIVE_CONTINUATION_RESPPROTO_MATCH_S_IDS_D_CACHED "ds4-server: responses live continuation RESPPROTO match=%s ids=%d cached=%d prompt=%d"
#define DS4_SERVER_TEXT_ANTHROPIC_LIVE_CONTINUATION_MATCH_TOOL_OUTPUT_IDS_IDS_D "ds4-server: anthropic live continuation match=tool-output-ids ids=%d cached=%d prompt=%d"
#define DS4_SERVER_TEXT_THINKING_LIVE_CONTINUATION_MATCH_VISIBLE_PREFIX_CACHED_D_PROMPT "ds4-server: thinking live continuation match=visible-prefix cached=%d prompt=%d"
#define DS4_SERVER_TEXT_RESPONSES_REPLAY_RESPPROTO_MISSING_REASONING_STATE_CONTINUING_FROM_VISIBLE "ds4-server: responses replay RESPPROTO missing reasoning state; continuing from visible history source=%s cached=%d prompt=%d"
#define DS4_SERVER_TEXT_RESPONSES_REPLAY_MISSING_REASONING_STATE_CONTINUING_FROM_VISIBLE_HISTORY "responses replay missing reasoning state; continuing from visible history source=%s cached=%d"
#define DS4_SERVER_TEXT_S_CTX_S_S_S_PROMPT_START "ds4-server: %s ctx=%s%s%s prompt start"
#define DS4_SERVER_TEXT_RESIDENT_SYSTEM_FRONTIER_CAPTURE_FAILED_S "ds4-server: resident system frontier capture failed: %s"
#define DS4_SERVER_TEXT_BOUNDARY_UNAVAILABLE "boundary unavailable"
#define DS4_SERVER_TEXT_CANCELLED_DURING_PREFILL "cancelled during prefill"
#define DS4_SERVER_TEXT_PREFILL_FAILED_S "prefill failed: %s"
#define DS4_SERVER_TEXT_CANCELLED_AFTER_PREFILL "cancelled after prefill"
#define DS4_SERVER_TEXT_S_CTX_S_S_S_PROMPT_DONE_FS "ds4-server: %s ctx=%s%s%s prompt done %.3fs"
#define DS4_SERVER_TEXT_RESPONSE_FRONTIER_CAPTURED_TOKENS_D_BYTES_F_MIB_LOCATION "ds4-server: response frontier captured tokens=%d bytes=%.2f MiB location=%s %.3fms"
#define DS4_SERVER_TEXT_RESPONSE_FRONTIER_CAPTURE_FAILED_S "ds4-server: response frontier capture failed: %s"
#define DS4_SERVER_TEXT_S_CTX_S_S_S_STREAM_CLOSED_DURING_PREFILL "ds4-server: %s ctx=%s%s%s stream closed during prefill"
#define DS4_SERVER_TEXT_S_CTX_S_S_S_SSE_HEADERS_FAILED "ds4-server: %s ctx=%s%s%s sse headers failed"
#define DS4_SERVER_TEXT_CHAT_CTX_S_ANTHROPIC_STREAM_START_FAILED "ds4-server: chat ctx=%s anthropic stream start failed"
#define DS4_SERVER_TEXT_CHAT_CTX_S_OPENAI_ROLE_CHUNK_FAILED "ds4-server: chat ctx=%s openai role chunk failed"
#define DS4_SERVER_TEXT_CHAT_CTX_S_S_S_RESPONSES_CREATED_EVENT_FAILED "ds4-server: chat ctx=%s%s%s responses created event failed"
#define DS4_SERVER_TEXT_CONTEXT_LIMIT "context limit"
#define DS4_SERVER_TEXT_OUTPUT_LIMIT "output limit"
#define DS4_SERVER_TEXT_PREFILL_DONE_DECODE_MAX_D_CTX_ROOM_D "prefill done; decode_max=%d ctx_room=%d"
#define DS4_SERVER_TEXT_S_CONSTRAINED_DECODER_HAS_NO_VALID_TOKEN "%s constrained decoder has no valid token"
#define DS4_SERVER_TEXT_STRUCTURED_OUTPUT "structured output"
#define DS4_SERVER_TEXT_DECODE_RETURNED_NO_TOKENS "decode returned no tokens"
#define DS4_SERVER_TEXT_STOP_TOKEN "stop token"
#define DS4_SERVER_TEXT_CLIENT_STREAM_WRITE_FAILED "client stream write failed"
#define DS4_SERVER_TEXT_CHAT_CTX_S_S_S_RECOVERED_A_COMPLETE_TOOL "ds4-server: chat ctx=%s%s%s recovered a complete tool call from unclosed reasoning after %d generated tokens"
#define DS4_SERVER_TEXT_RECOVERED_COMPLETE_TOOL_CALL_FROM_UNCLOSED_REASONING_AFTER_D "recovered complete tool call from unclosed reasoning after %d generated tokens"
#define DS4_SERVER_TEXT_CHAT_CTX_S_S_S_IGNORED_ORPHAN_TOOL_CALL "ds4-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens"
#define DS4_SERVER_TEXT_IGNORED_ORPHAN_TOOL_CALL_END_MARKER_AFTER_D_GENERATED "ignored orphan tool-call end marker after %d generated tokens"
#define DS4_SERVER_TEXT_ENTERED_TOOL_CALL_BLOCK_AFTER_D_GENERATED_TOKENS "entered tool-call block after %d generated tokens"
#define DS4_SERVER_TEXT_CLOSED_TOOL_CALL_BLOCK_AFTER_D_GENERATED_TOKENS "closed tool-call block after %d generated tokens"
#define DS4_SERVER_TEXT_PROGRESS_GEN_D_DSML_START_D_DSML_END_D "progress gen=%d dsml_start=%d dsml_end=%d"
#define DS4_SERVER_TEXT_CLIENT_STOP_SEQUENCE "client stop sequence"
#define DS4_SERVER_TEXT_SPECULATIVE_BOUNDARY_KEPT_D_DISCARDED_D_RESAMPLE_D "speculative boundary: kept=%d discarded=%d resample=%d"
#define DS4_SERVER_TEXT_OLD_TEXT_DRAFTS_CYCLES_LLU_PROPOSED_LLU_ACCEPTED_LLU_SAMPLED_ROWS "ds4-server: old_text drafts cycles=%llu proposed=%llu accepted=%llu sampled_rows=%llu fallbacks=%llu batches=32/64\n"
#define DS4_SERVER_TEXT_PHASE_PROFILE_CTX_S_GEN_D_MODE_S_WALL \
    "ds4-server: phase profile ctx=%s gen=%d mode=%s wall=%.3fms " \
    "forced_build=%.3fms/%llu forced_sync=%.3fms/%llu/%llu_tok " \
    "forced_prefix_probe=%.3fms sampling_mask_build=%.3fms " \
    "oracle_compare=%.3fms/%llu/%llu_div constraint_cpu=%.3fms " \
    "constraint_cpu_exposed=%.3fms constraint_cpu_overlapped=0.000ms " \
    "filter_setup=%.3fms filter=%.3fms filtered_sample=%.3fms/%llu " \
    "plain_sample=%.3fms/%llu eval=%.3fms/%llu residual=%.3fms " \
    "vocab=%llu filter_calls=%llu accepted=%llu finite_allowed=%llu " \
    "piece_bytes=%llu candidate_tokens_tested=%llu " \
    "parser_transition_count=%llu parser_bytes_visited=%llu " \
    "trie_nodes_visited=%llu subtrees_pruned=%llu " \
    "trie_leaf_tokens_emitted=%llu mask_cache_hit=%llu mask_cache_miss=%llu " \
    "grammar_compile_ms=%.3f grammar_jit_ms=%.3f " \
    "trie_compile_ms=%.3f trie_compiled_nodes=%llu trie_memory_bytes=%llu " \
    "static_mask_compile_ms=%.3f static_mask_memory_bytes=%llu " \
    "dynamic_frontier_size=%llu " \
    "search_static_steps=%llu search_static_fallbacks=%llu " \
    "mtp_constrained_target_only_steps=%llu " \
    "mtp_constrained_speculative_cycles=%llu " \
    "mtp_constrained_sampled_rows=%llu " \
    "mtp_constrained_accepted_drafts=%llu " \
    "mtp_constrained_fallbacks=%llu " \
    "constraint_state_checkpoint=%llu constraint_state_rollback=%llu " \
    "exhaustive_fallback_steps=%llu"
#define DS4_SERVER_TEXT_CANCELLED_DURING_GENERATION_AFTER_D_TOKENS "cancelled during generation after %d tokens"
#define DS4_SERVER_TEXT_STRUCTURED_OUTPUT_ENDED_BEFORE_A_SCHEMA_VALID_JSON_VALUE "structured output ended before a schema-valid JSON value"
#define DS4_SERVER_TEXT_INCOMPLETE_S_TOOL_CALL_STOP_S_TOKEN_D_GENERATED "ds4-server: incomplete %s tool call: stop=%s token=%d generated=%d limit=%d room=%d"
#define DS4_SERVER_TEXT_INCOMPLETE_TOOL_CALL_STOP_S_TOKEN_D_GENERATED_D "incomplete tool call: stop=%s token=%d generated=%d limit=%d room=%d"
#define DS4_SERVER_TEXT_CHAT_CTX_S_S_S_REPAIRED_UNTERMINATED_TOOL_CALL "ds4-server: chat ctx=%s%s%s repaired unterminated tool call (%d calls recovered)"
#define DS4_SERVER_TEXT_REPAIRED_UNTERMINATED_TOOL_CALL_D_CALLS_RECOVERED "repaired unterminated tool call (%d calls recovered)"
#define DS4_SERVER_TEXT_CHAT_CTX_S_S_S_UNTERMINATED_TOOL_CALL_CONTINUING "ds4-server: chat ctx=%s%s%s unterminated tool call; continuing with model-visible tool error"
#define DS4_SERVER_TEXT_UNTERMINATED_TOOL_CALL_CONTINUING_WITH_MODEL_VISIBLE_TOOL_ERROR "unterminated tool call; continuing with model-visible tool error"
#define DS4_SERVER_TEXT_UNTERMINATED_TOOL_CALL "unterminated tool call"
#define DS4_SERVER_TEXT_CHAT_CTX_S_S_S_TOOL_ERROR_CONTINUATION_APPENDED "ds4-server: chat ctx=%s%s%s tool-error continuation appended %d tokens"
#define DS4_SERVER_TEXT_TOOL_ERROR_CONTINUATION_APPENDED_D_TOKENS "tool-error continuation appended %d tokens"
#define DS4_SERVER_TEXT_INVALID_TOOL_CALL_RECOVERY_FAILED_S "invalid tool call recovery failed: %s"
#define DS4_SERVER_TEXT_UNTERMINATED_TOOL_CALL_S_TOKEN_D_GENERATED_D_LIMIT "unterminated tool call (%s; token=%d, generated=%d, limit=%d)"
#define DS4_SERVER_TEXT_CANCELLED_WHILE_FLUSHING_GENERATION "cancelled while flushing generation"
#define DS4_SERVER_TEXT_CHAT_CTX_S_S_S_INVALID_TOOL_CALL_CONTINUING "ds4-server: chat ctx=%s%s%s invalid tool call; continuing with model-visible tool error"
#define DS4_SERVER_TEXT_INVALID_TOOL_CALL_CONTINUING_WITH_MODEL_VISIBLE_TOOL_ERROR "invalid tool call; continuing with model-visible tool error"
#define DS4_SERVER_TEXT_CHAT_CTX_S_S_S_INVALID_TOOL_CALL_RETURNED "ds4-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s [text_len=%zu saw_start=%d saw_end=%d text_snippet: %.*s]"
#define DS4_SERVER_TEXT_CHAT_CTX_S_S_S_INVALID_TOOL_CALL_DSML_SNIPPET "ds4-server: chat ctx=%s%s%s invalid tool call dsml_snippet: %.*s"
#define DS4_SERVER_TEXT_INVALID_TOOL_CALL_RETURNED_AS_ASSISTANT_TEXT_FINISH_S "invalid tool call returned as assistant text finish=%s"
#define DS4_SERVER_TEXT_CANCELLED_DURING_RESPONSE_PARSING "cancelled during response parsing"
#define DS4_SERVER_TEXT_GENERATED_CAPABILITY_IS_NOT_ALLOWED_S "generated capability is not allowed: %s"
#define DS4_SERVER_TEXT_GENERATED_ARGUMENTS_DO_NOT_MATCH_SCHEMA_FOR_S "generated arguments do not match schema for %s"
#define DS4_SERVER_TEXT_CANCELLED_BEFORE_PUBLISHING_RESPONSE_STATE "cancelled before publishing response state"
#define DS4_SERVER_TEXT_S_CTX_S_S_S_CLIENT_DISCONNECTED "ds4-server: %s ctx=%s%s%s client disconnected"
#define DS4_SERVER_TEXT_CHAT_CTX_S_GEN_D_S_S_FINISH_S "ds4-server: chat ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs"
#define DS4_SERVER_TEXT_CHAT_CTX_S_GEN_D_S_S_FINISH_S_GENERATE_JOB_INNER "ds4-server: chat ctx=%s gen=%d%s%s finish=%s %.3fs"
#define DS4_SERVER_TEXT_RESPONSES_POST_RESPONSE_VISIBLE_CHECKPOINT_CANONICALIZATION_FAILED_CTX_S "ds4-server: Responses post-response visible checkpoint canonicalization failed ctx=%s"
#define DS4_SERVER_TEXT_CHAT_COMPLETIONS_POST_RESPONSE_THINKING_CHECKPOINT_CANONICALIZATION_FAILED_CTX "ds4-server: chat/completions post-response thinking checkpoint canonicalization failed ctx=%s"
#define DS4_SERVER_TEXT_QWEN_REQUEST_TIERS_CPU_PAIRS_LLU_GPU_HIT_PAIRS_LLU_PCIE_PAIRS_LLU \
    "ds4-server: Qwen request tiers CPU_pairs=%llu GPU_hit_pairs=%llu PCIe_pairs=%llu " \
    "SSD_bytes=%llu SSD_projection_reads=%llu expert_H2D_bytes=%llu KV_H2D_bytes=%llu KV_hits=%llu KV_misses=%llu CPU_wait_ms=%.3f " \
    "expert_RAM_bytes=%llu KV_RAM_bytes=%llu staging_bytes=%llu expert_VRAM_bytes=%llu " \
    "streamed_KV_VRAM_bytes=%llu KV_workspace_bytes=%llu hybrid_workspace_bytes=%llu hybrid_host_bytes=%llu CPU_H2D_bytes=%llu activation_D2H_bytes=%llu " \
    "CPU_stage_jobs=%llu staging_wait_ms=%.3f victim_select_ms=%.3f"
#define DS4_SERVER_TEXT_QWEN_REQUEST_TIERS_LOG \
    "ds4-server: Qwen request shared decode batches=%llu CPU_pairs=%llu GPU_hit_pairs=%llu PCIe_pairs=%llu " \
    "SSD_bytes=%llu SSD_projection_reads=%llu expert_H2D_bytes=%llu KV_H2D_bytes=%llu KV_hits=%llu KV_misses=%llu CPU_wait_ms=%.3f " \
    "CPU_H2D_bytes=%llu activation_D2H_bytes=%llu CPU_stage_jobs=%llu staging_wait_ms=%.3f victim_select_ms=%.3f " \
    "attribution=shared-do-not-sum-across-requests valid=%u"
#define DS4_SERVER_TEXT_QWEN_REQUEST_TIER_COUNTERS_INCOMPLETE_AFTER_ENGINE_COUNTER_RESET "ds4-server: Qwen request tier counters incomplete after engine counter reset"
#define DS4_SERVER_TEXT_SLOT_D_REQUEST_REUSES_NOTHING_EVICTING_CHECKPOINT_D_TOKENS \
    "slot %d: request reuses nothing; evicting checkpoint " \
    "(%d tokens, idle %lld s, %s)"
#define DS4_SERVER_TEXT_BAD_HTTP_REQUEST "bad HTTP request"
#define DS4_SERVER_TEXT_UNKNOWN_ENDPOINT "unknown endpoint"
#define DS4_SERVER_TEXT_SERVER_SHUTTING_DOWN "server shutting down"
#define DS4_SERVER_TEXT_INVALID_VALUE_FOR_S_S "ds4-server: invalid value for %s: %s"
#define DS4_SERVER_TEXT_MISSING_VALUE_FOR_S "ds4-server: missing value for %s"
#define DS4_SERVER_TEXT_CONTEXT_BUFFERS_F_MIB_CTX_D_BACKEND_S_PREFILL_CHUNK "ds4-server: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)"
#define DS4_SERVER_TEXT_D_RESIDENT_SESSIONS_REQUEST_AT_LEAST_F_GIB_OF \
    "ds4-server: %d resident sessions request at least %.2f GiB of " \
    "context buffers, or %.2f GiB when the model shares one prefill " \
    "workspace"
#define DS4_SERVER_TEXT_VALID_SERVER_BACKENDS_ARE_METAL_ROCM_CPU "ds4-server: valid server backends are: metal, rocm, cpu"
#define DS4_SERVER_TEXT_VALID_SERVER_BACKENDS_ARE_METAL_CUDA_CPU "ds4-server: valid server backends are: metal, cuda, cpu"
#define DS4_SERVER_TEXT_INVALID_DISTRIBUTED_OPTION "invalid distributed option"
#define DS4_SERVER_TEXT_INVALID_TENSOR_PARALLEL_OPTION "invalid tensor-parallel option"
#define DS4_SERVER_TEXT_SSD_STREAMING_CACHE_EXPERTS_MUST_BE_A_POSITIVE_COUNT "ds4-server: --ssd-streaming-cache-experts must be a positive count or <number>GB"
#define DS4_SERVER_TEXT_SSD_STREAMING_PRELOAD_EXPERTS_MUST_BE_POSITIVE "ds4-server: --ssd-streaming-preload-experts must be positive"
#define DS4_SERVER_TEXT_SIMULATE_USED_MEMORY_MUST_BE_A_POSITIVE_GIB_VALUE "ds4-server: --simulate-used-memory must be a positive GiB value, e.g. 64GB"
#define DS4_SERVER_TEXT_PREFILL_CHUNK_MUST_BE_POSITIVE "ds4-server: --prefill-chunk must be positive"
#define DS4_SERVER_TEXT_POWER_MUST_BE_BETWEEN_AND "ds4-server: --power must be between 1 and 100"
#define DS4_SERVER_TEXT_UNKNOWN_OPTION_S "ds4-server: unknown option: %s"
#define DS4_SERVER_TEXT_KV_CACHE_COLD_MAX_TOKENS_MUST_BE_OR_KV "ds4-server: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens"
#define DS4_SERVER_TEXT_ROLE_WORKER_IS_A_SERVING_MODE_START_TENSOR_PARALLEL "ds4-server: --role worker is a serving mode; start tensor-parallel workers with ./ds4"
#define DS4_SERVER_TEXT_INVALID_CONSTRAINT_MODE \
    "ds4-server: invalid DS4_CONSTRAINT_MODE '%s'; expected trie, " \
    "incremental, optimized, oracle_only or compare_new_vs_oracle\n"
#define DS4_SERVER_TEXT_FAILED_TO_CHDIR_TO_S_S "ds4-server: failed to chdir to %s: %s"
#define DS4_SERVER_TEXT_LOADED_MODEL_S "ds4-server: loaded model %s"
#define DS4_SERVER_TEXT_AUDITED_QWEN_Q4_K_S_SESSION_CACHE_BUDGET_MUST_BE_LLU "ds4-server: audited Qwen Q4_K_S session cache budget must be <= %llu MiB"
#define DS4_SERVER_TEXT_FAILED_TO_CREATE_S_SESSION_D_D "ds4-server: failed to create %s session %d/%d"
#define DS4_SERVER_TEXT_EXACT_DSML_TOOL_REPLAY_DISABLED_TOOL_HISTORY_USES_CANONICAL "ds4-server: exact DSML tool replay disabled; tool history uses canonical JSON rendering"
#define DS4_SERVER_TEXT_BATCHED_MODE_LOG "ds4-server: batched mode enabled resident_sessions=%d prefill_quantum=%d mixed_prefill_quantum=%d decode_coalesce_us=%ld"
#define DS4_SERVER_TEXT_MTP_SPECULATIVE_DECODING_IS_DISABLED_WHILE_NATIVE_SESSION_BATCHING "ds4-server: MTP speculative decoding is disabled while native session batching is active"
#define DS4_SERVER_TEXT_FAILED_TO_OPEN_TRACE_FILE_S_S "ds4-server: failed to open trace file %s: %s"
#define DS4_SERVER_TEXT_TRACING_SESSION_TO_S "ds4-server: tracing session to %s"
#define DS4_SERVER_TEXT_FAILED_TO_START_DECODE_COORDINATOR "ds4-server: failed to start decode coordinator"
#define DS4_SERVER_TEXT_FAILED_TO_START_SESSION_WORKER_D_D "ds4-server: failed to start session worker %d/%d"
#define DS4_SERVER_TEXT_FAILED_TO_START_WORKER "ds4-server: failed to start worker"
#define DS4_SERVER_TEXT_FAILED_TO_LISTEN_ON_S_D_S "ds4-server: failed to listen on %s:%d: %s"
#define DS4_SERVER_TEXT_LISTENING_ON_HTTP_S_D "ds4-server: listening on http://%s:%d"
#define DS4_SERVER_TEXT_LISTEN_POLL_FAILED_S "ds4-server: listen poll failed: %s"
#define DS4_SERVER_TEXT_ACCEPT_FAILED_S "ds4-server: accept failed: %s"
#define DS4_SERVER_TEXT_SHUTDOWN_REQUESTED_DRAINING_REQUESTS "ds4-server: shutdown requested, draining requests"
#define DS4_SERVER_TEXT_PERSISTING_RESIDENT_KV_CACHE_BEFORE_SHUTDOWN_SLOT_D_TOKENS "ds4-server: persisting resident KV cache before shutdown slot=%d tokens=%d"

/* Diagnostic formatting for the shared server prefix. */
#define DS4_SERVER_TEXT_MESSAGE "ds4-server: %s"
#define DS4_SERVER_TEXT_MESSAGE_LINE "ds4-server: %s\n"
#define DS4_SERVER_DIAGNOSTIC_DS4_CONSTRAINT_MODE "DS4_CONSTRAINT_MODE"
#define DS4_SERVER_TEXT_INVALID_OPTION_VALUE "ds4-server: invalid %s value: %s"
#define DS4_SERVER_DIAGNOSTIC_COMPLETION "completion"
#define DS4_SERVER_DIAGNOSTIC_CHAT "chat"
#define DS4_SERVER_DIAGNOSTIC_DS4_SERVER_S_CTX_S_GEN_D_S_S_FINISH_S_FS "ds4-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs"
#define DS4_SERVER_DIAGNOSTIC_DS4_SERVER_S_CTX_S_GEN_D_S_S_FINISH_S_ERROR_S_FS "ds4-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs"
#define DS4_SERVER_DIAGNOSTIC_GLM "GLM"
#define DS4_SERVER_DIAGNOSTIC_RESPPROTO " RESPPROTO"
#define DS4_SERVER_TEXT_PREFILL_PROGRESS_LOG "ds4-server: %s ctx=%s%s%s %s chunk %d/%d (%.1f%%) chunk=%.2f t/s avg=%.2f t/s %.3fs"
#define DS4_SERVER_DIAGNOSTIC_JSON_SCHEMA "json_schema"

#define DS4_SERVER_DIAGNOSTIC_PROTECTED "protected"
#define DS4_SERVER_DIAGNOSTIC_STALE "stale"
#define DS4_SERVER_DIAGNOSTIC_NONE "(none)"
#define DS4_SERVER_DIAGNOSTIC_NULL "(null)"
#define DS4_SERVER_DIAGNOSTIC_DEEPSEEK "DeepSeek"
#define DS4_SERVER_DIAGNOSTIC_FALSE "false"
#define DS4_SERVER_DIAGNOSTIC_TRUE "true"
#define DS4_SERVER_DIAGNOSTIC_UNKNOWN "unknown"
#define DS4_SERVER_DIAGNOSTIC_MISMATCH "mismatch"
#define DS4_SERVER_DIAGNOSTIC_MATCH "match"
#define DS4_SERVER_DIAGNOSTIC_ERROR "error"
#define DS4_SERVER_DIAGNOSTIC_OK "ok"
#define DS4_SERVER_DIAGNOSTIC_FILE " file="
#define DS4_SERVER_DIAGNOSTIC_DSML "dsml"

#endif
