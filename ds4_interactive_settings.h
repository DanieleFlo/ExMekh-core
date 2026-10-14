#ifndef DS4_INTERACTIVE_SETTINGS_H
#define DS4_INTERACTIVE_SETTINGS_H

/* Protocol identity, resource policy and presentation text have one owner.
 * Marker spellings are a fixed wire contract shared with the client. */
#define INTERACTIVE_MAX_TOOLS 16
#define INTERACTIVE_MAX_FIELDS 64
#define INTERACTIVE_MAX_SNAPSHOT (1024 * 1024)
#define INTERACTIVE_MAX_TEXT (1024 * 1024)
#define INTERACTIVE_MAX_VALUES (16 * 1024)
#define INTERACTIVE_MAX_RECORDS 64
#define INTERACTIVE_MAX_ACTIVE 8
#define INTERACTIVE_SNAPSHOT_BUDGET (8 * 1024 * 1024)
#define INTERACTIVE_STATE_BUDGET (128 * 1024 * 1024)
#define INTERACTIVE_CHECKPOINT_BUDGET (UINT64_C(8) * 1024 * 1024 * 1024)
#define INTERACTIVE_TTL 300
#define INTERACTIVE_RECEIPT_TTL 600

#define INTERACTIVE_ERROR_BYTES 160
#define INTERACTIVE_PROTOCOL_VERSION 2
#define INTERACTIVE_STRINGIFY_VALUE(value) #value
#define INTERACTIVE_STRINGIFY(value) INTERACTIVE_STRINGIFY_VALUE(value)
#define INTERACTIVE_VERSION_TEXT INTERACTIVE_STRINGIFY(INTERACTIVE_PROTOCOL_VERSION)
#define EDIT_BEHAVIOR_NAME "edit_file.v3"
#define EDIT_SNAPSHOT_KIND "file_snapshot"
#define EDIT_ID_DIGITS 3
#define EDIT_ID_SPACE 1000
#define EDIT_ID_FORMAT "%03u"
#define EDIT_ID_ATTRIBUTE_FORMAT "id=\"%s\""
#define EDIT_OPEN_PREFIX "<exmekh_edit path=\""
#define EDIT_OPEN_SUFFIX_FORMAT " id=\"%s\" />"
#define EDIT_SEPARATOR "<exmekh_new_text/>"
#define EDIT_SEPARATOR_FORMAT "<exmekh_new_text id=\"%s\"/>"
#define EDIT_DIAGNOSTIC_PROBE "<delete/>"

static const char interactive_configuration_error[] = "invalid or unsupported interactive v2 configuration (document must be one required string without enum/const)";
static const char interactive_expired_message[] = "interactive continuation expired";
static const char interactive_active_context_error[] = "agent context has an active interactive turn";
static const char interactive_request_error[] = "invalid interactive request";
static const char interactive_reused_request_error[] = "request_id reused with different payload";
static const char interactive_capacity_error[] = "interactive capacity reached";
static const char interactive_cancelled_before_pause_error[] = "interactive turn cancelled before suspension";
static const char interactive_aborted_message[] = "interactive generation ended before a result";
static const char interactive_unknown_scope_error[] = "unknown interactive continuation or mismatched scope";
static const char interactive_finalizing_error[] = "interactive operation is finalizing";
static const char interactive_cancelled_message[] = "interactive turn cancelled";
static const char interactive_exchange_unavailable_error[] = "interactive exchange is unavailable or already consumed";
static const char interactive_stream_mode_error[] = "resume must preserve the original streaming mode";
static const char interactive_exchange_limit_error[] = "interactive exchange limit exceeded";
static const char interactive_snapshot_budget_error[] = "interactive snapshot memory budget exceeded";
static const char interactive_control_error[] = "invalid interactive control payload";

static const char interactive_payload_frontier_log[] =
    "ds4-server: interactive payload frontier call_id=%s input_index=%zu temperature=%.3f "
    "unmasked_best=%d piece=%s raw_argmax_allowed=%s operation_probe_allowed=%s";

static const char interactive_rollback_pause_error[] = "interactive pause checkpoint is unavailable for rollback";

/* Product text for this domain; preserve formatting placeholders. */
#define DS4_INTERACTIVE_TEXT_INTERACTIVE_V2_DOES_NOT_ACCEPT_STOP_OR_IGNORE_EOS "interactive v2 does not accept stop or ignore_eos"
#define DS4_INTERACTIVE_TEXT_INTERACTIVE_CHECKPOINT_IS_UNAVAILABLE "interactive checkpoint is unavailable"
#define DS4_INTERACTIVE_TEXT_INTERACTIVE_GENERATED_TEXT_EXCEEDS_ITS_BYTE_LIMIT_OR_CONTAINS "interactive generated text exceeds its byte limit or contains NUL"
#define DS4_INTERACTIVE_TEXT_INTERACTIVE_GENERATION_ENDED_BEFORE_COMPLETION "interactive generation ended before completion"
#define DS4_INTERACTIVE_TEXT_INTERACTIVE_NATIVE_ARGUMENT_EXTRACTION_FAILED "interactive native argument extraction failed"
#define DS4_INTERACTIVE_TEXT_INTERACTIVE_ARGUMENT_EXTRACTION_FAILED "interactive argument extraction failed"
#define DS4_INTERACTIVE_TEXT_INTERACTIVE_OPERATION_WAS_CANCELLED_BEFORE_DISPATCH "interactive operation was cancelled before dispatch"

#endif
