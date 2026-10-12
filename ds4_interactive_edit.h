#ifndef DS4_INTERACTIVE_EDIT_H
#define DS4_INTERACTIVE_EDIT_H

#include "ds4_edit_constraint.h"
#include "ds4_edit_document.h"

static const char interactive_edit_description[] =
    "Edit workspace text files with one string containing consecutive edit fragments. "
    "Generate <edit path=\"relative/path\" then wait: the server authorizes each path. "
    "Then continue with ><old_text>selection</old_text><new_text>replacement</new_text></edit>. "
    "Only path is allowed on edit. The final </new_text> immediately before </edit> "
    "closes the body; earlier bare </new_text> tags are literal replacement content. "
    "For a literal </new_text></edit> sequence use CDATA or XML entities. "
    "Use XML entities or CDATA for literal markup and preserve all whitespace. "
    "Literal old_text starts at a physical line boundary and must occur exactly once. "
    "Use <all/> as old_text for a full rewrite, <line number=\"N\"/> to insert before line N "
    "(L+1 is EOF), and <delete/> as new_text only with all on an existing file. "
    "For a full rewrite or creation emit exactly <old_text><all/></old_text>, without old_text attributes. "
    "For deletion emit exactly <old_text><all/></old_text><new_text><delete/></new_text></edit>. "
    "Do not replace deletion with empty text or a newline. "
    "Create a missing file with all or line 1. Empty new_text removes selected content, not the file. "
    "For extremely long selections use empty <old_text start_line=\"N\" end_line=\"M\"></old_text>; "
    "the client validates the inclusive range. Repeated paths refer to the initial snapshot. "
    "Never use start_line/end_line for whole-file creation, rewrite or deletion. "
    "The client rejects overlapping edits and insertions at the same position. "
    "Existing files require valid inspect/read evidence. No file is changed before completion.";

typedef struct { int document; } interactive_edit_fields;

static void interactive_edit_free_config(void *config) { free(config); }

static bool interactive_edit_config(interactive_tool *tool,
        const schema_json_value *spec, const tool_schema_order *order) {
    const schema_json_value *fields = schema_json_object_get(spec, "fields");
    const char *name = interactive_string(fields, "document");
    int prop = interactive_property(order, name);
    const schema_json_value *schema = tool_schema_order_property_schema(order, prop);
    const char *type = interactive_string(schema, "type");
    if (!INTERACTIVE_KEYS(fields, "document") || fields->len != 1 ||
        !interactive_required(order, name) || !type || strcmp(type, "string") ||
        order->len != 1) return false;
    /* Finite document alternatives cannot generally adapt to an authorized
     * snapshot. Reject unsupported schema forms before decoding. */
    if (schema_json_object_get(schema, "enum") || schema_json_object_get(schema, "const")) return false;
    if (!INTERACTIVE_KEYS(schema, "type", "title", "description", "default", "examples",
                          "$id", "$schema", "minLength", "maxLength", "pattern")) return false;
    interactive_edit_fields *config = xmalloc(sizeof(*config));
    config->document = prop;
    interactive_tool_bind_config(tool, config);
    interactive_tool_define_field(tool, prop, INTERACTIVE_FIELD_CONSTRAINED);
    return true;
}

static void interactive_edit_free_state(void *opaque) {
    edit_document_free(opaque);
}

static bool interactive_edit_begin_input(void **opaque, const char *values, const char *end) {
    schema_json_value *input = schema_json_parse(values);
    const char *path = interactive_string(input, "path");
    if (!path || !path[0]) { schema_json_free(input); return false; }
    edit_document *state = *opaque;
    if (!state) {
        state = edit_document_new(end);
        *opaque = state;
    }
    bool valid = edit_document_begin_input(state, path);
    schema_json_free(input);
    return valid;
}

static size_t interactive_edit_reply_bytes(const void *opaque, const schema_json_value *reply) {
    if (edit_document_reuses_snapshot(opaque)) return 0;
    const char *text = interactive_string(reply, "text");
    return text ? strlen(text) : 0;
}

static size_t interactive_edit_reply_memory(const void *opaque, const schema_json_value *reply) {
    if (edit_document_reuses_snapshot(opaque)) return 0;
    size_t n = interactive_edit_reply_bytes(opaque, reply);
    return n > INTERACTIVE_MAX_SNAPSHOT ? SIZE_MAX : edit_document_snapshot_memory(n);
}

static bool interactive_edit_reply(void **opaque, const schema_json_value *reply, const char *end) {
    (void)end;
    const char *status = interactive_string(reply, "status");
    bool create = status && !strcmp(status, "create"), provide = status && !strcmp(status, "provide");
    const char *text = provide ? interactive_string(reply, "text") : "";
    const char *encoding = interactive_string(reply, "encoding");
    const schema_json_value *lines = schema_json_object_get(reply, "line_count");
    if ((!create && !provide) || !INTERACTIVE_KEYS(reply, "status", "text", "encoding", "line_count") ||
        (create && reply->len != 1) || (provide && (!text || strlen(text) > INTERACTIVE_MAX_SNAPSHOT ||
        !encoding || !encoding[0] || !lines || lines->kind != SCHEMA_JSON_NUMBER ||
        !interactive_utf8((const unsigned char *)text)))) return false;
    if (provide && lines->number != (double)edit_constraint_count_lines(text)) return false;
    return edit_document_provide(*opaque, text, create);
}

/* Translate a decoded restart back to the native transport representation. */
static size_t interactive_edit_native_offset(const char *raw, size_t decoded,
                                              const char *end) {
    const char *start = raw;
    while (decoded--) {
        if (ds4_tool_text_escaped_close(raw, end))
            raw += !strncmp(raw, "&amp;", 5) ? 5 : 4;
        else raw++;
    }
    return (size_t)(raw - start);
}

static size_t interactive_edit_characters(const void *opaque, const char *raw,
                                          size_t n, const char *end) {
    edit_document_checkpoint restart = edit_document_restart(opaque, n, n);
    char *tail = xstrndup(raw + restart.offset, n - restart.offset);
    ds4_tool_text_unescape(tail, end);
    size_t count = restart.characters;
    for (const unsigned char *p = (const unsigned char *)tail; *p; p++)
        if ((*p & 0xc0) != 0x80) count++;
    free(tail);
    return count;
}

static edit_document_view interactive_edit_probe(const void *opaque, const char *raw,
        size_t n, const char *end, size_t committed, bool complete, bool incremental, buf *selected) {
    edit_document_checkpoint restart = incremental ? edit_document_restart(opaque, n, committed) :
        (edit_document_checkpoint){0};
    size_t base = restart.offset;
    raw += base;
    n -= base;
    committed -= base;
    size_t safe = n;
    for (size_t i = n; i > 0; i--) {
        if (raw[i-1] == '&' && ds4_tool_text_partial_escape(raw+i-1, n-i+1, end)) {
            safe = i-1; break;
        }
    }
    char *encoded = xstrndup(raw, n);
    char *value = xstrdup(encoded);
    char *prefix = xstrndup(raw, committed < n ? committed : n);
    /* Native delimiter escapes are one transport layer, not XML entities. */
    ds4_tool_text_unescape(value, end);
    ds4_tool_text_unescape(prefix, end);
    edit_document_view view = edit_document_scan_from(opaque, value, strlen(value), strlen(prefix), complete, restart, selected);
    if (safe != n) {
        buf escape_selected = {0};
        buf *escape_prefix = selected ? selected : &escape_selected;
        if (selected) selected->len = 0;
        free(value); value = xstrndup(raw, safe); ds4_tool_text_unescape(value, end);
        edit_document_view before_escape = edit_document_scan_from(opaque, value,
            strlen(value), strlen(prefix), false, restart, escape_prefix);
        if (!view.valid && !complete) {
            free(view.path);
            view = before_escape;
            view.valid = view.valid && view.native_escape;
            if (view.valid && view.checkpoint.old_body)
                view.valid = edit_document_native_escape_possible(opaque, view.checkpoint, restart,
                    escape_prefix, raw + safe, n - safe, end);
        } else {
            view.checkpoint = before_escape.checkpoint;
            free(before_escape.path);
        }
        buf_free(&escape_selected);
    }
    if (safe != n) {
        view.speculative = false;
        view.payload_start = false;
    }
    size_t characters = restart.characters;
    for (size_t i = 0; i < view.checkpoint.offset; i++)
        if (((unsigned char)value[i] & 0xc0) != 0x80) characters++;
    view.checkpoint.characters = characters;
    view.checkpoint.offset = base + interactive_edit_native_offset(encoded, view.checkpoint.offset, end);
    free(encoded); free(prefix); free(value);
    return view;
}

static void interactive_edit_draft_suffix(const interactive_tool *tool, const void *state,
        int prop, const char *raw, size_t n, const char *end, buf *out, size_t limit) {
    const interactive_edit_fields *fields = interactive_tool_config(tool);
    if (prop != fields->document || !state) return;
    buf selected = {0}, xml = {0};
    edit_document_checkpoint restart = edit_document_restart(state, n, n);
    edit_document_view view = interactive_edit_probe(state, raw, n, end, n, false, true, &selected);
    if (view.valid && view.phase == EDIT_DOCUMENT_OLD && view.checkpoint.offset == n)
        edit_document_draft_suffix(state, view.checkpoint, restart, &selected, &xml, limit);
    /* XML entities and native tool escapes are distinct layers. Escaping the
     * complete XML proposal also protects entity spellings of native closes. */
    append_tool_body_text(out, xml.ptr, end);
    free(view.path); buf_free(&selected); buf_free(&xml);
}

static edit_document_view interactive_edit_view(const void *opaque, const char *raw,
        size_t n, const char *end, size_t committed, bool complete) {
    return interactive_edit_probe(opaque, raw, n, end, committed, complete, true, NULL);
}

static bool interactive_edit_plain_replacement(const void *state, const char *raw,
                                               size_t n, const char *end) {
    if (edit_document_plain_replacement(state, n)) return true;
    /* MTP's accepted shadow prefix may extend the real checkpoint. Validate
     * its tail without publishing a checkpoint or changing authorizations. */
    edit_document_view view = interactive_edit_view(state, raw, n, end, n, false);
    bool plain = view.valid && view.checkpoint.new_body && view.checkpoint.offset == n;
    free(view.path);
    return plain;
}

static void interactive_edit_commit_string(void *opaque, const char *raw,
                                            size_t n, const char *end) {
    if (!opaque) return;
    buf selected = {0};
    edit_document_view view = interactive_edit_probe(opaque, raw, n, end, n, false, true, &selected);
    if (view.valid) edit_document_commit_restart(opaque, view.checkpoint, &selected);
    buf_free(&selected);
    free(view.path);
}

static bool interactive_edit_string_prefix(const interactive_tool *tool, const void *state,
        int prop, const char *value, size_t n,
        bool complete, size_t committed, const char *end) {
    const interactive_edit_fields *fields = interactive_tool_config(tool);
    if (prop != fields->document) return false;
    edit_document_view view = interactive_edit_view(state, value, n, end, committed, complete);
    bool valid = view.valid;
    free(view.path);
    return valid;
}

static bool interactive_edit_string_prefix_oracle(const interactive_tool *tool, const void *state,
        int prop, const char *value, size_t n, bool complete, size_t committed, const char *end) {
    const interactive_edit_fields *fields = interactive_tool_config(tool);
    if (prop != fields->document) return false;
    edit_document_view view = interactive_edit_probe(state, value, n, end, committed, complete, false, NULL);
    bool valid = view.valid;
    free(view.path);
    return valid;
}

static bool interactive_edit_await(const interactive_tool *tool, const void *state, int prop,
        const char *value, size_t n, buf *values, size_t *index, const char *end) {
    const interactive_edit_fields *fields = interactive_tool_config(tool);
    if (prop != fields->document) return false;
    edit_document_view view = interactive_edit_view(state, value, n, end, n, false);
    bool waiting = view.valid && view.phase == EDIT_DOCUMENT_AWAIT && view.path;
    if (waiting) {
        buf_puts(values, "{\"path\":"); json_escape(values, view.path); buf_putc(values, '}');
        *index = view.index + 1;
    }
    free(view.path);
    return waiting;
}

static interactive_sampling_policy interactive_edit_sampling(const interactive_tool *tool, const void *state,
        int prop, const char *value, size_t n) {
    const interactive_edit_fields *fields = interactive_tool_config(tool);
    interactive_sampling_policy policy = {.greedy = true};
    if (prop != fields->document || !state) return policy;
    edit_document_view view = interactive_edit_view(state, value, n,
        edit_document_native_end(state), n, false);
    if (view.valid) {
        /* The epoch is opaque to the scheduler. It distinguishes edits and
         * parser phases so verifier carry cannot cross a policy boundary. */
        policy.epoch = ((view.index + 1) << 8) | (unsigned)view.phase;
        policy.body = view.speculative &&
            (view.phase == EDIT_DOCUMENT_NEW || view.phase == EDIT_DOCUMENT_OLD);
        policy.greedy = view.phase != EDIT_DOCUMENT_NEW;
        policy.payload_start = view.phase == EDIT_DOCUMENT_NEW && view.payload_start;
    }
    free(view.path);
    return policy;
}

static bool interactive_edit_complete(const interactive_tool *tool, const void *state,
                                       const schema_json_value *args) {
    (void)tool;
    if (!args || args->kind != SCHEMA_JSON_OBJECT || args->len != 1 ||
        args->member[0].value->kind != SCHEMA_JSON_STRING) return false;
    const char *value = args->member[0].value->string;
    edit_document_view view = edit_document_scan(state, value, strlen(value), strlen(value), true);
    bool valid = view.valid; free(view.path);
    return valid;
}

#endif
