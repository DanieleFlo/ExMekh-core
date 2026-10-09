#ifndef DS4_INTERACTIVE_EDIT_H
#define DS4_INTERACTIVE_EDIT_H

#include "ds4_edit_constraint.h"

/* All edit roles, markers and snapshot interpretation belong to this behavior.
 * The protocol stores opaque state and applies the compiled field policies. */
typedef struct {
    int selection;
    int replacement;
    char *selection_name;
} interactive_edit_fields;

static void interactive_edit_free_config(void *state) {
    interactive_edit_fields *fields = state;
    if (!fields) return;
    free(fields->selection_name);
    free(fields);
}

static bool interactive_edit_config(interactive_tool *tool,
                                     const schema_json_value *spec,
                                     const tool_schema_order *order) {
    const schema_json_value *fields = schema_json_object_get(spec, "fields");
    if (!INTERACTIVE_KEYS(fields, "resource", "selection", "replacement") || fields->len != 3)
        return false;

    int role[3];
    const char *roles[] = {"resource", "selection", "replacement"};
    for (int i = 0; i < 3; i++) {
        const char *name = interactive_string(fields, roles[i]);
        role[i] = interactive_property(order, name);
        const schema_json_value *schema = tool_schema_order_property_schema(order, role[i]);
        const char *type = interactive_string(schema, "type");
        if (role[i] < 0 || !interactive_required(order, name) || !type || strcmp(type, "string"))
            return false;
    }
    if (role[0] == role[1] || role[0] == role[2] || role[1] == role[2]) return false;

    const schema_json_value *markers = schema_json_object_get(spec, "markers");
    const char *line = interactive_string(markers, "line");
    const char *all = interactive_string(markers, "all");
    const char *del = interactive_string(markers, "delete");
    if (!INTERACTIVE_KEYS(markers, "line", "all", "delete") || !line || !all || !del ||
        strcmp(line, "@ex_meckh_line[<N>]") || strcmp(all, "@ex_meckh_all") ||
        strcmp(del, "@ex_meckh_delete")) return false;

    interactive_edit_fields *config = xmalloc(sizeof(*config));
    config->selection = role[1];
    config->replacement = role[2];
    config->selection_name = xstrdup(interactive_string(fields, "selection"));
    interactive_tool_bind_config(tool, config);
    interactive_tool_require_priority(tool, role[0]);
    interactive_tool_define_field(tool, role[1], INTERACTIVE_FIELD_AFTER_INPUT |
        INTERACTIVE_FIELD_CONSTRAINED | INTERACTIVE_FIELD_GREEDY | INTERACTIVE_FIELD_CLOSE_BOUNDARY, 0);
    interactive_tool_define_field(tool, role[2], INTERACTIVE_FIELD_AFTER_INPUT |
        INTERACTIVE_FIELD_CONSTRAINED | INTERACTIVE_FIELD_OPEN_BOUNDARY, UINT64_C(1) << role[1]);
    return true;
}

static size_t interactive_edit_reply_bytes(const schema_json_value *reply) {
    const char *text = interactive_string(reply, "text");
    return text ? strlen(text) : 0;
}

static void interactive_edit_free_state(void *state) {
    if (!state) return;
    edit_constraint_free(state);
    free(state);
}

static bool interactive_edit_reply(void **state, const schema_json_value *reply,
                                   const char *end) {
    const char *status = interactive_string(reply, "status");
    if (!status) return false;
    bool create = !strcmp(status, "create");
    bool provide = !strcmp(status, "provide");
    const char *text = provide ? interactive_string(reply, "text") : "";
    const char *encoding = interactive_string(reply, "encoding");
    const schema_json_value *lines = schema_json_object_get(reply, "line_count");
    if ((!create && !provide) || !INTERACTIVE_KEYS(reply, "status", "text", "encoding", "line_count") ||
        (create && reply->len != 1) ||
        (provide && (!text || strlen(text) > INTERACTIVE_MAX_SNAPSHOT || !encoding || !encoding[0] ||
         !lines || lines->kind != SCHEMA_JSON_NUMBER || !interactive_utf8((const unsigned char *)text))))
        return false;

    edit_constraint *snapshot = xmalloc(sizeof(*snapshot));
    edit_constraint_init(snapshot, text, create, end);
    if (provide && lines->number != (double)edit_constraint_line_count(snapshot)) {
        interactive_edit_free_state(snapshot);
        return false;
    }
    interactive_edit_free_state(*state);
    *state = snapshot;
    return true;
}

static bool interactive_edit_string_prefix(const interactive_tool *tool, const void *state,
                                            int prop, const schema_json_value *previous,
                                            const char *value, size_t n, bool complete) {
    const interactive_edit_fields *fields = interactive_tool_config(tool);
    if (prop == fields->selection)
        return edit_selection_prefix(state, value, n, complete, NULL);
    const char *selection = interactive_string(previous, fields->selection_name);
    bool all = selection && !strcmp(selection, "@ex_meckh_all");
    return edit_replacement_prefix(state, all, value, n, complete);
}

static bool interactive_edit_sampling_body(const interactive_tool *tool, const void *state,
                                            int prop, const char *value, size_t n) {
    const interactive_edit_fields *fields = interactive_tool_config(tool);
    if (prop == fields->replacement) return true;
    if (prop != fields->selection) return false;
    size_t candidates = 0;
    bool closable = edit_selection_prefix(state, value, n, true, &candidates);
    /* One starting line can still match internally elsewhere. Target-only
     * through global ambiguity preserves the verified greedy extension. */
    return candidates == 1 && closable;
}

#endif
