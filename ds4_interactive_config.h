#ifndef DS4_INTERACTIVE_CONFIG_H
#define DS4_INTERACTIVE_CONFIG_H

static const char *interactive_string(const schema_json_value *obj, const char *key) {
    const schema_json_value *v = schema_json_object_get(obj, key);
    return v && v->kind == SCHEMA_JSON_STRING ? v->string : NULL;
}

static bool interactive_keys(const schema_json_value *v, const char *const *keys,
                              size_t count) {
    if (!v || v->kind != SCHEMA_JSON_OBJECT) return false;
    for (size_t i = 0; i < v->len; i++) {
        bool known = false;
        for (size_t j = 0; j < count; j++)
            if (!strcmp(v->member[i].key, keys[j])) known = true;
        if (!known) return false;
        for (size_t j = 0; j < i; j++)
            if (!strcmp(v->member[i].key, v->member[j].key)) return false;
    }
    return true;
}

#define INTERACTIVE_KEYS(v, ...) interactive_keys(v, \
    (const char *const[]){__VA_ARGS__}, \
    sizeof((const char *const[]){__VA_ARGS__}) / sizeof(const char *))

static bool interactive_version(const schema_json_value *v) {
    const schema_json_value *version = schema_json_object_get(v, "version");
    return version && version->kind == SCHEMA_JSON_NUMBER && version->number == 1;
}

static bool interactive_request_options(const char *body) {
    schema_json_value *root = schema_json_parse(body);
    bool valid = root && !schema_json_object_get(root, "stop") &&
        !schema_json_object_get(root, "ignore_eos");
    schema_json_free(root);
    return valid;
}

static int interactive_property(const tool_schema_order *order, const char *name) {
    if (!name) return -1;
    for (int i = 0; i < order->len; i++)
        if (!strcmp(order->prop[i], name)) return i;
    return -1;
}

static bool interactive_required(const tool_schema_order *order, const char *name) {
    const schema_json_value *required = schema_json_object_get(order->schema, "required");
    if (!required || required->kind != SCHEMA_JSON_ARRAY) return false;
    for (size_t i = 0; i < required->len; i++)
        if (required->item[i]->kind == SCHEMA_JSON_STRING &&
            !strcmp(required->item[i]->string, name)) return true;
    return false;
}

/* Behaviors compile policy through this API, without depending on its storage. */
static void interactive_tool_bind_config(interactive_tool *tool, void *config) {
    tool->behavior_config = config;
}

static const void *interactive_tool_config(const interactive_tool *tool) {
    return tool->behavior_config;
}

static void interactive_tool_require_priority(interactive_tool *tool, int prop) {
    tool->first_required_mask |= UINT64_C(1) << prop;
}

static void interactive_tool_define_field(interactive_tool *tool, int prop,
                                          unsigned flags, uint64_t requires) {
    tool->field[prop] = (interactive_field){.flags = flags, .requires = requires};
}

static int interactive_next_priority(const interactive_tool *tool, uint64_t seen) {
    for (int i = 0; i < tool->first_count; i++) {
        if (!(seen & (UINT64_C(1) << tool->first[i]))) return tool->first[i];
    }
    return -1;
}

static bool interactive_priority_complete(const interactive_tool *tool, uint64_t seen) {
    return (seen & tool->first_mask) == tool->first_mask;
}

#include "ds4_interactive_edit.h"

static const interactive_behavior interactive_behaviors[] = {
    {
        .name = "edit_file.v1",
        .request_kind = "file_snapshot",
        .configure = interactive_edit_config,
        .string_prefix = interactive_edit_string_prefix,
        .sampling_body = interactive_edit_sampling_body,
        .reply_bytes = interactive_edit_reply_bytes,
        .prepare_reply = interactive_edit_reply,
        .free_state = interactive_edit_free_state,
        .free_config = interactive_edit_free_config,
    },
};

static bool interactive_field_has(const interactive_tool *tool, int prop,
                                  interactive_field_flag flag) {
    return tool && prop >= 0 && prop < INTERACTIVE_MAX_FIELDS &&
        (tool->field[prop].flags & flag) != 0;
}

static bool interactive_field_dependencies_met(const interactive_tool *tool, int prop,
                                               uint64_t seen) {
    if (!tool || prop < 0 || prop >= INTERACTIVE_MAX_FIELDS) return false;
    uint64_t required = tool->field[prop].requires;
    return (seen & required) == required;
}

static bool interactive_field_needs_previous(const interactive_tool *tool, int prop) {
    return tool && prop >= 0 && prop < INTERACTIVE_MAX_FIELDS && tool->field[prop].requires != 0;
}

static bool interactive_call_prepare_reply(interactive_call *call,
                                           const schema_json_value *reply) {
    return call->tool->behavior->prepare_reply(&call->behavior_state, reply, call->param_end);
}

static const interactive_behavior *interactive_behavior_find(const char *name) {
    for (size_t i = 0; name && i < sizeof(interactive_behaviors)/sizeof(interactive_behaviors[0]); i++)
        if (!strcmp(name, interactive_behaviors[i].name)) return &interactive_behaviors[i];
    return NULL;
}

static const server_model_syntax interactive_syntaxes[] = {
    SERVER_MODEL_SYNTAX_QWEN, SERVER_MODEL_SYNTAX_QWEN4,
    SERVER_MODEL_SYNTAX_DEEPSEEK, SERVER_MODEL_SYNTAX_DEEPSEEK41,
};

static bool interactive_syntax_supported(server_model_syntax syntax) {
    for (size_t i = 0; i < sizeof(interactive_syntaxes)/sizeof(interactive_syntaxes[0]); i++)
        if (syntax == interactive_syntaxes[i]) return true;
    return false;
}

static bool interactive_compile(request *r, char *err, size_t errlen) {
    interactive_request *ir = &r->interactive;
    if (!ir->wire) return true;
    const schema_json_value *v = ir->wire;
    const char *op = interactive_string(v, "operation");
    const char *id = interactive_string(v, "request_id");
    if (!INTERACTIVE_KEYS(v, "version", "operation", "request_id", "tools") ||
        !interactive_version(v) || !op || strcmp(op, "configure") ||
        !id || !id[0] || strlen(id) > 128 || r->api != API_RESPONSES ||
        !interactive_syntax_supported(r->model_syntax) ||
        r->stops.len || r->ignore_eos) goto invalid;
    const schema_json_value *tools = schema_json_object_get(v, "tools");
    if (!tools || tools->kind != SCHEMA_JSON_ARRAY || !tools->len ||
        tools->len > INTERACTIVE_MAX_TOOLS) goto invalid;
    for (size_t i = 0; i < tools->len; i++) {
        const schema_json_value *spec = tools->item[i];
        const char *name = interactive_string(spec, "name");
        const char *behavior = interactive_string(spec, "behavior");
        if (!INTERACTIVE_KEYS(spec, "name", "behavior", "fields", "first_fields", "markers") ||
            !name || !interactive_behavior_find(behavior) ||
            !id_list_contains(&r->allowed_tools, name)) goto invalid;
        for (int j = 0; j < ir->count; j++)
            if (!strcmp(ir->tools[j].name, name)) goto invalid;
        const tool_schema_order *order = tool_schema_orders_find(&r->tool_orders, name);
        if (!order || !order->schema || order->len > INTERACTIVE_MAX_FIELDS) goto invalid;
        const schema_json_value *first = schema_json_object_get(spec, "first_fields");
        if (!first || first->kind != SCHEMA_JSON_ARRAY || !first->len ||
            first->len > INTERACTIVE_MAX_FIELDS) goto invalid;
        interactive_tool *t = &ir->tools[ir->count++];
        t->name = xstrdup(name);
        t->behavior = interactive_behavior_find(behavior);
        if (!t->behavior->configure(t, spec, order)) goto invalid;
        for (size_t j = 0; j < first->len; j++) {
            if (first->item[j]->kind != SCHEMA_JSON_STRING) goto invalid;
            const char *field = first->item[j]->string;
            int prop = interactive_property(order, field);
            if (prop < 0 || !interactive_required(order, field) ||
                interactive_field_has(t, prop, INTERACTIVE_FIELD_AFTER_INPUT) ||
                (t->first_mask & (UINT64_C(1) << prop))) goto invalid;
            t->first[t->first_count++] = prop;
            t->first_mask |= UINT64_C(1) << prop;
        }
        if ((t->first_mask & t->first_required_mask) != t->first_required_mask) goto invalid;
    }
    return true;
invalid:
    snprintf(err, errlen, "invalid or unsupported interactive v1 configuration");
    return false;
}

static const interactive_tool *interactive_tool_for(const request *r,
                                               const tool_schema_order *order) {
    if (!order) return NULL;
    for (int i = 0; i < r->interactive.count; i++)
        if (!strcmp(r->interactive.tools[i].name, order->name))
            return &r->interactive.tools[i];
    return NULL;
}

static interactive_call *interactive_call_for(const request *r, size_t body_start) {
    for (interactive_call *c = r->interactive.calls; c; c = c->next)
        if (c->body_start == body_start) return c;
    return NULL;
}

#endif
