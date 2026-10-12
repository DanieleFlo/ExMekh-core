#ifndef DS4_INTERACTIVE_CONFIG_H
#define DS4_INTERACTIVE_CONFIG_H

/* Shared string validation belongs to the grammar/configuration boundary,
 * independent of continuation storage and HTTP lifecycle. */
static bool interactive_utf8_valid(const unsigned char *text, size_t n, bool complete) {
    size_t pos = 0;
    while (pos < n) {
        unsigned first = text[pos++], need;
        if (first < 0x80) continue;
        if (first >= 0xc2 && first <= 0xdf) need = 1;
        else if (first >= 0xe0 && first <= 0xef) need = 2;
        else if (first >= 0xf0 && first <= 0xf4) need = 3;
        else return false;
        for (unsigned i = 0; i < need; i++) {
            if (pos == n) return !complete;
            unsigned next = text[pos++];
            if ((next & 0xc0) != 0x80 || (!i && ((first == 0xe0 && next < 0xa0) ||
                (first == 0xed && next >= 0xa0) || (first == 0xf0 && next < 0x90) ||
                (first == 0xf4 && next >= 0x90)))) return false;
        }
    }
    return true;
}

static bool interactive_utf8(const unsigned char *text) {
    return interactive_utf8_valid(text, strlen((const char *)text), true);
}

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
    return version && version->kind == SCHEMA_JSON_NUMBER && version->number == 2;
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

static void interactive_tool_define_field(interactive_tool *tool, int prop,
                                          unsigned flags) {
    tool->field[prop] = (interactive_field){.flags = flags};
}

#include "ds4_interactive_edit.h"

static const interactive_behavior interactive_behaviors[] = {
    {
        .name = "edit_file.v2",
        .request_kind = "file_snapshot",
        .description = interactive_edit_description,
        .diagnostic_probe = "<delete/>",
        .configure = interactive_edit_config,
        .string_prefix = interactive_edit_string_prefix,
        .string_prefix_oracle = interactive_edit_string_prefix_oracle,
        .commit_string = interactive_edit_commit_string,
        .plain_replacement = interactive_edit_plain_replacement,
        .string_characters = interactive_edit_characters,
        .sampling = interactive_edit_sampling,
        .await_input = interactive_edit_await,
        .begin_input = interactive_edit_begin_input,
        .complete = interactive_edit_complete,
        .reply_bytes = interactive_edit_reply_bytes,
        .reply_memory = interactive_edit_reply_memory,
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
        if (!INTERACTIVE_KEYS(spec, "name", "behavior", "fields") ||
            !name || !interactive_behavior_find(behavior) ||
            !id_list_contains(&r->allowed_tools, name)) goto invalid;
        for (int j = 0; j < ir->count; j++)
            if (!strcmp(ir->tools[j].name, name)) goto invalid;
        const tool_schema_order *order = tool_schema_orders_find(&r->tool_orders, name);
        if (!order || !order->schema || order->len > INTERACTIVE_MAX_FIELDS) goto invalid;
        interactive_tool *t = &ir->tools[ir->count++];
        t->name = xstrdup(name);
        t->behavior = interactive_behavior_find(behavior);
        if (!t->behavior->configure(t, spec, order)) goto invalid;
    }
    return true;
invalid:
    snprintf(err, errlen, "invalid or unsupported interactive v2 configuration (document must be one required string without enum/const)");
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

/* Add the server-owned framing contract to the actual model-facing tool
 * description. The client's static registry hash remains the source identity. */
static void interactive_tool_descriptions(buf *schemas, const request *r) {
    if (!r->interactive.count) return;
    buf rendered = {0};
    const char *p = schemas->ptr;
    while (p && *p) {
        json_ws(&p);
        if (!*p) break;
        size_t nodes = 0;
        schema_json_value *tool = schema_json_parse_value(&p, 0, &nodes);
        if (!tool) { buf_free(&rendered); return; }
        const char *name = interactive_string(tool, "name");
        const char *guidance = NULL;
        for (int i = 0; name && i < r->interactive.count; i++)
            if (!strcmp(name, r->interactive.tools[i].name)) guidance = r->interactive.tools[i].behavior->description;
        if (guidance) {
            schema_json_value *description = (schema_json_value *)schema_json_object_get(tool, "description");
            if (!description) {
                description = xmalloc(sizeof(*description)); memset(description, 0, sizeof(*description));
                description->kind = SCHEMA_JSON_STRING;
                tool->member = xrealloc(tool->member, (tool->len + 1) * sizeof(*tool->member));
                tool->member[tool->len++] = (schema_json_member){.key = xstrdup("description"), .value = description};
            }
            if (description->kind == SCHEMA_JSON_STRING &&
                (!description->string || !strstr(description->string, guidance))) {
                buf text = {0};
                if (description->string && description->string[0]) {
                    buf_puts(&text, description->string); buf_putc(&text, ' ');
                }
                buf_puts(&text, guidance);
                free(description->string); description->string = buf_take(&text);
            }
        }
        if (rendered.len) buf_putc(&rendered, '\n');
        schema_json_serialize_value(&rendered, tool);
        schema_json_free(tool);
    }
    buf_free(schemas); *schemas = rendered;
}

#endif
