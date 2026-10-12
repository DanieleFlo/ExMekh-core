#ifndef DS4_INTERACTIVE_GRAMMAR_H
#define DS4_INTERACTIVE_GRAMMAR_H

/* Interactive string bodies use the literal native representation, including
 * leading/trailing newlines. In particular Qwen's optional framing-newline
 * heuristic must not remove bytes from an edit selection or replacement. */
static schema_json_value *interactive_parameter_value(
        const tool_schema_order *order, int prop, bool is_string,
        const char *raw, size_t start, size_t end, const dsml_syntax *syn) {
    if (!is_string) return agentic_parameter_value(order, prop, false, raw, start, end, syn);
    schema_json_value *v = xmalloc(sizeof(*v));
    memset(v, 0, sizeof(*v));
    v->kind = SCHEMA_JSON_STRING;
    v->string = xstrndup(raw + start, end - start);
    ds4_tool_text_unescape(v->string, syn->param_end);
    return v;
}

static char *interactive_arguments(const tool_schema_order *order,
                                    const char *raw, size_t start, size_t end,
                                    const dsml_syntax *syn) {
    buf b = {0};
    buf_putc(&b, '{');
    bool wrote = false;
    size_t pos = start;
    while (pos < end) {
        while (pos < end && isspace((unsigned char)raw[pos])) pos++;
        if (pos == end) break;
        int prop = -1;
        bool string = false;
        size_t after = 0;
        if (agentic_sim_param_header(order, raw, end, pos, syn,
                                     &prop, &string, &after) != AGENTIC_TAG_COMPLETE) goto fail;
        size_t body_end = after;
        bool quoted = false, escaped = false;
        while (body_end < end) {
            if ((string || !quoted || syn == &qwen_tool_syntax) &&
                raw_full_lit(raw, end, body_end, syn->param_end)) break;
            char c = raw[body_end++];
            if (!string) {
                if (quoted) {
                    if (escaped) escaped = false;
                    else if (c == '\\') escaped = true;
                    else if (c == '"') quoted = false;
                } else if (c == '"') quoted = true;
            }
        }
        if (body_end == end) goto fail;
        schema_json_value *value = interactive_parameter_value(
            order, prop, string, raw, after, body_end, syn);
        if (!value) goto fail;
        if (wrote) buf_putc(&b, ',');
        json_escape(&b, order->prop[prop]);
        buf_putc(&b, ':');
        schema_json_serialize_value(&b, value);
        schema_json_free(value);
        wrote = true;
        pos = body_end + strlen(syn->param_end);
    }
    buf_putc(&b, '}');
    return buf_take(&b);
fail:
    buf_free(&b);
    return NULL;
}

static bool interactive_param_allowed(const request *r,
        const dsml_decode_tracker *dt, int prop) {
    const interactive_tool *t = interactive_tool_for(r, dt->active_order);
    if (!t) return true;
    if (prop < 0 || prop >= INTERACTIVE_MAX_FIELDS || !dt->seen_props_exact) return false;
    return interactive_field_has(t, prop, INTERACTIVE_FIELD_CONSTRAINED);
}

static bool interactive_parameter_form(const request *r, const dsml_decode_tracker *dt,
                                        int prop, bool string) {
    if (!interactive_tool_for(r, dt->active_order)) return true;
    const schema_json_value *schema = tool_schema_order_property_schema(dt->active_order, prop);
    /* String=false JSON cannot bypass the dynamic string recognizer. */
    if (string) return schema_json_allows_kind(schema, SCHEMA_JSON_STRING, 0);
    const schema_json_kind kinds[] = {SCHEMA_JSON_NULL, SCHEMA_JSON_BOOL, SCHEMA_JSON_NUMBER,
                                     SCHEMA_JSON_ARRAY, SCHEMA_JSON_OBJECT};
    for (size_t i = 0; i < sizeof(kinds)/sizeof(kinds[0]); i++)
        if (schema_json_allows_kind(schema, kinds[i], 0)) return true;
    return false;
}

static bool interactive_string_possible(const request *r,
        const dsml_decode_tracker *dt, const char *raw, size_t end, bool complete, size_t new_from) {
    const interactive_tool *t = interactive_tool_for(r, dt->active_order);
    if (!t) return agentic_string_parameter_prefix_possible(dt->active_order,
        dt->active_prop, raw, dt->param_value_start, end, complete, dt->syn);
    const schema_json_value *schema = tool_schema_order_property_schema(dt->active_order, dt->active_prop);
    const schema_json_value *maximum = schema_json_object_get(schema, "maxLength");
    schema_json_value *decoded = interactive_parameter_value(dt->active_order, dt->active_prop,
        true, raw, dt->param_value_start, end, dt->syn);
    bool valid = interactive_utf8_valid((const unsigned char *)decoded->string,
        strlen(decoded->string), complete);
    size_t length = 0;
    for (const unsigned char *p = (const unsigned char *)decoded->string; *p; p++)
        if ((*p & 0xc0) != 0x80) length++;
    if (maximum && (double)length > maximum->number) valid = false;
    if (complete && valid) valid = schema_json_validate(schema, decoded, 0);
    schema_json_free(decoded);
    if (!valid) return false;
    if (!interactive_field_has(t, dt->active_prop, INTERACTIVE_FIELD_CONSTRAINED))
        return agentic_string_parameter_prefix_possible(dt->active_order,
            dt->active_prop, raw, dt->param_value_start, end, complete, dt->syn);
    interactive_call *call = interactive_call_for(r, dt->invoke_body_start);
    const char *value = raw + dt->param_value_start;
    size_t n = end - dt->param_value_start;
    size_t committed = new_from > dt->param_value_start ? new_from - dt->param_value_start : 0;
    if (committed > n) committed = n;
    return t->behavior->string_prefix(t, call ? call->behavior_state : NULL, dt->active_prop,
        value, n, complete, committed, dt->syn->param_end);
}

static bool interactive_value_valid(const request *r, const dsml_decode_tracker *dt,
                                     bool string, const char *raw, size_t end) {
    if (string && interactive_tool_for(r, dt->active_order))
        return interactive_string_possible(r, dt, raw, end, true, end);
    return agentic_validate_parameter_value(dt->active_order, dt->active_prop,
        string, raw, dt->param_value_start, end, dt->syn);
}

static bool interactive_invoke_valid(const request *r, const dsml_decode_tracker *dt,
                                     const char *raw, size_t end) {
    if (!interactive_tool_for(r, dt->active_order))
        return agentic_invoke_state_valid(dt, raw, dt->invoke_body_start, end, dt->syn);
    char *args = interactive_arguments(dt->active_order, raw, dt->invoke_body_start, end, dt->syn);
    schema_json_value *parsed = args ? schema_json_parse(args) : NULL;
    bool valid = parsed && schema_json_validate(dt->active_order->schema, parsed, 0);
    const interactive_tool *tool = interactive_tool_for(r, dt->active_order);
    interactive_call *call = interactive_call_for(r, dt->invoke_body_start);
    if (valid && tool && tool->behavior->await_input) {
        valid = call && call->authorized &&
            tool->behavior->complete(tool, call->behavior_state, parsed);
    }
    schema_json_free(parsed);
    free(args);
    return valid;
}

static interactive_sampling_policy interactive_sampling(const request *r,
        const dsml_decode_tracker *dt, const char *raw, size_t n) {
    const interactive_tool *tool = interactive_tool_for(r, dt->active_order);
    if (!tool || dt->mode != DSML_TRACK_STRING_BODY) return (interactive_sampling_policy){0};
    interactive_call *call = interactive_call_for(r, dt->invoke_body_start);
    interactive_sampling_policy policy = tool->behavior->sampling(tool,
        call ? call->behavior_state : NULL, dt->active_prop,
        raw + dt->param_value_start, n - dt->param_value_start);
    if (!call || !call->authorized) policy.body = false;
    return policy;
}

static bool interactive_sampling_body(const request *r, const dsml_decode_tracker *dt,
                                       const char *raw, size_t n) {
    return interactive_sampling(r, dt, raw, n).body;
}

static uint64_t interactive_sampling_epoch(const request *r, const dsml_decode_tracker *dt,
                                           const char *raw, size_t n) {
    return interactive_sampling(r, dt, raw, n).epoch;
}

static float interactive_temperature(const request *r, const dsml_decode_tracker *dt,
                                      float temperature, const char *raw, size_t n) {
    if (interactive_sampling(r, dt, raw, n).greedy) return 0.0f;
    if (r->interactive.count && dsml_decode_state_is_tool(dt->decode) &&
        !dsml_decode_state_uses_payload_sampling(dt->decode)) return 0.0f;
    return temperature;
}

#endif
