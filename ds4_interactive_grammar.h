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
    int first = interactive_next_priority(t, dt->seen_props);
    if (first >= 0) return prop == first;
    interactive_call *call = interactive_call_for(r, dt->invoke_body_start);
    if (!call || !call->authorized) return false;
    return interactive_field_dependencies_met(t, prop, dt->seen_props);
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
        const dsml_decode_tracker *dt, const char *raw, size_t end, bool complete) {
    const interactive_tool *t = interactive_tool_for(r, dt->active_order);
    if (!t) return agentic_string_parameter_prefix_possible(dt->active_order,
        dt->active_prop, raw, dt->param_value_start, end, complete, dt->syn);
    const schema_json_value *schema = tool_schema_order_property_schema(dt->active_order, dt->active_prop);
    if (complete) {
        schema_json_value *value = interactive_parameter_value(dt->active_order,
            dt->active_prop, true, raw, dt->param_value_start, end, dt->syn);
        bool valid = value && interactive_utf8((const unsigned char *)value->string) &&
            schema_json_validate(schema, value, 0);
        schema_json_free(value);
        if (!valid) return false;
    }
    if (!interactive_field_has(t, dt->active_prop, INTERACTIVE_FIELD_CONSTRAINED))
        return agentic_string_parameter_prefix_possible(dt->active_order,
            dt->active_prop, raw, dt->param_value_start, end, complete, dt->syn);
    interactive_call *call = interactive_call_for(r, dt->invoke_body_start);
    if (!call || !call->authorized) return false;
    const char *value = raw + dt->param_value_start;
    size_t n = end - dt->param_value_start;
    schema_json_value *previous = NULL;
    if (interactive_field_needs_previous(t, dt->active_prop)) {
        char *args = interactive_arguments(dt->active_order, raw,
                                           dt->invoke_body_start,
                                           dt->param_header_start, dt->syn);
        previous = schema_json_parse(args);
        free(args);
        if (!previous) return false;
    }
    bool valid = t->behavior->string_prefix(t, call->behavior_state, dt->active_prop,
                                            previous, value, n, complete);
    schema_json_free(previous);
    return valid;
}

static bool interactive_boundary(const request *r, const dsml_decode_tracker *dt,
                                   int closed_prop, size_t raw_len) {
    const interactive_tool *t = interactive_tool_for(r, dt->active_order);
    if (!t) return true;
    interactive_call *call = interactive_call_for(r, dt->invoke_body_start);
    if ((!call || !call->authorized) &&
        interactive_priority_complete(t, dt->seen_props)) return dt->pos == raw_len;
    if (interactive_field_has(t, closed_prop, INTERACTIVE_FIELD_CLOSE_BOUNDARY))
        return dt->pos == raw_len;
    return true;
}

static bool interactive_value_valid(const request *r, const dsml_decode_tracker *dt,
                                     bool string, const char *raw, size_t end) {
    if (string && interactive_tool_for(r, dt->active_order))
        return interactive_string_possible(r, dt, raw, end, true);
    return agentic_validate_parameter_value(dt->active_order, dt->active_prop,
        string, raw, dt->param_value_start, end, dt->syn);
}

static bool interactive_invoke_valid(const request *r, const dsml_decode_tracker *dt,
                                     const char *raw, size_t end) {
    if (!interactive_tool_for(r, dt->active_order))
        return agentic_invoke_state_valid(dt, raw, dt->invoke_body_start, end, dt->syn);
    char *args = interactive_arguments(dt->active_order, raw, dt->invoke_body_start, end, dt->syn);
    bool valid = args && schema_json_validate_raw(dt->active_order->schema, args);
    free(args);
    return valid;
}

static bool interactive_sampling_body(const request *r, const dsml_decode_tracker *dt,
                                       const char *raw, size_t n) {
    const interactive_tool *tool = interactive_tool_for(r, dt->active_order);
    if (!tool || dt->mode != DSML_TRACK_STRING_BODY) return false;
    interactive_call *call = interactive_call_for(r, dt->invoke_body_start);
    if (!call || !call->authorized) return false;
    return tool->behavior->sampling_body(tool, call->behavior_state, dt->active_prop,
                                         raw + dt->param_value_start,
                                         n - dt->param_value_start);
}

static float interactive_temperature(const request *r, const dsml_decode_tracker *dt,
                                      float temperature) {
    const interactive_tool *tool = interactive_tool_for(r, dt->active_order);
    if (interactive_field_has(tool, dt->active_prop, INTERACTIVE_FIELD_GREEDY)) return 0.0f;
    if (r->interactive.count && dsml_decode_state_is_tool(dt->decode) &&
        !dsml_decode_state_uses_payload_sampling(dt->decode)) return 0.0f;
    return temperature;
}

#endif
