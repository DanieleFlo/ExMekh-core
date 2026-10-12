#ifndef DS4_INTERACTIVE_RUNTIME_H
#define DS4_INTERACTIVE_RUNTIME_H

typedef enum {
    INTERACTIVE_RUNNING, INTERACTIVE_AWAITING, INTERACTIVE_COMPLETED,
    INTERACTIVE_REJECTED, INTERACTIVE_CANCELLED, INTERACTIVE_EXPIRED,
    INTERACTIVE_FAILED
} interactive_status;

struct interactive_receipt {
    char *id;
    char hash[41];
    char *result;
    interactive_receipt *next;
};

struct interactive_entry {
    server *owner;
    char id[40];
    char *context_id;
    stop_list allowed_tools, allowed_skills;
    interactive_status status;
    time_t expires;
    int slot_id;
    request paused;
    bool owns_request, streaming;
    bool queued, finalizing;
    size_t snapshot_bytes;
    size_t behavior_bytes;
    uint64_t checkpoint_bytes;
    char *text;
    uint64_t rng;
    int generated, prompt_tokens;
    thinking_state thinking;
    responses_stream stream;
    interactive_call *pending;
    interactive_receipt *receipts, *operation;
    job *running;
    interactive_entry *next;
};

static const char *interactive_status_name(interactive_status status) {
    static const char *names[] = {"generating", "awaiting_input", "completed", "rejected",
                                  "cancelled", "expired", "failed"};
    return names[status];
}

static bool interactive_active(const interactive_entry *e) {
    return e->status == INTERACTIVE_RUNNING || e->status == INTERACTIVE_AWAITING;
}

static char *interactive_capabilities_body(server_model_syntax syntax) {
    buf body = {0};
    buf_puts(&body, "{\"interactive\":{\"version\":2,\"apis\":[\"responses\"],\"behaviors\":[");
    for (size_t i = 0; i < sizeof(interactive_behaviors)/sizeof(interactive_behaviors[0]); i++) {
        if (i) buf_putc(&body, ',');
        json_escape(&body, interactive_behaviors[i].name);
    }
    buf_puts(&body, "],\"syntaxes\":[");
    for (size_t i = 0; i < sizeof(interactive_syntaxes)/sizeof(interactive_syntaxes[0]); i++) {
        if (i) buf_putc(&body, ',');
        json_escape(&body, server_model_syntax_name(interactive_syntaxes[i]));
    }
    buf_puts(&body, "],\"model_syntax\":");
    json_escape(&body, server_model_syntax_name(syntax));
    buf_printf(&body, ",\"model_supported\":%s,\"limits\":{\"snapshot_bytes\":%u,\"generated_text_bytes\":%u,\"first_values_bytes\":%u,"
        "\"tools\":%u,\"fields\":%u,\"records\":%u,\"active_continuations\":%u,\"exchanges\":%u,\"snapshot_budget_bytes\":%u,"
        "\"state_budget_bytes\":%u,\"edits_per_call\":%u,"
        "\"checkpoint_budget_bytes\":%llu,\"continuation_ttl_seconds\":%u,\"receipt_ttl_seconds\":%u},"
        "\"mtp\":\"verified_parameter\",\"batched_mtp\":false,\"restart_resume\":false}}\n",
        interactive_syntax_supported(syntax) ? "true" : "false", INTERACTIVE_MAX_SNAPSHOT,
        INTERACTIVE_MAX_TEXT, INTERACTIVE_MAX_VALUES, INTERACTIVE_MAX_TOOLS, INTERACTIVE_MAX_FIELDS,
        INTERACTIVE_MAX_RECORDS, INTERACTIVE_MAX_ACTIVE, INTERACTIVE_MAX_RECORDS, INTERACTIVE_SNAPSHOT_BUDGET,
        INTERACTIVE_STATE_BUDGET, INTERACTIVE_MAX_RECORDS,
        (unsigned long long)INTERACTIVE_CHECKPOINT_BUDGET, INTERACTIVE_TTL, INTERACTIVE_RECEIPT_TTL);
    return buf_take(&body);
}

static int interactive_request_slot(const request *r) {
    return r->interactive.resumed && r->interactive.entry ? r->interactive.entry->slot_id : -1;
}

static int interactive_prompt_tokens(const request *r, int fallback) {
    return r->interactive.resumed ? r->interactive.entry->prompt_tokens : fallback;
}

static char *interactive_acceptance_body(const request *r) {
    buf body = {0};
    buf_printf(&body, "{\"type\":\"interactive.accepted\",\"version\":2,\"continuation_id\":\"%s\"}",
               r->interactive.entry->id);
    return buf_take(&body);
}

typedef struct {
    uint64_t rng;
    int generated;
    thinking_state thinking;
} interactive_generation_state;

/* Consume the saved buffers exactly once. Response IDs and sequence numbers
 * belong to the new HTTP response; cursors and item IDs belong to the turn. */
static void interactive_resume_generation(const request *r, buf *text,
                                           responses_stream *stream,
                                           interactive_generation_state *state) {
    if (!r->interactive.resumed) return;
    interactive_entry *entry = r->interactive.entry;
    state->rng = entry->rng;
    state->generated = entry->generated;
    state->thinking = entry->thinking;
    if (!entry->text) return;
    buf_puts(text, entry->text);
    free(entry->text);
    entry->text = NULL;
    if (r->stream) {
        char response_id[sizeof(stream->response_id)];
        memcpy(response_id, stream->response_id, sizeof(response_id));
        int sequence = stream->sequence;
        responses_stream_free(stream);
        *stream = entry->stream;
        memset(&entry->stream, 0, sizeof(entry->stream));
        memcpy(stream->response_id, response_id, sizeof(response_id));
        stream->sequence = sequence;
    }
}

static bool interactive_checkpoint_reserve_locked(server *s, int slot_id,
        const char *context_id, uint64_t bytes, bool *pending) {
    interactive_entry *owner = NULL;
    uint64_t used = 0;
    for (interactive_entry *entry = s->interactive_entries; entry; entry = entry->next) {
        used += entry->checkpoint_bytes;
        if (entry->slot_id == slot_id && !strcmp(entry->context_id, context_id) && interactive_active(entry))
            owner = entry;
    }
    *pending = owner != NULL;
    if (!owner) return true;
    if (bytes > INTERACTIVE_CHECKPOINT_BUDGET ||
        used - owner->checkpoint_bytes > INTERACTIVE_CHECKPOINT_BUDGET - bytes) return false;
    owner->checkpoint_bytes = bytes;
    return true;
}

static void interactive_append_acceptance(buf *b, const request *r) {
    if (r->interactive.entry)
        buf_printf(b, ",\"interactive\":{\"version\":2,\"continuation_id\":\"%s\"}", r->interactive.entry->id);
}

static void interactive_append_interruption(buf *body, const interactive_entry *e,
                                             const char *status, const char *code,
                                             const char *message) {
    buf_printf(body, "{\"type\":\"interactive.%s\",\"version\":2,\"continuation_id\":\"%s\",\"status\":\"%s\",\"code\":",
        status, e->id, status);
    json_escape(body, code);
    buf_puts(body, ",\"message\":");
    json_escape(body, message);
    if (e->pending) {
        buf_printf(body, ",\"call_id\":\"%s\",\"exchange_id\":\"%s\",\"tool_name\":",
            e->pending->call_id, e->pending->exchange_id);
        json_escape(body, e->pending->tool->name);
        buf_printf(body, ",\"edit_index\":%zu", e->pending->input_index);
        buf_puts(body, ",\"values\":");
        buf_puts(body, e->pending->values);
    }
}

static char *interactive_interrupted_body(const interactive_entry *e,
                                          const char *status, const char *code,
                                          const char *message) {
    buf b = {0};
    interactive_append_interruption(&b, e, status, code, message);
    buf_puts(&b, ",\"output\":[]}");
    return buf_take(&b);
}

static void interactive_drop_checkpoint_locked(server *s, interactive_entry *e) {
    e->checkpoint_bytes = 0;
    if (e->slot_id < 0 || e->slot_id >= s->slot_count) return;
    server_slot *slot = &s->slots[e->slot_id];
    agent_context_checkpoint *cp = agent_context_checkpoint_find_locked(slot, e->context_id);
    if (cp) agent_context_checkpoint_remove_locked(slot, cp, true);
}

static void interactive_release_payload(interactive_entry *e) {
    if (e->owns_request) request_free(&e->paused);
    e->owns_request = false;
    free(e->text); e->text = NULL;
    responses_stream_free(&e->stream);
    e->pending = NULL;
    e->snapshot_bytes = 0;
    e->behavior_bytes = 0;
}

static void interactive_entry_free(interactive_entry *e) {
    interactive_release_payload(e);
    free(e->context_id);
    id_list_free(&e->allowed_tools); id_list_free(&e->allowed_skills);
    while (e->receipts) {
        interactive_receipt *r = e->receipts;
        e->receipts = r->next;
        free(r->id); free(r->result); free(r);
    }
    free(e);
}

/* Called after workers have drained, before the server destroys its storage. */
static void interactive_close(server *s) {
    while (s->interactive_entries) {
        interactive_entry *entry = s->interactive_entries;
        s->interactive_entries = entry->next;
        interactive_entry_free(entry);
    }
}

static void interactive_sweep_locked(server *s) {
    time_t now = time(NULL);
    interactive_entry **p = &s->interactive_entries;
    while (*p) {
        interactive_entry *e = *p;
        if (e->expires > now || e->running || e->queued || e->finalizing ||
            e->status == INTERACTIVE_RUNNING) {
            p = &e->next;
            continue;
        }
        if (e->status == INTERACTIVE_AWAITING) {
            char *expired = interactive_interrupted_body(e, "expired", "expired", "interactive continuation expired");
            if (e->operation) { free(e->operation->result); e->operation->result = expired; }
            else free(expired);
            interactive_drop_checkpoint_locked(s, e);
            interactive_release_payload(e);
            e->status = INTERACTIVE_EXPIRED;
            e->expires = now + INTERACTIVE_RECEIPT_TTL;
            p = &e->next;
        } else {
            *p = e->next;
            interactive_entry_free(e);
        }
    }
}

static bool interactive_scope_equal(const stop_list *a, const stop_list *b) {
    if (a->len != b->len) return false;
    for (int i = 0; i < a->len; i++) if (!id_list_contains(b, a->v[i])) return false;
    return true;
}

/* Sort object keys only; array order (tool specs, input items) is semantic. */
static int interactive_member_compare(const void *a, const void *b) {
    return strcmp(((const schema_json_member *)a)->key, ((const schema_json_member *)b)->key);
}

static void interactive_canonicalize(schema_json_value *v) {
    if (!v) return;
    for (size_t i = 0; i < v->len; i++) {
        if (v->kind == SCHEMA_JSON_OBJECT) interactive_canonicalize(v->member[i].value);
        if (v->kind == SCHEMA_JSON_ARRAY) interactive_canonicalize(v->item[i]);
    }
    if (v->kind == SCHEMA_JSON_OBJECT) qsort(v->member, v->len, sizeof(*v->member), interactive_member_compare);
}

static bool interactive_unique_keys(const schema_json_value *v) {
    if (!v) return false;
    for (size_t i = 0; i < v->len; i++) {
        if (v->kind == SCHEMA_JSON_OBJECT) {
            for (size_t j = 0; j < i; j++)
                if (!strcmp(v->member[i].key, v->member[j].key)) return false;
            if (!interactive_unique_keys(v->member[i].value)) return false;
        } else if (v->kind == SCHEMA_JSON_ARRAY && !interactive_unique_keys(v->item[i])) return false;
    }
    return true;
}

static bool interactive_nul_escape(const char *raw);

static bool interactive_hash_body(const char *body, char hash[41]) {
    schema_json_value *v = schema_json_parse(body);
    if (!v || !interactive_unique_keys(v) || interactive_nul_escape(body)) {
        schema_json_free(v);
        return false;
    }
    interactive_canonicalize(v);
    buf canonical = {0};
    schema_json_serialize_value(&canonical, v);
    ds4_kvstore_sha1_bytes_hex(canonical.ptr, canonical.len, hash);
    buf_free(&canonical); schema_json_free(v);
    return true;
}

static interactive_receipt *interactive_receipt_find(interactive_entry *e, const char *id) {
    for (interactive_receipt *r = e->receipts; r; r = r->next)
        if (!strcmp(r->id, id)) return r;
    return NULL;
}

static interactive_receipt *interactive_receipt_add(interactive_entry *e,
                                                   const char *id, const char *hash) {
    interactive_receipt *r = xmalloc(sizeof(*r));
    memset(r, 0, sizeof(*r));
    r->id = xstrdup(id);
    snprintf(r->hash, sizeof(r->hash), "%s", hash);
    r->next = e->receipts; e->receipts = e->operation = r;
    return r;
}

static void interactive_record_response_locked(const request *r, const char *body) {
    interactive_entry *e = r->interactive.entry;
    interactive_receipt *receipt = r->interactive.receipt;
    if (!e || !receipt) return;
    free(receipt->result);
    receipt->result = xstrdup(body);
    schema_json_value *v = schema_json_parse(body);
    const char *type = interactive_string(v, "type");
    const schema_json_value *response = schema_json_object_get(v, "response");
    const char *status = interactive_string(response ? response : v, "status");
    if (e->status == INTERACTIVE_RUNNING &&
        ((type && !strcmp(type, "response.completed")) ||
         (status && !strcmp(status, "completed")))) e->status = INTERACTIVE_COMPLETED;
    else if (e->status == INTERACTIVE_RUNNING &&
             ((type && !strcmp(type, "interactive.failed")) ||
              (status && (!strcmp(status, "failed") || !strcmp(status, "incomplete")))))
        e->status = INTERACTIVE_FAILED;
    if (!interactive_active(e)) e->expires = time(NULL) + INTERACTIVE_RECEIPT_TTL;
    schema_json_free(v);
}

static void interactive_record_response(const request *r, const char *body) {
    interactive_entry *e = r->interactive.entry;
    if (!e) return;
    pthread_mutex_lock(&e->owner->tool_mu);
    interactive_record_response_locked(r, body);
    pthread_mutex_unlock(&e->owner->tool_mu);
}

static void interactive_rebind(request *r) {
    for (interactive_call *c = r->interactive.calls; c; c = c->next)
        c->tool = &r->interactive.tools[c->tool_index];
}

static bool interactive_send(int fd, bool cors, bool stream, const char *body) {
    if (!stream) return http_response(fd, cors, 200, "application/json", body);
    schema_json_value *v = schema_json_parse(body);
    const char *type = interactive_string(v, "type");
    bool ok;
    if (type) {
        buf wire = {0};
        buf_printf(&wire, "event: %s\ndata: %s\n\n", type, body);
        ok = send_all(fd, wire.ptr, wire.len);
        buf_free(&wire);
    } else {
        buf wire = {0};
        buf_printf(&wire, "event: interactive.result\ndata: {\"type\":\"interactive.result\",\"response\":%s}\n\n", body);
        ok = send_all(fd, wire.ptr, wire.len);
        buf_free(&wire);
    }
    schema_json_free(v);
    return ok;
}

static char *interactive_status_body(const interactive_entry *e) {
    buf b = {0};
    buf_printf(&b, "{\"type\":\"interactive.status\",\"version\":2,\"continuation_id\":\"%s\",\"status\":\"%s\",\"result\":",
               e->id, interactive_status_name(e->status));
    buf_puts(&b, e->operation && e->operation->result ? e->operation->result : "null");
    buf_putc(&b, '}');
    return buf_take(&b);
}

/* Returns true when it already answered the HTTP request (including retries). */
static bool interactive_admit(server *s, int fd, request *r) {
    pthread_mutex_lock(&s->tool_mu);
    interactive_sweep_locked(s);
    interactive_entry *context = NULL;
    size_t records = 0, active = 0;
    for (interactive_entry *e = s->interactive_entries; e; e = e->next) {
        records++;
        if (interactive_active(e)) active++;
        if (r->agent_context_id && !strcmp(e->context_id, r->agent_context_id) && interactive_active(e)) context = e;
    }
    if (!r->interactive.count) {
        pthread_mutex_unlock(&s->tool_mu);
        if (!context) return false;
        http_error(fd, s->enable_cors, 409, "agent context has an active interactive turn");
        return true;
    }
    const char *id = interactive_string(r->interactive.wire, "request_id");
    char hash[41];
    if (!interactive_hash_body(r->raw_body, hash)) {
        pthread_mutex_unlock(&s->tool_mu);
        http_error(fd, s->enable_cors, 400, "invalid interactive request");
        return true;
    }
    for (interactive_entry *e = s->interactive_entries; e; e = e->next) {
        if (strcmp(e->context_id, r->agent_context_id)) continue;
        interactive_receipt *receipt = interactive_receipt_find(e, id);
        if (!receipt) continue;
        if (strcmp(receipt->hash, hash)) {
            pthread_mutex_unlock(&s->tool_mu);
            http_error(fd, s->enable_cors, 409, "request_id reused with different payload");
        } else {
            char *body = receipt->result ? xstrdup(receipt->result) : interactive_status_body(e);
            pthread_mutex_unlock(&s->tool_mu);
            if (r->stream) sse_headers(fd, s->enable_cors);
            interactive_send(fd, s->enable_cors, r->stream, body);
            free(body);
        }
        return true;
    }
    if (context || records >= INTERACTIVE_MAX_RECORDS || active >= INTERACTIVE_MAX_ACTIVE) {
        pthread_mutex_unlock(&s->tool_mu);
        http_error(fd, s->enable_cors, context ? 409 : 429,
                   context ? "agent context has an active interactive turn" : "interactive capacity reached");
        return true;
    }
    interactive_entry *e = xmalloc(sizeof(*e));
    memset(e, 0, sizeof(*e));
    e->owner = s;
    responses_random_id(e->id, sizeof(e->id), "cont_");
    e->context_id = xstrdup(r->agent_context_id);
    e->slot_id = -1;
    for (int i = 0; i < r->allowed_tools.len; i++) id_list_push_unique(&e->allowed_tools, r->allowed_tools.v[i]);
    for (int i = 0; i < r->allowed_skills.len; i++) id_list_push_unique(&e->allowed_skills, r->allowed_skills.v[i]);
    e->status = INTERACTIVE_RUNNING;
    e->streaming = r->stream;
    e->expires = time(NULL) + INTERACTIVE_TTL;
    r->interactive.receipt = interactive_receipt_add(e, id, hash);
    e->queued = true;
    e->next = s->interactive_entries; s->interactive_entries = e;
    r->interactive.entry = e;
    pthread_mutex_unlock(&s->tool_mu);
    return false;
}

static interactive_call *interactive_observe(request *r,
        const dsml_decode_tracker *dt, const char *raw, size_t n) {
    const interactive_tool *tool = interactive_tool_for(r, dt->active_order);
    if (!tool || !dt->inside_invoke || !dt->seen_props_exact ||
        dt->mode != DSML_TRACK_STRING_BODY || dt->pos != n) return NULL;
    interactive_call *call = interactive_call_for(r, dt->invoke_body_start);
    if (call && !call->authorized) return call;
    buf values = {0};
    size_t input_index = 0;
    if (!tool->behavior->await_input(tool, call ? call->behavior_state : NULL,
            dt->active_prop, raw + dt->param_value_start, n - dt->param_value_start,
            &values, &input_index, dt->syn->param_end)) { buf_free(&values); return NULL; }
    size_t count = 0;
    for (interactive_call *c = r->interactive.calls; c; c = c->next) count++;
    if (values.len > INTERACTIVE_MAX_VALUES || (!call && count >= INTERACTIVE_MAX_RECORDS)) {
        buf_free(&values); return NULL;
    }
    if (!call) {
        call = xmalloc(sizeof(*call)); memset(call, 0, sizeof(*call));
        call->body_start = dt->invoke_body_start;
        call->header_start = dt->invoke_header_start;
        call->param_end = dt->syn->param_end;
        call->wrapper_end = dt->syn->tool_calls_end;
        call->tool = tool;
        call->tool_index = (int)(tool - r->interactive.tools);
        responses_random_id(call->call_id, sizeof(call->call_id), "call_");
        call->next = r->interactive.calls; r->interactive.calls = call;
    }
    if (!tool->behavior->begin_input(&call->behavior_state, values.ptr, dt->syn->param_end)) {
        buf_free(&values); return NULL;
    }
    free(call->values); call->values = buf_take(&values);
    call->input_index = input_index;
    call->authorized = false;
    responses_random_id(call->exchange_id, sizeof(call->exchange_id), "exchange_");
    return call;
}

/* Existing phase profiling also audits the first payload distribution. No
 * RNG or session state is sampled/changed: report the raw argmax and whether
 * the grammar admits it, plus the behavior-owned operation probe. */
static void interactive_profile_frontier(server *s, server_slot *slot,
        const dsml_decode_tracker *dt, agentic_token_filter *filter, float temperature) {
    const request *r = filter->req;
    const interactive_tool *tool = interactive_tool_for(r, dt->active_order);
    interactive_call *call = interactive_call_for(r, dt->invoke_body_start);
    if (!tool || !call || !call->authorized || dt->mode != DSML_TRACK_STRING_BODY ||
        !interactive_sampling(r, dt, filter->raw, filter->raw_len).payload_start) return;
    int vocab = ds4_engine_vocab_size(s->engine);
    if (vocab <= 0) return;
    float *logits = xmalloc((size_t)vocab * sizeof(*logits));
    int best = -1;
    if (ds4_session_copy_logits(slot->session, logits, vocab) == vocab) {
        for (int i = 0; i < vocab; i++)
            if (isfinite(logits[i]) && (best < 0 || logits[i] > logits[best])) best = i;
    }
    free(logits);
    if (best < 0) return;
    size_t length = 0;
    char *piece = ds4_token_text(s->engine, best, &length);
    bool allowed = agentic_filter_token(filter, best, piece, length);
    bool probe = tool->behavior->diagnostic_probe &&
        agentic_filter_token(filter, -1, tool->behavior->diagnostic_probe, strlen(tool->behavior->diagnostic_probe));
    buf encoded = {0}; json_escape(&encoded, piece);
    server_log(DS4_LOG_GENERATION,
        "ds4-server: interactive payload frontier call_id=%s input_index=%zu temperature=%.3f "
        "unmasked_best=%d piece=%s raw_argmax_allowed=%s operation_probe_allowed=%s",
        call->call_id, call->input_index, temperature, best, encoded.ptr,
        allowed ? "true" : "false", probe ? "true" : "false");
    free(piece); buf_free(&encoded);
}

static bool interactive_suspend(server *s, server_slot *slot, job *j,
        interactive_call *call, buf *text, uint64_t rng, int generated,
        int prompt_tokens, thinking_state thinking, responses_stream *stream,
        char *err, size_t errlen) {
    interactive_entry *e = j->req.interactive.entry;
    if (!e || !agent_context_checkpoint_save(s, slot, e->context_id, err, errlen)) return false;
    buf b = {0};
    buf_printf(&b, "{\"type\":\"interactive.await_input\",\"version\":2,\"status\":\"awaiting_input\",\"continuation_id\":\"%s\",\"exchange_id\":\"%s\",\"call_id\":\"%s\",\"tool_name\":",
               e->id, call->exchange_id, call->call_id);
    json_escape(&b, call->tool->name);
    buf_puts(&b, ",\"behavior\":"); json_escape(&b, call->tool->behavior->name);
    buf_puts(&b, ",\"request\":{\"kind\":"); json_escape(&b, call->tool->behavior->request_kind);
    buf_printf(&b, ",\"edit_index\":%zu", call->input_index);
    buf_puts(&b, ",\"values\":");
    buf_puts(&b, call->values);
    buf_puts(&b, "},\"usage\":");
    append_responses_usage_json(&b, &j->req, prompt_tokens, generated);
    buf_putc(&b, '}');
    pthread_mutex_lock(&s->tool_mu);
    if (e->status != INTERACTIVE_RUNNING || job_cancelled(j)) {
        interactive_drop_checkpoint_locked(s, e);
        pthread_mutex_unlock(&s->tool_mu);
        buf_free(&b);
        snprintf(err, errlen, "interactive turn cancelled before suspension");
        return false;
    }
    interactive_record_response_locked(&j->req, b.ptr);
    bool is_stream = j->req.stream;
    e->paused = j->req;
    memset(&j->req, 0, sizeof(j->req));
    interactive_rebind(&e->paused);
    e->paused.agentic_return = false;
    free(e->paused.agentic_skill_call_id); e->paused.agentic_skill_call_id = NULL;
    e->paused.responses_requires_live_tool_state = false;
    e->paused.responses_requires_live_reasoning = false;
    id_list_free(&e->paused.responses_live_call_ids);
    free(e->paused.responses_live_suffix_text); e->paused.responses_live_suffix_text = NULL;
    e->owns_request = true;
    free(e->text);
    e->text = buf_take(text);
    e->rng = rng; e->generated = generated; e->prompt_tokens = prompt_tokens;
    e->thinking = thinking;
    e->stream = *stream; memset(stream, 0, sizeof(*stream));
    e->pending = call;
    e->status = INTERACTIVE_AWAITING;
    e->expires = time(NULL) + INTERACTIVE_TTL;
    /* Keep running set until generate_job releases the slot; admission can
     * queue a resume, but never evict this entry while the old job unwinds. */
    pthread_mutex_unlock(&s->tool_mu);
    bool ok = interactive_send(j->fd, s->enable_cors, is_stream, b.ptr);
    if (ok) job_mark_response_published(j);
    buf_free(&b);
    return true;
}

static bool interactive_nul_escape(const char *raw) {
    for (const char *p = raw; *p; p++) {
        if (*p != '\\') continue;
        if (!strncmp(p + 1, "u0000", 5)) return true;
        if (p[1]) p++;
    }
    return false;
}

static bool interactive_apply_call_ids(const request *r, const char *raw,
                                        tool_calls *calls) {
    dsml_decode_tracker dt;
    dsml_decode_tracker_init(&dt, r);
    if (r->think_mode != DS4_THINK_NONE) {
        const char *end = strstr(raw, "</think>");
        if (end) dt.pos = (size_t)(end + 8 - raw);
    }
    /* Final extraction uses the same recognizer boundaries as the mask;
     * override the parser's optional native framing heuristics only here. */
    int index = -1;
    size_t last_body = SIZE_MAX;
    size_t n = strlen(raw);
    size_t surface = dt.pos;
    for (size_t pos = dt.pos + 1; pos <= n; pos++) {
        if (dt.mode == DSML_TRACK_SEARCH && dt.pos < surface) dt.pos = surface;
        dsml_decode_tracker_update(&dt, raw, pos);
        if (dt.inside_invoke && dt.invoke_body_start != last_body) {
            last_body = dt.invoke_body_start;
            index++;
        }
        if (!dt.inside_invoke || dt.mode != DSML_TRACK_STRUCTURAL || !dt.active_order) continue;
        interactive_call *c = interactive_call_for(r, dt.invoke_body_start);
        if (!c || !c->authorized || !raw_full_lit(raw, n, dt.pos, dt.syn->invoke_end)) continue;
        /* Match invocation order instead of tool name: a batch can call the
         * same tool several times against different snapshots. */
        if (index < 0 || index >= calls->len) return false;
        if (strcmp(calls->v[index].name, c->tool->name)) return false;
        char *args = interactive_arguments(dt.active_order, raw,
                                           dt.invoke_body_start, dt.pos, dt.syn);
        if (!args || !schema_json_validate_raw(dt.active_order->schema, args)) { free(args); return false; }
        if (!interactive_invoke_valid(r, &dt, raw, dt.pos)) { free(args); return false; }
        free(calls->v[index].arguments); calls->v[index].arguments = args;
        free(calls->v[index].id); calls->v[index].id = xstrdup(c->call_id);
        /* Jump over the invocation close, avoiding a second rewrite while
         * the forgiving tracker waits for the complete closing tag. */
        pos = dt.pos + strlen(dt.syn->invoke_end) - 1;
    }
    for (int i = 0; i < calls->len; i++) {
        if (!interactive_tool_for(r, tool_schema_orders_find(&r->tool_orders, calls->v[i].name))) continue;
        bool known = false;
        for (interactive_call *c = r->interactive.calls; c; c = c->next)
            if (c->authorized && calls->v[i].id && !strcmp(calls->v[i].id, c->call_id)) known = true;
        if (!known) return false;
    }
    return true;
}

/* The committed thinking state gates the grammar before any tool is parsed.
 * Apply the same boundary to extraction: quoted/incomplete native tags in
 * reasoning must never hide </think> or become executable invocations. */
static bool interactive_parse_message(const request *r, const char *raw,
                                      char **content, char **reasoning,
                                      tool_calls *calls) {
    const char *surface = raw ? raw : "";
    const char *end = ds4_think_mode_enabled(r->think_mode) ? strstr(surface, "</think>") : NULL;
    if (ds4_think_mode_enabled(r->think_mode) && !end) return false;
    if (end) surface = end + 8;
    bool ok = r->model_syntax == SERVER_MODEL_SYNTAX_QWEN4 ?
        parse_qwen4_generated_message_ex(surface, false, content, reasoning, calls, &r->tool_orders) :
        parse_generated_message_ex_for_syntax(r->model_syntax, surface, false, content, reasoning, calls);
    if (end) {
        const char *body = raw;
        if (!strncmp(body, "<think>", 7)) body += 7;
        free(*reasoning);
        *reasoning = xstrndup(body, (size_t)(end - body));
    }
    return ok;
}

static void interactive_terminal(server *s, interactive_entry *e,
                                  interactive_status status, const char *code,
                                  const char *message, bool keep_calls) {
    request *r = &e->paused;
    interactive_call *pending = e->pending;
    buf b = {0};
    interactive_append_interruption(&b, e, interactive_status_name(status), code, message);
    tool_calls calls = {0};
    char *content = NULL, *reasoning = NULL;
    if (keep_calls && e->text && pending) {
        buf prefix = {0};
        buf_append(&prefix, e->text, pending->header_start);
        buf_puts(&prefix, pending->wrapper_end);
        /* Qwen may use separate wrappers. Removing the interrupted invoke
         * leaves its last wrapper empty, so the ordinary parser reports false
         * after collecting earlier complete calls. Validate those calls with
         * the structural recognizer before retaining them. */
        (void)interactive_parse_message(r, prefix.ptr, &content, &reasoning, &calls);
        if (calls.len) {
            if (interactive_apply_call_ids(r, prefix.ptr, &calls)) {
                assign_tool_call_ids(s, &calls, API_RESPONSES);
                tool_memory_remember(s, &calls);
            } else tool_calls_free(&calls);
        }
        buf_free(&prefix);
    }
    buf_puts(&b, ",\"output\":[");
    responses_tool_item *items = NULL;
    responses_tool_items_build(&items, &calls, 0);
    for (int i = 0; i < calls.len; i++) {
        if (i) buf_putc(&b, ',');
        responses_append_function_call_item(&b, &calls.v[i], &items[i], "completed", true, &r->tool_orders);
    }
    buf_puts(&b, "],\"usage\":");
    append_responses_usage_json(&b, r, e->prompt_tokens, e->generated);
    buf_putc(&b, '}');
    free(items); tool_calls_free(&calls); free(content); free(reasoning);
    pthread_mutex_lock(&s->tool_mu);
    interactive_record_response_locked(r, b.ptr);
    interactive_drop_checkpoint_locked(s, e);
    e->status = status;
    e->finalizing = false;
    e->expires = time(NULL) + INTERACTIVE_RECEIPT_TTL;
    interactive_release_payload(e);
    pthread_mutex_unlock(&s->tool_mu);
    buf_free(&b);
}

static void interactive_finish_request(server *s, request *r, bool cancelled) {
    interactive_entry *e = r->interactive.entry;
    if (!e) return;
    pthread_mutex_lock(&s->tool_mu);
    if (e->status == INTERACTIVE_RUNNING)
        e->status = cancelled ? INTERACTIVE_CANCELLED : INTERACTIVE_FAILED;
    if (r->interactive.receipt && !r->interactive.receipt->result) {
        char *failure = interactive_interrupted_body(e, interactive_status_name(e->status),
            "generation_aborted", "interactive generation ended before a result");
        interactive_record_response_locked(r, failure);
        free(failure);
    }
    e->queued = false;
    e->expires = time(NULL) + INTERACTIVE_RECEIPT_TTL;
    interactive_drop_checkpoint_locked(s, e);
    interactive_release_payload(e);
    pthread_mutex_unlock(&s->tool_mu);
}

static interactive_entry *interactive_job_begin(server *s, server_slot *slot,
                                                 job *j, bool *allowed) {
    interactive_entry *entry = j->req.interactive.entry;
    *allowed = true;
    if (!entry) return NULL;
    pthread_mutex_lock(&s->tool_mu);
    *allowed = entry->status == INTERACTIVE_RUNNING;
    entry->queued = false;
    entry->running = j;
    entry->slot_id = slot->id;
    pthread_mutex_unlock(&s->tool_mu);
    return entry;
}

static void interactive_job_end(server *s, job *j, interactive_entry *entry) {
    if (!entry) return;
    pthread_mutex_lock(&s->tool_mu);
    if (entry->running == j) entry->running = NULL;
    bool release = j->req.interactive.entry == entry && entry->status != INTERACTIVE_AWAITING;
    pthread_mutex_unlock(&s->tool_mu);
    if (release) interactive_finish_request(s, &j->req, job_cancelled(j));
}

static void interactive_fail_job(server *s, job *j, interactive_call *call,
                                  const char *code, const char *message,
                                  int prompt_tokens, int generated) {
    interactive_entry *entry = j->req.interactive.entry;
    pthread_mutex_lock(&s->tool_mu);
    entry->pending = call;
    buf body = {0};
    interactive_append_interruption(&body, entry, "failed", code, message);
    pthread_mutex_unlock(&s->tool_mu);
    buf_puts(&body, ",\"output\":[],\"usage\":");
    append_responses_usage_json(&body, &j->req, prompt_tokens, generated);
    buf_putc(&body, '}');
    interactive_record_response(&j->req, body.ptr);
    if (interactive_send(j->fd, s->enable_cors, j->req.stream, body.ptr))
        job_mark_response_published(j);
    buf_free(&body);
}

/* A control request is dispatched before Responses input rendering. It moves
 * the saved request ownership; no synthetic input/history is needed to resume. */
static int interactive_control(server *s, int fd, const char *body, request *out) {
    schema_json_value *root = schema_json_parse(body);
    const schema_json_value *agentic = schema_json_object_get(root, "agentic");
    const schema_json_value *wire = schema_json_object_get(agentic, "interactive");
    const char *op = interactive_string(wire, "operation");
    if (!op || !strcmp(op, "configure")) { schema_json_free(root); return 0; }
    request control = {0};
    control.api = API_RESPONSES;
    const schema_json_value *stream = schema_json_object_get(root, "stream");
    bool streaming = stream && stream->kind == SCHEMA_JSON_BOOL && stream->boolean;
    buf encoded = {0}; schema_json_serialize_value(&encoded, agentic);
    const char *parse = encoded.ptr;
    char err[160] = {0};
    bool parsed = parse_agentic_value(&parse, &control, err, sizeof(err));
    buf_free(&encoded);
    const char *id = interactive_string(wire, "request_id");
    const char *continuation = interactive_string(wire, "continuation_id");
    const char *exchange = interactive_string(wire, "exchange_id");
    bool status = !strcmp(op, "status"), cancel = !strcmp(op, "cancel"), resume = !strcmp(op, "resume");
    if (!parsed || control.agentic_return || !interactive_version(wire) ||
        !INTERACTIVE_KEYS(root, "agentic", "stream") ||
        !INTERACTIVE_KEYS(wire, "version", "operation", "request_id", "continuation_id", "exchange_id", "reply") ||
        (!status && !cancel && !resume) || !continuation ||
        (!status && (!id || !id[0] || strlen(id) > 128)) ||
        (resume && (!exchange || !exchange[0])) || interactive_nul_escape(body) ||
        (stream && stream->kind != SCHEMA_JSON_BOOL)) goto invalid;
    char hash[41];
    if (!interactive_hash_body(body, hash)) goto invalid;
    pthread_mutex_lock(&s->tool_mu);
    interactive_sweep_locked(s);
    interactive_entry *e = s->interactive_entries;
    while (e && strcmp(e->id, continuation)) e = e->next;
    if (!e || strcmp(e->context_id, control.agent_context_id) ||
        !interactive_scope_equal(&e->allowed_tools, &control.allowed_tools) ||
        !interactive_scope_equal(&e->allowed_skills, &control.allowed_skills)) {
        pthread_mutex_unlock(&s->tool_mu);
        http_error(fd, s->enable_cors, 409, "unknown interactive continuation or mismatched scope");
        goto answered;
    }
    interactive_receipt *receipt = status ? NULL : interactive_receipt_find(e, id);
    if (resume && !stream) streaming = e->streaming;
    if (receipt && strcmp(receipt->hash, hash)) {
        pthread_mutex_unlock(&s->tool_mu);
        http_error(fd, s->enable_cors, 409, "request_id reused with different payload");
        goto answered;
    }
    if (status || receipt) {
        char *result = receipt && receipt->result ? xstrdup(receipt->result) : interactive_status_body(e);
        pthread_mutex_unlock(&s->tool_mu);
        if (streaming) sse_headers(fd, s->enable_cors);
        interactive_send(fd, s->enable_cors, streaming, result); free(result);
        goto answered;
    }
    if (e->finalizing) {
        pthread_mutex_unlock(&s->tool_mu);
        http_error(fd, s->enable_cors, 409, "interactive operation is finalizing");
        goto answered;
    }
    if (cancel && e->status == INTERACTIVE_RUNNING) {
        interactive_receipt_add(e, id, hash);
        e->status = INTERACTIVE_CANCELLED;
        if (e->running) job_mark_cancelled(e->running);
        char *result = interactive_interrupted_body(e, "cancelled", "cancelled", "interactive turn cancelled");
        e->operation->result = xstrdup(result);
        pthread_mutex_unlock(&s->tool_mu);
        if (streaming) sse_headers(fd, s->enable_cors);
        interactive_send(fd, s->enable_cors, streaming, result); free(result);
        goto answered;
    }
    if (e->status != INTERACTIVE_AWAITING || !e->owns_request ||
        (resume && strcmp(e->pending->exchange_id, exchange))) {
        pthread_mutex_unlock(&s->tool_mu);
        http_error(fd, s->enable_cors, 409, "interactive exchange is unavailable or already consumed");
        goto answered;
    }
    if (resume && stream && streaming != e->streaming) {
        pthread_mutex_unlock(&s->tool_mu);
        http_error(fd, s->enable_cors, 400, "resume must preserve the original streaming mode");
        goto answered;
    }
    size_t receipts = 0;
    for (interactive_receipt *p = e->receipts; p; p = p->next) receipts++;
    /* The configure receipt is separate from the exchange budget. */
    if (!cancel && receipts > INTERACTIVE_MAX_RECORDS) {
        pthread_mutex_unlock(&s->tool_mu);
        http_error(fd, s->enable_cors, 429, "interactive exchange limit exceeded");
        goto answered;
    }
    const schema_json_value *reply = schema_json_object_get(wire, "reply");
    const char *answer = interactive_string(reply, "status");
    bool reject = answer && !strcmp(answer, "reject");
    if (reject && (!INTERACTIVE_KEYS(reply, "status", "code", "message") ||
        !interactive_string(reply, "code") || !interactive_string(reply, "message") ||
        strlen(interactive_string(reply, "code")) > 128 ||
        strlen(interactive_string(reply, "message")) > INTERACTIVE_MAX_VALUES)) {
        pthread_mutex_unlock(&s->tool_mu); goto invalid;
    }
    if (!cancel && !reject) {
        const interactive_behavior *behavior = e->pending->tool->behavior;
        size_t bytes = behavior->reply_bytes(e->pending->behavior_state, reply);
        size_t memory = behavior->reply_memory(e->pending->behavior_state, reply);
        size_t used = 0, memory_used = 0;
        for (interactive_entry *p = s->interactive_entries; p; p = p->next) {
            used += p->snapshot_bytes;
            memory_used += p->behavior_bytes;
        }
        if (used > INTERACTIVE_SNAPSHOT_BUDGET || bytes > INTERACTIVE_SNAPSHOT_BUDGET - used ||
            memory_used > INTERACTIVE_STATE_BUDGET || memory > INTERACTIVE_STATE_BUDGET - memory_used) {
            pthread_mutex_unlock(&s->tool_mu);
            http_error(fd, s->enable_cors, 429, "interactive snapshot memory budget exceeded");
            goto answered;
        }
        if (!interactive_call_prepare_reply(e->pending, reply)) {
            pthread_mutex_unlock(&s->tool_mu); goto invalid;
        }
        e->snapshot_bytes += bytes;
        e->behavior_bytes += memory;
    }
    interactive_receipt *operation = interactive_receipt_add(e, id, hash);
    e->paused.interactive.receipt = operation;
    e->status = INTERACTIVE_RUNNING;
    if (cancel || reject) {
        e->finalizing = true;
        pthread_mutex_unlock(&s->tool_mu);
        interactive_terminal(s, e, cancel ? INTERACTIVE_CANCELLED : INTERACTIVE_REJECTED,
            cancel ? "cancelled" : interactive_string(reply, "code"),
            cancel ? "interactive turn cancelled" : interactive_string(reply, "message"), !cancel);
        pthread_mutex_lock(&s->tool_mu);
        char *result = xstrdup(operation->result);
        pthread_mutex_unlock(&s->tool_mu);
        if (streaming) sse_headers(fd, s->enable_cors);
        interactive_send(fd, s->enable_cors, streaming, result); free(result);
        goto answered;
    }
    *out = e->paused;
    memset(&e->paused, 0, sizeof(e->paused));
    e->owns_request = false;
    interactive_rebind(out);
    out->interactive.resumed = true;
    /* Streaming mode is frozen with the original request; absent stream on a
     * control request inherits it. A switch would replay reasoning/deltas. */
    e->queued = true;
    e->pending->authorized = true;
    pthread_mutex_unlock(&s->tool_mu);
    schema_json_free(root); request_free(&control);
    return 2;
invalid:
    http_error(fd, s->enable_cors, 400, "invalid interactive control payload");
answered:
    schema_json_free(root); request_free(&control);
    return 1;
}

#endif
