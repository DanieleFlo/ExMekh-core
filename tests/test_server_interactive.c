#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "interactive FAIL %d: %s\n", __LINE__, #c); exit(1); \
} } while (0)

static void fixture_with_format(request *r, bool multiple, const char *format_schema) {
    request_init(r, REQ_CHAT, 256);
    r->api = API_RESPONSES;
    r->model_syntax = SERVER_MODEL_SYNTAX_QWEN4;
    r->think_mode = DS4_THINK_NONE;
    r->has_tools = r->agentic_present = true;
    r->agent_context_id = xstrdup("test:root");
    id_list_push_unique(&r->allowed_tools, "patch");
    buf registry = {0};
    buf_printf(&registry,
        "{\"name\":\"patch\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"after\":{\"type\":\"string\"},\"before\":{\"type\":\"string\"},"
        "\"target\":{\"type\":\"string\"},\"format\":%s},"
        "\"required\":[\"target\",\"before\",\"after\",\"format\"],\"additionalProperties\":false}}",
        format_schema);
    tool_schema_orders_add_json(&r->tool_orders, registry.ptr);
    buf_free(&registry);
    buf config = {0};
    buf_printf(&config,
        "{\"version\":1,\"operation\":\"configure\",\"request_id\":\"initial\",\"tools\":[{"
        "\"name\":\"patch\",\"behavior\":\"edit_file.v1\","
        "\"fields\":{\"resource\":\"target\",\"selection\":\"before\",\"replacement\":\"after\"},"
        "\"first_fields\":%s,\"markers\":{\"line\":\"@ex_meckh_line[<N>]\","
        "\"all\":\"@ex_meckh_all\",\"delete\":\"@ex_meckh_delete\"}}]}",
        multiple ? "[\"target\",\"format\"]" : "[\"target\"]");
    r->interactive.wire = schema_json_parse(config.ptr);
    buf raw = {0};
    buf_puts(&raw, "{\"agentic\":{\"agent_context_id\":\"test:root\",\"allowed_tools\":[\"patch\"],\"allowed_skills\":[],\"interactive\":");
    buf_puts(&raw, config.ptr); buf_puts(&raw, "}}");
    r->raw_body = buf_take(&raw);
    buf_free(&config);
    char err[160] = {0};
    CHECK(interactive_compile(r, err, sizeof(err)));
}

static void fixture(request *r, bool multiple) {
    fixture_with_format(r, multiple, "{\"type\":\"boolean\"}");
}

static bool candidate(request *r, const char *raw, const char *piece) {
    dsml_decode_tracker dt;
    dsml_decode_tracker_init(&dt, r);
    dsml_decode_tracker_update(&dt, raw, strlen(raw));
    agentic_token_filter filter = {.req = r, .tracker = &dt, .raw = raw,
        .raw_len = strlen(raw), .incremental_free_string = true, .use_incremental_string = true};
    bool valid = agentic_filter_token(&filter, 0, piece, strlen(piece));
    buf_free(&filter.scratch);
    return valid;
}

static interactive_call *pause_call(request *r, const char *raw) {
    dsml_decode_tracker dt;
    dsml_decode_tracker_init(&dt, r);
    dsml_decode_tracker_update(&dt, raw, strlen(raw));
    return interactive_observe(r, &dt, raw, strlen(raw));
}

static void provide_snapshot(interactive_call *call, const char *text, bool create) {
    buf body = {0};
    if (create) {
        buf_puts(&body, "{\"status\":\"create\"}");
    } else {
        size_t lines = 0;
        for (size_t i = 0; text[i]; i++) {
            if (i == 0 || text[i-1] == '\n' ||
                (text[i-1] == '\r' && text[i] != '\n')) lines++;
        }
        buf_puts(&body, "{\"status\":\"provide\",\"encoding\":\"utf-8\",\"text\":");
        json_escape(&body, text);
        buf_printf(&body, ",\"line_count\":%zu}", lines);
    }
    schema_json_value *reply = schema_json_parse(body.ptr);
    CHECK(interactive_call_prepare_reply(call, reply));
    call->authorized = true;
    schema_json_free(reply);
    buf_free(&body);
}

static const char *const path_prefix = "<tool_call><function=patch><parameter=target>src.py";
static const char *const paused_prefix = "<tool_call><function=patch><parameter=target>src.py</parameter>";

/* Independent brute-force oracle includes internal/overlapping occurrences. */
static unsigned occurrences(const char *text, size_t n, const char *value, size_t len) {
    unsigned result = 0;
    for (size_t i = 0; len && i + len <= n; i++)
        if (!memcmp(text + i, value, len)) result++;
    return result;
}

static void test_text_oracle(void) {
    uint32_t seed = 17;
    for (int test = 0; test < 400; test++) {
        char text[130]; size_t n = (size_t)(test % 128);
        const char alphabet[] = "ab \r\n";
        for (size_t i = 0; i < n; i++) {
            seed = seed * 1664525 + 1013904223;
            text[i] = alphabet[(seed >> 16) % 5];
        }
        text[n] = 0;
        edit_constraint c; edit_constraint_init(&c, text, false, "</parameter>");
        for (size_t start = 0; start < n; start++) {
            bool line = start == 0 || text[start-1] == '\n' ||
                (text[start-1] == '\r' && text[start] != '\n');
            if (!line) continue;
            for (size_t len = 1; len <= n - start; len++) {
                bool unique = occurrences(text, n, text + start, len) == 1;
                CHECK(edit_selection_prefix(&c, text + start, len, true, NULL) == unique);
                bool viable = false;
                for (size_t j = 0; j + len <= n; j++) {
                    bool candidate_line = j == 0 || text[j-1] == '\n' ||
                        (text[j-1] == '\r' && text[j] != '\n');
                    if (!candidate_line || memcmp(text + j, text + start, len)) continue;
                    for (size_t end = len; end <= n - j; end++)
                        if (occurrences(text, n, text + j, end) == 1) viable = true;
                }
                CHECK(edit_selection_prefix(&c, text + start, len, false, NULL) == viable);
            }
        }
        edit_constraint_free(&c);
    }
}

static void test_dsml_forms(void) {
    request r; fixture(&r, false); r.model_syntax = SERVER_MODEL_SYNTAX_QWEN;
    const char *prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"patch\">"
        "<｜DSML｜parameter name=\"target\" string=\"true\">src.py</｜DSML｜parameter>";
    CHECK(!candidate(&r, "<｜DSML｜tool_calls><｜DSML｜invoke name=\"patch\">",
        "<｜DSML｜parameter name=\"before\" string=\"false\">\"anything\""));
    interactive_call *call = pause_call(&r, prefix);
    CHECK(call); call->authorized = true;
    provide_snapshot(call, "a\na", false);
    CHECK(!candidate(&r, prefix, "<｜DSML｜parameter name=\"before\" string=\"false\">"));
    CHECK(!candidate(&r, prefix, "<｜DSML｜parameter name=\"before\" string=\"f"));
    CHECK(candidate(&r, prefix, "<｜DSML｜parameter name=\"before\" string=\"true\">"));
    buf full = {0}; buf_puts(&full, prefix);
    buf_puts(&full, "<｜DSML｜parameter name=\"before\" string=\"true\">a\n");
    CHECK(candidate(&r, full.ptr, "</｜DSML｜parameter>"));
    CHECK(!candidate(&r, full.ptr, "</｜DSML｜parameter><｜DSML｜parameter name=\"after\" string=\"true\">"));
    buf_free(&full); request_free(&r);
}

static void test_selections(void) {
    edit_constraint c;
    edit_constraint_init(&c, "foo\nprefix foo\nbar", false, "</parameter>");
    CHECK(c.lines == 3);
    CHECK(edit_selection_prefix(&c, "foo", 3, false, NULL));
    CHECK(!edit_selection_prefix(&c, "foo", 3, true, NULL));
    CHECK(!edit_selection_prefix(&c, "foo\n", 4, true, NULL));
    CHECK(edit_selection_prefix(&c, "foo\np", 5, true, NULL));
    CHECK(edit_selection_prefix(&c, "foo\nprefix foo\nbar", 18, false, NULL));
    CHECK(!edit_selection_prefix(&c, "refix", 5, false, NULL));
    CHECK(edit_selection_prefix(&c, "@ex_meckh_all", strlen("@ex_meckh_all"), true, NULL));
    CHECK(edit_selection_prefix(&c, "@ex_meckh_line[4]", strlen("@ex_meckh_line[4]"), true, NULL));
    CHECK(!edit_selection_prefix(&c, "@ex_meckh_line[5]", strlen("@ex_meckh_line[5]"), true, NULL));
    CHECK(!edit_selection_prefix(&c, "@ex_meckh_line[01]", strlen("@ex_meckh_line[01]"), true, NULL));
    CHECK(!edit_selection_prefix(&c, "@ex_meckh_delete", strlen("@ex_meckh_delete"), true, NULL));
    edit_constraint_free(&c);

    edit_constraint_init(&c, "xabc\nabc", false, "</parameter>");
    CHECK(!edit_selection_prefix(&c, "a", 1, false, NULL));
    CHECK(edit_selection_prefix(&c, "x", 1, false, NULL));
    edit_constraint_free(&c);
    edit_constraint_init(&c, "aaa\n  x\r\ny\r", false, "</parameter>");
    CHECK(c.lines == 3);
    CHECK(!edit_selection_prefix(&c, "a", 1, true, NULL));
    CHECK(!edit_selection_prefix(&c, "x", 1, false, NULL));
    CHECK(edit_selection_prefix(&c, "  x", 3, true, NULL));
    edit_constraint_free(&c);

    edit_constraint_init(&c, "", true, "</parameter>");
    CHECK(c.lines == 0);
    CHECK(edit_selection_prefix(&c, "@ex_meckh_line[1]", strlen("@ex_meckh_line[1]"), true, NULL));
    CHECK(!edit_selection_prefix(&c, "", 0, true, NULL));
    CHECK(!edit_replacement_prefix(&c, true, "@ex_meckh_delete", strlen("@ex_meckh_delete"), true));
    CHECK(edit_replacement_prefix(&c, true, "", 0, true));
    edit_constraint_free(&c);

    edit_constraint_init(&c, "é</parameter>\n", false, "</parameter>");
    CHECK(edit_selection_prefix(&c, "\xc3", 1, false, NULL));
    CHECK(!edit_selection_prefix(&c, "\xc3", 1, true, NULL));
    CHECK(edit_selection_prefix(&c, "é&lt;/parameter>", strlen("é&lt;/parameter>"), true, NULL));
    edit_constraint_free(&c);
}

static void test_grammar(void) {
    request r;
    fixture(&r, false);
    r.tool_choice_required = true;
    CHECK(candidate(&r, "Reasoning about tools.\n</think>", "<tool_call>"));
    r.tool_choice_required = false;
    CHECK(!candidate(&r, "<tool_call><function=patch>", "<parameter=before>"));
    CHECK(candidate(&r, path_prefix, "</parameter>"));
    CHECK(!candidate(&r, path_prefix, "</parameter><parameter=before>"));
    interactive_call *call = pause_call(&r, paused_prefix);
    CHECK(call && strstr(call->values, "src.py"));
    CHECK(!candidate(&r, paused_prefix, "<parameter=before>"));
    call->authorized = true;
    provide_snapshot(call, "foo\nprefix foo\nbar", false);
    CHECK(candidate(&r, paused_prefix, "<parameter=before>"));
    CHECK(!candidate(&r, paused_prefix, "<parameter=after>"));
    buf prefix = {0};
    buf_puts(&prefix, paused_prefix); buf_puts(&prefix, "<parameter=before>");
    CHECK(!candidate(&r, prefix.ptr, "refix"));
    CHECK(candidate(&r, prefix.ptr, "foo"));
    buf_puts(&prefix, "foo");
    dsml_decode_tracker sampling;
    dsml_decode_tracker_init(&sampling, &r);
    dsml_decode_tracker_update(&sampling, prefix.ptr, prefix.len);
    CHECK(!interactive_sampling_body(&r, &sampling, prefix.ptr, prefix.len));
    CHECK(interactive_temperature(&r, &sampling, 0.7f) == 0.0f);
    CHECK(!candidate(&r, prefix.ptr, "</parameter>"));
    CHECK(candidate(&r, prefix.ptr, "\np</parameter>"));
    CHECK(!candidate(&r, prefix.ptr, "\np</parameter><parameter=after>bad"));
    buf_puts(&prefix, "\np</parameter>");
    CHECK(candidate(&r, prefix.ptr, "<parameter=after>"));
    CHECK(!candidate(&r, prefix.ptr, "<parameter=after>new text"));
    buf_puts(&prefix, "<parameter=after>");
    dsml_decode_tracker_init(&sampling, &r);
    dsml_decode_tracker_update(&sampling, prefix.ptr, prefix.len);
    CHECK(interactive_temperature(&r, &sampling, 0.7f) == 0.7f);
    buf closing = {0}; buf_puts(&closing, prefix.ptr); buf_puts(&closing, "replacement</par");
    dsml_decode_tracker_update(&sampling, closing.ptr, closing.len);
    CHECK(interactive_temperature(&r, &sampling, 0.7f) == 0.0f);
    buf_free(&closing);
    CHECK(candidate(&r, prefix.ptr, "\nreplacement\n</parameter>"));
    CHECK(!candidate(&r, prefix.ptr, "@ex_meckh_delete</parameter>"));
    buf_puts(&prefix, "\nreplacement\n</parameter><parameter=format>true</parameter></function></tool_call>");
    tool_calls calls = {0}; char *content = NULL, *reasoning = NULL;
    CHECK(parse_generated_message_ex_for_syntax(r.model_syntax, prefix.ptr, false, &content, &reasoning, &calls));
    CHECK(interactive_apply_call_ids(&r, prefix.ptr, &calls));
    CHECK(calls.len == 1 && !strcmp(calls.v[0].id, call->call_id));
    schema_json_value *args = schema_json_parse(calls.v[0].arguments);
    CHECK(!strcmp(interactive_string(args, "before"), "foo\np"));
    CHECK(!strcmp(interactive_string(args, "after"), "\nreplacement\n"));
    schema_json_free(args); tool_calls_free(&calls); free(content); free(reasoning);
    buf_free(&prefix); request_free(&r);

    fixture(&r, true);
    CHECK(candidate(&r, path_prefix, "</parameter><parameter=format>true</parameter>"));
    CHECK(pause_call(&r, paused_prefix) == NULL);
    CHECK(!candidate(&r, paused_prefix, "<parameter=before>"));
    buf multi = {0}; buf_puts(&multi, paused_prefix);
    buf_puts(&multi, "<parameter=format>true</parameter>");
    call = pause_call(&r, multi.ptr);
    CHECK(call && strstr(call->values, "true"));
    call->authorized = true;
    provide_snapshot(call, "", true);
    CHECK(candidate(&r, multi.ptr, "<parameter=before>@ex_meckh_all</parameter>"));
    CHECK(!candidate(&r, multi.ptr, "<parameter=before>x"));
    buf_free(&multi); request_free(&r);
}

static void test_frozen_object_and_reply_ownership(void) {
    request r;
    fixture_with_format(&r, true,
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\"},"
        "\"b\":{\"type\":\"number\"}},\"required\":[\"a\",\"b\"],\"additionalProperties\":false}");
    buf raw = {0};
    buf_puts(&raw, paused_prefix);
    buf_puts(&raw, "<parameter=format>{\"a\":1,\"b\":2}</parameter>");
    interactive_call *call = pause_call(&r, raw.ptr);
    CHECK(call);
    provide_snapshot(call, "unique", false);
    void *saved = call->behavior_state;
    schema_json_value *bad = schema_json_parse(
        "{\"status\":\"provide\",\"text\":\"wrong\",\"encoding\":\"utf-8\",\"line_count\":0}");
    CHECK(!interactive_call_prepare_reply(call, bad));
    CHECK(call->behavior_state == saved);
    schema_json_free(bad);
    CHECK(candidate(&r, raw.ptr, "<parameter=before>unique</parameter>"));

    free(call->values);
    call->values = xstrdup("{\"format\":{\"b\":2,\"a\":1},\"target\":\"src.py\"}");
    buf_puts(&raw, "<parameter=before>unique</parameter><parameter=after>x</parameter></function></tool_call>");
    tool_calls calls = {0};
    char *content = NULL, *reasoning = NULL;
    CHECK(interactive_parse_message(&r, raw.ptr, &content, &reasoning, &calls));
    CHECK(interactive_apply_call_ids(&r, raw.ptr, &calls));
    free(call->values);
    call->values = xstrdup("{\"format\":{\"b\":2,\"a\":9},\"target\":\"src.py\"}");
    CHECK(!interactive_apply_call_ids(&r, raw.ptr, &calls));
    tool_calls_free(&calls);
    free(content);
    free(reasoning);
    buf_free(&raw);
    request_free(&r);
}

static char *control_body(const interactive_entry *e, const char *id,
                          const char *operation, const char *reply) {
    buf b = {0};
    buf_printf(&b, "{\"agentic\":{\"agent_context_id\":\"test:root\",\"allowed_tools\":[\"patch\"],"
        "\"allowed_skills\":[],\"interactive\":{\"version\":1,\"operation\":\"%s\","
        "\"request_id\":\"%s\",\"continuation_id\":\"%s\",\"exchange_id\":\"%s\"",
        operation, id, e->id, e->pending ? e->pending->exchange_id : "old");
    if (reply) { buf_puts(&b, ",\"reply\":"); buf_puts(&b, reply); }
    buf_puts(&b, "}}}");
    return buf_take(&b);
}

static void test_runtime(void) {
    server s = {0}; pthread_mutex_init(&s.tool_mu, NULL);
    request r; fixture(&r, false);
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    CHECK(!interactive_admit(&s, sv[0], &r));
    interactive_entry *e = r.interactive.entry;
    interactive_call *call = pause_call(&r, paused_prefix);
    CHECK(call);
    e->paused = r; memset(&r, 0, sizeof(r)); interactive_rebind(&e->paused);
    e->owns_request = true; e->pending = call;
    e->queued = false;
    e->text = xstrdup(paused_prefix); e->status = INTERACTIVE_AWAITING;
    e->generated = 23; e->rng = 77;
    char *bad = control_body(e, "step", "resume", "{\"status\":\"provide\",\"text\":\"a\",\"encoding\":\"utf-8\",\"line_count\":0}");
    CHECK(interactive_control(&s, sv[0], bad, &r) == 1);
    CHECK(e->status == INTERACTIVE_AWAITING && !call->authorized);
    free(bad);
    close(sv[0]); close(sv[1]); CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    char *body = control_body(e, "step", "resume", "{\"status\":\"provide\",\"text\":\"a\",\"encoding\":\"utf-8\",\"line_count\":1}");
    CHECK(interactive_control(&s, sv[0], body, &r) == 2);
    CHECK(r.interactive.resumed && call->authorized && !e->owns_request);
    CHECK(!strcmp(call->tool->name, "patch"));
    CHECK(e->generated == 23 && e->rng == 77);
    request unused = {0};
    CHECK(interactive_control(&s, sv[0], body, &unused) == 1);
    CHECK(unused.prompt.len == 0);
    buf escaped = {0};
    const char *key = strstr(body, "\"interactive\""); CHECK(key);
    buf_append(&escaped, body, (size_t)(key - body));
    buf_puts(&escaped, "\"\\u0069nteractive\"");
    buf_puts(&escaped, key + strlen("\"interactive\""));
    CHECK(interactive_control(&s, sv[0], escaped.ptr, &unused) == 1);
    buf_free(&escaped);
    char *changed = control_body(e, "step", "resume", "{\"status\":\"create\"}");
    CHECK(interactive_control(&s, sv[0], changed, &unused) == 1);
    free(changed); free(body);
    interactive_record_response(&r, "{\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}");
    e->status = INTERACTIVE_COMPLETED;
    e->expires = time(NULL) + 30;
    request_free(&r);
    interactive_release_payload(e);
    char *status = control_body(e, "status", "status", NULL);
    CHECK(interactive_control(&s, sv[0], status, &unused) == 1);
    shutdown(sv[0], SHUT_WR);
    char *output = read_socket_text(sv[1]);
    CHECK(strstr(output, "request_id reused with different payload"));
    CHECK(strstr(output, "response.completed"));
    free(output); free(status);
    close(sv[0]); close(sv[1]);
    interactive_entry_free(e); s.interactive_entries = NULL;
    pthread_mutex_destroy(&s.tool_mu);
    char a[41], b[41];
    CHECK(interactive_hash_body("{\"x\":1,\"y\":2}", a));
    CHECK(interactive_hash_body("{\"y\":2,\"x\":1}", b) && !strcmp(a, b));
    CHECK(!interactive_hash_body("{\"x\":1,\"x\":2}", a));
    CHECK(!interactive_hash_body("{\"x\":\"\\u0000\"}", a));
    CHECK(interactive_hash_body("{\"x\":\"\\\\u0000\"}", a));
}

static void test_reasoning_surface(void) {
    request r; fixture(&r, false); r.think_mode = DS4_THINK_HIGH;
    dsml_decode_tracker dt; dsml_decode_tracker_init(&dt, &r);
    thinking_state thinking = {.inside = true};
    buf raw = {0};
    buf_puts(&raw, "Quoted protocol: <tool_call><function=patch><parameter=target>bad</parameter>");
    generation_protocol_feed(&r, &thinking, &dt, raw.ptr, raw.len, raw.ptr, raw.len);
    CHECK(!interactive_observe(&r, &dt, raw.ptr, raw.len));
    buf_puts(&raw, "</think>");
    generation_protocol_feed(&r, &thinking, &dt, raw.ptr, raw.len, "</think>", 8);
    buf_puts(&raw, paused_prefix);
    generation_protocol_feed(&r, &thinking, &dt, raw.ptr, raw.len, paused_prefix, strlen(paused_prefix));
    interactive_call *call = interactive_observe(&r, &dt, raw.ptr, raw.len);
    CHECK(call && strstr(call->values, "src.py") && !strstr(call->values, "bad"));
    provide_snapshot(call, "", true);
    buf_puts(&raw, "<parameter=before>@ex_meckh_all</parameter><parameter=after>x</parameter>"
        "<parameter=format>true</parameter></function></tool_call>");
    tool_calls calls = {0}; char *content = NULL, *reasoning = NULL;
    CHECK(interactive_parse_message(&r, raw.ptr, &content, &reasoning, &calls));
    CHECK(interactive_apply_call_ids(&r, raw.ptr, &calls));
    CHECK(calls.len == 1 && !strcmp(calls.v[0].id, call->call_id));
    tool_calls_free(&calls); free(content); free(reasoning); buf_free(&raw); request_free(&r);
}

static void test_batch_reject_and_expiry(void) {
    server s = {.disable_exact_dsml_tool_replay = true};
    pthread_mutex_init(&s.tool_mu, NULL);
    request r; fixture(&r, false);
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    CHECK(!interactive_admit(&s, sv[0], &r));
    interactive_entry *e = r.interactive.entry;
    interactive_call *first = pause_call(&r, paused_prefix);
    CHECK(first); first->authorized = true;
    char first_id[40]; snprintf(first_id, sizeof(first_id), "%s", first->call_id);
    provide_snapshot(first, "", true);
    buf text = {0}; buf_puts(&text, paused_prefix);
    buf_puts(&text, "<parameter=before>@ex_meckh_all</parameter>"
        "<parameter=after>quoted <function=patch> literal</parameter>"
        "<parameter=format>true</parameter></function></tool_call>");
    buf_puts(&text, paused_prefix);
    interactive_call *second = pause_call(&r, text.ptr);
    CHECK(second && second != first);
    e->paused = r; memset(&r, 0, sizeof(r)); interactive_rebind(&e->paused);
    e->pending = second; e->owns_request = true; e->queued = false;
    e->text = buf_take(&text); e->status = INTERACTIVE_AWAITING;
    char *body = control_body(e, "reject", "resume",
        "{\"status\":\"reject\",\"code\":\"unread\",\"message\":\"read first\"}");
    CHECK(interactive_control(&s, sv[0], body, &r) == 1);
    CHECK(e->status == INTERACTIVE_REJECTED && !e->owns_request);
    schema_json_value *result = schema_json_parse(e->operation->result);
    const schema_json_value *output = schema_json_object_get(result, "output");
    CHECK(output && output->len == 1);
    CHECK(!strcmp(interactive_string(output->item[0], "call_id"), first_id));
    schema_json_free(result); free(body);
    interactive_entry_free(e); s.interactive_entries = NULL;
    close(sv[0]); close(sv[1]); CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    fixture(&r, false); CHECK(!interactive_admit(&s, sv[0], &r));
    e = r.interactive.entry;
    e->pending = pause_call(&r, paused_prefix); CHECK(e->pending);
    e->paused = r; memset(&r, 0, sizeof(r)); interactive_rebind(&e->paused);
    e->owns_request = true; e->queued = false; e->status = INTERACTIVE_AWAITING;
    e->expires = time(NULL) - 1;
    interactive_sweep_locked(&s);
    CHECK(e->status == INTERACTIVE_EXPIRED && !e->owns_request);
    CHECK(strstr(e->operation->result, "interactive.expired"));
    e->expires = time(NULL) - 1; interactive_sweep_locked(&s);
    CHECK(!s.interactive_entries);
    close(sv[0]); close(sv[1]); pthread_mutex_destroy(&s.tool_mu);
}

typedef struct {
    server *server;
    const char *body;
    int fd, result;
    request request;
} resume_probe;

static void *resume_probe_run(void *argument) {
    resume_probe *probe = argument;
    probe->result = interactive_control(probe->server, probe->fd, probe->body, &probe->request);
    return NULL;
}

static void test_concurrent_resume_and_queued_cancel(void) {
    server s = {0}; pthread_mutex_init(&s.tool_mu, NULL);
    request r; fixture(&r, false); r.stream = true;
    int sockets[2][2];
    for (int i = 0; i < 2; i++) CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets[i]) == 0);
    CHECK(!interactive_admit(&s, sockets[0][0], &r));
    interactive_entry *e = r.interactive.entry;
    e->pending = pause_call(&r, paused_prefix); CHECK(e->pending);
    e->paused = r; memset(&r, 0, sizeof(r)); interactive_rebind(&e->paused);
    e->queued = false; e->owns_request = true; e->status = INTERACTIVE_AWAITING;
    char *body = control_body(e, "resume", "resume", "{\"status\":\"create\"}");
    resume_probe probe[2] = {{.server=&s, .body=body, .fd=sockets[0][0]},
                             {.server=&s, .body=body, .fd=sockets[1][0]}};
    pthread_t thread[2];
    for (int i = 0; i < 2; i++) CHECK(pthread_create(&thread[i], NULL, resume_probe_run, &probe[i]) == 0);
    for (int i = 0; i < 2; i++) CHECK(pthread_join(thread[i], NULL) == 0);
    CHECK((probe[0].result == 2) + (probe[1].result == 2) == 1);
    request *claimed = probe[0].result == 2 ? &probe[0].request : &probe[1].request;
    CHECK(claimed->stream && e->queued && e->status == INTERACTIVE_RUNNING);
    char *cancel = control_body(e, "cancel", "cancel", NULL);
    CHECK(interactive_control(&s, sockets[0][0], cancel, &r) == 1);
    CHECK(e->status == INTERACTIVE_CANCELLED);
    CHECK(strstr(e->operation->result, "interactive.cancelled"));
    CHECK(!claimed->interactive.receipt->result); /* Cancel has a distinct receipt. */
    e->expires = time(NULL) - 1; interactive_sweep_locked(&s);
    CHECK(s.interactive_entries == e); /* An acquired queued request owns e. */
    e->queued = false;
    request_free(claimed); interactive_release_payload(e);
    interactive_sweep_locked(&s); CHECK(!s.interactive_entries);
    free(body); free(cancel);
    for (int i = 0; i < 2; i++) { close(sockets[i][0]); close(sockets[i][1]); }
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_large_snapshot(void) {
    char *text = xmalloc(INTERACTIVE_MAX_SNAPSHOT + 1);
    size_t length = 0;
    for (int line = 0; length + 80 < INTERACTIVE_MAX_SNAPSHOT; line++)
        length += (size_t)snprintf(text + length, 80, "repeated repeated repeated repeated repeated line %d\r\n", line);
    text[length] = 0;
    double start = now_sec();
    edit_constraint c; edit_constraint_init(&c, text, false, "</parameter>");
    double index_ms = (now_sec() - start) * 1000;
    const char *selection = "repeated repeated repeated repeated repeated line 999\r\n";
    CHECK(edit_selection_prefix(&c, selection, strlen(selection), true, NULL));
    start = now_sec();
    for (int i = 0; i < 1000; i++)
        CHECK(edit_selection_prefix(&c, selection, strlen(selection), true, NULL));
    double lookup_us = (now_sec() - start) * 1000;
    size_t line_capacity = 32;
    while (line_capacity < c.lines) line_capacity *= 2;
    size_t index_bytes = c.length + 1 + c.wire.cap + sizeof(c) +
        (c.length + 1 + line_capacity + 3 * c.lines) * sizeof(size_t);
    fprintf(stdout, "interactive snapshot benchmark bytes=%zu lines=%zu index_ms=%.3f lookup_us=%.3f index_bytes=%zu build_peak_bytes=%zu\n",
        length, c.lines, index_ms, lookup_us, index_bytes,
        index_bytes + (5 * c.length + 257) * sizeof(int));
    edit_constraint_free(&c); free(text);
}

static void test_disconnected_queued_request(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL); pthread_mutex_init(&s.mu, NULL);
    pthread_mutex_init(&s.model_mu, NULL); pthread_cond_init(&s.cv, NULL); pthread_cond_init(&s.model_cv, NULL);
    job j = {0}; fixture(&j.req, false);
    pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
    int sockets[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    CHECK(!interactive_admit(&s, sockets[0], &j.req));
    interactive_entry *e = j.req.interactive.entry;
    s.head = s.tail = &j;
    server_cancel_job(&s, &j);
    CHECK(j.done && !s.head && !e->queued && e->status == INTERACTIVE_CANCELLED);
    CHECK(strstr(j.req.interactive.receipt->result, "interactive.cancelled"));
    request_free(&j.req); interactive_entry_free(e);
    close(sockets[0]); close(sockets[1]);
    pthread_mutex_destroy(&j.mu); pthread_cond_destroy(&j.cv);
    pthread_mutex_destroy(&s.tool_mu); pthread_mutex_destroy(&s.mu); pthread_mutex_destroy(&s.model_mu);
    pthread_cond_destroy(&s.cv); pthread_cond_destroy(&s.model_cv);
}

static void test_configuration_and_replay_contract(void) {
    request r; fixture(&r, false);
    char error[160] = {0};
    r.model_syntax = SERVER_MODEL_SYNTAX_GLM;
    CHECK(!interactive_compile(&r, error, sizeof(error)));
    request_free(&r);
    CHECK(!interactive_request_options("{\"stop\":[\"x\"]}"));
    CHECK(!interactive_request_options("{\"ignore_eos\":false}"));
    CHECK(interactive_request_options("{\"temperature\":0.7}"));
    const char *agent = "{\"agent_context_id\":\"ctx\",\"allowed_tools\":[],\"allowed_skills\":[],"
        "\"operation\":\"return\",\"skill_call_id\":\"parent\",\"interactive\":{\"version\":1}}";
    memset(&r, 0, sizeof(r));
    CHECK(parse_agentic_value(&agent, &r, error, sizeof(error)));
    CHECK(r.agentic_return && r.interactive.wire);
    request_free(&r);
    const char *event = "{\"version\":1,\"type\":\"interactive.rejected\",\"call_id\":\"call-1\","
        "\"tool_name\":\"patch\",\"values\":{\"target\":\"<tool_call> & x\"},"
        "\"code\":\"unread\",\"message\":\"read first\"}";
    char *replay = interactive_replay_text(event, strlen(event));
    CHECK(replay && strstr(replay, "&lt;tool_call&gt;") && strstr(replay, "&amp;"));
    CHECK(!strstr(replay, "<tool_call>") && !strstr(replay, "function_call_output"));
    free(replay);
    event = "{\"version\":1,\"type\":\"interactive.failed\",\"code\":\"generation_aborted\",\"message\":\"failed early\"}";
    replay = interactive_replay_text(event, strlen(event)); CHECK(replay); free(replay);
}

int main(void) {
    test_selections(); test_text_oracle(); test_grammar(); test_dsml_forms(); test_runtime();
    test_batch_reject_and_expiry();
    test_reasoning_surface();
    test_concurrent_resume_and_queued_cancel(); test_large_snapshot();
    test_disconnected_queued_request();
    test_configuration_and_replay_contract();
    test_frozen_object_and_reply_ownership();
    puts("interactive server tests: ok");
    return 0;
}
