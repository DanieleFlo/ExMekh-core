#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "interactive FAIL %d: %s\n", __LINE__, #c); exit(1); \
} } while (0)

static void fixture_schema(request *r, const char *schema) {
    request_init(r, REQ_CHAT, 512);
    r->api = API_RESPONSES;
    r->model_syntax = SERVER_MODEL_SYNTAX_QWEN4;
    r->think_mode = DS4_THINK_NONE;
    r->has_tools = r->agentic_present = true;
    r->agent_context_id = xstrdup("test:root");
    id_list_push_unique(&r->allowed_tools, "patch");
    buf registry = {0};
    buf_printf(&registry,
        "{\"name\":\"patch\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"payload\":%s},\"required\":[\"payload\"],"
        "\"additionalProperties\":false}}", schema);
    tool_schema_orders_add_json(&r->tool_orders, registry.ptr);
    buf_free(&registry);
    const char *configuration =
        "{\"version\":2,\"operation\":\"configure\",\"request_id\":\"initial\",\"tools\":[{"
        "\"name\":\"patch\",\"behavior\":\"edit_file.v3\",\"fields\":{\"document\":\"payload\"}}]}";
    r->interactive.wire = schema_json_parse(configuration);
    buf raw = {0};
    buf_puts(&raw, "{\"agentic\":{\"agent_context_id\":\"test:root\",\"allowed_tools\":[\"patch\"],"
                    "\"allowed_skills\":[],\"interactive\":");
    buf_puts(&raw, configuration); buf_puts(&raw, "}}");
    r->raw_body = buf_take(&raw);
}

static void fixture(request *r) {
    fixture_schema(r, "{\"type\":\"string\"}");
    char err[256] = {0};
    CHECK(interactive_compile(r, err, sizeof(err)));
}

static bool candidate(request *r, const char *raw, const char *piece) {
    dsml_decode_tracker dt;
    dsml_decode_tracker_init(&dt, r);
    dsml_decode_tracker_update(&dt, raw, strlen(raw));
    agentic_token_filter filter = {.req=r, .tracker=&dt, .raw=raw, .raw_len=strlen(raw)};
    bool oracle = agentic_filter_token(&filter, 0, piece, strlen(piece));
    filter.use_incremental_string = true;
    filter.incremental_free_string = agentic_tracker_has_unbounded_string(&dt);
    bool valid = agentic_filter_token(&filter, 0, piece, strlen(piece));
    CHECK(valid == oracle);
    buf_free(&filter.scratch);
    return valid;
}

static interactive_call *pause_call(request *r, const char *raw) {
    dsml_decode_tracker dt;
    dsml_decode_tracker_init(&dt, r);
    dsml_decode_tracker_update(&dt, raw, strlen(raw));
    return interactive_observe(r, &dt, raw, strlen(raw));
}

static const char *const parameter_prefix = "<tool_call><function=patch><parameter=payload>";
static const char *const paused_prefix = "<tool_call><function=patch><parameter=payload><exmekh_edit path=\"src.py\"";

static void provide_snapshot(interactive_call *call, const char *text, bool create) {
    buf body = {0};
    if (create) buf_puts(&body, "{\"status\":\"create\"}");
    else {
        size_t lines = 0;
        for (size_t i = 0; text[i]; i++)
            if (!i || text[i-1] == '\n' || (text[i-1] == '\r' && text[i] != '\n')) lines++;
        buf_puts(&body, "{\"status\":\"provide\",\"encoding\":\"utf-8\",\"text\":");
        json_escape(&body, text);
        buf_printf(&body, ",\"line_count\":%zu}", lines);
    }
    schema_json_value *reply = schema_json_parse(body.ptr);
    CHECK(interactive_call_prepare_reply(call, reply));
    call->authorized = true;
    schema_json_free(reply); buf_free(&body);
}

static void append_checked(request *r, buf *raw, const char *piece) {
    if (!candidate(r, raw->ptr ? raw->ptr : "", piece)) {
        fprintf(stderr, "rejected candidate prefix=%s piece=%s\n", raw->ptr ? raw->ptr : "", piece);
        CHECK(false);
    }
    buf_puts(raw, piece);
}

static void finish_opening(request *r, interactive_call *call, buf *raw) {
    CHECK(call->authorized);
    edit_document *state = call->behavior_state;
    buf opening = {0};
    buf_printf(&opening, " id=\"%s\" />", state->item[call->input_index - 1].identifier);
    append_checked(r, raw, opening.ptr); buf_free(&opening);
}

static void finish_element(request *r, interactive_call *call, buf *raw,
                           const char *old, const char *replacement) {
    edit_document *state = call->behavior_state;
    buf part = {0}; append_tool_body_text(&part, old, call->param_end);
    if (part.len) append_checked(r, raw, part.ptr);
    buf_free(&part);
    buf_printf(&part, "<exmekh_new_text id=\"%s\"/>", state->item[call->input_index - 1].identifier);
    if (part.len) append_checked(r, raw, part.ptr);
    buf_free(&part);
    append_tool_body_text(&part, replacement, call->param_end);
    if (part.len) append_checked(r, raw, part.ptr);
    buf_free(&part);
}

static unsigned occurrences(const char *text, size_t n, const char *value, size_t len) {
    unsigned count = 0;
    for (size_t i = 0; len && i + len <= n; i++)
        if (!memcmp(text+i, value, len)) count++;
    return count;
}

static void test_text_oracle(void) {
    uint32_t seed = 17;
    for (int test = 0; test < 400; test++) {
        char text[130]; size_t n = (size_t)(test % 128);
        const char alphabet[] = "ab \r\n";
        for (size_t i = 0; i < n; i++) {
            seed = seed * 1664525 + 1013904223; text[i] = alphabet[(seed >> 16) % 5];
        }
        text[n] = 0;
        edit_constraint snapshot; edit_constraint_init(&snapshot, text);
        for (size_t start = 0; start < n; start++) {
            bool line = !start || text[start-1] == '\n' || (text[start-1] == '\r' && text[start] != '\n');
            for (size_t length = 1; length <= n-start; length++) {
                bool expected = line && occurrences(text, n, text+start, length) == 1;
                CHECK(edit_selection_prefix(&snapshot, text+start, length, true, NULL) == expected);
            }
        }
        edit_constraint_free(&snapshot);
    }
}

















static void test_response_metadata(void) {
    request r; fixture(&r); free(r.model); r.model=xstrdup("Qwen3.8-27B-UD-Q4_K_S");
    r.constrained_prefill_tokens=7; r.cache_read_tokens=4; r.cache_write_tokens=6;
    int sockets[2]; CHECK(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);
    CHECK(responses_final_response(sockets[0],false,&r,"unused","OK",NULL,NULL,"stop",10,20));
    shutdown(sockets[0],SHUT_WR);
    char *wire=read_socket_text(sockets[1]),*body=strstr(wire,"\r\n\r\n"); CHECK(body);
    body+=4;
    schema_json_value *parsed=schema_json_parse(body); CHECK(parsed);
    CHECK(schema_json_object_get(parsed,"parallel_tool_calls"));
    CHECK(schema_json_object_get(parsed,"tool_choice"));
    CHECK(schema_json_object_get(parsed,"tools"));
    const schema_json_value *usage=schema_json_object_get(parsed,"usage");
    const schema_json_value *input=schema_json_object_get(usage,"input_tokens_details");
    const schema_json_value *output=schema_json_object_get(usage,"output_tokens_details");
    CHECK(schema_json_object_get(input,"cache_write_tokens")->number==6);
    CHECK(schema_json_object_get(output,"constrained_prefill_tokens")->number==7);
    const char *artifact=getenv("DS4_INTERACTIVE_RESPONSE_FIXTURE");
    if (artifact) {
        FILE *file=fopen(artifact,"w"); CHECK(file); CHECK(fputs(body,file)>=0); CHECK(fclose(file)==0);
    }
    schema_json_free(parsed); free(wire); close(sockets[0]); close(sockets[1]); request_free(&r);
}
static char *control_body(const interactive_entry *e, const char *id,
                          const char *operation, const char *reply) {
    buf b = {0};
    buf_printf(&b, "{\"agentic\":{\"agent_context_id\":\"test:root\",\"allowed_tools\":[\"patch\"],"
        "\"allowed_skills\":[],\"interactive\":{\"version\":2,\"operation\":\"%s\","
        "\"request_id\":\"%s\",\"continuation_id\":\"%s\",\"exchange_id\":\"%s\"",
        operation, id, e->id, e->pending ? e->pending->exchange_id : "old");
    if (reply) { buf_puts(&b, ",\"reply\":"); buf_puts(&b, reply); }
    buf_puts(&b, "}}}");
    return buf_take(&b);
}

static void test_runtime(void) {
    server s = {0}; pthread_mutex_init(&s.tool_mu, NULL);
    request r; fixture(&r);
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
    request r; fixture(&r); r.stream = true;
    int sockets[2][2];
    for (int i = 0; i < 2; i++) CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets[i]) == 0);
    CHECK(!interactive_admit(&s, sockets[0][0], &r));
    interactive_entry *e = r.interactive.entry;
    e->pending = pause_call(&r, paused_prefix); CHECK(e->pending);
    e->paused = r; memset(&r, 0, sizeof(r)); interactive_rebind(&e->paused);
    e->queued = false; e->owns_request = true; e->status = INTERACTIVE_AWAITING;
    char *body = control_body(e, "resume", "resume", "{\"status\":\"provide\",\"text\":\"\",\"encoding\":\"utf-8\",\"line_count\":0}");
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


static void test_disconnected_queued_request(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL); pthread_mutex_init(&s.mu, NULL);
    pthread_mutex_init(&s.model_mu, NULL); pthread_cond_init(&s.cv, NULL); pthread_cond_init(&s.model_cv, NULL);
    job j = {0}; fixture(&j.req);
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


static void test_reject_second_element(void) {
    server s={.disable_exact_dsml_tool_replay=true}; pthread_mutex_init(&s.tool_mu,NULL);
    request r; fixture(&r);
    int sv[2]; CHECK(socketpair(AF_UNIX,SOCK_STREAM,0,sv)==0);
    CHECK(!interactive_admit(&s,sv[0],&r));
    interactive_entry *e=r.interactive.entry;
    buf raw={0}; buf_puts(&raw,paused_prefix);
    interactive_call *call=pause_call(&r,raw.ptr); CHECK(call);
    provide_snapshot(call,"",false); finish_opening(&r,call,&raw);
    finish_element(&r,call,&raw,"","first");
    append_checked(&r,&raw,"<exmekh_edit path=\"second.py\"");
    CHECK(pause_call(&r,raw.ptr)==call && call->input_index==2);
    e->paused=r; memset(&r,0,sizeof(r)); interactive_rebind(&e->paused);
    e->pending=call; e->owns_request=true; e->queued=false;
    e->text=buf_take(&raw); e->status=INTERACTIVE_AWAITING;
    char *body=control_body(e,"reject","resume","{\"status\":\"reject\",\"code\":\"unread\",\"message\":\"read second.py\"}");
    CHECK(interactive_control(&s,sv[0],body,&r)==1 && e->status==INTERACTIVE_REJECTED);
    schema_json_value *result=schema_json_parse(e->operation->result);
    const schema_json_value *output=schema_json_object_get(result,"output");
    CHECK(output && !output->len);
    CHECK(strstr(e->operation->result,"\"edit_index\":2") && strstr(e->operation->result,"second.py"));
    char *replay=interactive_replay_text(e->operation->result,strlen(e->operation->result));
    CHECK(replay && strstr(replay,"<edit_index>2</edit_index>")); free(replay);
    schema_json_free(result); free(body); interactive_entry_free(e);
    close(sv[0]); close(sv[1]); pthread_mutex_destroy(&s.tool_mu);
}

static void test_expiry_and_memory_budget(void) {
    server s={0}; pthread_mutex_init(&s.tool_mu,NULL);
    request r; fixture(&r);
    int sv[2]; CHECK(socketpair(AF_UNIX,SOCK_STREAM,0,sv)==0);
    CHECK(!interactive_admit(&s,sv[0],&r));
    interactive_entry *e=r.interactive.entry;
    e->pending=pause_call(&r,paused_prefix); CHECK(e->pending);
    e->paused=r; memset(&r,0,sizeof(r)); interactive_rebind(&e->paused);
    e->owns_request=true; e->queued=false; e->status=INTERACTIVE_AWAITING;
    e->text=xstrdup(paused_prefix);
    e->behavior_bytes=INTERACTIVE_STATE_BUDGET;
    char *body=control_body(e,"over-budget","resume","{\"status\":\"create\"}");
    CHECK(interactive_control(&s,sv[0],body,&r)==1 && e->status==INTERACTIVE_AWAITING);
    CHECK(!e->pending->authorized);
    CHECK(!candidate(&e->paused, paused_prefix, ">"));
    e->expires=time(NULL)-1; interactive_sweep_locked(&s);
    CHECK(e->status==INTERACTIVE_EXPIRED && !e->owns_request);
    CHECK(!e->behavior_bytes && !e->snapshot_bytes);
    CHECK(strstr(e->operation->result,"interactive.expired"));
    e->expires=time(NULL)-1; interactive_sweep_locked(&s);
    CHECK(!s.interactive_entries);
    free(body); close(sv[0]); close(sv[1]); pthread_mutex_destroy(&s.tool_mu);
    const char *historical="{\"version\":1,\"type\":\"interactive.failed\",\"code\":\"failure\",\"message\":\"old\"}";
    CHECK(!interactive_replay_text(historical,strlen(historical)));
    const char *current="{\"version\":2,\"type\":\"interactive.failed\",\"code\":\"failure\",\"message\":\"current\"}";
    char *replay=interactive_replay_text(current,strlen(current)); CHECK(replay); free(replay);
}

static void test_large_snapshot(void) {
    char *text=xmalloc(INTERACTIVE_MAX_SNAPSHOT+1); size_t length=0;
    for (int line=0; length+80<INTERACTIVE_MAX_SNAPSHOT; line++)
        length+=(size_t)snprintf(text+length,80,"repeated repeated repeated repeated repeated line %d\r\n",line);
    text[length]=0;
    double start=now_sec(); edit_constraint snapshot; edit_constraint_init(&snapshot,text);
    double index_ms=(now_sec()-start)*1000;
    const char *selection="repeated repeated repeated repeated repeated line 999\r\n";
    CHECK(edit_selection_prefix(&snapshot,selection,strlen(selection),true,NULL));
    start=now_sec();
    volatile unsigned verified=0;
    for (int i=0;i<1000;i++) verified+=edit_selection_prefix(&snapshot,selection,strlen(selection),true,NULL);
    CHECK(verified==1000);
    printf("interactive snapshot benchmark bytes=%zu lines=%zu index_ms=%.3f lookup_us=%.3f\n",
           length,snapshot.lines,index_ms,(now_sec()-start)*1000);
    edit_constraint_free(&snapshot); free(text);
}







static void test_selection_frontier_equivalence(void) {
    const char *snapshots[] = {"", "a", "same\nsame\nsame-tail", "b\ra\r\nb\na-end",
        "αx\nαy\n🙂x\n🙂y", "repeat 1\nrepeat 2\nrepeat 11\nrepeat 12\nlast"};
    const char *pieces[] = {"", "a", "same", "repeat ", "\r\n", "α", "🙂", "\xf0", "&", "last"};
    for (size_t s = 0; s < sizeof(snapshots)/sizeof(snapshots[0]); s++) {
        edit_constraint snapshot; edit_constraint_init(&snapshot, snapshots[s]);
        for (size_t start = 0; start <= strlen(snapshots[s]); start++) {
            if (start && snapshots[s][start - 1] != '\n' && snapshots[s][start - 1] != '\r') continue;
            const char *line = snapshots[s] + start;
            for (size_t committed = 0; committed <= strlen(line); committed++) {
                edit_selection_frontier frontier = {0}; edit_selection_frontier_init(&frontier, &snapshot);
                edit_selection_frontier_advance(&frontier, line, committed);
                for (size_t p = 0; p < sizeof(pieces)/sizeof(pieces[0]); p++) {
                    buf value = {0}; buf_append(&value, line, committed); buf_puts(&value, pieces[p]);
                    for (int complete = 0; complete < 2; complete++) {
                        size_t expected_count = 0, actual_count = 0;
                        bool expected = edit_selection_prefix(&snapshot, value.ptr, value.len, complete, &expected_count);
                        bool actual = edit_selection_frontier_prefix(&frontier, pieces[p], strlen(pieces[p]), complete, &actual_count);
                        CHECK(expected == actual && expected_count == actual_count);
                    }
                    buf_free(&value);
                }
                edit_selection_frontier_free(&frontier);
            }
        }
        edit_constraint_free(&snapshot);
    }
}









static void test_snapshot_draft_policy(void) {
    const char *names[] = {"DS4_OLD_TEXT_DRAFT_BATCH", "DS4_OLD_TEXT_DRAFT_DISABLE"};
    char *saved[2];
    for (int i = 0; i < 2; i++) {
        saved[i] = getenv(names[i]) ? xstrdup(getenv(names[i])) : NULL;
        unsetenv(names[i]);
    }
    snapshot_draft_policy policy = snapshot_draft_policy_init();
    snapshot_draft_policy_enter(&policy, 1);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 32);
    snapshot_draft_policy_observe(&policy, 32, 32);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 64);
    CHECK(snapshot_draft_policy_capacity(&policy, 35, 100) == 35);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 35) == 35);
    CHECK(snapshot_draft_policy_capacity(&policy, 31, 100) == 0);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 31) == 0);
    snapshot_draft_policy_observe(&policy, 64, 40);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 32);
    snapshot_draft_policy_observe(&policy, 32, 5);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 0);
    snapshot_draft_policy_observe(&policy, 0, 31);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 0);
    snapshot_draft_policy_observe(&policy, 0, 1);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 32);
    snapshot_draft_policy_observe(&policy, 32, 1);
    snapshot_draft_policy_observe(&policy, 0, 64);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 0);
    snapshot_draft_policy_enter(&policy, 2);
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 32);
    setenv(names[0], "64", 1); policy = snapshot_draft_policy_init();
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 64);
    setenv(names[0], "16", 1); policy = snapshot_draft_policy_init();
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 32);
    setenv(names[1], "0", 1); policy = snapshot_draft_policy_init();
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 32);
    setenv(names[1], "1", 1); policy = snapshot_draft_policy_init();
    CHECK(snapshot_draft_policy_capacity(&policy, 64, 100) == 0);
    for (int i = 0; i < 2; i++) {
        if (saved[i]) { setenv(names[i], saved[i], 1); free(saved[i]); }
        else unsetenv(names[i]);
    }
}

static void test_speculative_metric_ownership(void) {
    server_decode_phase_metrics phase = {0};
    server_constraint_step step = {0};
    constrained_speculative_sampler sampler = {.phase=&phase, .profile=true};
    constrained_speculative_profile_row(&sampler, &step, 0, false);
    constrained_speculative_profile_fallback(&sampler);
    CHECK(phase.mtp_constrained_sampled_rows == 1 && phase.mtp_constrained_fallbacks == 1);
    sampler.snapshot_drafts = true;
    constrained_speculative_profile_row(&sampler, &step, 0, false);
    constrained_speculative_profile_fallback(&sampler);
    CHECK(phase.old_text_sampled_rows == 1 && phase.old_text_fallbacks == 1);
    CHECK(phase.mtp_constrained_sampled_rows == 1 && phase.mtp_constrained_fallbacks == 1);
}



static void test_resumed_progress_counters(void) {
    FILE *output = tmpfile(); CHECK(output);
    int saved = dup(STDERR_FILENO); CHECK(saved >= 0);
    CHECK(dup2(fileno(output), STDERR_FILENO) >= 0);
    double start = now_sec() - 1, last = start;
    int completion = 115;
    log_decode_progress(REQ_CHAT, 1000, 115, true, true, false, true, false,
        start, 115, &last, &completion);
    fflush(stderr);
    CHECK(dup2(saved, STDERR_FILENO) >= 0); close(saved);
    rewind(output); char line[1024] = {0}; CHECK(fgets(line, sizeof(line), output)); fclose(output);
    CHECK(strstr(line, "ctx=1115..1115:0 gen=115"));
    CHECK(strstr(line, "chunk=0.00 t/s avg=0.00 t/s"));
    CHECK(completion == 115 && last > start);
}


static void separator_text(interactive_call *call, buf *out) {
    edit_document *state = call->behavior_state;
    buf_printf(out, "<exmekh_new_text id=\"%s\"/>", state->item[call->input_index - 1].identifier);
}

static void test_literal_contract(void) {
    request r; fixture(&r);
    buf raw = {0}; buf_puts(&raw, paused_prefix);
    CHECK(candidate(&r, parameter_prefix, "<exmekh_edit path=\"src.py\""));
    CHECK(!candidate(&r, parameter_prefix, "<edit path=\"src.py\""));
    interactive_call *call = pause_call(&r, raw.ptr); CHECK(call);
    CHECK(!candidate(&r, raw.ptr, " />"));
    provide_snapshot(call, "  old & <x/>\r\nold\nprefix old\n", false);
    edit_document *state = call->behavior_state;
    CHECK(strlen(state->item[0].identifier) == 3);
    finish_opening(&r, call, &raw);
    CHECK(!candidate(&r, raw.ptr, "old<exmekh_new_text/>"));
    CHECK(!candidate(&r, raw.ptr, "<all/>"));
    CHECK(!candidate(&r, raw.ptr, "<line number=\"1\"/>"));
    append_checked(&r, &raw, "  old & <x/>");
    buf separator = {0}; separator_text(call, &separator);
    append_checked(&r, &raw, separator.ptr);

    append_checked(&r, &raw, "<delete/><![CDATA[raw]]>&amp;\r\n ");
    CHECK(candidate(&r, raw.ptr, "</parameter></function></tool_call>"));
    CHECK(!candidate(&r, raw.ptr, separator.ptr));
    append_checked(&r, &raw, "</parameter></function></tool_call>");
    tool_calls calls = {0}; char *content = NULL, *reasoning = NULL;
    CHECK(interactive_parse_message(&r, raw.ptr, &content, &reasoning, &calls));
    CHECK(calls.len == 1);
    schema_json_value *args = schema_json_parse(calls.v[0].arguments);
    CHECK(strstr(interactive_string(args, "payload"), "<delete/><![CDATA[raw]]>&amp;\r\n "));
    schema_json_free(args); tool_calls_free(&calls); free(content); free(reasoning);
    buf_free(&separator); buf_free(&raw); request_free(&r);
}

static void test_empty_and_missing_id(void) {
    for (int empty = 0; empty < 2; empty++) {
        request r; fixture(&r);
        buf raw = {0}; buf_puts(&raw, paused_prefix);
        interactive_call *call = pause_call(&r, raw.ptr); CHECK(call);
        provide_snapshot(call, empty ? "" : "KEEP", false);
        finish_opening(&r, call, &raw);
        if (!empty) append_checked(&r, &raw, "KEEP");
        append_checked(&r, &raw, "<exmekh_new_text/>");
        CHECK(candidate(&r, raw.ptr, "</parameter></function></tool_call>"));
        CHECK(!candidate(&r, raw.ptr, "<exmekh_new_text/>"));
        append_checked(&r, &raw, "FILLED");
        CHECK(candidate(&r, raw.ptr, "</parameter></function></tool_call>"));
        buf_free(&raw); request_free(&r);
    }
    request r; fixture(&r);
    interactive_call *call = pause_call(&r, paused_prefix);
    schema_json_value *create = schema_json_parse("{\"status\":\"create\"}");
    CHECK(!interactive_call_prepare_reply(call, create));
    schema_json_free(create);
    provide_snapshot(call, "repeated\nprefix repeated", false);
    buf raw = {0}; buf_puts(&raw, paused_prefix); finish_opening(&r, call, &raw);
    append_checked(&r, &raw, "repeated");
    CHECK(!candidate(&r, raw.ptr, "<exmekh_new_text/>"));
    CHECK(!candidate(&r, raw.ptr, "</parameter>"));
    buf_free(&raw); request_free(&r);
}

static void test_ids_and_multiple_edits(void) {
    request r; fixture(&r);
    buf raw = {0}; buf_puts(&raw, paused_prefix);
    interactive_call *call = pause_call(&r, raw.ptr); CHECK(call);
    provide_snapshot(call, "ONE\nTWO\n", false);
    edit_document *state = call->behavior_state;
    char first[4]; memcpy(first, state->item[0].identifier, 4);
    finish_opening(&r, call, &raw); finish_element(&r, call, &raw, "ONE", "FIRST\r\n");
    append_checked(&r, &raw, "<exmekh_edit path=\"src.py\"");
    CHECK(pause_call(&r, raw.ptr) == call && call->input_index == 2);
    provide_snapshot(call, "ONE\nTWO\n", false);
    CHECK(strcmp(first, state->item[1].identifier));
    CHECK(state->item[0].snapshot == state->item[1].snapshot);
    finish_opening(&r, call, &raw); finish_element(&r, call, &raw, "TWO", "SECOND");
    CHECK(candidate(&r, raw.ptr, "</parameter></function></tool_call>"));
    buf_free(&raw); request_free(&r);

    edit_document *document = edit_document_new("</parameter>");
    buf snapshot = {0};
    for (int i = 0; i < 999; i++) buf_printf(&snapshot, "id=\"%03d\"\n", i);
    CHECK(edit_document_begin_input(document, "x"));
    CHECK(edit_document_provide(document, snapshot.ptr));
    CHECK(!strcmp(document->item[0].identifier, "999"));
    CHECK(edit_document_begin_input(document, "x"));
    CHECK(!edit_document_provide(document, snapshot.ptr));
    buf_free(&snapshot); edit_document_free(document);
}

static void check_views(edit_document *state, const char *value, size_t n, size_t committed, bool complete) {
    edit_document_checkpoint before = state->checkpoint;
    edit_document_view full = interactive_edit_probe(state, &(interactive_edit_input){
        .text = value, .length = n, .native_end = "</parameter>", .committed = committed,
        .complete = complete, .incremental = false}, NULL);
    edit_document_view fast = interactive_edit_probe(state, &(interactive_edit_input){
        .text = value, .length = n, .native_end = "</parameter>", .committed = committed,
        .complete = complete, .incremental = true}, NULL);
    CHECK(full.valid == fast.valid);
    if (full.valid) {
        CHECK(full.phase == fast.phase && full.index == fast.index);
        CHECK(full.checkpoint.offset == fast.checkpoint.offset);
        CHECK(full.checkpoint.old_body == fast.checkpoint.old_body);
        CHECK(full.checkpoint.new_body == fast.checkpoint.new_body);
    }
    CHECK(!memcmp(&before, &state->checkpoint, sizeof(before)));
    free(full.path); free(fast.path);
}

static void test_literal_incremental_and_drafts(void) {
    const char *snapshot = "UNIQUE <p>α中🙂 &amp; <![CDATA[raw]]>\r\nLAST";
    edit_document *state = edit_document_new("</parameter>");
    CHECK(edit_document_begin_input(state, "src.py"));
    CHECK(edit_document_provide(state, snapshot));
    buf value = {0};
    buf_printf(&value, "<exmekh_edit path=\"src.py\" id=\"%s\" />", state->item[0].identifier);
    size_t body_start = value.len;
    buf_puts(&value, snapshot);
    buf_printf(&value, "<exmekh_new_text id=\"%s\"/>", state->item[0].identifier);
    buf_puts(&value, "<p>NEW &lt; <![CDATA[x]]> α中🙂\r\n <exmekh_ordinary/>");
    for (size_t n = 0; n <= value.len; n++) {
        check_views(state, value.ptr, n, n, false);
        interactive_edit_commit_string(state, value.ptr, n, "</parameter>");
        const char *pieces[] = {"<", "&", "x", "</parameter>", "<exmekh_new_text/>", "\xf0\x9f"};
        for (size_t i = 0; i < sizeof(pieces)/sizeof(pieces[0]); i++) {
            buf probe = {0}; buf_append(&probe, value.ptr, n); buf_puts(&probe, pieces[i]);
            check_views(state, probe.ptr, probe.len, n, false);
            buf_free(&probe);
        }
    }
    check_views(state, value.ptr, value.len, value.len, true);
    /* Rewind probes must not reuse a later committed frontier. */
    buf draft = {0};
    size_t selected = body_start + strlen("UNIQUE ");
    edit_document_checkpoint restart = edit_document_restart(state, selected, selected);
    buf logical = {0};
    edit_document_view view = interactive_edit_probe(state, &(interactive_edit_input){
        .text = value.ptr, .length = selected, .native_end = "</parameter>", .committed = selected,
        .complete = false, .incremental = true}, &logical);
    edit_document_draft_suffix(state, view.checkpoint, restart, &logical, &draft, 4096);
    CHECK(!strcmp(draft.ptr ? draft.ptr : "", snapshot + strlen("UNIQUE ")));
    free(view.path);
    buf_free(&logical); buf_free(&draft); buf_free(&value); edit_document_free(state);
}

static void test_native_literal_escape(void) {
    request r; fixture(&r);
    buf raw = {0}; buf_puts(&raw, paused_prefix);
    interactive_call *call = pause_call(&r, raw.ptr); CHECK(call);
    provide_snapshot(call, "UNIQUE </parameter> &lt;/parameter>", false);
    finish_opening(&r, call, &raw);
    finish_element(&r, call, &raw, "UNIQUE </parameter> &lt;/parameter>", "</parameter> &lt;/parameter>");
    append_checked(&r, &raw, "</parameter></function></tool_call>");
    tool_calls calls={0}; char *content=NULL,*reasoning=NULL;
    CHECK(interactive_parse_message(&r, raw.ptr, &content, &reasoning, &calls));
    schema_json_value *args=schema_json_parse(calls.v[0].arguments);
    CHECK(strstr(interactive_string(args,"payload"), "UNIQUE </parameter> &lt;/parameter>"));
    schema_json_free(args); tool_calls_free(&calls); free(content); free(reasoning);
    buf_free(&raw); request_free(&r);
}

static void test_all_native_syntaxes(void) {
    const server_model_syntax models[] = {SERVER_MODEL_SYNTAX_QWEN, SERVER_MODEL_SYNTAX_QWEN4,
        SERVER_MODEL_SYNTAX_DEEPSEEK, SERVER_MODEL_SYNTAX_DEEPSEEK41};
    for (size_t i = 0; i < sizeof(models)/sizeof(models[0]); i++) {
        request r; fixture(&r); r.model_syntax = models[i];
        bool qwen = models[i] == SERVER_MODEL_SYNTAX_QWEN4;
        const char *prefix = qwen ? paused_prefix :
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"patch\">"
            "<｜DSML｜parameter name=\"payload\" string=\"true\"><exmekh_edit path=\"src.py\"";
        const char *close = qwen ? "</parameter></function></tool_call>" :
            "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
        buf raw = {0}; buf_puts(&raw, prefix);
        interactive_call *call = pause_call(&r, raw.ptr); CHECK(call);
        buf original = {0}; buf_puts(&original, "UNIQUE "); buf_puts(&original, call->param_end);
        provide_snapshot(call, original.ptr, false);
        finish_opening(&r, call, &raw);
        finish_element(&r, call, &raw, original.ptr, original.ptr);
        append_checked(&r, &raw, close);
        tool_calls calls = {0}; char *content = NULL, *reasoning = NULL;
        CHECK(interactive_parse_message(&r, raw.ptr, &content, &reasoning, &calls));
        CHECK(calls.len == 1);
        schema_json_value *args = schema_json_parse(calls.v[0].arguments);
        CHECK(strstr(interactive_string(args, "payload"), original.ptr));
        schema_json_free(args); tool_calls_free(&calls); free(content); free(reasoning);
        buf_free(&original); buf_free(&raw); request_free(&r);
    }
}

static void test_separator_literal_and_phase_integrity(void) {
    request r; fixture(&r);
    buf raw = {0}; buf_puts(&raw, paused_prefix);
    interactive_call *call = pause_call(&r, raw.ptr); CHECK(call);
    const char *source = "UNIQUE<exmekh_new_text/>TAIL";
    provide_snapshot(call, source, false); finish_opening(&r, call, &raw);
    CHECK(!candidate(&r, raw.ptr, "UNIQUE<exmekh_new_text/>FORGED"));
    finish_element(&r, call, &raw, source, "NEW");
    CHECK(candidate(&r, raw.ptr, "</parameter></function></tool_call>"));
    buf_free(&raw); request_free(&r);

    fixture(&r); buf_puts(&raw, paused_prefix);
    call = pause_call(&r, raw.ptr); CHECK(call);
    provide_snapshot(call, "KEEP", false); finish_opening(&r, call, &raw);
    append_checked(&r, &raw, "KEEP<exmekh_new_text/>");
    buf separator = {0}; separator_text(call, &separator);
    CHECK(!candidate(&r, raw.ptr, separator.ptr));
    buf_free(&separator); buf_free(&raw); request_free(&r);
}

static void test_rejected_snapshot_preserves_authorization(void) {
    edit_document *document = edit_document_new("</parameter>");
    CHECK(edit_document_begin_input(document, "same.py"));
    CHECK(edit_document_provide(document, "ORIGINAL"));
    CHECK(edit_document_begin_input(document, "same.py"));
    edit_document_item pending = document->item[1];
    CHECK(!edit_document_provide(document, "CONTRADICTORY"));
    CHECK(!memcmp(&pending, &document->item[1], sizeof(pending)));
    CHECK(document->snapshot_count == 1);
    CHECK(edit_document_provide(document, "ORIGINAL"));
    CHECK(document->item[0].snapshot == document->item[1].snapshot);
    CHECK(strcmp(document->item[0].identifier, document->item[1].identifier));
    edit_document_free(document);
}

static void test_completion_uses_configured_property(void) {
    request r; fixture(&r);
    interactive_call *call = pause_call(&r, paused_prefix); CHECK(call);
    provide_snapshot(call, "KEEP", false);
    edit_document *document = call->behavior_state;
    buf literal = {0};
    buf_printf(&literal, "<exmekh_edit path=\"src.py\" id=\"%s\" />KEEP", document->item[0].identifier);
    buf_printf(&literal, "<exmekh_new_text id=\"%s\"/>CHANGED", document->item[0].identifier);
    const char *keys[] = {"payload", "wrong_property"};
    for (size_t i = 0; i < sizeof(keys)/sizeof(keys[0]); i++) {
        buf body = {0}; buf_putc(&body, '{'); json_escape(&body, keys[i]); buf_putc(&body, ':');
        json_escape(&body, literal.ptr); buf_putc(&body, '}');
        schema_json_value *args = schema_json_parse(body.ptr);
        CHECK(interactive_edit_complete(call->tool, document, args) == (i == 0));
        schema_json_free(args); buf_free(&body);
    }
    buf_free(&literal); request_free(&r);
}


static void test_rollback_checkpoint_ownership_and_budget(void) {
    server s = {0}; server_slot slot = {0}; s.slots = &slot; s.slot_count = 1;
    pthread_mutex_init(&s.tool_mu, NULL);
    interactive_entry entry = {.slot_id = 0, .context_id = "owner", .status = INTERACTIVE_AWAITING};
    s.interactive_entries = &entry;
    agent_context_checkpoint *paused = xmalloc(sizeof(*paused)); memset(paused, 0, sizeof(*paused));
    paused->context_id = xstrdup("owner"); paused->checkpoint_bytes = 90;
    slot.agent_contexts = paused; entry.checkpoint_bytes = 90;
    agent_context_checkpoint *completed = xmalloc(sizeof(*completed)); memset(completed, 0, sizeof(*completed));
    completed->context_id = xstrdup("owner"); completed->checkpoint_bytes = 80;
    completed->responses_live.valid = true;
    id_list_push_unique(&completed->responses_live.call_ids, "last-completed-call");
    entry.rollback.checkpoint = completed; entry.rollback.reserved_bytes = 80;
    bool pending = false;
    CHECK(!interactive_checkpoint_reserve_locked(&s, 0, "owner", INTERACTIVE_CHECKPOINT_BUDGET - 79, &pending));
    CHECK(pending && entry.checkpoint_bytes == 90);
    CHECK(interactive_checkpoint_reserve_locked(&s, 0, "owner", 90, &pending));
    interactive_publish_rollback_locked(&s, &entry);
    CHECK(slot.agent_contexts == completed && !completed->next);
    CHECK(completed->restore_pending);
    char error[INTERACTIVE_ERROR_BYTES] = {0};
    /* The resident session is still paused. Saving on auxiliary eviction must
     * preserve the published rollback without reading that resident payload. */
    CHECK(agent_context_checkpoint_save(&s, &slot, "owner", error, sizeof(error)));
    CHECK(slot.agent_contexts == completed && completed->restore_pending);
    interactive_publish_rollback_locked(&s, &entry);
    CHECK(slot.agent_contexts == completed && completed->restore_pending);
    CHECK(!entry.rollback.checkpoint && !entry.rollback.reserved_bytes && !entry.checkpoint_bytes);
    CHECK(id_list_contains(&completed->responses_live.call_ids, "last-completed-call"));
    interactive_release_payload(&entry);
    CHECK(slot.agent_contexts == completed);
    agent_context_checkpoints_clear(&slot, true);
    pthread_mutex_destroy(&s.tool_mu);
}

int main(void) {
    test_rollback_checkpoint_ownership_and_budget();
    test_rejected_snapshot_preserves_authorization(); test_completion_uses_configured_property();
    test_text_oracle(); test_literal_contract(); test_empty_and_missing_id();
    test_ids_and_multiple_edits(); test_literal_incremental_and_drafts(); test_native_literal_escape(); test_all_native_syntaxes(); test_separator_literal_and_phase_integrity();
    test_selection_frontier_equivalence(); test_snapshot_draft_policy(); test_speculative_metric_ownership();
    test_resumed_progress_counters(); test_response_metadata(); test_runtime();
    test_concurrent_resume_and_queued_cancel(); test_disconnected_queued_request();
    test_reject_second_element(); test_expiry_and_memory_budget(); test_large_snapshot();
    puts("interactive literal server tests: ok"); return 0;
}
