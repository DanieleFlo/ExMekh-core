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
        "\"name\":\"patch\",\"behavior\":\"edit_file.v2\",\"fields\":{\"document\":\"payload\"}}]}";
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

static const char *const parameter_prefix = "<tool_call><function=patch><parameter=payload>";
static const char *const paused_prefix = "<tool_call><function=patch><parameter=payload><edit path=\"src.py\"";

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
    append_checked(r, raw, ">");
}

static void finish_element(request *r, interactive_call *call, buf *raw,
                           const char *old, const char *replacement) {
    CHECK(call->authorized);
    buf part = {0}; buf_printf(&part, "<old_text>%s</old_text>", old);
    append_checked(r, raw, part.ptr); buf_free(&part);
    append_checked(r, raw, "<new_text>");
    buf_printf(&part, "%s</new_text>", replacement);
    append_checked(r, raw, part.ptr); buf_free(&part);
    append_checked(r, raw, "</edit>");
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
        edit_constraint snapshot; edit_constraint_init(&snapshot, text, false);
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

static void test_document_constraints(void) {
    request r; fixture(&r);
    CHECK(candidate(&r, parameter_prefix, "<edit path=\"src.py\""));
    CHECK(!candidate(&r, parameter_prefix, "<edit path=\"src.py\">"));
    CHECK(!candidate(&r, parameter_prefix, "<edit path=\"src.py\" code=\"123456\">"));
    CHECK(!candidate(&r, parameter_prefix, "<edit path=\"src.py\"<old_text>"));
    CHECK(!candidate(&r, parameter_prefix, "</parameter>"));
    CHECK(!candidate(&r, "<tool_call><function=patch>", "<parameter=payload string=false>"));
    buf raw = {0}; buf_puts(&raw, paused_prefix);
    interactive_call *call = pause_call(&r, raw.ptr);
    CHECK(call && call->input_index == 1 && !call->authorized);
    CHECK(!candidate(&r, raw.ptr, "<old_text>"));
    CHECK(pause_call(&r, raw.ptr) == call);
    provide_snapshot(call, "foo\nprefix foo\nbar", false);
    finish_opening(&r, call, &raw);
    CHECK(!candidate(&r, raw.ptr, "<new_text>"));
    append_checked(&r, &raw, "<old_text>");
    CHECK(!candidate(&r, raw.ptr, "refix"));
    append_checked(&r, &raw, "foo");
    CHECK(!candidate(&r, raw.ptr, "</old_text>"));
    CHECK(!candidate(&r, raw.ptr, "\np</old_text><new_text>bad"));
    append_checked(&r, &raw, "\np");
    dsml_decode_tracker dt; dsml_decode_tracker_init(&dt, &r);
    dsml_decode_tracker_update(&dt, raw.ptr, raw.len);
    CHECK(interactive_sampling_body(&r, &dt, raw.ptr, raw.len));
    CHECK(interactive_temperature(&r, &dt, 0.7f, raw.ptr, raw.len) == 0);
    uint64_t old_epoch = interactive_sampling_epoch(&r, &dt, raw.ptr, raw.len);
    CHECK(candidate(&r, raw.ptr, "refix foo\nbar</old_text>"));
    append_checked(&r, &raw, "</old_text>");
    CHECK(!candidate(&r, raw.ptr, "<new_text>body"));
    append_checked(&r, &raw, "<new_text>");
    dsml_decode_tracker_update(&dt, raw.ptr, raw.len);
    CHECK(interactive_temperature(&r, &dt, 0.7f, raw.ptr, raw.len) == 0.7f);
    CHECK(interactive_sampling_body(&r, &dt, raw.ptr, raw.len));
    uint64_t new_epoch = interactive_sampling_epoch(&r, &dt, raw.ptr, raw.len);
    CHECK(new_epoch && new_epoch != old_epoch);
    CHECK(!candidate(&r, raw.ptr, "<delete/>"));
    CHECK(!candidate(&r, raw.ptr, "<delete"));
    CHECK(!candidate(&r, raw.ptr, "<!DOCTYPE x>"));
    CHECK(!candidate(&r, raw.ptr, "&unknown;"));
    CHECK(!candidate(&r, raw.ptr, "&#0;"));
    CHECK(!candidate(&r, raw.ptr, "&#xD800;"));
    CHECK(candidate(&r, raw.ptr, "\nordinary\n"));
    CHECK(candidate(&r, raw.ptr, "\n@ex_meckh\ndelete"));
    CHECK(candidate(&r, raw.ptr, "ex_meckh_delete"));
    CHECK(candidate(&r, raw.ptr, "literal </new_text> data"));
    CHECK(candidate(&r, raw.ptr, "done</new_text></edit>"));
    CHECK(candidate(&r, raw.ptr, "done</new_text>\r\n </edit>"));
    CHECK(candidate(&r, raw.ptr, "literal </new_text><!["));
    /* Trie probes grow a token with a fixed committed boundary. Every
     * intermediate prefix must remain viable before its lookahead resolves. */
    buf pending={0}; buf_puts(&pending,raw.ptr); buf_puts(&pending,"literal </new_text");
    const char *lookaheads[]={"><![", "></edit>", ">\n</edit>"};
    for (size_t i=0; i<sizeof(lookaheads)/sizeof(lookaheads[0]); i++) {
        for (size_t j=1; j<=strlen(lookaheads[i]); j++) {
            char *piece=xstrndup(lookaheads[i],j);
            CHECK(candidate(&r,pending.ptr,piece)); free(piece);
        }
    }
    buf_putc(&pending,'>');
    dsml_decode_tracker_update(&dt,pending.ptr,pending.len);
    CHECK(interactive_temperature(&r,&dt,0.7f,pending.ptr,pending.len)==0.7f);
    CHECK(!interactive_sampling_body(&r,&dt,pending.ptr,pending.len));
    CHECK(interactive_sampling_epoch(&r,&dt,pending.ptr,pending.len)==new_epoch);
    buf_free(&pending);
    CHECK(!candidate(&r, raw.ptr, "literal </new_text></parameter>"));
    CHECK(!candidate(&r, raw.ptr, "</new_text code=\"123456\">"));
    buf part = {0}; buf_puts(&part, "literal </new_text> data </new_text> again</new_text>");
    CHECK(candidate(&r, raw.ptr, "</new_text>"));
    /* The matching delimiter is available byte by byte, including its first '<'. */
    for (size_t i = 0; i < part.len; i++) {
        char byte[2] = {part.ptr[i], 0}; append_checked(&r, &raw, byte);
    }
    append_checked(&r, &raw, "</edit>");
    append_checked(&r, &raw, "</parameter></function></tool_call>");
    tool_calls calls = {0}; char *content = NULL, *reasoning = NULL;
    CHECK(interactive_parse_message(&r, raw.ptr, &content, &reasoning, &calls));
    CHECK(interactive_apply_call_ids(&r, raw.ptr, &calls));
    CHECK(calls.len == 1 && !strcmp(calls.v[0].id, call->call_id));
    schema_json_value *args = schema_json_parse(calls.v[0].arguments);
    CHECK(strstr(interactive_string(args, "payload"), "literal </new_text> data"));
    CHECK(strstr(interactive_string(args, "payload"), "data </new_text> again</new_text></edit>"));
    CHECK(!strstr(interactive_string(args, "payload"), " code="));
    schema_json_free(args);
    /* Final validation independently refuses a changed frozen resource. */
    buf changed = {0};
    const char *path = strstr(raw.ptr, "src.py"); CHECK(path);
    buf_append(&changed, raw.ptr, (size_t)(path - raw.ptr));
    buf_puts(&changed, "different.py"); buf_puts(&changed, path + strlen("src.py"));
    CHECK(!interactive_apply_call_ids(&r, changed.ptr, &calls));
    buf_free(&changed);
    tool_calls_free(&calls); free(content); free(reasoning);
    buf_free(&part); buf_free(&raw); request_free(&r);
}

static void test_special_operations(void) {
    for (int create = 0; create < 2; create++) {
        request r; fixture(&r);
        buf raw = {0}; buf_puts(&raw, paused_prefix);
        interactive_call *call = pause_call(&r, raw.ptr); CHECK(call);
        provide_snapshot(call, "", create != 0); finish_opening(&r, call, &raw);
        buf head = {0}; buf_puts(&head, raw.ptr);
        CHECK(candidate(&r, head.ptr, "<old_text><all/></old_text>"));
        CHECK(candidate(&r, head.ptr, "<old_text><line number=\"1\"/></old_text>"));
        CHECK(!candidate(&r, head.ptr, "<old_text></old_text>"));
        CHECK(!candidate(&r, head.ptr, "<old_text><line number=\"0\"/></old_text>"));
        CHECK(!candidate(&r, head.ptr, "<old_text><line number=\"2\"/></old_text>"));
        CHECK(!candidate(&r, head.ptr, "<old_text><line number=\"01\"/></old_text>"));
        const char *selection="<old_text><all/></old_text>";
        for (size_t i=0; selection[i]; i++) {
            char byte[2]={selection[i],0}; append_checked(&r,&raw,byte);
        }
        buf insertion={0}; buf_puts(&insertion,head.ptr);
        const char *line="<old_text><line number=\"1\"/></old_text>";
        for (size_t i=0; line[i]; i++) {
            char byte[2]={line[i],0}; append_checked(&r,&insertion,byte);
        }
        buf_free(&insertion);
        append_checked(&r, &raw, "<new_text>");
        CHECK(candidate(&r, raw.ptr, "<delete/>") == !create);
        CHECK(!candidate(&r, raw.ptr, "\n<delete"));
        buf close = {0}; buf_printf(&close, "%s</new_text>", create ? "" : "<delete/>");
        for (size_t i=0; i<close.len; i++) {
            char byte[2]={close.ptr[i],0}; append_checked(&r,&raw,byte);
        }
        append_checked(&r, &raw, "</edit></parameter></function></tool_call>");
        CHECK(!candidate(&r, head.ptr, "<old_text><all/>extra</old_text>"));
        buf_free(&head); buf_free(&close); buf_free(&raw); request_free(&r);
    }
}

static void test_multiple_elements_and_ranges(void) {
    request r; fixture(&r);
    buf raw = {0}; buf_puts(&raw, paused_prefix);
    interactive_call *call = pause_call(&r, raw.ptr); CHECK(call);
    provide_snapshot(call, "one\ntwo\nthree", false); finish_opening(&r, call, &raw);
    char first_id[40], first_exchange[40];
    strcpy(first_id, call->call_id); strcpy(first_exchange, call->exchange_id);
    finish_element(&r, call, &raw, "one", "first");
    append_checked(&r, &raw, "<edit path=\"src.py\"");
    interactive_call *second = pause_call(&r, raw.ptr);
    CHECK(second == call && second->input_index == 2 && !second->authorized);
    CHECK(!strcmp(first_id, second->call_id) && strcmp(first_exchange, second->exchange_id));
    CHECK(!candidate(&r, raw.ptr, " code=\"123456\">"));
    schema_json_value *changed = schema_json_parse("{\"status\":\"create\"}");
    CHECK(!interactive_call_prepare_reply(second, changed));
    schema_json_free(changed);
    changed = schema_json_parse("{\"status\":\"provide\",\"text\":\"changed\",\"encoding\":\"utf-8\",\"line_count\":1}");
    CHECK(!interactive_call_prepare_reply(second, changed));
    schema_json_free(changed);
    CHECK(!candidate(&r, raw.ptr, ">"));
    provide_snapshot(second, "one\ntwo\nthree", false);
    const edit_document *document = second->behavior_state;
    CHECK(edit_document_reuses_snapshot(document));
    schema_json_value *same_reply=schema_json_parse("{\"status\":\"provide\",\"text\":\"one\\ntwo\\nthree\"}");
    CHECK(!interactive_edit_reply_bytes(document,same_reply));
    CHECK(!interactive_edit_reply_memory(document,same_reply)); schema_json_free(same_reply);
    finish_opening(&r, second, &raw);
    /* Semantically wrong ranges pass through the server; the client owns them. */
    append_checked(&r, &raw, "<old_text start_line=\"999999999999999999999999999999\" end_line=\"0\"></old_text>");
    CHECK(!candidate(&r, raw.ptr, "<new_text><delete/>"));
    append_checked(&r, &raw, "<new_text>");
    buf close = {0}; buf_puts(&close, "range</new_text>");
    append_checked(&r, &raw, close.ptr); append_checked(&r, &raw, "</edit></parameter></function></tool_call>");
    tool_calls calls = {0}; char *content=NULL, *reasoning=NULL;
    CHECK(interactive_parse_message(&r, raw.ptr, &content, &reasoning, &calls));
    CHECK(interactive_apply_call_ids(&r, raw.ptr, &calls) && calls.len == 1);
    tool_calls_free(&calls); free(content); free(reasoning);
    buf_free(&close); buf_free(&raw); request_free(&r);
}

static void test_entities_cdata_and_native_delimiters(void) {
    const char *selections[] = {
        "&lt;x&gt;&amp;&quot;&apos;\r\n",
        "<![CDATA[<x>&\"'\r\n]]>",
        "&#60;x&#62;&#38;&#34;&#39;&#13;&#10;"
    };
    for (size_t test=0; test < sizeof(selections)/sizeof(selections[0]); test++) {
        request r; fixture(&r);
        buf raw = {0}; buf_puts(&raw, paused_prefix);
        interactive_call *call=pause_call(&r, raw.ptr); CHECK(call);
        provide_snapshot(call, "<x>&\"'\r\nkeep", false); finish_opening(&r, call, &raw);
        append_checked(&r, &raw, "<old_text>");
        for (size_t i=0; selections[test][i]; i++) {
            char byte[2]={selections[test][i], 0}; append_checked(&r, &raw, byte);
        }
        append_checked(&r, &raw, "</old_text>"); append_checked(&r, &raw, "<new_text>");
        buf literal={0}; buf_puts(&literal, "<![CDATA[</new_text></edit>]]><![CDATA[ raw </new_text>]]>");
        for (size_t i=0; i<literal.len; i++) {
            char byte[2]={literal.ptr[i],0}; append_checked(&r,&raw,byte);
        }
        append_checked(&r,&raw,"</new_text>"); append_checked(&r,&raw,"</edit>");
        buf_free(&literal); buf_free(&raw); request_free(&r);
    }
    request r; fixture(&r);
    buf raw={0}; buf_puts(&raw, paused_prefix);
    interactive_call *call=pause_call(&r, raw.ptr); CHECK(call);
    provide_snapshot(call, "é</parameter>\n", false); finish_opening(&r, call, &raw);
    const char *piece="<old_text><![CDATA[é&lt;/parameter>\n]]></old_text>";
    for (size_t i=0; piece[i]; i++) {
        char byte[2]={piece[i],0}; append_checked(&r, &raw, byte);
    }
    buf_free(&raw); request_free(&r);
}

static void test_configuration_and_description(void) {
    request r; char error[256]={0};
    fixture_schema(&r, "{\"type\":\"string\",\"enum\":[\"x\"]}");
    CHECK(!interactive_compile(&r, error, sizeof(error))); request_free(&r);
    fixture_schema(&r, "{\"type\":\"string\",\"const\":\"x\"}");
    CHECK(!interactive_compile(&r, error, sizeof(error))); request_free(&r);
    fixture(&r);
    schema_json_value *version=(schema_json_value *)schema_json_object_get(r.interactive.wire,"version");
    version->number=1; CHECK(!interactive_compile(&r,error,sizeof(error)));
    request_free(&r);
    fixture(&r);
    buf registry={0}; buf_puts(&registry,"{\"name\":\"patch\",\"description\":\"User description\",\"parameters\":{}}\n"
                                        "{\"name\":\"ordinary\",\"description\":\"untouched\"}");
    interactive_tool_descriptions(&registry,&r);
    CHECK(strstr(registry.ptr,"User description") && strstr(registry.ptr,"final </new_text>"));
    CHECK(!strstr(registry.ptr,"six-digit"));
    CHECK(strstr(registry.ptr,"untouched"));
    char *before=xstrdup(registry.ptr); interactive_tool_descriptions(&registry,&r);
    CHECK(!strcmp(before,registry.ptr)); free(before); buf_free(&registry);
    char *capabilities=interactive_capabilities_body(SERVER_MODEL_SYNTAX_QWEN);
    CHECK(strstr(capabilities,"edit_file.v2") && !strstr(capabilities,"edit_file.v1"));
    CHECK(strstr(capabilities,"\"version\":2") && strstr(capabilities,"state_budget_bytes"));
    free(capabilities); request_free(&r);
}

static void test_escape_frontiers(void) {
    request r; fixture(&r);
    buf raw={0}; buf_puts(&raw,paused_prefix);
    interactive_call *call=pause_call(&r,raw.ptr); CHECK(call);
    provide_snapshot(call,"x",false); finish_opening(&r,call,&raw);
    CHECK(!candidate(&r,raw.ptr,"<old_text&"));
    append_checked(&r,&raw,"<old_text>x");
    CHECK(!candidate(&r,raw.ptr,"&"));
    CHECK(!candidate(&r,raw.ptr,"&amp"));
    CHECK(!candidate(&r,raw.ptr,"]"));
    CHECK(!candidate(&r,raw.ptr,"]]"));
    append_checked(&r,&raw,"</old_text>");
    append_checked(&r,&raw,"<new_text>");
    CHECK(!candidate(&r,raw.ptr,"\xef\xbf\xbe"));
    CHECK(!candidate(&r,raw.ptr,"<![CDATA[\xef\xbf\xbe"));
    CHECK(candidate(&r,raw.ptr,"\xf0\x9f"));
    CHECK(candidate(&r,raw.ptr,"\xf0\x9f\x98\x80"));
    CHECK(!candidate(&r,"<tool_call><function=patch><parameter=payload><edit path=\"","\xef\xbf\xbe"));
    CHECK(!candidate(&r,"<tool_call><function=patch><parameter=payload><edit path=\"&#1",
                     "&lt;/parameter"));
    CHECK(!candidate(&r,"<tool_call><function=patch><parameter=payload><edit path=\"\xf0\x9f",
                     "&lt;/parameter"));
    buf partial={0}; buf_puts(&partial,raw.ptr); buf_puts(&partial,"</new_text></ed");
    CHECK(!candidate(&r,partial.ptr,"&"));
    CHECK(!candidate(&r,partial.ptr,"&amp;"));
    CHECK(!candidate(&r,partial.ptr,"&lt;"));
    const char *unfinished[] = {"<", "<![CDATA", "&#1", "\xf0\x9f"};
    for (size_t i=0; i<sizeof(unfinished)/sizeof(unfinished[0]); i++) {
        buf frontier={0}; buf_puts(&frontier,raw.ptr); buf_puts(&frontier,unfinished[i]);
        CHECK(candidate(&r,raw.ptr,unfinished[i]));
        CHECK(!candidate(&r,frontier.ptr,"&lt;/parameter"));
        buf_free(&frontier);
    }
    /* The live DSML failure ended in <&lt;/｜DSML｜parameter. Test both
     * transports directly; partial native escapes must remain extendable. */
    const char *ends[] = {"</parameter>", "</｜DSML｜parameter>"};
    const edit_document *document=call->behavior_state;
    const char *value=strstr(raw.ptr,"<edit path=");
    for (size_t i=0; i<sizeof(ends)/sizeof(ends[0]); i++) {
        buf frontier={0}; buf_puts(&frontier,value); buf_putc(&frontier,'<');
        buf_puts(&frontier,"&lt;"); buf_puts(&frontier,ends[i]+1);
        frontier.ptr[--frontier.len]=0;
        edit_document_view view=interactive_edit_view(document,frontier.ptr,frontier.len,
                                                       ends[i],frontier.len,false);
        CHECK(!view.valid); free(view.path); buf_free(&frontier);
    }
    CHECK(interactive_utf8_valid((const unsigned char *)"\xe0",1,false));
    CHECK(!interactive_utf8_valid((const unsigned char *)"\xe0",1,true));
    CHECK(!interactive_utf8_valid((const unsigned char *)"\xe0\x80",2,false));
    CHECK(!interactive_utf8_valid((const unsigned char *)"\xc3<",2,false));
    buf_free(&partial); buf_free(&raw); request_free(&r);

    fixture(&r); buf_puts(&raw,paused_prefix);
    call=pause_call(&r,raw.ptr); CHECK(call);
    provide_snapshot(call," A",false); finish_opening(&r,call,&raw);
    append_checked(&r,&raw,"<old_text>");
    CHECK(candidate(&r,raw.ptr,"&"));
    CHECK(candidate(&r,raw.ptr,"&#3"));
    CHECK(!candidate(&r,raw.ptr,"&l"));
    CHECK(!candidate(&r,raw.ptr,"&#1"));
    CHECK(!candidate(&r,raw.ptr,"&#xA"));
    append_checked(&r,&raw,"&#32;");
    append_checked(&r,&raw,"A</old_text>");
    buf_free(&raw); request_free(&r);
}

static void test_cdata_end_prefix_matches_snapshot(void) {
    request r; fixture(&r);
    buf raw = {0}; buf_puts(&raw, paused_prefix);
    interactive_call *call = pause_call(&r, raw.ptr); CHECK(call);
    provide_snapshot(call, "  value\nprefix value\naaa\nvalue", false);
    finish_opening(&r, call, &raw);
    append_checked(&r, &raw, "<old_text>aa<![CDATA[]]>");
    CHECK(!candidate(&r, raw.ptr, "]"));
    CHECK(!candidate(&r, raw.ptr, "]]"));
    CHECK(candidate(&r, raw.ptr, "a</old_text>"));
    buf_free(&raw); request_free(&r);

    fixture(&r); buf_puts(&raw, paused_prefix);
    call = pause_call(&r, raw.ptr); CHECK(call);
    provide_snapshot(call, "]]x", false); finish_opening(&r, call, &raw);
    append_checked(&r, &raw, "<old_text>");
    append_checked(&r, &raw, "]"); append_checked(&r, &raw, "]");
    CHECK(!candidate(&r, raw.ptr, ">"));
    CHECK(candidate(&r, raw.ptr, "</old_text>"));
    buf_free(&raw); request_free(&r);
}

static void test_native_syntaxes_and_reasoning(void) {
    const server_model_syntax syntaxes[] = {SERVER_MODEL_SYNTAX_QWEN, SERVER_MODEL_SYNTAX_DEEPSEEK,
                                          SERVER_MODEL_SYNTAX_DEEPSEEK41};
    for (size_t i=0; i<sizeof(syntaxes)/sizeof(syntaxes[0]); i++) {
        request r; fixture(&r); r.model_syntax=syntaxes[i]; r.think_mode=DS4_THINK_HIGH;
        dsml_decode_tracker dt; dsml_decode_tracker_init(&dt,&r);
        thinking_state thinking={.inside=true};
        buf raw={0}; buf_puts(&raw,"Quoted <tool_call><function=patch>ignored");
        generation_protocol_feed(&r,&thinking,&dt,raw.ptr,raw.len,raw.ptr,raw.len);
        CHECK(!interactive_observe(&r,&dt,raw.ptr,raw.len));
        buf_puts(&raw,"</think>");
        generation_protocol_feed(&r,&thinking,&dt,raw.ptr,raw.len,"</think>",8);
        const char *prefix="<｜DSML｜tool_calls><｜DSML｜invoke name=\"patch\">"
                           "<｜DSML｜parameter name=\"payload\" string=\"true\"><edit path=\"src.py\"";
        append_checked(&r,&raw,prefix);
        generation_protocol_feed(&r,&thinking,&dt,raw.ptr,raw.len,prefix,strlen(prefix));
        interactive_call *call=interactive_observe(&r,&dt,raw.ptr,raw.len); CHECK(call);
        provide_snapshot(call,"",false); finish_opening(&r,call,&raw);
        const char *selection="<old_text><all/></old_text>";
        for (size_t j=0; selection[j]; j++) {
            char byte[2]={selection[j],0}; append_checked(&r,&raw,byte);
        }
        append_checked(&r,&raw,"<new_text>");
        buf operation={0}; buf_puts(&operation,"<delete/></new_text>");
        for (size_t j=0; j<operation.len; j++) {
            char byte[2]={operation.ptr[j],0}; append_checked(&r,&raw,byte);
        }
        buf_free(&operation); append_checked(&r,&raw,"</edit>");
        append_checked(&r,&raw,"</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>");
        tool_calls calls={0}; char *content=NULL,*reasoning=NULL;
        CHECK(interactive_parse_message(&r,raw.ptr,&content,&reasoning,&calls));
        CHECK(interactive_apply_call_ids(&r,raw.ptr,&calls) && calls.len==1);
        CHECK(reasoning && strstr(reasoning,"ignored"));
        tool_calls_free(&calls); free(content); free(reasoning); buf_free(&raw); request_free(&r);
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
    provide_snapshot(call,"",true); finish_opening(&r,call,&raw);
    finish_element(&r,call,&raw,"<all/>","first");
    append_checked(&r,&raw,"<edit path=\"second.py\"");
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
    double start=now_sec(); edit_constraint snapshot; edit_constraint_init(&snapshot,text,false);
    double index_ms=(now_sec()-start)*1000;
    const char *selection="repeated repeated repeated repeated repeated line 999\r\n";
    CHECK(edit_selection_prefix(&snapshot,selection,strlen(selection),true,NULL));
    start=now_sec();
    volatile unsigned verified=0;
    for (int i=0;i<1000;i++) verified+=edit_selection_prefix(&snapshot,selection,strlen(selection),true,NULL);
    CHECK(verified==1000);
    printf("interactive snapshot benchmark bytes=%zu lines=%zu index_ms=%.3f lookup_us=%.3f\n",
           length,edit_constraint_line_count(&snapshot),index_ms,(now_sec()-start)*1000);
    edit_constraint_free(&snapshot); free(text);
}

int main(void) {
    test_text_oracle(); test_document_constraints(); test_special_operations();
    test_multiple_elements_and_ranges(); test_entities_cdata_and_native_delimiters();
    test_configuration_and_description(); test_escape_frontiers(); test_native_syntaxes_and_reasoning();
    test_cdata_end_prefix_matches_snapshot();
    test_response_metadata(); test_runtime();
    test_concurrent_resume_and_queued_cancel(); test_disconnected_queued_request();
    test_reject_second_element(); test_expiry_and_memory_budget(); test_large_snapshot();
    puts("interactive XML server tests: ok"); return 0;
}
