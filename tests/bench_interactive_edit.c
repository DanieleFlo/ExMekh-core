#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

/* Model-free parser cost, not inference throughput. Both paths receive the
 * same viable continuation; native fixtures verify their equivalence. */
static double measure(const edit_document *state, const buf *value, size_t committed,
                      bool incremental, size_t iterations) {
    double start = now_sec();
    for (size_t i = 0; i < iterations; i++) {
        edit_document_view view = interactive_edit_probe(state, &(interactive_edit_input){
            .text = value->ptr, .length = value->len, .native_end = "</parameter>", .committed = committed,
            .complete = false, .incremental = incremental}, NULL);
        if (!view.valid) { fprintf(stderr, "invalid benchmark probe\n"); exit(1); }
        free(view.path);
    }
    return now_sec() - start;
}

int main(int argc, char **argv) {
    const char *cases[] = {"new_plain", "new_markup", "new_unicode", "new_entities",
                          "old_literal", "old_unicode", "old_markup", "old_entities", "old_ambiguous"};
    const size_t sizes[] = {128, 1024, 4096, 16384};
    if (argc > 2) { fprintf(stderr, "usage: %s [case]\n", argv[0]); return 2; }
    bool measured = false;
    puts("case,prefix_bytes,mode,repeat,iterations,ns_per_probe");
    for (size_t test = 0; test < sizeof(cases)/sizeof(cases[0]); test++) {
        if (argc == 2 && strcmp(argv[1], cases[test])) continue;
        measured = true;
        bool old = test >= 4;
        const char *unit = test == 2 || test == 5 ? "α中🙂" :
                           test == 3 || test == 7 ? "&amp;&#x3b1;" :
                           test == 1 || test == 6 ? "<p>Hello world & text</p>\n" : "plain text ";
        const char *logical_unit = unit;
        for (size_t size = 0; size < sizeof(sizes)/sizeof(sizes[0]); size++) {
            buf body = {0}, snapshot = {0}, value = {0};
            buf_puts(&body, "UNIQUE:"); buf_puts(&snapshot, "UNIQUE:");
            while (body.len < sizes[size]) { buf_puts(&body, unit); buf_puts(&snapshot, logical_unit); }
            buf_puts(&snapshot, "tail");
            if (test == 8) {
                snapshot.len = 0;
                for (int line = 0; line < 32; line++) {
                    buf_puts(&snapshot, body.ptr); buf_printf(&snapshot, "tail%d\n", line);
                }
            }
            edit_document *state = edit_document_new("</parameter>");
            if (!edit_document_begin_input(state, "a") || !edit_document_provide(state, snapshot.ptr)) return 1;
            buf_printf(&value, "<exmekh_edit path=\"a\" id=\"%s\" />", state->item[0].identifier);
            if (!old) buf_printf(&value, "UNIQUE:<exmekh_new_text id=\"%s\"/>", state->item[0].identifier);
            buf_puts(&value, body.ptr);
            size_t committed = value.len;
            interactive_edit_commit_string(state, value.ptr, committed, "</parameter>");
            buf_puts(&value, old ? "t" : "next");
            for (int mode = 0; mode < 2; mode++) {
                /* One warmup, then five samples lasting at least ~100 ms. */
                size_t iterations = 100;
                double warmup = measure(state, &value, committed, mode != 0, iterations);
                if (warmup > 0) iterations = (size_t)(0.15 * iterations / warmup) + 1;
                if (iterations > 3000000) iterations = 3000000;
                for (int repeat = 0; repeat < 5; repeat++) {
                    double seconds = measure(state, &value, committed, mode != 0, iterations);
                    printf("%s,%zu,%s,%d,%zu,%.3f\n", cases[test], committed,
                        mode ? "incremental" : "full", repeat, iterations, seconds * 1e9 / iterations);
                }
            }
            edit_document_free(state); buf_free(&body); buf_free(&snapshot); buf_free(&value);
        }
    }
    return measured ? 0 : 2;
}
