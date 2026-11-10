/* Host-only protocol gates. No model, numeric context or GPU is opened. */
#define DS4_NO_GPU
#define DS4_NATIVE_SUITE
#define DS4_SERVER_TEST
#define main suite_legacy_tests_main
#include "../ds4_server.c"
#undef main
#include <assert.h>
#include "suite_progress_cases.inc"

/* The protocol test intentionally cannot construct a device adapter. */
ds4_status ds4_numeric_cuda(ds4_numeric **out, ds4_resources *r, uint64_t owner) {
    (void)r; (void)owner;
    if (out) *out = NULL;
    return DS4_STATUS_UNSUPPORTED;
}
typedef struct {
    server *srv;
    int fd;
    const char *path;
} catalog_client;
static void *catalog_request(void *opaque) {
    catalog_client *request = opaque;
    if (request->path) {
        http_request http = {.method = "GET"};
        snprintf(http.path, sizeof(http.path), "%s", request->path);
        assert(suite_route(request->srv, request->fd, &http));
    } else {
        assert(send_models(request->srv, request->fd));
    }
    return NULL;
}
static void test_catalog_during_inference(void) {
    suite_state q = {0};
    server s = {.suite = &q, .ctx_size = 8192, .default_tokens = 32};
    assert(suite_catalog_init(&q.catalog));
    assert(!pthread_mutex_init(&s.inference_mu, NULL));
    suite_catalog_model snapshot[SUITE_MODELS] = {
        {xstrdup("Qwen/Qwen-Image-2.1-Turbo"),
         xstrdup("{\"id\":\"Qwen/Qwen-Image-2.1-Turbo\"}")},
        {xstrdup("google/embeddinggemma-2"),
         xstrdup("{\"id\":\"google/embeddinggemma-2\"}")},
    };
    suite_catalog_replace(&q.catalog, snapshot);
    static const struct {
        const char *path;
        const char *model_id;
        int status;
    } cases[] = {
        {NULL, NULL, 200},
        {"/v1/models/Qwen/Qwen-Image-2.1-Turbo", "Qwen/Qwen-Image-2.1-Turbo", 200},
        {"/v1/models/google/embeddinggemma-2", "google/embeddinggemma-2", 200},
        {"/v1/models/Qwen-Image-missing", NULL, 404},
    };
    pthread_mutex_lock(&s.inference_mu);
    /* More polls than the connection limit must complete while compute owns
     * the inference lock. An abandoned discovery must not retain a client. */
    for (unsigned i = 0; i < 12; i++) {
        int pair[2]; assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, pair));
        unsigned route = i % (sizeof(cases) / sizeof(*cases));
        catalog_client request = {.srv = &s, .fd = pair[0], .path = cases[route].path};
        pthread_t client;
        assert(!pthread_create(&client, NULL, catalog_request, &request));
        struct pollfd readable = {.fd = pair[1], .events = POLLIN};
        assert(poll(&readable, 1, 1000) == 1);
        pthread_join(client, NULL);
        shutdown(pair[0], SHUT_WR);
        buf response = {0}; char block[4096]; ssize_t bytes;
        while ((bytes = recv(pair[1], block, sizeof(block), 0)) > 0)
            buf_append(&response, block, bytes);
        char status_line[32];
        snprintf(status_line, sizeof(status_line), "HTTP/1.1 %d", cases[route].status);
        assert(strstr(response.ptr, status_line));
        const char *body = strstr(response.ptr, "\r\n\r\n");
        assert(body);
        schema_json_value *catalog = schema_json_parse(body + 4);
        assert(catalog);
        if (!cases[route].path) {
            sj *models = suite_get(catalog, "data");
            assert(models && models->kind == SCHEMA_JSON_ARRAY && models->len == 4);
            assert(!strcmp(suite_str(models->item[2], "id"), "Qwen/Qwen-Image-2.1-Turbo"));
            assert(!strcmp(suite_str(models->item[3], "id"), "google/embeddinggemma-2"));
        } else if (cases[route].model_id) {
            assert(!strcmp(suite_str(catalog, "id"), cases[route].model_id));
        } else {
            assert(!strcmp(suite_str(suite_get(catalog, "error"), "code"), "model_not_found"));
        }
        schema_json_free(catalog); buf_free(&response);
        close(pair[0]); close(pair[1]);
    }
    pthread_mutex_unlock(&s.inference_mu);
    buf current = {0};
    suite_append_models(&s, &current);
    assert(!current.len); /* Refresh once the owner is idle. */
    assert(!suite_catalog_append_model(&q.catalog, "Qwen/Qwen-Image-2.1-Turbo", &current));
    buf_free(&current);
    suite_catalog_close(&q.catalog);
    pthread_mutex_destroy(&s.inference_mu);
}
int main(void) {
    test_catalog_during_inference();
    test_progress_contract();
    ds4_resources *resources = NULL;
    assert(!ds4_resources_create(&resources));
    suite_state diagnostic = {.resources = resources};
    buf ledger = {0};
    unsetenv("DS4_SUITE_RESOURCE_METRICS");
    suite_append_resources(&ledger, &diagnostic);
    assert(!ledger.len);
    setenv("DS4_SUITE_RESOURCE_METRICS", "0", 1);
    suite_append_resources(&ledger, &diagnostic);
    assert(!ledger.len);
    setenv("DS4_SUITE_RESOURCE_METRICS", "1", 1);
    buf_puts(&ledger, "{\"version\":1");
    suite_append_resources(&ledger, &diagnostic);
    buf_putc(&ledger, '}');
    schema_json_value *metrics = schema_json_parse(ledger.ptr);
    assert(metrics && suite_get(metrics, "resources"));
    assert(suite_get(suite_get(metrics, "resources"), "owners")->len == 5);
    schema_json_free(metrics); buf_free(&ledger);
    unsetenv("DS4_SUITE_RESOURCE_METRICS");
    assert(!ds4_resources_close(resources));
    schema_json_value *a = schema_json_parse("{\"z\":[3,2],\"a\":18446744073709551615}");
    schema_json_value *b = schema_json_parse("{\"a\":18446744073709551615,\"z\":[3,2]}");
    uint64_t seed;
    assert(suite_uint(suite_get(a, "a"), &seed) && seed == UINT64_MAX);
    schema_json_value *overflow = schema_json_parse("18446744073709551616");
    assert(!suite_uint(overflow, &seed)); schema_json_free(overflow);
    buf first = {0}, second = {0}; suite_canonical(&first, a); suite_canonical(&second, b);
    assert(!strcmp(first.ptr, second.ptr));
    buf_free(&first); buf_free(&second); schema_json_free(a); schema_json_free(b);
    const char headers[] = "POST /v1/files HTTP/1.1\r\nContent-Length: 42\r\nContent-Type: multipart/form-data; boundary=x\r\nIdempotency-Key: key\r\n\r\n";
    char type[256] = {0}, key[129] = {0}; bool stream = false;
    assert(suite_read_headers(headers, sizeof(headers)-1, type, sizeof(type), key, sizeof(key), &stream));
    assert(!strcmp(key, "key") && strstr(type, "boundary=x"));
    const char duplicate[] = "Content-Length: 1\r\nContent-Length: 2\r\n";
    memset(type, 0, sizeof(type)); memset(key, 0, sizeof(key));
    assert(!suite_read_headers(duplicate, sizeof(duplicate)-1, type, sizeof(type), key, sizeof(key), &stream));
    const char chunked[] = "Transfer-Encoding: chunked\r\n";
    assert(!suite_read_headers(chunked, sizeof(chunked)-1, type, sizeof(type), key, sizeof(key), &stream));
    suite_image_request image = {0};
    a = schema_json_parse("{\"model\":\"Qwen/Qwen-Image-2.1-Turbo\",\"prompt\":\"red\",\"size\":\"33x32\"}");
    assert(suite_prepare_image(NULL, a, false, &image) == DS4_STATUS_INVALID);
    schema_json_free(a);
    a = schema_json_parse("{\"prompt\":\"red\",\"size\":\"4294967552x32\"}");
    assert(suite_prepare_image(NULL, a, false, &image) == DS4_STATUS_INVALID);
    schema_json_free(a);
    a = schema_json_parse("{\"prompt\":\"red\",\"size\":\"32x32\",\"ds4\":{\"version\":1,\"seed\":18446744073709551615}}");
    assert(!suite_prepare_image(NULL, a, false, &image) && image.native.seed == UINT64_MAX);
    schema_json_free(a);
    /* Every independently advertised dimension is accepted in either axis. */
    for (unsigned w = 32; w <= 2048; w += 32)
        for (unsigned h = 32; h <= 2048; h += 32) {
            char request[128];
            snprintf(request, sizeof(request), "{\"prompt\":\"red\",\"size\":\"%ux%u\"}", w, h);
            a = schema_json_parse(request);
            assert(!suite_prepare_image(NULL, a, false, &image));
            assert(image.native.width == w && image.native.height == h);
            schema_json_free(a);
        }
    a = schema_json_parse("{\"prompt\":\"red\",\"size\":\"2048x2048\",\"n\":2}");
    assert(suite_prepare_image(NULL, a, false, &image) == DS4_STATUS_INVALID);
    schema_json_free(a);
    uint8_t pixel = 0;
    ds4_qwen_image_reference reference = {&pixel, 64, 96, 3};
    image.native.references = &reference; image.native.reference_count = 1;
    assert(ds4_qwen_image_request_validate(&image.native) == DS4_STATUS_UNSUPPORTED);
    reference.width = reference.height = 256;
    image.native.prompt = "";
    assert(!ds4_qwen_image_request_validate(&image.native));
    const char multipart[] = "--abc\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nred\r\n--abc\r\nContent-Disposition: form-data; name=\"ds4[seed]\"\r\n\r\n17\r\n--abc\r\nContent-Disposition: form-data; name=\"image\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\nA\0B\r\n--abc--\r\n";
    suite_http h = {.content_type = "multipart/form-data; boundary=abc", .body = multipart, .length = sizeof(multipart)-1};
    suite_part parts[72] = {0}; size_t count;
    assert(suite_multipart(&h, parts, &count) && count == 3 && parts[2].bytes == 3);
    int sockets[2]; assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, sockets));
    buf wire = {0};
    buf_printf(&wire, "POST /v1/images/edits HTTP/1.1\r\nContent-Type: %s\r\nContent-Length: %zu\r\n\r\n", h.content_type, h.length);
    buf_append(&wire, h.body, h.length);
    assert(send_all(sockets[0], wire.ptr, wire.len));
    http_request parsed = {0};
    assert(read_http_request(sockets[1], &parsed) && parsed.spool && parsed.body_len == h.length);
    assert(!memcmp(parsed.body, h.body, h.length));
    http_request_free(&parsed); buf_free(&wire); close(sockets[0]); close(sockets[1]);
    suite_state q = {.quota = SUITE_RECEIPT_MAX};
    snprintf(q.directory, sizeof(q.directory), "/tmp/ds4-suite-test-XXXXXX");
    assert(mkdtemp(q.directory));
    suite_file *f = suite_new_file(&q, "../../untrusted.png", "image/png", "ABC", 3);
    assert(f && strstr(f->path, "file-") && !strstr(f->path, "untrusted"));
    time_t created = f->created;
    f->expires += 120;
    buf metadata = {0}; suite_file_metadata(&metadata, f);
    a = schema_json_parse(metadata.ptr);
    assert(suite_uint(suite_get(a, "created_at"), &seed) && seed == (uint64_t)created);
    assert(suite_uint(suite_get(a, "expires_at"), &seed) && seed == (uint64_t)f->expires);
    schema_json_free(a); buf_free(&metadata);
    f->expires = time(NULL) - 1; suite_expire(&q);
    assert(!q.stored && !*f->id);
    suite_file *kept = suite_new_file(&q, "reference.png", "image/png", "ABC", 3);
    suite_file *pending = suite_new_file(&q, "output.png", "image/png", "ABCD", 4);
    assert(kept && pending && q.stored == 7);
    char removed_path[PATH_MAX];
    snprintf(removed_path, sizeof(removed_path), "%s", pending->path);
    suite_remove_file(&q, pending);
    suite_remove_file(&q, pending);
    assert(access(removed_path, F_OK) == -1 && errno == ENOENT);
    assert(q.stored == 3 && *kept->id && !*pending->id);
    suite_remove_file(&q, kept);
    assert(!q.stored);
    suite_http request = {.path = "/v1/embeddings", .key = "key"};
    suite_id(request.request_id, "req-");
    job j = {.suite_request = &request};
    server s = {.suite = &q};
    a = schema_json_parse("{\"model\":\"gemma\",\"input\":\"text\"}");
    suite_receipt *receipt = NULL;
    assert(!suite_receipt_lookup(&s, &j, a, &receipt) && receipt);
    assert(!*receipt->key); /* admission is not a commit */
    snprintf(receipt->key, sizeof(receipt->key), "%s", request.key);
    snprintf(receipt->request_id, sizeof(receipt->request_id), "%s", request.request_id);
    snprintf(receipt->path, sizeof(receipt->path), "%s/receipt", q.directory);
    const char committed[] = "{\"committed\":true}";
    assert(suite_write(receipt->path, committed, sizeof(committed)-1));
    receipt->bytes = sizeof(committed)-1; q.stored += receipt->bytes;
    receipt->code = 200; receipt->expires = time(NULL) + 600;
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, sockets)); j.fd = sockets[0];
    suite_receipt *ignored;
    assert(suite_receipt_lookup(&s, &j, a, &ignored));
    char response[2048]; ssize_t got = recv(sockets[1], response, sizeof(response)-1, 0);
    assert(got > 0); response[got] = 0;
    assert(strstr(response, "200 OK") && strstr(response, committed));
    close(sockets[0]); close(sockets[1]);
    receipt->expires = time(NULL)-1; suite_expire(&q); assert(!q.stored);
    request.deadline = now_sec() - 1;
    pthread_mutex_init(&j.mu, NULL);
    assert(suite_cancel(&j));
    pthread_mutex_destroy(&j.mu);
    schema_json_free(a);
    assert(!rmdir(q.directory));
    puts("suite HTTP: PASS integers, canonical receipts, multipart, limits, lifetime and deadlines");
    return 0;
}
