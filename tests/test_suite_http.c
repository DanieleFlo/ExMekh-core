/* Host-only protocol gates. No model, numeric context or GPU is opened. */
#define DS4_NO_GPU
#define DS4_NATIVE_SUITE
#define DS4_SERVER_TEST
#define main suite_legacy_tests_main
#include "../ds4_server.c"
#undef main
#include <assert.h>

/* The protocol test intentionally cannot construct a device adapter. */
ds4_status ds4_numeric_cuda(ds4_numeric **out, ds4_resources *r, uint64_t owner) {
    (void)r; (void)owner;
    if (out) *out = NULL;
    return DS4_STATUS_UNSUPPORTED;
}
int main(void) {
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
    char type[256] = {0}, key[129] = {0};
    assert(suite_read_headers(headers, sizeof(headers)-1, type, sizeof(type), key, sizeof(key)));
    assert(!strcmp(key, "key") && strstr(type, "boundary=x"));
    const char duplicate[] = "Content-Length: 1\r\nContent-Length: 2\r\n";
    memset(type, 0, sizeof(type)); memset(key, 0, sizeof(key));
    assert(!suite_read_headers(duplicate, sizeof(duplicate)-1, type, sizeof(type), key, sizeof(key)));
    const char chunked[] = "Transfer-Encoding: chunked\r\n";
    assert(!suite_read_headers(chunked, sizeof(chunked)-1, type, sizeof(type), key, sizeof(key)));
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
