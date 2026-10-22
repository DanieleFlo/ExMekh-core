/* Transport normalization and dispatch. Multipart bodies are mmap-backed
 * temporary files; only bounded headers and ordinary JSON live on the heap. */
typedef struct {
    char name[64], filename[256], mime[40];
    const char *data;
    size_t bytes;
} suite_part;
static bool suite_disposition(const char *line, const char *key, char *out, size_t capacity) {
    char needle[64]; snprintf(needle, sizeof(needle), "%s=\"", key);
    const char *p = strstr(line, needle);
    if (!p) return false;
    p += strlen(needle);
    const char *end = strchr(p, '"');
    if (!end || end == p || (size_t)(end - p) >= capacity) return false;
    for (const char *a = p; a < end; a++) if ((unsigned char)*a < 32 || *a == '\\') return false;
    memcpy(out, p, end - p); out[end - p] = 0;
    return true;
}
static bool suite_multipart(const suite_http *h, suite_part parts[72], size_t *count) {
    *count = 0;
    const char *boundary = strstr(h->content_type, "boundary=");
    if (!boundary) return false;
    boundary += 9;
    char mark[80]; size_t n = strlen(boundary);
    if (n >= 2 && boundary[0] == '"' && boundary[n - 1] == '"') boundary++, n -= 2;
    if (!n || n > 70 || strpbrk(boundary, "\r\n")) return false;
    memcpy(mark, "--", 2); memcpy(mark + 2, boundary, n); n += 2; mark[n] = 0;
    const char *p = h->body, *end = p + h->length;
    if ((size_t)(end - p) < n + 2 || memcmp(p, mark, n)) return false;
    p += n;
    while (p < end) {
        if (end - p >= 2 && !memcmp(p, "--", 2)) {
            p += 2;
            return p == end || (end - p == 2 && !memcmp(p, "\r\n", 2));
        }
        if (*count == 72 || end - p < 2 || memcmp(p, "\r\n", 2)) return false;
        p += 2;
        const char *head = memmem(p, end - p, "\r\n\r\n", 4);
        if (!head || head - p > 8192) return false;
        char *headers = xstrndup(p, head - p);
        suite_part *part = parts + (*count)++;
        bool ok = suite_disposition(headers, "name", part->name, sizeof(part->name));
        if (strstr(headers, "filename="))
            ok = ok && suite_disposition(headers, "filename", part->filename, sizeof(part->filename));
        const char *mime = strcasestr(headers, "Content-Type:");
        if (mime) {
            mime += 13; while (*mime == ' ') mime++;
            size_t length = strcspn(mime, "\r\n");
            if (length >= sizeof(part->mime)) ok = false;
            else memcpy(part->mime, mime, length);
        }
        free(headers);
        if (!ok) return false;
        p = head + 4; part->data = p;
        char delimiter[84]; snprintf(delimiter, sizeof(delimiter), "\r\n%s", mark);
        const char *next = p;
        do {
            next = memmem(next, end - next, delimiter, n + 2);
            if (!next) return false;
            const char *suffix = next + n + 2;
            if (end - suffix >= 2 && (!memcmp(suffix, "--", 2) || !memcmp(suffix, "\r\n", 2))) break;
            next++;
        } while (next < end);
        part->bytes = next - p;
        if (!part->bytes) return false;
        p = next + n + 2;
    }
    return false;
}
static bool suite_mime(const char *mime) {
    const char *types[] = {"image/png", "image/jpeg", "image/webp", "video/mp4", "video/webm",
        "audio/wav", "audio/mpeg", "audio/flac", "audio/ogg"};
    for (size_t i = 0; i < sizeof(types)/sizeof(*types); i++) if (!strcmp(mime, types[i])) return true;
    return false;
}
static void suite_file_metadata(buf *b, const suite_file *f) {
    buf_puts(b, "{\"object\":\"file\",\"id\":"); json_escape(b, f->id);
    buf_puts(b, ",\"filename\":"); json_escape(b, f->name);
    buf_printf(b, ",\"bytes\":%llu,\"purpose\":\"ds4_media\",\"created_at\":%lld,\"expires_at\":%lld}",
        (unsigned long long)f->bytes, (long long)f->created, (long long)f->expires);
}
static void suite_upload(server *s, job *j) {
    suite_part parts[72] = {0}; size_t count;
    if (!suite_multipart(j->suite_request, parts, &count) || count != 2) goto invalid;
    suite_part *file = NULL; bool purpose = false;
    for (size_t i = 0; i < count; i++) {
        suite_part *p = parts + i;
        if (!strcmp(p->name, "file") && *p->filename && suite_mime(p->mime)) file = p;
        else if (!strcmp(p->name, "purpose") && !*p->filename && p->bytes == 9 && !memcmp(p->data, "ds4_media", 9)) purpose = true;
        else goto invalid;
    }
    if (!file || !purpose) goto invalid;
    suite_file *stored = suite_new_file(s->suite, file->filename, file->mime, file->data, file->bytes);
    if (!stored) { suite_status_error(s, j, DS4_STATUS_NOMEM); return; }
    buf body = {0}; suite_file_metadata(&body, stored); suite_send(s, j, 200, body.ptr); buf_free(&body);
    return;
invalid:
    suite_error(s, j, 400, "invalid_request", DS4_SUITE_INVALID, "file");
}
static void suite_download(server *s, job *j) {
    const char *id = j->suite_request->path + strlen("/v1/files/");
    char name[64]; size_t length = strcspn(id, "/");
    if (length >= sizeof(name)) goto missing;
    memcpy(name, id, length); name[length] = 0;
    suite_file *f = suite_find_file(s->suite, name);
    if (!f) goto missing;
    if (!id[length]) {
        buf body = {0}; suite_file_metadata(&body, f); suite_send(s, j, 200, body.ptr); buf_free(&body);
        return;
    }
    if (strcmp(id + length, "/content")) goto missing;
    FILE *file = fopen(f->path, "rb");
    if (!file) goto missing;
    if (suite_headers(j->fd, s->enable_cors, 200, f->mime, f->bytes, j->suite_request->request_id)) {
        char block[65536]; size_t n;
        while ((n = fread(block, 1, sizeof(block), file)) && send_all(j->fd, block, n)) {}
    }
    fclose(file); return;
missing:
    suite_error(s, j, 404, "media_not_found", DS4_SUITE_MEDIA, "file_id");
}
static schema_json_value *suite_edit_json(suite_state *q, const suite_http *h,
                                          char transient[64][38], size_t *transients) {
    suite_part parts[72] = {0}; size_t count;
    if (!suite_multipart(h, parts, &count)) return NULL;
    buf fields = {0}, images = {0}, extension = {0};
    buf_puts(&fields, "{"); buf_puts(&images, "[");
    bool first = true, has_ds4 = false;
    for (size_t i = 0; i < count; i++) {
        suite_part *p = parts + i;
        if (!strcmp(p->name, "image") || !strcmp(p->name, "image[]")) {
            if (!*p->filename || *transients == 64 || strncmp(p->mime, "image/", 6) || !suite_mime(p->mime)) goto fail;
            suite_file *f = suite_new_file(q, p->filename, p->mime, p->data, p->bytes);
            if (!f) goto fail;
            if (*transients) buf_putc(&images, ',');
            json_escape(&images, f->id);
            snprintf(transient[(*transients)++], 38, "%s", f->id);
        } else if (!strncmp(p->name, "ds4[", 4)) {
            if (*p->filename || p->bytes > 128 || memchr(p->data, 0, p->bytes)) goto fail;
            const char *field = !strcmp(p->name, "ds4[version]") ? "version" :
                !strcmp(p->name, "ds4[seed]") ? "seed" :
                !strcmp(p->name, "ds4[sampling]") ? "sampling" : NULL;
            if (!field) goto fail;
            if (extension.len) buf_putc(&extension, ',');
            json_escape(&extension, field); buf_putc(&extension, ':');
            char *text = xstrndup(p->data, p->bytes);
            if (!strcmp(field, "sampling")) json_escape(&extension, text);
            else buf_puts(&extension, text);
            free(text);
        } else {
            if (*p->filename || p->bytes > 1048576 || memchr(p->data, 0, p->bytes)) goto fail;
            if (!first) buf_putc(&fields, ',');
            first = false;
            json_escape(&fields, p->name); buf_putc(&fields, ':');
            char *text = xstrndup(p->data, p->bytes);
            if (!strcmp(p->name, "n") || !strcmp(p->name, "ds4")) {
                schema_json_value *v = schema_json_parse(text);
                if (!v) { free(text); goto fail; }
                schema_json_free(v); buf_puts(&fields, text);
                has_ds4 |= !strcmp(p->name, "ds4");
            } else json_escape(&fields, text);
            free(text);
        }
    }
    if (!*transients) goto fail;
    if (!first) buf_putc(&fields, ',');
    buf_puts(&images, "]"); buf_puts(&fields, "\"images\":"); buf_puts(&fields, images.ptr);
    if (extension.len) {
        if (has_ds4) goto fail;
        buf_puts(&fields, ",\"ds4\":{"); buf_puts(&fields, extension.ptr); buf_putc(&fields, '}');
    } else if (!has_ds4) buf_puts(&fields, ",\"ds4\":{\"version\":1}");
    buf_putc(&fields, '}');
    schema_json_value *root = schema_json_parse(fields.ptr);
    buf_free(&fields); buf_free(&images); buf_free(&extension); return root;
fail:
    buf_free(&fields); buf_free(&images); buf_free(&extension); return NULL;
}
#include "ds4_server_suite_operations.inc"
static bool suite_route(server *s, int fd, http_request *hr) {
    bool operation = !strcmp(hr->method, "POST") &&
        (!strcmp(hr->path, "/v1/embeddings") || !strcmp(hr->path, "/v1/images/generations") ||
         !strcmp(hr->path, "/v1/images/edits") || !strcmp(hr->path, "/v1/files"));
    bool file = !strcmp(hr->method, "GET") && !strncmp(hr->path, "/v1/files/", 10);
    bool model = !strcmp(hr->method, "GET") && !strncmp(hr->path, "/v1/models/", 11) &&
        (strstr(hr->path, "embeddinggemma") || strstr(hr->path, "Qwen-Image"));
    if (!operation && !file && !model) return false;
    suite_http request = {.method = hr->method, .path = hr->path, .body = hr->body,
        .length = hr->body_len, .key = hr->idempotency_key, .content_type = hr->content_type};
    suite_id(request.request_id, "req-");
    request.deadline = now_sec() + (s->suite ? s->suite->timeout : 600);
    job j = {.fd = fd, .suite_request = &request};
    pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
    if (!s->suite) suite_error(s, &j, 503, "model_not_ready", DS4_SUITE_MODEL, "model");
    else {
        set_client_socket_nonblocking(fd);
        if (!enqueue(s, &j)) suite_status_error(s, &j, DS4_STATUS_CANCELLED);
        else wait_for_job_or_disconnect(s, &j);
    }
    pthread_cond_destroy(&j.cv); pthread_mutex_destroy(&j.mu);
    return true;
}
