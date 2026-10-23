/* Native service boundary. Included after the server's queue primitives so all
 * operations, including file expiry and receipts, have one serialized owner. */
typedef const schema_json_value sj;
static sj *suite_get(sj *v, const char *k) { return schema_json_object_get(v, k); }
static const char *suite_string(sj *v) {
    return v && v->kind == SCHEMA_JSON_STRING ? v->string : NULL;
}
static const char *suite_str(sj *v, const char *k) { return suite_string(suite_get(v, k)); }
static bool suite_fields(sj *v, const char *const *keys, size_t n) {
    if (!v || v->kind != SCHEMA_JSON_OBJECT) return false;
    for (size_t i = 0; i < v->len; i++) {
        bool found = false;
        for (size_t j = 0; j < n; j++) found |= !strcmp(v->member[i].key, keys[j]);
        for (size_t j = 0; j < i; j++)
            if (!strcmp(v->member[j].key, v->member[i].key)) return false;
        if (!found) return false;
    }
    return true;
}
#define SUITE_FIELDS(v, ...) suite_fields(v, (const char *const[]){__VA_ARGS__}, \
    sizeof((const char *const[]){__VA_ARGS__}) / sizeof(char *))
static bool suite_uint(sj *v, uint64_t *out) {
    if (!v || v->kind != SCHEMA_JSON_NUMBER || !v->string || !*v->string)
        return false;
    for (const char *p = v->string; *p; p++) if (!isdigit((unsigned char)*p)) return false;
    errno = 0;
    char *end;
    unsigned long long n = strtoull(v->string, &end, 10);
    if (errno || *end) return false;
    *out = n;
    return true;
}
static bool suite_option_uint(sj *v, const char *key, uint64_t *out) {
    sj *a = suite_get(v, key);
    return !a || suite_uint(a, out);
}
static void suite_id(char out[38], const char *prefix) {
    uint8_t raw[16];
    if (!random_bytes(raw, sizeof(raw))) die(DS4_SUITE_FAILED);
    size_t at = strlen(prefix);
    memcpy(out, prefix, at);
    for (size_t i = 0; i < sizeof(raw); i++) snprintf(out + at + i * 2, 3, "%02x", raw[i]);
}
static bool suite_headers(int fd, bool cors, int code, const char *type,
                          uint64_t bytes, const char *id) {
    buf b = {0};
    buf_printf(&b, "HTTP/1.1 %d %s\r\nContent-Length: %llu\r\nContent-Type: %s\r\n"
                   "x-request-id: %s\r\nConnection: close\r\n", code,
                   code == 200 ? "OK" : "Error", (unsigned long long)bytes, type, id);
    if (cors) append_cors_headers(&b);
    buf_puts(&b, "\r\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}
static bool suite_send(server *s, job *j, int code, const char *body) {
    const char *id = j->suite_request->request_id;
    size_t n = strlen(body);
    return suite_headers(j->fd, s->enable_cors, code, "application/json", n, id) &&
           send_all(j->fd, body, n);
}
static void suite_error(server *s, job *j, int code, const char *error,
                         const char *message, const char *param) {
    buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":"); json_escape(&b, message);
    buf_puts(&b, ",\"type\":\"invalid_request_error\",\"code\":"); json_escape(&b, error);
    buf_puts(&b, ",\"param\":"); if (param) json_escape(&b, param); else buf_puts(&b, "null");
    buf_puts(&b, "},\"ds4\":{\"version\":1,\"request_id\":");
    json_escape(&b, j->suite_request->request_id);
    buf_printf(&b, ",\"recovery_required\":%s}}", s->suite && s->suite->recovery ? "true" : "false");
    suite_send(s, j, code, b.ptr); buf_free(&b);
}
static void suite_status_error(server *s, job *j, ds4_status status) {
    server_log(DS4_LOG_DEFAULT, DS4_SUITE_STATUS, (unsigned)status, j->suite_request->path);
    if (status == DS4_STATUS_RESTORE || status == DS4_STATUS_DEVICE_LOST)
        suite_error(s, j, 503, "restore_failed", DS4_SUITE_RESTORE, NULL);
    else if (status == DS4_STATUS_NOMEM)
        suite_error(s, j, 503, "insufficient_resources", DS4_SUITE_RESOURCES, NULL);
    else if (status == DS4_STATUS_CANCELLED)
        suite_error(s, j, 409, "request_cancelled", DS4_SUITE_CANCELLED, NULL);
    else if (status == DS4_STATUS_INVALID)
        suite_error(s, j, 400, "invalid_request", DS4_SUITE_INVALID, NULL);
    else if (status == DS4_STATUS_UNSUPPORTED)
        suite_error(s, j, 400, "unsupported_parameter", DS4_SUITE_PARAMETER, NULL);
    else if (status == DS4_STATUS_CONTEXT_LENGTH)
        suite_error(s, j, 400, "context_length_exceeded", DS4_SUITE_INVALID, "input");
    else if (status == DS4_STATUS_NOT_READY || status == DS4_STATUS_BUSY)
        suite_error(s, j, 503, "model_not_ready", DS4_SUITE_MODEL, "model");
    else suite_error(s, j, 500, "inference_failed", DS4_SUITE_FAILED, NULL);
}
static bool suite_cancel(void *p) {
    job *j = p;
    return job_cancelled(j) || g_stop_requested || now_sec() >= j->suite_request->deadline;
}
static bool suite_read_headers(const char *data, size_t size, char *type,
                                size_t type_size, char *key, size_t key_size) {
    const char *p = data, *end = data + size;
    bool length_seen = false;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t n = p - line;
        if (n && line[n - 1] == '\r') n--;
        char *dest = NULL; size_t capacity = 0, prefix = 0;
        if (n >= 15 && !strncasecmp(line, "Content-Length:", 15)) {
            if (length_seen) return false;
            length_seen = true;
            const char *a = line + 15, *b = line + n;
            while (a < b && *a == ' ') a++;
            if (a == b) return false;
            for (; a < b; a++) if (!isdigit((unsigned char)*a)) return false;
        }
        if (n >= 18 && !strncasecmp(line, "Transfer-Encoding:", 18)) return false;
        if (n >= 13 && !strncasecmp(line, "Content-Type:", 13))
            dest = type, capacity = type_size, prefix = 13;
        if (n >= 16 && !strncasecmp(line, "Idempotency-Key:", 16))
            dest = key, capacity = key_size, prefix = 16;
        if (dest) {
            if (*dest) return false;
            while (prefix < n && isspace((unsigned char)line[prefix])) prefix++;
            if (n - prefix >= capacity) return false;
            memcpy(dest, line + prefix, n - prefix); dest[n - prefix] = 0;
        }
        if (p < end) p++;
    }
    return true;
}
static bool suite_write(const char *path, const void *p, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    const uint8_t *bytes = p; size_t at = 0;
    while (at < n) {
        ssize_t wrote = write(fd, bytes + at, n - at);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) break;
        at += wrote;
    }
    bool ok = at == n && !fsync(fd);
    if (close(fd)) ok = false;
    if (!ok) unlink(path);
    return ok;
}
static char *suite_read(const char *path, size_t limit, size_t *length) {
    FILE *f = fopen(path, "rb"); struct stat st;
    if (!f) return NULL;
    char *data = NULL;
    if (!fstat(fileno(f), &st) && st.st_size >= 0 && (uint64_t)st.st_size <= limit) {
        size_t n = st.st_size;
        data = malloc(n + 1);
        if (data && fread(data, 1, n, f) == n) {
            data[n] = 0; if (length) *length = n;
        } else { free(data); data = NULL; }
    }
    fclose(f); return data;
}
static void suite_expire(suite_state *q) {
    time_t now = time(NULL);
    for (size_t i = 0; i < SUITE_FILES; i++) {
        suite_file *f = q->files + i;
        if (*f->id && f->expires <= now) {
            unlink(f->path); q->stored -= f->bytes; memset(f, 0, sizeof(*f));
        }
    }
    for (size_t i = 0; i < SUITE_RECEIPTS; i++) {
        suite_receipt *r = q->receipts + i;
        if (*r->key && r->expires <= now) {
            unlink(r->path); q->stored -= r->bytes; memset(r, 0, sizeof(*r));
        }
    }
}
static suite_file *suite_find_file(suite_state *q, const char *id) {
    if (id) for (size_t i = 0; i < SUITE_FILES; i++)
        if (*q->files[i].id && !strcmp(q->files[i].id, id)) return q->files + i;
    return NULL;
}
static suite_file *suite_new_file(suite_state *q, const char *name,
                                  const char *mime, const void *data, size_t bytes) {
    if (!bytes || bytes > SUITE_UPLOAD_MAX || bytes > q->quota - q->stored) return NULL;
    for (size_t i = 0; i < SUITE_FILES; i++) if (!*q->files[i].id) {
        suite_file *f = q->files + i;
        suite_id(f->id, "file-");
        snprintf(f->path, sizeof(f->path), "%s/%s", q->directory, f->id);
        if (!suite_write(f->path, data, bytes)) { memset(f, 0, sizeof(*f)); return NULL; }
        snprintf(f->name, sizeof(f->name), "%s", name);
        snprintf(f->mime, sizeof(f->mime), "%s", mime);
        f->bytes = bytes; f->created = time(NULL); f->expires = f->created + 3600;
        q->stored += bytes;
        return f;
    }
    return NULL;
}
static ds4_status suite_decode(suite_state *q, suite_model *m, const char *id,
                                ds4_media_type type, bool audio, ds4_media **out) {
    *out = NULL;
    suite_file *f = suite_find_file(q, id);
    if (!f) return DS4_STATUS_NOT_READY;
    int fd = open(f->path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return DS4_STATUS_NOT_READY;
    void *p = mmap(NULL, f->bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return DS4_STATUS_NOMEM;
    ds4_media_decode_options options = {.limits = ds4_media_default_limits(),
        .video_audio = audio ? DS4_MEDIA_AUDIO_FIRST : DS4_MEDIA_AUDIO_EXCLUDE};
    if (type == DS4_MEDIA_VIDEO && m->processor)
        options.video = ds4_gemma_processor_video_recipe(m->processor);
    ds4_status status = ds4_media_decode(out, q->resources, 510, type, p, f->bytes, &options);
    munmap(p, f->bytes);
    if (!status && audio && !ds4_media_data(*out).has_audio_track) {
        ds4_media_free(*out); *out = NULL; return DS4_STATUS_INVALID;
    }
    f->expires = time(NULL) + 3600;
    return status;
}
static void suite_base64(buf *b, const void *bytes, size_t count) {
    const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const uint8_t *p = bytes;
    for (size_t i = 0; i < count; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16;
        if (i + 1 < count) v |= (uint32_t)p[i + 1] << 8;
        if (i + 2 < count) v |= p[i + 2];
        char out[4] = {alphabet[v >> 18], alphabet[v >> 12 & 63],
            i + 1 < count ? alphabet[v >> 6 & 63] : '=',
            i + 2 < count ? alphabet[v & 63] : '='};
        buf_append(b, out, 4);
    }
}
/* Object ordering is immaterial to a receipt; array and multipart file order
 * remain material. Exact integer lexemes avoid rounding uint64 image seeds. */
static void suite_canonical(buf *b, sj *v) {
    if (v->kind == SCHEMA_JSON_OBJECT) {
        buf_putc(b, '{'); const char *previous = NULL;
        for (size_t n = 0; n < v->len; n++) {
            size_t best = SIZE_MAX;
            for (size_t i = 0; i < v->len; i++) {
                const char *key = v->member[i].key;
                if ((!previous || strcmp(key, previous) > 0) &&
                    (best == SIZE_MAX || strcmp(key, v->member[best].key) < 0)) best = i;
            }
            if (best == SIZE_MAX) break;
            if (n) buf_putc(b, ',');
            previous = v->member[best].key;
            json_escape(b, previous); buf_putc(b, ':');
            suite_canonical(b, v->member[best].value);
        }
        buf_putc(b, '}');
    } else if (v->kind == SCHEMA_JSON_ARRAY) {
        buf_putc(b, '[');
        for (size_t i = 0; i < v->len; i++) {
            if (i) buf_putc(b, ',');
            suite_canonical(b, v->item[i]);
        }
        buf_putc(b, ']');
    } else if (v->kind == SCHEMA_JSON_NUMBER && v->string) buf_puts(b, v->string);
    else schema_json_serialize_value(b, v);
}
static ds4_status suite_drop_gemma(void *p) {
    suite_model *m = p;
    ds4_status s = ds4_gemma_multimodal_release_vision(m->embed);
    if (!s) s = ds4_gemma_multimodal_release_audio(m->embed);
    if (!s) s = ds4_gemma_multimodal_release_text(m->embed);
    return s;
}
static ds4_status suite_drop_image(void *p) {
    suite_model *m = p;
    ds4_status s = ds4_qwen_image_pipeline_close(m->image);
    if (!s) m->image = NULL;
    return s;
}
struct suite_operation {
    server *server;
    suite_state *state;
    suite_model *model;
    job *client; /* cleared when recovery outlives the HTTP request */
    ds4_engine_reclaim *reclaim;
    ds4_execution_victim victims[DS4_EXECUTOR_MAX_VICTIMS];
    size_t victim_count;
    ds4_qwen_image_job *image;
    ds4_gemma_multimodal_input *input;
    ds4_gemma_result *result;
    ds4_execution *execution;
    ds4_execution_group group;
    ds4_execution_step step;
    ds4_execution_variant variant;
    ds4_execution_request request;
    ds4_bundle_hint *hint;
    uint32_t dimensions;
    bool media_decode_error;
    double load_seconds[3], phase_seconds[4];
    uint64_t media_tokens[3];
};
static ds4_status suite_operation_close(suite_operation *o) {
    ds4_status s = ds4_bundle_hint_close(o->hint);
    if (s) return s;
    o->hint = NULL;
    s = ds4_qwen_image_job_close(o->image);
    if (s) return s;
    o->image = NULL;
    s = ds4_execution_close(o->execution);
    if (s) return s;
    o->execution = NULL;
    s = ds4_engine_reclaim_close(o->reclaim);
    if (s) return s;
    ds4_gemma_result_free(o->result);
    ds4_gemma_multimodal_input_free(o->input);
    free(o);
    return DS4_STATUS_OK;
}
static ds4_status suite_recover(server *s) {
    if (!s->suite || !s->suite->recovery) return DS4_STATUS_OK;
    pthread_mutex_lock(&s->inference_mu);
    suite_operation *o = s->suite->recovery;
    ds4_status status = DS4_STATUS_OK;
    if (o->image) (void)ds4_qwen_image_job_recover(o->image);
    else if (o->execution) (void)ds4_execution_recover(o->execution);
    /* Recover returns the original operation failure even after restoration
     * succeeded. Close is the authoritative lifetime/readiness check. */
    status = suite_operation_close(o);
    if (!status) s->suite->recovery = NULL;
    pthread_mutex_unlock(&s->inference_mu);
    return status;
}
static void suite_unavailable(server *s, job *j) {
    http_error(j->fd, s->enable_cors, 503, DS4_SUITE_RESTORE);
}
static bool suite_operation_cancel(void *p) {
    suite_operation *o = p;
    return o->client && suite_cancel(o->client);
}
static ds4_status suite_drain(void *p) {
    return ds4_numeric_fence(((suite_operation *)p)->state->numeric);
}
static ds4_status suite_noop(void *p) { (void)p; return DS4_STATUS_OK; }
static void suite_abort(void *p) {
    suite_operation *o = p;
    ds4_gemma_result_free(o->result); o->result = NULL;
}
static ds4_status suite_embed_step(void *p, uint32_t step) {
    (void)step; suite_operation *o = p;
    ds4_gemma_observer observer = {o, suite_operation_cancel, NULL};
    return ds4_gemma_multimodal_embed(o->model->embed, o->input,
                                      o->dimensions, &observer, &o->result);
}
static ds4_status suite_victims(server *s, suite_operation *o) {
    ds4_status status = ds4_engine_reclaim_create(&o->reclaim, s->engine,
                                                  o->state->directory);
    /* Unqualified legacy adapters still admit eviction-free requests. */
    if (status == DS4_STATUS_UNSUPPORTED) status = DS4_STATUS_OK;
    if (status) return status;
    size_t count = 0;
    const ds4_execution_victim *v = ds4_engine_reclaim_victims(o->reclaim, &count);
    if (count + 1 > DS4_EXECUTOR_MAX_VICTIMS) return DS4_STATUS_UNSUPPORTED;
    if (count) memcpy(o->victims, v, count * sizeof(*v));
    o->victim_count = count;
    suite_model *other = o->state->models + (o->model == o->state->models ? 1 : 0);
    uint64_t bytes = 0;
    if (other->embed) {
        ds4_gemma_residency r;
        status = ds4_gemma_multimodal_resident_weights(other->embed, &r);
        if (status) return status;
        bytes = r.text_bytes + r.vision_bytes + r.audio_bytes;
    } else if (other->image) {
        uint64_t resident[3], total[3];
        status = ds4_qwen_image_pipeline_residency(other->image, resident, total);
        if (status) return status;
        bytes = resident[0] + resident[1] + resident[2];
    }
    if (bytes) {
        ds4_execution_victim *cache = o->victims + o->victim_count++;
        *cache = (ds4_execution_victim){.kind = DS4_RECLAIM_CACHE, .context = other,
            .release = other->embed ? suite_drop_gemma : suite_drop_image, .cost = bytes};
        cache->reclaimed.bytes[DS4_MEMORY_DEVICE] = bytes;
    }
    return DS4_STATUS_OK;
}
static ds4_status suite_model_close(suite_model *m) {
    /* A later close may fail. Clear each released owner so retry is safe. */
    ds4_status status = ds4_qwen_image_pipeline_close(m->image);
    if (status) return status;
    m->image = NULL;
    status = ds4_gemma_multimodal_close(m->embed);
    if (status) return status;
    m->embed = NULL;
    ds4_gemma_processor_close(m->processor);
    m->processor = NULL;
    status = ds4_bundle_close(m->bundle);
    if (status) return status;
    m->bundle = NULL;
    free(m->asset);
    m->asset = NULL;
    return DS4_STATUS_OK;
}
static bool suite_close(server *s) {
    suite_state *q = s->suite;
    if (!q) return true;
    if (suite_recover(s)) return false;
    for (unsigned i = 0; i < 2; i++) {
        if (suite_model_close(q->models + i)) return false;
    }
    if (ds4_numeric_close(q->numeric)) return false;
    for (size_t i = 0; i < SUITE_FILES; i++) if (*q->files[i].id) unlink(q->files[i].path);
    for (size_t i = 0; i < SUITE_RECEIPTS; i++) if (*q->receipts[i].key) unlink(q->receipts[i].path);
    rmdir(q->directory); free(q); s->suite = NULL;
    return true;
}
static ds4_status suite_open(server *s, const char *path) {
    if (!path) return DS4_STATUS_OK;
    char *data = suite_read(path, 65536, NULL);
    schema_json_value *config = data ? schema_json_parse(data) : NULL;
    free(data);
    ds4_status status = DS4_STATUS_INVALID;
    if (!SUITE_FIELDS(config, "version", "media_directory", "storage_bytes", "timeout_seconds", "models")) goto done;
    uint64_t version = 0, quota = 4ull << 30, timeout = 600;
    if (!suite_uint(suite_get(config, "version"), &version) || version != 1 ||
        !suite_option_uint(config, "storage_bytes", &quota) || quota < SUITE_RECEIPT_MAX ||
        !suite_option_uint(config, "timeout_seconds", &timeout) || !timeout || timeout > 86400) goto done;
    sj *models = suite_get(config, "models");
    const char *directory = suite_str(config, "media_directory");
    if (!directory || strlen(directory) > PATH_MAX - 64 || !models ||
        models->kind != SCHEMA_JSON_ARRAY || !models->len || models->len > 2) goto done;
    suite_state *q = calloc(1, sizeof(*q));
    if (!q) { status = DS4_STATUS_NOMEM; goto done; }
    s->suite = q;
    q->runtime = ds4_engine_runtime(s->engine);
    q->resources = ds4_engine_resources(s->engine);
    q->quota = quota; q->timeout = timeout;
    snprintf(q->directory, sizeof(q->directory), "%s/ds4-suite-XXXXXX", directory);
    if (!q->runtime || !q->resources || !mkdtemp(q->directory)) goto done;
    status = ds4_numeric_cuda(&q->numeric, q->resources, 500);
    for (size_t i = 0; i < models->len && !status; i++) {
        sj *entry = models->item[i];
        if (!SUITE_FIELDS(entry, "model", "index", "root", "text_asset", "media_asset")) { status = DS4_STATUS_INVALID; break; }
        const char *id = suite_str(entry, "model"), *index = suite_str(entry, "index"),
            *root = suite_str(entry, "root"), *asset = suite_str(entry, "text_asset"),
            *media = suite_str(entry, "media_asset");
        unsigned slot;
        if (id && !strcmp(id, "Qwen/Qwen-Image-2.1-Turbo")) slot = 0;
        else if (id && !strcmp(id, "google/embeddinggemma-2")) slot = 1;
        else { status = DS4_STATUS_UNSUPPORTED; break; }
        suite_model *m = q->models + slot;
        if (!index || !root || !asset || m->bundle || (slot && !media) || (!slot && media)) { status = DS4_STATUS_INVALID; break; }
        m->asset = xstrdup(asset);
        status = ds4_bundle_open(&m->bundle, index, root, q->resources, 501 + slot);
        if (!status && strcmp(ds4_bundle_model(m->bundle), id)) status = DS4_STATUS_INVALID;
        if (!status && !slot) status = ds4_qwen_image_pipeline_open(&m->image,
            m->bundle, q->numeric, q->runtime, m->asset);
        if (!status && slot) status = ds4_gemma_processor_open(&m->processor,
            m->bundle, q->resources, 503, media, 64ull << 20);
        if (!status && slot) status = ds4_gemma_multimodal_open(&m->embed,
            m->bundle, q->numeric, q->runtime, m->asset, m->processor);
    }
done:
    schema_json_free(config);
    return status;
}
static void suite_append_model(buf *b, suite_state *q, suite_model *m) {
    bool embed = m->embed != NULL;
    bool resident = false;
    uint64_t component_bytes[3] = {0}, totals[3] = {0};
    if (embed) {
        ds4_gemma_residency r;
        if (!ds4_gemma_multimodal_resident_weights(m->embed, &r)) {
            component_bytes[0] = r.text_bytes; component_bytes[1] = r.vision_bytes; component_bytes[2] = r.audio_bytes;
            resident = r.text_bytes && r.vision_bytes && r.audio_bytes;
        }
    } else if (m->image && !ds4_qwen_image_pipeline_residency(m->image, component_bytes, totals)) {
        resident = component_bytes[0] == totals[0] && component_bytes[1] == totals[1] && component_bytes[2] == totals[2];
    }
    buf_puts(b, "{\"object\":\"model\",\"id\":"); json_escape(b, ds4_bundle_model(m->bundle));
    buf_puts(b, ",\"owned_by\":\"ds4.c\",\"created\":1792454400,\"ds4\":{\"version\":1,\"revision\":");
    json_escape(b, ds4_bundle_revision(m->bundle));
    buf_puts(b, ",\"profile\":"); json_escape(b, ds4_bundle_profile(m->bundle));
    buf_printf(b, ",\"configured\":true,\"resident\":%s,\"ready\":%s,\"qualified_backends\":[\"cuda-sm86\"],"
        "\"operations\":%s,\"input_modalities\":%s,\"context_tokens\":%u,\"embedding_dimensions\":%s,"
        "\"limits\":{\"file_bytes\":536870912,\"storage_bytes\":%llu,\"files\":256,\"receipts\":64,\"max_inflight_requests\":8,"
        "\"receipt_bytes\":67108864,\"receipt_ttl_seconds\":600,\"media_ttl_seconds\":3600,"
        "\"timeout_seconds\":%.0f,\"batch_items\":32,\"image_total_pixels\":4194304,\"reference_resolution\":0,\"output_formats\":[\"png\"]},\"components\":[",
        resident ? "true" : "false", q->recovery ? "false" : "true",
        embed ? "[\"embed\"]" : "[\"image_generate\",\"image_edit\"]",
        embed ? "[\"text\",\"image\",\"video\",\"audio\"]" : "[\"text\",\"image\"]",
        embed ? 8192 : 262144, embed ? "[768,512,256,128]" : "[]", (unsigned long long)q->quota, q->timeout);
    for (size_t i = 0; i < ds4_bundle_component_count(m->bundle); i++) {
        const ds4_bundle_component *c = ds4_bundle_component_at(m->bundle, i);
        if (i) buf_putc(b, ',');
        buf_puts(b, "{\"id\":"); json_escape(b, c->id);
        bool ready = false, optional = false;
        const char *modalities = "[\"text\"]";
        if (embed) {
            unsigned at = !strcmp(c->id, "text") ? 0 : !strcmp(c->id, "vision") ? 1 : 2;
            ready = component_bytes[at] != 0; optional = at != 0;
            modalities = at == 0 ? "[\"text\"]" : at == 1 ? "[\"image\",\"video\"]" : "[\"audio\"]";
        } else {
            unsigned at = !strcmp(c->id, "text_encoder") ? 0 : !strcmp(c->id, "transformer") ? 1 : 2;
            ready = component_bytes[at] && component_bytes[at] == totals[at];
            modalities = at == 1 ? "[\"text\",\"image\"]" : "[\"image\"]";
        }
        buf_printf(b, ",\"optional\":%s,\"input_modalities\":%s,\"storage_format\":\"safetensors\",\"compute_dtype\":\"bfloat16\","
                      "\"host_loaded\":true,\"device_resident\":%s}", optional ? "true" : "false", modalities, ready ? "true" : "false");
    }
    buf_puts(b, "]}}");
}
static void suite_append_models(server *s, buf *b) {
    if (s->suite) for (unsigned i = 0; i < 2; i++) if (s->suite->models[i].bundle) {
        buf_putc(b, ','); suite_append_model(b, s->suite, s->suite->models + i);
    }
}
static void suite_worker(server *, job *);
