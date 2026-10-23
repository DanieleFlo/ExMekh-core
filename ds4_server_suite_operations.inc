static ds4_status suite_prepare_embedding(suite_operation *o, sj *root, uint64_t *usage) {
    if (!SUITE_FIELDS(root, "model", "input", "dimensions", "encoding_format", "user", "ds4")) return DS4_STATUS_UNSUPPORTED;
    sj *extension = suite_get(root, "ds4");
    uint64_t version = 0, dimensions = 768;
    uint32_t task = DS4_GEMMA_RAW;
    const char *title = NULL;
    if (extension) {
        if (!SUITE_FIELDS(extension, "version", "task", "title", "overflow") ||
            !suite_uint(suite_get(extension, "version"), &version) || version != 1) return DS4_STATUS_INVALID;
        const char *name = suite_str(extension, "task");
        const char *names[] = {"raw", "search_query", "document", "code_query", "similarity"};
        if (suite_get(extension, "task")) {
            bool found = false;
            for (unsigned i = 0; name && i < 5; i++) if (!strcmp(name, names[i])) task = i, found = true;
            if (!found) return DS4_STATUS_INVALID;
        }
        const char *overflow = suite_str(extension, "overflow");
        if (suite_get(extension, "overflow") && (!overflow || strcmp(overflow, "error"))) return DS4_STATUS_UNSUPPORTED;
        title = suite_str(extension, "title");
        if (suite_get(extension, "title") && (!title || !*title || task != DS4_GEMMA_DOCUMENT)) return DS4_STATUS_INVALID;
    }
    const char *encoding = suite_str(root, "encoding_format");
    if (suite_get(root, "encoding_format") && (!encoding || (strcmp(encoding, "float") && strcmp(encoding, "base64")))) return DS4_STATUS_INVALID;
    if (suite_get(root, "user") && !suite_str(root, "user")) return DS4_STATUS_INVALID;
    if (!suite_option_uint(root, "dimensions", &dimensions) ||
        (dimensions != 128 && dimensions != 256 && dimensions != 512 && dimensions != 768)) return DS4_STATUS_INVALID;
    o->dimensions = dimensions;
    sj *input = suite_get(root, "input");
    if (!input) return DS4_STATUS_INVALID;
    bool scalar = input->kind == SCHEMA_JSON_STRING;
    if (!scalar && (input->kind != SCHEMA_JSON_ARRAY || !input->len)) return DS4_STATUS_INVALID;
    bool tokens = !scalar && (input->item[0]->kind == SCHEMA_JSON_NUMBER || input->item[0]->kind == SCHEMA_JSON_ARRAY);
    if (tokens) {
        bool one = input->item[0]->kind == SCHEMA_JSON_NUMBER;
        size_t count = one ? 1 : input->len;
        if (count > 32 || task || title) return DS4_STATUS_INVALID;
        uint32_t *ids[32] = {0}, lengths[32] = {0};
        ds4_status status = DS4_STATUS_OK;
        for (size_t i = 0; i < count && !status; i++) {
            sj *row = one ? input : input->item[i];
            if (row->kind != SCHEMA_JSON_ARRAY || !row->len) { status = DS4_STATUS_INVALID; break; }
            if (row->len > 8192) { status = DS4_STATUS_CONTEXT_LENGTH; break; }
            lengths[i] = row->len;
            ids[i] = calloc(row->len, 4);
            if (!ids[i]) { status = DS4_STATUS_NOMEM; break; }
            for (size_t j = 0; j < row->len; j++) {
                uint64_t token;
                if (!suite_uint(row->item[j], &token) || token >= 262144) { status = DS4_STATUS_INVALID; break; }
                ids[i][j] = token;
            }
        }
        if (!status) status = ds4_gemma_multimodal_prepare_tokens(o->model->embed,
            (const uint32_t *const *)ids, lengths, count, &o->input);
        for (size_t i = 0; i < count; i++) free(ids[i]);
        if (status) return status;
    } else {
        size_t count = scalar ? 1 : input->len;
        if (count > 32) return DS4_STATUS_INVALID;
        ds4_gemma_multimodal_request requests[32] = {0};
        ds4_gemma_content *parts = calloc(count * 64, sizeof(*parts));
        ds4_gemma_media *media[32 * 64] = {0}; size_t media_count = 0;
        buf titles[32] = {0};
        if (!parts) return DS4_STATUS_NOMEM;
        ds4_status status = DS4_STATUS_OK;
        for (size_t i = 0; i < count && !status; i++) {
            sj *item = scalar ? input : input->item[i];
            if (!scalar && item->kind != input->item[0]->kind) { status = DS4_STATUS_INVALID; break; }
            bool text = item->kind == SCHEMA_JSON_STRING;
            sj *content = NULL;
            if (!text) {
                if (!extension || !SUITE_FIELDS(item, "content")) { status = DS4_STATUS_INVALID; break; }
                content = suite_get(item, "content");
                if (!content || content->kind != SCHEMA_JSON_ARRAY || !content->len || content->len > 64) { status = DS4_STATUS_INVALID; break; }
            }
            requests[i] = (ds4_gemma_multimodal_request){.content = parts + i * 64,
                .count = text ? 1 : content->len, .task = task, .add_timestamps = true};
            bool title_added = false;
            for (size_t j = 0; j < requests[i].count && !status; j++) {
                sj *part = text ? item : content->item[j];
                const char *type = text ? "text" : suite_str(part, "type");
                ds4_gemma_content *dest = parts + i * 64 + j;
                if (type && !strcmp(type, "text")) {
                    if (!text && !SUITE_FIELDS(part, "type", "text")) { status = DS4_STATUS_UNSUPPORTED; break; }
                    dest->text = text ? suite_string(item) : suite_str(part, "text");
                    if (!dest->text || !*dest->text || strlen(dest->text) > 1048576 ||
                        (task && (!strncmp(dest->text, "task: ", 6) || !strncmp(dest->text, "title: ", 7)))) { status = DS4_STATUS_INVALID; break; }
                    if (title && !title_added) {
                        buf_puts(titles + i, "title: "); buf_puts(titles + i, title);
                        buf_puts(titles + i, " | text: "); buf_puts(titles + i, dest->text);
                        dest->text = titles[i].ptr; title_added = true;
                    }
                } else {
                    ds4_media_type kind = type && !strcmp(type, "image") ? DS4_MEDIA_IMAGE :
                        type && !strcmp(type, "video") ? DS4_MEDIA_VIDEO :
                        type && !strcmp(type, "audio") ? DS4_MEDIA_AUDIO : 0;
                    const char *audio = suite_str(part, "audio");
                    if (!kind || !SUITE_FIELDS(part, "type", "file_id", "audio") ||
                        (suite_get(part, "audio") && (kind != DS4_MEDIA_VIDEO || !audio ||
                         (strcmp(audio, "exclude") && strcmp(audio, "include"))))) { status = DS4_STATUS_INVALID; break; }
                    ds4_media *decoded = NULL;
                    status = suite_decode(o->state, o->model, suite_str(part, "file_id"), kind,
                        audio && !strcmp(audio, "include"), &decoded);
                    if (status && status != DS4_STATUS_NOT_READY && status != DS4_STATUS_NOMEM) o->media_decode_error = true;
                    if (!status) status = ds4_gemma_media_prepare(o->model->processor, decoded,
                        NULL, media + media_count);
                    ds4_media_free(decoded);
                    if (!status) dest->media = media[media_count++];
                }
            }
            if (title && !title_added) status = DS4_STATUS_INVALID;
        }
        if (!status) status = ds4_gemma_multimodal_prepare(o->model->embed, requests, count, &o->input);
        for (size_t i = 0; i < media_count; i++) ds4_gemma_media_free(media[i]);
        for (size_t i = 0; i < count; i++) buf_free(titles + i);
        free(parts);
        if (status) return status;
    }
    for (size_t i = 0;; i++) {
        const uint32_t *ids; uint32_t count;
        if (ds4_gemma_multimodal_input_tokens(o->input, i, &ids, &count)) break;
        *usage += count;
        for (uint32_t j = 0; j < count; j++) {
            o->media_tokens[0] += ids[j] == 258880;
            o->media_tokens[1] += ids[j] == 258884;
            o->media_tokens[2] += ids[j] == 258881;
        }
    }
    return DS4_STATUS_OK;
}
static ds4_status suite_run_embedding(suite_operation *o, buf *body, sj *root,
                                       uint64_t usage, ds4_execution_plan *plan) {
    ds4_runtime_estimate estimate;
    ds4_status status = ds4_gemma_multimodal_estimate(o->model->embed, o->input, &estimate);
    if (status) return status;
    ds4_gemma_plan native_plan;
    /* Admit a bounded native plan before considering LLM victims. The inner
     * adapter makes the same read-only decision under the operation lease. */
    ds4_status bounded = ds4_gemma_multimodal_plan(o->model->embed, o->input, &native_plan);
    if (!bounded && !native_plan.reclaim_device_bytes) estimate = native_plan.peak;
    o->group = (ds4_execution_group){.context = o->model, .load = suite_noop, .release = suite_drop_gemma};
    o->step = (ds4_execution_step){.id = 0, .groups = 1, .retain = 1};
    o->step.working.bytes[DS4_MEMORY_DEVICE] = estimate.device_bytes;
    o->step.working.bytes[DS4_MEMORY_HOST] = estimate.host_bytes;
    /* Session capacity excludes future LLM prefill/vision transients. Until
     * that working set is bounded, retain Gemma weights only inside this
     * request; publication must leave the original LLM headroom available. */
    o->variant = (ds4_execution_variant){.steps = &o->step, .step_count = 1};
    o->request = (ds4_execution_request){.groups = &o->group, .group_count = 1,
        .variants = &o->variant, .variant_count = 1, .victims = o->victims, .victim_count = o->victim_count};
    ds4_execution_ops ops = {.context = o, .execute = suite_embed_step, .drain = suite_drain,
        .commit = suite_noop, .abort = suite_abort, .cancelled = suite_operation_cancel};
    status = ds4_execution_create(&o->execution, o->state->runtime, o->state->resources, 500, &o->request, &ops);
    if (!status) status = ds4_execution_run(o->execution);
    if (status) return status;
    *plan = *ds4_execution_get_plan(o->execution);
    ds4_gemma_result_view view = ds4_gemma_result_data(o->result);
    const char *encoding = suite_str(root, "encoding_format");
    buf_puts(body, "{\"object\":\"list\",\"model\":"); json_escape(body, ds4_bundle_model(o->model->bundle));
    buf_printf(body, ",\"usage\":{\"prompt_tokens\":%llu,\"total_tokens\":%llu},\"data\":[",
        (unsigned long long)usage, (unsigned long long)usage);
    for (uint32_t i = 0; i < view.count; i++) {
        if (i) buf_putc(body, ',');
        buf_printf(body, "{\"object\":\"embedding\",\"index\":%u,\"embedding\":", i);
        const float *v = view.vectors + (size_t)i * view.dimensions;
        if (encoding && !strcmp(encoding, "base64")) {
            buf_putc(body, '"'); suite_base64(body, v, view.dimensions * 4); buf_putc(body, '"');
        } else {
            buf_putc(body, '[');
            for (uint32_t d = 0; d < view.dimensions; d++) buf_printf(body, "%s%.9g", d ? "," : "", v[d]);
            buf_putc(body, ']');
        }
        buf_putc(body, '}');
    }
    buf_puts(body, "]");
    return DS4_STATUS_OK;
}
typedef struct {
    ds4_qwen_image_request native;
    ds4_media *media[64];
    ds4_qwen_image_reference references[64];
    uint32_t count;
    bool opaque;
    const char *background;
} suite_image_request;
static void suite_image_request_free(suite_image_request *r) {
    for (size_t i = 0; i < 64; i++) ds4_media_free(r->media[i]);
}
static ds4_status suite_prepare_image(suite_operation *o, sj *root, bool edit,
                                       suite_image_request *r) {
    if (!SUITE_FIELDS(root, "model", "prompt", "images", "n", "size", "output_format", "background", "ds4")) return DS4_STATUS_UNSUPPORTED;
    const char *prompt = suite_str(root, "prompt"), *size = suite_str(root, "size"),
        *format = suite_str(root, "output_format"), *background = suite_str(root, "background");
    if (!prompt || (!*prompt && !edit) || strlen(prompt) > 1048576) return DS4_STATUS_INVALID;
    if (suite_get(root, "output_format") && (!format || strcmp(format, "png"))) return DS4_STATUS_UNSUPPORTED;
    if (suite_get(root, "background") && (!background || (strcmp(background, "auto") && strcmp(background, "opaque") && strcmp(background, "transparent")))) return DS4_STATUS_INVALID;
    r->opaque = background && !strcmp(background, "opaque");
    r->background = background ? background : "auto";
    uint64_t n = 1, seed;
    if (!random_bytes(&seed, sizeof(seed)) || !suite_option_uint(root, "n", &n) || !n || n > 8) return DS4_STATUS_INVALID;
    sj *extension = suite_get(root, "ds4");
    if (extension) {
        uint64_t version;
        if (!SUITE_FIELDS(extension, "version", "seed", "sampling") ||
            !suite_uint(suite_get(extension, "version"), &version) || version != 1 ||
            !suite_option_uint(extension, "seed", &seed)) return DS4_STATUS_INVALID;
        const char *sampling = suite_str(extension, "sampling");
        if (suite_get(extension, "sampling") && (!sampling || strcmp(sampling, "checkpoint"))) return DS4_STATUS_UNSUPPORTED;
    }
    if (seed > UINT64_MAX - n + 1) return DS4_STATUS_INVALID;
    uint64_t width = 1024, height = 1024;
    if (suite_get(root, "size")) {
        char *end;
        if (!size || *size < '1' || *size > '9') return DS4_STATUS_INVALID;
        errno = 0; width = strtoull(size, &end, 10);
        if (errno || width > 2048 || *end != 'x' || end[1] < '1' || end[1] > '9') return DS4_STATUS_INVALID;
        height = strtoull(end + 1, &end, 10);
        if (errno || *end) return DS4_STATUS_INVALID;
    }
    if (width < 32 || height < 32 || width > 2048 || height > 2048 || width % 32 || height % 32 ||
        (uint64_t)n * width * height > 4194304) return DS4_STATUS_INVALID;
    r->native = (ds4_qwen_image_request){.prompt = prompt, .width = width, .height = height, .seed = seed};
    r->count = n;
    sj *images = suite_get(root, "images");
    if (edit) {
        if (!extension || !images || images->kind != SCHEMA_JSON_ARRAY || !images->len || images->len > 64) return DS4_STATUS_INVALID;
        for (size_t i = 0; i < images->len; i++) {
            ds4_status s = suite_decode(o->state, o->model, suite_string(images->item[i]), DS4_MEDIA_IMAGE, false, r->media + i);
            if (s && s != DS4_STATUS_NOT_READY && s != DS4_STATUS_NOMEM) o->media_decode_error = true;
            if (s) return s;
            ds4_media_frame_view frame;
            s = ds4_media_frame(r->media[i], 0, &frame);
            if (s) return s;
            r->references[i] = (ds4_qwen_image_reference){frame.rgb, frame.width, frame.height, 3};
        }
        r->native.references = r->references; r->native.reference_count = images->len;
    } else if (images) return DS4_STATUS_UNSUPPORTED;
    return ds4_qwen_image_request_validate(&r->native);
}
static ds4_status suite_run_images(suite_operation *o, suite_image_request *r, buf *body,
                                    ds4_execution_plan *plan) {
    suite_state *q = o->state;
    if (!o->model->image) {
        ds4_status s = ds4_qwen_image_pipeline_open(&o->model->image, o->model->bundle,
            q->numeric, q->runtime, o->model->asset);
        if (s) return s;
    }
    buf_printf(body, "{\"created\":%lld,\"data\":[", (long long)time(NULL));
    for (uint32_t i = 0; i < r->count; i++) {
        ds4_qwen_image_observer observer = {o, suite_operation_cancel, NULL};
#ifdef DS4_SUITE_TEST_PHASE_OBSERVER
        observer.trace = DS4_SUITE_TEST_PHASE_OBSERVER;
#endif
        ds4_status status = ds4_qwen_image_job_create(&o->image, o->model->image,
            &r->native, o->victims, o->victim_count, &observer);
        if (!status) status = ds4_qwen_image_job_run(o->image);
        if (status) return status;
        const ds4_qwen_image_report *report = ds4_qwen_image_job_report(o->image);
        for (unsigned phase = 0; phase < 4; phase++) o->phase_seconds[phase] += report->phase_seconds[phase];
        for (unsigned phase = 0; phase < 3; phase++) o->load_seconds[phase] += report->load_seconds[phase];
        *plan = report->plan;
        ds4_qwen_image_rgba *rgba = NULL;
        status = ds4_qwen_image_job_take(o->image, &rgba);
        char path[PATH_MAX]; snprintf(path, sizeof(path), "%s/output.png", q->directory);
        if (!status && r->opaque) status = ds4_qwen_image_rgba_opaque(rgba);
        if (!status) status = ds4_qwen_image_rgba_save_png(rgba, path);
        ds4_qwen_image_rgba_free(rgba);
        size_t bytes = 0;
        char *png = status ? NULL : suite_read(path, 32ull << 20, &bytes);
        unlink(path);
        if (!status && !png) status = DS4_STATUS_NOMEM;
        suite_file *f = status ? NULL : suite_new_file(q, "image.png", "image/png", png, bytes);
        if (!status && !f) status = DS4_STATUS_NOMEM;
        if (!status) {
            if (i) buf_putc(body, ',');
            buf_puts(body, "{\"b64_json\":\""); suite_base64(body, png, bytes);
            buf_puts(body, "\",\"ds4\":{\"file_id\":"); json_escape(body, f->id);
            buf_printf(body, ",\"seed\":%llu,\"width\":%u,\"height\":%u,\"recipe\":{\"version\":1,\"rng_version\":1,"
                "\"steps\":8,\"guidance\":1,\"reference_resolution\":0,\"background\":\"%s\"}}}",
                (unsigned long long)r->native.seed, r->native.width, r->native.height, r->background);
        }
        free(png);
        if (status) return status;
        status = ds4_qwen_image_job_close(o->image);
        if (status) return status;
        o->image = NULL;
        if (i + 1 < r->count) {
            r->native.seed++;
            status = ds4_engine_reclaim_close(o->reclaim);
            if (status) return status;
            o->reclaim = NULL; o->victim_count = 0;
            memset(o->victims, 0, sizeof(o->victims));
            status = suite_victims(o->server, o);
            if (status) return status;
        }
    }
    buf_putc(body, ']'); return DS4_STATUS_OK;
}
static bool suite_receipt_lookup(server *s, job *j, sj *root, suite_receipt **receipt) {
    *receipt = NULL;
    if (!*j->suite_request->key) return false;
    suite_state *q = s->suite;
    buf canonical = {0};
    suite_canonical(&canonical, root);
    /* Multipart normalization creates temporary IDs. Hash their actual bytes
     * so SDK retries with a new MIME boundary still identify the same edit. */
    sj *images = suite_get(root, "images");
    if (images && images->kind == SCHEMA_JSON_ARRAY) {
        schema_json_value *copy = schema_json_parse(canonical.ptr);
        schema_json_value *refs = (schema_json_value *)suite_get(copy, "images");
        for (size_t i = 0; i < images->len; i++) {
            suite_file *f = suite_find_file(q, suite_string(images->item[i]));
            if (!f || !refs) continue;
            int fd = open(f->path, O_RDONLY | O_CLOEXEC);
            void *data = fd < 0 ? MAP_FAILED : mmap(NULL, f->bytes, PROT_READ, MAP_PRIVATE, fd, 0);
            if (fd >= 0) close(fd);
            if (data != MAP_FAILED) {
                ds4_sha256 digest; uint8_t value[32]; char hex[65];
                ds4_sha256_init(&digest); ds4_sha256_update(&digest, data, f->bytes);
                ds4_sha256_final(&digest, value); munmap(data, f->bytes);
                for (size_t j = 0; j < 32; j++) snprintf(hex + j * 2, 3, "%02x", value[j]);
                free(refs->item[i]->string); refs->item[i]->string = xstrdup(hex);
            }
        }
        buf_free(&canonical); suite_canonical(&canonical, copy); schema_json_free(copy);
    }
    ds4_sha256 hash;
    uint8_t fingerprint[32]; ds4_sha256_init(&hash);
    ds4_sha256_update(&hash, j->suite_request->path, strlen(j->suite_request->path) + 1);
    ds4_sha256_update(&hash, canonical.ptr, canonical.len);
    ds4_sha256_final(&hash, fingerprint); buf_free(&canonical);
    suite_receipt *empty = NULL;
    for (size_t i = 0; i < SUITE_RECEIPTS; i++) {
        suite_receipt *r = q->receipts + i;
        if (!*r->key) { if (!empty) empty = r; continue; }
        if (strcmp(r->key, j->suite_request->key)) continue;
        if (memcmp(fingerprint, r->hash, 32)) suite_error(s, j, 409, "idempotency_conflict", DS4_SUITE_CONFLICT, NULL);
        else {
            snprintf(j->suite_request->request_id, 38, "%s", r->request_id);
            char *body = suite_read(r->path, SUITE_RECEIPT_MAX, NULL);
            if (body) { suite_send(s, j, r->code, body); free(body); }
            else suite_error(s, j, 503, "model_not_ready", DS4_SUITE_FAILED, NULL);
        }
        return true;
    }
    if (!empty || q->quota - q->stored < SUITE_RECEIPT_MAX) {
        suite_status_error(s, j, DS4_STATUS_NOMEM); return true;
    }
    memcpy(empty->hash, fingerprint, 32);
    *receipt = empty;
    return false;
}
/* Detailed ledgers are for qualification. Keeping them opt-in prevents
 * diagnostic arrays from consuming the agent's conversation context. */
static void suite_append_resources(buf *body, suite_state *q) {
    const char *enabled = getenv("DS4_SUITE_RESOURCE_METRICS");
    if (!enabled || strcmp(enabled, "1")) return;
    buf_puts(body, ",\"resources\":{\"locations\":[\"device\",\"mmap\",\"resident\",\"host\",\"pinned\",\"checkpoint\",\"managed\"],\"owners\":[");
    const uint64_t owners[] = {0, 500, 501, 502, 503};
    for (unsigned i = 0; i < sizeof(owners) / sizeof(*owners); i++) {
        ds4_resource_totals totals;
        ds4_resources_totals(q->resources, owners[i], &totals);
        buf_printf(body, "%s{\"owner\":%llu,\"roots\":%zu,\"views\":%zu,\"allocated\":[",
            i ? "," : "", (unsigned long long)owners[i], totals.roots, totals.views);
        for (unsigned j = 0; j < DS4_MEMORY_COUNT; j++)
            buf_printf(body, "%s%llu", j ? "," : "", (unsigned long long)totals.allocated[j]);
        buf_puts(body, "],\"live\":[");
        for (unsigned j = 0; j < DS4_MEMORY_COUNT; j++)
            buf_printf(body, "%s%llu", j ? "," : "", (unsigned long long)totals.live[j]);
        buf_puts(body, "],\"reserved\":[");
        for (unsigned j = 0; j < DS4_MEMORY_COUNT; j++)
            buf_printf(body, "%s%llu", j ? "," : "", (unsigned long long)totals.reserved[j]);
        buf_puts(body, "]}");
    }
    buf_puts(body, "]}");
}
static void suite_worker(server *s, job *j) {
    pthread_mutex_lock(&s->inference_mu);
    suite_state *q = s->suite;
    suite_http *h = j->suite_request;
    if (suite_recover(s)) { suite_status_error(s, j, DS4_STATUS_RESTORE); goto done; }
    suite_expire(q);
    if (suite_cancel(j)) { suite_status_error(s, j, DS4_STATUS_CANCELLED); goto done; }
    if (!strcmp(h->method, "GET")) {
        if (!strncmp(h->path, "/v1/files/", 10)) suite_download(s, j);
        else {
            const char *id = h->path + 11;
            bool found = false;
            for (unsigned i = 0; i < 2; i++) if (q->models[i].bundle && !strcmp(id, ds4_bundle_model(q->models[i].bundle))) {
                buf body = {0}; suite_append_model(&body, q, q->models + i); suite_send(s, j, 200, body.ptr); buf_free(&body); found = true;
            }
            if (!found) suite_error(s, j, 404, "model_not_found", DS4_SUITE_MODEL, "model");
        }
        goto done;
    }
    if (!strcmp(h->path, "/v1/files")) { suite_upload(s, j); goto done; }
    char transient[64][38]; size_t transients = 0;
    bool edit = !strcmp(h->path, "/v1/images/edits");
    bool embed = !strcmp(h->path, "/v1/embeddings");
    schema_json_value *root = edit && !strncmp(h->content_type, "multipart/form-data;", 20)
        ? suite_edit_json(q, h, transient, &transients)
        : (h->body && !memchr(h->body, 0, h->length) && !strstr(h->body, "\\u0000") ? schema_json_parse(h->body) : NULL);
    suite_operation *o = NULL;
    suite_image_request image = {0};
    buf body = {0}; suite_receipt *receipt = NULL;
    if (!root || root->kind != SCHEMA_JSON_OBJECT) { suite_status_error(s, j, DS4_STATUS_INVALID); goto cleanup; }
    const char *model = suite_str(root, "model");
    suite_model *m = q->models + (embed ? 1 : 0);
    if (!model || !m->bundle || strcmp(model, ds4_bundle_model(m->bundle))) {
        bool known = false;
        for (unsigned i = 0; i < 2; i++) known |= q->models[i].bundle && model && !strcmp(model, ds4_bundle_model(q->models[i].bundle));
        suite_error(s, j, known ? 400 : 404, known ? "unsupported_operation" : "model_not_found", DS4_SUITE_MODEL, "model"); goto cleanup;
    }
    o = calloc(1, sizeof(*o));
    if (!o) { suite_status_error(s, j, DS4_STATUS_NOMEM); goto cleanup; }
    o->model = m; o->state = q; o->client = j; o->server = s;
    uint64_t usage = 0;
    double start = now_sec();
    ds4_gemma_cache_stats cache_before = m->processor ? ds4_gemma_processor_cache(m->processor) : (ds4_gemma_cache_stats){0};
    ds4_status status = embed ? suite_prepare_embedding(o, root, &usage) : suite_prepare_image(o, root, edit, &image);
    if (status) {
        if (status == DS4_STATUS_NOT_READY) suite_error(s, j, 404, "media_not_found", DS4_SUITE_MEDIA, "file_id");
        else if (o->media_decode_error) suite_error(s, j, 400, "unsupported_media_format", DS4_SUITE_MEDIA, "file_id");
        else if (status == DS4_STATUS_UNSUPPORTED) suite_error(s, j, 400, "unsupported_parameter", DS4_SUITE_PARAMETER, NULL);
        else suite_status_error(s, j, status);
        goto cleanup;
    }
    if (suite_receipt_lookup(s, j, root, &receipt)) goto cleanup;
    /* Bound output storage before any eviction or compute, including a full
     * receipt plus PNG bytes. No GPU result is retained by the HTTP client. */
    uint64_t output_reserve = embed ? SUITE_RECEIPT_MAX : SUITE_RECEIPT_MAX +
        (uint64_t)image.count * (image.native.width * image.native.height * 5ull + 65536);
    size_t free_files = 0;
    for (size_t i = 0; i < SUITE_FILES; i++) free_files += !*q->files[i].id;
    if (q->quota - q->stored < output_reserve || (!embed && free_files < image.count)) {
        suite_status_error(s, j, DS4_STATUS_NOMEM); goto cleanup;
    }
    /* A complete, validated API operation is a structured hint. Partial LLM
     * output never reaches this path. Advisory CPU I/O is cancellable and
     * bounded to 16 MiB of the configured bundle, with no GPU allocation. */
    (void)ds4_bundle_hint_start(&o->hint, m->bundle, 0, 16u << 20);
    status = suite_victims(s, o);
    uint64_t h2d_before, d2h_before;
    ds4_numeric_transfers(q->numeric, &h2d_before, &d2h_before);
    ds4_execution_plan plan = {0};
    if (!status) status = embed ? suite_run_embedding(o, &body, root, usage, &plan) : suite_run_images(o, &image, &body, &plan);
    if (status) {
        ds4_status close = suite_operation_close(o);
        if (close) { o->client = NULL; q->recovery = o; status = DS4_STATUS_RESTORE; }
        o = NULL; suite_status_error(s, j, status); goto cleanup;
    }
    buf_puts(&body, ",\"ds4\":{\"version\":1,\"request_id\":"); json_escape(&body, h->request_id);
    buf_puts(&body, ",\"revision\":"); json_escape(&body, ds4_bundle_revision(m->bundle));
    buf_puts(&body, ",\"profile\":"); json_escape(&body, ds4_bundle_profile(m->bundle));
    ds4_gemma_cache_stats cache = m->processor ? ds4_gemma_processor_cache(m->processor) : (ds4_gemma_cache_stats){0};
    uint64_t h2d_after, d2h_after;
    ds4_numeric_transfers(q->numeric, &h2d_after, &d2h_after);
    uint64_t moved = 0;
    for (size_t i = 0; i < o->victim_count; i++) if (plan.victims & (1u << i)) moved += o->victims[i].reclaimed.bytes[DS4_MEMORY_DEVICE];
    buf_printf(&body, ",\"plan\":{\"variant\":%zu,\"victims\":%u},\"reclaimed_device_bytes\":%llu,"
        "\"cache_hits\":%llu,\"total_seconds\":%.9f,\"usage\":{\"processed_tokens\":%llu,"
        "\"image_soft_tokens\":%llu,\"video_soft_tokens\":%llu,\"audio_soft_tokens\":%llu},"
        "\"transfers\":{\"native_h2d_bytes\":%llu,\"native_d2h_bytes\":%llu}",
        plan.variant, plan.victims, (unsigned long long)moved, (unsigned long long)(cache.hits - cache_before.hits),
        now_sec() - start, (unsigned long long)usage,
        (unsigned long long)o->media_tokens[0], (unsigned long long)o->media_tokens[1], (unsigned long long)o->media_tokens[2],
        (unsigned long long)(h2d_after - h2d_before), (unsigned long long)(d2h_after - d2h_before));
    if (embed) buf_puts(&body, ",\"load_seconds\":null,\"phase_seconds\":null");
    else buf_printf(&body, ",\"load_seconds\":[%.9f,%.9f,%.9f],\"phase_seconds\":[%.9f,%.9f,%.9f,%.9f]",
        o->load_seconds[0], o->load_seconds[1], o->load_seconds[2],
        o->phase_seconds[0], o->phase_seconds[1], o->phase_seconds[2], o->phase_seconds[3]);
    status = suite_operation_close(o);
    if (status) { o->client = NULL; q->recovery = o; o = NULL; suite_status_error(s, j, DS4_STATUS_RESTORE); goto cleanup; }
    o = NULL;
    /* Snapshot after request cleanup. Host media caches have their own owner;
     * native buffers, legacy weights and unowned global memory stay distinct. */
    suite_append_resources(&body, q);
    buf_puts(&body, "}}");
    if (body.len > SUITE_RECEIPT_MAX) { suite_status_error(s, j, DS4_STATUS_NOMEM); goto cleanup; }
    if (receipt) {
        char id[38]; suite_id(id, "rcp-");
        snprintf(receipt->path, sizeof(receipt->path), "%s/%s", q->directory, id);
        snprintf(receipt->key, sizeof(receipt->key), "%s", h->key);
        receipt->expires = time(NULL) + 600; receipt->code = 200;
        snprintf(receipt->request_id, 38, "%s", h->request_id);
        if (!suite_write(receipt->path, body.ptr, body.len)) {
            /* The receipt remains a tombstone until expiry: a transport retry
             * cannot run a completed generation after a disk write failure. */
            suite_status_error(s, j, DS4_STATUS_NOMEM); goto cleanup;
        }
        receipt->bytes = body.len; q->stored += body.len;
    }
    suite_send(s, j, 200, body.ptr);
cleanup:
    if (o) {
        ds4_status close = suite_operation_close(o);
        if (close) { o->client = NULL; q->recovery = o; }
    }
    suite_image_request_free(&image);
    schema_json_free(root); buf_free(&body);
    for (size_t i = 0; i < transients; i++) {
        suite_file *f = suite_find_file(q, transient[i]);
        if (f) { unlink(f->path); q->stored -= f->bytes; memset(f, 0, sizeof(*f)); }
    }
done:
    pthread_mutex_unlock(&s->inference_mu);
}
