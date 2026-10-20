/* Family internals stay inside ds4.c. The generic resource/checkpoint modules
 * have no knowledge of tensor layouts, sessions, model names or HTTP. */
#include <sys/statvfs.h>

struct ds4_session_checkpoint {
    ds4_session_payload_file file;
    bool mtp;
    ds4_resource *account;
};

static bool lifecycle_supported(const ds4_engine *e) {
#if defined(DS4_NO_GPU) || defined(__APPLE__) || defined(DS4_ROCM_BUILD)
    (void)e; return false;
#else
    return e && e->backend == DS4_BACKEND_CUDA &&
        e->shape.family == DS4_MODEL_FAMILY_QWEN35 &&
        g_qwen_file_type == DS4_QWEN_FTYPE_Q4_K_S && !e->ssd_streaming &&
        g_n_gpus == 1 && !e->multi_tier && !e->cuda_tensor_parallel && !e->tp.active &&
        e->distributed.role == DS4_DISTRIBUTED_NONE;
#endif
}

ds4_resources *ds4_engine_resources(ds4_engine *e) {
#if defined(DS4_NO_GPU) || defined(__APPLE__) || defined(DS4_ROCM_BUILD)
    (void)e; return NULL;
#else
    return e && e->backend == DS4_BACKEND_CUDA && g_suite_gpu_ready ? ds4_gpu_resources() : NULL;
#endif
}

static bool lifecycle_identity(ds4_engine *e) {
    ds4_model *models[] = {&e->model, &e->mtp_model, &e->vision_model};
    struct stat files[3] = {{0}};
    for (unsigned i = 0; i < 3; i++) {
        if (!models[i]->map) continue;
        if (fstat(models[i]->fd, &files[i])) return false;
        if (e->lifecycle_identity_ready) {
            const struct stat *old = &e->lifecycle_files[i], *now = &files[i];
            if (old->st_dev != now->st_dev || old->st_ino != now->st_ino ||
                old->st_size != now->st_size || old->st_mtime != now->st_mtime ||
                old->st_ctime != now->st_ctime ||
#ifdef __APPLE__
                old->st_mtimespec.tv_nsec != now->st_mtimespec.tv_nsec ||
                old->st_ctimespec.tv_nsec != now->st_ctimespec.tv_nsec)
#else
                old->st_mtim.tv_nsec != now->st_mtim.tv_nsec ||
                old->st_ctim.tv_nsec != now->st_ctim.tv_nsec)
#endif
                return false;
        }
    }
    if (e->lifecycle_identity_ready) return true;
    ds4_sha256 sha; ds4_sha256_init(&sha);
    for (unsigned i = 0; i < 3; i++) {
        ds4_sha256_update(&sha, &models[i]->size, sizeof(models[i]->size));
        if (!models[i]->map) continue;
        /* Bind to the opened artifact and its complete GGUF directory. Full
         * content hashes belong to the artifact manifest; scanning 15 GiB on
         * the first live vision request would defeat selective suspension.
         * ctime (including nanoseconds) detects in-place writes even if mtime
         * is reset. Relocated copies deliberately require a fresh checkpoint. */
        if (models[i]->tensor_data_pos > models[i]->size) return false;
        uint64_t identity[] = {(uint64_t)files[i].st_dev, (uint64_t)files[i].st_ino,
            (uint64_t)files[i].st_size, (uint64_t)files[i].st_mtime, (uint64_t)files[i].st_ctime,
#ifdef __APPLE__
            (uint64_t)files[i].st_mtimespec.tv_nsec, (uint64_t)files[i].st_ctimespec.tv_nsec};
#else
            (uint64_t)files[i].st_mtim.tv_nsec, (uint64_t)files[i].st_ctim.tv_nsec};
#endif
        ds4_sha256_update(&sha, identity, sizeof(identity));
        ds4_sha256_update(&sha, models[i]->map, models[i]->tensor_data_pos);
    }
    ds4_sha256_final(&sha, e->lifecycle_identity);
    memcpy(e->lifecycle_files, files, sizeof(files));
    e->lifecycle_identity_ready = true;
    return lifecycle_identity(e); /* Detect a file changed during the scan. */
}

static void lifecycle_config(ds4_session *s, bool mtp, uint8_t hash[32]) {
    const ds4_engine *e = s->engine;
    uint32_t values[] = {DS4_SUITE_ABI, (uint32_t)s->ctx_size, s->prefill_cap,
        e->shape.family, (uint32_t)e->backend, e->quality, (uint32_t)e->support_kind,
        (uint32_t)e->mtp_draft_tokens, mtp};
    ds4_sha256 sha; ds4_sha256_init(&sha);
    ds4_sha256_update(&sha, values, sizeof(values));
    ds4_sha256_update(&sha, &e->mtp_margin, sizeof(e->mtp_margin));
    ds4_sha256_final(&sha, hash);
}

const char *ds4_session_checkpoint_path(const ds4_session_checkpoint *p) { return p ? p->file.path : NULL; }
void ds4_session_checkpoint_free(ds4_session_checkpoint *p) {
    if (!p) return;
    ds4_session_payload_file_free(&p->file);
    (void)ds4_resource_release(p->account);
    free(p);
}

static ds4_status lifecycle_checkpoint_capture(ds4_session *s, const char *directory,
        const void *metadata, size_t metadata_bytes, ds4_session_checkpoint **out) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    if (!s || !directory || !directory[0] || (metadata_bytes && !metadata)) return DS4_STATUS_INVALID;
    if (!lifecycle_supported(s->engine)) return DS4_STATUS_UNSUPPORTED;
    if (s->lifecycle_parked) return DS4_STATUS_NOT_RESIDENT;
    if (!s->checkpoint_valid) return DS4_STATUS_NOT_READY;
    if (!lifecycle_identity(s->engine)) return DS4_STATUS_INVALID;
    uint64_t body = ds4_session_payload_bytes(s);
    if (!body || body > UINT64_MAX - DS4_CHECKPOINT_HEADER_BYTES ||
        metadata_bytes > UINT64_MAX - body - DS4_CHECKPOINT_HEADER_BYTES) return DS4_STATUS_INVALID;
    uint64_t total = body + DS4_CHECKPOINT_HEADER_BYTES + metadata_bytes;
    ds4_resources *registry = ds4_engine_resources(s->engine);
    struct statvfs disk;
    if (statvfs(directory, &disk) || (disk.f_frsize && disk.f_bavail > UINT64_MAX / disk.f_frsize))
        return DS4_STATUS_NOMEM;
    (void)ds4_resources_reconcile(registry, DS4_MEMORY_CHECKPOINT,
        (uint64_t)disk.f_bavail * disk.f_frsize, true);
    uint64_t reservation[DS4_MEMORY_COUNT] = {0};
    reservation[DS4_MEMORY_CHECKPOINT] = total;
    ds4_reservation *lease = NULL;
    ds4_status status = ds4_resources_reserve(registry, (uint64_t)(uintptr_t)s, reservation, &lease);
    if (status) return status;
    ds4_session_checkpoint *p = calloc(1, sizeof(*p));
    size_t length = strlen(directory);
    if (!p || length > SIZE_MAX - 32u) { free(p); ds4_reservation_cancel(lease); return DS4_STATUS_NOMEM; }
    p->file.path = malloc(length + 32u);
    if (!p->file.path) { free(p); ds4_reservation_cancel(lease); return DS4_STATUS_NOMEM; }
    snprintf(p->file.path, length + 32u, "%s/ds4-checkpoint.XXXXXX", directory);
    int fd = mkstemp(p->file.path);
    FILE *fp = fd >= 0 ? fdopen(fd, "w+b") : NULL;
    if (!fp && fd >= 0) close(fd);
    char error[256] = {0};
#ifndef DS4_NO_GPU
    p->mtp = s->qwen_graph.mtp_ready;
#endif
    uint8_t config[32];
    lifecycle_config(s, p->mtp, config);
    status = DS4_STATUS_NOMEM;
    if (fp && !fseeko(fp, DS4_CHECKPOINT_HEADER_BYTES, SEEK_SET) &&
        !ds4_session_save_payload(s, fp, error, sizeof(error))) {
        off_t end = ftello(fp);
        if (end >= 0 && (uint64_t)end == body + DS4_CHECKPOINT_HEADER_BYTES)
            status = ds4_checkpoint_seal(fp, s->engine->lifecycle_identity, config, body, metadata, metadata_bytes);
    }
    if (fp && fclose(fp)) status = DS4_STATUS_NOMEM;
    if (!status) {
        int directory_fd = open(directory, O_RDONLY | O_DIRECTORY);
        if (directory_fd < 0 || fsync(directory_fd)) status = DS4_STATUS_NOMEM;
        if (directory_fd >= 0 && close(directory_fd)) status = DS4_STATUS_NOMEM;
    }
    if (!status) {
        ds4_resource_spec spec = {.owner=(uint64_t)(uintptr_t)s, .requested=total,
            .allocated=total, .live=total, .location=DS4_MEMORY_CHECKPOINT, .authoritative=true};
        status = ds4_reservation_commit(lease, &spec, &p->account);
    }
    (void)ds4_reservation_cancel(lease);
    if (status) { ds4_session_checkpoint_free(p); return status; }
    p->file.bytes = total; *out = p;
    return DS4_STATUS_OK;
}

static ds4_status lifecycle_checkpoint_restore(ds4_session *s, const ds4_session_checkpoint *p,
        void *metadata, size_t *metadata_bytes) {
    if (!s || !p || !metadata_bytes) return DS4_STATUS_INVALID;
    if (!lifecycle_supported(s->engine)) return DS4_STATUS_UNSUPPORTED;
    if (s->engine->weights_evicted) return DS4_STATUS_NOT_RESIDENT;
    if (!lifecycle_identity(s->engine)) return DS4_STATUS_INVALID;
    uint8_t config[32]; lifecycle_config(s, p->mtp, config);
    FILE *fp = fopen(p->file.path, "rb");
    if (!fp) return DS4_STATUS_RESTORE;
    uint64_t body = 0;
    /* Validate auxiliary data into a private buffer too: a failed graph load
     * must not advance the caller's RNG or request state. */
    size_t capacity = *metadata_bytes;
    void *aux = capacity && metadata ? malloc(capacity) : NULL;
    if (capacity && metadata && !aux) { fclose(fp); return DS4_STATUS_NOMEM; }
    ds4_status status = ds4_checkpoint_validate(fp, s->engine->lifecycle_identity,
        config, &body, aux, &capacity);
    if (status) { *metadata_bytes = capacity; free(aux); fclose(fp); return status; }
#ifndef DS4_NO_GPU
    ds4_session *candidate = NULL;
    if (ds4_session_create(&candidate, s->engine, s->ctx_size)) status = DS4_STATUS_NOMEM;
    char error[256] = {0};
    if (!status) {
        candidate->qwen_graph.mtp_ready = p->mtp;
        if (ds4_session_load_payload(candidate, fp, body, error, sizeof(error))) status = DS4_STATUS_RESTORE;
    }
    if (fclose(fp)) status = DS4_STATUS_RESTORE;
    if (!status) {
        qwen_graph_free(&s->qwen_graph);
        s->qwen_graph = candidate->qwen_graph;
        memset(&candidate->qwen_graph, 0, sizeof(candidate->qwen_graph));
        s->qwen_graph_ready = true; candidate->qwen_graph_ready = false;
        token_vec old = s->checkpoint; s->checkpoint = candidate->checkpoint; candidate->checkpoint = old;
        ds4_vision_identity *images = s->checkpoint_images;
        s->checkpoint_images = candidate->checkpoint_images; candidate->checkpoint_images = images;
        s->checkpoint_image_count = candidate->checkpoint_image_count;
        memcpy(s->logits, candidate->logits, (size_t)DS4_N_VOCAB * sizeof(float));
        if (p->mtp) memcpy(s->mtp_logits, candidate->mtp_logits, (size_t)DS4_N_VOCAB * sizeof(float));
        s->mtp_draft_token = candidate->mtp_draft_token;
        s->mtp_draft_valid = candidate->mtp_draft_valid;
        s->checkpoint_valid = true;
        s->lifecycle_parked = false;
        if (s->parked_checkpoint && s->parked_checkpoint != p) {
            ds4_session_checkpoint_free(s->parked_checkpoint); s->parked_checkpoint = NULL;
        }
        ds4_resources_transfer_owner(ds4_engine_resources(s->engine),
            (uint64_t)(uintptr_t)candidate, (uint64_t)(uintptr_t)s);
        if (capacity) memcpy(metadata, aux, capacity);
        *metadata_bytes = capacity;
    }
    ds4_session_free(candidate);
#else
    fclose(fp); status = DS4_STATUS_UNSUPPORTED;
#endif
    free(aux); return status;
}

static ds4_status lifecycle_park(ds4_session *s, const char *directory) {
    if (!s || !directory || !directory[0]) return DS4_STATUS_INVALID;
    if (!lifecycle_supported(s->engine)) return DS4_STATUS_UNSUPPORTED;
    if (s->lifecycle_parked) return DS4_STATUS_OK;
#ifndef DS4_NO_GPU
    ds4_session_checkpoint *p = NULL;
    if (s->checkpoint_valid) {
        ds4_status status = lifecycle_checkpoint_capture(s, directory, NULL, 0, &p);
        if (status) return status;
    }
    if (!ds4_gpu_synchronize()) { ds4_session_checkpoint_free(p); return DS4_STATUS_INFERENCE; }
    ds4_session_checkpoint_free(s->parked_checkpoint); s->parked_checkpoint = p;
    qwen_graph_free(&s->qwen_graph); s->qwen_graph_ready = false;
    s->lifecycle_parked = true;
    return DS4_STATUS_OK;
#else
    (void)directory; return DS4_STATUS_UNSUPPORTED;
#endif
}
static ds4_status lifecycle_resume(ds4_session *s) {
    if (!s) return DS4_STATUS_INVALID;
    if (!lifecycle_supported(s->engine)) return DS4_STATUS_UNSUPPORTED;
    if (!s->lifecycle_parked) return DS4_STATUS_OK;
    if (s->engine->weights_evicted) return DS4_STATUS_NOT_RESIDENT;
    if (s->parked_checkpoint) {
        size_t bytes = 0;
        ds4_status status = lifecycle_checkpoint_restore(s, s->parked_checkpoint, NULL, &bytes);
        if (status) return status;
        ds4_session_checkpoint_free(s->parked_checkpoint); s->parked_checkpoint = NULL;
        return DS4_STATUS_OK;
    }
#ifndef DS4_NO_GPU
    ds4_resource_owner_scope owner __attribute__((cleanup(ds4_resource_owner_leave))) =
        ds4_resource_owner_enter((uint64_t)(uintptr_t)s);
    if (!qwen_graph_alloc(&s->qwen_graph, s->ctx_size, s->prefill_cap) ||
        (s->engine->support_kind == DS4_SUPPORT_QWEN35_MTP &&
         !qwen_graph_enable_mtp(&s->qwen_graph, s->engine->mtp_draft_tokens))) {
        qwen_graph_free(&s->qwen_graph); return DS4_STATUS_NOMEM;
    }
    s->qwen_graph_ready = true; s->lifecycle_parked = false;
    return DS4_STATUS_OK;
#else
    return DS4_STATUS_UNSUPPORTED;
#endif
}

ds4_status ds4_session_trim(ds4_session *s) {
    DS4_ENGINE_MAINTENANCE_SCOPE(s ? s->engine : NULL, DS4_STATUS_BUSY);
    if (!s) return DS4_STATUS_INVALID;
    if (!lifecycle_supported(s->engine)) return DS4_STATUS_UNSUPPORTED;
#ifndef DS4_NO_GPU
    if (!ds4_gpu_synchronize()) return DS4_STATUS_INFERENCE;
    qwen_graph_free_draft_workspace(&s->qwen_graph);
#endif
    return DS4_STATUS_OK;
}
ds4_status ds4_engine_weights_release(ds4_engine *e) {
    if (!e) return DS4_STATUS_INVALID;
    DS4_ENGINE_MAINTENANCE_SCOPE(e, DS4_STATUS_BUSY);
    if (!lifecycle_supported(e)) return DS4_STATUS_UNSUPPORTED;
    if (e->weights_evicted) return DS4_STATUS_OK;
    if (!lifecycle_identity(e)) return DS4_STATUS_INVALID;
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    if (!ds4_gpu_synchronize()) return DS4_STATUS_INFERENCE;
    ds4_gpu_legacy_model_release(); /* Invalidates captured weight addresses. */
    e->vision_map_ready = false; e->weights_evicted = true;
    return DS4_STATUS_OK;
#else
    return DS4_STATUS_UNSUPPORTED;
#endif
}
ds4_status ds4_engine_weights_load(ds4_engine *e) {
    if (!e) return DS4_STATUS_INVALID;
    DS4_ENGINE_MAINTENANCE_SCOPE(e, DS4_STATUS_BUSY);
    if (!lifecycle_supported(e)) return DS4_STATUS_UNSUPPORTED;
    if (!e->weights_evicted) return DS4_STATUS_OK;
    if (!lifecycle_identity(e)) return DS4_STATUS_INVALID;
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    bool ok = ds4_gpu_set_model_map_range(e->model.map, e->model.size,
        e->model.tensor_data_pos, e->model.size - e->model.tensor_data_pos, e->model.max_tensor_bytes) != 0;
    if (ok) ok = ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map) != 0;
    if (ok) ok = accelerator_cache_model_tensors(e->backend, &e->model, NULL, NULL, 0);
    if (ok && e->mtp_ready) {
        ok = ds4_gpu_set_model_fd_for_map(e->mtp_model.fd, e->mtp_model.map) != 0 &&
            accelerator_cache_model_tensors(e->backend, &e->mtp_model, NULL, NULL, 0);
        (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
    }
    if (!ok) { ds4_gpu_legacy_model_release(); return DS4_STATUS_NOMEM; }
    ds4_gpu_enable_q8_dequant_gemm(); e->weights_evicted = false;
    return DS4_STATUS_OK;
#else
    return DS4_STATUS_UNSUPPORTED;
#endif
}

/* Public mutation starts between execution scopes. Private vision orchestration
 * already owns the exclusive scope and calls the adapter helpers directly. */
ds4_status ds4_session_checkpoint_capture(ds4_session *s, const char *directory,
        const void *metadata, size_t bytes, ds4_session_checkpoint **out) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    DS4_ENGINE_MAINTENANCE_SCOPE(s ? s->engine : NULL, DS4_STATUS_BUSY);
    return lifecycle_checkpoint_capture(s, directory, metadata, bytes, out);
}
ds4_status ds4_session_checkpoint_restore(ds4_session *s, const ds4_session_checkpoint *p,
        void *metadata, size_t *bytes) {
    DS4_ENGINE_MAINTENANCE_SCOPE(s ? s->engine : NULL, DS4_STATUS_BUSY);
    return lifecycle_checkpoint_restore(s, p, metadata, bytes);
}
ds4_status ds4_session_park(ds4_session *s, const char *directory) {
    DS4_ENGINE_MAINTENANCE_SCOPE(s ? s->engine : NULL, DS4_STATUS_BUSY);
    return lifecycle_park(s, directory);
}
ds4_status ds4_session_resume(ds4_session *s) {
    DS4_ENGINE_MAINTENANCE_SCOPE(s ? s->engine : NULL, DS4_STATUS_BUSY);
    return lifecycle_resume(s);
}

ds4_status ds4_session_memory_query(ds4_session *s, ds4_session_memory *out) {
    DS4_ENGINE_SCOPE(s ? s->engine : NULL, DS4_STATUS_BUSY);
    if (!s || !out) return DS4_STATUS_INVALID;
    memset(out, 0, sizeof(*out));
    out->parked = s->lifecycle_parked; out->weights_resident = !s->engine->weights_evicted;
    if (!lifecycle_supported(s->engine)) return DS4_STATUS_OK; /* Explicitly unknown. */
    out->capacity_known = true;
    out->payload = s->parked_checkpoint ? s->parked_checkpoint->file.bytes - DS4_CHECKPOINT_HEADER_BYTES :
        ds4_session_payload_bytes(s);
    out->payload_known = s->checkpoint_valid && out->payload != 0;
#ifndef DS4_NO_GPU
    ds4_qwen_gpu_graph *g = &s->qwen_graph;
    uint64_t persistent = 0;
    for (unsigned i = 0; i < DS4_MAX_LAYER; i++) {
        out->recurrent += ds4_gpu_tensor_bytes(g->conv_state[i]) + ds4_gpu_tensor_bytes(g->recurrent_state[i]);
        persistent += ds4_gpu_tensor_bytes(g->key_cache[i]) + ds4_gpu_tensor_bytes(g->value_cache[i]);
    }
    out->mtp = ds4_gpu_tensor_bytes(g->mtp_pending_h) + ds4_gpu_tensor_bytes(g->mtp_key_cache) +
        ds4_gpu_tensor_bytes(g->mtp_value_cache);
    ds4_resource_totals resources;
    ds4_resources_totals(ds4_engine_resources(s->engine), (uint64_t)(uintptr_t)s, &resources);
    out->device_capacity = resources.allocated[DS4_MEMORY_DEVICE];
    uint64_t state = persistent + out->recurrent + out->mtp;
    out->scratch = out->device_capacity >= state ? out->device_capacity - state : 0;
    out->host_capacity = sizeof(*s) + (uint64_t)s->checkpoint.cap * sizeof(int) +
        (uint64_t)s->checkpoint_image_count * sizeof(ds4_vision_identity) +
        (uint64_t)DS4_N_VOCAB * sizeof(float) * (2u + (s->mtp_logits != NULL)) +
        (g->host_pos3 ? (uint64_t)g->ctx_cap * 12u : 0);
#endif
    return DS4_STATUS_OK;
}
