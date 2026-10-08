/* Dense Qwen context capacity: logical FP32 backing, explicit WSL DMA.
 * The attention kernels retain the original full-context reduction order. */
namespace qwen35_kv {
static const uint64_t row_bytes = 1024ull * sizeof(float);
struct cache {
    unsigned context, limit, cells, frontier;
    unsigned char *host;
    bool staged;
};
static std::unordered_map<const ds4_gpu_tensor *, cache> caches;
static void *stage[2];
static const ds4_gpu_tensor *stage_owner[2];
static unsigned stage_frontier[2];
static uint64_t stage_reuses;
static void *tiles[2][2];
static unsigned tile_cells;
static cudaStream_t copy_stream;
static cudaEvent_t ready[2], consumed[2];
static uint64_t stage_bytes, host_bytes, resident_bytes, h2d_bytes, d2h_bytes;
static uint64_t peak_host_bytes, peak_resident_bytes;
static uint64_t host_budget;
static bool host_budget_set;
static unsigned max_frontier;
static unsigned growths, spills;

static cache *find(const ds4_gpu_tensor *t) {
    auto it = caches.find(t); return it == caches.end() ? NULL : &it->second;
}
static void invalidate_stage(const ds4_gpu_tensor *t) {
    for (unsigned i = 0; i < 2; ++i) if (stage_owner[i] == t) {
        stage_owner[i] = NULL; stage_frontier[i] = 0;
    }
}
static int mode(void) {
    if (ds4_env_qwen_weight_streaming()) return 0;
    const char *s = getenv("DS4_CUDA_QWEN_KV_CAPACITY_MODE");
    if (!s || !s[0] || !strcmp(s, "off")) return 0;
    if (!strcmp(s, "grow")) return 1;
    if (!strcmp(s, "hybrid-attention")) return 2;
    fprintf(stderr, "ds4: invalid Qwen KV capacity mode: %s\n", s);
    return -1;
}
static int grow(ds4_gpu_tensor *t, unsigned used) {
    cache *c = find(t); if (!c) return 1;
    used = std::min(used, c->limit);
    if (used <= c->cells) return 1;
    const unsigned cells = std::min(c->limit, (used + 4095u) & ~4095u);
    void *next = NULL;
    if (!cuda_ok(cudaMalloc(&next, (size_t)cells * row_bytes), "dense KV growth")) return 0;
    peak_resident_bytes = std::max(peak_resident_bytes, resident_bytes + (uint64_t)cells * row_bytes);
    if (!cuda_ok(cudaMemset(next, 0, (size_t)cells * row_bytes), "dense KV growth zero") ||
        (c->cells && !cuda_ok(cudaMemcpy(next, t->ptr, (size_t)c->cells * row_bytes,
            cudaMemcpyDeviceToDevice), "dense KV growth copy"))) {
        cudaFree(next); return 0;
    }
    cudaFree(t->ptr); t->ptr = next;
    resident_bytes += (uint64_t)(cells - c->cells) * row_bytes;
    peak_resident_bytes = std::max(peak_resident_bytes, resident_bytes);
    c->cells = cells; ++growths; return 1;
}
static int release(const ds4_gpu_tensor *t) {
    auto it = caches.find(t); if (it == caches.end()) return 0;
    cudaDeviceSynchronize();
    invalidate_stage(t);
    cache &c = it->second;
    resident_bytes -= (uint64_t)c.cells * row_bytes;
    if (c.host) { host_bytes -= (uint64_t)c.context * row_bytes; free(c.host); }
    caches.erase(it);
    if (caches.empty()) {
        fprintf(stderr, "ds4: dense KV capacity frontier=%u peak_host=%.3f GiB peak_resident=%.3f GiB peak_stage=%.3f GiB growths=%u spills=%u H2D=%.3f GiB D2H=%.3f GiB NextN_stage_reuses=%llu\n",
            max_frontier, (double)peak_host_bytes / 1073741824.0,
            (double)peak_resident_bytes / 1073741824.0,
            (double)(2u * stage_bytes + 4ull * tile_cells * row_bytes) / 1073741824.0, growths, spills,
            (double)h2d_bytes / 1073741824.0, (double)d2h_bytes / 1073741824.0,
            (unsigned long long)stage_reuses);
        for (unsigned i = 0; i < 2; ++i) {
            cudaFree(stage[i]); stage[i] = NULL; stage_owner[i] = NULL; stage_frontier[i] = 0;
        }
        stage_reuses = 0;
        if (copy_stream) cudaStreamDestroy(copy_stream);
        copy_stream = NULL;
        for (unsigned b = 0; b < 2; ++b) {
            if (ready[b]) cudaEventDestroy(ready[b]);
            if (consumed[b]) cudaEventDestroy(consumed[b]);
            ready[b] = consumed[b] = NULL;
            for (unsigned i = 0; i < 2; ++i) { cudaFree(tiles[b][i]); tiles[b][i] = NULL; }
        }
        tile_cells = 0;
        stage_bytes = 0; growths = spills = 0; h2d_bytes = d2h_bytes = 0;
        peak_host_bytes = peak_resident_bytes = 0; max_frontier = 0;
        host_budget = 0; host_budget_set = false;
    }
    return 1;
}
static int fill(ds4_gpu_tensor *t, float value, uint64_t count) {
    cache *c = find(t); if (!c) return -1;
    if (value != 0.0f || count != (uint64_t)c->context * 1024u) return 0;
    if (!cuda_ok(cudaDeviceSynchronize(), "dense KV reset fence")) return 0;
    invalidate_stage(t);
    if (c->host) memset(c->host, 0, (size_t)c->context * row_bytes);
    c->frontier = 0;
    return !c->cells || cuda_ok(cudaMemset(t->ptr, 0, (size_t)c->cells * row_bytes), "dense KV reset");
}
static int io(const ds4_gpu_tensor *t, uint64_t off, void *data, uint64_t bytes, bool write) {
    cache *c = find(t); if (!c) return -1;
    if (!cuda_ok(cudaDeviceSynchronize(), "dense KV logical IO fence")) return 0;
    if (write) {
        invalidate_stage(t);
        if (!grow((ds4_gpu_tensor *)t, (unsigned)((off + bytes + row_bytes - 1u) / row_bytes))) return 0;
        c = find(t);
        if (c->host) memcpy(c->host + off, data, (size_t)bytes);
        c->frontier = std::max(c->frontier, (unsigned)((off + bytes + row_bytes - 1u) / row_bytes));
        const uint64_t end = std::min(off + bytes, (uint64_t)c->cells * row_bytes);
        return end <= off || cuda_ok(cudaMemcpy((char *)t->ptr + off, data,
            (size_t)(end - off), cudaMemcpyHostToDevice), "dense KV logical restore");
    }
    if (c->host) { memcpy(data, c->host + off, (size_t)bytes); return 1; }
    if (off + bytes > (uint64_t)c->cells * row_bytes) return 0;
    return cuda_ok(cudaMemcpy(data, (char *)t->ptr + off, (size_t)bytes,
        cudaMemcpyDeviceToHost), "dense KV logical checkpoint");
}
static int append(ds4_gpu_tensor *t, const void *rows, unsigned position, unsigned n) {
    cache *c = find(t); if (!c || position > c->context || n > c->context - position) return 0;
    if (!grow(t, position + n)) return 0;
    c = find(t);
    if (position < c->cells) {
        const unsigned live = std::min(n, c->cells - position);
        if (!cuda_ok(cudaMemcpyAsync((char *)t->ptr + (uint64_t)position * row_bytes,
            rows, (size_t)live * row_bytes, cudaMemcpyDeviceToDevice, 0), "dense KV resident append")) return 0;
    }
    if (c->host) {
        if (!qwen4_cuda::transfer_download(c->host + (uint64_t)position * row_bytes,
            rows, (uint64_t)n * row_bytes, 0)) return 0;
        d2h_bytes += (uint64_t)n * row_bytes;
    }
    if (c->staged) for (unsigned i = 0; i < 2; ++i) if (stage_owner[i] == t) {
        if (position > stage_frontier[i]) { invalidate_stage(t); break; }
        if (!cuda_ok(cudaMemcpyAsync((char *)stage[i] + (uint64_t)position * row_bytes,
                rows, (uint64_t)n * row_bytes, cudaMemcpyDeviceToDevice, 0), "NextN staged KV append")) return 0;
        stage_frontier[i] = position + n;
    }
    c->frontier = position + n; max_frontier = std::max(max_frontier, c->frontier); return 1;
}
static int materialize(ds4_gpu_tensor *k, ds4_gpu_tensor *v, unsigned used,
        ds4_gpu_tensor *key, ds4_gpu_tensor *value) {
    cache *a = find(k), *b = find(v);
    if (!a || !b || used > a->frontier || used > b->frontier) return 0;
    *key = *k; *value = *v;
    if (used <= a->cells && used <= b->cells) return 1;
    if (!a->host || !b->host || (uint64_t)used * row_bytes > stage_bytes) return 0;
    ds4_gpu_tensor *original[2] = {k, v};
    ds4_gpu_tensor *result[2] = {key, value};
    for (unsigned i = 0; i < 2; ++i) {
        cache *c = find(original[i]);
        if (c->staged && stage_owner[i] == original[i] && stage_frontier[i] >= used) {
            result[i]->ptr = stage[i]; result[i]->owner = 0; ++stage_reuses; continue;
        }
        stage_owner[i] = NULL; stage_frontier[i] = 0;
        const uint64_t resident = (uint64_t)std::min(c->cells, used) * row_bytes;
        if (resident && !cuda_ok(cudaMemcpyAsync(stage[i], original[i]->ptr, (size_t)resident,
                cudaMemcpyDeviceToDevice, 0), "dense KV stage resident")) return 0;
        const uint64_t remote = (uint64_t)used * row_bytes - resident;
        if (!qwen4_cuda::transfer_upload((char *)stage[i] + resident, c->host + resident, remote, 0)) return 0;
        h2d_bytes += remote; result[i]->ptr = stage[i]; result[i]->owner = 0;
        stage_owner[i] = original[i]; stage_frontier[i] = used;
    }
    ++spills; return 1;
}
}

static int qwen35_kv_alloc(ds4_gpu_tensor **key, ds4_gpu_tensor **value, unsigned context, bool mtp) {
    if (!key || !value || !context) return 0;
    *key = *value = NULL;
    const int mode = qwen35_kv::mode(); if (mode < 0) return 0;
    if (!mode) {
        *key = ds4_gpu_tensor_alloc((uint64_t)context * qwen35_kv::row_bytes);
        *value = ds4_gpu_tensor_alloc((uint64_t)context * qwen35_kv::row_bytes);
        if (*key && *value) return 1;
        ds4_gpu_tensor_free(*key); ds4_gpu_tensor_free(*value); *key = *value = NULL; return 0;
    }
    if (g_n_gpus != 1 || context > 262144u) return 0;
    unsigned limit = mode == 1 ? context : std::min(context,
        cuda_parse_u32_env_clamped("DS4_CUDA_QWEN_KV_RESIDENT", 32768u, 64u, 262144u, NULL));
    bool staged = false;
    if (mtp && mode == 2) {
        const char *placement = getenv("DS4_CUDA_QWEN_MTP_KV_MODE");
        if (!placement || !placement[0]) placement = "auto";
        if (strcmp(placement, "shared") && strcmp(placement, "auto") && strcmp(placement, "resident") && strcmp(placement, "staged")) {
            fprintf(stderr, "ds4: invalid NextN KV placement: %s\n", placement); return 0;
        }
        /* The target's existing prefill workspace is idle while its decode
         * consumes separate double-buffered tiles. Cache NextN there without
         * allocating another full-context KV pair. */
        staged = limit < context && (!strcmp(placement, "auto") || !strcmp(placement, "staged"));
        if (!staged && strcmp(placement, "shared") && limit < context) {
            /* All target caches are allocated lazily. Account for their
             * promised resident cells before committing the smaller drafter
             * cache, including one old/new growth allocation and headroom. */
            uint64_t reserved = 0;
            for (const auto &entry : qwen35_kv::caches)
                reserved += (uint64_t)(entry.second.limit - entry.second.cells) * qwen35_kv::row_bytes;
            const uint64_t needed = 3ull * context * qwen35_kv::row_bytes;
            const uint64_t headroom = 512ull << 20;
            size_t available = 0, total = 0;
            const bool fits = cuda_ok(cudaMemGetInfo(&available, &total), "NextN KV budget") &&
                available >= reserved && available - reserved >= headroom &&
                available - reserved - headroom >= needed;
            if (fits) limit = context;
            else if (!strcmp(placement, "resident")) {
                fprintf(stderr, "ds4: NextN resident KV exceeds budget (free=%.3f GiB target_reserved=%.3f GiB needed=%.3f GiB plus %.3f GiB headroom)\n",
                    available / 1073741824.0, reserved / 1073741824.0,
                    needed / 1073741824.0, headroom / 1073741824.0); return 0;
            }
        }
        fprintf(stderr, "ds4: NextN KV placement=%s resident_cells=%u context=%u reuse_prefill_stage=%u\n", placement, limit, context, staged);
    }
    if (mode == 2) {
        if (!qwen35_kv::host_budget_set) {
            uint64_t available = 0;
            if (!ds4_linux_nonmovable_memory(&available)) {
                fprintf(stderr, "ds4: cannot determine dense KV host memory budget\n"); return 0;
            }
            const uint64_t reserve = 6ull * 1073741824ull;
            qwen35_kv::host_budget = available > reserve ? available - reserve : 0;
            qwen35_kv::host_budget_set = true;
        }
        const uint64_t bytes = 2ull * context * qwen35_kv::row_bytes;
        if (qwen35_kv::host_bytes > qwen35_kv::host_budget || bytes > qwen35_kv::host_budget - qwen35_kv::host_bytes) {
            fprintf(stderr, "ds4: dense KV host budget exhausted: %.3f GiB needed, %.3f GiB already allocated, %.3f GiB budget\n",
                bytes / 1073741824.0, qwen35_kv::host_bytes / 1073741824.0, qwen35_kv::host_budget / 1073741824.0);
            return 0;
        }
    }
    if (mode == 2 && limit < context && (uint64_t)context * qwen35_kv::row_bytes > qwen35_kv::stage_bytes) {
        void *next[2] = {};
        for (unsigned i = 0; i < 2; ++i) {
            if (!cuda_ok(cudaMalloc(&next[i], (size_t)context * qwen35_kv::row_bytes), "dense KV layer stage")) {
                cudaFree(next[0]); cudaFree(next[1]); return 0;
            }
        }
        cudaDeviceSynchronize();
        for (unsigned i = 0; i < 2; ++i) {
            cudaFree(qwen35_kv::stage[i]); qwen35_kv::stage[i] = next[i];
            qwen35_kv::stage_owner[i] = NULL; qwen35_kv::stage_frontier[i] = 0;
        }
        qwen35_kv::stage_bytes = (uint64_t)context * qwen35_kv::row_bytes;
    }
    if (mode == 2 && limit < context && std::min(context, 8192u) > qwen35_kv::tile_cells) {
        const unsigned cells = std::min(context, 8192u);
        void *next[2][2] = {};
        bool ok = true;
        for (unsigned b = 0; b < 2 && ok; ++b) for (unsigned i = 0; i < 2 && ok; ++i)
            ok = cuda_ok(cudaMalloc(&next[b][i], (size_t)cells * qwen35_kv::row_bytes), "dense KV attention tile");
        if (!ok) { for (auto &pair : next) for (void *p : pair) cudaFree(p); return 0; }
        cudaDeviceSynchronize();
        for (unsigned b = 0; b < 2; ++b) for (unsigned i = 0; i < 2; ++i) {
            cudaFree(qwen35_kv::tiles[b][i]); qwen35_kv::tiles[b][i] = next[b][i];
        }
        qwen35_kv::tile_cells = cells;
        if (!qwen35_kv::copy_stream && !cuda_ok(cudaStreamCreateWithFlags(&qwen35_kv::copy_stream,
                cudaStreamNonBlocking), "dense KV copy stream")) return 0;
        for (unsigned b = 0; b < 2; ++b) {
            if (!qwen35_kv::ready[b] && !cuda_ok(cudaEventCreateWithFlags(&qwen35_kv::ready[b],
                    cudaEventDisableTiming), "dense KV tile ready")) return 0;
            if (!qwen35_kv::consumed[b] && !cuda_ok(cudaEventCreateWithFlags(&qwen35_kv::consumed[b],
                    cudaEventDisableTiming), "dense KV tile consumed")) return 0;
        }
    }
    ds4_gpu_tensor **tensors[2] = {key, value};
    for (unsigned i = 0; i < 2; ++i) {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(256u); if (!t) break;
        unsigned char *host = mode == 2 ? (unsigned char *)calloc(context, (size_t)qwen35_kv::row_bytes) : NULL;
        if (mode == 2 && !host) { ds4_gpu_tensor_free(t); break; }
        t->bytes = (uint64_t)context * qwen35_kv::row_bytes;
        qwen35_kv::caches.emplace(t, qwen35_kv::cache{context, limit, 0u, 0u, host, staged});
        if (host) {
            qwen35_kv::host_bytes += t->bytes;
            qwen35_kv::peak_host_bytes = std::max(qwen35_kv::peak_host_bytes, qwen35_kv::host_bytes);
        }
        *tensors[i] = t;
    }
    if (*key && *value) return 1;
    ds4_gpu_tensor_free(*key); ds4_gpu_tensor_free(*value); *key = *value = NULL; return 0;
}

extern "C" int ds4_gpu_qwen35_kv_alloc(ds4_gpu_tensor **key, ds4_gpu_tensor **value, unsigned context) {
    return qwen35_kv_alloc(key, value, context, false);
}
extern "C" int ds4_gpu_qwen35_mtp_kv_alloc(ds4_gpu_tensor **key, ds4_gpu_tensor **value, unsigned context) {
    return qwen35_kv_alloc(key, value, context, true);
}

extern "C" ds4_gpu_tensor *ds4_gpu_qwen35_mtp_rows_alloc(uint64_t bytes, uint32_t rows, uint32_t *resident_rows) {
    if (resident_rows) *resident_rows = 0;
    if (!resident_rows || !rows || rows > 15u || !bytes || bytes % rows ||
        g_n_gpus != 1 || qwen35_kv::mode() != 2) return NULL;
    const uint64_t row_size = bytes / rows;
    const uint64_t budget = (uint64_t)cuda_parse_u32_env_clamped(
        "DS4_CUDA_QWEN_MTP_DECODE_ROWS_MB", 1024u, 0u, 8192u, NULL) << 20;
    if (row_size > budget) {
        fprintf(stderr, "ds4: NextN decode rollback remains in RAM (%.3f GiB exceeds %.3f GiB arena budget)\n",
            bytes / 1073741824.0, budget / 1073741824.0); return NULL;
    }
    uint64_t promised = 0, growth = 0;
    for (const auto &entry : qwen35_kv::caches) {
        const auto &c = entry.second;
        promised += (uint64_t)(c.limit - c.cells) * qwen35_kv::row_bytes;
        if (c.cells < c.limit)
            growth = std::max(growth, (uint64_t)c.limit * qwen35_kv::row_bytes);
    }
    /* The large prefill score/dequant slab is idle here. The exact <=16-row
     * verifier uses small activation quantization and split-K partials (under
     * 16 MiB for this backend), not a full prefill score matrix. Drain and
     * invalidate consumers before returning that phase's temporary storage.
     * Its normal allocator grows again if a later prefill needs it. */
    const uint64_t released = g_cuda_tmp_bytes;
    if (g_cuda_tmp && g_cuda_tmp_bytes > (128ull << 20)) {
        if (!cuda_ok(cudaDeviceSynchronize(), "NextN prefill scratch drain")) return NULL;
        ds4_gpu_decode_graphs_invalidate();
        if (!cuda_ok(cudaFree(g_cuda_tmp), "NextN prefill scratch release")) return NULL;
        g_cuda_tmp = NULL; g_cuda_tmp_bytes = 0;
        fprintf(stderr, "ds4: NextN returned %.3f GiB prefill scratch before decode\n", released / 1073741824.0);
    }
    size_t available = 0, total = 0;
    const uint64_t headroom = (128ull << 20) + growth;
    if (!cuda_ok(cudaMemGetInfo(&available, &total), "NextN rollback budget") ||
        available < promised || available - promised < headroom ||
        available - promised - headroom < row_size) {
        fprintf(stderr, "ds4: NextN decode rollback remains in RAM (free=%.3f GiB promised=%.3f GiB needed=%.3f GiB headroom=%.3f GiB)\n",
            available / 1073741824.0, promised / 1073741824.0,
            bytes / 1073741824.0, headroom / 1073741824.0);
        return NULL;
    }
    const uint32_t admitted = (uint32_t)std::min<uint64_t>(rows,
        std::min<uint64_t>(budget, available - promised - headroom) / row_size);
    const uint64_t allocation = admitted * row_size;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    const cudaError_t result = cudaMalloc(&t->ptr, (size_t)allocation);
    if (result != cudaSuccess) {
        /* This is an optional optimization: clear the recoverable allocation
         * error before returning to the already allocated RAM/scratch path. */
        cudaGetLastError(); free(t); return NULL;
    }
    t->bytes = allocation; t->owner = 1; t->device_id = 0;
    g_gpu[0].used_bytes += allocation;
    *resident_rows = admitted;
    return t;
}
