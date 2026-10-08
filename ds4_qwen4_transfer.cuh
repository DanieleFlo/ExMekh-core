/* Included inside qwen4_cuda. Explicit bounded DMA staging, shared by experts
 * and KV. No mapped host reads: this also works with WSL's CUDA driver. */
static void *transfer_host[2];
static cudaEvent_t transfer_done[2];
static unsigned transfer_next;
static const uint64_t transfer_capacity = 64ull * 1048576ull;
static uint64_t transfer_h2d_bytes, transfer_d2h_bytes;
static bool transfer_batch_on;
static int transfer_batch_buffer = -1;
static uint64_t transfer_batch_used;
static cudaStream_t transfer_batch_stream;
struct transfer_pending_read { void *destination; uint64_t offset, bytes; };
static std::vector<transfer_pending_read> transfer_pending[2];
struct transfer_pending_write {
    ds4_read_task task;
    void *destination;
    void *staged;
    const void *source;
    uint64_t offset, bytes;
    bool async;
};
static std::vector<transfer_pending_write> transfer_writes[2];
static ds4_read_queue transfer_workers;
static bool transfer_workers_attempted;
static uint64_t transfer_upload_calls, transfer_h2d_submissions;
static uint64_t transfer_stage_jobs;
/* Workload metadata is independent of the native GEMM router cache. A hybrid
 * decode need not rebuild those lists after a large prefill. */
static unsigned transfer_workload_rows;
static double transfer_stage_wait;
static uintptr_t hybrid_staging_end(uintptr_t address);

static uintptr_t transfer_allocation_end(uintptr_t address, uint64_t bytes) {
    const uintptr_t prefill = (uintptr_t)g_stream_selected_cache.prefill_ptr;
    if (prefill && address >= prefill && address - prefill < g_stream_selected_cache.prefill_capacity)
        return prefill + g_stream_selected_cache.prefill_capacity;
    const uintptr_t hybrid_end = hybrid_staging_end(address);
    if (hybrid_end) return hybrid_end;
    for (const auto &arena : g_model_arenas) {
        const uintptr_t first = (uintptr_t)arena.device_ptr;
        if (address >= first && address - first < arena.bytes) return first + arena.bytes;
    }
    /* Contiguous addresses do not imply a single CUDA allocation. Unknown
     * destinations keep the original individual copy boundary. */
    return address + bytes;
}

static void transfer_stage_run(void *, unsigned, void *payload) {
    auto *write = (transfer_pending_write *)payload;
    memcpy(write->staged, write->source, write->bytes);
}

/* Payload and CUDA publication are distinct: workers only copy disjoint
 * ordinary-RAM ranges into pinned storage. The inference thread waits and
 * merges contiguous destinations before submitting any consumer. */
static int transfer_submit_writes(unsigned i) {
    auto &writes = transfer_writes[i];
    const double start = cuda_wall_sec();
    for (auto &write : writes) if (write.async) ds4_read_queue_wait(&transfer_workers, &write.task);
    transfer_stage_wait += cuda_wall_sec() - start;
    int ok = 1;
    for (size_t begin = 0; begin < writes.size();) {
        const auto &first = writes[begin]; uint64_t bytes = first.bytes;
        const uintptr_t destination = (uintptr_t)first.destination;
        const uintptr_t allocation_end = transfer_allocation_end(destination, bytes);
        size_t end = begin + 1;
        while (end < writes.size() && writes[end].offset == first.offset + bytes &&
               (char *)writes[end].destination == (char *)first.destination + bytes &&
               bytes <= allocation_end - destination && writes[end].bytes <= allocation_end - destination - bytes) {
            bytes += writes[end++].bytes;
        }
        if (!cuda_ok(cudaMemcpyAsync(first.destination, (char *)transfer_host[i] + first.offset,
                                    bytes, cudaMemcpyHostToDevice, transfer_batch_stream), "Qwen merged staged H2D")) { ok = 0; break; }
        transfer_h2d_submissions++; begin = end;
    }
    writes.clear();
    return ok;
}

static int transfer_finish_buffer(unsigned i) {
    if (!cuda_ok(cudaEventSynchronize(transfer_done[i]), "Qwen DMA buffer reuse")) return 0;
    for (const auto &read : transfer_pending[i])
        memcpy(read.destination, (const char *)transfer_host[i] + read.offset, read.bytes);
    transfer_pending[i].clear();
    return 1;
}

static int transfer_batch_flush(void) {
    if (transfer_batch_buffer < 0) return 1;
    if (!transfer_submit_writes((unsigned)transfer_batch_buffer)) return 0;
    const int ok = cuda_ok(cudaEventRecord(transfer_done[transfer_batch_buffer], transfer_batch_stream),
                          "Qwen DMA batch source consumed");
    transfer_batch_buffer = -1; transfer_batch_used = 0;
    return ok;
}
static int transfer_batch_begin(cudaStream_t stream) {
    if (!transfer_batch_flush()) return 0;
    transfer_batch_on = true; transfer_batch_stream = stream;
    return 1;
}
static int transfer_batch_end(void) {
    int ok = transfer_batch_flush(); transfer_batch_on = false;
    for (unsigned i = 0; i < 2; i++) if (!transfer_pending[i].empty())
        ok = transfer_finish_buffer(i) && ok;
    return ok;
}

/* A RAM slot is reserved before its deferred demotion finishes. Publish its
 * bytes only after DMA and the host copy, even if it is requested again in
 * the same batch. Other destinations stay pending until the group boundary. */
static int transfer_source_ready(const void *source, uint64_t bytes) {
    const uintptr_t first = (uintptr_t)source;
    for (unsigned i = 0; i < 2; i++) for (const auto &read : transfer_pending[i]) {
        const uintptr_t destination = (uintptr_t)read.destination;
        if (first < destination + read.bytes && destination < first + bytes) {
            if (!transfer_batch_flush() || !transfer_finish_buffer(i)) return 0;
            break;
        }
    }
    return 1;
}

static int transfer_prepare(void) {
    for (unsigned i = 0; i < 2; i++) {
        if (!transfer_host[i] && !cuda_ok(cudaMallocHost(&transfer_host[i], transfer_capacity), "Qwen DMA staging")) return 0;
        if (!transfer_done[i] && !cuda_ok(cudaEventCreateWithFlags(&transfer_done[i], cudaEventDisableTiming), "Qwen DMA lifetime")) return 0;
        if (transfer_pending[i].capacity() < 4096) transfer_pending[i].reserve(4096);
        if (transfer_writes[i].capacity() < 4096) transfer_writes[i].reserve(4096);
    }
    if (!transfer_workers_attempted) {
        transfer_workers_attempted = true;
        const char *env = getenv("DS4_CUDA_QWEN4_STAGING_WORKERS");
        const char *policy = getenv("DS4_CUDA_QWEN4_EXPERT_POLICY");
        unsigned workers = policy && !strcmp(policy, "stable") ? 2u : 0u;
        if (env && *env) {
            char *end = NULL; const unsigned long n = strtoul(env, &end, 10);
            if (end != env && !*end && *env != '-' && n <= 4u) workers = (unsigned)n;
        }
        const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (cpus > 0) workers = std::min(workers, (unsigned)cpus);
        if (workers) (void)ds4_read_queue_start(&transfer_workers, workers, transfer_stage_run, NULL);
    }
    return 1;
}

static void transfer_reset(void) {
    (void)transfer_batch_end();
    ds4_read_queue_stop(&transfer_workers);
    transfer_workers_attempted = false;
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_CACHE_STATS", 0))
        fprintf(stderr, "ds4: Qwen staged uploads=%llu H2D_submissions=%llu CPU_stage_jobs=%llu CPU_stage_wait_ms=%.3f staging_workers_drained=1\n",
            (unsigned long long)transfer_upload_calls, (unsigned long long)transfer_h2d_submissions,
            (unsigned long long)transfer_stage_jobs, transfer_stage_wait * 1000.0);
    for (unsigned i = 0; i < 2; i++) {
        if (transfer_done[i]) { (void)cudaEventSynchronize(transfer_done[i]); (void)cudaEventDestroy(transfer_done[i]); }
        if (transfer_host[i]) (void)cudaFreeHost(transfer_host[i]);
        transfer_host[i] = NULL; transfer_done[i] = NULL;
        transfer_pending[i].clear();
        transfer_writes[i].clear();
    }
    transfer_next = 0; transfer_h2d_bytes = transfer_d2h_bytes = 0;
    transfer_upload_calls = transfer_h2d_submissions = 0;
    transfer_stage_jobs = 0; transfer_stage_wait = 0;
    transfer_workload_rows = 0;
}

static int transfer_upload(void *dst, const void *src, uint64_t bytes, cudaStream_t stream) {
    if (!transfer_prepare() || !transfer_source_ready(src, bytes)) return 0;
    if (transfer_batch_on && stream == transfer_batch_stream) {
        for (uint64_t off = 0; off < bytes;) {
            if (transfer_batch_buffer < 0) {
                const unsigned i = transfer_next++ % 2u;
                if (!transfer_finish_buffer(i)) return 0;
                transfer_batch_buffer = (int)i;
            }
            const unsigned i = (unsigned)transfer_batch_buffer;
            if (transfer_writes[i].size() == transfer_writes[i].capacity()) {
                if (!transfer_submit_writes(i)) return 0;
            }
            const uint64_t n = std::min(transfer_capacity - transfer_batch_used, bytes - off);
            char *staged = (char *)transfer_host[i] + transfer_batch_used;
            transfer_writes[i].push_back({{}, (char *)dst + off, staged, (const char *)src + off, transfer_batch_used, n, false});
            auto &write = transfer_writes[i].back();
            /* Keep tiny decode copies inline. Explicit workers still live for
             * the engine lifetime, including cancellation/error cleanup. */
            if (transfer_workload_rows > 8 && ds4_read_queue_workers(&transfer_workers)) {
                write.async = ds4_read_queue_submit(&transfer_workers, &write.task, &write) != 0;
                transfer_stage_jobs += write.async;
            }
            if (!write.async) memcpy(staged, write.source, n);
            transfer_upload_calls++;
            transfer_batch_used += n; off += n; transfer_h2d_bytes += n;
            if (transfer_batch_used == transfer_capacity && !transfer_batch_flush()) return 0;
        }
        return 1;
    }
    if (!transfer_batch_flush()) return 0;
    for (uint64_t off = 0; off < bytes;) {
        const unsigned i = transfer_next++ % 2u;
        const uint64_t n = std::min(transfer_capacity, bytes - off);
        if (!transfer_finish_buffer(i)) return 0;
        memcpy(transfer_host[i], (const char *)src + off, n);
        if (!cuda_ok(cudaMemcpyAsync((char *)dst + off, transfer_host[i], n, cudaMemcpyHostToDevice, stream), "Qwen staged H2D") ||
            !cuda_ok(cudaEventRecord(transfer_done[i], stream), "Qwen DMA source consumed")) return 0;
        off += n; transfer_h2d_bytes += n;
        transfer_upload_calls++; transfer_h2d_submissions++;
    }
    return 1;
}

static int transfer_download(void *dst, const void *src, uint64_t bytes, cudaStream_t stream) {
    if (!transfer_prepare()) return 0;
    if (transfer_batch_on && stream == transfer_batch_stream) {
        for (uint64_t off = 0; off < bytes;) {
            if (transfer_batch_buffer < 0) {
                const unsigned i = transfer_next++ % 2u;
                if (!transfer_finish_buffer(i)) return 0;
                transfer_batch_buffer = (int)i;
            }
            const unsigned i = (unsigned)transfer_batch_buffer;
            /* A later D2H may read a slot just overwritten by an upload. */
            if (!transfer_submit_writes(i)) return 0;
            if (transfer_pending[i].size() == transfer_pending[i].capacity()) {
                if (!transfer_batch_flush() || !transfer_finish_buffer(i)) return 0;
                continue;
            }
            const uint64_t n = std::min(transfer_capacity - transfer_batch_used, bytes - off);
            if (!cuda_ok(cudaMemcpyAsync((char *)transfer_host[i] + transfer_batch_used,
                    (const char *)src + off, n, cudaMemcpyDeviceToHost, stream), "Qwen batched D2H")) return 0;
            transfer_pending[i].push_back({(char *)dst + off, transfer_batch_used, n});
            transfer_batch_used += n; off += n; transfer_d2h_bytes += n;
            if (transfer_batch_used == transfer_capacity && !transfer_batch_flush()) return 0;
        }
        return 1;
    }
    if (!transfer_batch_flush()) return 0;
    for (uint64_t off = 0; off < bytes;) {
        const unsigned i = transfer_next++ % 2u;
        const uint64_t n = std::min(transfer_capacity, bytes - off);
        if (!transfer_finish_buffer(i) ||
            !cuda_ok(cudaMemcpyAsync(transfer_host[i], (const char *)src + off, n, cudaMemcpyDeviceToHost, stream), "Qwen staged D2H") ||
            !cuda_ok(cudaEventRecord(transfer_done[i], stream), "Qwen DMA destination ready") ||
            !cuda_ok(cudaEventSynchronize(transfer_done[i]), "Qwen host bytes ready")) return 0;
        memcpy((char *)dst + off, transfer_host[i], n);
        off += n; transfer_d2h_bytes += n;
    }
    return 1;
}
