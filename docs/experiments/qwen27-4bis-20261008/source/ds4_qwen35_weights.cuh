/* Ordered dense layer streaming. RAM owns only the nonresident complement;
 * original quantized bytes and compute kernels remain unchanged. */
namespace qwen35_weights {
struct span {
    uint64_t offset, bytes, packed;
    uint32_t layer;
    bool remote;
};
struct job { unsigned layer, slot; uint64_t ticket; bool consumed; };
struct state {
    const void *map = NULL;
    uint64_t size = 0, resident_bytes = 0, host_bytes = 0, slot_bytes = 0;
    unsigned layers = 0, live_layer = UINT32_MAX;
    std::vector<span> spans;
    std::vector<uint64_t> layer_bytes, host_offset;
    std::vector<unsigned> order, ordinal;
    unsigned char *host = NULL, *resident = NULL, *slots[2] = {};
    unsigned char *pinned[2] = {};
    uint64_t pin_bytes = 64ull << 20, tickets = 0, wanted[2] = {}, submitted[2] = {};
    bool used[2] = {}, stop = false, failed = false;
    cudaStream_t copy = NULL;
    cudaEvent_t ready[2] = {}, consumed[2] = {}, dma[2] = {};
    int device = 0;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<job> jobs;
    uint64_t forwards = 0, transfers = 0, h2d = 0, pending_waits = 0;
    uint64_t minimum_free = 0;
    double submit_wait_ms = 0;
};
static state *s;
static uint64_t align(uint64_t n) { return (n + 255u) & ~255ull; }

static void worker(state *p) {
    bool ok = cuda_ok(cudaSetDevice(p->device), "dense weight worker device");
    unsigned stage = 0;
    bool dma_live[2] = {};
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lock(p->mutex);
            if (!ok) { p->failed = true; p->changed.notify_all(); return; }
            p->changed.wait(lock, [&] { return p->stop || !p->jobs.empty(); });
            if (p->stop) return;
            j = p->jobs.front(); p->jobs.pop_front();
        }
        if (j.consumed) ok = cuda_ok(cudaStreamWaitEvent(p->copy, p->consumed[j.slot], 0),
            "dense weight consumer dependency");
        const uint64_t bytes = p->layer_bytes[j.layer];
        for (uint64_t off = 0; ok && off < bytes; off += p->pin_bytes) {
            const unsigned b = stage++ % 2u;
            if (dma_live[b]) ok = cuda_ok(cudaEventSynchronize(p->dma[b]), "dense weight pinned reuse");
            const uint64_t n = std::min(p->pin_bytes, bytes - off);
            if (ok) memcpy(p->pinned[b], p->host + p->host_offset[j.layer] + off, n);
            if (ok) ok = cuda_ok(cudaMemcpyAsync(p->slots[j.slot] + off, p->pinned[b], n,
                cudaMemcpyHostToDevice, p->copy), "dense weight H2D");
            if (ok) ok = cuda_ok(cudaEventRecord(p->dma[b], p->copy), "dense weight DMA complete");
            dma_live[b] = ok;
        }
        if (ok) ok = cuda_ok(cudaEventRecord(p->ready[j.slot], p->copy), "dense weight ready");
        {
            std::lock_guard<std::mutex> lock(p->mutex);
            if (!ok) p->failed = true;
            else { p->submitted[j.slot] = j.ticket; ++p->transfers; p->h2d += bytes; }
            p->changed.notify_all();
        }
    }
}

static bool enqueue(unsigned ordinal) {
    if (!s || ordinal >= s->order.size()) return true;
    const unsigned b = ordinal % 2u;
    std::lock_guard<std::mutex> lock(s->mutex);
    if (s->failed || s->stop) return false;
    s->wanted[b] = ++s->tickets;
    s->jobs.push_back(job{s->order[ordinal], b, s->wanted[b], s->used[b]});
    s->changed.notify_all();
    return true;
}
}

static void qwen35_weight_release(void) {
    using namespace qwen35_weights;
    state *p = s; if (!p) return;
    {
        std::lock_guard<std::mutex> lock(p->mutex);
        p->stop = true; p->changed.notify_all();
    }
    if (p->worker.joinable()) p->worker.join();
    (void)cudaDeviceSynchronize();
    fprintf(stderr, "ds4: dense weight streaming forwards=%llu H2D=%.3f GiB transfers=%llu "
        "pending_waits=%llu submit_wait=%.3fms resident=%.3f GiB host=%.3f GiB ring=%.3f GiB pinned=%.3f GiB\n",
        (unsigned long long)p->forwards, p->h2d / 1073741824.0,
        (unsigned long long)p->transfers, (unsigned long long)p->pending_waits,
        p->submit_wait_ms, p->resident_bytes / 1073741824.0, p->host_bytes / 1073741824.0,
        2.0 * p->slot_bytes / 1073741824.0, 2.0 * p->pin_bytes / 1073741824.0);
    for (unsigned i = 0; i < 2; ++i) {
        cudaFree(p->slots[i]); cudaFreeHost(p->pinned[i]);
        if (p->ready[i]) cudaEventDestroy(p->ready[i]);
        if (p->consumed[i]) cudaEventDestroy(p->consumed[i]);
        if (p->dma[i]) cudaEventDestroy(p->dma[i]);
    }
    if (p->copy) cudaStreamDestroy(p->copy);
    cudaFree(p->resident); free(p->host);
    delete p; s = NULL;
}

extern "C" int ds4_gpu_qwen35_weight_stream_active(const void *map) {
    return qwen35_weights::s && qwen35_weights::s->map == map;
}
extern "C" uint64_t ds4_gpu_qwen35_weight_resident_bytes(const void *map) {
    const auto *p=qwen35_weights::s;
    return p && p->map==map ? p->resident_bytes+2u*p->slot_bytes : 0;
}

extern "C" int ds4_gpu_qwen35_weight_session_prepare(const void *map,
        uint64_t scratch_bytes, uint64_t graph_bytes) {
    if (!ds4_gpu_qwen35_weight_stream_active(map)) return 1;
    size_t free_b = 0, total_b = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_b, &total_b), "dense weight memory admission")) return 0;
    const uint64_t growth = scratch_bytes > g_cuda_tmp_bytes ? scratch_bytes - g_cuda_tmp_bytes : 0;
    const uint64_t floor = 1ull << 30;
    if (growth > free_b || graph_bytes > free_b - growth ||
        free_b - growth - graph_bytes < floor) {
        fprintf(stderr, "ds4: dense weight session exceeds VRAM budget: free=%.3f GiB "
            "graph=%.3f GiB scratch_growth=%.3f GiB minimum_free=1.000 GiB\n",
            free_b / 1073741824.0, graph_bytes / 1073741824.0, growth / 1073741824.0);
        return 0;
    }
    if (scratch_bytes && !cuda_tmp_alloc(scratch_bytes, "dense weight reserved CUDA scratch")) return 0;
    qwen35_weights::s->minimum_free = floor;
    ds4_gpu_print_memory_report(graph_bytes ? "weight streaming before graph" : "weight streaming after graph");
    return 1;
}

static const char *qwen35_weight_resolve(const void *map, uint64_t offset, uint64_t bytes) {
    using namespace qwen35_weights;
    if (!s || s->map != map || offset > s->size || bytes > s->size - offset) return NULL;
    auto it = std::upper_bound(s->spans.begin(), s->spans.end(), offset,
        [](uint64_t n, const span &r) { return n < r.offset; });
    if (it == s->spans.begin()) {
        fprintf(stderr,"ds4: dense weight offset absent: offset=%llu bytes=%llu\n",
            (unsigned long long)offset,(unsigned long long)bytes); return NULL;
    }
    const span &r = *--it;
    if (offset < r.offset || offset - r.offset > r.bytes || bytes > r.bytes - (offset - r.offset)) {
        fprintf(stderr,"ds4: dense weight span bounds: offset=%llu bytes=%llu span=%llu+%llu layer=%u\n",
            (unsigned long long)offset,(unsigned long long)bytes,(unsigned long long)r.offset,
            (unsigned long long)r.bytes,r.layer); return NULL;
    }
    if (!r.remote) return (const char *)s->resident + r.packed + offset - r.offset;
    if (r.layer != s->live_layer) {
        fprintf(stderr, "ds4: dense weight access outside its layer: wanted=%u live=%u\n", r.layer, s->live_layer);
        return NULL;
    }
    return (const char *)s->slots[s->ordinal[r.layer] % 2u] + r.packed + offset - r.offset;
}

extern "C" int ds4_gpu_qwen35_weight_stream_init(const void *map, uint64_t size,
        const ds4_qwen_weight_span *ranges, uint32_t count, uint32_t layers,
        uint64_t budget) {
    using namespace qwen35_weights;
    if (!map || !ranges || !count || !layers || layers > 128u || g_n_gpus != 1) return 0;
    /* Validate before replacing a working engine's storage. */
    std::vector<span> spans;
    std::vector<uint64_t> layer_bytes(layers, 0);
    uint64_t permanent = 0, total = 0;
    for (unsigned i = 0; i < count; ++i) {
        const auto &r = ranges[i];
        if (!r.bytes || r.bytes > UINT64_MAX - 255u || r.offset > size || r.bytes > size-r.offset ||
            (r.layer != UINT32_MAX && r.layer >= layers)) return 0;
        const uint64_t n = align(r.bytes);
        if (n > UINT64_MAX-total) return 0;
        spans.push_back(span{r.offset,r.bytes,0,r.layer,false}); total += n;
        if (r.layer == UINT32_MAX) permanent += n; else layer_bytes[r.layer] += n;
    }
    std::sort(spans.begin(),spans.end(),[](const span &a,const span &b) { return a.offset < b.offset; });
    for (unsigned i = 1; i < spans.size(); ++i)
        if (spans[i].offset < spans[i-1].offset + spans[i-1].bytes) return 0;
    if (budget < permanent) return 0;
    std::vector<bool> remote(layers,false);
    uint64_t host = 0;
    /* Evenly spaced offloaded layers leave compute windows between copies.
     * Increase the count until the actual variable-size GGUF pack fits. */
    for (unsigned n = 0; total-host > budget && n < layers; ++n) {
        std::fill(remote.begin(),remote.end(),false); host = 0;
        for (unsigned j = 0; j <= n; ++j) {
            const unsigned il = (unsigned)(((2ull*j+1u)*layers)/(2u*(n+1u)));
            remote[il] = true; host += layer_bytes[il];
        }
    }
    qwen35_weight_release();
    if (!cuda_register_model_map(map,size,false)) return 0;
    s = new state;
    s->map=map; s->size=size; s->layers=layers; s->spans=std::move(spans);
    s->layer_bytes=std::move(layer_bytes); s->ordinal.assign(layers,UINT32_MAX);
    s->host_offset.assign(layers,0); s->resident_bytes=total-host; s->host_bytes=host;
    uint64_t off = 0;
    for (unsigned il = 0; il < layers; ++il) if (remote[il]) {
        s->host_offset[il]=off; off+=s->layer_bytes[il];
        s->ordinal[il]=s->order.size(); s->order.push_back(il);
        s->slot_bytes=std::max(s->slot_bytes,s->layer_bytes[il]);
    }
    std::vector<uint64_t> cursor(layers,0);
    off = 0;
    for (span &r : s->spans) {
        r.remote=r.layer != UINT32_MAX && remote[r.layer];
        if (r.remote) { r.packed=cursor[r.layer]; cursor[r.layer]+=align(r.bytes); }
        else { r.packed=off; off+=align(r.bytes); }
    }
    if (host) s->host=(unsigned char *)calloc(1,host);
    bool ok = !host || s->host;
    if (ok && s->resident_bytes) ok = cuda_ok(cudaMalloc(&s->resident,s->resident_bytes),"dense resident weights");
    if (ok) ok = cuda_ok(cudaStreamCreateWithFlags(&s->copy,cudaStreamNonBlocking),"dense weight copy stream");
    if (ok) ok = cuda_ok(cudaGetDevice(&s->device),"dense weight device");
    s->pin_bytes=(uint64_t)cuda_parse_u32_env_clamped("DS4_CUDA_QWEN_WEIGHT_STAGE_MB",64,1,128,NULL)<<20;
    for (unsigned b=0; b<2 && ok; ++b) {
        if (host) ok = cuda_ok(cudaMalloc(&s->slots[b],s->slot_bytes),"dense weight ring slot");
        if (ok) ok = cuda_ok(cudaMallocHost(&s->pinned[b],s->pin_bytes),"dense weight pinned stage");
        if (ok) ok = cuda_ok(cudaEventCreateWithFlags(&s->ready[b],cudaEventDisableTiming),"dense weight ready event");
        if (ok) ok = cuda_ok(cudaEventCreateWithFlags(&s->consumed[b],cudaEventDisableTiming),"dense weight consume event");
        if (ok) ok = cuda_ok(cudaEventCreateWithFlags(&s->dma[b],cudaEventDisableTiming),"dense weight DMA event");
    }
    for (const span &r : s->spans) {
        if (!ok) break;
        const char *src=(const char *)map+r.offset;
        if (r.remote) memcpy(s->host+s->host_offset[r.layer]+r.packed,src,r.bytes);
        else for (uint64_t done=0; ok && done<r.bytes; done+=s->pin_bytes) {
            const uint64_t n=std::min(s->pin_bytes,r.bytes-done);
            memcpy(s->pinned[0],src+done,n);
            ok=cuda_ok(cudaMemcpy(s->resident+r.packed+done,s->pinned[0],n,cudaMemcpyHostToDevice),
                "dense resident weight upload");
        }
    }
    if (!ok) { qwen35_weight_release(); return 0; }
    cuda_model_discard_source_pages(map,size,0,size); cuda_model_drop_file_pages(0,size);
    fprintf(stderr,"ds4: dense weight plan resident=%.3f GiB host=%.3f GiB ring=%.3f GiB "
        "offloaded_layers=%zu/%u; KV and recurrent state stay in VRAM\n",
        s->resident_bytes/1073741824.0,host/1073741824.0,2.0*s->slot_bytes/1073741824.0,
        s->order.size(),layers);
    if (host) {
        try { s->worker=std::thread(worker,s); }
        catch (...) {
            fprintf(stderr,"ds4: dense weight copy worker creation failed\n");
            qwen35_weight_release(); return 0;
        }
    }
    return 1;
}

extern "C" int ds4_gpu_qwen35_weight_forward_begin(const void *map) {
    using namespace qwen35_weights;
    if (!s || s->map != map) return 1;
    if (s->minimum_free) {
        size_t free_b = 0, total_b = 0;
        if (!cuda_ok(cudaMemGetInfo(&free_b, &total_b), "dense weight forward memory check")) return 0;
        if (free_b < s->minimum_free) {
            fprintf(stderr, "ds4: dense weight forward stopped before paging: free=%.3f GiB "
                "minimum_free=%.3f GiB\n", free_b / 1073741824.0,
                s->minimum_free / 1073741824.0);
            return 0;
        }
    }
    s->live_layer=UINT32_MAX; ++s->forwards;
    return enqueue(0) && enqueue(1);
}
extern "C" int ds4_gpu_qwen35_weight_layer_begin(const void *map, uint32_t layer) {
    using namespace qwen35_weights;
    if (!s || s->map != map) return 1;
    if (layer>=s->layers) return 0;
    s->live_layer=layer;
    if (s->ordinal[layer]==UINT32_MAX) return 1;
    const unsigned b=s->ordinal[layer]%2u;
    const double t=cuda_wall_sec();
    {
        std::unique_lock<std::mutex> lock(s->mutex);
        s->changed.wait(lock,[&] { return s->failed || s->stop || s->submitted[b]==s->wanted[b]; });
        s->submit_wait_ms+=(cuda_wall_sec()-t)*1000.0;
        if (s->failed || s->stop) return 0;
    }
    const cudaError_t status=cudaEventQuery(s->ready[b]);
    if (status==cudaErrorNotReady) ++s->pending_waits;
    else if (!cuda_ok(status,"dense weight ready query")) return 0;
    return cuda_ok(cudaStreamWaitEvent(0,s->ready[b],0),"dense weight compute dependency");
}
extern "C" int ds4_gpu_qwen35_weight_layer_end(const void *map, uint32_t layer) {
    using namespace qwen35_weights;
    if (!s || s->map != map) return 1;
    if (layer>=s->layers) return 0;
    const unsigned ordinal=s->ordinal[layer];
    if (ordinal==UINT32_MAX) return 1;
    const unsigned b=ordinal%2u;
    if (!cuda_ok(cudaEventRecord(s->consumed[b],0),"dense weight consumer complete")) return 0;
    s->used[b]=true;
    return enqueue(ordinal+2u);
}
