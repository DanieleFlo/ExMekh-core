/* Included inside qwen4_cuda. Logical FP16 tensors backed by ordinary RAM;
 * only selected four-token pages occupy VRAM. All DMA is explicit. */
struct kv_cache {
    ds4_gpu_tensor *k, *v, *append_k, *append_v, *table;
    unsigned ctx, resident, max_tokens, Hkv, D, slots, refs;
    uint64_t row_bytes, host_bytes, valid_bytes[2], epoch, hits, misses, uploaded;
    unsigned hand;
    bool direct, table_dirty;
    uint64_t direct_rows;
    uint64_t groups, grouped_rows, staged_rows, joined_decode_publications, direct_append_deferred;
    void *host[2];
    void *device[2];
    std::vector<int> page_slot, slot_page, needed, missing;
    std::vector<uint64_t> marks, stamps;
};
static std::unordered_map<const void *, kv_cache *> kv_caches;
/* One deferred append shares the existing bounded DMA pool with the QSA
 * readback. Only device work may intervene; every logical host access,
 * rewind, next append and release publishes it before using its bytes. */
static kv_cache *kv_pending_append;
static int kv_publish_append(void) {
    if (!kv_pending_append) return 1;
    const int ok = transfer_batch_end();
    kv_pending_append = NULL;
    return ok;
}
static std::vector<int> kv_ids;
static std::vector<unsigned> kv_counts;
static uint64_t kv_total_hits, kv_total_misses, kv_total_uploaded;
/* Reuse one identity layout after its attention reader, as in Strata. An
 * optional second buffer also permits DMA during the attention itself.
 * Both variants protect the producer and consumer with distinct events. */
static ds4_gpu_tensor *kv_prefill_stage[2][2];
static uint64_t kv_prefill_stage_capacity;
static cudaStream_t kv_prefill_copy_stream;
static cudaEvent_t kv_prefill_ready[2], kv_prefill_consumed[2];
static kv_cache *kv_prefill_owner[2];
static unsigned kv_prefill_rows[2], kv_prefill_hand;

static bool kv_prefill_stage_requested(void) {
    const char *policy = getenv("DS4_CUDA_QWEN4_EXPERT_POLICY");
    return ds4_env_flag_enabled("DS4_CUDA_QWEN4_KV_PREFILL_STAGE", policy && !strcmp(policy, "stable"));
}

static bool kv_prefill_prefetch_requested(void) {
    const char *policy = getenv("DS4_CUDA_QWEN4_EXPERT_POLICY");
    return ds4_env_flag_enabled("DS4_CUDA_QWEN4_KV_PREFETCH", policy && !strcmp(policy, "stable"));
}

static int kv_prefill_invalidate(kv_cache *c) {
    for (unsigned b = 0; b < 2; ++b) if (!c || kv_prefill_owner[b] == c) {
        if (kv_prefill_ready[b] && !cuda_ok(cudaEventSynchronize(kv_prefill_ready[b]), "Qwen KV prefetch invalidate")) return 0;
        kv_prefill_owner[b] = NULL; kv_prefill_rows[b] = 0;
    }
    return 1;
}

static void kv_prefill_stage_prepare(uint64_t bytes, unsigned ctx) {
    if (!kv_prefill_stage_requested() || ctx > 262144u || bytes <= kv_prefill_stage_capacity) return;
    if (!cuda_ok(cudaStreamSynchronize(cuda_decode_stream()), "Qwen KV prefill staging lifetime")) return;
    if (kv_prefill_copy_stream && !cuda_ok(cudaStreamSynchronize(kv_prefill_copy_stream), "Qwen KV prefetch resize")) return;
    ds4_gpu_tensor *next[2][2] = {};
    const unsigned buffers = kv_prefill_prefetch_requested() ?
        cuda_parse_u32_env_clamped("DS4_CUDA_QWEN4_KV_PREFETCH_BUFFERS", 1u, 1u, 2u, NULL) : 1u;
    bool ok = true;
    for (unsigned b = 0; b < buffers && ok; ++b) for (unsigned p = 0; p < 2 && ok; ++p) {
        next[b][p] = ds4_gpu_tensor_alloc(bytes); ok = next[b][p] != NULL;
    }
    if (ok && kv_prefill_prefetch_requested()) {
        if (!kv_prefill_copy_stream) ok = cuda_ok(cudaStreamCreateWithFlags(&kv_prefill_copy_stream, cudaStreamNonBlocking), "Qwen KV prefetch stream");
        for (unsigned b = 0; b < buffers && ok; ++b) {
            if (!kv_prefill_ready[b]) ok = cuda_ok(cudaEventCreateWithFlags(&kv_prefill_ready[b], cudaEventDisableTiming), "Qwen KV prefetch ready");
            if (ok && !kv_prefill_consumed[b]) ok = cuda_ok(cudaEventCreateWithFlags(&kv_prefill_consumed[b], cudaEventDisableTiming), "Qwen KV prefetch consumed");
        }
    }
    if (!ok) { for (auto &pair : next) for (auto *t : pair) ds4_gpu_tensor_free(t); return; }
    for (unsigned b = 0; b < 2; ++b) for (unsigned p = 0; p < 2; ++p) {
        ds4_gpu_tensor_free(kv_prefill_stage[b][p]); kv_prefill_stage[b][p] = next[b][p];
    }
    kv_prefill_owner[0] = kv_prefill_owner[1] = NULL;
    kv_prefill_rows[0] = kv_prefill_rows[1] = kv_prefill_hand = 0;
    kv_prefill_stage_capacity = bytes;
    g_model_auto_cache_limit = 0;
}

static kv_cache *kv_lookup(const void *ptr) {
    const auto it = kv_caches.find(ptr);
    return it == kv_caches.end() ? NULL : it->second;
}
static bool kv_streamed(const void *ptr) { return kv_lookup(ptr) != NULL; }

static void kv_drop_page(kv_cache *c, unsigned page) {
    const int slot = c->page_slot[page];
    if (slot >= 0) { c->slot_page[(unsigned)slot] = -1; c->page_slot[page] = -1; c->table_dirty = true; }
}

static int kv_truncate_cache(kv_cache *c, unsigned rows) {
    if (!kv_publish_append()) return 0;
    const uint64_t bytes = std::min((uint64_t)rows * c->row_bytes, c->host_bytes);
    const uint64_t old = std::max(c->valid_bytes[0], c->valid_bytes[1]);
    /* Appending to the current frontier leaves the prefetched historical
     * prefix immutable. A rewind must drain and discard speculative copies. */
    if (bytes < old && !kv_prefill_invalidate(c)) return 0;
    for (unsigned page = rows / 4u; (uint64_t)page * 4u * c->row_bytes < old && page < c->page_slot.size(); page++)
        kv_drop_page(c, page);
    c->valid_bytes[0] = std::min(c->valid_bytes[0], bytes);
    c->valid_bytes[1] = std::min(c->valid_bytes[1], bytes);
    /* A completely reset session can append into the identity slots again.
     * Logical checkpoint writes explicitly leave this mode in kv_io. */
    if (!rows) c->direct = true;
    return 1;
}

static int kv_io(const void *ptr, uint64_t offset, void *data, uint64_t bytes, bool write) {
    kv_cache *c = kv_lookup(ptr);
    if (!c) return -1;
    if (!kv_publish_append()) return 0;
    const unsigned plane = ptr == c->device[0] ? 0u : 1u;
    const uint64_t logical = (uint64_t)c->ctx * c->row_bytes;
    if (!data || offset > logical || bytes > logical - offset ||
        (!write && (offset > c->valid_bytes[plane] || bytes > c->valid_bytes[plane] - offset))) return 0;
    if (!cuda_ok(cudaStreamSynchronize(cuda_decode_stream()), "Qwen KV host access lifetime")) return 0;
    if (write) {
        if (!kv_prefill_invalidate(c)) return 0;
        /* Logical checkpoint writes invalidate physical identity. Rebuild on
         * demand even when the restored frontier fits the resident budget. */
        c->direct = false;
        if (bytes) for (uint64_t page = offset / (c->row_bytes * 4u);
            page <= (offset + bytes - 1u) / (c->row_bytes * 4u); page++) kv_drop_page(c, (unsigned)page);
        memcpy((char *)c->host[plane] + offset, data, bytes);
        c->valid_bytes[plane] = std::max(c->valid_bytes[plane], offset + bytes);
    } else memcpy(data, (const char *)c->host[plane] + offset, bytes);
    return 1;
}

static void kv_release(const void *ptr) {
    const auto it = kv_caches.find(ptr);
    if (it == kv_caches.end()) return;
    kv_cache *c = it->second;
    (void)kv_publish_append();
    kv_caches.erase(it);
    if (--c->refs) return;
    (void)cudaStreamSynchronize(cuda_decode_stream());
    (void)kv_prefill_invalidate(c);
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_CACHE_STATS", 0))
        fprintf(stderr, "ds4: Qwen KV ctx=%u resident=%u direct_rows=%llu page_hits=%llu page_misses=%llu H2D_bytes=%llu groups=%llu grouped_rows=%llu staged_rows=%llu joined_decode_publications=%llu direct_append_deferred=%llu\n",
            c->ctx, c->resident, (unsigned long long)c->direct_rows,
            (unsigned long long)c->hits, (unsigned long long)c->misses, (unsigned long long)c->uploaded,
            (unsigned long long)c->groups, (unsigned long long)c->grouped_rows, (unsigned long long)c->staged_rows,
            (unsigned long long)c->joined_decode_publications, (unsigned long long)c->direct_append_deferred);
    kv_host_reserved -= c->host_bytes * 2u;
    free(c->host[0]); free(c->host[1]);
    ds4_gpu_tensor_free(c->append_k); ds4_gpu_tensor_free(c->append_v); ds4_gpu_tensor_free(c->table);
    delete c;
    if (kv_caches.empty()) {
        if (kv_prefill_copy_stream) (void)cudaStreamSynchronize(kv_prefill_copy_stream);
        for (unsigned b = 0; b < 2; ++b) {
            for (unsigned p = 0; p < 2; ++p) { ds4_gpu_tensor_free(kv_prefill_stage[b][p]); kv_prefill_stage[b][p] = NULL; }
            if (kv_prefill_ready[b]) (void)cudaEventDestroy(kv_prefill_ready[b]);
            if (kv_prefill_consumed[b]) (void)cudaEventDestroy(kv_prefill_consumed[b]);
            kv_prefill_ready[b] = kv_prefill_consumed[b] = NULL;
        }
        if (kv_prefill_copy_stream) (void)cudaStreamDestroy(kv_prefill_copy_stream);
        kv_prefill_copy_stream = NULL; kv_prefill_stage_capacity = 0; kv_prefill_hand = 0;
    }
}

static int kv_append(kv_cache *c, unsigned pos, unsigned T) {
    transfer_workload_rows = T;
    if (pos > c->ctx || T > c->ctx - pos || T > c->max_tokens) return 0;
    if ((uint64_t)pos * c->row_bytes > std::min(c->valid_bytes[0], c->valid_bytes[1])) {
        fprintf(stderr, "ds4: Qwen streamed KV append leaves a gap at %u\n", pos); return 0;
    }
    if (!kv_truncate_cache(c, pos)) return 0;
    const uint64_t bytes = (uint64_t)T * c->row_bytes, offset = (uint64_t)pos * c->row_bytes;
    if (pos + T > c->resident) c->direct = false;
    if (c->direct) {
        /* Before the first spill every logical page has an identity slot.
         * Populate new cells from their original device FP16 bytes, while the
         * ordinary RAM mirror still supports legacy logical checkpoints. */
        if (!cuda_ok(cudaMemcpyAsync((char *)c->k->ptr + offset, c->append_k->ptr, bytes,
                                      cudaMemcpyDeviceToDevice, cuda_decode_stream()), "Qwen resident K append") ||
            !cuda_ok(cudaMemcpyAsync((char *)c->v->ptr + offset, c->append_v->ptr, bytes,
                                      cudaMemcpyDeviceToDevice, cuda_decode_stream()), "Qwen resident V append")) return 0;
        for (unsigned page = pos / 4u; page < (pos + T + 3u) / 4u; page++) {
            c->page_slot[page] = (int)page; c->slot_page[page] = (int)page;
        }
        c->table_dirty = true;
    }
    /* K and V become host-visible at the same layer boundary. Publish both
     * through one bounded DMA batch rather than fencing each tiny decode
     * projection separately. Always drain on error before source reuse. */
    if (!transfer_batch_begin(cuda_decode_stream())) return 0;
    const int downloaded = transfer_download((char *)c->host[0] + offset, c->append_k->ptr, bytes, cuda_decode_stream()) &&
        transfer_download((char *)c->host[1] + offset, c->append_v->ptr, bytes, cuda_decode_stream());
    const bool defer = downloaded && T <= 8u &&
        ds4_env_flag_enabled("DS4_CUDA_QWEN4_KV_DECODE_BATCH", 0);
    if (defer) {
        kv_pending_append = c;
        /* Identity attention reads device K/V directly. Record the producer
         * event now, without waiting for its RAM mirror: a later publication
         * must not fence kernels queued after this append. The shared pinned
         * pool continues to protect the payload until that event completes. */
        if (c->direct) {
            const int sealed = transfer_batch_flush(); transfer_batch_on = false;
            if (!sealed) { (void)kv_publish_append(); return 0; }
            c->direct_append_deferred++;
        }
    }
    else {
        const int drained = transfer_batch_end();
        if (!downloaded || !drained) return 0;
    }
    c->valid_bytes[0] = c->valid_bytes[1] = offset + bytes;
    return 1;
}

/* Pack scattered host pages into contiguous physical slot runs. A run costs
 * two DMA submissions, rather than one submission per selected key/page. */
static int kv_upload_pages(kv_cache *c) {
    const uint64_t page_bytes = c->row_bytes * 4u;
    std::sort(c->missing.begin(), c->missing.end(), [&](int a, int b) { return c->page_slot[(unsigned)a] < c->page_slot[(unsigned)b]; });
    for (size_t begin = 0; begin < c->missing.size();) {
        const unsigned first = (unsigned)c->page_slot[(unsigned)c->missing[begin]];
        size_t end = begin + 1;
        while (end < c->missing.size() &&
            c->page_slot[(unsigned)c->missing[end]] == (int)(first + end - begin) &&
            (end + 1u - begin) * page_bytes <= transfer_capacity) end++;
        const uint64_t bytes = (end - begin) * page_bytes;
        for (unsigned plane = 0; plane < 2; plane++) {
            const unsigned i = transfer_next++ % 2u;
            if (!cuda_ok(cudaEventSynchronize(transfer_done[i]), "Qwen KV staging reuse")) return 0;
            for (size_t j = begin; j < end; j++)
                memcpy((char *)transfer_host[i] + (j - begin) * page_bytes,
                       (const char *)c->host[plane] + (uint64_t)c->missing[j] * page_bytes, page_bytes);
            void *dst = (char *)(plane ? c->v->ptr : c->k->ptr) + (uint64_t)first * page_bytes;
            if (!cuda_ok(cudaMemcpyAsync(dst, transfer_host[i], bytes, cudaMemcpyHostToDevice, cuda_decode_stream()), "Qwen KV selected pages") ||
                !cuda_ok(cudaEventRecord(transfer_done[i], cuda_decode_stream()), "Qwen KV pages copied")) return 0;
            c->uploaded += bytes; transfer_h2d_bytes += bytes;
            kv_total_uploaded += bytes;
        }
        begin = end;
    }
    const int ok = transfer_upload(c->table->ptr, c->page_slot.data(), c->page_slot.size() * sizeof(int), cuda_decode_stream());
    if (ok) c->table_dirty = false;
    return ok;
}

static int kv_resolve(kv_cache *c) {
    c->missing.clear();
    for (int page : c->needed) {
        const int slot = c->page_slot[(unsigned)page];
        if (slot >= 0) { c->stamps[(unsigned)slot] = c->epoch; c->hits++; kv_total_hits++; }
        else c->missing.push_back(page);
    }
    for (int page : c->missing) {
        unsigned scanned = 0;
        while (scanned < c->slots && c->stamps[c->hand] == c->epoch) { c->hand = (c->hand + 1u) % c->slots; scanned++; }
        if (scanned == c->slots) return 0;
        const unsigned slot = c->hand; c->hand = (c->hand + 1u) % c->slots;
        if (c->slot_page[slot] >= 0) c->page_slot[(unsigned)c->slot_page[slot]] = -1;
        c->slot_page[slot] = page; c->page_slot[(unsigned)page] = (int)slot;
        c->stamps[slot] = c->epoch; c->misses++;
        kv_total_misses++;
    }
    if (!c->missing.empty()) return kv_upload_pages(c);
    if (!c->table_dirty) return 1;
    const int ok = transfer_upload(c->table->ptr, c->page_slot.data(), c->page_slot.size() * sizeof(int), cuda_decode_stream());
    if (ok) c->table_dirty = false;
    return ok;
}

static int kv_dispatch(kv_cache *c, ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *count,
        ds4_gpu_tensor *partial, unsigned begin, unsigned T, unsigned original_T,
        unsigned H, unsigned Hkv, unsigned D, unsigned pos0, bool sparse, unsigned stride,
        unsigned keys, float scale, const __half *stage_k = NULL, const __half *stage_v = NULL) {
    const float *qp = (const float *)q->ptr + (uint64_t)begin * H * D;
    const float *gp = (const float *)gate->ptr + (uint64_t)begin * H * D;
    float *op = (float *)out->ptr + (uint64_t)begin * H * D;
    const int *ids = sparse ? (const int *)sel->ptr + (uint64_t)begin * stride : NULL;
    const unsigned *counts = sparse ? (const unsigned *)count->ptr + begin : NULL;
    const int *pages = stage_k || c->direct ? NULL : (const int *)c->table->ptr;
    const __half *kp = stage_k ? stage_k : (const __half *)c->k->ptr;
    const __half *vp = stage_v ? stage_v : (const __half *)c->v->ptr;
    const cudaStream_t stream = cuda_decode_stream();
    if (!partial && original_T >= 32 && D == 256 && H / Hkv <= 16 &&
        ds4_cuda_attn_tokentile_arch_ok() && ::qwen4_group_attention_arch_ok() &&
        !g_quality_mode && !getenv("DS4_QWEN4_NO_ATTN_MM")) {
        attention_group<<<dim3(Hkv,T),256,0,stream>>>(op, qp, gp, kp,
            vp, ids, counts, H, Hkv, pos0 + begin, stride, sparse, scale, pages);
        return launched();
    }
    const unsigned splits = partial ? std::min(64u, (keys + 31u) / 32u) : 1u;
    if (partial && !tensor(partial, (uint64_t)original_T * H * splits * (D + 2u) * 4u)) return 0;
    float *pp = partial ? (float *)partial->ptr + (uint64_t)begin * H * splits * (D + 2u) : NULL;
    const dim3 grid((H + 3u) / 4u, T, splits);
#define QWEN_STREAM_ATTN(DIM) attention<DIM><<<grid,128,0,stream>>>(op, pp, qp, gp, \
        kp, vp, ids, counts, H, Hkv, \
        pos0 + begin, stride, sparse, splits, (keys + splits - 1u) / splits, scale, pages)
    if (D == 32) { QWEN_STREAM_ATTN(32); }
    else if (D == 128) { QWEN_STREAM_ATTN(128); }
    else { QWEN_STREAM_ATTN(256); }
#undef QWEN_STREAM_ATTN
    if (!launched()) return 0;
    if (splits > 1) attn_merge<<<dim3(H,T),D,0,stream>>>(op, pp, gp, H, D, splits);
    return launched();
}

static int kv_attention(kv_cache *c, ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *count,
        ds4_gpu_tensor *partial, unsigned T, unsigned H, unsigned Hkv, unsigned D,
        unsigned pos0, bool sparse, unsigned stride, float scale) {
    transfer_workload_rows = T;
    /* Keep append and selected IDs in one batch only for this layer. Dense,
     * direct and prefill paths publish before their usual host accesses. */
    const bool direct_pending = kv_pending_append == c && c->direct && pos0 + (uint64_t)T <= c->resident;
    const bool joined = kv_pending_append == c && !c->direct && sparse && T <= 8u;
    if (!joined && !direct_pending && !kv_publish_append()) return 0;
    if (Hkv != c->Hkv || D != c->D || T > c->max_tokens || pos0 > c->ctx || T > c->ctx - pos0 ||
        ((uint64_t)pos0 + T) * c->row_bytes > std::min(c->valid_bytes[0], c->valid_bytes[1])) return 0;
    if (c->direct && pos0 + T <= c->resident) {
        c->direct_rows += T;
        return kv_dispatch(c, out, q, gate, sel, count, partial, 0, T, T,
                           H, Hkv, D, pos0, sparse, stride, sparse ? stride : pos0 + T, scale);
    }
    const uint64_t prefix_bytes = ((uint64_t)pos0 + T) * c->row_bytes;
    if (T >= 32 && kv_prefill_stage_requested() && prefix_bytes <= kv_prefill_stage_capacity &&
        kv_prefill_stage[0][0] && kv_prefill_stage[0][1]) {
        /* Original selection and original batched attention. Copy the whole
         * layer's immutable prefix once, avoiding CPU QSA readback and tiny
         * query groups when their union overflows the resident page cache. */
        unsigned b = 0, copied = 0;
        for (unsigned i = 0; i < 2; ++i) if (kv_prefill_owner[i] == c && kv_prefill_rows[i] == pos0) {
            b = i; copied = pos0; break;
        }
        if (kv_prefill_ready[b] &&
            (!cuda_ok(cudaStreamWaitEvent(cuda_decode_stream(), kv_prefill_ready[b], 0), "Qwen KV prefetched prefix ready") ||
             !cuda_ok(cudaStreamWaitEvent(cuda_decode_stream(), kv_prefill_consumed[b], 0), "Qwen KV prefill buffer consumer"))) return 0;
        const uint64_t offset = (uint64_t)copied * c->row_bytes, bytes = prefix_bytes - offset;
        if (!transfer_upload((char *)kv_prefill_stage[b][0]->ptr + offset, (char *)c->host[0] + offset, bytes, cuda_decode_stream()) ||
            !transfer_upload((char *)kv_prefill_stage[b][1]->ptr + offset, (char *)c->host[1] + offset, bytes, cuda_decode_stream())) return 0;
        c->staged_rows += T; c->uploaded += bytes * 2u; kv_total_uploaded += bytes * 2u;
        kv_prefill_owner[b] = NULL; kv_prefill_rows[b] = 0;
        const int ok = kv_dispatch(c, out, q, gate, sel, count, partial, 0, T, T, H, Hkv, D, pos0,
            sparse, stride, sparse ? stride : pos0 + T, scale,
            (const __half *)kv_prefill_stage[b][0]->ptr, (const __half *)kv_prefill_stage[b][1]->ptr);
        return ok && (!kv_prefill_consumed[b] || cuda_ok(cudaEventRecord(kv_prefill_consumed[b], cuda_decode_stream()), "Qwen KV prefill consumed"));
    }
    if (sparse) {
        if (kv_ids.size() < (uint64_t)T * stride || kv_counts.size() < T) return 0;
        if (!joined && !transfer_batch_begin(cuda_decode_stream())) return 0;
        const int downloaded = transfer_download(kv_ids.data(), sel->ptr, (uint64_t)T * stride * sizeof(int), cuda_decode_stream()) &&
            transfer_download(kv_counts.data(), count->ptr, T * sizeof(unsigned), cuda_decode_stream());
        const int drained = transfer_batch_end();
        if (joined) { kv_pending_append = NULL; c->joined_decode_publications++; }
        if (!downloaded || !drained) return 0;
    }
    const unsigned keys = sparse ? stride : pos0 + T;
    for (unsigned begin = 0; begin < T;) {
        if (++c->epoch == 0) { std::fill(c->marks.begin(), c->marks.end(), 0); std::fill(c->stamps.begin(), c->stamps.end(), 0); c->epoch = 1; }
        c->needed.clear();
        unsigned end = begin;
        for (; end < T; end++) {
            const unsigned n = sparse ? kv_counts[end] : pos0 + end + 1u;
            if (sparse && n > stride) return 0;
            const size_t before = c->needed.size();
            for (unsigned j = 0; j < n; j++) {
                const unsigned pos = sparse ? (unsigned)kv_ids[(uint64_t)end * stride + j] : j;
                if (pos > pos0 + end) continue; /* same causal masking as the resident kernels */
                const unsigned page = pos / 4u;
                if (page >= c->page_slot.size()) return 0;
                if (c->marks[page] != c->epoch) { c->marks[page] = c->epoch; c->needed.push_back((int)page); }
            }
            if (c->needed.size() > c->slots) {
                for (size_t j = before; j < c->needed.size(); j++) c->marks[(unsigned)c->needed[j]] = 0;
                c->needed.resize(before);
                break;
            }
        }
        if (end == begin) { fprintf(stderr, "ds4: Qwen KV resident budget cannot hold one query's selected pages\n"); return 0; }
        c->groups++; c->grouped_rows += end - begin;
        if (!kv_resolve(c) || !kv_dispatch(c, out, q, gate, sel, count, partial, begin, end - begin,
            T, H, Hkv, D, pos0, sparse, stride, keys, scale)) return 0;
        begin = end;
    }
    return 1;
}

extern "C" int ds4_gpu_qwen4_kv_alloc(ds4_gpu_tensor **k, ds4_gpu_tensor **v,
        unsigned ctx, unsigned resident, unsigned max_tokens, unsigned max_select, unsigned Hkv, unsigned D) {
    if (!k || !v || k == v) return 0;
    *k = *v = NULL;
    if (!g_qwen4_tiered_cache || g_n_gpus > 1 || !ctx || ctx > UINT_MAX - 3u || !resident || resident % 4u ||
        !max_tokens || max_tokens > ctx || !max_select || !Hkv || Hkv > 512 || (D != 32 && D != 128 && D != 256)) return 0;
    kv_cache *c = new kv_cache{};
    c->ctx = ctx; c->resident = std::min((ctx + 3u) & ~3u, resident); c->max_tokens = max_tokens;
    c->Hkv = Hkv; c->D = D; c->slots = c->resident / 4u; c->row_bytes = (uint64_t)Hkv * D * 2u;
    c->host_bytes = ((uint64_t)ctx + 3u) / 4u * 4u * c->row_bytes;
    if (max_tokens >= 32 && ctx > resident) kv_prefill_stage_prepare(c->host_bytes, ctx);
    c->host[0] = calloc(1, c->host_bytes); c->host[1] = calloc(1, c->host_bytes);
    c->k = ds4_gpu_tensor_alloc((uint64_t)c->resident * c->row_bytes);
    c->v = ds4_gpu_tensor_alloc((uint64_t)c->resident * c->row_bytes);
    c->append_k = ds4_gpu_tensor_alloc((uint64_t)max_tokens * c->row_bytes);
    c->append_v = ds4_gpu_tensor_alloc((uint64_t)max_tokens * c->row_bytes);
    const unsigned blocks = (ctx + 3u) / 4u;
    c->table = ds4_gpu_tensor_alloc((uint64_t)blocks * sizeof(int));
    if (!c->host[0] || !c->host[1] || !c->k || !c->v || !c->append_k || !c->append_v || !c->table || !transfer_prepare()) {
        free(c->host[0]); free(c->host[1]);
        ds4_gpu_tensor_free(c->k); ds4_gpu_tensor_free(c->v);
        ds4_gpu_tensor_free(c->append_k); ds4_gpu_tensor_free(c->append_v); ds4_gpu_tensor_free(c->table); delete c; return 0;
    }
    c->page_slot.assign(blocks, -1); c->slot_page.assign(c->slots, -1);
    c->marks.assign(blocks, 0); c->stamps.assign(c->slots, 0);
    c->needed.reserve(blocks); c->missing.reserve(c->slots);
    if (kv_ids.size() < (uint64_t)max_tokens * max_select) kv_ids.resize((uint64_t)max_tokens * max_select);
    if (kv_counts.size() < max_tokens) kv_counts.resize(max_tokens);
    (void)cudaMemsetAsync(c->table->ptr, 0xff, blocks * sizeof(int), cuda_decode_stream());
    c->k->bytes = c->v->bytes = (uint64_t)ctx * c->row_bytes;
    c->device[0] = c->k->ptr; c->device[1] = c->v->ptr;
    c->refs = 2; c->direct = true; kv_host_reserved += c->host_bytes * 2u;
    kv_caches[c->k->ptr] = c; kv_caches[c->v->ptr] = c;
    *k = c->k; *v = c->v; return 1;
}

extern "C" int ds4_gpu_qwen4_kv_truncate(const ds4_gpu_tensor *k, unsigned rows) {
    kv_cache *c = k ? kv_lookup(k->ptr) : NULL;
    if (!c) return 1;
    if (rows > c->ctx || !cuda_ok(cudaStreamSynchronize(cuda_decode_stream()), "Qwen KV frontier lifetime")) return 0;
    return kv_truncate_cache(c, rows);
}

extern "C" int ds4_gpu_qwen4_kv_prefetch(const ds4_gpu_tensor *key, unsigned pos0, unsigned T) {
    transfer_workload_rows = T;
    if (!kv_publish_append()) return 0;
    kv_cache *c = key ? kv_lookup(key->ptr) : NULL;
    if (!c || T < 32u || !kv_prefill_stage_requested() || !kv_prefill_prefetch_requested() ||
        pos0 + (uint64_t)T <= c->resident || !kv_prefill_copy_stream ||
        !kv_prefill_stage[0][0] || !kv_prefill_stage[0][1]) return 1;
    const uint64_t bytes = (uint64_t)pos0 * c->row_bytes;
    if (pos0 > c->ctx || T > c->ctx - pos0 || bytes > kv_prefill_stage_capacity ||
        bytes > std::min(c->valid_bytes[0], c->valid_bytes[1])) return 0;
    for (unsigned b = 0; b < 2; ++b)
        if (kv_prefill_owner[b] == c && kv_prefill_rows[b] == pos0) return 1;
    const unsigned buffers = kv_prefill_stage[1][0] && kv_prefill_stage[1][1] ? 2u : 1u;
    const unsigned b = kv_prefill_hand++ % buffers;
    if (!cuda_ok(cudaStreamWaitEvent(kv_prefill_copy_stream, kv_prefill_consumed[b], 0), "Qwen KV next-layer buffer reuse") ||
        !transfer_upload(kv_prefill_stage[b][0]->ptr, c->host[0], bytes, kv_prefill_copy_stream) ||
        !transfer_upload(kv_prefill_stage[b][1]->ptr, c->host[1], bytes, kv_prefill_copy_stream) ||
        !cuda_ok(cudaEventRecord(kv_prefill_ready[b], kv_prefill_copy_stream), "Qwen KV next-layer prefix copied")) return 0;
    kv_prefill_owner[b] = c; kv_prefill_rows[b] = pos0;
    c->uploaded += bytes * 2u; kv_total_uploaded += bytes * 2u;
    return 1;
}
