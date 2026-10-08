/* Whole-expert prefill groups. The original GU/down kernels and final slot
 * reduction are unchanged; only their order and copy dependencies differ. */
namespace qwen4_cuda {
struct ring_ahead_state {
    const void *map = NULL;
    uint64_t size = 0, offsets[3] = {}, bytes[3] = {};
    unsigned NE = 0, T = 0, width = 0, groups = 0;
    bool active = false;
    std::vector<const char *> pointers;
    std::vector<uint64_t> copied;
};
static ring_ahead_state ring_ahead;
static uint64_t ring_ahead_groups, ring_ahead_bytes, ring_ahead_unused;
/* Engine hints only: the real router always decides which slots compute.
 * A domain change merely defers an unpredicted transfer to its normal path. */
static std::unordered_map<uint64_t, std::vector<unsigned char>> ring_last_routes;

static unsigned ring_width(unsigned NE, uint64_t gb, uint64_t db) {
    uint64_t budget = 128ull * 1048576ull;
    const char *env = getenv("DS4_CUDA_QWEN4_PREFILL_RING_MB");
    if (env && *env) {
        char *end = NULL; const uint64_t mb = strtoull(env, &end, 10);
        if (*env != '-' && end != env && !*end && mb >= 1u && mb <= 1024u)
            budget = mb * 1048576ull;
    }
    const uint64_t blob = 2u*gb + db;
    return blob && blob <= budget/2u ? (unsigned)std::min<uint64_t>(NE, budget/(2u*blob)) : 0;
}

static int ring_ahead_clear(void) {
    if (ring_ahead.active && !cuda_ok(cudaStreamSynchronize(expert_copy_stream),
                                     "Qwen abandoned ahead uploads drained")) return 0;
    ring_ahead.active = false;
    return 1;
}
static void ring_ahead_reset(void) {
    (void)ring_ahead_clear();
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_CACHE_STATS", 0))
        fprintf(stderr, "ds4: Qwen prefill ahead_groups=%llu ahead_bytes=%llu unrouted_bytes=%llu\n",
            (unsigned long long)ring_ahead_groups, (unsigned long long)ring_ahead_bytes,
            (unsigned long long)ring_ahead_unused);
    ring_ahead = ring_ahead_state{};
    ring_last_routes.clear();
    ring_ahead_groups = ring_ahead_bytes = ring_ahead_unused = 0;
}
} // namespace qwen4_cuda

extern "C" int ds4_gpu_qwen4_moe_prefill_ahead(const void *map, uint64_t size,
        uint64_t go, uint64_t uo, uint64_t doff, uint32_t gu_type, uint32_t down_type,
        uint32_t NE, uint32_t T, uint32_t E, uint32_t F) {
    using namespace qwen4_cuda;
    if (!ring_ahead_clear()) return 0;
    if (!ds4_env_flag_enabled("DS4_CUDA_QWEN4_PREFILL_AHEAD", 0) ||
        !ds4_env_flag_enabled("DS4_CUDA_QWEN4_PREFILL_RING", 0) ||
        !tiered_cache(map) || !stable_requested() ||
        (ds4_env_flag_enabled("DS4_CUDA_QWEN4_HYBRID_MOE", 0) &&
         ds4_env_flag_enabled("DS4_CUDA_QWEN4_HYBRID_PREFILL", 1))) return 1;
    unsigned minimum = 1024u;
    const char *env = getenv("DS4_CUDA_QWEN4_PREFILL_AHEAD_MIN");
    if (env && *env) {
        char *end = NULL; const unsigned long value = strtoul(env, &end, 10);
        if (*env != '-' && end != env && !*end && value >= 9u && value <= 8192u) minimum = (unsigned)value;
    }
    if (T < minimum || E > 65536u || F > 65536u) return 1;
    const uint64_t gb = expert_row_bytes(gu_type, E)*F, db = expert_row_bytes(down_type, F)*E;
    if (!NE || NE > 512u || !E || !F || !gb || !db ||
        go > size || gb*NE > size-go || uo > size || gb*NE > size-uo ||
        doff > size || db*NE > size-doff) return 0;
    const bool all = ds4_env_flag_enabled("DS4_CUDA_QWEN4_PREFILL_AHEAD_ALL", 0);
    const auto history = ring_last_routes.find(go);
    /* Unlike Strata's pack, the local router leaves many experts unused even
     * at 4K. Avoid a speculative upload on the first cold chunk; subsequent
     * chunks reuse its observed working set, with ordinary misses as fallback. */
    if (!all && (history == ring_last_routes.end() || history->second.size() != NE)) return 1;
    const unsigned width = ring_width(NE, gb, db);
    if (!width) return 1;
    const unsigned groups = (NE+width-1u)/width;
    if (!expert_group_ensure(NE, groups, 3) || !expert_reads_finish()) return 0;
    auto &cache = g_stream_selected_cache;
    cache.logical_tier = 0;
    const uint64_t group_bytes = width*(2u*gb+db), table = (2u*group_bytes+255u)&~255ull;
    if (!cuda_stream_selected_ensure_bytes(&cache.prefill_ptr, &cache.prefill_capacity,
            table+(uint64_t)NE*3u*sizeof(char *), "Qwen bounded ahead ring")) return 0;
    ring_ahead.map = map; ring_ahead.size = size; ring_ahead.NE = NE; ring_ahead.T = T;
    ring_ahead.width = width; ring_ahead.groups = std::min(groups, 2u);
    ring_ahead.pointers.resize((uint64_t)NE*3u);
    std::fill(ring_ahead.pointers.begin(), ring_ahead.pointers.end(), nullptr);
    ring_ahead.copied.assign(NE, 0);
    const uint64_t offsets[] = {go, uo, doff}, bytes[] = {gb, gb, db};
    std::copy(offsets, offsets+3, ring_ahead.offsets); std::copy(bytes, bytes+3, ring_ahead.bytes);
    transfer_workload_rows = T;
    ring_ahead.active = true;
    const cudaStream_t copy = expert_copy_stream;
    /* Leave the previous consumer's epoch pinned. The copy stream waits on
     * each physical buffer; only routing later advances the cache epoch. */
    for (unsigned group = 0; group < ring_ahead.groups; ++group) {
        if (!cuda_ok(cudaStreamWaitEvent(copy, expert_group_consumed[group], 0), "Qwen ahead buffer lifetime") ||
            (ram_pageable && !transfer_batch_begin(copy))) goto failed;
        char *base = cache.prefill_ptr+group*group_bytes;
        const unsigned first = group*width, last = std::min(first+width, NE);
        for (unsigned w = 0; w < 3; ++w) for (unsigned e = first; e < last; ++e) {
            if (!all && !history->second[e]) continue;
            const uint64_t off = offsets[w]+e*bytes[w];
            const auto hit = expert_slot_by_offset.find(off);
            const char *ptr = NULL;
            if (hit != expert_slot_by_offset.end()) {
                expert_cache_touch(off, 1); ptr = expert_slots[hit->second].device_ptr;
            } else {
                char *dst = base+w*width*gb+(e-first)*bytes[w];
                if (!expert_transfer(map, off, bytes[w], dst, copy)) goto failed;
                ptr = dst; ring_ahead_bytes += bytes[w]; ring_ahead.copied[e] += bytes[w];
            }
            ring_ahead.pointers[(uint64_t)w*NE+e] = ptr;
        }
        if (ram_pageable && !transfer_batch_end()) goto failed;
        ring_ahead_groups++;
    }
    return 1;
failed:
    (void)transfer_batch_end(); (void)ring_ahead_clear();
    return 0;
}

extern "C" int ds4_gpu_qwen4_moe_prefill_ring_tensor(ds4_gpu_tensor *mid,
        ds4_gpu_tensor *part, const ds4_gpu_tensor *x, const ds4_gpu_tensor *lists,
        const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t go, uint64_t uo, uint64_t doff, uint32_t gu_type, uint32_t down_type,
        uint32_t NE, uint32_t T, uint32_t NS, uint32_t E, uint32_t F, uint32_t cap) {
    using namespace qwen4_cuda;
    if (!ds4_env_flag_enabled("DS4_CUDA_QWEN4_PREFILL_RING", 0) ||
        !tiered_cache(map) || !stable_requested() || T <= 8u) return -1;
    if (E > 65536u || F > 65536u) return -1;
    transfer_workload_rows = T;
    const uint64_t gb = expert_row_bytes(gu_type, E) * F;
    const uint64_t db = expert_row_bytes(down_type, F) * E;
    if (!NE || NE > 512u || !NS || NS > NE || !E || !F || !gb || !db || cap < T ||
        !tensor(mid, (uint64_t)T*NS*F*4) || !tensor(part, (uint64_t)T*NS*E*4) ||
        !tensor(x, (uint64_t)T*E*4) || !tensor(lists, (uint64_t)NE*cap*4) ||
        !tensor(counts, (uint64_t)NE*4) ||
        go > size || gb*NE > size-go || uo > size || gb*NE > size-uo ||
        doff > size || db*NE > size-doff) return 0;
    const cudaStream_t stream = cuda_decode_stream();
    const int *uses = NULL, *ids = NULL;
    /* Also drains consumers of the previous layer before any cache pin is
     * relaxed. An external, prebuilt list still uses the legacy API. */
    if (route_lists != lists->ptr || route_counts != counts->ptr || !route_selected ||
        route_T != T || route_NS != NS || route_NE != NE || route_cap != cap) return -1;
    if (!route_ready && !route_read(route_selected, T, NS, NE, stream)) return 0;
    if (route_ready && !cuda_ok(cudaStreamSynchronize(stream), "Qwen prior ring consumers")) return 0;
    route_ready = true;
    ids = route_host; uses = route_host+(uint64_t)T*NS;
    expert_cache_next_epoch();
    expert_cache_observe(go, gb, NE, T, NS, ids);
    profile_observe(go, T, NS, ids);
    expert_cache_observe(uo, gb, NE, T, NS, ids);
    expert_cache_observe(doff, db, NE, T, NS, ids);
    for (unsigned e = 0; e < NE; ++e) if (uses[e] > 0) {
        expert_cache_touch(go + e*gb, 1);
        expert_cache_touch(uo + e*gb, 1);
        expert_cache_touch(doff + e*db, 1);
    }

    /* Two whole-expert buffers share a byte budget, independent of GGUF
     * projection sizes. Pointer tables and counts are charged separately. */
    const uint64_t blob = 2u*gb + db;
    const unsigned width = ring_width(NE, gb, db);
    if (!width) return -1;
    const unsigned groups = (NE + width-1u)/width;
    if (!expert_group_ensure(NE, groups, 3)) return 0;
    auto &cache = g_stream_selected_cache;
    cache.logical_tier = 0;
    const uint64_t group_bytes = width*blob;
    const uint64_t table_offset = (2u*group_bytes+255u)&~255ull;
    if (!cuda_stream_selected_ensure_bytes(&cache.prefill_ptr, &cache.prefill_capacity,
        table_offset + (uint64_t)NE*3u*sizeof(char *), "Qwen whole-expert prefill ring")) return 0;
    const uint64_t count_bytes = (uint64_t)NE*groups*sizeof(int);
    const uint64_t pointer_offset = (count_bytes+sizeof(char *)-1u)&~(uint64_t)(sizeof(char *)-1u);
    int *host_counts = (int *)expert_group_host;
    const char **ptrs = (const char **)(expert_group_host+pointer_offset);
    memset(host_counts, 0, count_bytes);
    memset(ptrs, 0, (uint64_t)NE*3u*sizeof(char *));
    char *table = cache.prefill_ptr+table_offset;
    const cudaStream_t copy = expert_copy_stream;
    const uint64_t offsets[] = {go, uo, doff}, bytes[] = {gb, gb, db};
    const bool ahead = ring_ahead.active && ring_ahead.map == map && ring_ahead.size == size &&
        ring_ahead.NE == NE && ring_ahead.T == T && ring_ahead.width == width &&
        std::equal(offsets, offsets+3, ring_ahead.offsets) && std::equal(bytes, bytes+3, ring_ahead.bytes);
    if (ring_ahead.active && !ahead && !ring_ahead_clear()) return 0;
    if (ahead) for (unsigned e = 0; e < std::min(NE, ring_ahead.groups*width); ++e)
        for (unsigned w = 0; w < 3; ++w) expert_cache_touch(offsets[w]+e*bytes[w], 1);
    if (expert_reads_start()) {
        expert_read_cursor = 0;
        expert_read_offsets.clear(); expert_read_sizes.clear();
        for (unsigned first = 0; first < NE; first += width)
            for (unsigned w = 0; w < 3; ++w)
                for (unsigned e = first; e < std::min(first+width, NE); ++e) if (uses[e] > 0) {
                    expert_read_offsets.push_back(offsets[w]+e*bytes[w]);
                    expert_read_sizes.push_back(bytes[w]);
                }
        expert_reads_pump();
    }
    int ok = 0;
    for (unsigned group = 0; group < groups; ++group) {
        const unsigned first = group*width, last = std::min(first+width, NE);
        unsigned pairs = 0, active = 0;
        for (unsigned e = first; e < last; ++e) if (uses[e] > 0) {
            host_counts[(uint64_t)group*NE+e] = std::min(uses[e], (int)cap);
            pairs += host_counts[(uint64_t)group*NE+e]; ++active;
        }
        if (!pairs) continue;
        const unsigned buffer = group%2u;
        if (group >= 2u && !cuda_ok(cudaStreamWaitEvent(copy, expert_group_consumed[buffer], 0),
            "Qwen whole expert reuse")) goto ring_done;
        if (ram_pageable && !transfer_batch_begin(copy)) goto ring_done;
        char *base = cache.prefill_ptr+buffer*group_bytes;
        for (unsigned w = 0; w < 3; ++w) {
            const uint64_t projection_offset = w*width*gb;
            for (unsigned e = first; e < last; ++e) if (uses[e] > 0) {
                const uint64_t off = offsets[w]+e*bytes[w];
                const char *cached = NULL;
                if (!expert_cache_get(map, off, bytes[w], copy, &cached)) goto ring_done;
                char *dst = base+projection_offset+(e-first)*bytes[w];
                const char *staged = ahead && group < ring_ahead.groups ? ring_ahead.pointers[(uint64_t)w*NE+e] : NULL;
                if (!cached && !staged && !expert_transfer(map, off, bytes[w], dst, copy)) goto ring_done;
                ptrs[(uint64_t)w*NE+e] = cached ? cached : staged ? staged : dst;
            }
        }
        if (ram_pageable && !transfer_batch_end()) goto ring_done;
        int *group_counts = expert_group_counts+(uint64_t)group*NE;
        if (!cuda_ok(cudaMemcpyAsync(group_counts, host_counts+(uint64_t)group*NE,
            NE*sizeof(int), cudaMemcpyHostToDevice, copy), "Qwen whole expert counts")) goto ring_done;
        for (unsigned w = 0; w < 3; ++w)
            if (!cuda_ok(cudaMemcpyAsync(table+((uint64_t)w*NE+first)*sizeof(char *),
                ptrs+(uint64_t)w*NE+first, (last-first)*sizeof(char *),
                cudaMemcpyHostToDevice, copy), "Qwen whole expert pointers")) goto ring_done;
        if (!cuda_ok(cudaEventRecord(expert_copy_ready, copy), "Qwen whole expert ready") ||
            !cuda_ok(cudaStreamWaitEvent(stream, expert_copy_ready, 0), "Qwen whole expert dependency")) goto ring_done;
        ++expert_copy_groups;
        ++expert_ring_groups;
        if (!matrix_dispatch((float *)mid->ptr, (const float *)x->ptr, base, base+width*gb,
            (const int *)lists->ptr, group_counts, gu_type, NE, T, NS, NS, E, F, cap, false,
            (const char *const *)table, (const char *const *)table+NE, pairs, active) ||
            !matrix_dispatch((float *)part->ptr, (const float *)mid->ptr, base+2u*width*gb, NULL,
            (const int *)lists->ptr, group_counts, down_type, NE, T, NS, NS, F, E, cap, true,
            (const char *const *)table+2u*NE, NULL, pairs, active) ||
            !cuda_ok(cudaEventRecord(expert_group_consumed[buffer], stream),
                "Qwen whole expert consumed")) goto ring_done;
    }
    ok = 1;
    {
        auto &history = ring_last_routes[go]; history.resize(NE);
        for (unsigned e = 0; e < NE; ++e) history[e] = uses[e] > 0;
    }
ring_done:
    if (ahead) {
        for (unsigned e = 0; e < std::min(NE, ring_ahead.groups*width); ++e)
            if (!uses[e]) ring_ahead_unused += ring_ahead.copied[e];
        ring_ahead.active = false;
    }
    const int transfer_ok = transfer_batch_end();
    const int reads_ok = expert_reads_finish();
    route_reset();
    if (!ok || !transfer_ok || !reads_ok) {
        (void)cudaStreamSynchronize(copy);
        (void)cudaStreamSynchronize(stream);
        return 0;
    }
    return 1;
}
