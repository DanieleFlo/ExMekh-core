/* Included after the Flash CUDA dispatchers. The router remains device-owned;
 * a complete gate/up/down triplet has exactly one placement per forward. */
namespace qwen4_cuda {
struct hybrid_state {
    ds4_read_queue pool;
    ds4_read_task tasks[DS4_READ_QUEUE_WORKERS];
    unsigned workers, T, NS, E, F, gu_type, down_type, group_experts, list_cap;
    uint64_t capacity;
    int *ids, *mask[3], *cpu_pairs;
    const char **pointers[3];
    float *x, *output, *raw_gu, *cpu_mid;
    unsigned cpu_phase;
    ds4_gpu_tensor *d_mask[3], *d_ptrs[3], *lists, *counts, *d_cpu_pairs, *d_output, *staging;
    cudaStream_t copy_stream;
    cudaEvent_t copy_ready[2], consumed[2];
    ds4_gpu_tensor *d_raw_gu, *d_cpu_mid;
    cudaStream_t activation_stream;
    std::vector<const char *> cpu_gate, cpu_up, cpu_down;
    std::vector<float> scratch;
    std::vector<unsigned> misses, uses, cpu_groups;
    std::vector<unsigned> pair_offsets, pair_cursor, pair_ids;
    std::vector<unsigned char> cpu_assigned;
    unsigned n_cpu;
    bool cpu_buffers, row_kernels, compact_lists;
};
static hybrid_state hybrid_states[DS4_QWEN4_WORKLOAD_COUNT];
static uintptr_t hybrid_staging_end(uintptr_t address) {
    for (const auto &h : hybrid_states) if (h.staging) {
        const uintptr_t first = (uintptr_t)h.staging->ptr;
        if (address >= first && address - first < h.staging->bytes) return first + h.staging->bytes;
    }
    return 0;
}
static hybrid_state *hybrid_active = &hybrid_states[DS4_QWEN4_WORKLOAD_DECODE];
#define hybrid (*hybrid_active)
static uint64_t hybrid_gpu_hits, hybrid_cpu_pairs, hybrid_pcie_pairs, hybrid_windows;
static uint64_t hybrid_cpu_h2d_bytes, hybrid_activation_d2h_bytes;
static double hybrid_cpu_seconds;

static void hybrid_worker(void *context, unsigned worker, void *payload) {
    hybrid_state &h = *(hybrid_state *)context;
    (void)worker;
    const unsigned stripe = (unsigned)(uintptr_t)payload;
    for (unsigned group = stripe; group + 1u < h.cpu_groups.size(); group += h.workers) {
        for (unsigned j = h.cpu_groups[group]; j < h.cpu_groups[group + 1u]; j += 4u) {
            const unsigned rows = std::min(4u, h.cpu_groups[group + 1u] - j);
            const unsigned first = (unsigned)h.cpu_pairs[j]; const float *inputs[4]; float *outputs[4], *gates[4], *ups[4];
            for (unsigned r = 0; r < rows; r++) {
                const unsigned p = (unsigned)h.cpu_pairs[j + r];
                inputs[r] = h.cpu_phase ? h.cpu_mid + (uint64_t)(j + r) * h.F : h.x + (uint64_t)(p / h.NS) * h.E;
                outputs[r] = h.output + (uint64_t)(j + r) * h.E;
                gates[r] = h.raw_gu + (uint64_t)(j + r) * h.F;
                ups[r] = h.raw_gu + (h.capacity + j + r) * h.F;
            }
            if (!h.cpu_phase)
                ds4_qwen4_cpu_gate_up_rows_run((const uint8_t *)h.cpu_gate[first], (const uint8_t *)h.cpu_up[first],
                    h.gu_type, inputs, rows, h.E, h.F, gates, ups);
            else ds4_qwen4_cpu_down_rows_run((const uint8_t *)h.cpu_down[first], h.down_type,
                inputs, rows, h.E, h.F, outputs);
        }
    }
}

static void hybrid_reset_state(void) {
    ds4_read_queue_stop(&hybrid.pool);
    if (hybrid.capacity) (void)cudaStreamSynchronize(cuda_decode_stream());
    if (hybrid.copy_stream) (void)cudaStreamSynchronize(hybrid.copy_stream);
    if (hybrid.activation_stream) (void)cudaStreamSynchronize(hybrid.activation_stream);
    for (unsigned i = 0; i < 2; i++) {
        if (hybrid.copy_ready[i]) (void)cudaEventDestroy(hybrid.copy_ready[i]);
        if (hybrid.consumed[i]) (void)cudaEventDestroy(hybrid.consumed[i]);
    }
    if (hybrid.copy_stream) (void)cudaStreamDestroy(hybrid.copy_stream);
    if (hybrid.activation_stream) (void)cudaStreamDestroy(hybrid.activation_stream);
    for (unsigned i = 0; i < 3; i++) {
        if (hybrid.mask[i]) (void)cudaFreeHost(hybrid.mask[i]);
        if (hybrid.pointers[i]) (void)cudaFreeHost(hybrid.pointers[i]);
        ds4_gpu_tensor_free(hybrid.d_mask[i]); ds4_gpu_tensor_free(hybrid.d_ptrs[i]);
    }
    if (hybrid.ids) (void)cudaFreeHost(hybrid.ids);
    if (hybrid.cpu_pairs) (void)cudaFreeHost(hybrid.cpu_pairs);
    if (hybrid.x) (void)cudaFreeHost(hybrid.x);
    if (hybrid.output) (void)cudaFreeHost(hybrid.output);
    if (hybrid.raw_gu) (void)cudaFreeHost(hybrid.raw_gu);
    if (hybrid.cpu_mid) (void)cudaFreeHost(hybrid.cpu_mid);
    ds4_gpu_tensor_free(hybrid.d_raw_gu); ds4_gpu_tensor_free(hybrid.d_cpu_mid);
    ds4_gpu_tensor_free(hybrid.lists); ds4_gpu_tensor_free(hybrid.counts);
    ds4_gpu_tensor_free(hybrid.d_cpu_pairs); ds4_gpu_tensor_free(hybrid.d_output);
    ds4_gpu_tensor_free(hybrid.staging);
    hybrid = hybrid_state{};
}

static void hybrid_reset(void) {
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_CACHE_STATS", 0) && hybrid_windows)
        fprintf(stderr, "ds4: Qwen hybrid windows=%llu GPU_hit_pairs=%llu CPU_pairs=%llu PCIe_pairs=%llu CPU_H2D_bytes=%llu activation_D2H_bytes=%llu CPU_wait_ms=%.3f\n",
                (unsigned long long)hybrid_windows, (unsigned long long)hybrid_gpu_hits,
                (unsigned long long)hybrid_cpu_pairs, (unsigned long long)hybrid_pcie_pairs,
                (unsigned long long)hybrid_cpu_h2d_bytes, (unsigned long long)hybrid_activation_d2h_bytes, hybrid_cpu_seconds * 1000);
    for (auto &state : hybrid_states) { hybrid_active = &state; hybrid_reset_state(); }
    hybrid_active = &hybrid_states[DS4_QWEN4_WORKLOAD_DECODE];
    hybrid_gpu_hits = hybrid_cpu_pairs = hybrid_pcie_pairs = hybrid_windows = 0;
    hybrid_cpu_h2d_bytes = hybrid_activation_d2h_bytes = 0;
    hybrid_cpu_seconds = 0;
}

static unsigned hybrid_worker_count(unsigned T) {
    unsigned n = 4; float fraction;
    if (tuning_config(T, &fraction, &n) < 0) return 0;
    const long available = sysconf(_SC_NPROCESSORS_ONLN);
    return std::min(n, available > 1 ? (unsigned)available - 1u : 1u);
}

static int hybrid_finish_prefill(void) {
    auto &h = hybrid_states[DS4_QWEN4_WORKLOAD_PREFILL];
    if (!h.staging) return 1;
    /* This is a phase boundary, after the complete forward. Keep CPU pools
     * and metadata persistent; return the large expert copy workspace before
     * decoding gives the weight arena its final physical budget. */
    if (!cuda_ok(cudaStreamSynchronize(cuda_decode_stream()), "Qwen prefill staging consumed") ||
        !cuda_ok(cudaStreamSynchronize(h.copy_stream), "Qwen prefill copy drain") ||
        !cuda_ok(cudaStreamSynchronize(h.activation_stream), "Qwen prefill activation drain")) return 0;
    ds4_gpu_tensor_free(h.staging); h.staging = NULL;
    return 1;
}

static int hybrid_prepare(unsigned T, unsigned NS, unsigned NE, unsigned E, unsigned F) {
    hybrid_active = &hybrid_states[tuning_phase(T)];
    const uint64_t pairs = (uint64_t)std::max(T, 8u) * NS;
    const bool compact_lists = ds4_env_flag_enabled("DS4_CUDA_QWEN4_COMPACT_LISTS", 0);
    float fraction = 1; unsigned workers = 4;
    const bool cpu_buffers = tuning_config(T, &fraction, &workers) > 0 && fraction < 1;
    if (hybrid.capacity >= pairs && (!cpu_buffers || hybrid.cpu_buffers) && hybrid.E == E && hybrid.F == F && hybrid.NS == NS &&
        hybrid.uses.size() == NE && hybrid.workers == hybrid_worker_count(T) && hybrid.compact_lists == compact_lists) {
        if (!hybrid.staging) hybrid.staging = ds4_gpu_tensor_alloc(2ull * hybrid.group_experts *
            (2u * ds4_qwen4_cpu_row_bytes(16, E) * F + ds4_qwen4_cpu_row_bytes(10, F) * E));
        return hybrid.staging != NULL;
    }
    hybrid_reset_state();
    hybrid.NS = NS; hybrid.E = E; hybrid.F = F; hybrid.workers = hybrid_worker_count(T);
    hybrid.cpu_buffers = cpu_buffers;
    hybrid.compact_lists = compact_lists;
    /* Router top-k is unique within a token. Keep one token's capacity per
     * expert instead of reserving every (token, slot) for every expert.
     * Direct API callers with duplicate slots grow below, before consumption. */
    hybrid.list_cap = compact_lists ? std::max(T, 8u) : (unsigned)pairs;
    /* Large chunks amortize wide staging. Short prompts keep the smaller
     * group rather than withholding another 144 MiB from resident experts. */
    hybrid.group_experts = std::min(T >= 1024u ? 64u : 16u, NE);
    if (!hybrid.workers) { fprintf(stderr, "ds4: invalid Qwen CPU worker count\n"); return 0; }
    const uint64_t pointers = std::max(pairs, (uint64_t)NE) * 3u;
    if (!cuda_ok(cudaMallocHost((void **)&hybrid.ids, pairs * sizeof(int)), "Qwen hybrid routing") ||
        !cuda_ok(cudaMallocHost((void **)&hybrid.cpu_pairs, pairs * sizeof(int)), "Qwen CPU destinations")) goto fail;
    if (cpu_buffers && (
        !cuda_ok(cudaMallocHost((void **)&hybrid.x, (pairs / NS) * E * sizeof(float)), "Qwen CPU activations") ||
        !cuda_ok(cudaMallocHost((void **)&hybrid.output, pairs * E * sizeof(float)), "Qwen CPU outputs") ||
        !cuda_ok(cudaMallocHost((void **)&hybrid.raw_gu, 2u * pairs * F * sizeof(float)), "Qwen CPU raw gate up") ||
        !cuda_ok(cudaMallocHost((void **)&hybrid.cpu_mid, pairs * F * sizeof(float)), "Qwen CPU activated rows"))) goto fail;
    for (unsigned i = 0; i < 3; i++) {
        if (!cuda_ok(cudaMallocHost((void **)&hybrid.mask[i], pairs * sizeof(int)), "Qwen GPU routing masks") ||
            !cuda_ok(cudaMallocHost((void **)&hybrid.pointers[i], pointers * sizeof(char *)), "Qwen GPU pointer staging")) goto fail;
        hybrid.d_mask[i] = ds4_gpu_tensor_alloc(pairs * sizeof(int));
        hybrid.d_ptrs[i] = ds4_gpu_tensor_alloc(pointers * sizeof(char *));
        if (!hybrid.d_mask[i] || !hybrid.d_ptrs[i]) goto fail;
    }
    hybrid.lists = ds4_gpu_tensor_alloc((uint64_t)NE * hybrid.list_cap * sizeof(int));
    hybrid.counts = ds4_gpu_tensor_alloc((uint64_t)NE * sizeof(int));
    hybrid.d_cpu_pairs = ds4_gpu_tensor_alloc(pairs * sizeof(int));
    if (cpu_buffers) {
        hybrid.d_output = ds4_gpu_tensor_alloc(pairs * E * sizeof(float));
        hybrid.d_raw_gu = ds4_gpu_tensor_alloc(2u * pairs * F * sizeof(float));
        hybrid.d_cpu_mid = ds4_gpu_tensor_alloc(pairs * F * sizeof(float));
    }
    hybrid.staging = ds4_gpu_tensor_alloc(2ull * hybrid.group_experts *
        (2u * ds4_qwen4_cpu_row_bytes(16, E) * F + ds4_qwen4_cpu_row_bytes(10, F) * E));
    if (!hybrid.lists || !hybrid.counts || !hybrid.d_cpu_pairs ||
        (cpu_buffers && (!hybrid.d_output || !hybrid.d_raw_gu || !hybrid.d_cpu_mid)) || !hybrid.staging || !transfer_prepare()) goto fail;
    if (!cuda_ok(cudaStreamCreateWithFlags(&hybrid.copy_stream, cudaStreamNonBlocking), "Qwen hybrid copy stream")) goto fail;
    if (!cuda_ok(cudaStreamCreateWithFlags(&hybrid.activation_stream, cudaStreamNonBlocking), "Qwen CPU activation stream")) goto fail;
    for (unsigned i = 0; i < 2; i++)
        if (!cuda_ok(cudaEventCreateWithFlags(&hybrid.copy_ready[i], cudaEventDisableTiming), "Qwen hybrid copies ready") ||
            !cuda_ok(cudaEventCreateWithFlags(&hybrid.consumed[i], cudaEventDisableTiming), "Qwen hybrid group consumed")) goto fail;
    hybrid.cpu_gate.resize(pairs); hybrid.cpu_up.resize(pairs); hybrid.cpu_down.resize(pairs);
    hybrid.scratch.resize((uint64_t)hybrid.workers * F * 4u); hybrid.uses.resize(NE); hybrid.misses.reserve(NE);
    hybrid.pair_offsets.resize(NE + 1u); hybrid.pair_cursor.resize(NE);
    hybrid.pair_ids.resize(pairs); hybrid.cpu_assigned.resize(NE);
    hybrid.cpu_groups.reserve(NE + 1u);
    expert_read_offsets.reserve(NE * 3u); expert_read_sizes.reserve(NE * 3u);
    if (!ds4_read_queue_start(&hybrid.pool, hybrid.workers, hybrid_worker, &hybrid)) goto fail;
    hybrid.capacity = pairs;
    return 1;
fail:
    hybrid_reset_state(); return 0;
}

__global__ void hybrid_scatter(float *part, const float *src, const int *pairs,
                               unsigned n, unsigned NS, unsigned NO, unsigned E) {
    const unsigned j = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n && d < E) {
        const unsigned p = (unsigned)pairs[j];
        part[((uint64_t)(p / NS) * NO + p % NS) * E + d] = src[(uint64_t)j * E + d];
    }
}

/* Keep CUDA's exp/reciprocal rounding at this small FP32 boundary. All three
 * expert weight projections remain on the CPU; no activation quantization. */
__global__ void hybrid_cpu_activate(float *mid, const float *raw, uint64_t count, uint64_t stride) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) mid[i] = silu(raw[i]) * raw[stride + i];
}

static int hybrid_gpu_phase(unsigned phase, ds4_gpu_tensor *mid, ds4_gpu_tensor *part,
        const ds4_gpu_tensor *x, const char *sg, const char *su, const char *sd,
        unsigned st, unsigned NE, unsigned T, unsigned NS, unsigned E, unsigned F, bool tables_ready = false) {
    hybrid_state &h = hybrid;
    /* CPU rows use the qualified warp-FP32 dot order. Mixed prefill must
     * use that same order for GPU hits/misses, otherwise physical residency
     * changes rounding after a checkpoint or between independent sessions.
     * GPU-only prefill retains its original tiled GEMMs. */
    const bool row_kernels = h.row_kernels;
    const unsigned n = T * NS, table = row_kernels ? n : NE, NO = NS + (st != UINT_MAX);
    const cudaStream_t stream = cuda_decode_stream();
    unsigned pairs = 0, active = 0; bool seen[512] = {};
    for (unsigned p = 0; p < n; p++) if (h.mask[phase][p] >= 0) {
        const unsigned e = (unsigned)h.mask[phase][p]; pairs++;
        if (!seen[e]) { seen[e] = true; active++; }
    }
    if (!pairs && (phase || st == UINT_MAX)) return 1;
    if (!tables_ready && (!cuda_ok(cudaMemcpyAsync(h.d_mask[phase]->ptr, h.mask[phase], n * sizeof(int), cudaMemcpyHostToDevice, stream), "Qwen placement mask") ||
        !cuda_ok(cudaMemcpyAsync(h.d_ptrs[phase]->ptr, h.pointers[phase], (uint64_t)table * 3u * sizeof(char *), cudaMemcpyHostToDevice, stream), "Qwen complete expert pointers"))) return 0;
    const char *const *ptr = (const char *const *)h.d_ptrs[phase]->ptr;
    if (row_kernels) {
        return moe_mv_dispatch((float *)mid->ptr, (const float *)x->ptr, (const int *)h.d_mask[phase]->ptr,
                    NULL, NULL, sg, su, h.gu_type, st, NE, T, NS, E, F, false, ptr, ptr + table, phase ? 5u : 1u) &&
               moe_mv_dispatch((float *)part->ptr, (const float *)mid->ptr, (const int *)h.d_mask[phase]->ptr,
                    NULL, NULL, sd, NULL, h.down_type, st, NE, T, NS, F, E, true, ptr + table * 2u, NULL, phase ? 5u : 1u);
    }
    if (!phase && st != UINT_MAX &&
        (!moe_mv_dispatch((float *)mid->ptr, (const float *)x->ptr, (const int *)h.d_mask[phase]->ptr,
             NULL, NULL, sg, su, h.gu_type, st, NE, T, NS, E, F, false, NULL, NULL, 2u) ||
         !moe_mv_dispatch((float *)part->ptr, (const float *)mid->ptr, (const int *)h.d_mask[phase]->ptr,
             NULL, NULL, sd, NULL, h.down_type, st, NE, T, NS, F, E, true, NULL, NULL, 2u))) return 0;
    expert_lists<<<1,256,0,stream>>>((int *)h.lists->ptr, (int *)h.counts->ptr,
                                   (const int *)h.d_mask[phase]->ptr, n, NE, h.list_cap);
    if (!launched()) return 0;
    return matrix_dispatch((float *)mid->ptr, (const float *)x->ptr, NULL, NULL,
              (const int *)h.lists->ptr, (const int *)h.counts->ptr, h.gu_type, NE, T, NS, NO, E, F, h.list_cap,
              false, ptr, ptr + table, pairs, active) &&
           matrix_dispatch((float *)part->ptr, (const float *)mid->ptr, NULL, NULL,
              (const int *)h.lists->ptr, (const int *)h.counts->ptr, h.down_type, NE, T, NS, NO, F, E, h.list_cap,
              true, ptr + table * 2u, NULL, pairs, active);
}

/* Heterogeneous triplets use the existing bounded SSD reader pool. Its
 * requests reserve/publish RAM on the inference thread, never in workers. */
static void hybrid_reads_begin(const hybrid_state &h, uint64_t go, uint64_t uo, uint64_t d,
                               uint64_t gb, uint64_t db) {
    if (!expert_reads_start()) return;
    expert_read_cursor = 0; expert_read_offsets.clear(); expert_read_sizes.clear();
    for (unsigned e : h.misses) {
        expert_read_offsets.push_back(go + e * gb); expert_read_sizes.push_back(gb);
        expert_read_offsets.push_back(uo + e * gb); expert_read_sizes.push_back(gb);
        expert_read_offsets.push_back(d + e * db); expert_read_sizes.push_back(db);
    }
    expert_reads_pump();
}
struct hybrid_reads_scope {
    bool pending;
    explicit hybrid_reads_scope(bool active) : pending(active) {}
    int finish() { if (!pending) return 1; pending = false; return expert_reads_finish(); }
    ~hybrid_reads_scope() { (void)finish(); }
};
} // namespace qwen4_cuda

/* -1: not applicable; 0: error; 1: all routed contributions are ready on the
 * decode stream. Shared projections keep the ordinary CUDA arithmetic. */
extern "C" int ds4_gpu_qwen4_moe_hybrid_tensor(ds4_gpu_tensor *mid, ds4_gpu_tensor *part,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected, const void *map, uint64_t size,
        uint64_t go, uint64_t uo, uint64_t d, unsigned gu_type, unsigned down_type,
        unsigned NE, unsigned T, unsigned NS, unsigned E, unsigned F,
        uint64_t sgo, uint64_t suo, uint64_t sdo, unsigned st) {
    using namespace qwen4_cuda;
    if (!tiered_cache(map) || !ds4_env_flag_enabled("DS4_CUDA_QWEN4_HYBRID_MOE", 0) ||
        gu_type != 16 || down_type != 10 || T > 8192) return -1;
    float fraction; unsigned workers;
    const int configured = tuning_config(T, &fraction, &workers);
    if (configured <= 0) return configured < 0 ? 0 : -1;
    const bool row_kernels = T <= 8u || fraction < 1.0f;
    const unsigned NO = NS + (st != UINT_MAX), n = T * NS, table = row_kernels ? n : NE;
    const uint64_t gb = expert_row_bytes(gu_type, E) * F, db = expert_row_bytes(down_type, F) * E;
    if (!T || !NS || NS > NE || NE > 512 || !gb || !db ||
        !tensor(x, (uint64_t)T * E * 4u) || !tensor(selected, (uint64_t)n * 4u) ||
        !tensor(mid, (uint64_t)T * NO * F * 4u) || !tensor(part, (uint64_t)T * NO * E * 4u) ||
        go > size || gb * NE > size - go || uo > size || gb * NE > size - uo ||
        d > size || db * NE > size - d || !hybrid_prepare(T, NS, NE, E, F)) return 0;
    hybrid_state &h = hybrid; h.T = T; h.gu_type = gu_type; h.down_type = down_type; h.n_cpu = 0;
    transfer_workload_rows = T;
    h.row_kernels = row_kernels;
    const cudaStream_t stream = cuda_decode_stream();
    if (!cuda_ok(cudaMemcpyAsync(h.ids, selected->ptr, n * sizeof(int), cudaMemcpyDeviceToHost, stream), "Qwen hybrid selected IDs") ||
        !cuda_ok(cudaStreamSynchronize(stream), "Qwen hybrid router ready")) return 0;
    profile_observe(go, T, NS, h.ids);
    expert_cache_next_epoch();
    expert_cache_observe(go, gb, NE, T, NS, h.ids); expert_cache_observe(uo, gb, NE, T, NS, h.ids);
    expert_cache_observe(d, db, NE, T, NS, h.ids);
    std::fill(h.uses.begin(), h.uses.end(), 0u); h.misses.clear();
    for (unsigned p = 0; p < n; p++) {
        const int e = h.ids[p];
        if (e < 0 || (unsigned)e >= NE) return 0;
        h.uses[(unsigned)e]++;
    }
    if (!row_kernels && h.compact_lists) {
        const unsigned required = *std::max_element(h.uses.begin(), h.uses.end());
        if (required > h.list_cap) {
            /* A duplicated expert within one token is valid for this low-level
             * API. Never truncate its contributions. Earlier phases may still
             * consume the old lists, so drain before replacing their storage. */
            if (!cuda_ok(cudaStreamSynchronize(stream), "Qwen compact routing lists consumed")) return 0;
            ds4_gpu_tensor *lists = ds4_gpu_tensor_alloc((uint64_t)NE * required * sizeof(int));
            if (!lists) return 0;
            ds4_gpu_tensor_free(h.lists); h.lists = lists; h.list_cap = required;
        }
    }
    h.pair_offsets[0] = 0;
    std::fill(h.cpu_assigned.begin(), h.cpu_assigned.end(), 0);
    for (unsigned e = 0; e < NE; e++) {
        h.pair_cursor[e] = h.pair_offsets[e];
        h.pair_offsets[e + 1u] = h.pair_offsets[e] + h.uses[e];
        if (h.uses[e]) {
            expert_cache_touch(go + (uint64_t)e * gb, 1); expert_cache_touch(uo + (uint64_t)e * gb, 1);
            expert_cache_touch(d + (uint64_t)e * db, 1);
        }
    }
    for (unsigned p = 0; p < n; p++) h.pair_ids[h.pair_cursor[(unsigned)h.ids[p]]++] = p;
    for (unsigned phase = 0; phase < 2; phase++) {
        std::fill(h.mask[phase], h.mask[phase] + n, -1);
        std::fill(h.pointers[phase], h.pointers[phase] + (uint64_t)table * 3u, (const char *)NULL);
    }
    for (unsigned e = 0; e < NE; e++) if (h.uses[e]) {
        const auto g = expert_slot_by_offset.find(go + (uint64_t)e * gb);
        const auto u = expert_slot_by_offset.find(uo + (uint64_t)e * gb);
        const auto w = expert_slot_by_offset.find(d + (uint64_t)e * db);
        if (g == expert_slot_by_offset.end() || u == expert_slot_by_offset.end() || w == expert_slot_by_offset.end()) { h.misses.push_back(e); continue; }
        for (unsigned j = h.pair_offsets[e]; j < h.pair_offsets[e + 1u]; j++) {
            const unsigned p = h.pair_ids[j];
            const unsigned q = row_kernels ? p : e; h.mask[0][p] = (int)e;
            h.pointers[0][q] = expert_slots[g->second].device_ptr;
            h.pointers[0][table + q] = expert_slots[u->second].device_ptr;
            h.pointers[0][2u * table + q] = expert_slots[w->second].device_ptr;
            hybrid_gpu_hits++;
        }
    }
    const char *sg = st != UINT_MAX ? weight(map, size, sgo, row_bytes(st, E) * F) : NULL;
    const char *su = st != UINT_MAX ? weight(map, size, suo, row_bytes(st, E) * F) : NULL;
    const char *sd = st != UINT_MAX ? weight(map, size, sdo, row_bytes(st, F) * E) : NULL;
    if (st != UINT_MAX && (!sg || !su || !sd)) return 0;
    /* Queue the resident work before loading/working the CPU's misses. */
    if (!hybrid_gpu_phase(0, mid, part, x, sg, su, sd, st, NE, T, NS, E, F)) return 0;
    std::sort(h.misses.begin(), h.misses.end(), [&](unsigned a, unsigned b) {
        return h.uses[a] != h.uses[b] ? h.uses[a] > h.uses[b] : a < b;
    });
    if (T > 8u) hybrid_reads_begin(h, go, uo, d, gb, db);
    hybrid_reads_scope reads(T > 8u);
    const unsigned pcie = (unsigned)floorf(fraction * h.misses.size() + 0.5f);
    h.cpu_groups.clear();
    for (unsigned j = pcie; j < h.misses.size(); j++) {
        const unsigned e = h.misses[j]; const char *g = NULL, *u = NULL, *w = NULL;
        if (!expert_ram_source(go + e * gb, gb, &g) || !expert_ram_source(uo + e * gb, gb, &u) ||
            !expert_ram_source(d + e * db, db, &w)) return 0;
        if (!g || !u || !w) continue; /* bounded RAM pressure: this triplet goes to GPU */
        h.cpu_assigned[e] = 1;
        h.cpu_groups.push_back(h.n_cpu);
        for (unsigned i = h.pair_offsets[e]; i < h.pair_offsets[e + 1u]; i++) {
            const unsigned p = h.pair_ids[i];
            h.cpu_pairs[h.n_cpu++] = (int)p; h.cpu_gate[p] = g; h.cpu_up[p] = u; h.cpu_down[p] = w;
        }
    }
    h.cpu_groups.push_back(h.n_cpu);
    if (h.n_cpu) {
        /* The router fence also published x. Download only when the placement
         * actually contains CPU work; this stream does not wait for the
         * already launched resident kernels on the compute stream. */
        if (!cuda_ok(cudaMemcpyAsync(h.x, x->ptr, (uint64_t)T * E * 4u,
                cudaMemcpyDeviceToHost, h.activation_stream), "Qwen hybrid CPU input") ||
            !cuda_ok(cudaStreamSynchronize(h.activation_stream), "Qwen CPU input ready")) return 0;
        hybrid_activation_d2h_bytes += (uint64_t)T * E * 4u;
        h.cpu_phase = 0;
        unsigned submitted = 0;
        for (; submitted < h.workers; submitted++)
            if (!ds4_read_queue_submit(&h.pool, &h.tasks[submitted], (void *)(uintptr_t)submitted)) break;
        if (submitted != h.workers) {
            for (unsigned i = 0; i < submitted; i++) ds4_read_queue_wait(&h.pool, &h.tasks[i]);
            return 0;
        }
    }
    bool ok = true;
    unsigned group = 0, active = 0, phase = 1;
    uint64_t transient = 0, transient_end = 0;
    const uint64_t group_bytes = (uint64_t)h.group_experts * (gb * 2u + db);
    const cudaStream_t copy = h.copy_stream;
    auto begin_group = [&]() {
        const unsigned buffer = group % 2u; phase = 1u + buffer;
        if (!cuda_ok(cudaEventSynchronize(h.copy_ready[buffer]), "Qwen hybrid host table reuse") ||
            !cuda_ok(cudaStreamWaitEvent(copy, h.consumed[buffer], 0), "Qwen hybrid device group reuse") ||
            !transfer_batch_begin(copy)) return false;
        std::fill(h.mask[phase], h.mask[phase] + n, -1);
        std::fill(h.pointers[phase], h.pointers[phase] + (uint64_t)table * 3u, (const char *)NULL);
        transient = buffer * group_bytes; transient_end = transient + group_bytes;
        return true;
    };
    auto finish_group = [&]() {
        if (!active) return true;
        const unsigned buffer = group % 2u;
        if (!transfer_batch_end() ||
            !cuda_ok(cudaMemcpyAsync(h.d_mask[phase]->ptr, h.mask[phase], n * sizeof(int), cudaMemcpyHostToDevice, copy), "Qwen group routing") ||
            !cuda_ok(cudaMemcpyAsync(h.d_ptrs[phase]->ptr, h.pointers[phase], (uint64_t)table * 3u * sizeof(char *), cudaMemcpyHostToDevice, copy), "Qwen group triplets") ||
            !cuda_ok(cudaEventRecord(h.copy_ready[buffer], copy), "Qwen hybrid group ready") ||
            !cuda_ok(cudaStreamWaitEvent(stream, h.copy_ready[buffer], 0), "Qwen hybrid copy dependency") ||
            !hybrid_gpu_phase(phase, mid, part, x, sg, su, sd, st, NE, T, NS, E, F, true) ||
            !cuda_ok(cudaEventRecord(h.consumed[buffer], stream), "Qwen hybrid group consumed")) return false;
        active = 0; group++; expert_copy_groups++;
        return true;
    };
    for (unsigned e : h.misses) {
        if (h.cpu_assigned[e]) continue;
        if (!active && !begin_group()) { ok = false; break; }
        const char *g = NULL, *u = NULL, *w = NULL;
        ok = expert_cache_get(map, go + e * gb, gb, copy, &g) &&
             expert_cache_get(map, uo + e * gb, gb, copy, &u) && expert_cache_get(map, d + e * db, db, copy, &w);
        if (!ok) break;
        /* Capacity pressure does not omit a route or use an unregistered host
         * alias. Keep all missing projections in persistent transient VRAM. */
        const uint64_t offsets[] = {go + e * gb, uo + e * gb, d + e * db};
        const uint64_t sizes[] = {gb, gb, db};
        const char **weights[] = {&g, &u, &w};
        for (unsigned k = 0; k < 3 && ok; k++) if (!*weights[k]) {
            if (transient > transient_end || sizes[k] > transient_end - transient) { ok = false; break; }
            char *dst = (char *)h.staging->ptr + transient;
            ok = expert_transfer(map, offsets[k], sizes[k], dst, copy); *weights[k] = dst; transient += sizes[k];
        }
        if (!ok) break;
        for (unsigned j = h.pair_offsets[e]; j < h.pair_offsets[e + 1u]; j++) {
            const unsigned p = h.pair_ids[j];
            const unsigned q = row_kernels ? p : e; h.mask[phase][p] = (int)e;
            h.pointers[phase][q] = g; h.pointers[phase][table + q] = u; h.pointers[phase][table * 2u + q] = w;
            hybrid_pcie_pairs++;
        }
        active++;
        if (active == h.group_experts && !finish_group()) { ok = false; break; }
    }
    if (ok) ok = finish_group();
    if (!transfer_batch_end()) ok = false;
    if (!reads.finish()) ok = false;
    if (h.n_cpu) {
        const double wait_start = cuda_wall_sec();
        for (unsigned i = 0; i < h.workers; i++) ds4_read_queue_wait(&h.pool, &h.tasks[i]);
        if (ok) {
            const uint64_t count = (uint64_t)h.n_cpu * F, stride = h.capacity * F;
            const cudaStream_t act = h.activation_stream;
            ok = cuda_ok(cudaMemcpyAsync(h.d_raw_gu->ptr, h.raw_gu, count * sizeof(float),
                    cudaMemcpyHostToDevice, act), "Qwen CPU gate results") &&
                 cuda_ok(cudaMemcpyAsync((float *)h.d_raw_gu->ptr + stride, h.raw_gu + stride,
                    count * sizeof(float), cudaMemcpyHostToDevice, act), "Qwen CPU up results");
            if (ok) {
                hybrid_cpu_activate<<<(count + 255u) / 256u, 256, 0, act>>>(
                    (float *)h.d_cpu_mid->ptr, (const float *)h.d_raw_gu->ptr, count, stride);
                ok = launched() && cuda_ok(cudaMemcpyAsync(h.cpu_mid, h.d_cpu_mid->ptr,
                    count * sizeof(float), cudaMemcpyDeviceToHost, act), "Qwen CPU activated results");
            }
            if (!cuda_ok(cudaStreamSynchronize(act), "Qwen CPU activation consumed")) ok = false;
            if (ok) {
                hybrid_cpu_h2d_bytes += 2u * count * sizeof(float);
                hybrid_activation_d2h_bytes += count * sizeof(float);
                h.cpu_phase = 1;
                unsigned submitted = 0;
                for (; submitted < h.workers; ++submitted)
                    if (!ds4_read_queue_submit(&h.pool, &h.tasks[submitted], (void *)(uintptr_t)submitted)) break;
                for (unsigned i = 0; i < submitted; ++i) ds4_read_queue_wait(&h.pool, &h.tasks[i]);
                if (submitted != h.workers) ok = false;
            }
        }
        hybrid_cpu_seconds += cuda_wall_sec() - wait_start;
        if (ok) ok = cuda_ok(cudaMemcpyAsync(h.d_output->ptr, h.output, (uint64_t)h.n_cpu * E * 4u, cudaMemcpyHostToDevice, stream), "Qwen CPU expert results") &&
                     cuda_ok(cudaMemcpyAsync(h.d_cpu_pairs->ptr, h.cpu_pairs, h.n_cpu * sizeof(int), cudaMemcpyHostToDevice, stream), "Qwen CPU expert destinations");
        if (ok) {
            hybrid_cpu_h2d_bytes += (uint64_t)h.n_cpu * E * 4u;
            hybrid_scatter<<<dim3((E + 255u) / 256u, h.n_cpu),256,0,stream>>>((float *)part->ptr,
                (const float *)h.d_output->ptr, (const int *)h.d_cpu_pairs->ptr, h.n_cpu, NS, NO, E);
            ok = launched();
        }
    }
    hybrid_cpu_pairs += h.n_cpu; hybrid_windows++;
    if (!ok) (void)cudaStreamSynchronize(copy);
    return ok ? 1 : 0;
}
#undef hybrid
