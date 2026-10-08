/* Stable, complete expert placement. Inspired by Strata's per-layer cache
 * quotas and host complement; no kernels or quantization are changed. */
struct stable_expert_layer {
    ds4_gpu_qwen4_expert_layout layout;
    unsigned quota;
    bool selected = false;
    std::vector<uint64_t> counts;
    std::vector<unsigned> order;
};
static std::vector<stable_expert_layer> stable_layers;
static std::unordered_map<uint64_t, size_t> stable_layer_by_gate;
static std::unordered_map<uint64_t, bool> stable_offsets;
static bool stable_prefill = true;
static unsigned mtp_resident_layers;
static bool mtp_resident_loaded;

static bool stable_requested(void) {
    const char *policy = getenv("DS4_CUDA_QWEN4_EXPERT_POLICY");
    return policy && !strcmp(policy, "stable");
}

static void stable_reset(void) {
    stable_layers.clear(); stable_layer_by_gate.clear(); stable_offsets.clear(); stable_prefill = true;
    mtp_resident_layers = 0; mtp_resident_loaded = false;
}

static void stable_observe(uint64_t gate, unsigned NE, unsigned T, unsigned NS, const int *ids) {
    if (T) stable_prefill = tuning_phase(T) == DS4_QWEN4_WORKLOAD_PREFILL;
    const auto found = stable_layer_by_gate.find(gate);
    if (found == stable_layer_by_gate.end()) return;
    auto &layer = stable_layers[found->second];
    if (layer.selected || NE != layer.layout.experts) return;
    for (uint64_t i = 0; i < (uint64_t)T * NS; i++)
        if (ids[i] >= 0 && (unsigned)ids[i] < NE) layer.counts[ids[i]]++;
    /* A loaded profile seeds the ranking; otherwise use the entire first
     * routed chunk, not just the last 128 rows of the recent-policy window. */
    const auto &p = layer.layout;
    const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
    const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
    std::sort(layer.order.begin(), layer.order.end(), [&](unsigned a, unsigned b) {
        const uint64_t pa = profile_frequency(p.gate + (uint64_t)a * gb, 0, 0);
        const uint64_t pb = profile_frequency(p.gate + (uint64_t)b * gb, 0, 0);
        if (pa != pb) return pa > pb;
        return layer.counts[a] != layer.counts[b] ? layer.counts[a] > layer.counts[b] : a < b;
    });
    for (unsigned i = 0; i < layer.quota; i++) {
        const unsigned e = layer.order[i];
        stable_offsets[p.gate + (uint64_t)e * gb] = true;
        stable_offsets[p.up + (uint64_t)e * gb] = true;
        stable_offsets[p.down + (uint64_t)e * db] = true;
    }
    layer.selected = true;
}

static uint64_t expert_placement_budget(void) {
    const uint64_t limit = cuda_model_cache_limit_bytes();
    uint64_t budget = limit > dense_required ? limit - dense_required : 0;
    /* Match the allocator's physical ceiling and outstanding dense reserve.
     * The extra 256 MiB accounts for lazy prefill scratch, not weight bytes. */
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        uint64_t slack = 0;
        for (const auto &arena : g_model_arenas) slack += arena.bytes - arena.used;
        const uint64_t reserve = 2ull * 1073741824ull + 256ull * 1048576ull +
            (dense_required > dense_reserved ? dense_required - dense_reserved : 0);
        const uint64_t room = (uint64_t)free_bytes + slack;
        budget = std::min(budget, room > reserve ? room - reserve : 0);
    }
    const char *env = getenv("DS4_CUDA_QWEN4_EXPERT_CACHE_MB");
    if (env && *env) {
        char *end = NULL; const uint64_t mb = strtoull(env, &end, 10);
        if (end != env && !*end && *env != '-' && mb <= UINT64_MAX / 1048576ull)
            budget = std::min<uint64_t>(budget, mb * 1048576ull);
    }
    return budget;
}

/* Reserve the draft's complete experts inside the existing quota. No extra
 * model copy or VRAM budget: the remaining layers share the remainder. */
static void stable_prepare(const ds4_gpu_qwen4_expert_layout *layers, unsigned count, unsigned nextn) {
    if (!stable_layers.empty() || mtp_resident_loaded) return;
    const uint64_t budget = expert_placement_budget();
    uint64_t draft_bytes = 0;
    if (nextn && ds4_env_flag_enabled("DS4_CUDA_QWEN4_MTP_RESIDENT", 0)) {
        for (unsigned l = count - nextn; l < count; l++) {
            const auto &p = layers[l];
            draft_bytes += (2u * expert_row_bytes(p.gu_type, p.embd) * p.ff +
                expert_row_bytes(p.down_type, p.ff) * p.embd) * p.experts;
        }
        if (draft_bytes <= budget) mtp_resident_layers = nextn;
        else {
            fprintf(stderr, "ds4: Qwen NextN resident experts need %.2f GiB, quota %.2f GiB; bounded routing retained\n",
                draft_bytes / 1073741824.0, budget / 1073741824.0);
            draft_bytes = 0;
        }
    }
    if (!stable_requested()) return;
    uint64_t model_bytes = 0;
    for (unsigned l = 0; l < count; l++) {
        const auto &p = layers[l];
        model_bytes += (2u * expert_row_bytes(p.gu_type, p.embd) * p.ff +
            expert_row_bytes(p.down_type, p.ff) * p.embd) * p.experts;
    }
    const uint64_t other_bytes = model_bytes - draft_bytes;
    const double fraction = other_bytes ? std::min(1.0, (double)(budget - draft_bytes) / other_bytes) : 0;
    uint64_t reserved = 0, complete = 0;
    stable_layers.reserve(count); stable_layer_by_gate.reserve(count);
    for (unsigned l = 0; l < count; l++) {
        const auto &p = layers[l];
        const unsigned quota = l >= count - mtp_resident_layers ? p.experts : (unsigned)(fraction * p.experts);
        stable_expert_layer layer;
        layer.layout = p; layer.quota = quota;
        layer.counts.assign(p.experts, 0); layer.order.resize(p.experts);
        for (unsigned e = 0; e < p.experts; e++) layer.order[e] = e;
        stable_layer_by_gate[p.gate] = l;
        stable_layers.push_back(std::move(layer));
        if (l >= count - mtp_resident_layers) {
            const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
            const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
            for (unsigned e = 0; e < p.experts; e++) {
                stable_offsets[p.gate + (uint64_t)e * gb] = true;
                stable_offsets[p.up + (uint64_t)e * gb] = true;
                stable_offsets[p.down + (uint64_t)e * db] = true;
            }
            stable_layers.back().selected = true;
        }
        reserved += quota * (2u * expert_row_bytes(p.gu_type, p.embd) * p.ff +
            expert_row_bytes(p.down_type, p.ff) * p.embd);
        complete += quota;
    }
    stable_offsets.reserve((size_t)complete * 3u);
    fprintf(stderr, "ds4: Qwen stable placement %llu complete experts across %u layers; %.2f GiB quota, no GPU eviction\n",
        (unsigned long long)complete, count, reserved / 1073741824.0);
}
