/* Learned routing is an engine preference, independent of session frontiers.
 * The profile contains counts only, never pointers, weights or routing rules. */
namespace qwen4_cuda {
static std::vector<ds4_gpu_qwen4_expert_layout> profile_layers;
static std::unordered_map<uint64_t, size_t> profile_layer_by_gate;
static std::vector<uint64_t> profile_counts, profile_prior;
static std::unordered_map<uint64_t, uint64_t> profile_initial_frequency;
static uint64_t profile_fingerprint, profile_routes;
static const uint64_t profile_magic = 0x31504851345344ull; /* DS4QHP1 */

static uint64_t profile_hash(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < bytes; i++) hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}

static void profile_observe(uint64_t gate, unsigned T, unsigned NS, const int *ids) {
    const auto found = profile_layer_by_gate.find(gate);
    if (found == profile_layer_by_gate.end() || profile_counts.empty()) return;
    const unsigned NE = profile_layers[found->second].experts;
    const uint64_t base = (uint64_t)found->second * NE;
    for (uint64_t p = 0; p < (uint64_t)T * NS; p++) if (ids[p] >= 0 && (unsigned)ids[p] < NE) {
        uint64_t &value = profile_counts[base + (unsigned)ids[p]];
        value = ds4_qwen4_expert_frequency_add(value, 1);
    }
    profile_routes += T;
}

static uint64_t profile_frequency(uint64_t offset, uint64_t recent, unsigned rows) {
    if (profile_initial_frequency.empty() || rows >= 128) return recent;
    const auto prior = profile_initial_frequency.find(offset);
    return prior == profile_initial_frequency.end() ? recent :
        (uint64_t)((recent * (double)rows + prior->second * (double)(128u - rows)) / 128.0);
}

static void profile_reset(void) {
    const char *path = getenv("DS4_CUDA_QWEN4_EXPERT_PROFILE");
    if (path && *path && profile_routes && !profile_counts.empty()) {
        const std::string temporary = std::string(path) + ".tmp." + std::to_string((long long)getpid());
        FILE *fp = fopen(temporary.c_str(), "wb");
        const uint64_t checksum = profile_hash(14695981039346656037ull, profile_counts.data(), profile_counts.size() * 8u);
        const uint64_t header[] = {profile_magic, profile_fingerprint, profile_counts.size(), checksum};
        bool ok = fp && fwrite(header, sizeof(header), 1, fp) == 1 &&
                  fwrite(profile_counts.data(), 8, profile_counts.size(), fp) == profile_counts.size();
        if (fp && fclose(fp)) ok = false;
        if (ok) ok = rename(temporary.c_str(), path) == 0;
        if (!ok) { (void)remove(temporary.c_str()); fprintf(stderr, "ds4: could not save Qwen expert profile %s\n", path); }
    }
    profile_layers.clear(); profile_layer_by_gate.clear(); profile_counts.clear(); profile_prior.clear();
    profile_initial_frequency.clear(); tuning_reset();
    profile_fingerprint = profile_routes = 0;
}

static int profile_preload(void) {
    if (profile_prior.empty() || !ds4_env_flag_enabled("DS4_CUDA_QWEN4_PROFILE_PRELOAD", 1)) return 1;
    const unsigned NE = profile_layers[0].experts;
    std::vector<size_t> order(profile_prior.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    auto cost = [&](size_t i) {
        const auto &p = profile_layers[i / NE];
        return 2u * expert_row_bytes(p.gu_type, p.embd) * p.ff + expert_row_bytes(p.down_type, p.ff) * p.embd;
    };
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const double x = (double)profile_prior[a] / cost(a), y = (double)profile_prior[b] / cost(b);
        return x != y ? x > y : a < b;
    });
    expert_cache_next_epoch();
    uint64_t complete = 0;
    for (size_t i : order) {
        if (!profile_prior[i]) break;
        const auto &p = profile_layers[i / NE]; const unsigned e = (unsigned)(i % NE);
        const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
        const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
        const uint64_t offsets[] = {p.gate + e * gb, p.up + e * gb, p.down + e * db};
        const uint64_t sizes[] = {gb, gb, db};
        uint64_t required = 0;
        for (unsigned w = 0; w < 3; w++) if (!expert_slot_by_offset.count(offsets[w])) required += sizes[w];
        const uint64_t limit = cuda_model_cache_limit_bytes();
        uint64_t budget = limit > dense_required ? limit - dense_required : 0;
        const char *env = getenv("DS4_CUDA_QWEN4_EXPERT_CACHE_MB");
        if (env && *env) budget = std::min<uint64_t>(budget, strtoull(env, NULL, 10) * 1048576ull);
        if (expert_reserved > budget || required > budget - expert_reserved ||
            g_model_range_bytes > limit || required > limit - g_model_range_bytes) continue;
        bool hit = true;
        for (unsigned w = 0; w < 3; w++) {
            expert_frequency[offsets[w]] = profile_initial_frequency[offsets[w]];
            const char *ptr = NULL;
            if (!expert_cache_get(g_model_host_base, offsets[w], sizes[w], cuda_decode_stream(), &ptr)) return 0;
            hit = hit && ptr;
        }
        complete += hit;
    }
    if (!cuda_ok(cudaStreamSynchronize(cuda_decode_stream()), "Qwen profile preload ready")) return 0;
    fprintf(stderr, "ds4: Qwen learned profile preloaded %llu complete experts\n", (unsigned long long)complete);
    if (!ds4_env_flag_enabled("DS4_CUDA_QWEN4_HYBRID_MOE", 0) ||
        !ds4_env_flag_enabled("DS4_CUDA_QWEN4_HOST_PRELOAD", 1)) return 1;
    uint64_t required = 64ull * 1048576ull;
    for (const auto &p : profile_layers) {
        const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
        const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
        for (unsigned e = 0; e < p.experts; e++) {
            if (expert_slot_by_offset.count(p.gate + e * gb) && expert_slot_by_offset.count(p.up + e * gb) &&
                expert_slot_by_offset.count(p.down + e * db)) continue;
            required += gb * 2u + db + g_model_direct_align * 12u;
        }
    }
    if (!ram_budget_set || required > ram_budget) {
        fprintf(stderr, "ds4: Qwen host complement requires %.2f GiB; bounded demand loading retained\n", required / 1073741824.0);
        return 1;
    }
    expert_cache_next_epoch();
    for (const auto &p : profile_layers) {
        const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
        const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
        for (unsigned e = 0; e < p.experts; e++) {
            if (expert_slot_by_offset.count(p.gate + e * gb) && expert_slot_by_offset.count(p.up + e * gb) &&
                expert_slot_by_offset.count(p.down + e * db)) continue;
            const char *g, *u, *d;
            if (!expert_ram_source(p.gate + e * gb, gb, &g) || !expert_ram_source(p.up + e * gb, gb, &u) ||
                !expert_ram_source(p.down + e * db, db, &d)) return 0;
            if (!g || !u || !d) { fprintf(stderr, "ds4: Qwen host preload reached its physical budget\n"); return 1; }
        }
    }
    fprintf(stderr, "ds4: Qwen complete CPU complement resident in RAM\n");
    return 1;
}
} // namespace qwen4_cuda

/* Runs before any consumers, once per engine. Capacity failures leave the
 * already admitted triplets ordinary and bounded; I/O errors remain errors. */
static int qwen4_mtp_resident_preload(const ds4_gpu_qwen4_expert_layout *layers, unsigned count) {
    using namespace qwen4_cuda;
    if (!mtp_resident_layers || mtp_resident_loaded) return 1;
    const double started = cuda_wall_sec();
    const cudaStream_t stream = cuda_decode_stream();
    expert_cache_next_epoch();
    bool complete = true;
    for (unsigned l = count - mtp_resident_layers; l < count; l++) {
        const auto &p = layers[l];
        const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
        const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
        for (unsigned e = 0; e < p.experts; e++) {
            const char *g = NULL, *u = NULL, *d = NULL;
            if (!expert_cache_get(g_model_host_base, p.gate + (uint64_t)e * gb, gb, stream, &g) ||
                !expert_cache_get(g_model_host_base, p.up + (uint64_t)e * gb, gb, stream, &u) ||
                !expert_cache_get(g_model_host_base, p.down + (uint64_t)e * db, db, stream, &d)) return 0;
            complete = complete && g && u && d;
        }
    }
    if (!cuda_ok(cudaStreamSynchronize(stream), "Qwen resident NextN ready")) return 0;
    if (!complete) {
        fprintf(stderr, "ds4: Qwen NextN resident preload reached physical quota; bounded routing retained\n");
        mtp_resident_layers = 0;
        return 1;
    }
    uint64_t bytes = 0;
    for (unsigned l = count - mtp_resident_layers; l < count; l++) {
        const auto &p = layers[l];
        const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
        const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
        for (unsigned e = 0; e < p.experts; e++) {
            const uint64_t offsets[] = {p.gate + (uint64_t)e * gb, p.up + (uint64_t)e * gb, p.down + (uint64_t)e * db};
            for (uint64_t off : offsets) {
                auto &slot = expert_slots[expert_slot_by_offset.at(off)];
                slot.permanent = 1;
                bytes += slot.capacity;
            }
        }
    }
    /* All replacement paths must see the new ownership immediately. */
    decode_rank_reset(); gpu_victim_heaps.clear();
    mtp_resident_loaded = true;
    fprintf(stderr, "ds4: Qwen NextN %u layer(s) complete experts resident, %.2f GiB inside weight quota, %.1f ms startup\n",
        mtp_resident_layers, bytes / 1073741824.0, (cuda_wall_sec() - started) * 1000.0);
    return 1;
}

extern "C" int ds4_gpu_qwen4_runtime_prepare(const void *map, uint64_t size,
        const ds4_gpu_qwen4_expert_layout *layers, uint32_t count, uint32_t prefill, uint32_t nextn_layers) {
    using namespace qwen4_cuda;
    if (!tiered_cache(map)) return 1;
    const bool hybrid_on = ds4_env_flag_enabled("DS4_CUDA_QWEN4_HYBRID_MOE", 0);
    const bool kv_on = ds4_env_flag_enabled("DS4_CUDA_QWEN4_KV_STREAMING", 0);
    const bool resident_mtp = nextn_layers && ds4_env_flag_enabled("DS4_CUDA_QWEN4_MTP_RESIDENT", 0);
    const char *path = getenv("DS4_CUDA_QWEN4_EXPERT_PROFILE");
    if (!hybrid_on && !kv_on && !stable_requested() && !resident_mtp && (!path || !*path)) return 1;
    if (!layers || !count || count > 256 || nextn_layers > count || !layers[0].experts || layers[0].experts > 512) return 0;
    for (uint32_t l = 0; l < count; l++) {
        if (layers[l].experts != layers[0].experts || !layers[l].slots || layers[l].slots > layers[l].experts) return 0;
        const uint64_t g = expert_row_bytes(layers[l].gu_type, layers[l].embd) * layers[l].ff * layers[l].experts;
        const uint64_t d = expert_row_bytes(layers[l].down_type, layers[l].ff) * layers[l].embd * layers[l].experts;
        if (!g || !d || layers[l].gate > size || g > size - layers[l].gate ||
            layers[l].up > size || g > size - layers[l].up || layers[l].down > size || d > size - layers[l].down) return 0;
    }
    uint64_t fingerprint = profile_hash(14695981039346656037ull, &size, sizeof(size));
    fingerprint = profile_hash(fingerprint, map, (size_t)std::min<uint64_t>(size, 1048576ull));
    fingerprint = profile_hash(fingerprint, layers, count * sizeof(*layers));
    cudaDeviceProp prop; int driver = 0;
    if (cudaGetDeviceProperties(&prop, g_gpu[0].device_id) != cudaSuccess || cudaDriverGetVersion(&driver) != cudaSuccess) return 0;
    uint64_t tuning_key = profile_hash(fingerprint, prop.uuid.bytes, sizeof(prop.uuid.bytes));
    const bool compact_lists = ds4_env_flag_enabled("DS4_CUDA_QWEN4_COMPACT_LISTS", 0);
    const bool ahead = ds4_env_flag_enabled("DS4_CUDA_QWEN4_PREFILL_AHEAD", 0);
    /* Preserve qualified preferences when new experiments are disabled.
     * Their unchanged scheduler must not lose its v57 calibration. */
    const uint32_t scheduler_version = prefill > 256u ? 35u :
        resident_mtp ? 33u : ahead ? 31u : compact_lists ? 29u : 28u;
    tuning_key = profile_hash(tuning_key, &scheduler_version, sizeof(scheduler_version));
    tuning_key = profile_hash(tuning_key, &driver, sizeof(driver));
    if (resident_mtp) tuning_key = profile_hash(tuning_key, &nextn_layers, sizeof(nextn_layers));
    tuning_key = profile_hash(tuning_key, &cache_ctx, sizeof(cache_ctx));
    tuning_key = profile_hash(tuning_key, &prefill, sizeof(prefill));
    const uint64_t weight_budget = cuda_model_cache_limit_bytes();
    tuning_key = profile_hash(tuning_key, &weight_budget, sizeof(weight_budget));
    const char *budget_names[] = {"DS4_CUDA_QWEN4_RAM_CACHE_MB", "DS4_CUDA_QWEN4_EXPERT_CACHE_MB", "DS4_CUDA_QWEN4_EXPERT_POLICY", "DS4_CUDA_QWEN4_PREFILL_RING", "DS4_CUDA_QWEN4_PREFILL_RING_MB", "DS4_CUDA_QWEN4_PLE_OVERLAP", "DS4_CUDA_QWEN4_PLE_PREFETCH", "DS4_CUDA_QWEN4_KV_PREFILL_STAGE", "DS4_CUDA_QWEN4_KV_PREFETCH", "DS4_CUDA_QWEN4_KV_PREFETCH_BUFFERS", "DS4_CUDA_QWEN4_STAGING_WORKERS", "DS4_CUDA_QWEN4_DECODE_GRAPHS", "DS4_CUDA_DECODE_GRAPHS", "DS4_CUDA_QWEN4_KV_DECODE_BATCH", "DS4_CUDA_QWEN4_QSA_REGISTER_KEYS", "DS4_CUDA_QWEN4_GRAPH_DEFAULT_STREAM", "DS4_CUDA_QWEN4_DECODE_RANKS"};
    for (const char *name : budget_names) {
        const char *value = getenv(name); const size_t length = value ? strlen(value) : 0;
        tuning_key = profile_hash(tuning_key, &length, sizeof(length));
        if (length) tuning_key = profile_hash(tuning_key, value, length);
    }
    tuning_key = profile_hash(tuning_key, &kv_on, sizeof(kv_on));
    if (kv_on) {
        const char *value = getenv("DS4_CUDA_QWEN4_KV_RESIDENT");
        const uint64_t resident = value && *value ? strtoull(value, NULL, 10) : 32768u;
        tuning_key = profile_hash(tuning_key, &resident, sizeof(resident));
    }
    if (compact_lists || ahead) tuning_key = profile_hash(tuning_key, &compact_lists, sizeof(compact_lists));
    const char *ahead_names[] = {"DS4_CUDA_QWEN4_PREFILL_AHEAD", "DS4_CUDA_QWEN4_PREFILL_AHEAD_MIN", "DS4_CUDA_QWEN4_PREFILL_AHEAD_ALL"};
    if (ahead) for (const char *name : ahead_names) {
        const char *value = getenv(name); const size_t length = value ? strlen(value) : 0;
        tuning_key = profile_hash(tuning_key, &length, sizeof(length));
        if (length) tuning_key = profile_hash(tuning_key, value, length);
    }
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    tuning_key = profile_hash(tuning_key, &cpus, sizeof(cpus));
    tuning_load(tuning_key);
    float fraction; unsigned workers;
    if (hybrid_on) for (unsigned phase = 0; phase < 3; phase++) {
        const unsigned T = phase == 2u ? std::min(prefill, 8192u) : phase == 1u ? 8u : 1u;
        const int config = tuning_config(T, &fraction, &workers);
        if (config < 0) { fprintf(stderr, "ds4: invalid Qwen hybrid option\n"); return 0; }
        /* Validate every family here, but reserve its persistent workspace
         * only on first use. An unused decode family must not evict weights
         * while an MTP request explicitly keeps its previous GPU path. */
    }
    if ((hybrid_on || kv_on) && !transfer_prepare()) return 0;
    g_model_auto_cache_limit = 0;
    stable_prepare(layers, count, nextn_layers);
    if (!qwen4_mtp_resident_preload(layers, count)) return 0;
    if (!path || !*path) return 1;
    if (!profile_layers.empty()) return 1;
    profile_fingerprint = fingerprint;
    profile_layers.assign(layers, layers + count);
    for (uint32_t l = 0; l < count; l++) profile_layer_by_gate[layers[l].gate] = l;
    profile_counts.assign((uint64_t)count * layers[0].experts, 0);
    FILE *fp = fopen(path, "rb");
    if (fp) {
        uint64_t header[4];
        profile_prior.resize(profile_counts.size());
        const bool ok = fread(header, sizeof(header), 1, fp) == 1 && header[0] == profile_magic &&
            header[1] == profile_fingerprint && header[2] == profile_counts.size() &&
            fread(profile_prior.data(), 8, profile_prior.size(), fp) == profile_prior.size() && fgetc(fp) == EOF &&
            header[3] == profile_hash(14695981039346656037ull, profile_prior.data(), profile_prior.size() * 8u);
        fclose(fp);
        if (ok) {
            profile_counts = profile_prior;
            for (uint32_t l = 0; l < count; l++) {
                const auto &p = layers[l]; uint64_t total = 0;
                for (unsigned e = 0; e < p.experts; e++) total = ds4_qwen4_expert_frequency_add(total, profile_prior[(uint64_t)l * p.experts + e]);
                const double denominator = std::max(1.0, (double)total / p.slots);
                const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
                const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
                for (unsigned e = 0; e < p.experts; e++) {
                    const uint64_t frequency = (uint64_t)(profile_prior[(uint64_t)l * p.experts + e] * 1048576.0 / denominator);
                    profile_initial_frequency[p.gate + (uint64_t)e * gb] = frequency;
                    profile_initial_frequency[p.up + (uint64_t)e * gb] = frequency;
                    profile_initial_frequency[p.down + (uint64_t)e * db] = frequency;
                }
            }
            fprintf(stderr, "ds4: loaded Qwen expert profile %s (%zu entries)\n", path, profile_prior.size());
        } else {
            profile_prior.clear(); fprintf(stderr, "ds4: incompatible/corrupt Qwen expert profile ignored: %s\n", path);
        }
    }
    if (!profile_prior.empty()) for (const auto &p : stable_layers)
        stable_observe(p.layout.gate, p.layout.experts, 0, 0, NULL);
    return profile_preload();
}

extern "C" int ds4_gpu_qwen4_prefill_finish(void) {
    using namespace qwen4_cuda;
    if (!ring_ahead_clear()) return 0;
    if (!hybrid_finish_prefill()) return 0;
    if (stable_layers.empty()) return 1;
    const cudaStream_t stream = cuda_decode_stream();
    if (!cuda_ok(cudaStreamSynchronize(stream), "Qwen prefill experts consumed")) return 0;
    stable_prefill = false;
    stable_offsets.clear();
    expert_cache_next_epoch();
    /* Recent final-prompt routing prepares decode. Pin desired hits across
     * all layers before replacing anything; charge all copies to prefill. */
    for (auto &layer : stable_layers) {
        const auto &p = layer.layout;
        if (!expert_windows.count(p.gate)) continue;
        const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
        const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
        const auto &window = expert_windows.at(p.gate);
        for (unsigned e = 0; e < p.experts; e++) layer.counts[e] = window.counts[2u * p.experts + e];
        std::sort(layer.order.begin(), layer.order.end(), [&](unsigned a, unsigned b) {
            const uint64_t x = layer.counts[a], y = layer.counts[b];
            return x != y ? x > y : a < b;
        });
        for (unsigned i = 0; i < layer.quota; i++) {
            const unsigned e = layer.order[i];
            if (!layer.counts[e]) break;
            stable_offsets[p.gate + (uint64_t)e * gb] = true;
            stable_offsets[p.up + (uint64_t)e * gb] = true;
            stable_offsets[p.down + (uint64_t)e * db] = true;
            expert_cache_touch(p.gate + (uint64_t)e * gb, 1);
            expert_cache_touch(p.up + (uint64_t)e * gb, 1);
            expert_cache_touch(p.down + (uint64_t)e * db, 1);
        }
    }
    if (!transfer_batch_begin(stream)) return 0;
    bool ok = true;
    for (const auto &layer : stable_layers) {
        const auto &p = layer.layout;
        if (!expert_windows.count(p.gate)) continue;
        const uint64_t gb = expert_row_bytes(p.gu_type, p.embd) * p.ff;
        const uint64_t db = expert_row_bytes(p.down_type, p.ff) * p.embd;
        for (unsigned i = 0; i < layer.quota && ok; i++) {
            const unsigned e = layer.order[i]; const char *g = NULL, *u = NULL, *d = NULL;
            if (!layer.counts[e]) break;
            ok = expert_cache_get(g_model_host_base, p.gate + (uint64_t)e * gb, gb, stream, &g) &&
                 expert_cache_get(g_model_host_base, p.up + (uint64_t)e * gb, gb, stream, &u) &&
                 expert_cache_get(g_model_host_base, p.down + (uint64_t)e * db, db, stream, &d);
        }
        if (!ok) break;
    }
    const int drained = transfer_batch_end();
    return ok && drained && cuda_ok(cudaStreamSynchronize(stream), "Qwen decode cache ready");
}

extern "C" ds4_qwen4_workload ds4_gpu_qwen4_workload_scope(ds4_qwen4_workload workload) {
    return g_n_gpus == 1 ? qwen4_cuda::tuning_set_workload(workload) :
        qwen4_cuda::tuning_workload_scope;
}
