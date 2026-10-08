/* Persistent exact replacement indices for small hybrid forwards. Included
 * inside qwen4_cuda; metadata only, never changes the router or arithmetic.
 * Prefill retains the bulk heaps. Decode updates only changed leaves instead
 * of rebuilding every RAM/VRAM candidate list at each layer. */
struct decode_rank_node { size_t gpu, other; };
struct decode_rank_tree {
    uint64_t bytes = 0;
    size_t leaves = 0;
    std::vector<decode_rank_node> nodes;
};
static std::unordered_map<uint64_t, decode_rank_tree> decode_gpu_ranks, decode_ram_ranks;
static std::vector<size_t> decode_gpu_pins, decode_ram_pins;
static bool decode_rank_active;

static size_t decode_rank_min(const std::vector<ds4_qwen4_expert_slot> &slots, size_t a, size_t b) {
    if (a == SIZE_MAX) return b;
    if (b == SIZE_MAX) return a;
    return ds4_qwen4_expert_rank_before(slots.data(), a, b) ? a : b;
}
static decode_rank_node decode_rank_merge(const std::vector<ds4_qwen4_expert_slot> &slots,
                                          decode_rank_node a, decode_rank_node b) {
    return {decode_rank_min(slots, a.gpu, b.gpu), decode_rank_min(slots, a.other, b.other)};
}
static decode_rank_node decode_rank_leaf(bool ram, uint64_t bytes, size_t i) {
    const auto &slots = ram ? ram_slots : expert_slots;
    const auto &slot = slots[i];
    if (slot.capacity < bytes || slot.pinned_epoch == expert_epoch || slot.permanent) return {SIZE_MAX, SIZE_MAX};
    if (!ram || expert_slot_by_offset.count(slot.offset)) return {i, SIZE_MAX};
    return {SIZE_MAX, i};
}
static void decode_rank_build(decode_rank_tree &tree, bool ram) {
    const auto &slots = ram ? ram_slots : expert_slots;
    tree.leaves = 128;
    while (tree.leaves < slots.size()) tree.leaves *= 2u;
    tree.nodes.assign(tree.leaves * 2u, {SIZE_MAX, SIZE_MAX});
    for (size_t i = 0; i < slots.size(); ++i)
        tree.nodes[tree.leaves + i] = decode_rank_leaf(ram, tree.bytes, i);
    for (size_t i = tree.leaves - 1u; i; --i)
        tree.nodes[i] = decode_rank_merge(slots, tree.nodes[i * 2u], tree.nodes[i * 2u + 1u]);
}
static void decode_rank_update(decode_rank_tree &tree, bool ram, size_t i) {
    if (i >= tree.leaves) { decode_rank_build(tree, ram); return; }
    const auto &slots = ram ? ram_slots : expert_slots;
    size_t node = tree.leaves + i;
    tree.nodes[node] = decode_rank_leaf(ram, tree.bytes, i);
    while ((node /= 2u))
        tree.nodes[node] = decode_rank_merge(slots, tree.nodes[node * 2u], tree.nodes[node * 2u + 1u]);
}
static void decode_rank_reset(void) {
    decode_gpu_ranks.clear(); decode_ram_ranks.clear();
    decode_gpu_pins.clear(); decode_ram_pins.clear(); decode_rank_active = false;
}
static uint64_t decode_rank_bytes(void) {
    uint64_t bytes = (decode_gpu_pins.capacity() + decode_ram_pins.capacity()) * sizeof(size_t);
    for (const auto &entry : decode_gpu_ranks) bytes += entry.second.nodes.capacity() * sizeof(decode_rank_node);
    for (const auto &entry : decode_ram_ranks) bytes += entry.second.nodes.capacity() * sizeof(decode_rank_node);
    return bytes;
}
static void decode_rank_next_epoch(void) {
    const bool enabled = transfer_workload_rows && transfer_workload_rows <= 8u &&
        ds4_env_flag_enabled("DS4_CUDA_QWEN4_HYBRID_MOE", 0) &&
        ds4_env_flag_enabled("DS4_CUDA_QWEN4_DECODE_RANKS", 1);
    if (!enabled) { if (decode_rank_active) decode_rank_reset(); return; }
    const double start = cuda_wall_sec();
    if (!decode_rank_active) {
        /* At most 512 complete experts plus demotions in a layer. Reserve
         * once; ordinary forwards never allocate pin-list storage. */
        decode_gpu_pins.reserve(8192); decode_ram_pins.reserve(8192);
        decode_rank_active = true;
    }
    for (size_t i : decode_gpu_pins)
        for (auto &entry : decode_gpu_ranks) decode_rank_update(entry.second, false, i);
    for (size_t i : decode_ram_pins)
        for (auto &entry : decode_ram_ranks) decode_rank_update(entry.second, true, i);
    decode_gpu_pins.clear(); decode_ram_pins.clear();
    victim_cpu_seconds += cuda_wall_sec() - start;
}
static void decode_rank_changed(bool ram, size_t i, bool newly_pinned = false) {
    if (!decode_rank_active) return;
    const double start = cuda_wall_sec();
    auto &pins = ram ? decode_ram_pins : decode_gpu_pins;
    if (newly_pinned) pins.push_back(i);
    auto &trees = ram ? decode_ram_ranks : decode_gpu_ranks;
    for (auto &entry : trees) decode_rank_update(entry.second, ram, i);
    victim_cpu_seconds += cuda_wall_sec() - start;
}
static void decode_rank_ram_residency(uint64_t offset) {
    if (!decode_rank_active) return;
    const auto found = ram_slot_by_offset.find(offset);
    if (found != ram_slot_by_offset.end()) decode_rank_changed(true, found->second);
}
static size_t decode_rank_victim(bool ram, uint64_t bytes) {
    auto &trees = ram ? decode_ram_ranks : decode_gpu_ranks;
    auto &tree = trees[bytes];
    if (!tree.leaves) { tree.bytes = bytes; decode_rank_build(tree, ram); }
    return tree.nodes[1].gpu != SIZE_MAX ? tree.nodes[1].gpu : tree.nodes[1].other;
}
