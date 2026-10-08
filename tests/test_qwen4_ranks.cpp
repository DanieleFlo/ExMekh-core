#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include "ds4_qwen4_expert_cache.h"

static std::vector<ds4_qwen4_expert_slot> expert_slots, ram_slots;
static std::unordered_map<uint64_t, size_t> expert_slot_by_offset, ram_slot_by_offset;
static uint64_t expert_epoch;
static unsigned transfer_workload_rows = 1;
static double victim_cpu_seconds;
static bool enabled = true;
static bool ds4_env_flag_enabled(const char *, int) { return enabled; }
static double cuda_wall_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#include "ds4_qwen4_ranks.cuh"

static uint32_t random_state = 0x735deca1;
static uint32_t next_random() {
    random_state ^= random_state << 13; random_state ^= random_state >> 17;
    return random_state ^= random_state << 5;
}
static size_t ram_oracle(uint64_t bytes) {
    size_t best = SIZE_MAX;
    for (unsigned pass = 0; pass < 2 && best == SIZE_MAX; ++pass)
        for (size_t i = 0; i < ram_slots.size(); ++i) {
            const auto &s = ram_slots[i];
            if (s.capacity < bytes || s.pinned_epoch == expert_epoch ||
                (!pass && !expert_slot_by_offset.count(s.offset))) continue;
            if (best == SIZE_MAX || ds4_qwen4_expert_rank_before(ram_slots.data(), i, best)) best = i;
        }
    return best;
}
static void verify() {
    for (uint64_t bytes : {1ull, 64ull, 65ull, 128ull, 257ull}) {
        assert(decode_rank_victim(false, bytes) ==
            ds4_qwen4_expert_victim(expert_slots.data(), expert_slots.size(), bytes, expert_epoch));
        assert(decode_rank_victim(true, bytes) == ram_oracle(bytes));
    }
}
static void pin(bool ram, size_t i) {
    auto &slot = (ram ? ram_slots : expert_slots)[i];
    const bool newly_pinned = slot.pinned_epoch != expert_epoch;
    slot.last_used = slot.pinned_epoch = expert_epoch;
    decode_rank_changed(ram, i, newly_pinned);
}
int main() {
    for (size_t i = 0; i < 129; ++i) {
        expert_slots.push_back({i * 4096u, 64u << (i % 3u), i % 7u, i % 5u, 0, nullptr, 0});
        expert_slot_by_offset[i * 4096u] = i;
        ram_slots.push_back({i * 4096u, 64u << (i % 3u), i % 7u, i % 5u, 0, nullptr, 0});
        ram_slot_by_offset[i * 4096u] = i;
    }
    for (unsigned epoch = 1; epoch <= 4096; ++epoch) {
        expert_epoch = epoch;
        transfer_workload_rows = epoch % 17u ? (epoch % 8u + 1u) : 128u;
        decode_rank_next_epoch();
        if (!decode_rank_active) continue;
        /* First query occurs after pinning, including indices initially
         * absent from the tree. Previous-epoch pins must later reappear. */
        for (unsigned j = 0; j < 20; ++j) {
            const bool ram = j & 1u;
            auto &slots = ram ? ram_slots : expert_slots;
            const size_t i = next_random() % slots.size();
            slots[i].frequency = next_random() % 11u;
            decode_rank_changed(ram, i);
            pin(ram, i);
        }
        verify();
        const size_t i = next_random() % expert_slots.size();
        const uint64_t old = expert_slots[i].offset;
        expert_slot_by_offset.erase(old); decode_rank_ram_residency(old);
        expert_slots[i].offset = (4096ull + epoch) * 4096u;
        expert_slot_by_offset[expert_slots[i].offset] = i;
        decode_rank_changed(false, i);
        decode_rank_ram_residency(expert_slots[i].offset);
        /* Change the RAM preference from other to GPU-resident and back. */
        const size_t r = next_random() % ram_slots.size();
        ram_slot_by_offset.erase(ram_slots[r].offset);
        ram_slots[r].offset = expert_slots[i].offset;
        ram_slot_by_offset[ram_slots[r].offset] = r;
        decode_rank_changed(true, r); verify();
        expert_slot_by_offset.erase(expert_slots[i].offset);
        decode_rank_ram_residency(expert_slots[i].offset); verify();
        expert_slot_by_offset[expert_slots[i].offset] = i;
        decode_rank_ram_residency(expert_slots[i].offset); verify();
        if (epoch % 31u == 0 && expert_slots.size() < 1025u) {
            const size_t n = expert_slots.size();
            const uint64_t offset = (10000ull + epoch) * 4096u;
            expert_slots.push_back({offset, 128, 3, 0, 0, nullptr, 0});
            expert_slot_by_offset[offset] = n;
            decode_rank_changed(false, n);
            ram_slots.push_back({offset, 128, 3, 0, 0, nullptr, 0});
            ram_slot_by_offset[offset] = n;
            decode_rank_changed(true, n); verify();
        }
    }
    ++expert_epoch; decode_rank_next_epoch();
    for (size_t i = 0; i < expert_slots.size(); ++i) { pin(false, i); pin(true, i); }
    verify(); // all pinned and oversized requests return no victim
    ++expert_epoch; decode_rank_next_epoch(); verify();
    /* Permanent draft residents remain excluded even after ordinary epoch
     * pins expire, after tree rebuilds, and with every other slot pinned. */
    for (size_t i = 0; i < expert_slots.size(); ++i) {
        expert_slots[i].permanent = i % 3u == 0;
        decode_rank_changed(false, i);
    }
    verify();
    ++expert_epoch; decode_rank_next_epoch(); verify();
    for (size_t i = 0; i < expert_slots.size(); ++i) {
        expert_slots[i].permanent = 1;
        decode_rank_changed(false, i);
    }
    assert(decode_rank_victim(false, 1) == SIZE_MAX);
    ++expert_epoch; decode_rank_next_epoch(); verify();
    enabled = false; ++expert_epoch; decode_rank_next_epoch(); assert(!decode_rank_active);
    enabled = true; ++expert_epoch; decode_rank_next_epoch(); verify();
    assert(decode_rank_bytes() > 0);
    decode_rank_reset();
    puts("PASS: 4096 epochs, exact GPU/RAM oracle, ties, pin restore, permanent residents, residency changes, growth, prefill reset and disabled fallback");
}
