#ifndef DS4_QWEN4_EXPERT_CACHE_H
#define DS4_QWEN4_EXPERT_CACHE_H

#include <stddef.h>
#include <stdint.h>

/* Metadata only: weight bytes and dot products never change. Capacity is the
 * original arena allocation, while the live GGUF range may be smaller. */
typedef struct {
    uint64_t offset, capacity, frequency, last_used, pinned_epoch;
    char *device_ptr;
    /* Engine-owned draft weights may stay resident across routing epochs. */
    unsigned permanent;
} ds4_qwen4_expert_slot;

static inline uint64_t ds4_qwen4_expert_frequency_add(uint64_t old, uint64_t uses) {
    return uses > UINT64_MAX - old ? UINT64_MAX : old + uses;
}

/* Freeze once after graph allocation. Leave at least 6 GiB, and at least a
 * quarter of currently available memory, for PLE/page cache/agent/checkpoints. */
static inline uint64_t ds4_qwen4_ram_auto_budget(uint64_t available, uint64_t limit) {
    uint64_t reserve = 6ull * 1073741824ull;
    if (available / 4u > reserve) reserve = available / 4u;
    const uint64_t usable = available > reserve ? available - reserve : 0;
    return usable < limit ? usable : limit;
}

/* Exact bounded routing history shared by the runtime and policy tests. */
static inline void ds4_qwen4_expert_observe(int *history, unsigned *counts,
        unsigned experts, unsigned selected, unsigned *cursor, unsigned *rows,
        unsigned tokens, const int *ids) {
    const unsigned widths[] = {32, 64, 128};
    for (unsigned t = 0; t < tokens; t++) {
        for (unsigned w = 0; w < 3; w++) {
            unsigned *c = counts + w * experts;
            if (*rows >= widths[w]) {
                const unsigned old = (*cursor + 128u - widths[w]) % 128u;
                for (unsigned k = 0; k < selected; k++) {
                    const int e = history[old * selected + k];
                    if (e >= 0 && (unsigned)e < experts) c[e]--;
                }
            }
            for (unsigned k = 0; k < selected; k++) {
                const int e = ids[(uint64_t)t * selected + k];
                if (e >= 0 && (unsigned)e < experts) c[e]++;
            }
        }
        for (unsigned k = 0; k < selected; k++) {
            history[*cursor * selected + k] = ids[(uint64_t)t * selected + k];
        }
        *cursor = (*cursor + 1u) % 128u;
        if (*rows < 128) (*rows)++;
    }
}

/* Compare the two recent 32-row distributions against their sampling noise.
 * Sparse expert counts have large raw total variation even with stable routing.
 * Pearson discrepancy scales by occupied bins; 3/6 standard-deviation margins
 * avoid shortening the window for ordinary sparse-count noise. This is a
 * cache heuristic, never a router decision. The longest history is 128 rows. */
static inline unsigned ds4_qwen4_expert_window(const unsigned *last32,
        const unsigned *last64, unsigned experts, unsigned rows, unsigned selected) {
    if (rows < 64 || !selected) return 32;
    double discrepancy = 0;
    unsigned active = 0;
    for (unsigned e = 0; e < experts; e++) {
        const double a = last32[e];
        const double b = last64[e] - last32[e];
        const double total = a + b;
        if (total) {
            const double difference = a - b;
            discrepancy += difference * difference / total;
            active++;
        }
    }
    const double df = active > 1 ? active - 1 : 1;
    const double excess = discrepancy - df;
    if (excess <= 0) return 128;
    const double squared = excess * excess;
    if (squared > 72 * df) return 32;
    if (squared > 18 * df) return 64;
    return 128;
}

/* Same ordering as the linear scan, including its lowest-index tie-break. */
static inline int ds4_qwen4_expert_rank_before(const ds4_qwen4_expert_slot *slots,
        size_t a, size_t b) {
    const ds4_qwen4_expert_slot *x = slots + a;
    const ds4_qwen4_expert_slot *y = slots + b;
    if (x->frequency != y->frequency) return x->frequency < y->frequency;
    if (x->last_used != y->last_used) return x->last_used < y->last_used;
    return a < b;
}

/* Every offset in the current request is pinned BEFORE selecting a victim.
 * Keep frequent experts; ties replace the least recently requested. A bounded recent
 * window lets a former hot set lose priority after the workload changes. */
static inline size_t ds4_qwen4_expert_victim(const ds4_qwen4_expert_slot *slots,
        size_t count, uint64_t bytes, uint64_t epoch) {
    size_t victim = SIZE_MAX;
    for (size_t i = 0; i < count; i++) {
        const ds4_qwen4_expert_slot *s = slots + i;
        if (s->capacity < bytes || s->pinned_epoch == epoch || s->permanent) continue;
        if (victim == SIZE_MAX || ds4_qwen4_expert_rank_before(slots, i, victim)) {
            victim = i;
        }
    }
    return victim;
}

#endif
