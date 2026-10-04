#include "ds4_qwen4_expert_cache.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    const uint64_t GiB = 1073741824ull;
    assert(ds4_qwen4_ram_auto_budget(48*GiB, 16*GiB) == 16*GiB);
    assert(ds4_qwen4_ram_auto_budget(10*GiB, 16*GiB) == 4*GiB);
    assert(ds4_qwen4_ram_auto_budget(6*GiB, 16*GiB) == 0);
    assert(ds4_qwen4_ram_auto_budget(0, 16*GiB) == 0);
    assert(ds4_qwen4_ram_auto_budget(UINT64_MAX, 16*GiB) == 16*GiB);
    assert(ds4_qwen4_ram_auto_budget(48*GiB, 0) == 0);
    ds4_qwen4_expert_slot slots[] = {
        {0, 64, 100, 1, 0, NULL},
        {64, 64, 2, 2, 0, NULL},
        {128, 64, 2, 3, 0, NULL}
    };
    assert(ds4_qwen4_expert_victim(slots, 3, 64, 10) == 1); /* hot survives; cold LRU leaves */
    slots[1].pinned_epoch = 10;
    assert(ds4_qwen4_expert_victim(slots, 3, 64, 10) == 2); /* later requested hit protected */
    slots[0].pinned_epoch = slots[2].pinned_epoch = 10;
    assert(ds4_qwen4_expert_victim(slots, 3, 64, 10) == SIZE_MAX); /* stage, never steal active */
    assert(ds4_qwen4_expert_victim(slots, 3, 65, 11) == SIZE_MAX); /* no undersized allocation */
    slots[0].frequency = 0; slots[0].capacity = 128;
    assert(ds4_qwen4_expert_victim(slots, 3, 65, 11) == 0); /* mixed capacity */
    assert(ds4_qwen4_expert_frequency_add(UINT64_MAX - 1, 3) == UINT64_MAX);
    unsigned last32[] = {32, 0}, stable64[] = {64, 0}, changed64[] = {32, 32};
    assert(ds4_qwen4_expert_window(last32, stable64, 2, 128, 1) == 128);
    assert(ds4_qwen4_expert_window(last32, changed64, 2, 128, 1) == 32);
    unsigned medium32[] = {24, 8}, medium64[] = {36, 28};
    assert(ds4_qwen4_expert_window(medium32, medium64, 2, 128, 1) == 64);
    assert(ds4_qwen4_expert_window(last32, stable64, 2, 1, 1) == 32); /* startup */
    unsigned sparse32[256] = {0}, sparse64[256] = {0};
    for (unsigned e = 0; e < 256; e++) { sparse32[e] = e % 2; sparse64[e] = 1; }
    assert(ds4_qwen4_expert_window(sparse32, sparse64, 256, 128, 4) == 128); /* sparse noise alone */
    int history[256] = {0}, route[2] = {0, 1};
    unsigned counts[24] = {0}, cursor = 0, rows = 0;
    for (unsigned t = 0; t < 1000; t++) ds4_qwen4_expert_observe(history, counts, 8, 2, &cursor, &rows, 1, route);
    assert(rows == 128 && counts[0] == 32 && counts[8] == 64 && counts[16] == 128);
    route[0] = 6; route[1] = 7;
    for (unsigned t = 0; t < 32; t++) ds4_qwen4_expert_observe(history, counts, 8, 2, &cursor, &rows, 1, route);
    assert(ds4_qwen4_expert_window(counts, counts + 8, 8, rows, 2) == 32);
    assert(counts[0] == 0 && counts[6] == 32); /* former hot set loses recent priority */
    for (unsigned t = 0; t < 96; t++) ds4_qwen4_expert_observe(history, counts, 8, 2, &cursor, &rows, 1, route);
    for (unsigned w = 0; w < 3; w++) {
        unsigned sum = 0;
        for (unsigned e = 0; e < 8; e++) { sum += counts[w * 8 + e]; if (e < 6) assert(counts[w * 8 + e] == 0); }
        assert(sum == (32u << w) * 2); /* conservation, and no distant history */
    }
    puts("Qwen expert policy: hot retention, LRU tie, pinning, capacities and adaptive recent window PASS");
    return 0;
}
