/* No Metal device or GGUF required: policy must be testable on CPU builds. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_metal_memory.h"
#include "ds4_resources.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    const uint64_t gib = UINT64_C(1) << 30;
    uint64_t reserve = 2 * gib;
    assert(!setenv("DS4_METAL_HOST_RESERVE_BYTES", "2147483648", 1));
    assert(ds4_metal_reclaimable_bytes(10, 80, 20, 4096) == 30 * 4096);
    assert(ds4_metal_reclaimable_bytes(10, 20, 80, 16384) == 30 * 16384);
    assert(ds4_metal_reclaimable_bytes(10, 80, 0, 4096) == 10 * 4096);
    assert(ds4_metal_reclaimable_bytes(UINT64_MAX, 1, 1, 16384) == UINT64_MAX);
    assert(ds4_metal_reclaimable_bytes(UINT64_MAX / 2, 0, 0, 16384) == UINT64_MAX);
    assert(ds4_metal_reclaimable_above_floor(10, 80, 20, 15, 4096) == 15 * 4096);
    assert(ds4_metal_reclaimable_above_floor(10, 10, 80, 20, 4096) == 20 * 4096);
    assert(ds4_metal_reclaimable_above_floor(10, 80, 20, 20, 4096) == 10 * 4096);
    assert(ds4_metal_reclaimable_above_floor(10, 80, 20, UINT64_MAX, 4096) == 10 * 4096);
    assert(ds4_metal_reclaimable_above_floor(10, 80, 20, 0, 4096) == 30 * 4096);
    assert(ds4_metal_reclaimable_above_floor(UINT64_MAX, 80, 20, 15, 4096) == UINT64_MAX);
    /* The old wired workload left floor20; the released workload has 270
     * noncompressed pages. Do not spend the protected difference on weights. */
    assert(ds4_metal_filecache_floor(20, 270) == 100);
    assert(ds4_metal_filecache_floor(120, 270) == 120);
    assert(ds4_metal_filecache_floor(0, 0) == 0);
    assert(ds4_metal_filecache_floor(0, UINT64_MAX) > UINT64_MAX / 3);
    /* A warmed file cache is not all available to an anonymous SSD arena.
     * Keep current free pages spendable, without adding a physical reserve. */
    uint64_t floor = ds4_metal_filecache_floor(20, 270);
    assert(ds4_metal_reclaimable_above_floor(10, 80, 120, floor, 16384) == 30 * 16384);
    assert(ds4_metal_reclaimable_above_floor(10, 80, 80, floor, 16384) == 10 * 16384);
    assert(ds4_metal_incremental_budget(8 * gib, 18 * gib, 4 * gib, 2 * gib) == 6 * gib);
    assert(ds4_metal_incremental_budget(18 * gib, 12 * gib, 4 * gib, 2 * gib) == 6 * gib);
    assert(!ds4_metal_incremental_budget(2 * gib, 18 * gib, 0, 2 * gib));
    assert(!ds4_metal_incremental_budget(18 * gib, 4 * gib, 5 * gib, 0));
    assert(!ds4_metal_incremental_budget(0, 18 * gib, 0, 2 * gib));

    ds4_resources *r = NULL;
    assert(!ds4_resources_create(&r));
    assert(!ds4_resources_set_unified_memory(r, true));
    ds4_metal_memory memory = {.physical = 24 * gib,
                               .host_available = 8 * gib,
                               .recommended = 18 * gib,
                               .allocated = 4 * gib,
                               .pressure = 1,
                               .unified = true};
    assert(ds4_metal_memory_capacity(&memory, DS4_METAL_SAFETY_RESERVE_BYTES) == 6 * gib);
    assert(ds4_metal_stream_cache_budget(&memory, 2 * gib, 16 * gib, 0, reserve) == 4 * gib);
    assert(ds4_metal_stream_cache_budget(&memory, 2 * gib, 16 * gib, 3 * gib, reserve) == 3 * gib);
    assert(ds4_metal_stream_cache_budget(&memory, 2 * gib, 16 * gib, 8 * gib, reserve) == 4 * gib);
    assert(ds4_metal_stream_cache_budget(&memory, 2 * gib, gib, 0, reserve) == gib);
    assert(!ds4_metal_stream_cache_budget(&memory, 6 * gib, 16 * gib, 0, reserve));
    assert(!ds4_metal_stream_cache_budget(&memory, UINT64_MAX, UINT64_MAX, UINT64_MAX, reserve));
    assert(!ds4_metal_stream_cache_budget(&memory, 0, 0, 0, reserve));
    assert(!setenv("DS4_METAL_HOST_RESERVE_BYTES", "0", 1));
    assert(ds4_metal_host_reserve_bytes() == 0);
    reserve = 0;
    assert(ds4_metal_stream_cache_budget(&memory, 2 * gib, 16 * gib, 0, reserve) == 6 * gib);
    assert(ds4_metal_stream_cache_budget(&memory, 2 * gib, 16 * gib, 3 * gib, reserve) == 3 * gib);
    assert(ds4_metal_stream_cache_rebudget(&memory, 2 * gib, gib, 16 * gib, 0, reserve) == 9 * gib);
    assert(!ds4_metal_resources_set_budget(r, &memory, ds4_metal_host_reserve_bytes()));
    uint64_t no_reserve_free = 0;
    assert(!ds4_resources_available(r, DS4_MEMORY_DEVICE, &no_reserve_free));
    assert(no_reserve_free == 8 * gib);
    memory.pressure = DS4_METAL_PRESSURE_CRITICAL;
    assert(!ds4_metal_stream_cache_budget(&memory, 0, UINT64_MAX, 0, reserve));
    memory.pressure = 1;
    assert(!unsetenv("DS4_METAL_HOST_RESERVE_BYTES"));
    assert(ds4_metal_host_reserve_bytes() == 2 * gib);
    const char *invalid[] = {"", "-1", "1GB", " 0", "1.5", "18446744073709551616"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
        assert(!setenv("DS4_METAL_HOST_RESERVE_BYTES", invalid[i], 1));
        assert(ds4_metal_host_reserve_bytes() == 2 * gib);
    }
    assert(!setenv("DS4_METAL_HOST_RESERVE_BYTES", "2147483648", 1));
    reserve = 2 * gib;
    /* These samples exclude the two GiB already held by a live cache.
     * Growth, contraction and the reserve are recomputed without double debit. */
    ds4_metal_memory changing = memory;
    changing.host_available = 4 * gib;
    changing.allocated = 6 * gib;
    assert(ds4_metal_stream_cache_rebudget(&changing, 2 * gib, gib, 16 * gib, 0, reserve) == 3 * gib);
    changing.host_available = 2 * gib;
    assert(ds4_metal_stream_cache_rebudget(&changing, 2 * gib, gib, 16 * gib, 0, reserve) == gib);
    changing.host_available = gib;
    assert(!ds4_metal_stream_cache_rebudget(&changing, 2 * gib, gib, 16 * gib, 0, reserve));
    changing.host_available = 8 * gib;
    assert(ds4_metal_stream_cache_rebudget(&changing, 2 * gib, gib, 16 * gib, gib, reserve) == gib);
    changing.pressure = DS4_METAL_PRESSURE_CRITICAL;
    assert(!ds4_metal_stream_cache_rebudget(&changing, 2 * gib, 0, 16 * gib, 0, reserve));
    assert(!ds4_metal_resources_set_budget(r, &memory, DS4_METAL_SAFETY_RESERVE_BYTES));
    uint64_t free = 0;
    assert(!ds4_resources_available(r, DS4_MEMORY_DEVICE, &free) && free == 6 * gib);
    ds4_resource *root = NULL, *alias = NULL;
    ds4_resource_spec spec = {.owner = 42,
                              .requested = 2 * gib,
                              .allocated = 2 * gib,
                              .live = 2 * gib,
                              .location = DS4_MEMORY_DEVICE};
    assert(!ds4_resource_register(r, &spec, &root));
    assert(!ds4_resource_view(root, 0, gib, &alias));
    assert(!ds4_resources_available(r, DS4_MEMORY_HOST, &free) && free == 4 * gib);
    /* A fresh host sample excludes live storage. Subtract only the reserve. */
    ds4_resource_budget cap = {.limit = 9 * gib};
    assert(!ds4_resources_set_budget(r, DS4_MEMORY_DEVICE, &cap));
    assert(!ds4_metal_resources_reconcile(r, &memory));
    ds4_resource_budget preserved;
    assert(!ds4_resources_get_budget(r, DS4_MEMORY_DEVICE, &preserved));
    assert(preserved.limit == cap.limit && preserved.margin == cap.margin);
    assert(!ds4_resources_available(r, DS4_MEMORY_DEVICE, &free) && free == 6 * gib);
    uint64_t peak[DS4_MEMORY_COUNT] = {
        [DS4_MEMORY_DEVICE] = 4 * gib, [DS4_MEMORY_HOST] = gib, [DS4_MEMORY_PINNED] = gib + 1};
    assert(ds4_resources_can_reserve(r, peak, NULL) == DS4_STATUS_NOMEM);
    peak[DS4_MEMORY_PINNED]--;
    ds4_reservation *lease = NULL;
    assert(!ds4_resources_reserve(r, 42, peak, &lease));
    assert(!ds4_resources_available(r, DS4_MEMORY_HOST, &free) && !free);
    memory.pressure = 2;
    assert(!ds4_metal_resources_reconcile(r, &memory));
    ds4_resource_totals totals;
    ds4_resources_totals(r, 42, &totals);
    assert(totals.reserved[DS4_MEMORY_DEVICE] == peak[DS4_MEMORY_DEVICE]);
    assert(ds4_metal_memory_capacity(&memory, DS4_METAL_SAFETY_RESERVE_BYTES) == 6 * gib);
    assert(memory.host_available == 8 * gib);
    memory.pressure = DS4_METAL_PRESSURE_CRITICAL;
    assert(!ds4_metal_stream_cache_budget(&memory, 0, UINT64_MAX, 0, reserve));
    assert(!ds4_metal_resources_reconcile(r, &memory));
    assert(!ds4_metal_memory_capacity(&memory, DS4_METAL_SAFETY_RESERVE_BYTES));
    assert(memory.host_available == 8 * gib); /* Telemetry remains a measurement. */
    assert(!ds4_reservation_cancel(lease));
    memory.pressure = 2;
    memory.host_available = DS4_METAL_SAFETY_RESERVE_BYTES;
    assert(!ds4_metal_memory_capacity(&memory, DS4_METAL_SAFETY_RESERVE_BYTES));
    memory.host_available = 8 * gib;
    memory.pressure = 0; /* Unknown is distinct from elevated pressure. */
    assert(!ds4_metal_resources_reconcile(r, &memory));
    assert(!ds4_resources_available(r, DS4_MEMORY_DEVICE, &free) && free == 6 * gib);
    /* Failed sampling also invalidates stale capacity without losing owners. */
    ds4_metal_memory unavailable = {0};
    assert(!ds4_metal_resources_reconcile(r, &unavailable));
    assert(!ds4_resources_available(r, DS4_MEMORY_DEVICE, &free) && !free);
    assert(!ds4_resource_release(root));
    assert(!ds4_resource_release(alias));
    assert(!ds4_resources_close(r));
    puts("PASS: conservative Metal host pages, shared physical budget, reserve, staging and "
         "pressure");
    return 0;
}
