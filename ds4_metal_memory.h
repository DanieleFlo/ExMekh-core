#ifndef DS4_METAL_MEMORY_H
#define DS4_METAL_MEMORY_H
#include "ds4_resources.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define DS4_METAL_SAFETY_RESERVE_BYTES (UINT64_C(2) << 30)
#define DS4_METAL_PRESSURE_CRITICAL 4u

/* One resource policy for the LLM, suite and standalone model probes.
 * Zero is an explicit no-reserve profile; malformed overrides retain the
 * historical default. Actual graph/restore costs are still accounted for. */
static inline uint64_t ds4_metal_host_reserve_bytes(void) {
    const char *value = getenv("DS4_METAL_HOST_RESERVE_BYTES");
    if (!value || !*value) return DS4_METAL_SAFETY_RESERVE_BYTES;
    uint64_t bytes = 0;
    for (const char *p = value; *p; p++) {
        if (*p < '0' || *p > '9' || bytes > (UINT64_MAX - (unsigned)(*p - '0')) / 10)
            return DS4_METAL_SAFETY_RESERVE_BYTES;
        bytes = bytes * 10 + (unsigned)(*p - '0');
    }
    return bytes;
}

typedef struct {
    uint64_t host_available, allocated, recommended, max_buffer, physical;
    unsigned pressure; /* XNU: 0 unknown, 1 normal, 2 warning, 4 critical. */
    bool unified;
} ds4_metal_memory;

/* Pure policy arithmetic, also exercised by the CPU-only foundation test.
 * Inactive anonymous pages require compression/swap. Bound the reclaimable
 * inactive cache by the file-backed page count (Strata 46c152a, arena_budget.h).
 * Speculative pages are already in free_count; purgeable pages can overlap
 * the inactive list and are deliberately not added a second time. */
static inline uint64_t ds4_metal_reclaimable_bytes(uint64_t free_pages, uint64_t inactive_pages,
                                                   uint64_t file_pages, uint64_t page_bytes) {
    uint64_t cache = inactive_pages < file_pages ? inactive_pages : file_pages;
    if (free_pages > UINT64_MAX - cache)
        return UINT64_MAX;
    uint64_t pages = free_pages + cache;
    return page_bytes && pages > UINT64_MAX / page_bytes ? UINT64_MAX : pages * page_bytes;
}
/* mmap uploads compete with the file cache. XNU prefers anonymous pageout
 * below this measured floor, so only the file-backed excess is available. */
static inline uint64_t ds4_metal_reclaimable_above_floor(uint64_t free_pages,
                                                        uint64_t inactive_pages,
                                                        uint64_t file_pages,
                                                        uint64_t file_floor_pages,
                                                        uint64_t page_bytes) {
    uint64_t excess = file_pages > file_floor_pages ? file_pages - file_floor_pages : 0;
    return ds4_metal_reclaimable_bytes(free_pages, inactive_pages, excess, page_bytes);
}
/* XNU f6217f8 vm_pageout.c recomputes this only during pageout. A floor
 * measured while a large GPU cache was wired can remain stale after release.
 * Use current noncompressed pages and the macOS divisor, retaining any larger
 * measured floor. This is file-cache policy, not an extra physical reserve. */
static inline uint64_t ds4_metal_filecache_floor(uint64_t measured, uint64_t noncompressed) {
    uint64_t current = (noncompressed / 27) * 10 + (noncompressed % 27) * 10 / 27;
    return measured > current ? measured : current;
}
static inline uint64_t ds4_metal_incremental_budget(uint64_t host, uint64_t recommended,
                                                    uint64_t allocated, uint64_t margin) {
    uint64_t device = recommended > allocated ? recommended - allocated : 0;
    uint64_t budget = host < device ? host : device;
    return budget > margin ? budget - margin : 0;
}

/* Warning pressure still admits measured capacity after the safety reserve.
 * Critical pressure admits no additional storage. Unknown pressure uses the
 * conservative VM sample; telemetry itself is never rewritten by policy. */
static inline uint64_t ds4_metal_host_capacity(const ds4_metal_memory *memory) {
    return memory->pressure >= DS4_METAL_PRESSURE_CRITICAL ? 0 : memory->host_available;
}
static inline uint64_t ds4_metal_device_capacity(const ds4_metal_memory *memory) {
    return memory->recommended > memory->allocated ? memory->recommended - memory->allocated : 0;
}
static inline uint64_t ds4_metal_memory_capacity(const ds4_metal_memory *memory, uint64_t margin) {
    return ds4_metal_incremental_budget(ds4_metal_host_capacity(memory), memory->recommended,
                                        memory->allocated, margin);
}

/* Plan only additional storage. Live Metal allocations are already excluded
 * by memory_capacity; future graph/auxiliary storage must be reserved first.
 * A caller's nonzero limit is a ceiling, never permission to spend the reserve. */
static inline uint64_t ds4_metal_stream_cache_budget(const ds4_metal_memory *memory,
                                                     uint64_t fixed_bytes,
                                                     uint64_t useful_bytes,
                                                     uint64_t requested_bytes,
                                                     uint64_t reserve_bytes) {
    uint64_t capacity = ds4_metal_memory_capacity(memory, reserve_bytes);
    uint64_t cache = capacity > fixed_bytes ? capacity - fixed_bytes : 0;
    if (cache > useful_bytes) cache = useful_bytes;
    if (requested_bytes && cache > requested_bytes) cache = requested_bytes;
    return cache;
}
/* A refreshed sample already excludes live cache storage. Credit it once
 * when planning its replacement, before subtracting the physical reserve.
 * Critical pressure must release capacity even when the cache owns memory. */
static inline uint64_t ds4_metal_stream_cache_rebudget(const ds4_metal_memory *memory,
                                                       uint64_t owned_bytes,
                                                       uint64_t future_bytes,
                                                       uint64_t useful_bytes,
                                                       uint64_t requested_bytes,
                                                       uint64_t reserve_bytes) {
    if (memory->pressure >= DS4_METAL_PRESSURE_CRITICAL) return 0;
    ds4_metal_memory reclaim = *memory;
    reclaim.host_available = owned_bytes > UINT64_MAX - reclaim.host_available
        ? UINT64_MAX : reclaim.host_available + owned_bytes;
    reclaim.allocated = reclaim.allocated > owned_bytes ? reclaim.allocated - owned_bytes : 0;
    return ds4_metal_stream_cache_budget(&reclaim, future_bytes, useful_bytes,
                                         requested_bytes, reserve_bytes);
}

/* Both Metal owners apply the same physical-budget policy. Reconciliation
 * updates measurements only: explicit caps and outstanding leases survive. */
static inline ds4_status ds4_metal_resources_reconcile(ds4_resources *resources,
                                                       const ds4_metal_memory *memory) {
    ds4_status status =
        ds4_resources_reconcile(resources, DS4_MEMORY_HOST, ds4_metal_host_capacity(memory), true);
    return status ? status
                  : ds4_resources_reconcile(resources, DS4_MEMORY_DEVICE,
                                            ds4_metal_device_capacity(memory), true);
}
static inline ds4_status ds4_metal_resources_set_budget(ds4_resources *resources,
                                                        const ds4_metal_memory *memory,
                                                        uint64_t margin) {
    ds4_resource_budget host = {.limit = memory->physical,
                                .margin = margin,
                                .global_free = ds4_metal_host_capacity(memory),
                                .global_known = true};
    ds4_resource_budget device = {.limit = memory->recommended,
                                  .margin = margin,
                                  .global_free = ds4_metal_device_capacity(memory),
                                  .global_known = true};
    ds4_status status = ds4_resources_set_budget(resources, DS4_MEMORY_HOST, &host);
    return status ? status : ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &device);
}
#endif
