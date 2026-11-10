#ifndef DS4_METAL_DEVICE_H
#define DS4_METAL_DEVICE_H

/* Platform sampling stays inside the Objective-C backends. The engine and
 * CPU tests depend only on the policy and resource API in ds4_metal_memory.h. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ds4_metal_memory.h"
#include <mach/mach.h>
#include <sys/sysctl.h>

/* Best-effort read-only XNU telemetry: 1 normal, 2 warning, 4 critical; 0 means
 * unavailable, never a claim of normal pressure. No private symbol is linked. */
static inline unsigned ds4_metal_host_pressure(void) {
    int level = 0;
    size_t bytes = sizeof(level);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &bytes, NULL, 0) ||
        bytes != sizeof(level) || level < 1)
        return 0;
    return (unsigned)level;
}
static inline bool ds4_metal_vm_pages(const char *name, unsigned *pages) {
    size_t bytes = sizeof(*pages);
    return !sysctlbyname(name, pages, &bytes, NULL, 0) && bytes == sizeof(*pages);
}
static inline bool ds4_metal_host_memory_above_floor(uint64_t *out, unsigned *pressure,
                                                     bool respect_file_floor) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    if (!out)
        return false;
    host_t host = mach_host_self();
    bool valid =
        host_page_size(host, &page) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count) == KERN_SUCCESS;
    (void)mach_port_deallocate(mach_task_self(), host);
    if (!valid)
        return false;
    /* XNU host_statistics64 can return a cached snapshot for one second
     * after only 2–10 calls. Dynamic cache growth/release must use current
     * free pages. These UINT sysctls are read-only counters (XNU f6217f8,
     * vm_unix.c). Unlike Mach free_count, this free count excludes speculative.
     * Keep the inactive allowance bounded by current pageable file pages. */
    unsigned free_pages, speculative, file_pages;
    if (!ds4_metal_vm_pages("vm.page_free_count", &free_pages) ||
        !ds4_metal_vm_pages("vm.page_speculative_count", &speculative) ||
        !ds4_metal_vm_pages("vm.page_pageable_external_count", &file_pages))
        return false;
    uint64_t floor = 0;
    if (respect_file_floor) {
        size_t bytes = sizeof(floor);
        unsigned anonymous, throttled;
        if (sysctlbyname("vm.vm_page_filecache_min", &floor, &bytes, NULL, 0) ||
            (bytes != sizeof(uint32_t) && bytes != sizeof(uint64_t)) ||
            !ds4_metal_vm_pages("vm.page_pageable_internal_count", &anonymous) ||
            !ds4_metal_vm_pages("vm.page_throttled_count", &throttled))
            return false;
        /* Pageable already includes speculative pages. Throttled pages are
         * excluded from AVAILABLE_NON_COMPRESSED_MEMORY (vm_page.h). */
        uint64_t pages = (uint64_t)free_pages + file_pages + anonymous;
        floor = ds4_metal_filecache_floor(floor, pages > throttled ? pages - throttled : 0);
    }
    unsigned level = ds4_metal_host_pressure();
    if (pressure)
        *pressure = level;
    /* Speculative file pages are credited with free, so cannot also bound
     * inactive reclaim when the Mach inactive snapshot is cached. */
    uint64_t nonspeculative = file_pages > speculative ? file_pages - speculative : 0;
    *out =
        ds4_metal_reclaimable_above_floor((uint64_t)free_pages + speculative,
                                          vm.inactive_count, nonspeculative, floor, page);
    return true;
}
static inline bool ds4_metal_device_memory_above_floor(id<MTLDevice> device, ds4_metal_memory *out,
                                                       bool respect_file_floor) {
    if (!out)
        return false;
    *out = (ds4_metal_memory){0};
    if (!device || !ds4_metal_host_memory_above_floor(&out->host_available, &out->pressure,
                                                     respect_file_floor))
        return false;
    out->physical = NSProcessInfo.processInfo.physicalMemory;
    out->allocated = device.currentAllocatedSize;
    out->recommended = device.recommendedMaxWorkingSetSize;
    out->max_buffer = device.maxBufferLength;
    out->unified = device.hasUnifiedMemory;
    return true;
}
static inline bool ds4_metal_device_memory(id<MTLDevice> device, ds4_metal_memory *out) {
    return ds4_metal_device_memory_above_floor(device, out, false);
}
/* Owned anonymous buffers compete with the file-cache floor even when
 * filled by pread. Resident file-backed wrapping uses device_memory above. */
static inline bool ds4_metal_owned_device_memory(id<MTLDevice> device, ds4_metal_memory *out) {
    return ds4_metal_device_memory_above_floor(device, out, true);
}
#endif
