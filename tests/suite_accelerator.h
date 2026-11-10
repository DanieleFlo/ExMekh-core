#ifndef DS4_TEST_SUITE_ACCELERATOR_H
#define DS4_TEST_SUITE_ACCELERATOR_H
#include "ds4_numeric.h"
#include "ds4_metal_memory.h"
#ifndef __APPLE__
extern int cudaMemGetInfo(size_t *, size_t *);
#endif
static inline ds4_status suite_numeric_create(ds4_numeric **out, ds4_resources *r, uint64_t owner) {
    if (!out)
        return DS4_STATUS_INVALID;
    *out = NULL;
#ifdef __APPLE__
    ds4_status status = ds4_resources_set_unified_memory(r, true);
    if (status)
        return status;
    status = ds4_numeric_metal(out, r, owner);
    if (status)
        return status;
    /* Model probes use the same configured host reserve as the service. */
    status = ds4_numeric_metal_budget(*out, ds4_metal_host_reserve_bytes());
    if (!status)
        return DS4_STATUS_OK;
    ds4_status cleanup = ds4_numeric_close(*out);
    if (!cleanup)
        *out = NULL;
    /* An unsuccessful close retains caller ownership for retry. */
    if (cleanup)
        return cleanup;
    return status;
#else
    return ds4_numeric_cuda(out, r, owner);
#endif
}
static inline int suite_memory_info(ds4_numeric *n, size_t *available, size_t *total) {
    if (available)
        *available = 0;
    if (total)
        *total = 0;
    if (!n || !available || !total)
        return 1;
#ifdef __APPLE__
    ds4_resources *r = ds4_numeric_resources(n, NULL);
    ds4_resource_budget b;
    if (ds4_resources_get_budget(r, DS4_MEMORY_DEVICE, &b))
        return 1;
    uint64_t free_bytes = 0;
    if (ds4_resources_available(r, DS4_MEMORY_DEVICE, &free_bytes))
        return 1;
    *total = b.limit;
    *available = free_bytes;
    return 0;
#else
    (void)n;
    int status = cudaMemGetInfo(available, total);
    if (status)
        *available = *total = 0;
    return status;
#endif
}
#endif
