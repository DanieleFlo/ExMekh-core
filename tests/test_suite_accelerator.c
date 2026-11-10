/* Probe factory ownership on injected failures, without opening a GPU. */
#include "suite_accelerator.h"
#include <assert.h>
#include <stdio.h>

static ds4_status unified_status, create_status, budget_status, close_status;
static ds4_status query_status, available_status;
static unsigned closes;
static uint64_t reserve = 2ull * 1024 * 1024 * 1024;
static ds4_numeric *context = (ds4_numeric *)(uintptr_t)1;
static ds4_resources *resources = (ds4_resources *)(uintptr_t)2;

ds4_status ds4_resources_set_unified_memory(ds4_resources *r, bool enabled) {
    assert(r == resources && enabled);
    return unified_status;
}
static ds4_status create(ds4_numeric **out, ds4_resources *r, uint64_t owner) {
    assert(r == resources && owner == 100);
    *out = create_status ? NULL : context;
    return create_status;
}
ds4_status ds4_numeric_metal(ds4_numeric **out, ds4_resources *r, uint64_t owner) {
    return create(out, r, owner);
}
ds4_status ds4_numeric_cuda(ds4_numeric **out, ds4_resources *r, uint64_t owner) {
    return create(out, r, owner);
}
ds4_status ds4_numeric_metal_budget(ds4_numeric *n, uint64_t margin) {
    assert(n == context && margin == reserve);
    return budget_status;
}
ds4_status ds4_numeric_close(ds4_numeric *n) {
    assert(n == context);
    closes++;
    return close_status;
}
ds4_resources *ds4_numeric_resources(ds4_numeric *n, uint64_t *owner) {
    assert(n == context && !owner);
    return resources;
}
ds4_status ds4_resources_get_budget(ds4_resources *r, ds4_memory_location location,
                                     ds4_resource_budget *out) {
    assert(r == resources && location == DS4_MEMORY_DEVICE);
    *out = (ds4_resource_budget){.limit = 34};
    return query_status;
}
ds4_status ds4_resources_available(ds4_resources *r, ds4_memory_location location,
                                    uint64_t *out) {
    assert(r == resources && location == DS4_MEMORY_DEVICE);
    *out = 12;
    return available_status;
}
#ifndef __APPLE__
int cudaMemGetInfo(size_t *available, size_t *total) {
    *available = 12;
    *total = 34;
    return query_status || available_status;
}
#endif

int main(void) {
    assert(!setenv("DS4_METAL_HOST_RESERVE_BYTES", "2147483648", 1));
    ds4_numeric *out = context;
    assert(suite_numeric_create(NULL, resources, 100) == DS4_STATUS_INVALID);
#ifdef __APPLE__
    unified_status = DS4_STATUS_BUSY;
    assert(suite_numeric_create(&out, resources, 100) == DS4_STATUS_BUSY);
    assert(!out && !closes);
    unified_status = DS4_STATUS_OK;
#endif
    create_status = DS4_STATUS_NOMEM;
    out = context;
    assert(suite_numeric_create(&out, resources, 100) == DS4_STATUS_NOMEM);
    assert(!out && !closes);
    create_status = DS4_STATUS_OK;
#ifdef __APPLE__
    budget_status = DS4_STATUS_NOT_READY;
    assert(suite_numeric_create(&out, resources, 100) == DS4_STATUS_NOT_READY);
    assert(!out && closes == 1);
    close_status = DS4_STATUS_DEVICE_LOST;
    assert(suite_numeric_create(&out, resources, 100) == DS4_STATUS_DEVICE_LOST);
    assert(out == context && closes == 2);
    close_status = DS4_STATUS_OK;
    assert(ds4_numeric_close(out) == DS4_STATUS_OK);
    budget_status = DS4_STATUS_OK;
#endif
    assert(suite_numeric_create(&out, resources, 100) == DS4_STATUS_OK);
    assert(out == context);
    assert(ds4_numeric_close(out) == DS4_STATUS_OK);
    reserve = 0;
    assert(!setenv("DS4_METAL_HOST_RESERVE_BYTES", "0", 1));
    assert(suite_numeric_create(&out, resources, 100) == DS4_STATUS_OK);
    assert(ds4_numeric_close(out) == DS4_STATUS_OK);
    size_t available = 42, total = 42;
    assert(suite_memory_info(NULL, &available, &total));
    assert(!available && !total);
    total = 42;
    assert(suite_memory_info(context, NULL, &total) && !total);
    available = 42;
    assert(suite_memory_info(context, &available, NULL) && !available);
    assert(!suite_memory_info(context, &available, &total));
    assert(available == 12 && total == 34);
    query_status = DS4_STATUS_NOT_READY;
    assert(suite_memory_info(context, &available, &total));
    assert(!available && !total);
    query_status = DS4_STATUS_OK;
    available_status = DS4_STATUS_NOT_READY;
    assert(suite_memory_info(context, &available, &total));
    assert(!available && !total);
    puts("suite accelerator factory ownership: PASS");
    return 0;
}
