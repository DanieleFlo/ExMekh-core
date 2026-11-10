/* A refused context close must leave its synchronous upload storage usable. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_numeric.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    ds4_resources *resources = NULL;
    ds4_numeric *numeric = NULL;
    ds4_reservation *lease = NULL;
    assert(!ds4_resources_create(&resources));
    assert(!ds4_resources_set_unified_memory(resources, true));
    uint64_t demand[DS4_MEMORY_COUNT] = {[DS4_MEMORY_HOST] = 16u << 20};
    assert(!ds4_resources_reserve(resources, 901, demand, &lease));
    assert(!ds4_reservation_use(lease, 7));
    ds4_reservation_scope scope = ds4_reservation_enter(lease);
    assert(!ds4_numeric_metal(&numeric, resources, 901));
    ds4_reservation_leave(&scope);
    ds4_resource_totals before, after;
    ds4_resources_totals(resources, 901, &before);
    assert(ds4_numeric_close(numeric) == DS4_STATUS_BUSY);
    ds4_resources_totals(resources, 901, &after);
    assert(before.roots == after.roots);
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++)
        assert(before.allocated[i] == after.allocated[i]);

    char path[] = "/tmp/ds4-numeric-close.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0 && !unlink(path));
    const float expected[] = {1.25f, -2.5f, 3.75f, 8.0f};
    assert(write(fd, expected, sizeof(expected)) == (ssize_t)sizeof(expected));
    ds4_num_buffer *buffer = NULL;
    assert(!ds4_num_alloc(numeric, 4, DS4_NUM_F32, &buffer));
    assert(!ds4_num_buffer_stage_file(buffer, 0, fd, 0, sizeof(expected)));
    float actual[4];
    assert(!ds4_num_read(numeric, ds4_num_data(buffer), actual, sizeof(actual)));
    assert(!memcmp(actual, expected, sizeof(actual)));
    assert(!ds4_num_free(buffer));
    assert(!close(fd));
    ds4_resources_complete(resources, 7);
    assert(!ds4_numeric_close(numeric));
    assert(!ds4_reservation_cancel(lease));
    ds4_resources_totals(resources, 0, &after);
    assert(!after.roots && !after.views);
    assert(!ds4_resources_close(resources));
    puts("Metal Numeric close BUSY preserves upload and retry: PASS");
    return 0;
}
