/* Inspect advisory ranges without issuing I/O or allocating model storage. */
#define fcntl record_fcntl
#define posix_fadvise record_fadvise
#include "../ds4_bundle.c"
#include "../ds4_weight_cache.c"
#include <assert.h>
#include <stdarg.h>

typedef struct { int fd; uint64_t offset, bytes; } hint;
static hint recorded[32];
static size_t hint_count;
static int record_hint(int fd, uint64_t offset, uint64_t bytes) {
    assert(hint_count < sizeof(recorded) / sizeof(recorded[0]));
    recorded[hint_count++] = (hint){fd, offset, bytes};
    return -1; /* Advice failures must never turn an authoritative load ready. */
}
#ifdef __APPLE__
int record_fcntl(int fd, int command, ...) {
    assert(command == F_RDADVISE);
    va_list arguments;
    va_start(arguments, command);
    const struct radvisory *advice = va_arg(arguments, struct radvisory *);
    int result = record_hint(fd, advice->ra_offset, advice->ra_count);
    va_end(arguments);
    return result;
}
#else
int record_fadvise(int fd, off_t offset, off_t bytes, int advice) {
    assert(advice == POSIX_FADV_WILLNEED);
    return record_hint(fd, offset, bytes);
}
#endif
int main(void) {
    ds4_weight_tensor tensors[] = {
        {.file = 0, .offset = 8, .bytes = 100},
        {.file = 0, .offset = 128, .bytes = (1u << 20) + 17},
        {.file = 1, .offset = 64, .bytes = 1024}};
    mapping maps[] = {{.fd = 17}, {.fd = 29}};
    ds4_bundle bundle = {.tensors = tensors, .tensor_count = 3, .maps = maps};
    uint64_t offsets[] = {UINT64_MAX, 0, (1u << 20) + 32};
    ds4_weight_group group = {.bundle = &bundle, .offsets = offsets,
                              .bytes = (1u << 20) + 32 + 1024};
    ds4_weight_group *groups[] = {&group};
    ds4_weight_cache cache;
    assert(!ds4_weight_cache_init(&cache, groups, 1));
    ds4_weight_budget before, after;
    assert(!ds4_weight_cache_budget(&cache, &before));
    assert(ds4_weight_group_hint(NULL, 1) == DS4_STATUS_INVALID);
    assert(ds4_weight_group_hint(&group, 0) == DS4_STATUS_INVALID);
    assert(ds4_weight_group_hint(&group, (16u << 20) + 1) == DS4_STATUS_INVALID);
    assert(ds4_weight_cache_hint(&cache, 1, 1) == DS4_STATUS_INVALID);
    assert(hint_count == 0);
    assert(!ds4_weight_cache_hint(&cache, 0, (1u << 20) + 17 + 500));
    assert(hint_count == 3);
    assert(recorded[0].fd == 17 && recorded[0].offset == 128 && recorded[0].bytes == (1u << 20));
    assert(recorded[1].fd == 17 && recorded[1].offset == 128 + (1u << 20) && recorded[1].bytes == 17);
    assert(recorded[2].fd == 29 && recorded[2].offset == 64 && recorded[2].bytes == 500);
    assert(!ds4_weight_cache_budget(&cache, &after));
    assert(before.total == after.total && before.minimum == after.minimum);
    assert(before.resident == after.resident && before.limit == after.limit);
    assert(before.loads == after.loads && before.hits == after.hits);
    assert(before.evictions == after.evictions && before.transferred == after.transferred);
    assert(!group.storage && !group.resident);
    group.storage = &bundle;
    assert(!ds4_weight_group_hint(&group, 4096) && hint_count == 3);
    group.storage = NULL;
    hint_count = 0;
    assert(!ds4_weight_group_hint(&group, 16u << 20));
    assert(hint_count == 3 && recorded[2].bytes == 1024);
    puts("group hint: PASS selected ranges, cumulative bound, failed advice and unchanged ownership");
}
