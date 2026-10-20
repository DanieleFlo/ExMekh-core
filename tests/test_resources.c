#define _POSIX_C_SOURCE 200809L
#include "ds4_resources.h"
#include "ds4_checkpoint.h"
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static int destroyed, fail_destroy;
static ds4_status destroy(void *opaque) {
    assert(opaque == &destroyed);
    if (fail_destroy) return DS4_STATUS_INFERENCE;
    destroyed++; return DS4_STATUS_OK;
}
static ds4_resource_spec spec(uint64_t owner, uint64_t bytes) {
    ds4_resource_spec s = {.owner=owner, .requested=bytes, .allocated=bytes,
        .live=bytes, .location=DS4_MEMORY_DEVICE, .format=32, .device=0};
    return s;
}
static void lifetime(void) {
    ds4_resources *r = NULL; assert(!ds4_resources_create(&r));
    ds4_resource *base, *view, *nested;
    ds4_resource_spec s = spec(17, 1024); s.destroy = destroy; s.opaque = &destroyed;
    assert(!ds4_resource_register(r, &s, &base));
    assert(!ds4_resource_view(base, 512, 512, &view));
    assert(!ds4_resource_view(view, 128, 128, &nested));
    ds4_resource *invalid = (void *)1;
    assert(ds4_resource_view(view, UINT64_MAX, 1, &invalid) == DS4_STATUS_INVALID && !invalid);
    ds4_resource_info info; assert(!ds4_resource_inspect(nested, &info));
    assert(info.offset == 640 && info.spec.allocated == 0);
    ds4_resource_totals totals; ds4_resources_totals(r, 17, &totals);
    assert(totals.roots == 1 && totals.views == 2 && totals.allocated[0] == 1024);
    assert(!ds4_resource_use(nested, 9));
    assert(ds4_resource_release(base) == DS4_STATUS_BUSY);
    ds4_resources_complete(r, 8); assert(ds4_resource_release(view) == DS4_STATUS_BUSY);
    ds4_resources_complete(r, 9);
    assert(!ds4_resource_release(base) && !destroyed);
    assert(!ds4_resource_release(view) && !destroyed);
    fail_destroy = 1;
    assert(ds4_resource_release(nested) == DS4_STATUS_INFERENCE);
    assert(!ds4_resource_inspect(nested, &info));
    fail_destroy = 0; assert(!ds4_resource_release(nested) && destroyed == 1);
    ds4_resources_totals(r, 0, &totals);
    assert(!totals.roots && !totals.views && !totals.allocated[0]);
    s = spec(99, UINT64_MAX); assert(!ds4_resource_register(r, &s, &base));
    s = spec(99, 1); assert(ds4_resource_register(r, &s, &view) == DS4_STATUS_NOMEM);
    assert(!ds4_resource_release(base));
    s = spec(1, 10); s.live = 11; assert(ds4_resource_register(r, &s, &base) == DS4_STATUS_INVALID);
    assert(!ds4_resources_close(r));
}
static void budgets(void) {
    ds4_resources *r = NULL; assert(!ds4_resources_create(&r));
    ds4_resource_budget b = {1000, 10, 700, true};
    assert(!ds4_resources_set_budget(r, DS4_MEMORY_DEVICE, &b));
    ds4_resource *p; ds4_resource_spec s = spec(4, 400);
    assert(!ds4_resource_register(r, &s, &p));
    uint64_t available; assert(!ds4_resources_available(r, 0, &available) && available == 290);
    /* Fresh global free already includes p: do not subtract p again. */
    assert(!ds4_resources_reconcile(r, 0, 700, true));
    assert(!ds4_resources_available(r, 0, &available) && available == 590);
    uint64_t bytes[DS4_MEMORY_COUNT] = {200, 0, 0, 30, 50, 0};
    ds4_reservation *a, *other;
    assert(!ds4_resources_reserve(r, 4, bytes, &a));
    assert(!ds4_reservation_use(a, 42));
    assert(ds4_reservation_cancel(a) == DS4_STATUS_BUSY);
    ds4_resource_totals totals; ds4_resources_totals(r, 4, &totals);
    assert(totals.reserved[0] == 200 && totals.reserved[DS4_MEMORY_PINNED] == 50);
    bytes[0] = 391; assert(ds4_resources_reserve(r, 7, bytes, &other) == DS4_STATUS_NOMEM);
    assert(ds4_resources_close(r) == DS4_STATUS_BUSY);
    s = spec(4, 100); ds4_resource *committed;
    assert(!ds4_reservation_commit(a, &s, &committed));
    assert(ds4_resource_release(committed) == DS4_STATUS_BUSY);
    ds4_resources_complete(r, 42);
    assert(!ds4_reservation_cancel(a)); assert(!ds4_resource_release(committed));
    assert(!ds4_resource_release(p));
    b = (ds4_resource_budget){.limit=100};
    assert(!ds4_resources_set_budget(r, DS4_MEMORY_HOST, &b));
    uint64_t host_peak[DS4_MEMORY_COUNT] = {0};
    host_peak[DS4_MEMORY_HOST] = host_peak[DS4_MEMORY_PINNED] = 60;
    assert(ds4_resources_reserve(r, 4, host_peak, &other) == DS4_STATUS_NOMEM);
    ds4_resources_totals(r, 0, &totals);
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++) assert(!totals.allocated[i] && !totals.reserved[i]);
    assert(!ds4_resources_close(r));
}
static void trimming(void) {
    ds4_resources *r; assert(!ds4_resources_create(&r));
    ds4_resource *a, *b, *v, *account_only;
    ds4_resource_spec s = spec(1, 100); s.rebuildable = true;
    s.destroy = destroy; s.opaque = &destroyed;
    assert(!ds4_resource_register(r, &s, &a)); s.owner = 2;
    assert(!ds4_resource_register(r, &s, &b));
    s.owner = 1; s.destroy = NULL;
    assert(!ds4_resource_register(r, &s, &account_only));
    assert(!ds4_resource_view(a, 0, 10, &v));
    uint64_t freed; assert(!ds4_resources_trim(r, 1, &freed) && !freed);
    assert(!ds4_resource_release(v));
    fail_destroy = 1;
    assert(ds4_resources_trim(r, 1, &freed) == DS4_STATUS_INFERENCE && !freed);
    ds4_resource_info info;
    assert(!ds4_resource_inspect(a, &info));
    fail_destroy = 0;
    int prior = destroyed;
    assert(!ds4_resources_trim(r, 1, &freed) && freed == 100);
    assert(destroyed == prior + 1);
    assert(!ds4_resource_inspect(account_only, &info) && info.spec.allocated == 100);
    assert(!ds4_resource_inspect(b, &info) && info.spec.owner == 2);
    assert(!ds4_resource_release(account_only));
    assert(!ds4_resource_release(b)); assert(!ds4_resources_close(r));
}
static void *concurrent(void *opaque) {
    ds4_resources *r = opaque;
    for (int i = 0; i < 2000; i++) {
        uint64_t bytes[DS4_MEMORY_COUNT] = {40, 0, 0, 20, 20, 0};
        ds4_reservation *p = NULL;
        ds4_status status = ds4_resources_reserve(r, 8, bytes, &p);
        if (status == DS4_STATUS_NOMEM) continue;
        assert(!status);
        ds4_resource_totals totals; ds4_resources_totals(r, 0, &totals);
        assert(totals.reserved[0] <= 128);
        assert(!ds4_reservation_cancel(p));
    }
    return NULL;
}
static void concurrency(void) {
    ds4_resources *r; assert(!ds4_resources_create(&r));
    ds4_resource_budget b = {.limit=128}; assert(!ds4_resources_set_budget(r, 0, &b));
    pthread_t threads[8];
    for (unsigned i = 0; i < 8; i++) assert(!pthread_create(&threads[i], NULL, concurrent, r));
    for (unsigned i = 0; i < 8; i++) assert(!pthread_join(threads[i], NULL));
    assert(!ds4_resources_close(r));
}
static void checkpoints(void) {
    uint8_t identity[32] = {17}, config[32] = {42};
    FILE *fp = tmpfile(); assert(fp);
    const char body[] = "KV recurrent MTP mRoPE";
    assert(!fseek(fp, DS4_CHECKPOINT_HEADER_BYTES, SEEK_SET));
    assert(fwrite(body, 1, sizeof(body), fp) == sizeof(body));
    uint64_t rng = 0x123456789abcdef, got = 0, payload = 0;
    assert(!ds4_checkpoint_seal(fp, identity, config, sizeof(body), &rng, sizeof(rng)));
    size_t bytes = sizeof(got);
    assert(!ds4_checkpoint_validate(fp, identity, config, &payload, &got, &bytes));
    assert(payload == sizeof(body) && got == rng && ftell(fp) == DS4_CHECKPOINT_HEADER_BYTES);
    /* Reusing a stream must not leave a stale trailer outside the envelope. */
    assert(!fseek(fp, 0, SEEK_END));
    assert(fwrite("stale trailer", 1, 13, fp) == 13);
    assert(!ds4_checkpoint_seal(fp, identity, config, sizeof(body), &rng, sizeof(rng)));
    bytes = sizeof(got);
    assert(!ds4_checkpoint_validate(fp, identity, config, &payload, &got, &bytes));
    identity[1] = 1; bytes = sizeof(got);
    assert(ds4_checkpoint_validate(fp, identity, config, &payload, &got, &bytes) == DS4_STATUS_INVALID);
    identity[1] = 0; bytes = 0;
    assert(ds4_checkpoint_validate(fp, identity, config, &payload, NULL, &bytes) == DS4_STATUS_NOMEM);
    assert(bytes == sizeof(rng));
    /* Overflowing declared extents and a stale ABI are rejected before output. */
    const long positions[] = {8, 16, 24};
    for (unsigned i = 0; i < 3; i++) {
        assert(!fseek(fp, positions[i], SEEK_SET)); int saved = fgetc(fp); assert(saved != EOF);
        assert(!fseek(fp, positions[i], SEEK_SET)); assert(fputc(saved ^ 128, fp) != EOF); assert(!fflush(fp));
        bytes = sizeof(got);
        assert(ds4_checkpoint_validate(fp, identity, config, &payload, &got, &bytes) == DS4_STATUS_INVALID);
        assert(!fseek(fp, positions[i], SEEK_SET)); assert(fputc(saved, fp) != EOF); assert(!fflush(fp));
    }
    assert(!fseek(fp, DS4_CHECKPOINT_HEADER_BYTES + 3, SEEK_SET)); assert(fputc(255, fp) != EOF);
    assert(!fflush(fp)); bytes = sizeof(got); got = 999;
    assert(ds4_checkpoint_validate(fp, identity, config, &payload, &got, &bytes) == DS4_STATUS_INVALID && got == 999);
    fclose(fp);
}
int main(void) {
    lifetime(); budgets(); trimming(); concurrency(); checkpoints();
    puts("PASS: resource lifetime, aliases, events, budgets, concurrent reservations and checkpoint validation");
    return 0;
}
