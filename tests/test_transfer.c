#include "ds4_transfer.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  ds4_resources *resources;
  void *dst[2];
  const void *src[2];
  uint64_t bytes[2];
  bool pending[2], fail_wait, fail_copy, fail_alloc;
  unsigned allocations, copies;
} fake;
static ds4_status allocate(void *p, uint64_t bytes, void **out) {
  fake *f = p;
  if (f->fail_alloc && f->allocations)
    return DS4_STATUS_NOMEM;
  *out = malloc(bytes);
  if (!*out)
    return DS4_STATUS_NOMEM;
  f->allocations++;
  return DS4_STATUS_OK;
}
static void release(void *p, void *memory) {
  fake *f = p;
  ds4_resource_totals totals;
  ds4_resources_totals(f->resources, 40, &totals);
  /* The budget must still cover storage while its destructor runs. */
  assert(totals.allocated[DS4_MEMORY_PINNED] == 254);
  f->allocations--;
  free(memory);
}
static ds4_status copy(void *p, unsigned slot, void *dst, const void *src,
                       uint64_t bytes) {
  fake *f = p;
  assert(!f->pending[slot]);
  if (f->fail_copy)
    return DS4_STATUS_INFERENCE;
  f->copies++;
  f->pending[slot] = true;
  f->dst[slot] = dst;
  f->src[slot] = src;
  f->bytes[slot] = bytes;
  return DS4_STATUS_OK;
}
static ds4_status wait(void *p, unsigned slot) {
  fake *f = p;
  if (f->fail_wait)
    return DS4_STATUS_INFERENCE;
  if (f->pending[slot])
    memcpy(f->dst[slot], f->src[slot], f->bytes[slot]);
  f->pending[slot] = false;
  return DS4_STATUS_OK;
}
static ds4_status publish(void *p) {
  ds4_status s = wait(p, 0);
  return s ? s : wait(p, 1);
}
int main(void) {
  ds4_resources *r = NULL;
  assert(!ds4_resources_create(&r));
  fake f = {.resources = r};
  ds4_transfer_ops ops = {&f, allocate, release, copy, wait, publish};
  ds4_transfer *t = NULL;
  ds4_resource_budget budget = {.limit = 253};
  assert(!ds4_resources_set_budget(r, DS4_MEMORY_PINNED, &budget));
  assert(ds4_transfer_create(&t, r, 40, 127, DS4_MEMORY_PINNED, &ops) ==
         DS4_STATUS_NOMEM);
  assert(!t && !f.allocations);
  budget.limit = 254;
  assert(!ds4_resources_set_budget(r, DS4_MEMORY_PINNED, &budget));
  f.fail_alloc = true;
  assert(ds4_transfer_create(&t, r, 40, 127, DS4_MEMORY_PINNED, &ops) ==
         DS4_STATUS_NOMEM);
  assert(!t && !f.allocations);
  f.fail_alloc = false;
  assert(!ds4_transfer_create(&t, r, 40, 127, DS4_MEMORY_PINNED, &ops));
  uint8_t source[16391], dest[16391];
  for (unsigned cycle = 0; cycle < 100; cycle++) {
    for (unsigned i = 0; i < sizeof(source); i++)
      source[i] = (i * 23 + cycle) % 251;
    memset(dest, 0, sizeof(dest));
    assert(!ds4_transfer_write(t, dest, source, sizeof(source)));
    assert(!ds4_transfer_publish(t));
    assert(!memcmp(source, dest, sizeof(source)));
    ds4_resource_totals totals;
    ds4_resources_totals(r, 40, &totals);
    assert(totals.allocated[DS4_MEMORY_PINNED] == 254 &&
           !totals.reserved[DS4_MEMORY_PINNED]);
  }
  f.fail_copy = true;
  assert(ds4_transfer_write(t, dest, source, sizeof(source)) ==
         DS4_STATUS_INFERENCE);
  f.fail_copy = false;
  assert(!ds4_transfer_write(t, dest, source, sizeof(source)));
  f.fail_wait = true;
  assert(ds4_transfer_close(t) == DS4_STATUS_INFERENCE && f.allocations == 2);
  ds4_resource_totals failed;
  ds4_resources_totals(r, 40, &failed);
  assert(failed.allocated[DS4_MEMORY_PINNED] == 254);
  f.fail_wait = false;
  assert(!ds4_transfer_close(t) && !f.allocations);
  /* A constructor failure under an in-flight operation must return its
   * slots and accounting immediately; the compute event owns neither. */
  uint64_t demand[DS4_MEMORY_COUNT] = {0};
  demand[DS4_MEMORY_HOST] = 65536;
  demand[DS4_MEMORY_PINNED] = 254;
  ds4_reservation *lease = NULL;
  assert(!ds4_resources_reserve(r, 40, demand, &lease));
  assert(!ds4_reservation_use(lease, 7));
  ds4_reservation_scope scope = ds4_reservation_enter(lease);
  f.fail_alloc = true;
  assert(ds4_transfer_create(&t, r, 40, 127, DS4_MEMORY_PINNED, &ops) ==
         DS4_STATUS_NOMEM);
  assert(!t && !f.allocations);
  f.fail_alloc = false;
  assert(!ds4_transfer_create(&t, r, 40, 127, DS4_MEMORY_PINNED, &ops));
  assert(!ds4_transfer_write(t, dest, source, sizeof(source)));
  assert(!ds4_transfer_close(t) && !f.allocations);
  ds4_reservation_leave(&scope);
  assert(ds4_reservation_cancel(lease) == DS4_STATUS_BUSY);
  ds4_resources_complete(r, 7);
  assert(!ds4_reservation_cancel(lease));
  ds4_resource_totals totals;
  ds4_resources_totals(r, 0, &totals);
  assert(!totals.roots && !totals.views);
  assert(!ds4_resources_close(r));
  puts("transfer: PASS bounded staging, deferred DMA, faults, drain/retry and "
       "100 alternations");
}
