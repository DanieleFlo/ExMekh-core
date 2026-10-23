/* A failed fence must keep both storage and its accounting recoverable. */
#include "ds4_numeric_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int failed, live, allocation_failed, close_failed, closed;
} state;
static ds4_status allocate(void *p, uint64_t n, void **out) {
  if (((state *)p)->allocation_failed) {
    *out = NULL;
    return DS4_STATUS_NOMEM;
  }
  *out = malloc(n);
  if (!*out)
    return DS4_STATUS_NOMEM;
  ((state *)p)->live++;
  return DS4_STATUS_OK;
}
static ds4_status release(void *p, void *memory) {
  if (((state *)p)->failed)
    return DS4_STATUS_DEVICE_LOST;
  free(memory);
  ((state *)p)->live--;
  return DS4_STATUS_OK;
}
static ds4_status fence(void *p) {
  return ((state *)p)->failed ? DS4_STATUS_DEVICE_LOST : DS4_STATUS_OK;
}
static void close_backend(void *p) { ((state *)p)->closed++; }
static ds4_status prepare_close(void *p) {
  return ((state *)p)->close_failed ? DS4_STATUS_INFERENCE : DS4_STATUS_OK;
}
int main(void) {
  assert(!ds4_numeric_close(NULL));
  ds4_resources *r = NULL;
  ds4_numeric *n = NULL;
  ds4_num_buffer *a = NULL, *b = NULL;
  state s = {0};
  const ds4_numeric_ops ops = {.allocate = allocate,
                               .release = release,
                               .fence = fence,
                               .close = close_backend,
                               .prepare_close = prepare_close};
  assert(!ds4_resources_create(&r));
  assert(!ds4_numeric_create(&n, &ops, &s, r, 901, false));
  assert(!ds4_num_alloc(n, 128, DS4_NUM_F32, &a));
  assert(!ds4_num_alloc(n, 128, DS4_NUM_BF16, &b));
  s.failed = 1;
  assert(ds4_num_free(a) == DS4_STATUS_DEVICE_LOST);    /* owner keeps a */
  assert(ds4_num_dispose(b) == DS4_STATUS_DEVICE_LOST); /* context keeps b */
  assert(ds4_numeric_recover(n) == DS4_STATUS_DEVICE_LOST);
  assert(ds4_numeric_close(n) == DS4_STATUS_DEVICE_LOST);
  ds4_resource_totals totals;
  ds4_resources_totals(r, 901, &totals);
  assert(totals.roots == 5 && s.live == 2);
  s.failed = 0;
  assert(!ds4_numeric_recover(n));
  assert(s.live == 1);
  assert(ds4_numeric_close(n) == DS4_STATUS_BUSY);
  assert(!ds4_num_free(a));
  uint64_t demand[DS4_MEMORY_COUNT] = {0};
  demand[DS4_MEMORY_HOST] = 4096;
  ds4_reservation *lease = NULL;
  assert(!ds4_resources_reserve(r, 901, demand, &lease));
  assert(!ds4_reservation_use(lease, 7));
  ds4_reservation_scope scope = ds4_reservation_enter(lease);
  assert(!ds4_num_alloc(n, 128, DS4_NUM_F32, &a));
  ds4_reservation_leave(&scope);
  assert(ds4_num_free(a) == DS4_STATUS_BUSY);
  assert(s.live == 1); /* The registry must gate actual destruction too. */
  ds4_resources_complete(r, 7);
  assert(!ds4_num_free(a));
  assert(!s.live);
  assert(!ds4_reservation_cancel(lease));
  assert(!ds4_resources_reserve(r, 901, demand, &lease));
  assert(!ds4_reservation_use(lease, 8));
  scope = ds4_reservation_enter(lease);
  s.allocation_failed = 1;
  assert(ds4_num_alloc(n, 128, DS4_NUM_F32, &a) == DS4_STATUS_NOMEM);
  assert(!a);
  ds4_reservation_leave(&scope);
  assert(ds4_numeric_close(n) == DS4_STATUS_BUSY);
  assert(!s.live);
  ds4_resources_complete(r, 8);
  assert(!ds4_numeric_recover(n));
  assert(!ds4_reservation_cancel(lease));
  s.close_failed = 1;
  assert(ds4_numeric_close(n) == DS4_STATUS_INFERENCE && !s.closed);
  ds4_resources_totals(r, 901, &totals);
  assert(totals.roots == 1 && totals.allocated[DS4_MEMORY_HOST]);
  s.close_failed = 0;
  assert(!ds4_numeric_close(n));
  assert(s.closed == 1);
  /* A context created under an in-flight lease must preserve its backend
   * until its own metadata can be released; the caller can then retry. */
  assert(!ds4_resources_reserve(r, 901, demand, &lease));
  assert(!ds4_reservation_use(lease, 9));
  scope = ds4_reservation_enter(lease);
  assert(!ds4_numeric_create(&n, &ops, &s, r, 901, false));
  ds4_reservation_leave(&scope);
  assert(ds4_numeric_close(n) == DS4_STATUS_BUSY && s.closed == 1);
  ds4_resources_complete(r, 9);
  assert(!ds4_numeric_close(n) && s.closed == 2);
  assert(!ds4_reservation_cancel(lease));
  ds4_resources_totals(r, 0, &totals);
  assert(!totals.roots && !totals.views && !s.live);
  assert(!ds4_resources_close(r));
  puts("numeric failed-fence ownership: PASS");
  return 0;
}
