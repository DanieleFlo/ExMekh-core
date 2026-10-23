#include "ds4_weight_cache.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
  bool fail_free;
  uint64_t live, peak;
} fake;
static ds4_status allocate(void *p, uint64_t bytes, void **out) {
  fake *f = p;
  uint64_t *storage = malloc(bytes + 8);
  if (!storage)
    return DS4_STATUS_NOMEM;
  *storage = bytes;
  *out = storage + 1;
  f->live += bytes;
  if (f->peak < f->live)
    f->peak = f->live;
  return DS4_STATUS_OK;
}
static ds4_status write_bytes(void *p, void *dst, uint64_t off, const void *src,
                              uint64_t bytes) {
  (void)p;
  memcpy((char *)dst + off, src, bytes);
  return DS4_STATUS_OK;
}
static ds4_status release(void *p, void *storage) {
  fake *f = p;
  if (f->fail_free)
    return DS4_STATUS_INFERENCE;
  uint64_t *head = (uint64_t *)storage - 1;
  f->live -= *head;
  free(head);
  return DS4_STATUS_OK;
}
static ds4_status fence(void *p) {
  (void)p;
  return DS4_STATUS_OK;
}
int main(int argc, char **argv) {
  assert(argc == 3);
  ds4_resources *r = NULL;
  ds4_bundle *b = NULL;
  assert(!ds4_resources_create(&r));
  assert(!ds4_bundle_open(&b, argv[1], argv[2], r, 42));
  fake f = {0};
  ds4_weight_backend ops = {&f, allocate, write_bytes, fence, release};
  ds4_weight_group *groups[3] = {0};
  size_t index = 0;
  const ds4_weight_tensor *tensor = ds4_bundle_tensor(b, index);
  for (unsigned i = 0; i < 3; i++)
    assert(!ds4_weight_group_create_selected(groups + i, b, tensor->component,
                                             &index, 1, &ops));
  ds4_weight_cache c;
  assert(!ds4_weight_cache_init(&c, groups, 3));
  ds4_weight_budget budget;
  assert(ds4_weight_cache_budget(NULL, &budget) == DS4_STATUS_INVALID);
  assert(ds4_weight_cache_budget(&c, NULL) == DS4_STATUS_INVALID);
  assert(!ds4_weight_cache_missing(NULL));
  uint64_t one = ds4_weight_cache_minimum(&c);
  assert(ds4_weight_cache_limit(&c, one - 1) == DS4_STATUS_INVALID);
  assert(!ds4_weight_cache_limit(&c, one * 2));
  assert(ds4_weight_cache_preload(&c, 4) == DS4_STATUS_INVALID);
  assert(!ds4_weight_cache_preload(&c, 3) && !f.live);
  for (unsigned pass = 0; pass < 50; pass++)
    for (unsigned i = 0; i < 3; i++) {
      assert(!ds4_weight_cache_acquire(&c, i));
      assert(f.live <= one * 2);
      const void *storage;
      uint64_t off;
      assert(!ds4_weight_group_tensor(groups[i], tensor->name, &storage, &off));
      assert(!memcmp((const char *)storage + off, ds4_bundle_tensor_data(b, 0),
                     tensor->bytes));
    }
  assert(!ds4_weight_cache_budget(&c, &budget));
  assert(budget.evictions && budget.hits && budget.loads == 101);
  assert(budget.limit == one * 2 && budget.resident <= budget.limit);
  assert(!ds4_weight_cache_limit(&c, UINT64_MAX));
  assert(!ds4_weight_cache_preload(&c, 3));
  assert(!ds4_weight_cache_budget(&c, &budget));
  uint64_t loads = budget.loads;
  for (unsigned pass = 0; pass < 8; pass++)
    for (unsigned i = 0; i < 3; i++)
      assert(!ds4_weight_cache_acquire(&c, i));
  assert(!ds4_weight_cache_budget(&c, &budget));
  assert(budget.loads == loads && f.live == one * 3);
  f.fail_free = true;
  assert(ds4_weight_cache_release(&c) == DS4_STATUS_INFERENCE && f.live);
  f.fail_free = false;
  assert(!ds4_weight_cache_release(&c) && !f.live);
  for (unsigned i = 0; i < 3; i++)
    assert(!ds4_weight_group_close(groups[i]));
  assert(!ds4_bundle_close(b));
  assert(!ds4_resources_close(r));
  puts("weight cache: PASS mixed/resident traffic, 50 visits, exact bytes and "
       "failed-free recovery");
}
