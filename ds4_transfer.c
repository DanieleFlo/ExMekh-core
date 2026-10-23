#include "ds4_transfer.h"
#include "ds4_read_queue.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  ds4_read_task task;
  void *host;
  const void *source;
  uint64_t bytes;
  unsigned index;
  struct ds4_transfer *transfer;
} transfer_slot;
struct ds4_transfer {
  ds4_transfer_ops ops;
  ds4_read_queue queue;
  transfer_slot slots[2];
  ds4_resource *account, *metadata;
  uint64_t capacity;
  FILE *trace;
  uint64_t owner;
};
/* The registry credits the budget after this destructor returns. Keeping
 * release here prevents a competing admission from reusing live host bytes. */
static ds4_status release_slots(void *opaque) {
  ds4_transfer *t = opaque;
  for (unsigned i = 0; i < 2; i++)
    if (t->slots[i].host) {
      t->ops.release(t->ops.context, t->slots[i].host);
      t->slots[i].host = NULL;
    }
  return DS4_STATUS_OK;
}
static void trace_event(ds4_transfer *t, const char *event, unsigned slot,
                        uint64_t bytes) {
  if (!t->trace)
    return;
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  fprintf(t->trace,
          "{\"time\":%.9f,\"owner\":%llu,\"event\":\"%s\",\"slot\":%u,"
          "\"bytes\":%llu}\n",
          now.tv_sec + now.tv_nsec * 1e-9, (unsigned long long)t->owner, event,
          slot, (unsigned long long)bytes);
}
static void read_slot(void *context, unsigned worker, void *payload) {
  (void)context;
  (void)worker;
  transfer_slot *s = payload;
  trace_event(s->transfer, "read_start", s->index, s->bytes);
  memcpy(s->host, s->source, s->bytes);
  trace_event(s->transfer, "read_done", s->index, s->bytes);
}
ds4_status ds4_transfer_create(ds4_transfer **out, ds4_resources *r,
                               uint64_t owner, uint64_t bytes,
                               ds4_memory_location location,
                               const ds4_transfer_ops *ops) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!r || !owner || !ops || !ops->allocate || !ops->release || !ops->copy ||
      !ops->wait || !ops->publish || !bytes || bytes > SIZE_MAX / 2 ||
      (location != DS4_MEMORY_PINNED && location != DS4_MEMORY_HOST))
    return DS4_STATUS_INVALID;
  ds4_transfer *t = calloc(1, sizeof(*t));
  if (!t)
    return DS4_STATUS_NOMEM;
  t->ops = *ops;
  t->capacity = bytes;
  t->owner = owner;
  for (unsigned i = 0; i < 2; i++) {
    t->slots[i].index = i;
    t->slots[i].transfer = t;
  }
  const char *trace = getenv("DS4_SUITE_TRANSFER_TRACE");
  if (trace)
    t->trace = fopen(trace, "a");
  ds4_status s = ds4_resource_register_host(r, owner, sizeof(*t), &t->metadata);
  ds4_resource_spec spec = {.owner = owner,
                            .location = location,
                            .device = -1,
                            .requested = bytes * 2,
                            .allocated = bytes * 2,
                            .live = bytes * 2,
                            .destroy = release_slots,
                            .opaque = t};
  /* DMA is fenced by wait(), including constructor recovery. Borrow the
   * operation's budget without inheriting an unrelated compute event. */
  ds4_reservation *lease = NULL;
  uint64_t demand[DS4_MEMORY_COUNT] = {0};
  demand[location] = bytes * 2;
  if (!s)
    s = ds4_resources_reserve(r, owner, demand, &lease);
  if (!s)
    s = ds4_reservation_commit(lease, &spec, &t->account);
  ds4_reservation_cancel(lease);
  for (unsigned i = 0; i < 2 && !s; i++)
    s = ops->allocate(ops->context, bytes, &t->slots[i].host);
  if (!s && !ds4_read_queue_start(&t->queue, 1, read_slot, t))
    s = DS4_STATUS_NOMEM;
  if (!s && ds4_read_queue_workers(&t->queue) != 1)
    abort();
  if (s) {
    if (t->account)
      ds4_resource_release(t->account);
    if (t->metadata)
      ds4_resource_release(t->metadata);
    if (t->trace)
      fclose(t->trace);
    free(t);
    return s;
  }
  *out = t;
  return DS4_STATUS_OK;
}
static ds4_status prepare(ds4_transfer *t, unsigned slot, const void *src,
                          uint64_t n) {
  ds4_status s = t->ops.wait(t->ops.context, slot);
  if (s)
    return s;
  trace_event(t, "slot_ready", slot, n);
  transfer_slot *p = &t->slots[slot];
  p->source = src;
  p->bytes = n;
  return ds4_read_queue_submit(&t->queue, &p->task, p) ? DS4_STATUS_OK
                                                       : DS4_STATUS_BUSY;
}
ds4_status ds4_transfer_write(ds4_transfer *t, void *dst, const void *src,
                              uint64_t n) {
  if (!t || !dst || !src || !n || n > SIZE_MAX ||
      (uintptr_t)src > UINTPTR_MAX - n || (uintptr_t)dst > UINTPTR_MAX - n)
    return DS4_STATUS_INVALID;
  uint64_t at = 0, chunk = n < t->capacity ? n : t->capacity;
  unsigned current = 0;
  ds4_status s = prepare(t, current, src, chunk);
  if (s)
    return s;
  while (at < n) {
    transfer_slot *p = &t->slots[current];
    ds4_read_queue_wait(&t->queue, &p->task);
    uint64_t next = at + chunk, size = n - next;
    if (size > t->capacity)
      size = t->capacity;
    /* Read the next page-cache slice while DMA consumes this one. */
    if (size)
      s = prepare(t, current ^ 1, (const char *)src + next, size);
    bool queued = size && !s;
    if (!s)
      s = t->ops.copy(t->ops.context, current, (char *)dst + at, p->host,
                      chunk);
    if (!s)
      trace_event(t, "copy_submit", current, chunk);
    if (s) {
      if (queued)
        ds4_read_queue_wait(&t->queue, &t->slots[current ^ 1].task);
      return s;
    }
    at = next;
    chunk = size;
    current ^= 1;
  }
  return DS4_STATUS_OK;
}
ds4_status ds4_transfer_publish(ds4_transfer *t) {
  return t ? t->ops.publish(t->ops.context) : DS4_STATUS_INVALID;
}
ds4_status ds4_transfer_close(ds4_transfer *t) {
  if (!t)
    return DS4_STATUS_OK;
  for (unsigned i = 0; i < 2; i++) {
    ds4_status s = t->ops.wait(t->ops.context, i);
    if (s)
      return s;
  }
  ds4_read_queue_stop(&t->queue);
  ds4_status s = ds4_resource_release(t->account);
  if (s)
    return s;
  t->account = NULL;
  s = ds4_resource_release(t->metadata);
  if (s)
    return s;
  if (t->trace)
    fclose(t->trace);
  free(t);
  return DS4_STATUS_OK;
}
