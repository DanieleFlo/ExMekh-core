#ifndef DS4_TRANSFER_H
#define DS4_TRANSFER_H
#include "ds4_resources.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_transfer ds4_transfer;
/* Two bounded host slots. Only the persistent read worker touches mmap source
 * pages; accelerator callbacks run on the caller's serial execution thread.
 * wait(slot) fences its previous copy before a worker can overwrite that slot.
 * publish orders compute after submitted copies without a host-wide fence. */
typedef struct {
  void *context;
  ds4_status (*allocate)(void *, uint64_t, void **);
  void (*release)(void *, void *);
  ds4_status (*copy)(void *, unsigned slot, void *, const void *, uint64_t);
  ds4_status (*wait)(void *, unsigned slot);
  ds4_status (*publish)(void *);
} ds4_transfer_ops;
ds4_status ds4_transfer_create(ds4_transfer **, ds4_resources *, uint64_t owner,
                               uint64_t slot_bytes, ds4_memory_location,
                               const ds4_transfer_ops *);
ds4_status ds4_transfer_write(ds4_transfer *, void *, const void *, uint64_t);
ds4_status ds4_transfer_publish(ds4_transfer *);
/* Drain failure preserves the handle and all accounted storage for recovery. */
ds4_status ds4_transfer_close(ds4_transfer *);
#ifdef __cplusplus
}
#endif
#endif
