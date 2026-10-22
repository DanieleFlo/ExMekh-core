#ifndef DS4_HOST_MEMORY_H
#define DS4_HOST_MEMORY_H
/* Private synchronous host storage. The registry must outlive each buffer. */
#include "ds4_resources.h"
#include <stdlib.h>
typedef struct {
  void *data;
  ds4_resource *account;
} ds4_host_buffer;
static inline ds4_status ds4_host_alloc(ds4_resources *rs, uint64_t owner,
                                        uint64_t bytes, ds4_host_buffer *out) {
  *out = (ds4_host_buffer){0};
  if (!bytes)
    return DS4_STATUS_OK;
  if (bytes > SIZE_MAX)
    return DS4_STATUS_INVALID;
  ds4_status s = ds4_resource_register_host(rs, owner, bytes, &out->account);
  if (s)
    return s;
  out->data = calloc(1, (size_t)bytes);
  if (!out->data) {
    ds4_resource_release(out->account);
    *out = (ds4_host_buffer){0};
    return DS4_STATUS_NOMEM;
  }
  return DS4_STATUS_OK;
}
static inline void ds4_host_free(ds4_host_buffer *b) {
  free(b->data);
  if (b->account)
    ds4_resource_release(b->account);
  *b = (ds4_host_buffer){0};
}
#endif
