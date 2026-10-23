#ifndef DS4_RESOURCES_H
#define DS4_RESOURCES_H

#include "ds4_runtime.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* These are separate ledgers: mmap storage is not RSS, and pinned pages are
 * not also charged as anonymous pages. Global free memory already excludes
 * owned allocations; it must never have owned bytes subtracted a second time. */
typedef enum {
    DS4_MEMORY_DEVICE, DS4_MEMORY_MMAP, DS4_MEMORY_RESIDENT,
    DS4_MEMORY_HOST, DS4_MEMORY_PINNED, DS4_MEMORY_CHECKPOINT,
    /* Virtual managed capacity can migrate; do not call it physical VRAM/RSS. */
    DS4_MEMORY_MANAGED,
    DS4_MEMORY_COUNT
} ds4_memory_location;
typedef struct ds4_resources ds4_resources;
typedef struct ds4_resource ds4_resource;
typedef struct ds4_reservation ds4_reservation;
enum { DS4_RESOURCE_OWNER_RUNTIME = 1, DS4_RESOURCE_OWNER_LEGACY_WEIGHTS = 2 };
/* Thread-local owner for legacy backend allocation sites. Scoped C callers
 * restore the previous owner on every return; standalone backends use owner 1. */
typedef struct { uint64_t previous; } ds4_resource_owner_scope;
ds4_resource_owner_scope ds4_resource_owner_enter(uint64_t owner);
void ds4_resource_owner_leave(ds4_resource_owner_scope *);
uint64_t ds4_resource_current_owner(void);
typedef ds4_status (*ds4_resource_destructor)(void *);
typedef struct {
    uint64_t owner, requested, allocated, live;
    ds4_memory_location location;
    uint32_t format;
    int device;
    bool authoritative, rebuildable;
    ds4_resource_destructor destroy;
    void *opaque;
} ds4_resource_spec;
typedef struct {
    uint64_t requested[DS4_MEMORY_COUNT], allocated[DS4_MEMORY_COUNT];
    uint64_t live[DS4_MEMORY_COUNT], reserved[DS4_MEMORY_COUNT];
    size_t roots, views;
} ds4_resource_totals;
typedef struct {
    uint64_t limit, margin, global_free;
    bool global_known;
} ds4_resource_budget;
typedef struct {
    ds4_resource_spec spec;
    size_t references;
    uint64_t last_event, offset;
    bool view;
} ds4_resource_info;

ds4_status ds4_resources_create(ds4_resources **out);
/* Refuses destruction while any resources or reservations remain. */
ds4_status ds4_resources_close(ds4_resources *resources);
ds4_status ds4_resources_set_budget(ds4_resources *, ds4_memory_location,
                                     const ds4_resource_budget *);
ds4_status ds4_resources_get_budget(ds4_resources *, ds4_memory_location, ds4_resource_budget *);
ds4_status ds4_resources_available(ds4_resources *, ds4_memory_location, uint64_t *);
/* Capacity visible to a nested adapter: the active operation lease, or
 * globally available capacity when no lease is active on this thread. */
ds4_status ds4_resources_working_capacity(ds4_resources *, ds4_memory_location, uint64_t *);
/* Atomic, read-only admission using the registry's own budget policy. Reclaimed
 * bytes are a projection, not an eviction; NULL means no reclamation. */
ds4_status ds4_resources_can_reserve(ds4_resources *, const uint64_t bytes[DS4_MEMORY_COUNT],
                                    const uint64_t reclaimed[DS4_MEMORY_COUNT]);
ds4_status ds4_resources_reconcile(ds4_resources *, ds4_memory_location, uint64_t global_free, bool known);
ds4_status ds4_resources_transfer_owner(ds4_resources *, uint64_t from, uint64_t to);
ds4_status ds4_resource_register(ds4_resources *, const ds4_resource_spec *, ds4_resource **);
/* CPU-only metadata/storage, consumed synchronously by its owner. Borrows the
 * current reservation's budget without inheriting an unrelated device event.
 * Async staging/pinned storage must use register/use and its actual event. */
ds4_status ds4_resource_register_host(ds4_resources *, uint64_t owner,
                                      uint64_t bytes, ds4_resource **);
/* Views retain storage and never charge its bytes again. */
ds4_status ds4_resource_view(ds4_resource *, uint64_t offset, uint64_t bytes, ds4_resource **);
ds4_status ds4_resource_retain(ds4_resource *);
ds4_status ds4_resource_release(ds4_resource *);
ds4_status ds4_resource_inspect(ds4_resource *, ds4_resource_info *);
ds4_status ds4_resource_set_live(ds4_resource *, uint64_t);
/* Events use one monotonically ordered, fenced timeline per registry. */
ds4_status ds4_resource_use(ds4_resource *, uint64_t event);
void ds4_resources_complete(ds4_resources *, uint64_t event);
void ds4_resources_totals(ds4_resources *, uint64_t owner, ds4_resource_totals *);
/* Only unaliased, idle resources with a destructor can actually be trimmed.
 * Accounting-only entries remain registered until their owner releases them. */
ds4_status ds4_resources_trim(ds4_resources *, uint64_t owner, uint64_t *released);
int ds4_resources_dump(ds4_resources *, FILE *);

/* Reserve all simultaneous working/staging/conversion copies in one call.
 * Commit transfers a reservation to live storage without a budget gap.
 * Cancellation must drain a marked in-flight copy before returning budget. */
ds4_status ds4_resources_reserve(ds4_resources *, uint64_t owner,
        const uint64_t bytes[DS4_MEMORY_COUNT], ds4_reservation **);
ds4_status ds4_reservation_use(ds4_reservation *, uint64_t event);
ds4_status ds4_reservation_commit(ds4_reservation *, const ds4_resource_spec *, ds4_resource **);
ds4_status ds4_reservation_cancel(ds4_reservation *);

/* Allocations made in this thread borrow from the operation's reservation,
 * retain their own owner, and return capacity to the lease on free. Cancel
 * detaches surviving allocations. Leave scopes in reverse order on their
 * entering thread. Any active scope, including on another thread, pins its
 * lease against cancellation. The caller owns the handle across scope entry. */
typedef struct { ds4_reservation *previous, *active; } ds4_reservation_scope;
ds4_reservation_scope ds4_reservation_enter(ds4_reservation *);
void ds4_reservation_leave(ds4_reservation_scope *);

#ifdef __cplusplus
}
#endif
#endif
