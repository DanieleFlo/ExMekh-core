#define _POSIX_C_SOURCE 200809L
#include "ds4_resources.h"
#include <inttypes.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static __thread uint64_t current_owner = DS4_RESOURCE_OWNER_RUNTIME;
ds4_resource_owner_scope ds4_resource_owner_enter(uint64_t owner) {
    ds4_resource_owner_scope scope = {current_owner};
    if (owner) current_owner = owner;
    return scope;
}
void ds4_resource_owner_leave(ds4_resource_owner_scope *scope) { current_owner = scope->previous; }
uint64_t ds4_resource_current_owner(void) { return current_owner; }

struct ds4_resources {
    pthread_mutex_t mutex;
    ds4_resource *head;
    ds4_reservation *leases;
    ds4_resource_budget budget[DS4_MEMORY_COUNT];
    ds4_resource_totals totals;
    uint64_t completed;
    size_t reservations;
    bool destroying;
};
struct ds4_resource {
    ds4_resources *registry;
    ds4_resource *next, *backing;
    ds4_resource_info info;
    ds4_reservation *lease;
};
struct ds4_reservation {
    ds4_reservation *next;
    ds4_reservation *parent;
    ds4_resources *registry;
    uint64_t owner, bytes[DS4_MEMORY_COUNT], event;
    size_t scopes;
};

static __thread ds4_reservation *current_reservation;
ds4_reservation_scope ds4_reservation_enter(ds4_reservation *lease) {
    ds4_reservation_scope scope = {current_reservation, lease};
    if (lease) {
        pthread_mutex_lock(&lease->registry->mutex);
        lease->scopes++;
        pthread_mutex_unlock(&lease->registry->mutex);
    }
    current_reservation = lease;
    return scope;
}
void ds4_reservation_leave(ds4_reservation_scope *scope) {
    if (scope->active) {
        ds4_reservation *lease = scope->active;
        pthread_mutex_lock(&lease->registry->mutex);
        lease->scopes--;
        pthread_mutex_unlock(&lease->registry->mutex);
        scope->active = NULL;
    }
    current_reservation = scope->previous;
}

static bool location_valid(ds4_memory_location l) { return (unsigned)l < DS4_MEMORY_COUNT; }
static uint64_t subtract(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }
static uint64_t add(uint64_t a, uint64_t b) { return b > UINT64_MAX - a ? UINT64_MAX : a + b; }
static uint64_t available_after(ds4_resources *r, ds4_memory_location l, const uint64_t *reclaimed) {
    const ds4_resource_budget *b = &r->budget[l];
    uint64_t credit = reclaimed ? reclaimed[l] : 0;
    uint64_t allocated = subtract(r->totals.allocated[l], credit), reserved = r->totals.reserved[l];
    if (l == DS4_MEMORY_HOST) {
        const ds4_memory_location shared[] = {DS4_MEMORY_PINNED, DS4_MEMORY_RESIDENT};
        for (size_t i = 0; i < 2; i++) {
            uint64_t part = reclaimed ? reclaimed[shared[i]] : 0;
            allocated = add(allocated, subtract(r->totals.allocated[shared[i]], part));
            reserved = add(reserved, r->totals.reserved[shared[i]]);
            credit = add(credit, part);
        }
    }
    uint64_t free = subtract(b->limit, allocated);
    uint64_t global = add(b->global_free, credit);
    if (b->global_known && global < free) free = global;
    free = subtract(subtract(free, reserved), b->margin);
    if (l == DS4_MEMORY_PINNED || l == DS4_MEMORY_RESIDENT) {
        uint64_t host = available_after(r, DS4_MEMORY_HOST, reclaimed);
        if (host < free) free = host;
    }
    return free;
}
static uint64_t available(ds4_resources *r, ds4_memory_location l) {
    return available_after(r, l, NULL);
}
static ds4_status can_reserve(ds4_resources *r, const uint64_t *bytes, const uint64_t *reclaimed) {
    if (r->destroying) return DS4_STATUS_BUSY;
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++)
        if (reclaimed && reclaimed[i] > r->totals.allocated[i]) return DS4_STATUS_INVALID;
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++)
        if (bytes[i] > available_after(r, i, reclaimed)) return DS4_STATUS_NOMEM;
    uint64_t host = bytes[DS4_MEMORY_HOST], pinned = bytes[DS4_MEMORY_PINNED], resident = bytes[DS4_MEMORY_RESIDENT];
    if (pinned > UINT64_MAX-host || resident > UINT64_MAX-host-pinned ||
        host+pinned+resident > available_after(r, DS4_MEMORY_HOST, reclaimed)) return DS4_STATUS_NOMEM;
    return DS4_STATUS_OK;
}
ds4_status ds4_resources_create(ds4_resources **out) {
    if (!out) return DS4_STATUS_INVALID;
    *out = calloc(1, sizeof(**out));
    if (!*out) return DS4_STATUS_NOMEM;
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr)) { free(*out); *out = NULL; return DS4_STATUS_NOT_READY; }
    int rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    if (!rc) rc = pthread_mutex_init(&(*out)->mutex, &attr);
    pthread_mutexattr_destroy(&attr);
    if (rc) { free(*out); *out = NULL; return DS4_STATUS_NOT_READY; }
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++) (*out)->budget[i].limit = UINT64_MAX;
    return DS4_STATUS_OK;
}
ds4_status ds4_resources_close(ds4_resources *r) {
    if (!r) return DS4_STATUS_OK;
    pthread_mutex_lock(&r->mutex);
    if (r->head || r->reservations || r->destroying) {
        pthread_mutex_unlock(&r->mutex); return DS4_STATUS_BUSY;
    }
    pthread_mutex_unlock(&r->mutex);
    pthread_mutex_destroy(&r->mutex); free(r); return DS4_STATUS_OK;
}
ds4_status ds4_resources_set_budget(ds4_resources *r, ds4_memory_location l,
                                     const ds4_resource_budget *b) {
    if (!r || !b || !location_valid(l)) return DS4_STATUS_INVALID;
    pthread_mutex_lock(&r->mutex); r->budget[l] = *b;
    pthread_mutex_unlock(&r->mutex); return DS4_STATUS_OK;
}
ds4_status ds4_resources_available(ds4_resources *r, ds4_memory_location l, uint64_t *out) {
    if (!r || !out || !location_valid(l)) return DS4_STATUS_INVALID;
    pthread_mutex_lock(&r->mutex); *out = available(r, l);
    pthread_mutex_unlock(&r->mutex); return DS4_STATUS_OK;
}
ds4_status ds4_resources_can_reserve(ds4_resources *r, const uint64_t bytes[DS4_MEMORY_COUNT],
                                    const uint64_t reclaimed[DS4_MEMORY_COUNT]) {
    if (!r || !bytes) return DS4_STATUS_INVALID;
    pthread_mutex_lock(&r->mutex);
    ds4_status status = can_reserve(r, bytes, reclaimed);
    pthread_mutex_unlock(&r->mutex);
    return status;
}
ds4_status ds4_resources_get_budget(ds4_resources *r, ds4_memory_location l, ds4_resource_budget *out) {
    if (!r || !out || !location_valid(l)) return DS4_STATUS_INVALID;
    pthread_mutex_lock(&r->mutex); *out = r->budget[l];
    pthread_mutex_unlock(&r->mutex); return DS4_STATUS_OK;
}
ds4_status ds4_resources_reconcile(ds4_resources *r, ds4_memory_location l, uint64_t bytes, bool known) {
    if (!r || !location_valid(l)) return DS4_STATUS_INVALID;
    pthread_mutex_lock(&r->mutex);
    r->budget[l].global_free = bytes; r->budget[l].global_known = known;
    pthread_mutex_unlock(&r->mutex); return DS4_STATUS_OK;
}
ds4_status ds4_resources_transfer_owner(ds4_resources *r, uint64_t from, uint64_t to) {
    if (!r || !from || !to) return DS4_STATUS_INVALID;
    pthread_mutex_lock(&r->mutex);
    if (r->destroying) { pthread_mutex_unlock(&r->mutex); return DS4_STATUS_BUSY; }
    for (ds4_resource *p = r->head; p; p = p->next)
        if (p->info.spec.owner == from) p->info.spec.owner = to;
    for (ds4_reservation *p = r->leases; p; p = p->next)
        if (p->owner == from) p->owner = to;
    pthread_mutex_unlock(&r->mutex); return DS4_STATUS_OK;
}
static bool spec_valid(const ds4_resource_spec *s) {
    return s && s->owner && location_valid(s->location) && s->live <= s->allocated &&
        s->requested <= s->allocated && !(s->authoritative && s->rebuildable);
}
static void insert(ds4_resources *r, ds4_resource *p, const ds4_resource_spec *s) {
    p->registry = r; p->info.spec = *s; p->info.references = 1;
    p->next = r->head; r->head = p;
    r->totals.roots++;
    r->totals.requested[s->location] += s->requested;
    r->totals.allocated[s->location] += s->allocated;
    r->totals.live[s->location] += s->live;
    /* Keep a global-free sample conservative until the next reconciliation. */
    if (r->budget[s->location].global_known)
        r->budget[s->location].global_free = subtract(r->budget[s->location].global_free, s->allocated);
    if ((s->location == DS4_MEMORY_PINNED || s->location == DS4_MEMORY_RESIDENT) && r->budget[DS4_MEMORY_HOST].global_known)
        r->budget[DS4_MEMORY_HOST].global_free = subtract(r->budget[DS4_MEMORY_HOST].global_free, s->allocated);
}
ds4_status ds4_resource_register(ds4_resources *r, const ds4_resource_spec *s, ds4_resource **out) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    if (!r || !spec_valid(s)) return DS4_STATUS_INVALID;
    ds4_resource *p = calloc(1, sizeof(*p));
    if (!p) return DS4_STATUS_NOMEM;
    pthread_mutex_lock(&r->mutex);
    ds4_reservation *lease = current_reservation;
    if (lease && lease->registry != r) lease = NULL;
    ds4_status status = r->destroying ? DS4_STATUS_BUSY :
        s->allocated > (lease ? lease->bytes[s->location] : available(r, s->location)) ?
        DS4_STATUS_NOMEM : DS4_STATUS_OK;
    if (!status) {
        if (lease) {
            lease->bytes[s->location] -= s->allocated;
            r->totals.reserved[s->location] -= s->allocated;
            p->lease = lease;
            p->info.last_event = lease->event;
        }
        insert(r, p, s); *out = p;
    }
    pthread_mutex_unlock(&r->mutex);
    if (status) free(p);
    return status;
}
ds4_status ds4_resource_view(ds4_resource *base, uint64_t offset, uint64_t bytes, ds4_resource **out) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    if (!base) return DS4_STATUS_INVALID;
    ds4_resources *r = base->registry;
    pthread_mutex_lock(&r->mutex);
    ds4_resource *root = base->backing ? base->backing : base;
    uint64_t extent = base->info.view ? base->info.spec.requested : base->info.spec.allocated;
    if (r->destroying || offset > extent || bytes > extent - offset ||
        root->info.references == SIZE_MAX) {
        pthread_mutex_unlock(&r->mutex); return DS4_STATUS_INVALID;
    }
    ds4_resource *p = calloc(1, sizeof(*p));
    if (!p) { pthread_mutex_unlock(&r->mutex); return DS4_STATUS_NOMEM; }
    p->registry = r; p->backing = root;
    p->info = base->info; p->info.view = true; p->info.references = 1;
    p->info.offset = base->info.offset + offset;
    p->info.spec.requested = bytes; p->info.spec.allocated = p->info.spec.live = 0;
    p->info.spec.destroy = NULL; p->info.spec.opaque = NULL;
    root->info.references++; p->next = r->head; r->head = p; r->totals.views++;
    *out = p; pthread_mutex_unlock(&r->mutex); return DS4_STATUS_OK;
}
ds4_status ds4_resource_retain(ds4_resource *p) {
    if (!p) return DS4_STATUS_INVALID;
    ds4_resources *r = p->registry;
    pthread_mutex_lock(&r->mutex);
    ds4_status status = r->destroying ? DS4_STATUS_BUSY :
        p->info.references == SIZE_MAX ? DS4_STATUS_INVALID : DS4_STATUS_OK;
    if (!status) p->info.references++;
    pthread_mutex_unlock(&r->mutex); return status;
}
static ds4_status release(ds4_resource *p) {
    ds4_resources *r = p->registry;
    ds4_resource *root = p->backing ? p->backing : p;
    if (r->destroying || root->info.last_event > r->completed) return DS4_STATUS_BUSY;
    if (p->info.references > 1) { p->info.references--; return DS4_STATUS_OK; }
    if (!p->backing && p->info.spec.destroy) {
        r->destroying = true;
        ds4_status status = p->info.spec.destroy(p->info.spec.opaque);
        r->destroying = false;
        if (status) return status;
    }
    /* An alias is not consumed when its backing destructor fails. */
    if (p->backing) {
        ds4_status status = release(root);
        if (status) return status;
    }
    ds4_resource **link = &r->head;
    while (*link != p) link = &(*link)->next;
    *link = p->next;
    if (p->backing) r->totals.views--;
    else {
        const ds4_resource_spec *s = &p->info.spec;
        r->totals.roots--;
        r->totals.requested[s->location] -= s->requested;
        r->totals.allocated[s->location] -= s->allocated;
        r->totals.live[s->location] -= s->live;
        if (p->lease) {
            p->lease->bytes[s->location] += s->allocated;
            r->totals.reserved[s->location] += s->allocated;
        }
        /* Do not credit global_free: backend destruction or an unrelated
         * process may change physical residency. Refresh its sample explicitly. */
    }
    free(p);
    return DS4_STATUS_OK;
}
ds4_status ds4_resource_release(ds4_resource *p) {
    if (!p) return DS4_STATUS_OK;
    ds4_resources *r = p->registry; pthread_mutex_lock(&r->mutex);
    ds4_status status = release(p);
    pthread_mutex_unlock(&r->mutex); return status;
}
ds4_status ds4_resource_inspect(ds4_resource *p, ds4_resource_info *out) {
    if (!p || !out) return DS4_STATUS_INVALID;
    ds4_resources *r = p->registry; pthread_mutex_lock(&r->mutex);
    *out = p->info;
    if (p->backing) out->last_event = p->backing->info.last_event;
    pthread_mutex_unlock(&r->mutex); return DS4_STATUS_OK;
}
ds4_status ds4_resource_set_live(ds4_resource *p, uint64_t bytes) {
    if (!p) return DS4_STATUS_INVALID;
    ds4_resources *r = p->registry; pthread_mutex_lock(&r->mutex);
    ds4_status status = r->destroying ? DS4_STATUS_BUSY :
        p->backing || bytes > p->info.spec.allocated ? DS4_STATUS_INVALID : DS4_STATUS_OK;
    if (!status) {
        r->totals.live[p->info.spec.location] -= p->info.spec.live;
        p->info.spec.live = bytes; r->totals.live[p->info.spec.location] += bytes;
    }
    pthread_mutex_unlock(&r->mutex); return status;
}
ds4_status ds4_resource_use(ds4_resource *p, uint64_t event) {
    if (!p) return DS4_STATUS_INVALID;
    ds4_resources *r = p->registry; pthread_mutex_lock(&r->mutex);
    if (p->backing) p = p->backing;
    if (event > p->info.last_event) p->info.last_event = event;
    pthread_mutex_unlock(&r->mutex); return DS4_STATUS_OK;
}
void ds4_resources_complete(ds4_resources *r, uint64_t event) {
    if (!r) return;
    pthread_mutex_lock(&r->mutex);
    if (event > r->completed) r->completed = event;
    pthread_mutex_unlock(&r->mutex);
}
void ds4_resources_totals(ds4_resources *r, uint64_t owner, ds4_resource_totals *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!r) return;
    pthread_mutex_lock(&r->mutex);
    if (!owner) *out = r->totals;
    else {
        for (ds4_reservation *p = r->leases; p; p = p->next)
            if (p->owner == owner)
                for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++) out->reserved[i] += p->bytes[i];
        for (ds4_resource *p = r->head; p; p = p->next) {
            const ds4_resource_spec *s = &p->info.spec;
            if (s->owner != owner) continue;
            if (p->backing) { out->views++; continue; }
            out->roots++; out->requested[s->location] += s->requested;
            out->allocated[s->location] += s->allocated; out->live[s->location] += s->live;
        }
    }
    pthread_mutex_unlock(&r->mutex);
}
ds4_status ds4_resources_trim(ds4_resources *r, uint64_t owner, uint64_t *released) {
    if (!r || !owner || !released) return DS4_STATUS_INVALID;
    *released = 0; pthread_mutex_lock(&r->mutex);
    ds4_status status = DS4_STATUS_OK;
    for (ds4_resource *p = r->head, *next; p; p = next) {
        next = p->next;
        if (p->backing || p->info.spec.owner != owner || !p->info.spec.rebuildable ||
            !p->info.spec.destroy ||
            p->info.references != 1 || p->info.last_event > r->completed) continue;
        uint64_t bytes = p->info.spec.allocated;
        status = release(p);
        if (status) break;
        *released = bytes > UINT64_MAX - *released ? UINT64_MAX : *released + bytes;
    }
    pthread_mutex_unlock(&r->mutex); return status;
}
int ds4_resources_dump(ds4_resources *r, FILE *fp) {
    if (!r || !fp) return -1;
    pthread_mutex_lock(&r->mutex);
    int rc = 0;
    for (unsigned i = 0; i < DS4_MEMORY_COUNT && !rc; i++) {
        const ds4_resource_budget *b = &r->budget[i];
        rc = fprintf(fp, "B %u %" PRIu64 " %" PRIu64 " %" PRIu64 " %u %" PRIu64 " %" PRIu64 "\n",
            i, b->limit, b->margin, b->global_free, b->global_known,
            r->totals.allocated[i], r->totals.reserved[i]) < 0 ? -1 : 0;
    }
    for (ds4_resource *p = r->head; p && !rc; p = p->next) {
        const ds4_resource_spec *s = &p->info.spec;
        rc = fprintf(fp, "R %" PRIu64 " %u %d %u %" PRIu64 " %" PRIu64 " %" PRIu64
            " %zu %" PRIu64 " %u %u %u\n", s->owner, s->location, s->device, s->format,
            s->requested, s->allocated, s->live, p->info.references,
            (p->backing ? p->backing : p)->info.last_event,
            s->authoritative, s->rebuildable, p->info.view) < 0 ? -1 : 0;
    }
    pthread_mutex_unlock(&r->mutex); return rc;
}
ds4_status ds4_resources_reserve(ds4_resources *r, uint64_t owner,
        const uint64_t bytes[DS4_MEMORY_COUNT], ds4_reservation **out) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    if (!r || !owner || !bytes) return DS4_STATUS_INVALID;
    ds4_reservation *p = calloc(1, sizeof(*p));
    if (!p) return DS4_STATUS_NOMEM;
    pthread_mutex_lock(&r->mutex);
    ds4_reservation *parent = current_reservation;
    if (parent && parent->registry != r) parent = NULL;
    ds4_status status = r->destroying ? DS4_STATUS_BUSY : DS4_STATUS_OK;
    if (!status && !parent) status = can_reserve(r, bytes, NULL);
    for (unsigned i = 0; parent && i < DS4_MEMORY_COUNT && !status; i++)
        if (bytes[i] > parent->bytes[i]) status = DS4_STATUS_NOMEM;
    if (!status) {
        p->registry = r; p->owner = owner; memcpy(p->bytes, bytes, sizeof(p->bytes));
        p->parent = parent;
        p->next = r->leases; r->leases = p;
        for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++) {
            if (parent) parent->bytes[i] -= bytes[i];
            else r->totals.reserved[i] += bytes[i];
        }
        r->reservations++; *out = p;
    }
    pthread_mutex_unlock(&r->mutex); if (status) free(p); return status;
}
ds4_status ds4_reservation_use(ds4_reservation *p, uint64_t event) {
    if (!p) return DS4_STATUS_INVALID;
    pthread_mutex_lock(&p->registry->mutex);
    if (event > p->event) p->event = event;
    pthread_mutex_unlock(&p->registry->mutex); return DS4_STATUS_OK;
}
ds4_status ds4_reservation_commit(ds4_reservation *p, const ds4_resource_spec *s, ds4_resource **out) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    if (!p || !spec_valid(s) || s->owner != p->owner) return DS4_STATUS_INVALID;
    ds4_resources *r = p->registry;
    ds4_resource *resource = calloc(1, sizeof(*resource));
    if (!resource) return DS4_STATUS_NOMEM;
    pthread_mutex_lock(&r->mutex);
    ds4_status status = r->destroying ? DS4_STATUS_BUSY :
        s->allocated > p->bytes[s->location] ? DS4_STATUS_NOMEM : DS4_STATUS_OK;
    if (!status) {
        p->bytes[s->location] -= s->allocated;
        r->totals.reserved[s->location] -= s->allocated;
        insert(r, resource, s); resource->info.last_event = p->event;
        resource->lease = p->parent; *out = resource;
    }
    pthread_mutex_unlock(&r->mutex); if (status) free(resource); return status;
}
ds4_status ds4_reservation_cancel(ds4_reservation *p) {
    if (!p) return DS4_STATUS_OK;
    ds4_resources *r = p->registry; pthread_mutex_lock(&r->mutex);
    bool children = false;
    for (ds4_reservation *child = r->leases; child; child = child->next)
        if (child->parent == p) children = true;
    if (r->destroying || p->event > r->completed || children || p->scopes) {
        pthread_mutex_unlock(&r->mutex); return DS4_STATUS_BUSY;
    }
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++) {
        if (p->parent) p->parent->bytes[i] += p->bytes[i];
        else r->totals.reserved[i] -= p->bytes[i];
    }
    for (ds4_resource *resource = r->head; resource; resource = resource->next)
        if (resource->lease == p) resource->lease = p->parent;
    ds4_reservation **link = &r->leases;
    while (*link != p) link = &(*link)->next;
    *link = p->next;
    r->reservations--; pthread_mutex_unlock(&r->mutex); free(p); return DS4_STATUS_OK;
}
