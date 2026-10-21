#include "ds4_executor.h"
#include <stdlib.h>
#include <string.h>

static bool sum(uint64_t *a, uint64_t b) {
    if (b > UINT64_MAX - *a)
        return false;
    *a += b;
    return true;
}
static bool demand_add(ds4_demand *a, const ds4_demand *b) {
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++)
        if (!sum(&a->bytes[i], b->bytes[i]))
            return false;
    return true;
}
static void demand_max(ds4_demand *a, const ds4_demand *b) {
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++)
        if (b->bytes[i] > a->bytes[i])
            a->bytes[i] = b->bytes[i];
}
static uint64_t group_mask(size_t n) { return n == 64 ? UINT64_MAX : (UINT64_C(1) << n) - 1; }

static bool valid(const ds4_execution_request *r) {
    if (!r || !r->variants || !r->variant_count || r->group_count > DS4_EXECUTOR_MAX_GROUPS ||
        r->victim_count > DS4_EXECUTOR_MAX_VICTIMS || (r->group_count && !r->groups) ||
        (r->victim_count && !r->victims) || (r->resident_groups & ~group_mask(r->group_count)))
        return false;
    for (size_t i = 0; i < r->group_count; i++)
        if (!r->groups[i].load || !r->groups[i].release)
            return false;
    for (size_t i = 0; i < r->victim_count; i++) {
        const ds4_execution_victim *v = &r->victims[i];
        if (!v->release || (unsigned)v->kind > DS4_RECLAIM_WEIGHTS ||
            (v->kind != DS4_RECLAIM_CACHE && (!v->restore || !v->quiesce)) ||
            (v->save && !v->discard) || (v->kind == DS4_RECLAIM_SESSION && !v->save))
            return false;
    }
    for (size_t i = 0; i < r->variant_count; i++) {
        const ds4_execution_variant *v = &r->variants[i];
        if (!v->steps || !v->step_count)
            return false;
        uint64_t live = r->resident_groups;
        for (size_t j = 0; j < v->step_count; j++) {
            const ds4_execution_step *s = &v->steps[j];
            if ((s->groups | s->retain) & ~group_mask(r->group_count))
                return false;
            live |= s->groups;
            if (s->retain & ~live)
                return false;
            live = s->retain | r->resident_groups;
        }
        if ((v->keep & ~group_mask(r->group_count)) || (v->keep & ~live))
            return false;
    }
    return true;
}

static bool phase_peak(const ds4_execution_request *r, size_t variant, ds4_execution_plan *p) {
    uint64_t live = r->resident_groups;
    p->cost = r->variants[variant].cost;
    const ds4_execution_variant *v = &r->variants[variant];
    for (size_t i = 0; i < v->step_count; i++) {
        const ds4_execution_step *s = &v->steps[i];
        uint64_t added = s->groups & ~live;
        live |= s->groups;
        ds4_demand now = s->working;
        for (size_t g = 0; g < r->group_count; g++) {
            uint64_t bit = UINT64_C(1) << g;
            if ((live & ~r->resident_groups) & bit)
                if (!demand_add(&now, &r->groups[g].memory))
                    return false;
            if (added & bit)
                if (!sum(&p->cost, r->groups[g].load_cost))
                    return false;
        }
        demand_max(&p->peak, &now);
        live = s->retain | r->resident_groups;
    }
    return true;
}

ds4_status ds4_execution_plan_build(ds4_resources *resources, const ds4_execution_request *r,
                                    ds4_execution_plan *out) {
    if (!resources || !out || !valid(r))
        return DS4_STATUS_INVALID;
    bool found = false, previous_valid = false;
    ds4_execution_plan previous = {0};
    for (size_t variant = 0; variant < r->variant_count; variant++) {
        ds4_execution_plan base = {.variant = variant};
        if (!phase_peak(r, variant, &base))
            return DS4_STATUS_INVALID;
        for (uint32_t mask = 0; mask < (UINT32_C(1) << r->victim_count); mask++) {
            ds4_execution_plan p = base;
            p.victims = mask;
            ds4_demand restore = {0}, reclaimed = {0};
            bool overflow = false;
            for (size_t i = 0; i < r->victim_count; i++) {
                if (!(mask & (UINT32_C(1) << i)))
                    continue;
                const ds4_execution_victim *victim = &r->victims[i];
                overflow |= !demand_add(&reclaimed, &victim->reclaimed) ||
                            !demand_add(&p.checkpoint, &victim->checkpoint) ||
                            !sum(&p.cost, victim->cost);
            }
            for (size_t i = r->victim_count; i-- > 0;) {
                if (!(mask & (UINT32_C(1) << i)))
                    continue;
                const ds4_execution_victim *victim = &r->victims[i];
                if (victim->restore) {
                    overflow |= !demand_add(&restore, &victim->reclaimed);
                    ds4_demand temporary = restore;
                    overflow |= !demand_add(&temporary, &victim->restore_extra);
                    demand_max(&p.peak, &temporary);
                }
            }
            ds4_demand total = p.peak;
            overflow |= !demand_add(&total, &p.checkpoint);
            if (overflow)
                continue;
            ds4_status admission = ds4_resources_can_reserve(resources, p.checkpoint.bytes, NULL);
            if (!admission)
                admission = ds4_resources_can_reserve(resources, total.bytes, reclaimed.bytes);
            if (admission == DS4_STATUS_BUSY)
                return admission;
            if (admission)
                continue;
            /* Eviction-free execution always wins over an eviction plan. */
            if (!found || (!mask && out->victims) ||
                (((out->victims == 0) == (mask == 0)) && p.cost < out->cost)) {
                *out = p;
                found = true;
            }
            if (mask == r->previous_victims && (!previous_valid || p.cost < previous.cost)) {
                previous = p;
                previous_valid = true;
            }
        }
    }
    if (!found)
        return DS4_STATUS_NOMEM;
    if (out->victims && previous_valid && previous.cost >= out->cost &&
        previous.cost - out->cost <= r->hysteresis_cost)
        *out = previous;
    return DS4_STATUS_OK;
}

struct ds4_execution {
    ds4_runtime *runtime;
    ds4_resources *resources;
    uint64_t owner, loaded;
    const ds4_execution_request *request;
    ds4_execution_ops ops;
    ds4_execution_plan plan;
    ds4_execution_state state;
    ds4_status result;
    bool device_lost;
    uint32_t saved, released;
    ds4_reservation *reservation;
};

static ds4_status note(ds4_execution *e, ds4_execution_event event, size_t index,
                       ds4_status status) {
    if (e->ops.journal)
        e->ops.journal(e->ops.context, event, index, status);
    return status;
}
static bool cancelled(ds4_execution *e) {
    return e->ops.cancelled && e->ops.cancelled(e->ops.context);
}
static ds4_status callback(ds4_status (*fn)(void *), void *context) {
    return fn ? fn(context) : DS4_STATUS_OK;
}

ds4_status ds4_execution_create(ds4_execution **out, ds4_runtime *runtime, ds4_resources *resources,
                                uint64_t owner, const ds4_execution_request *request,
                                const ds4_execution_ops *ops) {
    if (!out)
        return DS4_STATUS_INVALID;
    *out = NULL;
    if (!runtime || !resources || !owner || !valid(request) || !ops || !ops->execute ||
        !ops->drain || !ops->commit || !ops->abort)
        return DS4_STATUS_INVALID;
    ds4_execution *e = calloc(1, sizeof(*e));
    if (!e)
        return DS4_STATUS_NOMEM;
    e->runtime = runtime;
    e->resources = resources;
    e->owner = owner;
    e->request = request;
    e->ops = *ops;
    ds4_runtime_retain(runtime);
    *out = e;
    return DS4_STATUS_OK;
}

static ds4_status release_groups(ds4_execution *e, uint64_t retain) {
    for (size_t i = e->request->group_count; i-- > 0;) {
        uint64_t bit = UINT64_C(1) << i;
        if (!(e->loaded & bit) || (retain & bit))
            continue;
        const ds4_execution_group *g = &e->request->groups[i];
        ds4_status status = note(e, DS4_EXEC_RELEASE, i, g->release(g->context));
        if (status)
            return status;
        e->loaded &= ~bit;
    }
    return DS4_STATUS_OK;
}

/* Restoration never re-executes compute. Publication belongs to finish, only
 * after every restore and cleanup succeeds and cancellation is checked again.
 */
static ds4_status restore(ds4_execution *e) {
    ds4_status status = e->ops.drain(e->ops.context);
    uint64_t keep = !e->result && !e->plan.victims && !cancelled(e)
                        ? e->request->variants[e->plan.variant].keep
                        : 0;
    if (!status)
        status = release_groups(e, keep);
    if (status)
        return status;
    for (size_t i = e->request->victim_count; i-- > 0;) {
        uint32_t bit = UINT32_C(1) << i;
        const ds4_execution_victim *v = &e->request->victims[i];
        if (e->released & bit) {
            status = note(e, DS4_EXEC_RESTORE, i, callback(v->restore, v->context));
            if (status)
                return status;
            e->released &= ~bit;
        }
        if (e->saved & bit) {
            v->discard(v->context);
            e->saved &= ~bit;
        }
    }
    return DS4_STATUS_OK;
}

static ds4_status cleanup_failed(ds4_execution *e, ds4_status status) {
    e->device_lost = status == DS4_STATUS_DEVICE_LOST;
    if (e->device_lost)
        e->result = DS4_STATUS_DEVICE_LOST;
    e->state = e->device_lost ? DS4_EXECUTION_UNAVAILABLE : DS4_EXECUTION_RECOVERABLE;
    return e->device_lost ? DS4_STATUS_DEVICE_LOST : DS4_STATUS_RESTORE;
}
static ds4_status finish(ds4_execution *e) {
    if (e->device_lost) {
        e->state = DS4_EXECUTION_UNAVAILABLE;
        return DS4_STATUS_DEVICE_LOST;
    }
    ds4_reservation_scope scope = ds4_reservation_enter(e->reservation);
    ds4_status status = restore(e);
    ds4_reservation_leave(&scope);
    if (status)
        return cleanup_failed(e, status);
    status = ds4_reservation_cancel(e->reservation);
    if (status) {
        e->state = DS4_EXECUTION_RECOVERABLE;
        return DS4_STATUS_RESTORE;
    }
    e->reservation = NULL;
    if (!e->result && cancelled(e))
        e->result = DS4_STATUS_CANCELLED;
    if (!e->result) {
        e->result = note(e, DS4_EXEC_COMMIT, 0, e->ops.commit(e->ops.context));
        if (e->result == DS4_STATUS_DEVICE_LOST) {
            e->device_lost = true;
            e->state = DS4_EXECUTION_UNAVAILABLE;
            return DS4_STATUS_DEVICE_LOST;
        }
    }
    if (e->result) {
        /* A final cancellation/commit failure cannot publish or retain newly
         * loaded cache groups. Keep recovery state if release is obstructed. */
        status = release_groups(e, 0);
        if (status)
            return cleanup_failed(e, status);
        e->ops.abort(e->ops.context);
    }
    e->state = e->result ? DS4_EXECUTION_FAILED : DS4_EXECUTION_COMPLETE;
    return e->result;
}

static ds4_status save_victims(ds4_execution *e) {
    ds4_status status = note(e, DS4_EXEC_QUIESCE, SIZE_MAX, e->ops.drain(e->ops.context));
    for (size_t i = 0; i < e->request->victim_count && !status; i++) {
        uint32_t bit = UINT32_C(1) << i;
        if (!(e->plan.victims & bit))
            continue;
        if (cancelled(e)) {
            status = DS4_STATUS_CANCELLED;
            break;
        }
        const ds4_execution_victim *v = &e->request->victims[i];
        status = note(e, DS4_EXEC_QUIESCE, i, callback(v->quiesce, v->context));
        if (!status && v->save) {
            e->saved |= bit;
            status = note(e, DS4_EXEC_SAVE, i, v->save(v->context));
        }
    }
    return status;
}

static ds4_status release_victims(ds4_execution *e) {
    ds4_status status = DS4_STATUS_OK;
    for (size_t i = 0; i < e->request->victim_count && !status; i++) {
        uint32_t bit = UINT32_C(1) << i;
        if (!(e->plan.victims & bit))
            continue;
        if (cancelled(e)) {
            status = DS4_STATUS_CANCELLED;
            break;
        }
        const ds4_execution_victim *v = &e->request->victims[i];
        /* Mark before the callback: a partial release must be recoverable. */
        e->released |= bit;
        status = note(e, DS4_EXEC_RELEASE, i, v->release(v->context));
    }
    return status;
}

static ds4_status execute_steps(ds4_execution *e) {
    ds4_status status = DS4_STATUS_OK;
    const ds4_execution_variant *variant = &e->request->variants[e->plan.variant];
    for (size_t step = 0; step < variant->step_count && !status; step++) {
        if (cancelled(e)) {
            status = DS4_STATUS_CANCELLED;
            break;
        }
        const ds4_execution_step *s = &variant->steps[step];
        for (size_t i = 0; i < e->request->group_count && !status; i++) {
            if (cancelled(e)) {
                status = DS4_STATUS_CANCELLED;
                break;
            }
            uint64_t bit = UINT64_C(1) << i;
            if (!(s->groups & bit) || ((e->loaded | e->request->resident_groups) & bit))
                continue;
            e->loaded |= bit; /* Release even a failed partial load. */
            const ds4_execution_group *g = &e->request->groups[i];
            status = note(e, DS4_EXEC_LOAD, i, g->load(g->context));
        }
        if (!status && cancelled(e))
            status = DS4_STATUS_CANCELLED;
        if (!status)
            status = note(e, DS4_EXEC_STEP, step, e->ops.execute(e->ops.context, s->id));
        if (!status)
            status = callback(e->ops.drain, e->ops.context);
        if (!status)
            status = release_groups(e, s->retain);
    }
    return status;
}

ds4_status ds4_execution_run(ds4_execution *e) {
    if (!e || e->state != DS4_EXECUTION_NEW)
        return DS4_STATUS_INVALID;
    ds4_runtime_transaction transaction = {0};
    ds4_status status = ds4_runtime_transaction_begin(e->runtime, &transaction);
    if (status)
        return status;
    ds4_resource_owner_scope owner __attribute__((cleanup(ds4_resource_owner_leave))) =
        ds4_resource_owner_enter(e->owner);
    e->state = DS4_EXECUTION_RUNNING;
    status = note(e, DS4_EXEC_VALIDATE, 0, callback(e->ops.validate, e->ops.context));
    if (!status)
        status = ds4_execution_plan_build(e->resources, e->request, &e->plan);
    if (!status && cancelled(e))
        status = DS4_STATUS_CANCELLED;
    if (!status)
        status = note(e, DS4_EXEC_RESERVE, 0,
                      ds4_resources_reserve(e->resources, e->owner, e->plan.checkpoint.bytes,
                                            &e->reservation));
    ds4_reservation_scope staging = ds4_reservation_enter(e->reservation);
    if (!status)
        status = save_victims(e);
    ds4_reservation_leave(&staging);
    if (e->reservation) {
        ds4_status release = ds4_reservation_cancel(e->reservation);
        if (!release)
            e->reservation = NULL;
        else if (!status)
            status = release;
    }
    if (!status)
        status = release_victims(e);
    if (!status)
        status = note(
            e, DS4_EXEC_RESERVE, 1,
            ds4_resources_reserve(e->resources, e->owner, e->plan.peak.bytes, &e->reservation));
    ds4_reservation_scope working = ds4_reservation_enter(e->reservation);
    if (!status)
        status = execute_steps(e);
    ds4_reservation_leave(&working);
    if (!status && cancelled(e))
        status = DS4_STATUS_CANCELLED;
    e->result = status;
    e->device_lost = status == DS4_STATUS_DEVICE_LOST;
    status = finish(e);
    ds4_runtime_transaction_end(&transaction);
    return status;
}

ds4_status ds4_execution_recover(ds4_execution *e) {
    if (!e || (e->state != DS4_EXECUTION_RECOVERABLE && e->state != DS4_EXECUTION_UNAVAILABLE))
        return DS4_STATUS_INVALID;
    ds4_runtime_transaction transaction = {0};
    ds4_status status = ds4_runtime_transaction_begin(e->runtime, &transaction);
    if (status)
        return status;
    ds4_resource_owner_scope owner __attribute__((cleanup(ds4_resource_owner_leave))) =
        ds4_resource_owner_enter(e->owner);
    if (e->device_lost) {
        /* Explicit cleanup after repair never turns a lost device operation
         * into a successful result, and never replays its compute. */
        status = callback(e->ops.validate, e->ops.context);
        if (status) {
            ds4_runtime_transaction_end(&transaction);
            return status;
        }
        e->device_lost = false;
    }
    status = finish(e);
    ds4_runtime_transaction_end(&transaction);
    return status;
}
ds4_execution_state ds4_execution_get_state(const ds4_execution *e) {
    return e ? e->state : DS4_EXECUTION_FAILED;
}
const ds4_execution_plan *ds4_execution_get_plan(const ds4_execution *e) {
    return e ? &e->plan : NULL;
}
ds4_status ds4_execution_recovery_query(const ds4_execution *e, ds4_execution_recovery *out) {
    if (!e || !out)
        return DS4_STATUS_INVALID;
    *out = (ds4_execution_recovery){e->state, e->result, e->saved, e->released, e->loaded};
    return DS4_STATUS_OK;
}
ds4_status ds4_execution_close(ds4_execution *e) {
    if (!e)
        return DS4_STATUS_OK;
    if (e->state == DS4_EXECUTION_RUNNING || e->state == DS4_EXECUTION_RECOVERABLE ||
        e->state == DS4_EXECUTION_UNAVAILABLE)
        return DS4_STATUS_BUSY;
    ds4_runtime_release(e->runtime);
    free(e);
    return DS4_STATUS_OK;
}
