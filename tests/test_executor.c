#include "ds4_executor.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    ds4_resources *resources;
    ds4_resource *allocation;
    uint64_t owner, bytes;
    unsigned loads, releases, saves, restores, discarded, executed, committed, aborted;
    bool fail_load, fail_save, fail_release, fail_restore, fail_execute, cancel;
    bool fail_drain, device_lost, cancel_after_execute;
    bool *cancel_on_load, *cancel_on_save, *cancel_on_restore;
} fixture;
static ds4_status load(void *opaque) {
    fixture *f = opaque;
    f->loads++;
    if (f->allocation) return DS4_STATUS_OK;
    ds4_resource_spec s = {.owner=f->owner, .requested=f->bytes, .allocated=f->bytes,
        .live=f->bytes, .location=DS4_MEMORY_DEVICE};
    ds4_status status = ds4_resource_register(f->resources, &s, &f->allocation);
    if (f->cancel_on_load) *f->cancel_on_load = true;
    return status ? status : f->fail_load ? DS4_STATUS_NOMEM : DS4_STATUS_OK;
}
static ds4_status release(void *opaque) {
    fixture *f = opaque; f->releases++;
    if (f->fail_release) return DS4_STATUS_INFERENCE;
    ds4_status status = ds4_resource_release(f->allocation);
    if (!status) f->allocation = NULL;
    return status;
}
static ds4_status save(void *opaque) {
    fixture *f = opaque; f->saves++;
    if (f->cancel_on_save) *f->cancel_on_save = true;
    return f->fail_save ? DS4_STATUS_NOMEM : DS4_STATUS_OK;
}
static ds4_status restore(void *opaque) {
    fixture *f = opaque; f->restores++;
    if (f->cancel_on_restore) *f->cancel_on_restore = true;
    if (f->device_lost) return DS4_STATUS_DEVICE_LOST;
    return f->fail_restore ? DS4_STATUS_RESTORE : load(f);
}
static void discard(void *opaque) { ((fixture *)opaque)->discarded++; }
static ds4_status ready(void *opaque) { (void)opaque; return DS4_STATUS_OK; }
static ds4_status execute(void *opaque, uint32_t id) {
    fixture *f = opaque; f->executed++;
    assert(id == 17 || id == 18);
    if (f->device_lost) return DS4_STATUS_DEVICE_LOST;
    if (f->cancel_after_execute) f->cancel = true;
    return f->fail_execute ? DS4_STATUS_INFERENCE : DS4_STATUS_OK;
}
static ds4_status drain(void *opaque) {
    fixture *f=opaque;
    return f->fail_drain && f->executed ? DS4_STATUS_INFERENCE : DS4_STATUS_OK;
}
static ds4_status commit(void *opaque) { ((fixture *)opaque)->committed++; return DS4_STATUS_OK; }
static void abort_result(void *opaque) { ((fixture *)opaque)->aborted++; }
static bool cancelled(void *opaque) { return ((fixture *)opaque)->cancel; }

static void budgets(ds4_resources *r, uint64_t bytes) {
    ds4_resource_budget budget = {.limit=bytes};
    assert(!ds4_resources_set_budget(r, DS4_MEMORY_DEVICE, &budget));
}
int main(void) {
    ds4_resources *resources = NULL;
    ds4_runtime *runtime = NULL;
    assert(!ds4_resources_create(&resources) && !ds4_runtime_create(&runtime, NULL, NULL));
    fixture resident = {.resources=resources, .owner=10, .bytes=80};
    fixture group = {.resources=resources, .owner=20, .bytes=60};
    fixture operation = {0};
    assert(!load(&resident));
    ds4_execution_group groups[] = {{.memory.bytes={[DS4_MEMORY_DEVICE]=60},
        .load_cost=5, .context=&group, .load=load, .release=release}};
    ds4_execution_step steps[] = {{.id=17, .groups=1, .retain=1}, {.id=18, .groups=1}};
    ds4_execution_variant variants[] = {{steps, 2, 0, false}};
    ds4_execution_victim victims[] = {{.kind=DS4_RECLAIM_SESSION,
        .reclaimed.bytes={[DS4_MEMORY_DEVICE]=80}, .cost=10, .context=&resident,
        .quiesce=ready, .save=save, .release=release, .restore=restore, .discard=discard}};
    ds4_execution_request request = {.groups=groups, .group_count=1, .variants=variants,
        .variant_count=1, .victims=victims, .victim_count=1};
    ds4_execution_ops ops = {.context=&operation, .validate=ready, .execute=execute,
        .drain=drain, .commit=commit, .abort=abort_result, .cancelled=cancelled};
    ds4_execution_plan plan;
    victims[0].kind = (ds4_reclaim_kind)-1;
    assert(ds4_execution_plan_build(resources, &request, &plan) == DS4_STATUS_INVALID);
    victims[0].kind = DS4_RECLAIM_SESSION;
    budgets(resources, 160);
    assert(!ds4_execution_plan_build(resources, &request, &plan) && !plan.victims && plan.cost == 5);
    budgets(resources, 100);
    assert(!ds4_execution_plan_build(resources, &request, &plan) && plan.victims == 1);
    budgets(resources, 50);
    assert(ds4_execution_plan_build(resources, &request, &plan) == DS4_STATUS_NOMEM);
    budgets(resources, 100);
    enum {
        FAULT_NONE, FAULT_SAVE, FAULT_RELEASE, FAULT_LOAD, FAULT_COMPUTE,
        FAULT_RESTORE, FAULT_CANCEL_INITIAL, FAULT_DRAIN, FAULT_GROUP_RELEASE,
        FAULT_DEVICE_COMPUTE, FAULT_CANCEL_COMPUTE, FAULT_DEVICE_RESTORE,
        FAULT_CANCEL_LOAD, FAULT_CANCEL_SAVE, FAULT_CANCEL_RESTORE,
        FAULT_CANCEL_RECOVERY, FAULT_COUNT
    };
    for (unsigned fault = FAULT_NONE; fault < FAULT_COUNT; fault++) {
        operation = (fixture){0};
        resident.fail_save = fault == FAULT_SAVE;
        resident.fail_release = fault == FAULT_RELEASE;
        group.fail_load = fault == FAULT_LOAD;
        operation.fail_execute = fault == FAULT_COMPUTE;
        resident.fail_restore = fault == FAULT_RESTORE || fault == FAULT_CANCEL_RECOVERY;
        operation.cancel = fault == FAULT_CANCEL_INITIAL;
        operation.fail_drain = fault == FAULT_DRAIN;
        group.fail_release = fault == FAULT_GROUP_RELEASE;
        operation.device_lost = fault == FAULT_DEVICE_COMPUTE;
        operation.cancel_after_execute = fault == FAULT_CANCEL_COMPUTE;
        resident.device_lost = fault == FAULT_DEVICE_RESTORE;
        group.cancel_on_load = fault == FAULT_CANCEL_LOAD ? &operation.cancel : NULL;
        resident.cancel_on_save = fault == FAULT_CANCEL_SAVE ? &operation.cancel : NULL;
        resident.cancel_on_restore = fault == FAULT_CANCEL_RESTORE ? &operation.cancel : NULL;
        ds4_execution *job = NULL;
        assert(!ds4_execution_create(&job, runtime, resources, 30, &request, &ops));
        ds4_status status = ds4_execution_run(job);
        bool fatal = fault == FAULT_DEVICE_COMPUTE || fault == FAULT_DEVICE_RESTORE;
        bool recoverable = fault == FAULT_RESTORE || fault == FAULT_CANCEL_RECOVERY ||
            fault == FAULT_DRAIN || fault == FAULT_GROUP_RELEASE;
        if (recoverable || fatal) {
            assert(status == (fatal ? DS4_STATUS_DEVICE_LOST : DS4_STATUS_RESTORE));
            assert(ds4_execution_get_state(job) == (fatal ? DS4_EXECUTION_UNAVAILABLE : DS4_EXECUTION_RECOVERABLE));
            assert(!operation.committed && ds4_execution_close(job) == DS4_STATUS_BUSY);
            ds4_execution_recovery pending;
            assert(!ds4_execution_recovery_query(job,&pending) && pending.released_victims);
            if (fatal) assert(pending.operation_status == DS4_STATUS_DEVICE_LOST);
            unsigned executed = operation.executed;
            resident.fail_restore = false;
            resident.device_lost = false;
            operation.fail_drain = group.fail_release = false;
            if (fault == FAULT_CANCEL_RECOVERY) operation.cancel = true;
            ds4_status expected = DS4_STATUS_INFERENCE;
            if (fault == FAULT_RESTORE) expected = DS4_STATUS_OK;
            if (fault == FAULT_CANCEL_RECOVERY) expected = DS4_STATUS_CANCELLED;
            if (fatal) expected = DS4_STATUS_DEVICE_LOST;
            assert(ds4_execution_recover(job) == expected);
            assert(operation.executed == executed);
        } else assert((status == DS4_STATUS_OK) == (fault == FAULT_NONE));
        assert(resident.allocation && !group.allocation);
        assert(operation.committed == (fault == FAULT_NONE || fault == FAULT_RESTORE));
        assert(operation.aborted == (fault != FAULT_NONE && fault != FAULT_RESTORE));
        if (fault == FAULT_CANCEL_LOAD || fault == FAULT_CANCEL_SAVE) assert(!operation.executed);
        assert(!ds4_execution_close(job));
        resident.fail_release = false;
    }
    fixture cache={.resources=resources,.owner=11,.bytes=20}, weights={.resources=resources,.owner=12,.bytes=100};
    budgets(resources,1000); assert(!load(&cache) && !load(&weights));
    ds4_execution_victim choices[3]={
        {.kind=DS4_RECLAIM_CACHE,.reclaimed.bytes={[DS4_MEMORY_DEVICE]=20},.cost=1,.context=&cache,.release=release},
        victims[0],
        {.kind=DS4_RECLAIM_WEIGHTS,.reclaimed.bytes={[DS4_MEMORY_DEVICE]=100},.cost=3,.context=&weights,.release=release,.restore=restore,.quiesce=ready}};
    choices[1].cost=5;
    ds4_execution_request matrix=request; matrix.victims=choices; matrix.victim_count=3;
    budgets(resources,260); assert(!ds4_execution_plan_build(resources,&matrix,&plan) && plan.victims==0);
    budgets(resources,240); assert(!ds4_execution_plan_build(resources,&matrix,&plan) && plan.victims==1);
    budgets(resources,210); assert(!ds4_execution_plan_build(resources,&matrix,&plan) && plan.victims==4);
    matrix.previous_victims=2; matrix.hysteresis_cost=2;
    assert(!ds4_execution_plan_build(resources,&matrix,&plan) && plan.victims==2);
    matrix.previous_victims=0; matrix.hysteresis_cost=0;
    choices[2].cost=9;
    assert(!ds4_execution_plan_build(resources,&matrix,&plan) && plan.victims==2);
    ds4_execution_step large={.id=17,.groups=1,.working.bytes={[DS4_MEMORY_DEVICE]=140}};
    ds4_execution_variant alternatives[]={{&large,1,0,false},{steps,2,100,true}};
    matrix.variants=alternatives; matrix.variant_count=1;
    budgets(resources,240); assert(!ds4_execution_plan_build(resources,&matrix,&plan) && plan.victims==6);
    matrix.variant_count=2;
    budgets(resources,270); assert(!ds4_execution_plan_build(resources,&matrix,&plan) && !plan.victims && plan.variant==1);
    assert(!release(&cache) && !release(&weights)); budgets(resources,100);
    /* Scoped allocation and nested checkpoint reservations consume the same
     * peak rather than charging it twice; freeing returns reusable capacity. */
    assert(!release(&resident));
    uint64_t bytes[DS4_MEMORY_COUNT] = {[DS4_MEMORY_DEVICE]=100};
    ds4_reservation *lease, *child;
    assert(!ds4_resources_reserve(resources, 30, bytes, &lease));
    ds4_reservation_scope scope = ds4_reservation_enter(lease);
    bytes[0] = 60;
    assert(!ds4_resources_reserve(resources, 20, bytes, &child));
    assert(ds4_reservation_cancel(lease) == DS4_STATUS_BUSY);
    assert(!ds4_reservation_cancel(child));
    group.fail_load = false;
    assert(!load(&group) && !release(&group) && !load(&group));
    ds4_reservation_leave(&scope);
    assert(!ds4_reservation_cancel(lease) && !release(&group));
    ds4_resource_totals totals; ds4_resources_totals(resources, 0, &totals);
    assert(!totals.roots && !totals.reserved[0]);
    assert(!ds4_resources_close(resources)); ds4_runtime_release(runtime);
    puts("executor: PASS admission, repeated visits, cancellation, faults, recovery and reservation reuse");
    return 0;
}
