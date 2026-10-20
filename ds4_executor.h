#ifndef DS4_EXECUTOR_H
#define DS4_EXECUTOR_H
#include "ds4_resources.h"
#ifdef __cplusplus
extern "C" {
#endif

#define DS4_EXECUTOR_MAX_VICTIMS 16u
#define DS4_EXECUTOR_MAX_GROUPS 64u
typedef struct { uint64_t bytes[DS4_MEMORY_COUNT]; } ds4_demand;

typedef struct {
    ds4_demand memory;
    uint64_t load_cost;
    void *context;
    ds4_status (*load)(void *);
    ds4_status (*release)(void *);
} ds4_execution_group;

typedef struct {
    uint32_t id;
    uint64_t groups, retain; /* Required now; retained after this visit. */
    ds4_demand working;
} ds4_execution_step;

typedef struct {
    const ds4_execution_step *steps;
    size_t step_count;
    uint64_t cost;
    bool streamed;
} ds4_execution_variant;

typedef enum { DS4_RECLAIM_CACHE, DS4_RECLAIM_SESSION, DS4_RECLAIM_WEIGHTS } ds4_reclaim_kind;
typedef struct {
    ds4_reclaim_kind kind;
    ds4_demand reclaimed, checkpoint, restore_extra;
    uint64_t cost;
    void *context;
    ds4_status (*quiesce)(void *);
    ds4_status (*save)(void *);
    ds4_status (*release)(void *);
    ds4_status (*restore)(void *);
    void (*discard)(void *); /* Only after validated restoration. */
} ds4_execution_victim;

typedef struct {
    const ds4_execution_group *groups;
    size_t group_count;
    uint64_t resident_groups;
    const ds4_execution_variant *variants;
    size_t variant_count;
    const ds4_execution_victim *victims;
    size_t victim_count;
    uint32_t previous_victims;
    uint64_t hysteresis_cost;
} ds4_execution_request;

typedef struct {
    size_t variant;
    uint32_t victims;
    ds4_demand peak, checkpoint;
    uint64_t cost;
} ds4_execution_plan;

ds4_status ds4_execution_plan_build(ds4_resources *, const ds4_execution_request *, ds4_execution_plan *);

typedef enum {
    DS4_EXECUTION_NEW, DS4_EXECUTION_RUNNING, DS4_EXECUTION_RECOVERABLE,
    DS4_EXECUTION_COMPLETE, DS4_EXECUTION_FAILED, DS4_EXECUTION_UNAVAILABLE
} ds4_execution_state;
typedef enum {
    DS4_EXEC_VALIDATE, DS4_EXEC_RESERVE, DS4_EXEC_QUIESCE, DS4_EXEC_SAVE,
    DS4_EXEC_RELEASE, DS4_EXEC_LOAD, DS4_EXEC_STEP, DS4_EXEC_RESTORE, DS4_EXEC_COMMIT
} ds4_execution_event;
typedef struct {
    void *context;
    ds4_status (*validate)(void *);
    ds4_status (*execute)(void *, uint32_t step);
    ds4_status (*drain)(void *);
    ds4_status (*commit)(void *);
    void (*abort)(void *);
    bool (*cancelled)(void *);
    void (*journal)(void *, ds4_execution_event, size_t, ds4_status);
} ds4_execution_ops;
typedef struct ds4_execution ds4_execution;
/* Request descriptors remain immutable and callback contexts alive until close. Compute
 * is never retried. Recover retries only drain/release/restore, ignoring cancel.
 * DEVICE_LOST retains resources and checkpoints as UNAVAILABLE. Explicit
 * recovery validates the repaired backend and still fails the original result. */
ds4_status ds4_execution_create(ds4_execution **, ds4_runtime *, ds4_resources *, uint64_t owner,
                              const ds4_execution_request *, const ds4_execution_ops *);
ds4_status ds4_execution_run(ds4_execution *);
ds4_status ds4_execution_recover(ds4_execution *);
ds4_execution_state ds4_execution_get_state(const ds4_execution *);
const ds4_execution_plan *ds4_execution_get_plan(const ds4_execution *);
typedef struct {
    ds4_execution_state state;
    ds4_status operation_status;
    uint32_t saved_victims, released_victims;
    uint64_t loaded_groups;
} ds4_execution_recovery;
/* Query between calls or from a journal callback on the execution thread. */
ds4_status ds4_execution_recovery_query(const ds4_execution *, ds4_execution_recovery *);
ds4_status ds4_execution_close(ds4_execution *);
#ifdef __cplusplus
}
#endif
#endif
