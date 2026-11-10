#ifndef DS4_OPERATION_PROGRESS_H
#define DS4_OPERATION_PROGRESS_H
#include <stdbool.h>
#include <stdint.h>

/* Model/backend-independent metadata. Strings are borrowed for the callback;
 * consumers copy them before returning. total_units=0 and fraction<0 mean
 * unknown. Publishing never fences, reads tensors, allocates or performs I/O. */
typedef struct {
    const char *phase, *unit;
    uint64_t completed_units, total_units;
    double overall_fraction;
    bool estimated;
} ds4_operation_progress;
typedef struct {
    void *context;
    void (*update)(void *, const ds4_operation_progress *);
} ds4_operation_observer;
static inline void ds4_operation_publish(const ds4_operation_observer *observer,
                                         ds4_operation_progress progress) {
    if (observer && observer->update)
        observer->update(observer->context, &progress);
}
#endif
