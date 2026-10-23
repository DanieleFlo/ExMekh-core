/* Qualification-only server. The release target does not compile this hook.
 * A local barrier stops at a fenced native boundary until the actual HTTP
 * peer disconnects. Cancellation and cleanup remain the production path. */
#include "../ds4_qwen_image_observer.h"
static ds4_status phase_barrier(void *, const char *, uint32_t, ds4_num_view);
#define DS4_NATIVE_SUITE
#define DS4_SUITE_TEST_PHASE_OBSERVER phase_barrier
#include "../ds4_server.c"

static ds4_status phase_barrier(void *opaque, const char *stage, uint32_t layer,
                                ds4_num_view view) {
    (void)view;
    suite_operation *operation = opaque;
    const char *path = getenv("DS4_SUITE_TEST_PHASE_BARRIER");
    if (!path) return DS4_STATUS_OK;
    FILE *control = fopen(path, "r");
    if (!control) return DS4_STATUS_OK;
    char selected[64]; unsigned target;
    int fields = fscanf(control, "%63s %u", selected, &target);
    fclose(control);
    if (fields != 2 || strcmp(stage, selected) || layer != target)
        return DS4_STATUS_OK;
    char ready[PATH_MAX];
    if (snprintf(ready, sizeof(ready), "%s.ready", path) >= (int)sizeof(ready))
        return DS4_STATUS_INVALID;
    FILE *signal = fopen(ready, "w");
    if (!signal) return DS4_STATUS_INFERENCE;
    fprintf(signal, "%s %u\n", stage, layer);
    if (fclose(signal)) return DS4_STATUS_INFERENCE;
    double deadline = now_sec() + 15;
    while (!suite_operation_cancel(operation) && now_sec() < deadline)
        usleep(1000);
    if (!suite_operation_cancel(operation)) return DS4_STATUS_INFERENCE;
    /* Record the ledger after the worker's error cleanup, not at this point:
     * tensors and the operation lease are still intentionally alive here. */
    return DS4_STATUS_CANCELLED;
}
