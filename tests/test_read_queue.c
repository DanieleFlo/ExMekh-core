#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include "ds4_read_queue.h"

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    unsigned started, completed;
    bool hold;
} fixture;

typedef struct { unsigned value, result; int error; } request;

static void read_fixture(void *context, unsigned worker, void *payload) {
    fixture *f = (fixture *)context;
    request *r = (request *)payload;
    assert(worker < DS4_READ_QUEUE_WORKERS);
    pthread_mutex_lock(&f->mutex);
    f->started++;
    pthread_cond_broadcast(&f->changed);
    while (f->hold) pthread_cond_wait(&f->changed, &f->mutex);
    pthread_mutex_unlock(&f->mutex);
    r->result = r->value * 17u;
    r->error = r->value == 7u ? EIO : 0;
    pthread_mutex_lock(&f->mutex);
    f->completed++;
    pthread_mutex_unlock(&f->mutex);
}

int main(void) {
    ds4_read_queue queue;
    fixture f = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, true};
    ds4_read_task tasks[DS4_READ_QUEUE_DEPTH + 1] = {{0}};
    request rows[DS4_READ_QUEUE_DEPTH + 1] = {{0}};
    assert(!ds4_read_queue_start(&queue, 0, read_fixture, &f));
    assert(!ds4_read_queue_start(&queue, DS4_READ_QUEUE_WORKERS + 1u, read_fixture, &f));
    assert(!ds4_read_queue_start(&queue, 4, NULL, &f));
    assert(ds4_read_queue_start(&queue, 4, read_fixture, &f));
    for (unsigned i = 0; i < DS4_READ_QUEUE_DEPTH; i++) {
        rows[i].value = i + 1u;
        assert(ds4_read_queue_submit(&queue, &tasks[i], &rows[i]));
    }
    /* The callbacks are gated: all submitted tasks remain live. This proves
     * the limit without scheduler timing assumptions or a busy sleep. */
    assert(!ds4_read_queue_submit(&queue, &tasks[DS4_READ_QUEUE_DEPTH], &rows[DS4_READ_QUEUE_DEPTH]));
    pthread_mutex_lock(&f.mutex);
    while (f.started < 4) pthread_cond_wait(&f.changed, &f.mutex);
    f.hold = false;
    pthread_cond_broadcast(&f.changed);
    pthread_mutex_unlock(&f.mutex);
    for (unsigned i = 0; i < DS4_READ_QUEUE_DEPTH; i++) {
        ds4_read_queue_wait(&queue, &tasks[i]);
        assert(rows[i].result == rows[i].value * 17u);
        assert(rows[i].error == (rows[i].value == 7u ? EIO : 0));
    }
    /* Reuse completed task objects. stop must drain even if the caller never
     * explicitly waits for this second batch, before freeing its buffers. */
    for (unsigned i = 0; i < DS4_READ_QUEUE_DEPTH; i++) {
        rows[i].value += 20u; rows[i].result = 0;
        assert(ds4_read_queue_submit(&queue, &tasks[i], &rows[i]));
    }
    assert(ds4_read_queue_workers(&queue) == 4u);
    ds4_read_queue_stop(&queue);
    assert(ds4_read_queue_workers(&queue) == 0u);
    assert(f.completed == 2u * DS4_READ_QUEUE_DEPTH);
    for (unsigned i = 0; i < DS4_READ_QUEUE_DEPTH; i++) assert(rows[i].result == rows[i].value * 17u);
    ds4_read_queue_stop(&queue);
    assert(!ds4_read_queue_submit(&queue, &tasks[0], &rows[0]));
    assert(ds4_read_queue_start(&queue, 1, read_fixture, &f));
    assert(ds4_read_queue_submit(&queue, &tasks[0], &rows[0]));
    ds4_read_queue_wait(&queue, &tasks[0]);
    ds4_read_queue_stop(&queue);
    pthread_cond_destroy(&f.changed); pthread_mutex_destroy(&f.mutex);
    puts("read queue: bounded concurrency, completion, error payload, reuse, drain and restart PASS");
    return 0;
}
