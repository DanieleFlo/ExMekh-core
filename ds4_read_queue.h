/* Bounded host work queue. The caller owns task payloads and buffers until
 * wait completes; workers never own cache metadata or accelerator objects. */
#ifndef DS4_READ_QUEUE_H
#define DS4_READ_QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define DS4_READ_QUEUE_WORKERS 8u
#define DS4_READ_QUEUE_DEPTH 16u

typedef void (*ds4_read_queue_fn)(void *, unsigned, void *);

typedef struct ds4_read_task {
    struct ds4_read_task *next;
    void *payload;
    bool done;
} ds4_read_task;

typedef struct ds4_read_queue ds4_read_queue;
typedef struct {
    ds4_read_queue *queue;
    unsigned index;
    pthread_t thread;
} ds4_read_worker;

struct ds4_read_queue {
    pthread_mutex_t mutex;
    pthread_cond_t ready, complete;
    ds4_read_task *head, *tail;
    ds4_read_queue_fn run;
    void *context;
    ds4_read_worker workers[DS4_READ_QUEUE_WORKERS];
    unsigned n_workers, pending;
    bool stop;
};

static void *ds4_read_queue_worker(void *arg) {
    ds4_read_worker *worker = (ds4_read_worker *)arg;
    ds4_read_queue *q = worker->queue;
    for (;;) {
        pthread_mutex_lock(&q->mutex);
        while (!q->head && !q->stop) pthread_cond_wait(&q->ready, &q->mutex);
        if (!q->head && q->stop) { pthread_mutex_unlock(&q->mutex); break; }
        ds4_read_task *task = q->head;
        q->head = task->next;
        if (!q->head) q->tail = NULL;
        pthread_mutex_unlock(&q->mutex);
        q->run(q->context, worker->index, task->payload);
        pthread_mutex_lock(&q->mutex);
        task->done = true;
        q->pending--;
        pthread_cond_broadcast(&q->complete);
        pthread_mutex_unlock(&q->mutex);
    }
    return NULL;
}

/* Stop drains submitted work before joining: no callback can outlive the
 * caller's model descriptors, host buffers or error recovery. */
static void ds4_read_queue_stop(ds4_read_queue *q) {
    if (!q->n_workers) return;
    pthread_mutex_lock(&q->mutex);
    while (q->pending) pthread_cond_wait(&q->complete, &q->mutex);
    q->stop = true;
    pthread_cond_broadcast(&q->ready);
    pthread_mutex_unlock(&q->mutex);
    for (unsigned i = 0; i < q->n_workers; i++)
        if (pthread_join(q->workers[i].thread, NULL)) abort(); /* a worker must not outlive its buffers */
    pthread_cond_destroy(&q->complete);
    pthread_cond_destroy(&q->ready);
    pthread_mutex_destroy(&q->mutex);
    q->n_workers = 0;
}

static int ds4_read_queue_start(ds4_read_queue *q, unsigned workers,
                               ds4_read_queue_fn run, void *context) {
    if (!workers || workers > DS4_READ_QUEUE_WORKERS || !run) return 0;
    memset(q, 0, sizeof(*q));
    if (pthread_mutex_init(&q->mutex, NULL)) return 0;
    if (pthread_cond_init(&q->ready, NULL)) {
        pthread_mutex_destroy(&q->mutex); return 0;
    }
    if (pthread_cond_init(&q->complete, NULL)) {
        pthread_cond_destroy(&q->ready); pthread_mutex_destroy(&q->mutex); return 0;
    }
    q->run = run; q->context = context;
    for (unsigned i = 0; i < workers; i++) {
        q->workers[i].queue = q; q->workers[i].index = i;
        if (pthread_create(&q->workers[i].thread, NULL, ds4_read_queue_worker, &q->workers[i])) {
            if (q->n_workers) ds4_read_queue_stop(q);
            else {
                pthread_cond_destroy(&q->complete); pthread_cond_destroy(&q->ready);
                pthread_mutex_destroy(&q->mutex);
            }
            return 0;
        }
        q->n_workers++;
    }
    return 1;
}

static unsigned ds4_read_queue_workers(const ds4_read_queue *q) { return q->n_workers; }

static int ds4_read_queue_submit(ds4_read_queue *q, ds4_read_task *task, void *payload) {
    if (!q->n_workers || !task) return 0;
    pthread_mutex_lock(&q->mutex);
    if (q->stop || q->pending >= DS4_READ_QUEUE_DEPTH) {
        pthread_mutex_unlock(&q->mutex); return 0;
    }
    task->next = NULL; task->payload = payload; task->done = false;
    if (q->tail) q->tail->next = task;
    else q->head = task;
    q->tail = task; q->pending++;
    pthread_cond_signal(&q->ready);
    pthread_mutex_unlock(&q->mutex);
    return 1;
}

static void ds4_read_queue_wait(ds4_read_queue *q, ds4_read_task *task) {
    pthread_mutex_lock(&q->mutex);
    while (!task->done) pthread_cond_wait(&q->complete, &q->mutex);
    pthread_mutex_unlock(&q->mutex);
}

#endif
