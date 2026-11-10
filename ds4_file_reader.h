/* Private host-side owner for bounded exact/indexed file reads. The reader
 * duplicates the descriptor; destinations stay borrowed until each call returns.
 * Only the calling thread publishes errors and counters. No GPU or cache state. */
#ifndef DS4_FILE_READER_H
#define DS4_FILE_READER_H

#include "ds4_read_queue.h"
#include <errno.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>

enum { DS4_FILE_READ_CHUNK_BYTES = 8u << 20, DS4_FILE_READ_PARALLEL_BYTES = 32u << 20 };

typedef struct {
    uint64_t offset, requested;
    ssize_t received;
    int error_number;
} ds4_file_read_error;

typedef struct {
    void *destination;
    uint64_t offset, bytes, completed;
    ds4_file_read_error error;
} ds4_file_read_job;

typedef struct {
    int fd;
    uint64_t size, bytes_read;
    unsigned workers;
    bool queue_tried;
    ds4_read_queue queue;
} ds4_file_reader;

static bool ds4_file_reader_fail(ds4_file_read_error *error, uint64_t offset,
                                 uint64_t bytes, ssize_t received, int number) {
    if (error)
        *error = (ds4_file_read_error){offset, bytes, received, number};
    return false;
}

static ds4_file_reader *ds4_file_reader_open(int fd, uint64_t size, unsigned workers) {
    if (fd < 0 || size > INT64_MAX || !workers || workers > DS4_READ_QUEUE_WORKERS)
        return NULL;
    int owned = dup(fd);
    if (owned < 0)
        return NULL;
    ds4_file_reader *reader = calloc(1, sizeof(*reader));
    if (!reader) {
        close(owned);
        return NULL;
    }
    reader->fd = owned;
    reader->size = size;
    reader->workers = workers;
    return reader;
}

static void ds4_file_reader_close(ds4_file_reader *reader) {
    if (!reader)
        return;
    ds4_read_queue_stop(&reader->queue);
    close(reader->fd);
    free(reader);
}

static uint64_t ds4_file_reader_bytes(const ds4_file_reader *reader) {
    return reader ? reader->bytes_read : 0;
}

static void ds4_file_reader_run(void *context, unsigned worker, void *payload) {
    (void)worker;
    const ds4_file_reader *reader = context;
    ds4_file_read_job *job = payload;
    while (job->completed < job->bytes) {
        uint64_t remaining = job->bytes - job->completed;
        size_t part = (size_t)(remaining < DS4_FILE_READ_CHUNK_BYTES
                                  ? remaining : DS4_FILE_READ_CHUNK_BYTES);
        uint64_t offset = job->offset + job->completed;
        ssize_t got = pread(reader->fd, (char *)job->destination + job->completed,
                            part, (off_t)offset);
        if (got < 0 && errno == EINTR)
            continue;
        if (got <= 0) {
            ds4_file_reader_fail(&job->error, offset, part, got, got ? errno : EIO);
            return;
        }
        job->completed += (uint64_t)got;
    }
}

static bool ds4_file_reader_publish(ds4_file_reader *reader, const ds4_file_read_job *job,
                                    ds4_file_read_error *error) {
    reader->bytes_read += job->completed;
    if (job->error.error_number) {
        if (error)
            *error = job->error;
        return false;
    }
    return true;
}

/* Both read modes prepare at most DS4_READ_QUEUE_DEPTH validated jobs;
 * only this executor owns queue
 * startup, borrowed payload lifetimes and ordered result publication. */
static bool ds4_file_reader_run_batch(ds4_file_reader *reader, ds4_file_read_job *jobs,
                                      unsigned count, ds4_file_read_error *error) {
    if (!reader->queue_tried) {
        reader->queue_tried = true;
        ds4_read_queue_start(&reader->queue, reader->workers, ds4_file_reader_run, reader);
    }
    bool parallel = ds4_read_queue_workers(&reader->queue) != 0;
    ds4_read_task tasks[DS4_READ_QUEUE_DEPTH] = {0};
    unsigned submitted = 0;
    for (; submitted < count; submitted++) {
        if (parallel) {
            if (!ds4_read_queue_submit(&reader->queue, &tasks[submitted], &jobs[submitted]))
                break;
        } else {
            ds4_file_reader_run(reader, 0, &jobs[submitted]);
        }
    }
    bool ok = true;
    for (unsigned i = 0; i < submitted; i++) {
        if (parallel)
            ds4_read_queue_wait(&reader->queue, &tasks[i]);
        bool complete = ds4_file_reader_publish(reader, &jobs[i], ok ? error : NULL);
        ok = complete && ok;
    }
    if (!ok)
        return false;
    if (submitted != count)
        return ds4_file_reader_fail(error, jobs[submitted].offset, jobs[submitted].bytes, -1, EBUSY);
    return true;
}

/* Copy indexed slices in the caller's order, including repeated indices.
 * Validate the whole request before writing; drain every submitted job before
 * returning an I/O error. Pool-start failure keeps the same synchronous API. */
static bool ds4_file_reader_read_indexed(ds4_file_reader *reader, void *destination,
                                         uint64_t offset, uint64_t stride,
                                         const uint32_t *indices, unsigned count,
                                         ds4_file_read_error *error) {
    if (error)
        *error = (ds4_file_read_error){0};
    if (!reader || offset > reader->size ||
        (count && (!destination || !indices || !stride || stride > SIZE_MAX / count)))
        return ds4_file_reader_fail(error, offset, stride, -1, EINVAL);
    if (!count)
        return true;
    uint64_t rows = (reader->size - offset) / stride;
    for (unsigned i = 0; i < count; i++)
        if (indices[i] >= rows)
            return ds4_file_reader_fail(error, offset, stride, -1, EINVAL);
    for (unsigned first = 0; first < count;) {
        ds4_file_read_job jobs[DS4_READ_QUEUE_DEPTH] = {0};
        unsigned batch = count - first;
        if (batch > DS4_READ_QUEUE_DEPTH)
            batch = DS4_READ_QUEUE_DEPTH;
        for (unsigned j = 0; j < batch; j++) {
            unsigned i = first + j;
            jobs[j] = (ds4_file_read_job){.destination = (char *)destination + i * stride,
                                           .offset = offset + indices[i] * stride,
                                           .bytes = stride};
        }
        if (!ds4_file_reader_run_batch(reader, jobs, batch, error))
            return false;
        first += batch;
    }
    return true;
}

/* A failed read can modify part of the destination. Its owner must publish it
 * only on success. Large dense tensors use the shared batch executor,
 * writing disjoint pieces of the same buffer.
 * Small reads never start the pool; no extra staging storage is needed. */
static bool ds4_file_reader_read(ds4_file_reader *reader, void *destination,
                                 uint64_t offset, uint64_t bytes,
                                 ds4_file_read_error *error) {
    if (error)
        *error = (ds4_file_read_error){0};
    if (!reader || offset > reader->size || bytes > reader->size - offset ||
        bytes > SIZE_MAX || (bytes && !destination))
        return ds4_file_reader_fail(error, offset, bytes, -1, EINVAL);
    if (bytes >= DS4_FILE_READ_PARALLEL_BYTES && reader->workers > 1) {
        ds4_file_read_job jobs[DS4_READ_QUEUE_WORKERS] = {0};
        uint64_t stride = bytes / reader->workers;
        for (unsigned i = 0; i < reader->workers; i++)
            jobs[i] = (ds4_file_read_job){.destination = (char *)destination + i * stride,
                                          .offset = offset + i * stride,
                                          .bytes = stride};
        if (!ds4_file_reader_run_batch(reader, jobs, reader->workers, error))
            return false;
        uint64_t completed = stride * reader->workers;
        if (completed == bytes)
            return true;
        destination = (char *)destination + completed;
        offset += completed;
        bytes -= completed;
    }
    ds4_file_read_job job = {.destination = destination, .offset = offset, .bytes = bytes};
    ds4_file_reader_run(reader, 0, &job);
    return ds4_file_reader_publish(reader, &job, error);
}

#endif
