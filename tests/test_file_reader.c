#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>

/* Exercise complete and partial pool-start failure without a production flag. */
static unsigned starts_before_failure = UINT_MAX, thread_starts;
static int controlled_thread_create(pthread_t *thread, const pthread_attr_t *attributes,
                                     void *(*start)(void *), void *argument) {
    thread_starts++;
    if (!starts_before_failure)
        return EAGAIN;
    if (starts_before_failure != UINT_MAX)
        starts_before_failure--;
    return pthread_create(thread, attributes, start, argument);
}
#define pthread_create controlled_thread_create
#include "ds4_read_queue.h"
#undef pthread_create

/* Reject a submission through the queue API, without inspecting its state. */
static unsigned submits_before_failure = UINT_MAX;
static int controlled_queue_submit(ds4_read_queue *queue, ds4_read_task *task, void *payload) {
    if (!submits_before_failure)
        return 0;
    if (submits_before_failure != UINT_MAX)
        submits_before_failure--;
    return ds4_read_queue_submit(queue, task, payload);
}
#define ds4_read_queue_submit controlled_queue_submit
#include "ds4_file_reader.h"
#undef ds4_read_queue_submit

enum { PREFIX = 17, ROWS = 40, STRIDE = 256 };

static void check_rows(const unsigned char *source, const unsigned char *actual,
                        const uint32_t *indices, unsigned count) {
    for (unsigned i = 0; i < count; i++)
        assert(!memcmp(actual + i * STRIDE, source + PREFIX + indices[i] * STRIDE, STRIDE));
}

static void test_bounds(ds4_file_reader *reader, unsigned char *output, size_t bytes) {
    memset(output, 0xA5, bytes);
    uint32_t invalid[] = {0, ROWS};
    uint64_t before = ds4_file_reader_bytes(reader);
    ds4_file_read_error error;
    assert(!ds4_file_reader_read_indexed(reader, output, PREFIX, STRIDE, invalid, 2, &error));
    assert(error.error_number == EINVAL);
    assert(!ds4_file_reader_read_indexed(reader, output, PREFIX, UINT64_MAX, invalid, 2, &error));
    assert(!ds4_file_reader_read(reader, output, UINT64_MAX, 32, &error));
    assert(!ds4_file_reader_read(reader, NULL, 0, 1, &error));
    assert(!ds4_file_reader_read(NULL, output, 0, 1, &error));
    assert(ds4_file_reader_read(reader, NULL, PREFIX + ROWS * STRIDE, 0, &error));
    assert(ds4_file_reader_read_indexed(reader, NULL, 0, 0, NULL, 0, &error));
    assert(!error.error_number && ds4_file_reader_bytes(reader) == before);
    for (size_t i = 0; i < bytes; i++)
        assert(output[i] == 0xA5);
}

static void test_partial_eof(ds4_file_reader *reader, int fd, const unsigned char *source,
                             unsigned char *output, size_t source_bytes) {
    const uint32_t indices[] = {0, 5, 15, 1};
    const off_t end = PREFIX + 5 * STRIDE + STRIDE / 2;
    assert(!ftruncate(fd, end));
    memset(output, 0xA5, 4 * STRIDE);
    uint64_t before = ds4_file_reader_bytes(reader);
    ds4_file_read_error error;
    errno = ENOTTY;
    assert(!ds4_file_reader_read_indexed(reader, output, PREFIX, STRIDE, indices, 4, &error));
    assert(error.error_number == EIO && error.offset == (uint64_t)end && !error.received);
    assert(error.requested == STRIDE / 2);
    assert(ds4_file_reader_bytes(reader) == before + 2 * STRIDE + STRIDE / 2);
    assert(!memcmp(output, source + PREFIX, STRIDE));
    assert(!memcmp(output + STRIDE, source + PREFIX + 5 * STRIDE, STRIDE / 2));
    assert(!memcmp(output + 3 * STRIDE, source + PREFIX + STRIDE, STRIDE));
    for (unsigned i = 0; i < STRIDE; i++)
        assert(output[2 * STRIDE + i] == 0xA5);
    /* Restore the same file: a failed batch must not poison subsequent reads. */
    assert(!ftruncate(fd, (off_t)source_bytes));
    assert(pwrite(fd, source, source_bytes, 0) == (ssize_t)source_bytes);
    assert(ds4_file_reader_read_indexed(reader, output, PREFIX, STRIDE, indices, 4, &error));
    assert(!error.error_number);
    check_rows(source, output, indices, 4);
}

static void test_pool_failure(int fd, const unsigned char *source, size_t bytes,
                              const uint32_t *indices, unsigned char *output) {
    for (unsigned fail_after = 0; fail_after <= 2; fail_after += 2) {
        starts_before_failure = fail_after;
        ds4_file_reader *reader = ds4_file_reader_open(fd, bytes, 4);
        assert(reader);
        assert(ds4_file_reader_read_indexed(reader, output, PREFIX, STRIDE, indices, ROWS, NULL));
        check_rows(source, output, indices, ROWS);
        assert(ds4_file_reader_bytes(reader) == ROWS * STRIDE);
        unsigned attempts = thread_starts;
        assert(ds4_file_reader_read_indexed(reader, output, PREFIX, STRIDE, indices, ROWS, NULL));
        assert(thread_starts == attempts && ds4_file_reader_bytes(reader) == 2 * ROWS * STRIDE);
        test_partial_eof(reader, fd, source, output, bytes);
        assert(thread_starts == attempts);
        ds4_file_reader_close(reader);
    }
    starts_before_failure = UINT_MAX;
}

static void test_submission_failure(ds4_file_reader *reader, int fd,
                                     const unsigned char *source, unsigned char *output,
                                     size_t source_bytes) {
    const uint32_t indices[] = {5, 2, 9, 1};
    uint64_t before = ds4_file_reader_bytes(reader);
    ds4_file_read_error error;
    memset(output, 0xA5, 4 * STRIDE);
    submits_before_failure = 2;
    assert(!ds4_file_reader_read_indexed(reader, output, PREFIX, STRIDE, indices, 4, &error));
    assert(error.error_number == EBUSY && error.offset == PREFIX + indices[2] * STRIDE);
    assert(error.requested == STRIDE && error.received == -1);
    assert(ds4_file_reader_bytes(reader) == before + 2 * STRIDE);
    check_rows(source, output, indices, 2);
    for (unsigned i = 2 * STRIDE; i < 4 * STRIDE; i++)
        assert(output[i] == 0xA5);
    /* An earlier I/O error takes precedence, but both submitted jobs drain. */
    assert(!ftruncate(fd, PREFIX + 3 * STRIDE));
    submits_before_failure = 2;
    assert(!ds4_file_reader_read_indexed(reader, output, PREFIX, STRIDE, indices, 4, &error));
    assert(error.error_number == EIO && error.offset == PREFIX + indices[0] * STRIDE);
    assert(ds4_file_reader_bytes(reader) == before + 3 * STRIDE);
    submits_before_failure = UINT_MAX;
    assert(!ftruncate(fd, (off_t)source_bytes));
    assert(pwrite(fd, source, source_bytes, 0) == (ssize_t)source_bytes);
    assert(ds4_file_reader_read_indexed(reader, output, PREFIX, STRIDE, indices, 4, &error));
    assert(!error.error_number);
    check_rows(source, output, indices, 4);
}

static void check_complete_read(ds4_file_reader *reader, const unsigned char *source,
                                 unsigned char *output, size_t offset, size_t bytes) {
    uint64_t before = ds4_file_reader_bytes(reader);
    ds4_file_read_error error;
    assert(ds4_file_reader_read(reader, output, offset, bytes, &error));
    assert(!error.error_number && !memcmp(output, source + offset, bytes));
    assert(ds4_file_reader_bytes(reader) == before + bytes);
}

static void check_complete_eof(ds4_file_reader *reader, int fd, const unsigned char *source,
                                unsigned char *output, size_t offset, size_t bytes,
                                size_t available) {
    assert(!ftruncate(fd, (off_t)(offset + available)));
    memset(output, 0xA5, bytes);
    uint64_t before = ds4_file_reader_bytes(reader);
    ds4_file_read_error error;
    assert(!ds4_file_reader_read(reader, output, offset, bytes, &error));
    assert(error.error_number == EIO && error.offset == offset + available);
    assert(ds4_file_reader_bytes(reader) == before + available);
    assert(!memcmp(output, source + offset, available));
    for (size_t i = available; i < bytes; i++)
        assert(output[i] == 0xA5);
    assert(!ftruncate(fd, (off_t)(offset + bytes)));
    assert(pwrite(fd, source, offset + bytes, 0) == (ssize_t)(offset + bytes));
    check_complete_read(reader, source, output, offset, bytes);
}

static void test_large_reads(unsigned workers, unsigned fail_after) {
    const size_t bytes = (32u << 20) + 3, offset = 17, size = offset + bytes;
    unsigned char *source = malloc(size), *output = malloc(bytes);
    assert(source && output);
    for (size_t i = 0; i < size; i++)
        source[i] = (unsigned char)(i * 17 + i / 4096);
    FILE *file = tmpfile();
    assert(file && fwrite(source, 1, size, file) == size && !fflush(file));
    int fd = fileno(file);
    starts_before_failure = fail_after;
    ds4_file_reader *reader = ds4_file_reader_open(fd, size, workers);
    assert(reader);
    unsigned attempts = thread_starts;
    check_complete_read(reader, source, output, offset, (32u << 20) - 1);
    assert(thread_starts == attempts); /* below the retained threshold stays lazy */
    const size_t lengths[] = {8u << 20, (8u << 20) + 3, 16u << 20, 24u << 20,
                              32u << 20, bytes};
    for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++)
        check_complete_read(reader, source, output, offset, lengths[i]);
    attempts = thread_starts;
    /* File boundaries are independent of the implementation's worker splits. */
    const size_t medium = (8u << 20) + 3;
    check_complete_eof(reader, fd, source, output, offset, medium, 0);
    check_complete_eof(reader, fd, source, output, offset, medium, medium / 3);
    check_complete_eof(reader, fd, source, output, offset, medium, medium - 1);
    check_complete_eof(reader, fd, source, output, offset, bytes, 0);
    check_complete_eof(reader, fd, source, output, offset, bytes, bytes / 3);
    check_complete_eof(reader, fd, source, output, offset, bytes, bytes - 1);
    assert(thread_starts == attempts); /* failed pool startup is not retried */
    ds4_file_reader_close(reader);
    starts_before_failure = UINT_MAX;
    assert(!fclose(file));
    free(output);
    free(source);
}

int main(void) {
    unsigned char source[PREFIX + ROWS * STRIDE], output[ROWS * STRIDE];
    uint32_t indices[ROWS];
    for (unsigned i = 0; i < sizeof(source); i++)
        source[i] = (unsigned char)(i * 17 + i / STRIDE);
    for (unsigned i = 0; i < ROWS; i++)
        indices[i] = (i * 7) % ROWS;
    indices[0] = indices[1] = 3;
    FILE *file = tmpfile();
    assert(file && fwrite(source, 1, sizeof(source), file) == sizeof(source) && !fflush(file));
    int fd = fileno(file), borrowed = dup(fd);
    assert(borrowed >= 0);
    ds4_file_reader *reader = ds4_file_reader_open(borrowed, sizeof(source), 4);
    assert(reader && !close(borrowed));
    assert(!ds4_file_reader_bytes(reader));
    unsigned attempts = thread_starts;
    test_bounds(reader, output, sizeof(output));
    assert(thread_starts == attempts);
    ds4_file_read_error error;
    assert(ds4_file_reader_read(reader, output, 5, 601, &error));
    assert(!error.error_number && !memcmp(output, source + 5, 601));
    assert(ds4_file_reader_read_indexed(reader, output, PREFIX, STRIDE, indices, ROWS, &error));
    check_rows(source, output, indices, ROWS);
    assert(ds4_file_reader_bytes(reader) == 601 + ROWS * STRIDE);
    test_submission_failure(reader, fd, source, output, sizeof(source));
    test_partial_eof(reader, fd, source, output, sizeof(source));
    ds4_file_reader_close(reader);
    test_pool_failure(fd, source, sizeof(source), indices, output);
    assert(!ds4_file_reader_open(-1, sizeof(source), 4));
    assert(!ds4_file_reader_open(fd, UINT64_MAX, 4));
    assert(!ds4_file_reader_open(fd, sizeof(source), 0));
    int closed = dup(fd);
    assert(closed >= 0 && !close(closed));
    assert(!ds4_file_reader_open(closed, sizeof(source), 4));
    reader = ds4_file_reader_open(fd, sizeof(source), 4);
    assert(reader && !ds4_file_reader_bytes(reader));
    ds4_file_reader_close(reader);
    ds4_file_reader_close(NULL);
    assert(!fclose(file));
    for (unsigned workers = 1; workers <= DS4_READ_QUEUE_WORKERS; workers *= 2)
        test_large_reads(workers, UINT_MAX);
    test_large_reads(4, 0);
    test_large_reads(4, 2);
    puts("file reader: descriptor ownership, indexed batches, bounds, EOF, counters, recovery and pool fallback PASS");
    return 0;
}
