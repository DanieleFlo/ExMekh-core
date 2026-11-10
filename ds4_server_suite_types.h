#ifndef DS4_SERVER_SUITE_TYPES_H
#define DS4_SERVER_SUITE_TYPES_H
#include "ds4_gemma_multimodal.h"
#include "ds4_qwen_image_pipeline.h"
#include "ds4_suite_text.h"
#include "ds4_sha256.h"
#include <pthread.h>
#include <sys/mman.h>
#define SUITE_FILES 256
#define SUITE_MODELS 2
#define SUITE_RECEIPTS 64
#define SUITE_IMAGE_MAX 8
#define SUITE_UPLOAD_MAX (512ull * 1024 * 1024)
#define SUITE_RECEIPT_MAX (64ull * 1024 * 1024)
#define SUITE_PROGRESS_HEARTBEAT_SECONDS 10.0
typedef struct {
    char phase[48], unit[24];
    uint64_t completed_units, total_units, revision;
    uint32_t item_index, item_count, active_items;
    double overall_fraction;
    bool estimated;
} suite_progress;
typedef enum {
    SUITE_EVENT_STARTED,
    SUITE_EVENT_PROGRESS,
    SUITE_EVENT_HEARTBEAT,
    SUITE_EVENT_COMPLETED,
    SUITE_EVENT_FAILED,
    SUITE_EVENT_CANCELLED,
} suite_event;
typedef struct {
    char *body;
    int code;
    suite_event event;
} suite_terminal;
typedef struct {
    char id[38], path[PATH_MAX], mime[40], name[256];
    uint64_t bytes;
    time_t created, expires;
} suite_file;
typedef struct {
    char key[129], path[PATH_MAX], request_id[38];
    uint8_t hash[32];
    time_t expires;
    uint64_t bytes;
    int code;
} suite_receipt;
typedef struct {
    ds4_bundle *bundle;
    ds4_qwen_image_pipeline *image;
    ds4_gemma_multimodal *embed;
    ds4_gemma_processor *processor;
    char *asset;
} suite_model;
typedef struct suite_operation suite_operation;
typedef struct {
    char *id;
    char *json;
} suite_catalog_model;
typedef struct {
    pthread_mutex_t mutex;
    suite_catalog_model models[SUITE_MODELS];
} suite_catalog;
typedef struct {
    ds4_runtime *runtime;
    ds4_resources *resources;
    ds4_numeric *numeric;
    suite_model models[SUITE_MODELS];
    suite_file files[SUITE_FILES];
    suite_receipt receipts[SUITE_RECEIPTS];
    char directory[PATH_MAX - 64];
    uint64_t quota, stored, device_working_set;
    double timeout, image_timeout;
    suite_operation *recovery;
    suite_catalog catalog;
} suite_state;
typedef struct {
    const char *method, *path, *content_type, *key;
    const char *body;
    size_t length;
    char request_id[38];
    double deadline;
    bool stream;
    double started;
    /* Owned by the connection; worker access is protected by job.mu. */
    suite_progress progress;
    suite_terminal terminal;
} suite_http;
#endif
