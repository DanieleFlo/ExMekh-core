/* Bounded explicit SSD storage for Qwen models. Arithmetic stays in their kernels.
 * Dense retained plans recycle at capacity pressure; routed plans at layers.
 * Fence consumers before reuse. Router compaction preserves slot/order/weights. */
#include "ds4_file_reader.h"

enum { QWEN4_STREAM_ENTRIES = 96, QWEN4_STREAM_EXPERTS = 512, QWEN4_STREAM_READ_WORKERS = 4 };
typedef struct {
    uint64_t offset, bytes;
    ds4_gpu_tensor *tensor;
    bool ready;
} qwen4_stream_resident;
static struct {
    const void *map;
    uint64_t size, limit, live, reserved;
    ds4_file_reader *reader;
    qwen4_stream_resident *resident;
    unsigned resident_count;
    ds4_gpu_weight_region *layers;
    unsigned layer_count;
    uint64_t layer_capacity;
    ds4_gpu_weight_region head;
    uint64_t layer_transient, fallback_transient;
    bool layer_transport;
    struct {
        ds4_gpu_tensor *tensor;
        uint64_t offset, bytes;
        bool ready, pinned;
    } layer_slots[2];
    struct {
        uint64_t offset, bytes, capacity, expert_stride;
        ds4_gpu_tensor *tensor;
        bool ready, pinned;
    } entries[QWEN4_STREAM_ENTRIES];
    unsigned count, experts, tokens, slots, expert_domain;
    uint32_t ids[QWEN4_STREAM_EXPERTS];
    const ds4_gpu_tensor *selection;
    ds4_gpu_tensor *compact;
} qwen4_stream;
/* ARC owns these references separately from the resettable C metadata. Each
 * is the last encoded consumer, not merely the last completed read. */
static id<MTLCommandBuffer> qwen35_stream_last_use[2];

/* Only for private owned storage being removed after the queue is fenced.
 * ARC release alone can leave its pages in Metal's pool while a fresh memory
 * plan needs them for a graph restore. Common/reusable regions never enter
 * here, and the discarded buffer is never bound or read again. */
static void qwen4_stream_discard(ds4_gpu_tensor *tensor) {
    if (tensor)
        [ds4_gpu_tensor_buffer(tensor) setPurgeableState:MTLPurgeableStateEmpty];
    ds4_gpu_tensor_free(tensor);
}

static bool qwen4_stream_matches(const void *map, uint64_t size) {
    return qwen4_stream.map && map == qwen4_stream.map && size == qwen4_stream.size;
}
/* A binding is pinned until its dispatch has encoded all arguments. Eviction
 * fences the queue before overwriting or releasing any previously used slot. */
static void qwen4_stream_dispatch_encoded(id<MTLCommandBuffer> cb) {
    for (unsigned i = 0; i < 2; i++)
        if (qwen4_stream.layer_slots[i].pinned) {
            if (cb) qwen35_stream_last_use[i] = cb;
            qwen4_stream.layer_slots[i].pinned = false;
        }
    for (unsigned i = 0; i < qwen4_stream.count; i++)
        qwen4_stream.entries[i].pinned = false;
}
int ds4_gpu_qwen4_stream_active(void) { return qwen4_stream.map != NULL; }
static void qwen4_stream_clear_transient(void) {
    for (unsigned i = 0; i < qwen4_stream.count; i++) {
        qwen4_stream.live -= qwen4_stream.entries[i].capacity;
        ds4_gpu_tensor_free(qwen4_stream.entries[i].tensor);
    }
    qwen4_stream.live -= ds4_gpu_tensor_bytes(qwen4_stream.compact);
    ds4_gpu_tensor_free(qwen4_stream.compact);
    qwen4_stream.count = 0;
    qwen4_stream.experts = 0;
    qwen4_stream.selection = NULL;
    qwen4_stream.compact = NULL;
}
int ds4_gpu_qwen4_stream_layer(const void *map, uint64_t size) {
    if (!qwen4_stream_matches(map, size))
        return 1;
    bool reusable = qwen4_stream.resident_count && !qwen4_stream.compact &&
                    !qwen4_stream.selection;
    for (unsigned i = 0; reusable && i < qwen4_stream.count; i++)
        reusable = qwen4_stream.entries[i].ready;
    if (reusable) {
        /* Old encoded consumers remain in queue order. Eviction still
         * drains before reuse; a failed read instead takes the clear below.
         * A boundary also ends any abandoned pre-encode argument pins. */
        qwen4_stream_dispatch_encoded(g_batch_cb);
        return ds4_gpu_metal_resources_reconcile();
    }
    bool batch = ds4_gpu_commands_active();
    if (!ds4_gpu_synchronize())
        return 0;
    qwen4_stream_clear_transient();
    if (!ds4_gpu_metal_resources_reconcile())
        return 0;
    return !batch || ds4_gpu_begin_commands();
}
int ds4_gpu_qwen4_stream_close(void) {
    if (qwen4_stream.map) {
        if (!ds4_gpu_synchronize())
            return 0;
        ds4_file_reader_close(qwen4_stream.reader);
        qwen4_stream.reader = NULL;
        qwen4_stream_clear_transient();
        for (unsigned i = 0; i < qwen4_stream.resident_count; i++)
            qwen4_stream_discard(qwen4_stream.resident[i].tensor);
        for (unsigned i = 0; i < 2; i++) {
            qwen35_stream_last_use[i] = nil;
            qwen4_stream_discard(qwen4_stream.layer_slots[i].tensor);
        }
        free(qwen4_stream.layers);
        free(qwen4_stream.resident);
    }
    memset(&qwen4_stream, 0, sizeof(qwen4_stream));
    return 1;
}
int ds4_gpu_qwen4_stream_open(const void *map, uint64_t size, int fd, uint64_t limit) {
    if (!map || !size || fd < 0 || size > INT64_MAX)
        return 0;
    /* Acquire the replacement before releasing the working owner. A failed
     * dup or queue drain must leave the existing stream usable. */
    ds4_file_reader *reader = ds4_file_reader_open(fd, size, QWEN4_STREAM_READ_WORKERS);
    if (!reader)
        return 0;
    if (!ds4_gpu_qwen4_stream_close()) {
        ds4_file_reader_close(reader);
        return 0;
    }
    qwen4_stream.map = map;
    qwen4_stream.size = size;
    qwen4_stream.reader = reader;
    qwen4_stream.limit = limit;
    return 1;
}
void ds4_gpu_qwen4_stream_memory(uint64_t *owned, uint64_t *read_bytes) {
    if (owned)
        *owned = qwen4_stream.live;
    if (read_bytes)
        *read_bytes = ds4_file_reader_bytes(qwen4_stream.reader);
}
/* Lazy immutable regions own quota before they own storage. Transient misses
 * must not spend it, or a retained head requested last could fail to load. */
static uint64_t qwen4_stream_available(void) {
    return qwen4_stream.limit - qwen4_stream.live - qwen4_stream.reserved;
}
static ds4_gpu_tensor *qwen4_stream_alloc(uint64_t bytes) {
    if (!bytes || bytes > qwen4_stream_available())
        return NULL;
    ds4_resource_owner_scope scope = ds4_resource_owner_enter(DS4_RESOURCE_OWNER_LEGACY_WEIGHTS);
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    ds4_resource_owner_leave(&scope);
    if (t)
        qwen4_stream.live += bytes;
    else {
        uint64_t host = 0, device = 0;
        ds4_resources_available(ds4_gpu_resources(), DS4_MEMORY_HOST, &host);
        ds4_resources_available(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &device);
        fprintf(stderr, "ds4: SSD allocation denied bytes=%llu host=%llu device=%llu\n",
                (unsigned long long)bytes, (unsigned long long)host, (unsigned long long)device);
    }
    return t;
}
static bool qwen4_stream_read_result(bool ok, const ds4_file_read_error *error) {
    if (!ok) {
        fprintf(stderr, "ds4: SSD pread failed offset=%llu bytes=%llu got=%zd errno=%d\n",
                (unsigned long long)error->offset, (unsigned long long)error->requested,
                error->received, error->error_number);
        errno = error->error_number;
    }
    return ok;
}
static bool qwen4_stream_read(void *dst, uint64_t offset, uint64_t bytes) {
    /* Let already encoded work run while the host reads a new buffer. The
     * reader still completes before binding it; recycling always fences. */
    if (qwen4_stream.resident_count && bytes >= (8u << 20) &&
        g_batch_cb && g_batch_has_work && !ds4_gpu_flush_commands())
        return false;
    ds4_file_read_error error;
    bool ok = ds4_file_reader_read(qwen4_stream.reader, dst, offset, bytes, &error);
    return qwen4_stream_read_result(ok, &error);
}
static bool qwen4_stream_read_experts(void *dst, uint64_t offset, uint64_t stride) {
    ds4_file_read_error error;
    bool ok = ds4_file_reader_read_indexed(qwen4_stream.reader, dst, offset, stride,
                                          qwen4_stream.ids, qwen4_stream.experts, &error);
    return qwen4_stream_read_result(ok, &error);
}
int ds4_gpu_qwen4_stream_retain(const void *map, uint64_t size, const uint64_t *offsets,
                                const uint64_t *bytes, unsigned count, uint64_t transient) {
    if (!qwen4_stream_matches(map, size) || !offsets || !bytes || !count ||
        qwen4_stream.count || qwen4_stream.resident_count || qwen4_stream.layer_count || !transient)
        return 0;
    for (unsigned i = 0; i < count; i++)
        if (!bytes[i] || offsets[i] > size || bytes[i] > size - offsets[i])
            return 0;
    /* The first region is the head. If it cannot stay resident beside the
     * largest transient layer, keep the original streaming policy instead. */
    if (transient >= qwen4_stream.limit || bytes[0] > qwen4_stream.limit - transient)
        return 1;
    qwen4_stream_resident *regions = calloc(count, sizeof(*regions));
    if (!regions)
        return 0;
    uint64_t available = qwen4_stream.limit - transient;
    unsigned selected = 0;
    for (unsigned i = 0; i < count; i++) {
        if (bytes[i] > available)
            continue;
        regions[selected++] = (qwen4_stream_resident){.offset = offsets[i], .bytes = bytes[i]};
        available -= bytes[i];
    }
    qwen4_stream.resident = regions;
    qwen4_stream.resident_count = selected;
    qwen4_stream.reserved = qwen4_stream.limit - transient - available;
    return 1;
}
static bool qwen35_stream_region_valid(ds4_gpu_weight_region region, uint64_t size) {
    return region.bytes && region.offset <= size && region.bytes <= size - region.offset;
}
static bool qwen35_stream_regions_overlap(ds4_gpu_weight_region a, ds4_gpu_weight_region b) {
    return a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes;
}
typedef struct {
    qwen4_stream_resident *regions;
    unsigned count;
    uint64_t reserved;
    bool use_slots;
} qwen35_stream_layout;

/* Selection is metadata-only. Minimum transport storage is reserved before
 * immutable regions, and the same selection is reused when RAM changes. */
static bool qwen35_stream_select_layout(qwen35_stream_layout *layout,
    uint64_t limit, ds4_gpu_weight_region head, const ds4_gpu_weight_region *layers,
    unsigned count, uint64_t largest, uint64_t transient, uint64_t fallback) {
    *layout = (qwen35_stream_layout){0};
    layout->regions = calloc((size_t)count + 1, sizeof(*layout->regions));
    if (!layout->regions) return false;
    layout->use_slots = transient < limit && largest <= (limit - transient) / 2 &&
                        head.bytes <= limit - transient - 2 * largest;
    uint64_t reserve = layout->use_slots ? transient : fallback;
    if (reserve >= limit) return true;
    uint64_t remaining = limit - reserve;
    if (layout->use_slots) {
        remaining -= 2 * largest;
        layout->reserved = 2 * largest;
    }
    if (head.bytes > remaining) return true;
    layout->regions[layout->count++] =
        (qwen4_stream_resident){.offset = head.offset, .bytes = head.bytes};
    remaining -= head.bytes;
    layout->reserved += head.bytes;
    for (unsigned i = 0; i < count; i++) {
        if (layers[i].bytes > remaining) continue;
        layout->regions[layout->count++] =
            (qwen4_stream_resident){.offset = layers[i].offset, .bytes = layers[i].bytes};
        remaining -= layers[i].bytes;
        layout->reserved += layers[i].bytes;
    }
    return true;
}
static bool qwen35_stream_layout_matches(const qwen35_stream_layout *layout) {
    if (layout->count != qwen4_stream.resident_count ||
        layout->use_slots != qwen4_stream.layer_transport) return false;
    for (unsigned i = 0; i < layout->count; i++)
        if (layout->regions[i].offset != qwen4_stream.resident[i].offset ||
            layout->regions[i].bytes != qwen4_stream.resident[i].bytes) return false;
    return true;
}
/* Takes ownership of the layout after old GPU consumers have been fenced.
 * A common region transfers its storage and failed-read state together. */
static void qwen35_stream_adopt_layout(qwen35_stream_layout *layout) {
    for (unsigned i = 0; i < layout->count; i++) {
        for (unsigned j = 0; j < qwen4_stream.resident_count; j++) {
            qwen4_stream_resident *old = &qwen4_stream.resident[j];
            qwen4_stream_resident *next = &layout->regions[i];
            if (next->offset != old->offset || next->bytes != old->bytes) continue;
            next->tensor = old->tensor;
            next->ready = old->ready;
            old->tensor = NULL;
            if (next->tensor) layout->reserved -= next->bytes;
            break;
        }
    }
    for (unsigned i = 0; i < qwen4_stream.resident_count; i++) {
        qwen4_stream_resident *old = &qwen4_stream.resident[i];
        if (old->tensor) qwen4_stream.live -= old->bytes;
        qwen4_stream_discard(old->tensor);
    }
    free(qwen4_stream.resident);
    for (unsigned i = 0; i < 2; i++) {
        qwen35_stream_last_use[i] = nil;
        qwen4_stream.layer_slots[i].pinned = false;
        if (!layout->use_slots) {
            if (qwen4_stream.layer_slots[i].tensor)
                qwen4_stream.live -= qwen4_stream.layer_capacity;
            qwen4_stream_discard(qwen4_stream.layer_slots[i].tensor);
            memset(&qwen4_stream.layer_slots[i], 0, sizeof(qwen4_stream.layer_slots[i]));
        } else if (qwen4_stream.layer_slots[i].tensor) {
            layout->reserved -= qwen4_stream.layer_capacity;
        }
    }
    qwen4_stream.resident = layout->regions;
    qwen4_stream.resident_count = layout->count;
    qwen4_stream.layer_transport = layout->use_slots;
    qwen4_stream.reserved = layout->reserved;
    layout->regions = NULL;
}
int ds4_gpu_qwen35_stream_configure(const void *map, uint64_t size,
                                    const ds4_gpu_qwen35_stream_plan *plan) {
    if (!qwen4_stream_matches(map, size) || !plan || !plan->layers || !plan->layer_count ||
        plan->layer_count == UINT_MAX || qwen4_stream.count || qwen4_stream.resident_count ||
        qwen4_stream.layer_count || !plan->vocab_size ||
        !qwen35_stream_region_valid(plan->head, size) ||
        !qwen35_stream_region_valid(plan->embedding, size))
        return 0;
    uint64_t largest = 0;
    for (unsigned i = 0; i < plan->layer_count; i++) {
        ds4_gpu_weight_region layer = plan->layers[i];
        if (!qwen35_stream_region_valid(layer, size)) return 0;
        if (qwen35_stream_regions_overlap(layer, plan->head) ||
            qwen35_stream_regions_overlap(layer, plan->embedding)) return -1;
        for (unsigned j = 0; j < i; j++)
            if (qwen35_stream_regions_overlap(layer, plan->layers[j])) return -1;
        if (largest < layer.bytes) largest = layer.bytes;
    }
    uint64_t transient = plan->embedding.bytes;
    if (qwen35_ud_supported(plan->embedding_type)) {
        if (transient % plan->vocab_size) return 0;
        uint64_t row = transient / plan->vocab_size;
        if (row > UINT64_MAX / QWEN4_STREAM_ENTRIES) return 0;
        transient = row * QWEN4_STREAM_ENTRIES;
    }
    if (plan->extra_bytes > UINT64_MAX - transient) return 0;
    transient += plan->extra_bytes;
    uint64_t fallback = plan->embedding.bytes > largest ? plan->embedding.bytes : largest;
    if (plan->extra_bytes > UINT64_MAX - fallback) return 0;
    fallback += plan->extra_bytes;
    qwen35_stream_layout layout;
    if (!qwen35_stream_select_layout(&layout, qwen4_stream.limit, plan->head, plan->layers,
                                      plan->layer_count, largest, transient, fallback)) return 0;
    ds4_gpu_weight_region *layers = calloc(plan->layer_count, sizeof(*layers));
    if (!layers) {
        free(layout.regions);
        return 0;
    }
    memcpy(layers, plan->layers, plan->layer_count * sizeof(*layers));
    qwen4_stream.layers = layers;
    qwen4_stream.layer_count = plan->layer_count;
    qwen4_stream.layer_capacity = largest;
    qwen4_stream.head = plan->head;
    qwen4_stream.layer_transient = transient;
    qwen4_stream.fallback_transient = fallback;
    qwen35_stream_adopt_layout(&layout);
    return 1;
}
int ds4_gpu_qwen4_stream_resize(const void *map, uint64_t size, uint64_t limit) {
    if (!qwen4_stream_matches(map, size)) return 0;
    if (limit == qwen4_stream.limit) return 1;
    qwen35_stream_layout layout = {0};
    if (qwen4_stream.layer_count &&
        !qwen35_stream_select_layout(&layout, limit, qwen4_stream.head, qwen4_stream.layers,
                                      qwen4_stream.layer_count, qwen4_stream.layer_capacity,
                                      qwen4_stream.layer_transient,
                                      qwen4_stream.fallback_transient)) return 0;
    if ((!qwen4_stream.layer_count || qwen35_stream_layout_matches(&layout)) &&
        limit >= qwen4_stream.live + qwen4_stream.reserved) {
        free(layout.regions);
        qwen4_stream.limit = limit;
        return 1;
    }
    bool batch = ds4_gpu_commands_active();
    if (!ds4_gpu_synchronize()) {
        free(layout.regions);
        return 0;
    }
    qwen4_stream_clear_transient();
    qwen35_stream_adopt_layout(&layout);
    qwen4_stream.limit = limit;
    return ds4_gpu_metal_resources_reconcile() && (!batch || ds4_gpu_begin_commands());
}
static uint64_t qwen4_stream_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS
        ? info.phys_footprint : 0;
}
int ds4_gpu_qwen4_stream_trim(const void *map, uint64_t size) {
    if (!qwen4_stream_matches(map, size)) return 0;
    uint64_t owned = qwen4_stream.live, before = qwen4_stream_footprint();
    if (!ds4_gpu_qwen4_stream_resize(map, size, 0)) return 0;
    /* GPU completion precedes physical page retirement on Apple Silicon.
     * Admit the next workload against actual capacity, never synthetic credit.
     * A bounded wait cannot guarantee retirement; timeout still resamples. */
    if (owned && before >= owned) {
        const double deadline = ds4_gpu_now_ms() + 1000;
        uint64_t current;
        while ((current = qwen4_stream_footprint()) && current > before - owned &&
               ds4_gpu_now_ms() < deadline) {
            const struct timespec pause = {.tv_nsec = 1000000};
            (void)nanosleep(&pause, NULL);
        }
    }
    return ds4_gpu_metal_resources_reconcile();
}
int ds4_gpu_qwen4_stream_read(const void *map, uint64_t size, uint64_t offset, void *destination,
                              uint64_t bytes) {
    if (!qwen4_stream_matches(map, size))
        return -1;
    return destination && qwen4_stream_read(destination, offset, bytes);
}
static id<MTLBuffer> qwen4_stream_load(uint64_t offset, uint64_t bytes, uint64_t expert_stride) {
    if (offset > qwen4_stream.size || bytes > qwen4_stream.size - offset)
        return nil;
    uint64_t allocated = expert_stride ? expert_stride * qwen4_stream.experts : bytes;
    for (unsigned i = 0; i < qwen4_stream.count; i++) {
        if (qwen4_stream.entries[i].offset == offset &&
            qwen4_stream.entries[i].bytes == allocated &&
            qwen4_stream.entries[i].expert_stride == expert_stride) {
            qwen4_stream.entries[i].pinned = true;
            return qwen4_stream.entries[i].ready
                       ? ds4_gpu_tensor_buffer(qwen4_stream.entries[i].tensor)
                       : nil;
        }
    }
    if (!allocated || allocated > qwen4_stream.limit)
        return nil;
    unsigned slot = qwen4_stream.count;
    if (qwen4_stream.count == QWEN4_STREAM_ENTRIES ||
        allocated > qwen4_stream_available()) {
        bool batch = ds4_gpu_commands_active();
        if (!ds4_gpu_synchronize())
            return nil;
        /* Keep arguments already acquired for this dispatch. Reuse equal
         * capacities: an oversized pinned slot could otherwise consume space
         * promised to the dispatch's remaining bindings. */
        for (unsigned i = 0; i < qwen4_stream.count; i++) {
            if (!qwen4_stream.entries[i].pinned && qwen4_stream.entries[i].capacity == allocated) {
                slot = i;
                break;
            }
        }
        if (slot == qwen4_stream.count) {
            for (unsigned i = 0; i < qwen4_stream.count;) {
                if (qwen4_stream.entries[i].pinned) {
                    i++;
                    continue;
                }
                qwen4_stream.live -= qwen4_stream.entries[i].capacity;
                ds4_gpu_tensor_free(qwen4_stream.entries[i].tensor);
                qwen4_stream.entries[i] = qwen4_stream.entries[--qwen4_stream.count];
            }
            slot = qwen4_stream.count;
        }
        if (!ds4_gpu_metal_resources_reconcile() || (batch && !ds4_gpu_begin_commands()))
            return nil;
    }
    if (slot == qwen4_stream.count) {
        if (slot == QWEN4_STREAM_ENTRIES)
            return nil;
        ds4_gpu_tensor *t = qwen4_stream_alloc(allocated);
        if (!t)
            return nil;
        qwen4_stream.count++;
        qwen4_stream.entries[slot].tensor = t;
        qwen4_stream.entries[slot].capacity = allocated;
    }
    ds4_gpu_tensor *t = qwen4_stream.entries[slot].tensor;
    qwen4_stream.entries[slot].offset = offset;
    qwen4_stream.entries[slot].bytes = allocated;
    qwen4_stream.entries[slot].expert_stride = expert_stride;
    qwen4_stream.entries[slot].ready = false;
    qwen4_stream.entries[slot].pinned = true;
    uint8_t *dst = ds4_gpu_tensor_buffer(t).contents;
    if (expert_stride) {
        if (!qwen4_stream_read_experts(dst, offset, expert_stride))
            return nil;
    } else if (!qwen4_stream_read(dst, offset, bytes))
        return nil;
    qwen4_stream.entries[slot].ready = true;
    return ds4_gpu_tensor_buffer(t);
}
/* The retained owner keeps quota, storage and readiness together across I/O
 * failures. Region lookup never exposes a partially read allocation. */
static id<MTLBuffer> qwen4_stream_resident_buffer(qwen4_stream_resident *region) {
    if (!region->tensor) {
        qwen4_stream.reserved -= region->bytes;
        ds4_gpu_tensor *tensor = qwen4_stream_alloc(region->bytes);
        if (!tensor) {
            qwen4_stream.reserved += region->bytes;
            return nil;
        }
        region->tensor = tensor;
    }
    if (!region->ready) {
        /* Keep failed storage owned, since freeing can defer reclamation
         * behind unrelated GPU work. Retry in place; never expose partial
         * bytes or allocate another buffer for the same reservation. */
        if (!qwen4_stream_read(ds4_gpu_tensor_buffer(region->tensor).contents,
                               region->offset, region->bytes))
            return nil;
        region->ready = true;
    }
    return ds4_gpu_tensor_buffer(region->tensor);
}
static id<MTLBuffer> qwen35_stream_layer_buffer(unsigned layer) {
    ds4_gpu_weight_region region = qwen4_stream.layers[layer];
    unsigned slot = layer % 2;
    if (qwen4_stream.layer_slots[slot].ready &&
        qwen4_stream.layer_slots[slot].offset == region.offset &&
        qwen4_stream.layer_slots[slot].bytes == region.bytes) {
        qwen4_stream.layer_slots[slot].pinned = true;
        return ds4_gpu_tensor_buffer(qwen4_stream.layer_slots[slot].tensor);
    }
    /* Arguments acquired by the current dispatch cannot be displaced. Once
     * encoded, only this slot's last consumer must finish before a CPU write.
     * The next queued layer can keep executing while the reader refills it. */
    if (qwen4_stream.layer_slots[slot].pinned) return nil;
    if (g_batch_cb && g_batch_has_work && !ds4_gpu_flush_commands()) return nil;
    id<MTLCommandBuffer> consumer = qwen35_stream_last_use[slot];
    if (consumer) {
        if (consumer == g_batch_cb && !ds4_gpu_flush_commands()) return nil;
        if (consumer.status < MTLCommandBufferStatusCommitted) return nil;
        [consumer waitUntilCompleted];
        /* The pending queue records profiling once at its normal drain. A
         * partial wait must not retire unrelated expert-cache consumers. */
        if (consumer.status == MTLCommandBufferStatusError) {
            ds4_gpu_invalidate_completion_counters();
            return nil;
        }
        qwen35_stream_last_use[slot] = nil;
    }
    if (!qwen4_stream.layer_slots[slot].tensor) {
        qwen4_stream.reserved -= qwen4_stream.layer_capacity;
        ds4_gpu_tensor *tensor = qwen4_stream_alloc(qwen4_stream.layer_capacity);
        if (!tensor) {
            qwen4_stream.reserved += qwen4_stream.layer_capacity;
            return nil;
        }
        qwen4_stream.layer_slots[slot].tensor = tensor;
    }
    qwen4_stream.layer_slots[slot].offset = region.offset;
    qwen4_stream.layer_slots[slot].bytes = region.bytes;
    qwen4_stream.layer_slots[slot].ready = false;
    if (!qwen4_stream_read(ds4_gpu_tensor_buffer(qwen4_stream.layer_slots[slot].tensor).contents,
                           region.offset, region.bytes)) return nil;
    qwen4_stream.layer_slots[slot].ready = true;
    qwen4_stream.layer_slots[slot].pinned = true;
    return ds4_gpu_tensor_buffer(qwen4_stream.layer_slots[slot].tensor);
}
static id<MTLBuffer> qwen4_stream_range(uint64_t offset, uint64_t bytes, uint64_t *inner) {
    *inner = 0;
    for (unsigned i = 0; i < qwen4_stream.resident_count; i++) {
        qwen4_stream_resident *region = &qwen4_stream.resident[i];
        if (offset < region->offset || offset - region->offset > region->bytes ||
            bytes > region->bytes - (offset - region->offset))
            continue;
        id<MTLBuffer> buffer = qwen4_stream_resident_buffer(region);
        if (buffer)
            *inner = offset - region->offset;
        return buffer;
    }
    for (unsigned i = 0; qwen4_stream.layer_transport && i < qwen4_stream.layer_count; i++) {
        ds4_gpu_weight_region region = qwen4_stream.layers[i];
        if (offset < region.offset || offset - region.offset > region.bytes ||
            bytes > region.bytes - (offset - region.offset)) continue;
        id<MTLBuffer> buffer = qwen35_stream_layer_buffer(i);
        if (buffer) *inner = offset - region.offset;
        return buffer;
    }
    return qwen4_stream_load(offset, bytes, 0);
}
static bool qwen4_stream_selection(const ds4_gpu_tensor *selection, unsigned tokens, unsigned slots,
                                   unsigned domain) {
    if (qwen4_stream.selection == selection && qwen4_stream.tokens == tokens &&
        qwen4_stream.slots == slots && qwen4_stream.expert_domain == domain)
        return true;
    uint64_t pairs = (uint64_t)tokens * slots;
    if (!tokens || !slots || !domain || domain > QWEN4_STREAM_EXPERTS || qwen4_stream.selection ||
        pairs > ds4_gpu_tensor_bytes(selection) / 4)
        return false;
    bool batch = ds4_gpu_commands_active();
    if (!ds4_gpu_synchronize())
        return false;
    const int32_t *source =
        (const int32_t *)((const char *)ds4_gpu_tensor_buffer(selection).contents +
                          ds4_gpu_tensor_offset(selection));
    int32_t remap[QWEN4_STREAM_EXPERTS];
    uint32_t ids[QWEN4_STREAM_EXPERTS];
    unsigned count = 0;
    for (unsigned i = 0; i < domain; i++)
        remap[i] = -1;
    bool valid = true;
    for (uint64_t i = 0; i < pairs; i++) {
        int32_t id = source[i];
        if (id < 0 || (unsigned)id >= domain) {
            fprintf(stderr, "ds4: SSD router ID %d outside domain %u at pair %llu\n", id, domain,
                    (unsigned long long)i);
            valid = false;
            break;
        }
        if (remap[id] < 0) {
            remap[id] = count;
            ids[count++] = id;
        }
    }
    ds4_gpu_tensor *compact = valid ? qwen4_stream_alloc(pairs * 4) : NULL;
    if (compact) {
        int32_t *destination = ds4_gpu_tensor_buffer(compact).contents;
        for (uint64_t i = 0; i < pairs; i++)
            destination[i] = remap[source[i]];
        memcpy(qwen4_stream.ids, ids, count * sizeof(ids[0]));
        qwen4_stream.experts = count;
        qwen4_stream.expert_domain = domain;
        qwen4_stream.selection = selection;
        qwen4_stream.compact = compact;
        qwen4_stream.tokens = tokens;
        qwen4_stream.slots = slots;
    }
    bool resumed = !batch || ds4_gpu_begin_commands();
    return compact && resumed;
}
/* The compact layout and remapped IDs are one owner's invariant. Callers get
 * an ordinary binding and never inspect the cache's internal representation. */
static id<MTLBuffer> qwen4_stream_expert_binding(uint64_t offset, uint64_t expert_bytes,
                                                 unsigned domain, const ds4_gpu_tensor **selection,
                                                 unsigned tokens, unsigned slots,
                                                 uint32_t *packed) {
    if (!qwen4_stream_selection(*selection, tokens, slots, domain)) {
        fprintf(stderr, "ds4: SSD router binding failed (tokens=%u slots=%u domain=%u)\n", tokens,
                slots, domain);
        return nil;
    }
    id<MTLBuffer> buffer = qwen4_stream_load(offset, expert_bytes * domain, expert_bytes);
    if (!buffer)
        fprintf(
            stderr,
            "ds4: SSD expert read failed offset=%llu stride=%llu selected=%u live=%llu cap=%llu\n",
            (unsigned long long)offset, (unsigned long long)expert_bytes, qwen4_stream.experts,
            (unsigned long long)qwen4_stream.live, (unsigned long long)qwen4_stream.limit);
    if (buffer) {
        *selection = qwen4_stream.compact;
        *packed = qwen4_stream.experts;
    }
    return buffer;
}

/* Split token rows before binding a union that cannot fit. Every chunk keeps
 * the original slot order and uses the same per-row arithmetic kernels. */
static unsigned qwen4_stream_moe_token_cap(uint64_t expert_bytes, unsigned domain, unsigned slots,
                                           uint64_t shared_bytes, unsigned matrices,
                                           unsigned tokens) {
    uint64_t fixed = shared_bytes * matrices;
    uint64_t limit = qwen4_stream.limit;
    if (!slots || fixed >= limit)
        return 0;
    uint64_t union_bytes = expert_bytes * domain * matrices;
    if (union_bytes <= limit - fixed && (uint64_t)tokens * slots * 4 <= limit - fixed - union_bytes)
        return tokens;
    uint64_t per_token = slots * (expert_bytes * matrices + 4);
    uint64_t cap = (limit - fixed) / per_token;
    return cap < tokens ? (unsigned)cap : tokens;
}
