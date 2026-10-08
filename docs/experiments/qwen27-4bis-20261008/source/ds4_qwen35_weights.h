#ifndef DS4_QWEN35_WEIGHTS_H
#define DS4_QWEN35_WEIGHTS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* UINT32_MAX marks embedding/output or another permanent tensor. */
typedef struct {
    uint64_t offset, bytes;
    uint32_t layer;
} ds4_qwen_weight_span;
int ds4_gpu_qwen35_weight_stream_init(const void *map, uint64_t size,
        const ds4_qwen_weight_span *spans, uint32_t count, uint32_t layers,
        uint64_t resident_budget);
int ds4_gpu_qwen35_weight_stream_active(const void *map);
uint64_t ds4_gpu_qwen35_weight_resident_bytes(const void *map);
/* Preallocate shared scratch, admit upcoming graph bytes, then check again
 * with zero arguments after graph allocation. No KV or recurrent offload. */
int ds4_gpu_qwen35_weight_session_prepare(const void *map,
        uint64_t scratch_bytes, uint64_t graph_bytes);
int ds4_gpu_qwen35_weight_forward_begin(const void *map);
int ds4_gpu_qwen35_weight_layer_begin(const void *map, uint32_t layer);
int ds4_gpu_qwen35_weight_layer_end(const void *map, uint32_t layer);
#ifdef __cplusplus
}
#endif
#endif
