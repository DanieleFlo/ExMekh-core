/* Flash expert arithmetic on original GGUF bytes and float activations.
 * The scalar row readers in ds4.c remain the independent FP64 oracle. */
#ifndef DS4_QWEN4_CPU_H
#define DS4_QWEN4_CPU_H

#include <stdint.h>

/* Storage includes Q2_K padding in the final 256-element block. */
static inline uint64_t ds4_qwen4_cpu_row_bytes(unsigned type, unsigned n) {
    if (type == 16 && n % 256u == 0) return (uint64_t)(n / 256u) * 66u;
    if (type == 10 && n % 32u == 0) return (uint64_t)((n + 255u) / 256u) * 84u;
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
void ds4_qwen4_cpu_expert_run(const uint8_t *gate, const uint8_t *up,
        const uint8_t *down, unsigned gu_type, unsigned down_type,
        const float *x, unsigned E, unsigned F, float *mid, float *out);
void ds4_qwen4_cpu_expert_rows_run(const uint8_t *gate, const uint8_t *up,
        const uint8_t *down, unsigned gu_type, unsigned down_type,
        const float *const *x, unsigned rows, unsigned E, unsigned F,
        float *mid, float *const *out);
void ds4_qwen4_cpu_gate_up_rows_run(const uint8_t *gate, const uint8_t *up,
        unsigned type, const float *const *x, unsigned rows, unsigned E, unsigned F,
        float *const *gate_out, float *const *up_out);
void ds4_qwen4_cpu_down_rows_run(const uint8_t *down, unsigned type,
        const float *const *mid, unsigned rows, unsigned E, unsigned F, float *const *out);
#ifdef __cplusplus
}
#endif
#endif
