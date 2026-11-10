// Isolated experiment: BF16 storage and matrix operands, FP32 accumulation.
// No production kernel depends on this file.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Shape { uint rows, inner, columns; };

kernel void direct_bf16(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant Shape &shape [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint subgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat a[32 * 32], b[32 * 32];
    threadgroup float result[32 * 32];
    uint row = tile.y * 32, col = tile.x * 32;
    uint subrow = (subgroup / 2) * 16, subcol = (subgroup % 2) * 16;
    simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c01 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c10 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c11 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint start = 0; start < shape.inner; start += 32) {
        for (uint i = tid; i < 1024; i += 128) {
            uint r = i / 32, c = i % 32;
            a[i] = row + r < shape.rows && start + c < shape.inner
                 ? x[(row + r) * shape.inner + start + c] : bfloat(0.0f);
            b[i] = col + c < shape.columns && start + r < shape.inner
                 ? w[(col + c) * shape.inner + start + r] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_bfloat8x8 a0, a1, b0, b1;
            simdgroup_load(a0, a + subrow * 32 + k, 32);
            simdgroup_load(a1, a + (subrow + 8) * 32 + k, 32);
            simdgroup_load(b0, b + k * 32 + subcol, 32);
            simdgroup_load(b1, b + k * 32 + subcol + 8, 32);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, result + subrow * 32 + subcol, 32);
    simdgroup_store(c01, result + subrow * 32 + subcol + 8, 32);
    simdgroup_store(c10, result + (subrow + 8) * 32 + subcol, 32);
    simdgroup_store(c11, result + (subrow + 8) * 32 + subcol + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 128) {
        uint r = i / 32, c = i % 32;
        if (row + r < shape.rows && col + c < shape.columns)
            y[(row + r) * shape.columns + col + c] = result[i];
    }
}
