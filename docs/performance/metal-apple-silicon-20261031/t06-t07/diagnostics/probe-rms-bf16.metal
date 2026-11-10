// Isolated RMSNorm variants; no production dependency.
#include <metal_stdlib>
using namespace metal;
#define VARIANT 0
#define ROUND_NORMALIZED 1
struct Shape { uint rows, inner, columns; };
inline float round_bf16(float f) {
    uint u = as_type<uint>(f);
    u += 0x7fff + ((u >> 16) & 1);
    return as_type<float>(u & 0xffff0000);
}
kernel void norm_bf16(device const ushort *x [[buffer(0)]],
                     device const ushort *w [[buffer(1)]],
                     device float *y [[buffer(2)]],
                     constant Shape &shape [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
#if VARIANT == 4 || VARIANT == 5
    if (tid>=32) return;
    float levels[4]={};
    uint count=shape.inner/32;
    // Cascade 16 iterations at a time, keeping 32 independent FP32 sums.
    for(uint i=0;i<count;i++) {
        float z=as_type<float>(uint(x[row*shape.inner+i*32+tid])<<16);
        levels[0]+=z*z;
        if((i+1)%16==0) {
            for(uint level=1;level<4;level++) {
                levels[level]+=levels[level-1];levels[level-1]=0;
                if((i+1)&(15u<<(level*4))) break;
            }
        }
    }
    float value=((levels[0]+levels[1])+levels[2])+levels[3];
    uint pairs=(shape.inner/4-count*8)/2;
    if(tid<8) {
        for(uint i=0;i<pairs;i++) {
            uint j=(count*8+i*2+tid/4)*4+tid%4;
            float z=as_type<float>(uint(x[row*shape.inner+j])<<16);
            value+=z*z;
        }
    }
    float grouped=simd_shuffle(value,tid%8);
    grouped+=simd_shuffle(value,tid%8+8);
    grouped+=simd_shuffle(value,tid%8+16);
    grouped+=simd_shuffle(value,tid%8+24);
    float total=0;
    for(uint j=(count*8+pairs*2)*4;j<shape.inner;j++) {
        float z=as_type<float>(uint(x[row*shape.inner+j])<<16);
        total+=z*z;
    }
    for(uint lane=0;lane<8;lane++) total+=simd_shuffle(grouped,lane);
#if VARIANT == 5
    float inverse=pow(total/shape.inner+1e-6f,-0.5f);
#else
    float inverse=1.0f/sqrt(total/shape.inner+1e-6f);
#endif
    for(uint j=tid;j<shape.inner;j+=32) {
        float z=as_type<float>(uint(x[row*shape.inner+j])<<16)*inverse;
#if ROUND_NORMALIZED
        z=round_bf16(z);
#endif
        y[row*shape.inner+j]=round_bf16(z*as_type<float>(uint(w[j])<<16));
    }
#else
    threadgroup float partial[8];
    float sum = 0;
    for (uint j=tid; j<shape.inner; j+=256) {
        float z=as_type<float>(uint(x[row*shape.inner+j])<<16);
        sum += z*z;
    }
    sum=simd_sum(sum);
    if (!(tid&31)) partial[tid>>5]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=tid<8 ? partial[tid] : 0;
    sum=simd_sum(sum);
    if (!tid) partial[0]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance=partial[0]/shape.inner+1e-6f;
#if VARIANT == 0
    float denominator=sqrt(variance);
#elif VARIANT == 1
    float factor=1.0f/sqrt(variance);
#elif VARIANT == 2
    float factor=rsqrt(variance);
#else
    float factor=pow(variance,-0.5f);
#endif
    for (uint j=tid; j<shape.inner; j+=256) {
        float z=as_type<float>(uint(x[row*shape.inner+j])<<16);
#if VARIANT == 0
        z /= denominator;
#else
        z *= factor;
#endif
#if ROUND_NORMALIZED
        z=round_bf16(z);
#endif
        z *= as_type<float>(uint(w[j])<<16);
        y[row*shape.inner+j]=round_bf16(z);
    }
#endif
}
