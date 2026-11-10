// Accumulation-order diagnosis, not a production GEMM.
#include <metal_stdlib>
using namespace metal;
#define VARIANT 0
struct Shape { uint rows, inner, columns; };
kernel void direct_bf16(device const ushort *x [[buffer(0)]],
                       device const ushort *w [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant Shape &shape [[buffer(3)]],
                       uint2 tile [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    for (uint i=tid; i<1024; i+=128) {
        uint row=tile.y*32+i/32, col=tile.x*32+i%32;
        if (row>=shape.rows || col>=shape.columns) continue;
        float accum[32]={};
        float correction=0;
#if VARIANT == 6
        uint aligned=shape.inner/32*32;
        for (uint k=0; k<aligned; k+=32) {
            #pragma unroll
            for (uint lane=0;lane<32;lane++) {
                float a=as_type<float>(uint(x[row*shape.inner+k+lane])<<16);
                float b=as_type<float>(uint(w[col*shape.inner+k+lane])<<16);
                accum[lane]=fma(a,b,accum[lane]);
            }
        }
        for (uint span=16;span>=4;span/=2)
            for (uint lane=0;lane<span;lane++) accum[lane]+=accum[lane+span];
        float result=(accum[0]+accum[1])+(accum[2]+accum[3]);
        float tail[4]={};
        uint aligned8=shape.inner/8*8;
        for (uint k=aligned;k<aligned8;k+=8)
            for (uint lane=0;lane<4;lane++) {
                float a0=as_type<float>(uint(x[row*shape.inner+k+lane])<<16);
                float b0=as_type<float>(uint(w[col*shape.inner+k+lane])<<16);
                float a1=as_type<float>(uint(x[row*shape.inner+k+lane+4])<<16);
                float b1=as_type<float>(uint(w[col*shape.inner+k+lane+4])<<16);
                tail[lane]=fma(a1,b1,fma(a0,b0,tail[lane]));
            }
        result+=(tail[0]+tail[1])+(tail[2]+tail[3]);
        for (uint k=aligned8;k<shape.inner;k++) {
            float a=as_type<float>(uint(x[row*shape.inner+k])<<16);
            float b=as_type<float>(uint(w[col*shape.inner+k])<<16);
            result=fma(a,b,result);
        }
        y[row*shape.columns+col]=result;
#else
        for (uint kk=0; kk<shape.inner; kk++) {
            uint k=VARIANT==1 ? shape.inner-1-kk : kk;
            float a=as_type<float>(uint(x[row*shape.inner+k])<<16);
            float b=as_type<float>(uint(w[col*shape.inner+k])<<16);
#if VARIANT == 2 || VARIANT == 3
            accum[k%4]=fma(a,b,accum[k%4]);
#elif VARIANT == 4
            accum[k%8]=fma(a,b,accum[k%8]);
#elif VARIANT == 5
            float term=a*b-correction;
            float total=accum[0]+term;
            correction=(total-accum[0])-term;
            accum[0]=total;
#else
            accum[0]=fma(a,b,accum[0]);
#endif
        }
#if VARIANT == 2
        y[row*shape.columns+col]=((accum[0]+accum[1])+accum[2])+accum[3];
#elif VARIANT == 3
        y[row*shape.columns+col]=(accum[0]+accum[1])+(accum[2]+accum[3]);
#elif VARIANT == 4
        y[row*shape.columns+col]=((accum[0]+accum[1])+(accum[2]+accum[3]))+
                                   ((accum[4]+accum[5])+(accum[6]+accum[7]));
#else
        y[row*shape.columns+col]=accum[0];
#endif
#endif
    }
}
