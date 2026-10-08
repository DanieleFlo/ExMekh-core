/* Independent ordered matmul oracle: different weights reuse both DMA slots.
 * This catches an overwritten in-flight slot or a stale layer-to-slot map. */
#include "../ds4_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void run(unsigned budget) {
    float *map = calloc(7,256);
    assert(map);
    ds4_qwen_weight_span spans[7];
    for (unsigned layer=0;layer<7;++layer) {
        spans[layer]=(ds4_qwen_weight_span){layer*256u,64u,layer==6u?UINT32_MAX:layer};
        float *w=(float *)((char *)map+layer*256u);
        for (unsigned j=0;j<16;++j) w[j]=(float)((layer+1u)*(j+1u))/16.0f;
    }
    assert(ds4_gpu_qwen35_weight_stream_init(map,7u*256u,spans,7,6,budget));
    assert(ds4_gpu_qwen35_weight_stream_active(map));
    /* A graph that would exhaust VRAM is rejected before allocating it;
     * rejection must leave the engine reusable. */
    assert(!ds4_gpu_qwen35_weight_session_prepare(map, UINT64_MAX, 0));
    assert(!ds4_gpu_qwen35_weight_session_prepare(map, 0, UINT64_MAX));
    assert(ds4_gpu_qwen35_weight_session_prepare(map, 32u, 64u));
    assert(ds4_gpu_qwen35_weight_session_prepare(map, 0, 0));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(16), *y=ds4_gpu_tensor_alloc(16);
    assert(x && y);
    const float input[4]={1,-2,3,-4};
    assert(ds4_gpu_tensor_write(x,0,input,sizeof(input)));
    if (budget==512u)
        assert(!ds4_gpu_matmul_f32_tensor(y,map,7u*256u,0,4,4,x,1));
    for (unsigned cycle=0;cycle<12;++cycle) {
        assert(ds4_gpu_qwen35_weight_forward_begin(map));
        for (unsigned layer=0;layer<6;++layer) {
            assert(ds4_gpu_qwen35_weight_layer_begin(map,layer));
            assert(ds4_gpu_matmul_f32_tensor(y,map,7u*256u,layer*256u,4,4,x,1));
            assert(ds4_gpu_qwen35_weight_layer_end(map,layer));
            float actual[4]; assert(ds4_gpu_tensor_read(y,0,actual,sizeof(actual)));
            const float *w=(const float *)((const char *)map+layer*256u);
            for (unsigned r=0;r<4;++r) {
                float expected=0;
                for (unsigned c=0;c<4;++c) expected+=w[r*4u+c]*input[c];
                assert(actual[r]==expected);
            }
        }
        assert(ds4_gpu_matmul_f32_tensor(y,map,7u*256u,6u*256u,4,4,x,1));
        assert(ds4_gpu_end_commands());
    }
    /* Validation errors must preserve the active plan. */
    spans[0].bytes=8u*256u;
    assert(!ds4_gpu_qwen35_weight_stream_init(map,7u*256u,spans,7,6,budget));
    assert(ds4_gpu_qwen35_weight_stream_active(map));
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);
    ds4_gpu_cleanup(); free(map);
}
int main(void) {
    setenv("DS4_CUDA_QWEN_WEIGHT_STAGE_MB","1",1);
    assert(ds4_gpu_init()); run(512); /* five remote layers, both slots wrap */
    assert(ds4_gpu_init()); run(7u*256u); /* all resident */
    puts("dense weight streaming ordered oracle: PASS");
    return 0;
}
