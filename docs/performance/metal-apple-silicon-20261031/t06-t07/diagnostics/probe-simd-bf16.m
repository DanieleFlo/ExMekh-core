#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { uint32_t rows, inner, columns; } Shape;

static uint32_t dimension(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || !*text || *end || !value || value > 65536) return 0;
    return (uint32_t)value;
}

int main(int argc, const char **argv) { @autoreleasepool {
    if (argc != 8 && argc != 9) {
        fprintf(stderr, "usage: probe shader x.bf16 w.bf16 y.f32 rows inner columns [norm_bf16]\n");
        return 2;
    }
    Shape shape = {dimension(argv[5]), dimension(argv[6]), dimension(argv[7])};
    uint64_t xb = (uint64_t)shape.rows * shape.inner * 2;
    bool norm = argc == 9 && strcmp(argv[8], "norm_bf16") == 0;
    uint64_t wb = norm ? (uint64_t)shape.inner * 2 : (uint64_t)shape.columns * shape.inner * 2;
    uint64_t yb = (uint64_t)shape.rows * (norm ? shape.inner : shape.columns) * 4;
    if (!xb || !wb || !yb || xb + wb + yb > 128 * 1024 * 1024) return 2;
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                            encoding:NSUTF8StringEncoding error:&error];
    NSData *xd = [NSData dataWithContentsOfFile:@(argv[2])];
    NSData *wd = [NSData dataWithContentsOfFile:@(argv[3])];
    if (!source || xd.length != xb || wd.length != wb) {
        fprintf(stderr, "invalid source or operand bytes\n"); return 3;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 4;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    id<MTLFunction> function = [library newFunctionWithName:norm ? @"norm_bf16" : @"direct_bf16"];
    id<MTLComputePipelineState> pipeline = function ?
        [device newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < (norm ? 256 : 128)) {
        fprintf(stderr, "unsupported pipeline: %s\n", error.localizedDescription.UTF8String);
        return 5;
    }
    id<MTLBuffer> x = [device newBufferWithBytes:xd.bytes length:xb options:MTLResourceStorageModeShared];
    id<MTLBuffer> w = [device newBufferWithBytes:wd.bytes length:wb options:MTLResourceStorageModeShared];
    id<MTLBuffer> y = [device newBufferWithLength:yb options:MTLResourceStorageModeShared];
    if (!x || !w || !y) return 6;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:w offset:0 atIndex:1];
    [encoder setBuffer:y offset:0 atIndex:2];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    [encoder dispatchThreadgroups:norm ? MTLSizeMake(shape.rows,1,1) :
                                        MTLSizeMake((shape.columns+31)/32, (shape.rows+31)/32, 1)
             threadsPerThreadgroup:MTLSizeMake(norm ? 256 : 128,1,1)];
    [encoder endEncoding];
    [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "command: %s\n", command.error.localizedDescription.UTF8String); return 7;
    }
    if (![[NSData dataWithBytesNoCopy:y.contents length:yb freeWhenDone:NO]
          writeToFile:@(argv[4]) options:NSDataWritingAtomic error:&error]) return 8;
    printf("{\"gpu_seconds\":%.9g,\"operand_and_output_bytes\":%llu,"
           "\"threadgroup_bytes\":%lu,\"device\":\"%s\"}\n",
           command.GPUEndTime-command.GPUStartTime, (unsigned long long)(xb+wb+yb),
           (unsigned long)pipeline.staticThreadgroupMemoryLength,device.name.UTF8String);
    return 0;
}}
