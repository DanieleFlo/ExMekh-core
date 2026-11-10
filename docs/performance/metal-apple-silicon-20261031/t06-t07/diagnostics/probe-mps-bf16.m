#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdint.h>
int main(void) { @autoreleasepool { @try {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) return 2;
    uint16_t input[16];for(unsigned i=0;i<16;i++) input[i]=0x3f80;
    id<MTLBuffer> x=[device newBufferWithBytes:input length:sizeof(input) options:MTLResourceStorageModeShared];
    id<MTLBuffer> w=[device newBufferWithBytes:input length:sizeof(input) options:MTLResourceStorageModeShared];
    id<MTLBuffer> y=[device newBufferWithLength:8 options:MTLResourceStorageModeShared];
    MPSMatrixDescriptor *xd=[MPSMatrixDescriptor matrixDescriptorWithRows:2 columns:8 rowBytes:16 dataType:MPSDataTypeBFloat16];
    MPSMatrixDescriptor *yd=[MPSMatrixDescriptor matrixDescriptorWithRows:2 columns:2 rowBytes:4 dataType:MPSDataTypeBFloat16];
    MPSMatrix *xm=[[MPSMatrix alloc]initWithBuffer:x descriptor:xd];
    MPSMatrix *wm=[[MPSMatrix alloc]initWithBuffer:w descriptor:xd];
    MPSMatrix *ym=[[MPSMatrix alloc]initWithBuffer:y descriptor:yd];
    MPSMatrixMultiplication *op=[[MPSMatrixMultiplication alloc]initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:2 resultColumns:2 interiorColumns:8 alpha:1 beta:0];
    op.options=MPSKernelOptionsNone;
    id<MTLCommandBuffer> command=[[device newCommandQueue]commandBuffer];
    [op encodeToCommandBuffer:command leftMatrix:xm rightMatrix:wm resultMatrix:ym];
    [command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"command: %s\n",command.error.localizedDescription.UTF8String);return 3;}
    uint16_t *result=y.contents;
    for(unsigned i=0;i<4;i++)if(result[i]!=0x4100){fprintf(stderr,"value %u: %04x\n",i,result[i]);return 4;}
    puts("Native BF16 MPSMatrix 2x8 dot =8 exactly");return 0;
} @catch(NSException *e){fprintf(stderr,"%s: %s\n",e.name.UTF8String,e.reason.UTF8String);return 5;}}}
