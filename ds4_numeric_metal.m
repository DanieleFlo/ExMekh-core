/* Native suite owner. Shared buffers have ordinary addresses for view slicing;
 * every arithmetic operation is encoded on Metal. No CPU numeric fallback. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "ds4_numeric_internal.h"
#include "ds4_suite_text.h"
#include "ds4_metal_device.h"
#include "ds4_numeric_metal_args.h"
#include "build/numeric_metal_source.inc"
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "ds4_file_reader.h"

@interface DS4MetalNumeric : NSObject {
  @private
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    NSMutableDictionary<NSValue *, id<MTLBuffer>> *_buffers;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *_pipelines;
    id<MTLBuffer> _dummy;
    ds4_resources *_resources;
    ds4_numeric *_context;
    void *_upload;
    uint64_t _uploadPage;
    ds4_resource *_uploadAccount;
}
- (instancetype)initWithResources:(ds4_resources *)resources owner:(uint64_t)owner
                            status:(ds4_status *)status;
- (void)bindContext:(ds4_numeric *)context;
- (ds4_status)allocateBytes:(uint64_t)bytes result:(void **)result;
- (ds4_status)releaseAddress:(void *)address;
- (ds4_status)copyTo:(void *)destination
                from:(const void *)source
               bytes:(uint64_t)bytes
                kind:(int)kind;
- (ds4_status)readTo:(void *)destination descriptor:(int)fd
              offset:(uint64_t)offset bytes:(uint64_t)bytes;
- (ds4_status)allocateTemporary:(uint64_t)count
                           type:(uint32_t)type
                         result:(ds4_num_buffer **)result;
- (ds4_status)writeTemporary:(ds4_num_buffer *)buffer data:(const void *)data bytes:(uint64_t)bytes;
- (ds4_status)runKernel:(NSString *)name
              arguments:(ds4_nm_args)arguments
                  views:(const ds4_num_view *)views
                  count:(uint32_t)count
                   work:(uint64_t)work
                 groups:(bool)groups;
- (ds4_status)projectBFloat16:(ds4_num_view)y
                       left:(ds4_num_view)x
                      right:(ds4_num_view)w
                       bias:(ds4_num_view)bias
                       rows:(uint32_t)rows
                      inner:(uint32_t)inner
                    columns:(uint32_t)columns
             transposeInput:(bool)transposeInput;
- (ds4_status)convolve:(ds4_num_view)y
                 input:(ds4_num_view)x
                weight:(ds4_num_view)w
                  bias:(ds4_num_view)bias
            descriptor:(const ds4_num_convolution *)shape;
- (ds4_status)multiply:(ds4_num_view)y
                  left:(ds4_num_view)x
                 right:(ds4_num_view)w
                 batch:(uint32_t)batch
                  rows:(uint32_t)rows
                 inner:(uint32_t)inner
               columns:(uint32_t)columns
         transposeLeft:(bool)transposeLeft
        transposeRight:(bool)transposeRight;
- (ds4_status)setBudgetMargin:(uint64_t)margin;
- (ds4_status)refreshAvailability;
- (ds4_status)closeUpload;
@end

@implementation DS4MetalNumeric
- (instancetype)initWithResources:(ds4_resources *)resources owner:(uint64_t)owner
                            status:(ds4_status *)status {
    self = [super init];
    if (!self) {
        *status = DS4_STATUS_NOMEM;
        return nil;
    }
    _resources = resources;
    _device = MTLCreateSystemDefaultDevice();
    if (!_device || !_device.hasUnifiedMemory) {
        *status = DS4_STATUS_UNSUPPORTED;
        return nil;
    }
    _queue = [_device newCommandQueue];
    _buffers = [NSMutableDictionary new];
    _pipelines = [NSMutableDictionary new];
    _dummy = [_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    if (!_queue || !_dummy) {
        *status = DS4_STATUS_NOMEM;
        return nil;
    }
    long page = sysconf(_SC_PAGESIZE);
    if (page < (long)sizeof(void *) || page > DS4_FILE_READ_CHUNK_BYTES || (page & (page - 1))) {
        *status = DS4_STATUS_NOT_READY;
        return nil;
    }
    _uploadPage = (uint64_t)page;
    *status = ds4_resource_register_host(resources, owner, DS4_FILE_READ_CHUNK_BYTES, &_uploadAccount);
    if (*status) return nil;
    if (posix_memalign(&_upload, (size_t)page, DS4_FILE_READ_CHUNK_BYTES)) {
        *status = DS4_STATUS_NOMEM;
        return nil;
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *))
        options.mathMode = MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError *error = nil;
    id<MTLLibrary> library =
        [_device newLibraryWithSource:[NSString stringWithUTF8String:numeric_metal_source]
                              options:options
                                error:&error];
    if (!library) {
        fprintf(stderr, DS4_SUITE_METAL_COMPILE, error.localizedDescription.UTF8String);
        *status = DS4_STATUS_UNSUPPORTED;
        return nil;
    }
    for (NSString *name in @[
             @"nm_point", @"nm_norm", @"nm_channel", @"nm_rope", @"nm_reorder",
             @"nm_pool", @"nm_nearest", @"nm_attention", @"nm_epilogue", @"nm_conv",
             @"nm_bf16_linear", @"nm_bf16_linear_transposed", @"nm_bf16_rms", @"nm_bf16_layernorm",
             @"nm_bf16_linear_tile", @"nm_bf16_conv2d_1", @"nm_bf16_conv2d_3",
             @"nm_ordered_softmax", @"nm_bf16_attention", @"nm_bf16_conv", @"nm_bf16_channel_rms"
         ]) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        id<MTLComputePipelineState> pipeline =
            function ? [_device newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256 ||
            pipeline.threadExecutionWidth != 32) {
            // Ordered generic kernels cover shapes without optional tiles.
            if ([name isEqualToString:@"nm_bf16_linear_tile"] ||
                [name isEqualToString:@"nm_bf16_conv2d_1"] ||
                [name isEqualToString:@"nm_bf16_conv2d_3"])
                continue;
            *status = DS4_STATUS_UNSUPPORTED;
            return nil;
        }
        _pipelines[name] = pipeline;
    }
    *status = DS4_STATUS_OK;
    return self;
}
- (void)bindContext:(ds4_numeric *)context {
    _context = context;
    context->backend_bytes = DS4_FILE_READ_CHUNK_BYTES;
    context->peak_bytes = DS4_FILE_READ_CHUNK_BYTES;
}
- (ds4_status)closeUpload {
    ds4_status status = ds4_resource_release(_uploadAccount);
    if (status) return status;
    _uploadAccount = NULL;
    free(_upload);
    _upload = NULL;
    return DS4_STATUS_OK;
}
- (void)dealloc {
    /* Constructor failures have no GPU consumers or externally visible owner. */
    (void)[self closeUpload];
}
- (id<MTLBuffer>)bufferAtAddress:(const void *)address
                           bytes:(uint64_t)bytes
                          offset:(NSUInteger *)offset {
    if (!address && !bytes) {
        *offset = 0;
        return _dummy;
    }
    uintptr_t pointer = (uintptr_t)address;
    for (NSValue *key in _buffers) {
        uintptr_t base = (uintptr_t)key.pointerValue;
        id<MTLBuffer> buffer = _buffers[key];
        if (pointer >= base && pointer - base <= buffer.length &&
            bytes <= buffer.length - (pointer - base)) {
            *offset = pointer - base;
            return buffer;
        }
    }
    return nil;
}
- (ds4_status)allocateBytes:(uint64_t)bytes result:(void **)result {
    *result = NULL;
    if (bytes > _device.maxBufferLength)
        return DS4_STATUS_NOMEM;
    id<MTLBuffer> buffer = [_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer)
        return DS4_STATUS_NOMEM;
    *result = buffer.contents;
    _buffers[[NSValue valueWithPointer:*result]] = buffer;
    return DS4_STATUS_OK;
}
- (ds4_status)refreshAvailability {
    ds4_metal_memory memory = {0};
    if (!ds4_metal_owned_device_memory(_device, &memory))
        return DS4_STATUS_NOT_READY;
    return ds4_metal_resources_reconcile(_resources, &memory);
}
- (ds4_status)releaseAddress:(void *)address {
    NSValue *key = [NSValue valueWithPointer:address];
    if (!_buffers[key])
        return DS4_STATUS_INVALID;
    [_buffers removeObjectForKey:key];
    [self refreshAvailability];
    return DS4_STATUS_OK;
}
- (ds4_status)copyTo:(void *)destination
                from:(const void *)source
               bytes:(uint64_t)bytes
                kind:(int)kind {
    NSUInteger offset;
    /* Commands complete before returning. Unified storage needs no staging;
     * only the device side of a host/device transfer belongs to this owner. */
    if ((kind != DS4_NUM_COPY_TO_DEVICE && ![self bufferAtAddress:source
                                                            bytes:bytes
                                                           offset:&offset]) ||
        (kind != DS4_NUM_COPY_FROM_DEVICE && ![self bufferAtAddress:destination
                                                              bytes:bytes
                                                             offset:&offset]))
        return DS4_STATUS_INVALID;
    memmove(destination, source, bytes);
    return DS4_STATUS_OK;
}
- (ds4_status)readTo:(void *)destination descriptor:(int)fd
              offset:(uint64_t)offset bytes:(uint64_t)bytes {
    NSUInteger bufferOffset;
    if (![self bufferAtAddress:destination bytes:bytes offset:&bufferOffset])
        return DS4_STATUS_INVALID;
    /* XNU routes a misaligned NOCACHE vector through UBC. DS4B tensor offsets
     * need not be page-aligned, so read aligned windows into bounded host
     * storage first. The bundle checks identity before publishing the group. */
    if (fcntl(fd, F_NOCACHE, 1))
        return DS4_STATUS_INFERENCE;
    struct stat file;
    if (fstat(fd, &file) || file.st_size < 0 || offset > (uint64_t)file.st_size ||
        bytes > (uint64_t)file.st_size - offset)
        return DS4_STATUS_INFERENCE;
    ds4_file_reader *reader = ds4_file_reader_open(fd, (uint64_t)file.st_size, 1);
    if (!reader)
        return DS4_STATUS_NOMEM;
    bool ok = true;
    for (uint64_t done = 0; done < bytes && ok;) {
        uint64_t at = offset + done, prefix = at % _uploadPage;
        uint64_t payload = bytes - done;
        if (payload > DS4_FILE_READ_CHUNK_BYTES - prefix)
            payload = DS4_FILE_READ_CHUNK_BYTES - prefix;
        uint64_t read = (prefix + payload + _uploadPage - 1) / _uploadPage * _uploadPage;
        at -= prefix;
        if (read > (uint64_t)file.st_size - at) read = (uint64_t)file.st_size - at;
        ok = ds4_file_reader_read(reader, _upload, at, read, NULL);
        if (ok) memcpy((char *)destination + done, (char *)_upload + prefix, payload);
        done += payload;
    }
    ds4_file_reader_close(reader);
    return ok ? DS4_STATUS_OK : DS4_STATUS_INFERENCE;
}
- (ds4_status)allocateTemporary:(uint64_t)count
                           type:(uint32_t)type
                         result:(ds4_num_buffer **)result {
    return ds4_num_alloc(_context, count, type, result);
}
- (ds4_status)writeTemporary:(ds4_num_buffer *)buffer
                        data:(const void *)data
                       bytes:(uint64_t)bytes {
    return ds4_num_write(_context, ds4_num_data(buffer), data, bytes);
}
- (ds4_status)completeCommand:(id<MTLCommandBuffer>)command {
    if (!command)
        return DS4_STATUS_NOMEM;
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, DS4_SUITE_METAL_EXECUTE, command.error.localizedDescription.UTF8String);
        return DS4_STATUS_INFERENCE;
    }
    return DS4_STATUS_OK;
}
- (ds4_status)runKernel:(NSString *)name
              arguments:(ds4_nm_args)arguments
                  views:(const ds4_num_view *)views
                  count:(uint32_t)count
                   work:(uint64_t)work
                 groups:(bool)groups {
    if (work > UINT32_MAX || count >= sizeof(arguments.t) / sizeof(arguments.t[0]))
        return DS4_STATUS_UNSUPPORTED;
    id<MTLComputePipelineState> pipeline = _pipelines[name];
    if (!pipeline)
        return DS4_STATUS_INVALID;
    id<MTLCommandBuffer> command = [_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder)
        return DS4_STATUS_NOMEM;
    for (uint32_t i = 0; i < count; i++)
        arguments.t[i + 1] = views[i].type;
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:&arguments length:sizeof(arguments) atIndex:0];
    for (uint32_t i = 0; i < count; i++) {
        NSUInteger offset;
        id<MTLBuffer> buffer =
            [self bufferAtAddress:views[i].data
                            bytes:views[i].count * ds4_num_type_bytes(views[i].type)
                           offset:&offset];
        if (!buffer) {
            [encoder endEncoding];
            return DS4_STATUS_INVALID;
        }
        [encoder setBuffer:buffer offset:offset atIndex:i + 1];
    }
    MTLSize threads = MTLSizeMake(256, 1, 1), grid = MTLSizeMake(work, 1, 1);
    if (groups)
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    else
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return [self completeCommand:command];
}
- (ds4_status)projectBFloat16:(ds4_num_view)y
                       left:(ds4_num_view)x
                      right:(ds4_num_view)w
                       bias:(ds4_num_view)bias
                       rows:(uint32_t)rows
                      inner:(uint32_t)inner
                    columns:(uint32_t)columns
             transposeInput:(bool)transposeInput {
    ds4_nm_args arguments = {0};
    arguments.u[0] = rows;
    arguments.u[1] = inner;
    arguments.u[2] = columns;
    arguments.u[3] = bias.data != NULL;
    ds4_num_view views[] = {y, x, w, bias};
    const uint64_t tiles = ((uint64_t)rows + 3) / 4 * (columns / 2);
    if (!transposeInput && rows >= 64 && inner >= 1024 && columns >= 1024 &&
        inner % 32 == 0 && columns % 2 == 0 && tiles <= UINT32_MAX / 32 &&
        _pipelines[@"nm_bf16_linear_tile"])
        return [self runKernel:@"nm_bf16_linear_tile"
                     arguments:arguments views:views count:4 work:tiles * 32 groups:false];
    return [self runKernel:transposeInput ? @"nm_bf16_linear_transposed" : @"nm_bf16_linear"
                 arguments:arguments views:views count:4 work:(uint64_t)rows * columns groups:false];
}
- (ds4_status)convolve:(ds4_num_view)y
                 input:(ds4_num_view)x
                weight:(ds4_num_view)w
                  bias:(ds4_num_view)bias
            descriptor:(const ds4_num_convolution *)shape {
    // Numeric slices y to the validated output, so each axis fits this grid.
    if (y.count > UINT32_MAX)
        return DS4_STATUS_UNSUPPORTED;
    ds4_nm_args a = {0};
    a.n[0] = y.count;
    a.u[0] = shape->in_channels;
    a.u[1] = shape->out_channels;
    a.u[2] = shape->groups;
    a.u[24] = bias.data != NULL;
    for (uint32_t i = 0; i < 3; i++) {
        a.u[3 + i] = shape->input[i];
        a.u[6 + i] = shape->kernel[i];
        a.u[9 + i] = shape->stride[i];
        a.u[12 + i] = shape->dilation[i];
        a.u[15 + i] = shape->pad_before[i];
        a.u[21 + i] = (uint32_t)ds4_num_conv_axis_output(shape, i);
    }
    ds4_num_view views[] = {y, x, w, bias};
    bool bf16 = x.type == DS4_NUM_BF16;
    uint32_t kernel = shape->kernel[1];
    if (bf16 && shape->input[0] == 1 && shape->kernel[0] == 1 && a.u[21] == 1 &&
        shape->pad_before[0] == 0 && kernel == shape->kernel[2] &&
        (kernel == 1 || kernel == 3) && (shape->in_channels / shape->groups) % 4 == 0 &&
        y.count <= UINT32_MAX / 4) {
        NSString *name = kernel == 1 ? @"nm_bf16_conv2d_1" : @"nm_bf16_conv2d_3";
        if (_pipelines[name]) {
            uint64_t spatial = (uint64_t)a.u[22] * a.u[23];
            uint64_t pairs = (spatial + 1) / 2;
            uint64_t work = (uint64_t)shape->batch * shape->out_channels * pairs * 4;
            return [self runKernel:name arguments:a views:views count:4 work:work groups:false];
        }
    }
    return [self runKernel:bf16 ? @"nm_bf16_conv" : @"nm_conv"
                 arguments:a views:views count:4 work:a.n[0] groups:false];
}
- (MPSMatrix *)matrixForView:(ds4_num_view)view
                        rows:(uint32_t)rows
                     columns:(uint32_t)columns
                       batch:(uint32_t)batch {
    NSUInteger offset;
    id<MTLBuffer> buffer = [self bufferAtAddress:view.data
                                           bytes:(uint64_t)rows * columns * batch * 4
                                          offset:&offset];
    if (!buffer)
        return nil;
    MPSMatrixDescriptor *descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                              columns:columns
                                             matrices:batch
                                             rowBytes:(NSUInteger)columns * 4
                                          matrixBytes:(NSUInteger)rows * columns * 4
                                             dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:descriptor];
}
- (ds4_status)multiply:(ds4_num_view)y
                  left:(ds4_num_view)x
                 right:(ds4_num_view)w
                 batch:(uint32_t)batch
                  rows:(uint32_t)rows
                 inner:(uint32_t)inner
               columns:(uint32_t)columns
         transposeLeft:(bool)transposeLeft
        transposeRight:(bool)transposeRight {
    MPSMatrix *left = [self matrixForView:x
                                     rows:transposeLeft ? inner : rows
                                  columns:transposeLeft ? rows : inner
                                    batch:batch];
    MPSMatrix *right = [self matrixForView:w
                                      rows:transposeRight ? columns : inner
                                   columns:transposeRight ? inner : columns
                                     batch:batch];
    MPSMatrix *result = [self matrixForView:y rows:rows columns:columns batch:batch];
    MPSMatrixMultiplication *operation =
        [[MPSMatrixMultiplication alloc] initWithDevice:_device
                                          transposeLeft:transposeLeft
                                         transposeRight:transposeRight
                                             resultRows:rows
                                          resultColumns:columns
                                        interiorColumns:inner
                                                  alpha:1
                                                   beta:0];
    operation.options = MPSKernelOptionsNone;
    operation.batchSize = batch;
    id<MTLCommandBuffer> command = [_queue commandBuffer];
    if (!left || !right || !result || !operation || !command)
        return DS4_STATUS_NOMEM;
    [operation encodeToCommandBuffer:command leftMatrix:left rightMatrix:right resultMatrix:result];
    return [self completeCommand:command];
}
- (ds4_status)setBudgetMargin:(uint64_t)margin {
    ds4_metal_memory memory;
    if (!ds4_metal_owned_device_memory(_device, &memory))
        return DS4_STATUS_NOT_READY;
    return ds4_metal_resources_set_budget(_resources, &memory, margin);
}
@end

static DS4MetalNumeric *nm_state(void *p) { return (__bridge DS4MetalNumeric *)p; }
static ds4_status nm_alloc(void *p, uint64_t bytes, void **out) {
    @autoreleasepool {
        return [nm_state(p) allocateBytes:bytes result:out];
    }
}
static ds4_status nm_release(void *p, void *address) {
    @autoreleasepool {
        return [nm_state(p) releaseAddress:address];
    }
}
static ds4_status nm_copy(void *p, void *destination, const void *source, uint64_t bytes,
                          int kind) {
    @autoreleasepool {
        return [nm_state(p) copyTo:destination from:source bytes:bytes kind:kind];
    }
}
static ds4_status nm_fence(void *p) {
    (void)p;
    return DS4_STATUS_OK;
}
static ds4_status nm_refresh_memory(void *p) {
    return [nm_state(p) refreshAvailability];
}
static void nm_close(void *p) { (void)CFBridgingRelease(p); }
static ds4_status nm_prepare_close(void *p) { return [nm_state(p) closeUpload]; }
static ds4_status nm_run(void *p, NSString *name, ds4_nm_args arguments, ds4_num_view *views,
                         uint32_t count, uint64_t work, bool groups) {
    @autoreleasepool {
        return [nm_state(p) runKernel:name
                            arguments:arguments
                                views:views
                                count:count
                                 work:work
                               groups:groups];
    }
}
static ds4_status nm_point(void *p, ds4_num_view y, ds4_num_view x, ds4_num_view b, uint32_t op,
                           float scalar) {
    ds4_nm_args a = {0};
    a.n[0] = x.count;
    a.n[1] = b.count;
    a.u[0] = op;
    a.f[0] = scalar;
    ds4_num_view v[] = {y, x, b};
    return nm_run(p, @"nm_point", a, v, 3, x.count, false);
}
static ds4_status nm_norm(void *p, ds4_num_view y, ds4_num_view x, ds4_num_view w, ds4_num_view b,
                          uint32_t width, float eps, uint32_t kind, uint32_t mode, bool pre) {
    ds4_nm_args a = {0};
    a.u[0] = width;
    a.u[1] = kind;
    a.u[2] = mode;
    a.u[3] = pre;
    a.u[4] = b.data != NULL;
    a.f[0] = eps;
    if (!mode)
        w = (ds4_num_view){0};
    ds4_num_view v[] = {y, x, w, b};
    NSString *kernel = @"nm_norm";
    if (x.type == DS4_NUM_BF16) {
        if (kind == DS4_NUM_RMS)
            kernel = @"nm_bf16_rms";
        else if (kind == DS4_NUM_LAYER && width % 8 == 0 &&
                 width <= DS4_NM_BF16_LAYERNORM_MAX_WIDTH)
            kernel = @"nm_bf16_layernorm";
    }
    return nm_run(p, kernel, a, v, 4, x.count / width, true);
}
static ds4_status nm_channel(void *p, ds4_num_view y, ds4_num_view x, ds4_num_view scale,
                             uint32_t channels, uint32_t spatial, float eps, bool rms) {
    ds4_nm_args a = {0};
    a.u[0] = channels;
    a.u[1] = spatial;
    a.u[2] = rms;
    a.f[0] = eps;
    ds4_num_view v[] = {y, x, scale};
    NSString *kernel = rms && x.type == DS4_NUM_BF16 ? @"nm_bf16_channel_rms" : @"nm_channel";
    return nm_run(p, kernel, a, v, 3, spatial, true);
}
static ds4_status nm_channel_l2(void *p, ds4_num_view y, ds4_num_view x, uint32_t c, uint32_t s,
                                float e) {
    return nm_channel(p, y, x, (ds4_num_view){0}, c, s, e, false);
}
static ds4_status nm_channel_rms(void *p, ds4_num_view y, ds4_num_view x, ds4_num_view w,
                                 uint32_t c, uint32_t s, float e) {
    return nm_channel(p, y, x, w, c, s, e, true);
}
static ds4_status nm_rope(void *p, ds4_num_view y, ds4_num_view x, ds4_num_view co, ds4_num_view si,
                          uint32_t heads, uint32_t dim, bool adjacent, uint32_t rounding) {
    ds4_nm_args a = {0};
    a.n[0] = x.count;
    a.u[0] = heads;
    a.u[1] = dim;
    a.u[2] = adjacent;
    a.u[3] = rounding;
    ds4_num_view v[] = {y, x, co, si};
    return nm_run(p, @"nm_rope", a, v, 4, x.count / 2, false);
}
static ds4_status nm_reorder(void *p, ds4_num_view y, ds4_num_view x, const ds4_num_reindex *r) {
    ds4_nm_args a = {0};
    a.n[0] = 1;
    a.u[0] = r->rank;
    for (uint32_t i = 0; i < r->rank; i++) {
        a.n[0] *= r->shape[i];
        a.shape[i] = r->shape[i];
        a.stride[i] = r->stride[i];
    }
    ds4_num_view v[] = {y, x};
    return nm_run(p, @"nm_reorder", a, v, 2, a.n[0], false);
}
static ds4_status nm_nearest(void *p, ds4_num_view y, ds4_num_view x, const ds4_num_resample *r) {
    ds4_nm_args a = {0};
    a.n[0] = (uint64_t)r->batch * r->channels;
    for (uint32_t i = 0; i < 3; i++) {
        a.u[i] = r->input[i];
        a.u[3 + i] = r->output[i];
        a.n[0] *= r->output[i];
    }
    ds4_num_view v[] = {y, x};
    return nm_run(p, @"nm_nearest", a, v, 2, a.n[0], false);
}
/* Index/bounds buffers use the numeric owner, so admission includes metadata
 * staging too. They are immutable until the synchronous submission finishes. */
static ds4_status nm_indices(DS4MetalNumeric *s, const uint32_t *data, uint64_t count,
                             ds4_num_buffer **out) {
    ds4_status status = [s allocateTemporary:count type:DS4_NUM_F32 result:out];
    if (!status)
        status = [s writeTemporary:*out data:data bytes:count * 4];
    return status;
}
static ds4_status nm_temp_free(ds4_num_buffer *b, ds4_status status) {
    ds4_status release = ds4_num_dispose(b);
    return status ? status : release;
}
static ds4_status nm_pool(void *p, ds4_num_view y, ds4_num_view x, const uint32_t *indices,
                          uint32_t rows, uint32_t group, uint32_t width, float divisor) {
    ds4_num_buffer *b = NULL;
    ds4_status status = nm_indices(nm_state(p), indices, (uint64_t)rows * group, &b);
    ds4_nm_args a = {0};
    a.n[0] = (uint64_t)rows * width;
    a.u[0] = group;
    a.u[1] = width;
    a.f[0] = divisor;
    ds4_num_view v[] = {y, x, ds4_num_data(b)};
    if (!status)
        status = nm_run(p, @"nm_pool", a, v, 3, a.n[0], false);
    return nm_temp_free(b, status);
}
static ds4_status nm_softmax(void *p, ds4_num_view y, ds4_num_view x, const uint32_t *bounds,
                             uint32_t rows, uint32_t width, float scale) {
    ds4_num_buffer *b = NULL;
    ds4_status status = nm_indices(nm_state(p), bounds, (uint64_t)rows * 2, &b);
    ds4_nm_args a = {0};
    a.u[0] = width;
    a.f[0] = scale;
    ds4_num_view v[] = {y, x, ds4_num_data(b)};
    if (!status) {
        status = nm_run(p, @"nm_ordered_softmax", a, v, 3, rows, true);
    }
    return nm_temp_free(b, status);
}
static ds4_status nm_attend(void *p, ds4_num_view y, ds4_num_view q, ds4_num_view k, ds4_num_view v,
                            const uint32_t *bounds, const ds4_num_attention *r) {
    ds4_num_buffer *b = NULL;
    ds4_status status = nm_indices(nm_state(p), bounds, (uint64_t)r->batch * r->queries * 2, &b);
    ds4_nm_args a = {0};
    a.u[0] = r->batch;
    a.u[1] = r->queries;
    a.u[2] = r->keys;
    a.u[3] = r->q_heads;
    a.u[4] = r->kv_heads;
    a.u[5] = r->dim;
    a.u[6] = r->round_scores;
    a.u[7] = r->round_probabilities;
    a.u[8] = r->relative_keys.data ? r->relative_rows : 0;
    a.u[9] = (uint32_t)r->relative_offset;
    a.f[0] = r->scale;
    a.f[1] = r->softcap;
    ds4_num_view views[] = {y, q, k, v, r->relative_keys, ds4_num_data(b)};
    if (!status) {
        bool ordered_bf16 = y.type == DS4_NUM_BF16 && q.type == DS4_NUM_BF16 &&
                            k.type == DS4_NUM_BF16 && v.type == DS4_NUM_BF16 &&
                            r->round_scores && r->round_probabilities &&
                            !r->relative_keys.data && r->dim % 8 == 0 && r->softcap == 0;
        NSString *kernel = ordered_bf16 ? @"nm_bf16_attention" : @"nm_attention";
        status = nm_run(p, kernel, a, views, 6,
                        (uint64_t)r->batch * r->queries * r->q_heads, true);
    }
    return nm_temp_free(b, status);
}
static ds4_status nm_conv(void *p, ds4_num_view y, ds4_num_view x, ds4_num_view w,
                          ds4_num_view bias, const ds4_num_convolution *r) {
    @autoreleasepool {
        return [nm_state(p) convolve:y input:x weight:w bias:bias descriptor:r];
    }
}
static ds4_status nm_float(DS4MetalNumeric *s, ds4_num_view x, ds4_num_buffer **temp,
                           ds4_num_view *out) {
    *out = x;
    if (x.type == DS4_NUM_F32)
        return DS4_STATUS_OK;
    ds4_status status = [s allocateTemporary:x.count type:DS4_NUM_F32 result:temp];
    if (!status) {
        *out = ds4_num_data(*temp);
        status = nm_point((__bridge void *)s, *out, x, (ds4_num_view){0}, DS4_NUM_CAST, 0);
    }
    return status;
}
/* BF16 projections retain the oracle's FP32 accumulation order without staging.
 * Other layouts and dtypes use MPS with lossless F32 operands and owned temps. */
static ds4_status nm_gemm(void *p, ds4_num_view y, ds4_num_view x, ds4_num_view w,
                          ds4_num_view bias, uint32_t batch, uint32_t rows, uint32_t in,
                          uint32_t out, bool ta, bool tb) {
    @autoreleasepool {
        DS4MetalNumeric *s = nm_state(p);
        if (batch == 1 && tb && x.type == DS4_NUM_BF16 &&
            w.type == DS4_NUM_BF16 && y.type == DS4_NUM_BF16) {
            return [s projectBFloat16:y left:x right:w bias:bias
                                rows:rows inner:in columns:out transposeInput:ta];
        }
        ds4_num_buffer *xt = NULL, *wt = NULL, *yt = NULL;
        ds4_num_view xf = {0}, wf = {0}, yf = ds4_num_slice(y, 0, (uint64_t)batch * rows * out);
        ds4_status status = nm_float(s, x, &xt, &xf);
        if (!status)
            status = nm_float(s, w, &wt, &wf);
        if (!status && y.type != DS4_NUM_F32) {
            status = [s allocateTemporary:yf.count type:DS4_NUM_F32 result:&yt];
            yf = ds4_num_data(yt);
        }
        if (!status) {
            status = [s multiply:yf
                            left:xf
                           right:wf
                           batch:batch
                            rows:rows
                           inner:in
                         columns:out
                   transposeLeft:ta
                  transposeRight:tb];
        }
        if (!status && (yt || bias.data)) {
            ds4_nm_args a = {0};
            a.n[0] = (uint64_t)batch * rows * out;
            a.u[0] = out;
            a.u[1] = bias.data != NULL;
            ds4_num_view v[] = {y, yf, bias};
            status = nm_run(p, @"nm_epilogue", a, v, 3, a.n[0], false);
        }
        status = nm_temp_free(yt, status);
        status = nm_temp_free(wt, status);
        return nm_temp_free(xt, status);
    }
}
static ds4_status nm_linear(void *p, ds4_num_view y, ds4_num_view x, ds4_num_view w, ds4_num_view b,
                            uint32_t rows, uint32_t in, uint32_t out) {
    return nm_gemm(p, y, x, w, b, 1, rows, in, out, false, true);
}
static ds4_status nm_matmul(void *p, ds4_num_view y, ds4_num_view x, ds4_num_view w, uint32_t batch,
                            uint32_t rows, uint32_t in, uint32_t out, bool ta, bool tb) {
    return nm_gemm(p, y, x, w, (ds4_num_view){0}, batch, rows, in, out, ta, tb);
}
static uint64_t nm_linear_workspace(uint32_t rows, uint32_t in, uint32_t out) {
    uint64_t x = (uint64_t)rows * in, w = (uint64_t)in * out, y = (uint64_t)rows * out;
    if (x > UINT64_MAX - w || x + w > UINT64_MAX - y || x + w + y > UINT64_MAX / 4)
        return UINT64_MAX;
    return (x + w + y) * 4; /* Lossless BF16 expansion of both operands and result. */
}
static ds4_status nm_weight_read(void *p, void *destination, int fd,
                                uint64_t offset, uint64_t bytes) {
    return [nm_state(p) readTo:destination descriptor:fd offset:offset bytes:bytes];
}
static const ds4_numeric_ops nm_ops = {.allocate = nm_alloc,
                                       .release = nm_release,
                                       .copy = nm_copy,
                                       .fence = nm_fence,
                                       .close = nm_close,
                                       .linear = nm_linear,
                                       .norm = nm_norm,
                                       .point = nm_point,
                                       .rope = nm_rope,
                                       .attend = nm_attend,
                                       .conv = nm_conv,
                                       .reorder = nm_reorder,
                                       .pool = nm_pool,
                                       .nearest = nm_nearest,
                                       .softmax = nm_softmax,
                                       .matmul_batch = nm_matmul,
                                       .channel_l2 = nm_channel_l2,
                                       .channel_rms = nm_channel_rms,
                                       .linear_workspace = nm_linear_workspace,
                                       .refresh_memory = nm_refresh_memory,
                                       .prepare_close = nm_prepare_close,
                                       .weight_read = nm_weight_read};

ds4_status ds4_numeric_metal(ds4_numeric **out, ds4_resources *resources, uint64_t owner) {
    if (!out)
        return DS4_STATUS_INVALID;
    *out = NULL;
    if (!resources || !owner)
        return DS4_STATUS_INVALID;
    @autoreleasepool {
        ds4_status status;
        DS4MetalNumeric *state = [[DS4MetalNumeric alloc] initWithResources:resources
                                                                      owner:owner
                                                                     status:&status];
        if (!state)
            return status;
        void *retained = (__bridge_retained void *)state;
        status = ds4_numeric_create(out, &nm_ops, retained, resources, owner, true);
        if (status)
            nm_close(retained);
        else
            [state bindContext:*out];
        return status;
    }
}
ds4_status ds4_numeric_metal_budget(ds4_numeric *context, uint64_t margin) {
    if (!context || context->ops != &nm_ops)
        return DS4_STATUS_INVALID;
    return [nm_state(context->state) setBudgetMargin:margin];
}
