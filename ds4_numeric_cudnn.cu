/* Native cuDNN C API, confined to the CUDA backend. Fixed FP32 accumulation,
 * deterministic algorithms and bounded workspace; no Python runtime linkage. */
#include "ds4_numeric_internal.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <limits.h>
#include <stdlib.h>

typedef struct {
  cudnnHandle_t handle;
  cudaStream_t stream;
  ds4_resource *metadata;
} numeric_cudnn;
static ds4_status check(cudnnStatus_t s) {
  return s == CUDNN_STATUS_SUCCESS         ? DS4_STATUS_OK
         : s == CUDNN_STATUS_ALLOC_FAILED  ? DS4_STATUS_NOMEM
         : s == CUDNN_STATUS_NOT_SUPPORTED ? DS4_STATUS_UNSUPPORTED
                                           : DS4_STATUS_INFERENCE;
}
extern "C" ds4_status ds4_numeric_cudnn_create(void **out, void *stream,
                                               ds4_resources *resources,
                                               uint64_t owner) {
  *out = NULL;
  numeric_cudnn *s = (numeric_cudnn *)calloc(1, sizeof(*s));
  if (!s)
    return DS4_STATUS_NOMEM;
  ds4_resource_spec spec = {};
  spec.owner = owner;
  spec.location = DS4_MEMORY_HOST;
  spec.device = -1;
  spec.requested = spec.allocated = spec.live = sizeof(*s);
  ds4_status status = ds4_resource_register(resources, &spec, &s->metadata);
  if (!status)
    status = check(cudnnCreate(&s->handle));
  if (!status)
    status = check(cudnnSetStream(s->handle, (cudaStream_t)stream));
  if (status) {
    if (s->handle)
      cudnnDestroy(s->handle);
    if (s->metadata)
      ds4_resource_release(s->metadata);
    free(s);
    return status;
  }
  s->stream = (cudaStream_t)stream;
  *out = s;
  return DS4_STATUS_OK;
}
extern "C" void ds4_numeric_cudnn_close(void *state) {
  numeric_cudnn *s = (numeric_cudnn *)state;
  if (!s)
    return;
  cudnnDestroy(s->handle);
  ds4_resource_release(s->metadata);
  free(s);
}
__global__ static void pad_kernel(void *out, const void *in,
                                  ds4_num_convolution a, uint32_t unit,
                                  uint64_t count) {
  uint64_t dims[3];
  for (int i = 0; i < 3; i++)
    dims[i] = (uint64_t)a.input[i] + a.pad_before[i] + a.pad_after[i];
  for (uint64_t i = (uint64_t)blockIdx.x * 256 + threadIdx.x; i < count;
       i += (uint64_t)gridDim.x * 256) {
    uint64_t rem = i, index = 0, stride = 1;
    bool valid = true;
    for (int d = 2; d >= 0; d--) {
      int64_t p = (int64_t)(rem % dims[d]) - a.pad_before[d];
      rem /= dims[d];
      if (p < 0 || p >= a.input[d])
        valid = false;
      else
        index += p * stride;
      stride *= a.input[d];
    }
    index += rem * stride;
    if (unit == 2)
      ((uint16_t *)out)[i] = valid ? ((const uint16_t *)in)[index] : 0;
    else
      ((float *)out)[i] = valid ? ((const float *)in)[index] : 0;
  }
}
extern "C" ds4_status ds4_numeric_cudnn_conv(void *state, ds4_numeric *context,
                                             ds4_num_view y, ds4_num_view x,
                                             ds4_num_view w, ds4_num_view bias,
                                             const ds4_num_convolution *a) {
  numeric_cudnn *s = (numeric_cudnn *)state;
  cudnnTensorDescriptor_t xd = NULL, yd = NULL, bd = NULL;
  cudnnFilterDescriptor_t wd = NULL;
  cudnnConvolutionDescriptor_t cd = NULL;
  ds4_num_buffer *padded = NULL, *workspace = NULL;
  ds4_status status = DS4_STATUS_OK;
  int input[5], output[5], filter[5], xs[5], ys[5], pad[3], stride[3],
      dilation[3];
  bool symmetric = true;
  const bool image = a->input[0] == 1 && a->kernel[0] == 1 &&
                     !a->pad_before[0] && !a->pad_after[0] &&
                     a->stride[0] == 1 && a->dilation[0] == 1;
  for (int d = 0; d < 3; d++)
    if (a->pad_before[d] != a->pad_after[d])
      symmetric = false;
  input[0] = output[0] = a->batch;
  input[1] = a->in_channels;
  output[1] = filter[0] = a->out_channels;
  filter[1] = a->in_channels / a->groups;
  if (a->batch > INT_MAX || a->in_channels > INT_MAX ||
      a->out_channels > INT_MAX)
    return DS4_STATUS_UNSUPPORTED;
  for (int d = 0; d < 3; d++) {
    uint64_t padded_dim =
        (uint64_t)a->input[d] + a->pad_before[d] + a->pad_after[d];
    if (padded_dim > INT_MAX || a->stride[d] > INT_MAX ||
        a->dilation[d] > INT_MAX)
      return DS4_STATUS_UNSUPPORTED;
    input[d + 2] = symmetric ? a->input[d] : padded_dim;
    output[d + 2] =
        (padded_dim - (uint64_t)a->dilation[d] * (a->kernel[d] - 1) - 1) /
            a->stride[d] +
        1;
    filter[d + 2] = a->kernel[d];
    pad[d] = symmetric ? a->pad_before[d] : 0;
    stride[d] = a->stride[d];
    dilation[d] = a->dilation[d];
  }
  xs[4] = ys[4] = 1;
  for (int d = 3; d >= 0; d--) {
    if ((uint64_t)xs[d + 1] * input[d + 1] > INT_MAX ||
        (uint64_t)ys[d + 1] * output[d + 1] > INT_MAX)
      return DS4_STATUS_UNSUPPORTED;
    xs[d] = xs[d + 1] * input[d + 1];
    ys[d] = ys[d + 1] * output[d + 1];
  }
  cudnnDataType_t dtype =
      x.type == DS4_NUM_BF16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT;
  float alpha = 1, beta = 0;
  ds4_num_view xx = x;
#define CUDNN_TRY(expr)                                                        \
  do {                                                                         \
    status = check(expr);                                                      \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
  if (!symmetric) {
    uint64_t count = (uint64_t)input[0] * xs[0];
    status = ds4_num_alloc(context, count, x.type, &padded);
    if (status)
      goto cleanup;
    xx = ds4_num_data(padded);
    unsigned blocks =
        (unsigned)((count + 255) / 256 > 65535 ? 65535 : (count + 255) / 256);
    pad_kernel<<<blocks, 256, 0, s->stream>>>(
        xx.data, x.data, *a, ds4_num_type_bytes(x.type), count);
    if (cudaGetLastError() != cudaSuccess) {
      status = DS4_STATUS_INFERENCE;
      goto cleanup;
    }
  }
  CUDNN_TRY(cudnnCreateTensorDescriptor(&xd));
  CUDNN_TRY(cudnnCreateTensorDescriptor(&yd));
  CUDNN_TRY(cudnnCreateFilterDescriptor(&wd));
  CUDNN_TRY(cudnnCreateConvolutionDescriptor(&cd));
  if (image) {
    CUDNN_TRY(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, dtype, input[0],
                                         input[1], input[3], input[4]));
    CUDNN_TRY(cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, dtype,
                                         output[0], output[1], output[3],
                                         output[4]));
    CUDNN_TRY(cudnnSetFilter4dDescriptor(wd, dtype, CUDNN_TENSOR_NCHW,
                                         filter[0], filter[1], filter[3],
                                         filter[4]));
    CUDNN_TRY(cudnnSetConvolution2dDescriptor(
        cd, pad[1], pad[2], stride[1], stride[2], dilation[1], dilation[2],
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  } else {
    CUDNN_TRY(cudnnSetTensorNdDescriptor(xd, dtype, 5, input, xs));
    CUDNN_TRY(cudnnSetTensorNdDescriptor(yd, dtype, 5, output, ys));
    CUDNN_TRY(
        cudnnSetFilterNdDescriptor(wd, dtype, CUDNN_TENSOR_NCHW, 5, filter));
    CUDNN_TRY(cudnnSetConvolutionNdDescriptor(cd, 3, pad, stride, dilation,
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));
  }
  CUDNN_TRY(cudnnSetConvolutionGroupCount(cd, a->groups));
  CUDNN_TRY(cudnnSetConvolutionMathType(
      cd, x.type == DS4_NUM_BF16 ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH));
  {
    cudnnConvolutionFwdAlgoPerf_t candidates[8];
    int count = 0;
    CUDNN_TRY(cudnnGetConvolutionForwardAlgorithm_v7(s->handle, xd, wd, cd, yd,
                                                     8, &count, candidates));
    bool selected = false;
    for (int i = 0; i < count; i++) {
      if (candidates[i].status != CUDNN_STATUS_SUCCESS ||
          candidates[i].determinism != CUDNN_DETERMINISTIC ||
          candidates[i].memory > 64u * 1024u * 1024u)
        continue;
      if (x.type == DS4_NUM_F32 && candidates[i].mathType != CUDNN_FMA_MATH)
        continue;
      if (candidates[i].memory) {
        status = ds4_num_alloc(context, (candidates[i].memory + 3) / 4,
                               DS4_NUM_F32, &workspace);
        if (status == DS4_STATUS_NOMEM)
          continue;
        if (status)
          goto cleanup;
      }
      CUDNN_TRY(cudnnSetConvolutionMathType(cd, candidates[i].mathType));
      CUDNN_TRY(cudnnConvolutionForward(
          s->handle, &alpha, xd, xx.data, wd, w.data, cd, candidates[i].algo,
          workspace ? ds4_num_data(workspace).data : NULL, candidates[i].memory,
          &beta, yd, y.data));
      selected = true;
      break;
    }
    if (!selected) {
      status = DS4_STATUS_NOMEM;
      goto cleanup;
    }
  }
  /* PyTorch Conv casts convolution output before adding bias. cuDNN's
   * tensor add preserves this boundary, unlike a fused GEMM epilogue. */
  if (bias.data) {
    int dims[] = {1, (int)a->out_channels, 1, 1, 1},
        strides[] = {(int)a->out_channels, 1, 1, 1, 1};
    CUDNN_TRY(cudnnCreateTensorDescriptor(&bd));
    if (image)
      CUDNN_TRY(cudnnSetTensor4dDescriptor(bd, CUDNN_TENSOR_NCHW, dtype, 1,
                                           a->out_channels, 1, 1));
    else
      CUDNN_TRY(cudnnSetTensorNdDescriptor(bd, dtype, 5, dims, strides));
    CUDNN_TRY(
        cudnnAddTensor(s->handle, &alpha, bd, bias.data, &alpha, yd, y.data));
  }
cleanup:
  if (bd)
    cudnnDestroyTensorDescriptor(bd);
  if (xd)
    cudnnDestroyTensorDescriptor(xd);
  if (yd)
    cudnnDestroyTensorDescriptor(yd);
  if (wd)
    cudnnDestroyFilterDescriptor(wd);
  if (cd)
    cudnnDestroyConvolutionDescriptor(cd);
  ds4_status freed = ds4_num_dispose(workspace),
             other = ds4_num_dispose(padded);
  if (!freed)
    freed = other;
  return status ? status : freed;
}
