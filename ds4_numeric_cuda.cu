/* Native suite baseline kernels. Semantics: pinned Transformers/Diffusers;
 * online softmax: FlashAttention/llama.cpp, independently implemented here.
 * No legacy LLM kernels, global quality flags or offset caches are changed. */
#include "ds4_numeric_internal.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

typedef struct {
  cudaStream_t stream;
  cublasHandle_t blas;
  cublasLtHandle_t lt;
  ds4_numeric *context;
  void *cudnn;
  void *workspace;
  ds4_resource *workspace_account;
  ds4_resource *metadata;
} numeric_cuda;
static ds4_status status(cudaError_t e) {
  if (e == cudaSuccess)
    return DS4_STATUS_OK;
  if (e == cudaErrorMemoryAllocation)
    return DS4_STATUS_NOMEM;
  if (e == cudaErrorIllegalAddress || e == cudaErrorLaunchFailure ||
      e == cudaErrorECCUncorrectable)
    return DS4_STATUS_DEVICE_LOST;
  return DS4_STATUS_INFERENCE;
}
static ds4_status launch_status(void) { return status(cudaGetLastError()); }
__device__ static float readv(ds4_num_view v, uint64_t i) {
  return v.type == DS4_NUM_F32 ? ((float *)v.data)[i]
                               : __bfloat162float(((__nv_bfloat16 *)v.data)[i]);
}
__device__ static float roundv(float f, uint32_t t) {
  return t == DS4_NUM_F32 ? f : __bfloat162float(__float2bfloat16_rn(f));
}
__device__ static void writev(ds4_num_view v, uint64_t i, float f) {
  if (v.type == DS4_NUM_F32)
    ((float *)v.data)[i] = f;
  else
    ((__nv_bfloat16 *)v.data)[i] = __float2bfloat16_rn(f);
}
static ds4_status gpu_allocate(void *p, uint64_t n, void **out) {
  (void)p;
  return status(cudaMalloc(out, n));
}
static ds4_status gpu_release(void *p, void *storage) {
  (void)p;
  return status(cudaFree(storage));
}
static ds4_status gpu_fence(void *p) {
  return status(cudaStreamSynchronize(((numeric_cuda *)p)->stream));
}
static ds4_status gpu_copy(void *p, void *out, const void *in, uint64_t n,
                           int kind) {
  cudaMemcpyKind k = kind == 0   ? cudaMemcpyHostToDevice
                     : kind == 1 ? cudaMemcpyDeviceToHost
                                 : cudaMemcpyDeviceToDevice;
  ds4_status s =
      status(cudaMemcpyAsync(out, in, n, k, ((numeric_cuda *)p)->stream));
  /* Host bytes may be temporary. Complete transfers before returning. */
  return s ? s : gpu_fence(p);
}
static void gpu_close(void *p) {
  numeric_cuda *s = (numeric_cuda *)p;
  if (s->workspace)
    cudaFree(s->workspace);
  if (s->workspace_account)
    ds4_resource_release(s->workspace_account);
  cublasDestroy(s->blas);
  if (s->lt)
    cublasLtDestroy(s->lt);
  ds4_numeric_cudnn_close(s->cudnn);
  cudaStreamDestroy(s->stream);
  if (s->metadata)
    ds4_resource_release(s->metadata);
  free(s);
}
__global__ static void bias_fill_kernel(ds4_num_view y, ds4_num_view b,
                                        uint32_t width, uint64_t n) {
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (uint64_t)blockDim.x * gridDim.x)
    writev(y, i, readv(b, i % width));
}
static unsigned blocks(uint64_t n) {
  return (unsigned)((n + 255) / 256 > 65535 ? 65535 : (n + 255) / 256);
}
static ds4_status bias_linear(numeric_cuda *s, ds4_num_view y, ds4_num_view x,
                              ds4_num_view w, ds4_num_view bias, uint32_t rows,
                              uint32_t in, uint32_t out) {
  cublasLtMatmulDesc_t desc = NULL;
  cublasLtMatrixLayout_t ad = NULL, bd = NULL, cd = NULL;
  cublasLtMatmulPreference_t preference = NULL;
  ds4_num_buffer *workspace = NULL;
  ds4_status status =
      ds4_num_alloc(s->context, 256u * 1024u, DS4_NUM_F32, &workspace);
  if (status)
    return status;
  size_t workspace_bytes = 1024u * 1024u;
  float alpha = 1, beta = 0;
  cublasOperation_t transpose = CUBLAS_OP_T;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  uint32_t alignment = 16;
#define BLAS_TRY(expr)                                                         \
  do {                                                                         \
    if ((expr) != CUBLAS_STATUS_SUCCESS) {                                     \
      status = DS4_STATUS_INFERENCE;                                           \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)
  BLAS_TRY(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  BLAS_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                          &transpose, sizeof(transpose)));
  BLAS_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                          &epilogue, sizeof(epilogue)));
  BLAS_TRY(cublasLtMatmulDescSetAttribute(
      desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias.data, sizeof(bias.data)));
  BLAS_TRY(cublasLtMatrixLayoutCreate(&ad, CUDA_R_16BF, in, out, in));
  BLAS_TRY(cublasLtMatrixLayoutCreate(&bd, CUDA_R_16BF, in, rows, in));
  BLAS_TRY(cublasLtMatrixLayoutCreate(&cd, CUDA_R_16BF, out, rows, out));
  BLAS_TRY(cublasLtMatmulPreferenceCreate(&preference));
  BLAS_TRY(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
      sizeof(reduction)));
  BLAS_TRY(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignment,
      sizeof(alignment)));
  BLAS_TRY(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignment,
      sizeof(alignment)));
  BLAS_TRY(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &alignment,
      sizeof(alignment)));
  BLAS_TRY(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &alignment,
      sizeof(alignment)));
  BLAS_TRY(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes)));
  {
    cublasLtMatmulHeuristicResult_t result;
    int returned = 0;
    BLAS_TRY(cublasLtMatmulAlgoGetHeuristic(s->lt, desc, ad, bd, cd, cd,
                                            preference, 1, &result, &returned));
    if (!returned) {
      status = DS4_STATUS_UNSUPPORTED;
      goto cleanup;
    }
    BLAS_TRY(cublasLtMatmul(s->lt, desc, &alpha, w.data, ad, x.data, bd, &beta,
                            y.data, cd, y.data, cd, &result.algo,
                            ds4_num_data(workspace).data, workspace_bytes,
                            s->stream));
  }
cleanup:
  if (preference)
    cublasLtMatmulPreferenceDestroy(preference);
  if (ad)
    cublasLtMatrixLayoutDestroy(ad);
  if (bd)
    cublasLtMatrixLayoutDestroy(bd);
  if (cd)
    cublasLtMatrixLayoutDestroy(cd);
  if (desc)
    cublasLtMatmulDescDestroy(desc);
  ds4_status freed = ds4_num_dispose(workspace);
  return status ? status : freed;
}
static ds4_status gpu_linear(void *p, ds4_num_view y, ds4_num_view x,
                             ds4_num_view w, ds4_num_view b, uint32_t rows,
                             uint32_t in, uint32_t out) {
  numeric_cuda *s = (numeric_cuda *)p;
  if (x.type == DS4_NUM_BF16 && b.data)
    return bias_linear(s, y, x, w, b, rows, in, out);
  float alpha = 1, beta = b.data ? 1 : 0;
  ds4_num_view dest = y;
  if (b.data) {
    bias_fill_kernel<<<blocks((uint64_t)rows * out), 256, 0, s->stream>>>(
        y, b, out, (uint64_t)rows * out);
    ds4_status st = launch_status();
    if (st)
      return st;
  }
  cudaDataType_t type = x.type == DS4_NUM_F32 ? CUDA_R_32F : CUDA_R_16BF;
  cublasComputeType_t compute =
      x.type == DS4_NUM_F32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
  cublasStatus_t e =
      cublasGemmEx(s->blas, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &alpha,
                   w.data, type, in, x.data, type, in, &beta, dest.data,
                   dest.type == DS4_NUM_F32 ? CUDA_R_32F : CUDA_R_16BF, out,
                   compute, CUBLAS_GEMM_DEFAULT);
  ds4_status st =
      e == CUBLAS_STATUS_SUCCESS ? DS4_STATUS_OK : DS4_STATUS_INFERENCE;
  return st;
}
__device__ static float warp_sum(float v) {
  for (int d = 16; d; d /= 2)
    v += __shfl_down_sync(0xffffffff, v, d);
  return __shfl_sync(0xffffffff, v, 0);
}
__global__ static void norm_kernel(ds4_num_view y, ds4_num_view x,
                                   ds4_num_view w, ds4_num_view bias,
                                   uint32_t width, float eps, uint32_t kind,
                                   uint32_t mode, bool pre) {
  __shared__ float partial[8], mean, inv;
  uint64_t row = blockIdx.x;
  float z = 0;
  if (kind == DS4_NUM_LAYER)
    for (uint32_t j = threadIdx.x; j < width; j += 256)
      z += readv(x, row * width + j);
  z = warp_sum(z);
  if (threadIdx.x % 32 == 0)
    partial[threadIdx.x / 32] = z;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0;
    for (int j = 0; j < 8; j++)
      sum += partial[j];
    mean = sum / width;
  }
  __syncthreads();
  z = 0;
  for (uint32_t j = threadIdx.x; j < width; j += 256) {
    float d = readv(x, row * width + j) - mean;
    z += d * d;
  }
  z = warp_sum(z);
  if (threadIdx.x % 32 == 0)
    partial[threadIdx.x / 32] = z;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0;
    for (int j = 0; j < 8; j++)
      sum += partial[j];
    inv =
        kind == DS4_NUM_L2 ? fmaxf(sqrtf(sum), eps) : rsqrtf(sum / width + eps);
  }
  __syncthreads();
  for (uint32_t j = threadIdx.x; j < width; j += 256) {
    float v = kind == DS4_NUM_L2 ? readv(x, row * width + j) / inv
                                 : (readv(x, row * width + j) - mean) * inv;
    if (pre)
      v = roundv(v, x.type);
    if (kind == DS4_NUM_LAYER && mode == DS4_NUM_SCALE_DIRECT && bias.data) {
      writev(y, row * width + j, fmaf(v, readv(w, j), readv(bias, j)));
      continue;
    }
    if (mode != DS4_NUM_SCALE_NONE)
      v *= readv(w, j) + (mode == DS4_NUM_SCALE_ZERO_CENTERED ? 1 : 0);
    writev(y, row * width + j, v + (bias.data ? readv(bias, j) : 0));
  }
}
/* The square and the mean are distinct FP32 operations. Four independent
 * accumulators preserve the row reduction order before the BF16 normalization
 * cast; folding that cast into the scale changes Qwen's head norms. */
__global__ static void rms_rows_kernel(ds4_num_view y, ds4_num_view x,
                                       ds4_num_view w, ds4_num_view bias,
                                       uint32_t width, float eps, uint32_t mode,
                                       bool pre) {
  uint64_t row = (uint64_t)blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= x.count / width)
    return;
  uint32_t lane = threadIdx.x % 32;
  float acc[4] = {0};
  for (uint32_t j = lane * 4; j < width; j += 128)
    for (uint32_t k = 0; k < 4; k++) {
      float value = readv(x, row * width + j + k);
      acc[k] += value * value;
    }
  float sum = ((acc[0] + acc[1]) + acc[2]) + acc[3];
  sum = warp_sum(sum);
  float inv = rsqrtf(sum * (1.0f / width) + eps);
  for (uint32_t j = lane; j < width; j += 32) {
    float value = readv(x, row * width + j) * inv;
    if (pre)
      value = roundv(value, x.type);
    if (mode != DS4_NUM_SCALE_NONE)
      value *= readv(w, j) + (mode == DS4_NUM_SCALE_ZERO_CENTERED ? 1 : 0);
    writev(y, row * width + j, value + (bias.data ? readv(bias, j) : 0));
  }
}
typedef struct {
  float mean, m2, count;
} moment;
__device__ static moment moment_join(moment left, moment right) {
  float count = left.count + right.count;
  if (count == 0)
    return {0, 0, 0};
  float reciprocal = 1.0f / count, wa = right.count * reciprocal,
        wb = left.count * reciprocal;
  float delta = left.mean - right.mean;
  return {fmaf(wa, right.mean, wb * left.mean),
          fmaf((delta * delta) * right.count, wb, right.m2 + left.m2), count};
}
/* Vectorized Welford order agrees with the independent PyTorch LayerNorm
 * oracle on BF16 ties. FMA is explicit only where the moments/affine require
 * it; the rest of this module remains compiled without fast math. */
__global__ static void layer_rows_kernel(ds4_num_view y, ds4_num_view x,
                                         ds4_num_view weight, ds4_num_view bias,
                                         uint32_t width, float eps,
                                         uint32_t mode) {
  __shared__ moment warps[4];
  __shared__ float mean, inv;
  uint64_t row = blockIdx.x;
  uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  moment m = {0, 0, 0};
  for (uint32_t block = threadIdx.x * 4; block < width; block += 512)
    for (uint32_t k = 0; k < 4; k++) {
      float value = readv(x, row * width + block + k), delta = value - m.mean,
            count = m.count + 1;
      float next = fmaf(delta, 1.0f / count, m.mean);
      m = {next, fmaf(delta, value - next, m.m2), count};
    }
  for (int off = 16; off; off /= 2) {
    moment other = {__shfl_down_sync(0xffffffff, m.mean, off),
                    __shfl_down_sync(0xffffffff, m.m2, off),
                    __shfl_down_sync(0xffffffff, m.count, off)};
    m = moment_join(m, other);
  }
  if (!lane)
    warps[warp] = m;
  __syncthreads();
  for (uint32_t off = 2; off; off /= 2) {
    if (!lane && warp < off)
      warps[warp] = moment_join(warps[warp], warps[warp + off]);
    __syncthreads();
  }
  if (!threadIdx.x) {
    mean = warps[0].mean;
    inv = rsqrtf(warps[0].m2 / width + eps);
  }
  __syncthreads();
  for (uint32_t j = threadIdx.x; j < width; j += 128) {
    float z = (readv(x, row * width + j) - mean) * inv;
    float gamma =
        mode == DS4_NUM_SCALE_NONE
            ? 1
            : readv(weight, j) + (mode == DS4_NUM_SCALE_ZERO_CENTERED ? 1 : 0);
    writev(y, row * width + j, fmaf(gamma, z, bias.data ? readv(bias, j) : 0));
  }
}
static ds4_status gpu_norm(void *p, ds4_num_view y, ds4_num_view x,
                           ds4_num_view w, ds4_num_view b, uint32_t width,
                           float eps, uint32_t kind, uint32_t mode, bool pre) {
  if (x.count / width > 2147483647)
    return DS4_STATUS_UNSUPPORTED;
  if (kind == DS4_NUM_LAYER && width % 4 == 0) {
    layer_rows_kernel<<<(unsigned)(x.count / width), 128, 0,
                        ((numeric_cuda *)p)->stream>>>(y, x, w, b, width, eps,
                                                       mode);
    return launch_status();
  }
  if (kind == DS4_NUM_RMS && width >= 128 && width % 4 == 0 &&
      x.count / width >= 16) {
    rms_rows_kernel<<<(unsigned)((x.count / width + 3) / 4), 128, 0,
                      ((numeric_cuda *)p)->stream>>>(y, x, w, b, width, eps,
                                                     mode, pre);
    return launch_status();
  }
  norm_kernel<<<(unsigned)(x.count / width), 256, 0,
                ((numeric_cuda *)p)->stream>>>(y, x, w, b, width, eps, kind,
                                               mode, pre);
  return launch_status();
}
__global__ static void point_kernel(ds4_num_view y, ds4_num_view x,
                                    ds4_num_view b, uint32_t op, float scalar) {
  for (uint64_t i = (uint64_t)blockIdx.x * 256 + threadIdx.x; i < x.count;
       i += (uint64_t)gridDim.x * 256) {
    float z = readv(x, i);
    switch (op) {
    case DS4_NUM_ADD:
      z += readv(b, i % b.count);
      break;
    case DS4_NUM_MUL:
      z *= readv(b, i % b.count);
      break;
    case DS4_NUM_SILU:
      z = z / (1 + expf(-z));
      break;
    case DS4_NUM_SIGMOID:
      z = 1 / (1 + expf(-z));
      break;
    case DS4_NUM_GELU:
      z = 0.5f * z * (1 + erff(z * 0.7071067811865475f));
      break;
    case DS4_NUM_GELU_TANH:
      z = 0.5f * z *
          (1 + tanhf(0.7978845608028654f * (z + 0.044715f * z * z * z)));
      break;
    case DS4_NUM_SCALE:
      z *= scalar;
      break;
    case DS4_NUM_RELU:
      z = fmaxf(z, 0);
      break;
    case DS4_NUM_SOFTPLUS:
      z = z > 20 ? z : log1pf(expf(z));
      break;
    case DS4_NUM_TANH:
      z = tanhf(z);
      break;
    case DS4_NUM_MIN:
      z = fminf(z, readv(b, i % b.count));
      break;
    case DS4_NUM_MAX:
      z = fmaxf(z, readv(b, i % b.count));
      break;
    case DS4_NUM_SIN:
      z = sinf(z);
      break;
    case DS4_NUM_COS:
      z = cosf(z);
      break;
    default:
      break;
    }
    writev(y, i, z);
  }
}
static ds4_status gpu_point(void *p, ds4_num_view y, ds4_num_view x,
                            ds4_num_view b, uint32_t op, float scalar) {
  point_kernel<<<blocks(x.count), 256, 0, ((numeric_cuda *)p)->stream>>>(
      y, x, b, op, scalar);
  return launch_status();
}
__global__ static void rope_kernel(ds4_num_view y, ds4_num_view x,
                                   ds4_num_view co, ds4_num_view si,
                                   uint32_t heads, uint32_t dim, bool adj,
                                   uint32_t rounding) {
  for (uint64_t i = (uint64_t)blockIdx.x * 256 + threadIdx.x; i < x.count / 2;
       i += (uint64_t)gridDim.x * 256) {
    uint32_t d = i % (dim / 2);
    uint64_t h = i / (dim / 2), a = h * dim + (adj ? 2 * d : d),
             b = a + (adj ? 1 : dim / 2), p = (h / heads) * (dim / 2) + d;
    float c = readv(co, p), sn = readv(si, p), u = readv(x, a), v = readv(x, b),
          uc = u * c, vs = v * sn, vc = v * c, us = u * sn;
    if (rounding == 1) {
      uc = roundv(uc, x.type);
      vs = roundv(vs, x.type);
      vc = roundv(vc, x.type);
      us = roundv(us, x.type);
    }
    writev(y, a, rounding == 2 ? fmaf(u, c, -vs) : uc - vs);
    writev(y, b, rounding == 2 ? fmaf(u, sn, vc) : vc + us);
  }
}
static ds4_status gpu_rope(void *p, ds4_num_view y, ds4_num_view x,
                           ds4_num_view co, ds4_num_view si, uint32_t heads,
                           uint32_t dim, bool adj, uint32_t rounding) {
  rope_kernel<<<blocks(x.count / 2), 256, 0, ((numeric_cuda *)p)->stream>>>(
      y, x, co, si, heads, dim, adj, rounding);
  return launch_status();
}
__device__ static float attention_score(ds4_num_view q, ds4_num_view k,
                                        uint64_t qi, uint64_t ki,
                                        ds4_num_attention a) {
  float z = 0;
  for (uint32_t d = threadIdx.x % 32; d < a.dim; d += 32)
    z += readv(q, qi + d) * readv(k, ki + d);
  z = warp_sum(z);
  if (a.relative_keys.data) {
    int64_t index = (int64_t)((ki / a.dim / a.kv_heads) % a.keys) -
                    (int64_t)((qi / a.dim / a.q_heads) % a.queries) +
                    a.relative_offset;
    if (index >= 0 && index < a.relative_rows) {
      float relative = 0;
      uint64_t offset =
          ((uint64_t)index * a.q_heads + (qi / a.dim) % a.q_heads) * a.dim;
      for (uint32_t d = threadIdx.x % 32; d < a.dim; d += 32)
        relative += readv(q, qi + d) * readv(a.relative_keys, offset + d);
      z += warp_sum(relative);
    }
  }
  if (a.round_scores)
    z = roundv(z, q.type);
  z *= a.scale;
  if (a.round_scores)
    z = roundv(z, q.type);
  return a.softcap > 0 ? a.softcap * tanhf(z / a.softcap) : z;
}
/* Four independent query warps per CTA. Keys are visited in 32-row tiles;
 * the running maximum/denominator never stores scores. A second pass permits
 * the eager oracle's BF16 probability cast without an N*N allocation. */
__global__ static void attention_kernel(ds4_num_view y, ds4_num_view q,
                                        ds4_num_view k, ds4_num_view v,
                                        const uint32_t *bounds,
                                        ds4_num_attention a) {
  uint64_t row = (uint64_t)blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= (uint64_t)a.batch * a.queries * a.q_heads)
    return;
  uint32_t lane = threadIdx.x % 32, h = row % a.q_heads,
           r = (row / a.q_heads) % a.queries, b = row / (a.q_heads * a.queries);
  uint64_t qi = row * a.dim;
  uint32_t lo = bounds[2 * ((uint64_t)b * a.queries + r)],
           hi = bounds[2 * ((uint64_t)b * a.queries + r) + 1];
  float m = -INFINITY, l = 0, acc[16] = {0};
  for (uint32_t tile = lo; tile < hi; tile += 32)
    for (uint32_t j = tile; j < hi && j < tile + 32; j++) {
      uint64_t ki = (((uint64_t)b * a.keys + j) * a.kv_heads +
                     h / (a.q_heads / a.kv_heads)) *
                    a.dim;
      float z = attention_score(q, k, qi, ki, a), nm = fmaxf(m, z);
      l = l * expf(m - nm) + expf(z - nm);
      m = nm;
    }
  for (uint32_t tile = lo; tile < hi; tile += 32)
    for (uint32_t j = tile; j < hi && j < tile + 32; j++) {
      uint64_t ki = (((uint64_t)b * a.keys + j) * a.kv_heads +
                     h / (a.q_heads / a.kv_heads)) *
                    a.dim;
      float prob = expf(attention_score(q, k, qi, ki, a) - m) / l;
      if (a.round_probabilities)
        prob = roundv(prob, q.type);
      for (uint32_t d = lane; d < a.dim; d += 32)
        acc[d / 32] += prob * readv(v, ki + d);
    }
  for (uint32_t d = lane; d < a.dim; d += 32)
    writev(y, qi + d, acc[d / 32]);
}
/* BF16 eager attention rounds the QK GEMM and the normalized probabilities.
 * Keep those boundaries and Tensor Core accumulation, with fixed 256-row
 * query tiles. Scratch is O(heads*256*keys), never O(queries*keys). */
__global__ static void softmax_rows_kernel(ds4_num_view scores,
                                           const uint32_t *bounds,
                                           uint32_t batch, uint32_t first,
                                           uint32_t rows, ds4_num_attention a) {
  uint32_t row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= rows)
    return;
  uint32_t lo = bounds[2 * ((uint64_t)batch * a.queries + first + row)],
           hi = bounds[2 * ((uint64_t)batch * a.queries + first + row) + 1];
  float m = -INFINITY;
  for (uint32_t j = lane; j < a.keys; j += 32) {
    float z = j >= lo && j < hi
                  ? roundv(readv(scores, (uint64_t)row * a.keys + j) * a.scale,
                           scores.type)
                  : -INFINITY;
    if (a.softcap > 0 && j >= lo && j < hi)
      z = a.softcap * tanhf(z / a.softcap);
    m = fmaxf(m, z);
  }
  for (int off = 16; off; off /= 2)
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
  float sum = 0;
  for (uint32_t j = lane; j < a.keys; j += 32)
    if (j >= lo && j < hi) {
      float z = roundv(readv(scores, (uint64_t)row * a.keys + j) * a.scale,
                       scores.type);
      if (a.softcap > 0)
        z = a.softcap * tanhf(z / a.softcap);
      sum += expf(z - m);
    }
  for (int off = 16; off; off /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, off);
  for (uint32_t j = lane; j < a.keys; j += 32) {
    float prob = 0;
    if (j >= lo && j < hi) {
      float z = roundv(readv(scores, (uint64_t)row * a.keys + j) * a.scale,
                       scores.type);
      if (a.softcap > 0)
        z = a.softcap * tanhf(z / a.softcap);
      prob = expf(z - m) / sum;
    }
    writev(scores, (uint64_t)row * a.keys + j, prob);
  }
}
__global__ static void pack_attention_kv(ds4_num_view out, ds4_num_view in,
                                         ds4_num_attention a) {
  for (uint64_t i = (uint64_t)blockIdx.x * 256 + threadIdx.x; i < out.count;
       i += (uint64_t)gridDim.x * 256) {
    uint64_t d = i % a.dim, key = (i / a.dim) % a.keys,
             head = (i / a.dim / a.keys) % a.q_heads,
             b = i / (a.dim * a.q_heads * a.keys);
    writev(out, i,
           readv(in, ((b * a.keys + key) * a.kv_heads +
                      head / (a.q_heads / a.kv_heads)) *
                             a.dim +
                         d));
  }
}
__global__ static void scatter_attention(ds4_num_view y, ds4_num_view temp,
                                         ds4_num_attention a, uint32_t batch,
                                         uint32_t first, uint32_t rows) {
  for (uint64_t i = (uint64_t)blockIdx.x * 256 + threadIdx.x;
       i < (uint64_t)a.q_heads * rows * a.dim; i += (uint64_t)gridDim.x * 256) {
    uint64_t d = i % a.dim, row = (i / a.dim) % rows, head = i / (a.dim * rows);
    writev(y,
           (((uint64_t)batch * a.queries + first + row) * a.q_heads + head) *
                   a.dim +
               d,
           readv(temp, i));
  }
}
static ds4_status bf16_attention(numeric_cuda *s, ds4_num_view y,
                                 ds4_num_view q, ds4_num_view k, ds4_num_view v,
                                 const uint32_t *bounds,
                                 const ds4_num_attention *a) {
  ds4_num_buffer *scratch = NULL, *packed_v = NULL, *packed_k = NULL,
                 *result = NULL;
  const uint32_t tile_rows = 256;
  ds4_status st =
      ds4_num_alloc(s->context, (uint64_t)a->q_heads * tile_rows * a->keys,
                    DS4_NUM_BF16, &scratch);
  if (st)
    return st;
  uint64_t packed_count = (uint64_t)a->batch * a->q_heads * a->keys * a->dim;
  st = ds4_num_alloc(s->context, packed_count, v.type, &packed_v);
  if (!st)
    st = ds4_num_alloc(s->context, packed_count, k.type, &packed_k);
  if (!st)
    st = ds4_num_alloc(s->context, (uint64_t)a->q_heads * tile_rows * a->dim,
                       y.type, &result);
  ds4_num_view vv = ds4_num_data(packed_v), kk = ds4_num_data(packed_k),
               yy = ds4_num_data(result);
  if (!st) {
    pack_attention_kv<<<blocks(packed_count), 256, 0, s->stream>>>(vv, v, *a);
    pack_attention_kv<<<blocks(packed_count), 256, 0, s->stream>>>(kk, k, *a);
    st = launch_status();
  }
  ds4_num_view scores = ds4_num_data(scratch);
  float alpha = 1, beta = 0;
  for (uint32_t batch = 0; batch < a->batch && !st; batch++)
    for (uint32_t first = 0; first < a->queries && !st; first += tile_rows) {
      uint32_t rows =
          a->queries - first < tile_rows ? a->queries - first : tile_rows;
      const char *qp =
          (const char *)q.data +
          ((uint64_t)batch * a->queries + first) * a->q_heads * a->dim * 2;
      const char *kp = (const char *)kk.data +
                       (uint64_t)batch * a->q_heads * a->keys * a->dim * 2;
      const char *vp = (const char *)vv.data +
                       (uint64_t)batch * a->q_heads * a->keys * a->dim * 2;
      cublasStatus_t e = cublasGemmStridedBatchedEx(
          s->blas, CUBLAS_OP_T, CUBLAS_OP_N, a->keys, rows, a->dim, &alpha, kp,
          CUDA_R_16BF, a->dim, (long long)a->keys * a->dim, qp, CUDA_R_16BF,
          a->q_heads * a->dim, a->dim, &beta, scores.data, CUDA_R_16BF, a->keys,
          (long long)rows * a->keys, a->q_heads, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT);
      if (e != CUBLAS_STATUS_SUCCESS) {
        st = DS4_STATUS_INFERENCE;
        break;
      }
      for (uint32_t h = 0; h < a->q_heads; h++)
        softmax_rows_kernel<<<(rows + 3) / 4, 128, 0, s->stream>>>(
            ds4_num_slice(scores, (uint64_t)h * rows * a->keys,
                          (uint64_t)rows * a->keys),
            bounds, batch, first, rows, *a);
      st = launch_status();
      if (st)
        break;
      e = cublasGemmStridedBatchedEx(
          s->blas, CUBLAS_OP_N, CUBLAS_OP_N, a->dim, rows, a->keys, &alpha, vp,
          CUDA_R_16BF, a->dim, (long long)a->keys * a->dim, scores.data,
          CUDA_R_16BF, a->keys, (long long)rows * a->keys, &beta, yy.data,
          CUDA_R_16BF, a->dim, (long long)rows * a->dim, a->q_heads,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      if (e != CUBLAS_STATUS_SUCCESS)
        st = DS4_STATUS_INFERENCE;
      if (!st) {
        scatter_attention<<<blocks((uint64_t)a->q_heads * rows * a->dim), 256,
                            0, s->stream>>>(y, yy, *a, batch, first, rows);
        st = launch_status();
      }
    }
  ds4_status cleanup = ds4_num_dispose(result),
             other = ds4_num_dispose(packed_k);
  if (!cleanup)
    cleanup = other;
  other = ds4_num_dispose(packed_v);
  if (!cleanup)
    cleanup = other;
  other = ds4_num_dispose(scratch);
  if (!cleanup)
    cleanup = other;
  return st ? st : cleanup;
}
static ds4_status gpu_attend(void *p, ds4_num_view y, ds4_num_view q,
                             ds4_num_view k, ds4_num_view v,
                             const uint32_t *bounds,
                             const ds4_num_attention *a) {
  numeric_cuda *s = (numeric_cuda *)p;
  ds4_num_buffer *bb = NULL;
  uint64_t n = (uint64_t)a->batch * a->queries * 2;
  ds4_status st = ds4_num_alloc(s->context, n, DS4_NUM_F32, &bb);
  if (st)
    return st;
  ds4_num_view bv = ds4_num_data(bb);
  st = ds4_num_write(s->context, bv, bounds, n * 4);
  if (!st && q.type == DS4_NUM_BF16 && a->round_scores &&
      a->round_probabilities && !a->relative_keys.data) {
    st = bf16_attention(s, y, q, k, v, (const uint32_t *)bv.data, a);
  } else if (!st) {
    attention_kernel<<<
        (unsigned)(((uint64_t)a->batch * a->queries * a->q_heads + 3) / 4), 128,
        0, s->stream>>>(y, q, k, v, (const uint32_t *)bv.data, *a);
    st = launch_status();
  }
  ds4_status cleanup = ds4_num_dispose(bb);
  return st ? st : cleanup;
}
static ds4_status gpu_conv(void *p, ds4_num_view y, ds4_num_view x,
                           ds4_num_view w, ds4_num_view b,
                           const ds4_num_convolution *a) {
  numeric_cuda *s = (numeric_cuda *)p;
  return ds4_numeric_cudnn_conv(s->cudnn, s->context, y, x, w, b, a);
}
__global__ static void reorder_kernel(ds4_num_view y, ds4_num_view x,
                                      ds4_num_reindex a) {
  for (uint64_t i = (uint64_t)blockIdx.x * 256 + threadIdx.x; i < y.count;
       i += (uint64_t)gridDim.x * 256) {
    uint64_t rem = i, index = 0;
    for (uint32_t d = a.rank; d > 0; d--) {
      index += (rem % a.shape[d - 1]) * a.stride[d - 1];
      rem /= a.shape[d - 1];
    }
    writev(y, i, readv(x, index));
  }
}
static ds4_status gpu_reorder(void *state, ds4_num_view y, ds4_num_view x,
                              const ds4_num_reindex *a) {
  reorder_kernel<<<blocks(y.count), 256, 0, ((numeric_cuda *)state)->stream>>>(
      y, x, *a);
  return launch_status();
}
__global__ static void pool_kernel(ds4_num_view y, ds4_num_view x,
                                   const uint32_t *indices, uint32_t rows,
                                   uint32_t group, uint32_t width,
                                   float divisor) {
  for (uint64_t i = (uint64_t)blockIdx.x * 256 + threadIdx.x;
       i < (uint64_t)rows * width; i += (uint64_t)gridDim.x * 256) {
    uint32_t row = i / width, d = i % width;
    float sum = 0;
    for (uint32_t j = 0; j < group; j++) {
      uint32_t ix = indices[(uint64_t)row * group + j];
      if (ix != UINT32_MAX)
        sum += readv(x, (uint64_t)ix * width + d);
    }
    writev(y, i, sum / divisor);
  }
}
static ds4_status gpu_pool(void *state, ds4_num_view y, ds4_num_view x,
                           const uint32_t *indices, uint32_t rows,
                           uint32_t group, uint32_t width, float divisor) {
  numeric_cuda *s = (numeric_cuda *)state;
  ds4_num_buffer *ib = NULL;
  ds4_status status =
      ds4_num_alloc(s->context, (uint64_t)rows * group, DS4_NUM_F32, &ib);
  if (status)
    return status;
  ds4_num_view v = ds4_num_data(ib);
  status = ds4_num_write(s->context, v, indices, (uint64_t)rows * group * 4);
  if (!status) {
    pool_kernel<<<blocks((uint64_t)rows * width), 256, 0, s->stream>>>(
        y, x, (const uint32_t *)v.data, rows, group, width, divisor);
    status = launch_status();
  }
  ds4_status cleanup = ds4_num_dispose(ib);
  return status ? status : cleanup;
}
__global__ static void nearest_kernel(ds4_num_view y, ds4_num_view x,
                                      ds4_num_resample a) {
  for (uint64_t i = (uint64_t)blockIdx.x * 256 + threadIdx.x; i < y.count;
       i += (uint64_t)gridDim.x * 256) {
    uint64_t rem = i, index = 0, step = 1;
    for (int d = 2; d >= 0; d--) {
      uint64_t pos = rem % a.output[d];
      rem /= a.output[d];
      uint64_t src = ((2 * pos + 1) * a.input[d]) / (2ull * a.output[d]);
      index += src * step;
      step *= a.input[d];
    }
    writev(y, i, readv(x, rem * step + index));
  }
}
static ds4_status gpu_nearest(void *state, ds4_num_view y, ds4_num_view x,
                              const ds4_num_resample *a) {
  nearest_kernel<<<blocks(y.count), 256, 0, ((numeric_cuda *)state)->stream>>>(
      y, x, *a);
  return launch_status();
}
static ds4_status gpu_softmax(void *state, ds4_num_view y, ds4_num_view x,
                              const uint32_t *bounds, uint32_t rows,
                              uint32_t width, float scale) {
  numeric_cuda *s = (numeric_cuda *)state;
  ds4_num_buffer *ib = NULL;
  ds4_status status =
      ds4_num_alloc(s->context, (uint64_t)rows * 2, DS4_NUM_F32, &ib);
  if (status)
    return status;
  ds4_num_view bv = ds4_num_data(ib);
  status = ds4_num_write(s->context, bv, bounds, (uint64_t)rows * 8);
  if (!status && y.data != x.data)
    status = ds4_num_copy(s->context, y, x);
  if (!status) {
    ds4_num_attention a = {};
    a.batch = 1;
    a.queries = rows;
    a.keys = width;
    a.scale = scale;
    softmax_rows_kernel<<<(rows + 3) / 4, 128, 0, s->stream>>>(
        y, (const uint32_t *)bv.data, 0, 0, rows, a);
    status = launch_status();
  }
  ds4_status cleanup = ds4_num_dispose(ib);
  return status ? status : cleanup;
}
static ds4_status gpu_matmul_batch(void *p, ds4_num_view y, ds4_num_view x,
                                   ds4_num_view w, uint32_t batch,
                                   uint32_t rows, uint32_t in, uint32_t out,
                                   bool transpose_b) {
  numeric_cuda *s = (numeric_cuda *)p;
  float alpha = 1, beta = 0;
  cublasStatus_t status = cublasSgemmStridedBatched(
      s->blas, transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, out, rows,
      in, &alpha, (const float *)w.data, transpose_b ? in : out,
      (long long)out * in, (const float *)x.data, in, (long long)rows * in,
      &beta, (float *)y.data, out, (long long)rows * out, batch);
  return status == CUBLAS_STATUS_SUCCESS ? DS4_STATUS_OK : DS4_STATUS_INFERENCE;
}
static const ds4_numeric_ops gpu_ops = {
    gpu_allocate, gpu_release, gpu_copy,    gpu_fence,
    gpu_close,    gpu_linear,  gpu_norm,    gpu_point,
    gpu_rope,     gpu_attend,  gpu_conv,    gpu_reorder,
    gpu_pool,     gpu_nearest, gpu_softmax, gpu_matmul_batch};
extern "C" ds4_status ds4_numeric_cuda(ds4_numeric **out, ds4_resources *r,
                                       uint64_t owner) {
  if (!out || !r || !owner)
    return DS4_STATUS_INVALID;
  *out = NULL;
  numeric_cuda *s = (numeric_cuda *)calloc(1, sizeof(*s));
  if (!s)
    return DS4_STATUS_NOMEM;
  ds4_resource_spec metadata = {};
  metadata.owner = owner;
  metadata.location = DS4_MEMORY_HOST;
  metadata.device = -1;
  metadata.requested = metadata.allocated = metadata.live = sizeof(*s);
  ds4_status admitted = ds4_resource_register(r, &metadata, &s->metadata);
  if (admitted) {
    free(s);
    return admitted;
  }
  ds4_status st =
      status(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking));
  if (st) {
    ds4_resource_release(s->metadata);
    free(s);
    return st;
  }
  if (cublasCreate(&s->blas) != CUBLAS_STATUS_SUCCESS) {
    cudaStreamDestroy(s->stream);
    ds4_resource_release(s->metadata);
    free(s);
    return DS4_STATUS_NOMEM;
  }
  if (cublasLtCreate(&s->lt) != CUBLAS_STATUS_SUCCESS) {
    gpu_close(s);
    return DS4_STATUS_NOMEM;
  }
  if (cublasSetStream(s->blas, s->stream) != CUBLAS_STATUS_SUCCESS ||
      cublasSetMathMode(s->blas,
                        CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION) !=
          CUBLAS_STATUS_SUCCESS) {
    gpu_close(s);
    return DS4_STATUS_INFERENCE;
  }
  st = ds4_numeric_cudnn_create(&s->cudnn, (void *)s->stream, r, owner);
  if (st) {
    gpu_close(s);
    return st;
  }
  st = ds4_numeric_create(out, &gpu_ops, s, r, owner, true);
  if (st)
    gpu_close(s);
  else {
    s->context = *out;
    const uint64_t bytes = 8519680;
    ds4_resource_spec spec = {};
    spec.owner = owner;
    spec.location = DS4_MEMORY_DEVICE;
    spec.device = 0;
    spec.requested = spec.allocated = spec.live = bytes;
    st = ds4_resource_register(r, &spec, &s->workspace_account);
    if (!st)
      st = status(cudaMalloc(&s->workspace, bytes));
    if (!st && cublasSetWorkspace(s->blas, s->workspace, bytes) !=
                   CUBLAS_STATUS_SUCCESS)
      st = DS4_STATUS_INFERENCE;
    if (st) {
      ds4_numeric_close(*out);
      *out = NULL;
    } else {
      s->context->backend_bytes = bytes;
      s->context->peak_bytes = bytes;
    }
  }
  return st;
}
