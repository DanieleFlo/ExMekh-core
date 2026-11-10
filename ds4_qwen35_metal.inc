/* Qwen27 wrappers use the selected resident or bounded SSD weight owner.
 * All bounds are checked before encoding mutating kernels. */
typedef struct {
    uint32_t pos, rows, capacity, spatial, snapshots;
} qwen35_args;
static int qwen35_encode(const char *name, const void *args, size_t args_bytes,
                         const qwen4_bind *binds, unsigned count, MTLSize grid, unsigned threads) {
    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(name);
    /* These reductions use 32-lane SIMD groups, independent of the chip name. */
    if (!pipeline || pipeline.threadExecutionWidth != 32 ||
        threads > pipeline.maxTotalThreadsPerThreadgroup ||
        pipeline.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength)
        return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb)
        return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    if (!enc)
        return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:args_bytes atIndex:0];
    for (unsigned i = 0; i < count; ++i)
        [enc setBuffer:binds[i].buf offset:binds[i].off atIndex:i + 1];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, name);
}
static int qwen35_dispatch(const char *name, const qwen35_args *args, const qwen4_bind *binds,
                           unsigned count, MTLSize grid, unsigned threads) {
    return qwen35_encode(name, args, sizeof(*args), binds, count, grid, threads);
}

static int qwen35_ud_supported(uint32_t type) {
    return type == 11 || type == 13 || type == 14 || type == 17 || type == 18 ||
           (type >= 20 && type <= 23);
}
/* Streaming keeps only the requested rows. Capture IDs in command order and
 * validate the entire batch before its first output write. Scalar lookups
 * own the quantized row binding; this orchestration only borrows tensor views. */
static int qwen35_stream_gather_rows(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens,
                                    const void *map, uint64_t size, uint64_t offset,
                                    uint32_t type, uint32_t width, uint32_t rows, uint32_t vocab) {
    uint64_t row_bytes = (uint64_t)width * sizeof(float);
    if (ds4_gpu_tensor_bytes(out) < rows * row_bytes)
        return 0;
    int32_t *ids = malloc((size_t)rows * sizeof(*ids));
    bool ok = ids && ds4_gpu_tensor_read(tokens, 0, ids, (uint64_t)rows * sizeof(*ids));
    for (uint32_t i = 0; ok && i < rows; i++)
        ok = ids[i] >= 0 && (uint32_t)ids[i] < vocab;
    for (uint32_t i = 0; ok && i < rows; i++) {
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(out, i * row_bytes, row_bytes);
        ok = row && ds4_gpu_embed_token_quant_tensor(row, map, size, offset, type,
                                                     vocab, (uint32_t)ids[i], width);
        ds4_gpu_tensor_free(row);
    }
    free(ids);
    return ok;
}
static int qwen35_ud_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens,
                            const ds4_gpu_tensor *x, const void *map, uint64_t size,
                            uint64_t offset, uint32_t type, uint64_t in_dim, uint64_t out_dim,
                            uint64_t rows, uint32_t vocab, uint32_t token) {
    if (!g_initialized && !ds4_gpu_init())
        return 0;
    if (!qwen35_ud_supported(type) || !rows || rows > UINT32_MAX || !in_dim ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || (x ? !out_dim : !vocab) ||
        (!x && !tokens && token >= vocab))
        return 0;
    uint32_t block = type == 20 ? 32 : 256;
    uint32_t bytes = type == 11 || type == 21 ? 110
                     : type == 13             ? 176
                     : type == 14             ? 210
                     : type == 17             ? 74
                     : type == 18             ? 98
                     : type == 20             ? 18
                     : type == 22             ? 82
                                              : 136;
    if (in_dim % block || in_dim / block > UINT32_MAX / bytes)
        return 0;
    uint64_t row_bytes = in_dim / block * bytes, weight_rows = x ? out_dim : vocab;
    uint64_t output_width = x ? out_dim : in_dim;
    if (weight_rows > UINT64_MAX / row_bytes || rows > UINT64_MAX / output_width / 4 ||
        rows > UINT64_MAX / in_dim / 4)
        return 0;
    uint64_t weight_bytes = weight_rows * row_bytes;
    if (offset > size || weight_bytes > size - offset)
        return 0;
    if (!x && tokens && qwen4_stream_matches(map, size))
        return qwen35_stream_gather_rows(out, tokens, map, size, offset, type,
                                          (uint32_t)in_dim, (uint32_t)rows, vocab);
    /* A scalar lookup consumes one quantized row. Keep the original table
     * bounds check, then rebase the unchanged gather kernel onto that row;
     * streaming the whole vocabulary here rereads it for every decode token. */
    if (!x && !tokens) {
        offset += (uint64_t)token * row_bytes;
        weight_bytes = row_bytes;
        token = 0;
        vocab = 1;
    }
    struct {
        uint32_t in_dim, out_dim, rows, row_bytes, vocab, token, gather;
    } args = {
        (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)rows, (uint32_t)row_bytes, vocab, token,
        tokens != NULL};
    qwen4_bind b[3];
    if (!qwen4_bind_weight(&b[0], map, size, offset, weight_bytes, "27B UD weights") ||
        !qwen4_bind_tensor(&b[1],
                           x        ? x
                           : tokens ? tokens
                                    : out,
                           x        ? rows * in_dim * 4
                           : tokens ? rows * 4
                                    : 4,
                           "27B UD input") ||
        !qwen4_bind_tensor(&b[2], out, rows * output_width * 4, "27B UD output"))
        return 0;
    char name[64];
    /* A depth-four verifier has five rows. Keep one weight decode for that
     * batch without imposing its register tile on the ordinary prefill. */
    const char *operation = "gather";
    uint32_t token_tile = 4, output_tile = 4;
    if (x) {
        operation = "project";
        if (rows == 1) {
            operation = "project_pair";
            output_tile = 8;
        } else if (rows == 5) {
            operation = "project5";
            token_tile = 5;
        }
    }
    snprintf(name, sizeof(name), "ds4_qwen_ud_%s_%u", operation, type);
    uint32_t unit = (type == 17 || type == 18 || type == 21 || type == 22) ? 16 : 32;
    MTLSize grid = x ? MTLSizeMake((out_dim + output_tile - 1) / output_tile,
                                   (rows + token_tile - 1) / token_tile, 1)
                     : MTLSizeMake((in_dim / unit + 63) / 64, rows, 1);
    return qwen35_encode(name, &args, sizeof(args), b, 3, grid, x ? 128 : 64);
}

static int qwen35_attention_impl(ds4_gpu_tensor *out, ds4_gpu_tensor *q, ds4_gpu_tensor *k,
                                 ds4_gpu_tensor *v, ds4_gpu_tensor *kc, ds4_gpu_tensor *vc,
                                 const void *map, uint64_t size, uint64_t qoff, uint64_t koff,
                                 uint32_t pos, uint32_t rows, uint32_t capacity,
                                 const ds4_gpu_tensor *positions) {
    qwen35_args args = {pos, rows, capacity, positions != NULL, 0};
    const uint64_t cache = (uint64_t)capacity * 1024 * sizeof(float);
    qwen4_bind b[6] = {0}, qb[3] = {0}, ab[4] = {0};
    if (!rows || pos >= capacity || rows > capacity - pos ||
        !qwen4_bind_tensor(&b[0], k, (uint64_t)rows * 1024 * 4, "27B K") ||
        !qwen4_bind_tensor(&b[1], v, (uint64_t)rows * 1024 * 4, "27B V") ||
        !qwen4_bind_tensor(&b[2], kc, cache, "27B KV") ||
        !qwen4_bind_tensor(&b[3], vc, cache, "27B KV") ||
        !qwen4_bind_weight(&b[4], map, size, koff, 256 * 4, "27B K norm") ||
        !qwen4_bind_tensor(&b[5], positions ? positions : k, positions ? (uint64_t)rows * 12 : 4,
                           "27B positions"))
        return 0;
    if (out && (!qwen4_bind_tensor(&qb[0], q, (uint64_t)rows * 12288 * 4, "27B Q") ||
                !qwen4_bind_weight(&qb[1], map, size, qoff, 256 * 4, "27B Q norm") ||
                !qwen4_bind_tensor(&ab[0], out, (uint64_t)rows * 6144 * 4, "27B attention")))
        return 0;
    qb[2] = b[5];
    ab[1] = qb[0];
    ab[2] = b[2];
    ab[3] = b[3];
    if (out &&
        !qwen35_dispatch("ds4_qwen35_prepare_q", &args, qb, 3, MTLSizeMake(24, rows, 1), 256))
        return 0;
    if (!qwen35_dispatch("ds4_qwen35_prepare_kv", &args, b, 6, MTLSizeMake(4, rows, 1), 256))
        return 0;
    return !out ||
           qwen35_dispatch("ds4_qwen35_attention", &args, ab, 4, MTLSizeMake(24, rows, 1), 256);
}

int ds4_gpu_qwen35_full_attention_rows_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *q,
                                              ds4_gpu_tensor *k, ds4_gpu_tensor *v,
                                              ds4_gpu_tensor *kc, ds4_gpu_tensor *vc,
                                              const void *map, uint64_t size, uint64_t qo,
                                              uint64_t ko, uint32_t pos, uint32_t rows,
                                              uint32_t cap) {
    if (!out)
        return 0;
    return qwen35_attention_impl(out, q, k, v, kc, vc, map, size, qo, ko, pos, rows, cap, NULL);
}
int ds4_gpu_qwen35_full_attention_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *q, ds4_gpu_tensor *k,
                                         ds4_gpu_tensor *v, ds4_gpu_tensor *kc, ds4_gpu_tensor *vc,
                                         const void *map, uint64_t size, uint64_t qo, uint64_t ko,
                                         uint32_t pos, uint32_t cap) {
    return ds4_gpu_qwen35_full_attention_rows_tensor(out, q, k, v, kc, vc, map, size, qo, ko, pos,
                                                     1, cap);
}
int ds4_gpu_qwen35_full_attention_mrope_rows_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *q,
                                                    ds4_gpu_tensor *k, ds4_gpu_tensor *v,
                                                    ds4_gpu_tensor *kc, ds4_gpu_tensor *vc,
                                                    const void *map, uint64_t size, uint64_t qo,
                                                    uint64_t ko, uint32_t pos, uint32_t rows,
                                                    uint32_t cap, const ds4_gpu_tensor *positions) {
    if (!out || !positions)
        return 0;
    return qwen35_attention_impl(out, q, k, v, kc, vc, map, size, qo, ko, pos, rows, cap,
                                 positions);
}
int ds4_gpu_qwen35_store_kv_rows_tensor(ds4_gpu_tensor *k, ds4_gpu_tensor *v, ds4_gpu_tensor *kc,
                                        ds4_gpu_tensor *vc, const void *map, uint64_t size,
                                        uint64_t ko, uint32_t pos, uint32_t rows, uint32_t cap) {
    return qwen35_attention_impl(NULL, NULL, k, v, kc, vc, map, size, 0, ko, pos, rows, cap, NULL);
}
int ds4_gpu_qwen35_store_kv_mrope_rows_tensor(ds4_gpu_tensor *k, ds4_gpu_tensor *v,
                                              ds4_gpu_tensor *kc, ds4_gpu_tensor *vc,
                                              const void *map, uint64_t size, uint64_t ko,
                                              uint32_t pos, uint32_t rows, uint32_t cap,
                                              const ds4_gpu_tensor *positions) {
    if (!positions)
        return 0;
    return qwen35_attention_impl(NULL, NULL, k, v, kc, vc, map, size, 0, ko, pos, rows, cap,
                                 positions);
}

int ds4_gpu_qwen35_gated_delta_net_rows_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *qkv,
                                               const ds4_gpu_tensor *z, const ds4_gpu_tensor *alpha,
                                               const ds4_gpu_tensor *beta, ds4_gpu_tensor *conv,
                                               ds4_gpu_tensor *state, ds4_gpu_tensor *conv_snap,
                                               ds4_gpu_tensor *state_snap, const void *map,
                                               uint64_t size, uint64_t ao, uint64_t co,
                                               uint64_t dto, uint64_t no, uint32_t rows) {
    const uint64_t cb = 10240ull * 4 * 4, sb = 48ull * 128 * 128 * 4;
    qwen4_bind b[10], c[4];
    qwen35_args args = {0, rows, 0, 0, state_snap != NULL};
    if (!rows || !qwen4_bind_tensor(&b[0], out, (uint64_t)rows * 6144 * 4, "27B GDN output") ||
        !qwen4_bind_tensor(&b[1], qkv, (uint64_t)rows * 10240 * 4, "27B QKV") ||
        !qwen4_bind_tensor(&b[2], z, (uint64_t)rows * 6144 * 4, "27B Z") ||
        !qwen4_bind_tensor(&b[3], alpha, (uint64_t)rows * 48 * 4, "27B alpha") ||
        !qwen4_bind_tensor(&b[4], beta, (uint64_t)rows * 48 * 4, "27B beta") ||
        !qwen4_bind_tensor(&b[5], state, sb, "27B recurrent state") ||
        !qwen4_bind_weight(&b[6], map, size, ao, 48 * 4, "27B decay") ||
        !qwen4_bind_weight(&b[7], map, size, dto, 48 * 4, "27B delta bias") ||
        !qwen4_bind_weight(&b[8], map, size, no, 128 * 4, "27B GDN norm") ||
        !qwen4_bind_tensor(&b[9], state_snap ? state_snap : state,
                           state_snap ? (uint64_t)(rows - 1) * sb : sb, "27B state rollback") ||
        !qwen4_bind_tensor(&c[1], conv, cb, "27B convolution state") ||
        !qwen4_bind_weight(&c[2], map, size, co, cb, "27B convolution weights") ||
        !qwen4_bind_tensor(&c[3], conv_snap ? conv_snap : conv,
                           conv_snap ? (uint64_t)(rows - 1) * cb : cb, "27B convolution rollback"))
        return 0;
    c[0] = b[1];
    qwen35_args ca = args;
    ca.snapshots = conv_snap != NULL;
    return qwen35_dispatch("ds4_qwen35_conv", &ca, c, 4, MTLSizeMake(40, 1, 1), 256) &&
           qwen35_dispatch("ds4_qwen35_gdn", &args, b, 10, MTLSizeMake(48, 1, 1), 512);
}
int ds4_gpu_qwen35_gated_delta_net_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *qkv,
                                          const ds4_gpu_tensor *z, const ds4_gpu_tensor *alpha,
                                          const ds4_gpu_tensor *beta, ds4_gpu_tensor *conv,
                                          ds4_gpu_tensor *state, const void *map, uint64_t size,
                                          uint64_t ao, uint64_t co, uint64_t dto, uint64_t no) {
    return ds4_gpu_qwen35_gated_delta_net_rows_tensor(out, qkv, z, alpha, beta, conv, state, NULL,
                                                      NULL, map, size, ao, co, dto, no, 1);
}
