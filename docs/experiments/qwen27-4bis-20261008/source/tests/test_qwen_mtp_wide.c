/* Exact parity of every wide-verifier rollback frontier when the per-layer
 * GPU row snapshots are reused and their immutable contents live in RAM. */
#include "../ds4.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %d: %s (%s)\n", \
    __LINE__, #c, err); exit(1); } } while (0)

static uint64_t wide_hash_bytes(uint64_t hash, const void *data, uint64_t bytes) {
    const unsigned char *p = data;
    for (uint64_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

static uint64_t state_hash(ds4_qwen_gpu_graph *g, unsigned char *scratch, char *err) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t il = 0; il < DS4_N_LAYER; ++il) if (qwen_layer_is_recurrent(il)) {
        ds4_gpu_tensor *states[] = {g->conv_state[il], g->recurrent_state[il]};
        for (unsigned j = 0; j < 2; ++j) {
            uint64_t bytes = ds4_gpu_tensor_bytes(states[j]);
            CHECK(ds4_gpu_tensor_read(states[j], 0, scratch, bytes));
            hash = wide_hash_bytes(hash, scratch, bytes);
        }
    }
    return hash;
}

static void projection_parity(ds4_engine *e, uint32_t rows, char *err) {
    const ds4_tensor *weights[24] = {0};
    unsigned available = 0;
    /* The first layer's FFN may use IQ formats. Inspect all projections so
     * the test cannot silently skip the Q4/Q5 kernels it is qualifying. */
    for (uint32_t il = 0; il < DS4_N_LAYER; ++il) {
        const ds4_layer_weights *l = &e->weights.layer[il];
        const ds4_tensor *candidates[] = {l->ffn_gate, l->ffn_up, l->ffn_down,
            l->qwen_attn_qkv, l->qwen_attn_gate, l->qwen_ssm_out,
            l->qwen_ssm_alpha, l->qwen_ssm_beta, l->qwen_attn_q,
            l->qwen_attn_k, l->qwen_attn_v, l->attn_output, e->weights.output};
        for (unsigned i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
            const ds4_tensor *w = candidates[i];
            if (w && w->type < 24u && w->ndim == 2) {
                weights[w->type] = w;
                available |= 1u << w->type;
            }
        }
    }
    CHECK((available & (1u<<12)) && (available & (1u<<14)));
    for (unsigned iw = 0; iw < 24; ++iw) {
        const ds4_tensor *w = weights[iw];
        if (!w) continue;
        const uint64_t in = w->dim[0], out = w->dim[1];
        float *x = malloc((size_t)rows * in * sizeof(float));
        float *batch = malloc((size_t)rows * out * sizeof(float));
        float *serial = malloc((size_t)out * sizeof(float));
        CHECK(x && batch && serial);
        for (uint64_t i = 0; i < (uint64_t)rows * in; ++i)
            x[i] = (float)((int)((i * 131u + 73u) % 997u) - 498) / 1024.0f;
        ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc((uint64_t)rows * in * sizeof(float));
        ds4_gpu_tensor *gy = ds4_gpu_tensor_alloc((uint64_t)rows * out * sizeof(float));
        ds4_gpu_tensor *one = ds4_gpu_tensor_alloc(out * sizeof(float));
        CHECK(gx && gy && one && ds4_gpu_tensor_write(gx, 0, x, (uint64_t)rows * in * sizeof(float)));
        ds4_qwen_execution_context context = {DS4_QWEN_STAGE_MTP_VERIFY, 0};
        CHECK(qwen_graph_matmul(context, gy, &e->model, w, in, out, gx, rows));
        CHECK(ds4_gpu_tensor_read(gy, 0, batch, (uint64_t)rows * out * sizeof(float)));
        context.stage = DS4_QWEN_STAGE_DECODE;
        for (uint32_t row = 0; row < rows; ++row) {
            ds4_gpu_tensor *view = ds4_gpu_tensor_view(gx, (uint64_t)row * in * sizeof(float), in * sizeof(float));
            CHECK(view && qwen_graph_matmul(context, one, &e->model, w, in, out, view, 1));
            CHECK(ds4_gpu_tensor_read(one, 0, serial, out * sizeof(float)));
            if (memcmp(batch + (uint64_t)row * out, serial, out * sizeof(float)))
                fprintf(stderr,"projection mismatch type=%u in=%" PRIu64 " out=%" PRIu64 " T=%u row=%u\n",w->type,in,out,rows,row);
            CHECK(!memcmp(batch + (uint64_t)row * out, serial, out * sizeof(float)));
            ds4_gpu_tensor_free(view);
        }
        fprintf(stderr, "PASS: quant type=%u shape=%" PRIu64 "x%" PRIu64 " rows=%u, "
            "every wide projection bit-exact with scalar decode\n", w->type, in, out, rows);
        ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gy); ds4_gpu_tensor_free(one);
        free(x); free(batch); free(serial);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char err[256] = {0};
    const uint32_t widths[] = {3, 5, 7, 9, 11, 13, 14, 15};
    uint64_t expected_state[15], expected_logits[15], expected_continuation[15];
    unsigned char *scratch = malloc(16u * 1024u * 1024u);
    CHECK(scratch);
    for (unsigned iw = 0; iw < sizeof(widths) / sizeof(widths[0]); ++iw) {
      const uint32_t width = widths[iw];
      const char *only_width = getenv("DS4_TEST_QWEN_MTP_WIDTH");
      if (only_width && width != (uint32_t)strtoul(only_width, NULL, 10)) continue;
      const bool weight_stream = ds4_env_flag_enabled("DS4_TEST_QWEN_WEIGHT_STREAMING", 0);
      for (unsigned variant = 0; variant < (weight_stream ? 3u : 2u); ++variant) {
        const unsigned host = variant == 1u;
        if (weight_stream)
            CHECK(setenv("DS4_CUDA_QWEN_WEIGHT_STREAMING", variant == 2u ? "1" : "0", 1) == 0);
        CHECK(setenv("DS4_CUDA_QWEN_MTP_HOST_ROWS", host ? "1" : "0", 1) == 0);
        CHECK(setenv("DS4_CUDA_QWEN_KV_CAPACITY_MODE", host ? "hybrid-attention" : "off", 1) == 0);
        CHECK(setenv("DS4_CUDA_QWEN_KV_RESIDENT", "64", 1) == 0);
        CHECK(setenv("DS4_CUDA_QWEN_MTP_KV_MODE", host ? "resident" : "shared", 1) == 0);
        ds4_engine *e = NULL;
        ds4_session *s = NULL;
        ds4_engine_options options = {.model_path = argv[1], .mtp_path = argv[2],
            .backend = DS4_BACKEND_CUDA, .context_size = 512,
            .prefill_chunk = 256, .mtp_draft_tokens = (int)width};
        CHECK(ds4_engine_open(&e, &options) == 0 && ds4_engine_has_mtp(e));
        CHECK(ds4_engine_mtp_draft_tokens(e) == (int)width);
        if (variant == 0u) {
            if (!iw) for (uint32_t tail = 2; tail <= 5; ++tail)
                projection_parity(e, tail, err);
            projection_parity(e, width + 1u, err);
        }
        CHECK(ds4_session_create(&s, e, options.context_size) == 0);
        ds4_tokens pattern = {0}, prompt = {0};
        ds4_tokenize_text(e, "Write a clear C function and explain its boundary values.\n", &pattern);
        CHECK(pattern.len > 0);
        for (unsigned i = 0; i < 256; ++i) ds4_tokens_push(&prompt, pattern.v[i % pattern.len]);
        CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
        ds4_qwen_gpu_graph *g = &s->qwen_graph;
        CHECK(g->mtp_ready && g->mtp_snapshot_rows == width && g->mtp_verify_rows == width + 1u);
        CHECK(g->mtp_host_rows == (host != 0));
        CHECK(g->mtp_decode_rows == NULL);
        int tokens[16], tops[16];
        for (uint32_t i = 0; i <= width; ++i) tokens[i] = pattern.v[i % pattern.len];
        CHECK(qwen_graph_forward_rows(g, &e->model, &e->weights, tokens, prompt.len,
            width + 1u, DS4_QWEN_STAGE_MTP_VERIFY, s->logits, tops));
        const bool decode_rows = host && ds4_env_flag_enabled("DS4_CUDA_QWEN_MTP_DECODE_ROWS", 0) &&
            ds4_env_flag_enabled("DS4_TEST_QWEN_MTP_EXPECT_ROWS", 1);
        CHECK((g->mtp_decode_rows != NULL) == decode_rows);
        if (decode_rows) CHECK(g->mtp_decode_rows_valid == width);
        const char *resident = getenv("DS4_TEST_QWEN_MTP_RESIDENT_ROWS");
        if (host && resident) CHECK(g->mtp_decode_rows_capacity == (uint32_t)strtoul(resident, NULL, 10));
        if (host && ds4_env_flag_enabled("DS4_CUDA_QWEN_MTP_DECODE_ROWS", 0))
            CHECK(g->mtp_decode_rows_tried);
        for (uint32_t row = 0; row < width; ++row) {
            CHECK(qwen_graph_mtp_restore_recurrent_row(g, row));
            CHECK(qwen_graph_mtp_select_verifier_row(g, row, s->logits));
            uint64_t hs = state_hash(g, scratch, err);
            uint64_t hl = wide_hash_bytes(0, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float));
            CHECK(qwen_graph_forward_token(g, &e->model, &e->weights,
                (uint32_t)tokens[row + 1u], prompt.len + row + 1u, s->logits));
            uint64_t hc = wide_hash_bytes(0, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float));
            if (variant == 0u) {
                expected_state[row] = hs; expected_logits[row] = hl; expected_continuation[row] = hc;
            } else {
                CHECK(hs == expected_state[row]); CHECK(hl == expected_logits[row]);
                CHECK(hc == expected_continuation[row]);
            }
        }
        if (host) {
            /* Lean V(2) has no prebatch snapshot: its row-zero GPU reuse
             * must reproduce the independent full GPU-row oracle above. */
            ds4_session_invalidate(s);
            CHECK(!g->mtp_decode_rows_valid);
            CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
            CHECK(g->mtp_decode_rows == NULL && !g->mtp_decode_rows_tried);
            g->mtp_row0_gpu = true;
            CHECK(qwen_graph_forward_rows(g, &e->model, &e->weights, tokens, prompt.len,
                2u, DS4_QWEN_STAGE_MTP_VERIFY, s->logits, tops));
            CHECK(qwen_graph_mtp_restore_recurrent_row(g, 0u));
            CHECK(!qwen_graph_mtp_restore_recurrent_row(g, 1u));
            CHECK(qwen_graph_mtp_select_verifier_row(g, 0u, s->logits));
            CHECK(state_hash(g, scratch, err) == expected_state[0]);
            CHECK(wide_hash_bytes(0, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) == expected_logits[0]);
            CHECK(qwen_graph_forward_token(g, &e->model, &e->weights,
                (uint32_t)tokens[1], prompt.len + 1u, s->logits));
            CHECK(wide_hash_bytes(0, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) == expected_continuation[0]);
            /* Reenter wide verification after the prefill returned its arena;
             * subsequent cycles reuse it without allocating another bank. */
            if (decode_rows) {
                CHECK(qwen_graph_forward_rows(g, &e->model, &e->weights, tokens,
                    prompt.len + 2u, width + 1u, DS4_QWEN_STAGE_MTP_VERIFY, s->logits, tops));
                ds4_gpu_tensor *arena = g->mtp_decode_rows;
                CHECK(arena && qwen_graph_mtp_restore_recurrent_row(g, width - 1u));
                CHECK(qwen_graph_forward_rows(g, &e->model, &e->weights, tokens,
                    prompt.len + 2u, width + 1u, DS4_QWEN_STAGE_MTP_VERIFY, s->logits, tops));
                CHECK(g->mtp_decode_rows == arena);
            }
            uint64_t one_layer = ds4_gpu_tensor_bytes(g->mtp_conv_rows_scratch) +
                ds4_gpu_tensor_bytes(g->mtp_recurrent_rows_scratch);
            uint64_t host_bytes = 0;
            for (uint32_t il = 0; il < DS4_N_LAYER; ++il) if (qwen_layer_is_recurrent(il))
                host_bytes += (ds4_gpu_tensor_bytes(g->conv_state[il]) +
                    ds4_gpu_tensor_bytes(g->recurrent_state[il])) * width;
            CHECK(one_layer > 0 && host_bytes == one_layer * 48u);
            fprintf(stderr, "PASS: %u proposals, every rollback state/logit/continuation bit-exact; "
                "lean V(2) GPU reuse exact; host=%" PRIu64 " shared_GPU=%" PRIu64 " bytes\n", width, host_bytes, one_layer);
        }
        if (variant == 2u)
            fprintf(stderr, "PASS: %u proposals, weight streaming matches the independent resident "
                "oracle for every recurrent/conv state, complete logits and continuation\n", width);
        ds4_tokens_free(&prompt); ds4_tokens_free(&pattern);
        ds4_session_free(s); ds4_engine_close(e);
      }
    }
    free(scratch);
    return 0;
}
