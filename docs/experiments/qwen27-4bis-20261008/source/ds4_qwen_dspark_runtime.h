/* Reuse DS4 embeddings/GEMMs/norms/SwiGLU, target head and Qwen verifier. */
static bool qwen_dspark_graph_enable(ds4_qwen_gpu_graph *g, const ds4_model *m,
        const ds4_qwen_dspark_weights *weights, uint32_t width) {
#if defined(__linux__) && !defined(DS4_ROCM_BUILD)
    ds4_qwen_dspark_graph *d = calloc(1, sizeof(*d));
    if (!d || !g->mtp_ready || !width || width > 15) { free(d); return false; }
    d->model = m; d->weights = weights; d->context = g->ctx_cap;
    d->cap = g->prefill_cap; d->width = width;
#define QDSP_ALLOC(field, bytes) do { d->field = ds4_gpu_tensor_alloc(bytes); if (!d->field) goto fail; } while (0)
    QDSP_ALLOC(features, (uint64_t)d->cap * 25600 * 4);
    QDSP_ALLOC(projected, (uint64_t)d->cap * 5120 * 4);
    QDSP_ALLOC(normalized, (uint64_t)d->cap * 5120 * 4);
    QDSP_ALLOC(k, (uint64_t)d->cap * 1024 * 4); QDSP_ALLOC(v, (uint64_t)d->cap * 1024 * 4);
    QDSP_ALLOC(half, (uint64_t)d->cap * 1024 * 2);
    QDSP_ALLOC(markov, 256 * 4); QDSP_ALLOC(bias, 248320 * 4);
    QDSP_ALLOC(corrected, 248320 * 4); QDSP_ALLOC(confidence_input, 5376 * 4);
    QDSP_ALLOC(confidence_value, 4);
#undef QDSP_ALLOC
    for (unsigned i = 0; i < width; ++i) {
        d->base_rows[i] = ds4_gpu_tensor_view(g->mtp_verify_logits, (uint64_t)i * 248320 * 4, 248320 * 4);
        d->hidden_rows[i] = ds4_gpu_tensor_view(g->mtp_target_rows, (uint64_t)i * 5120 * 4, 5120 * 4);
        if (!d->base_rows[i] || !d->hidden_rows[i]) goto fail;
    }
    for (unsigned i = 0; i < 5; ++i) {
        d->keys[i] = malloc((uint64_t)d->context * 1024 * 2);
        d->values[i] = malloc((uint64_t)d->context * 1024 * 2);
        if (!d->keys[i] || !d->values[i]) goto fail;
    }
    d->attention_workspace = ds4_gpu_qwen_dspark_workspace_alloc(d->context, width);
    if (!d->attention_workspace) goto fail;
    ds4_gpu_tensor_free(g->mtp_key_cache); g->mtp_key_cache = NULL;
    ds4_gpu_tensor_free(g->mtp_value_cache); g->mtp_value_cache = NULL;
    g->dspark = d;
    fprintf(stderr,"ds4: Qwen DSpark session ctx=%u width=%u draft_RAM_KV=%.3f GiB target_KV=FP32 staging=shared-bounded\n",
        d->context,d->width,(double)(10ull*d->context*1024*2)/1073741824.0);
    return true;
fail:
    qwen_dspark_graph_free(d); return false;
#else
    (void)g; (void)m; (void)weights; (void)width; return false;
#endif
}

static bool qwen_dspark_capture_layer(ds4_qwen_gpu_graph *g, uint32_t layer, uint32_t rows) {
#if defined(__linux__) && !defined(DS4_ROCM_BUILD)
    if (!g->dspark) return true;
    static const uint32_t taps[] = {5, 19, 33, 47, 61};
    for (unsigned i = 0; i < 5; ++i) if (taps[i] == layer)
        return ds4_gpu_qwen_dspark_features_tensor(g->dspark->features, g->cur, rows, i) != 0;
    return true;
#else
    (void)g; (void)layer; (void)rows; return true;
#endif
}

static bool qwen_dspark_capture_finish(ds4_qwen_gpu_graph *g, uint32_t position, uint32_t rows) {
#if defined(__linux__) && !defined(DS4_ROCM_BUILD)
    ds4_qwen_dspark_graph *d = g->dspark; if (!d) return true;
    if (position > d->frontier || position > d->context || rows > d->context-position) return false;
    const ds4_model *m = d->model; const ds4_qwen_dspark_weights *w = d->weights;
    const ds4_qwen_execution_context ctx = {DS4_QWEN_STAGE_PREFILL, 0};
    bool ok = qwen_graph_matmul(ctx, d->projected, m, w->fc, 25600, 5120, d->features, rows);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(d->normalized, d->projected,
        m->map, m->size, w->hidden_norm->abs_offset, 5120, rows, 1e-6f) != 0;
    for (unsigned i = 0; ok && i < 5; ++i) {
        const ds4_qwen_dspark_layer *l = &w->layer[i];
        ok = qwen_graph_matmul(ctx, d->k, m, l->k, 5120, 1024, d->normalized, rows) &&
             qwen_graph_matmul(ctx, d->v, m, l->v, 5120, 1024, d->normalized, rows);
        if (ok) ok = ds4_gpu_qwen_dspark_norm_rope_tensor(d->k, m->map, m->size,
            l->k_norm->abs_offset, 8, rows, position) != 0;
        if (ok) ok = ds4_gpu_tensor_copy_f32_to_f16(d->half, 0, d->k, 0, (uint64_t)rows*1024) &&
            ds4_gpu_tensor_read(d->half, 0, d->keys[i]+(uint64_t)position*1024, (uint64_t)rows*1024*2);
        if (ok) ok = ds4_gpu_tensor_copy_f32_to_f16(d->half, 0, d->v, 0, (uint64_t)rows*1024) &&
            ds4_gpu_tensor_read(d->half, 0, d->values[i]+(uint64_t)position*1024, (uint64_t)rows*1024*2);
    }
    if (ok) {
        d->frontier = position + rows;
        g->mtp_cache_len = position + rows;
    }
    return ok;
#else
    (void)g; (void)position; (void)rows; return true;
#endif
}

static bool qwen_dspark_propose(ds4_qwen_gpu_graph *g, const ds4_model *target,
        const ds4_weights *target_weights, int first_token, uint32_t position,
        uint32_t width, float threshold, int *drafts, int *count) {
#if defined(__linux__) && !defined(DS4_ROCM_BUILD)
    ds4_qwen_dspark_graph *d = g->dspark;
    if (!d || !width || width > d->width || position != d->frontier) return false;
    const ds4_model *m = d->model; const ds4_qwen_dspark_weights *w = d->weights;
    const ds4_qwen_execution_context ctx = {DS4_QWEN_STAGE_MTP_DRAFT, 0};
    int32_t ids[15]; for (unsigned i = 0; i < width; ++i) ids[i] = i ? 248070 : first_token;
    bool ok = ds4_gpu_tensor_write(g->tokens, 0, ids, (uint64_t)width * 4) && ds4_gpu_begin_commands();
    if (ok) ok = ds4_gpu_embed_tokens_quant_tensor(g->cur, g->tokens, target->map, target->size,
        target_weights->token_embd->abs_offset, target_weights->token_embd->type, 248320, width, 5120) != 0;
    for (unsigned i = 0; ok && i < 5; ++i) {
        const ds4_qwen_dspark_layer *l = &w->layer[i];
        ok = ds4_gpu_rms_norm_weight_rows_tensor(g->attn_norm, g->cur, m->map, m->size,
            l->input_norm->abs_offset, 5120, width, 1e-6f) != 0;
        if (ok) ok = qwen_graph_matmul(ctx, g->q_gate, m, l->q, 5120, 4096, g->attn_norm, width) &&
            qwen_graph_matmul(ctx, g->k, m, l->k, 5120, 1024, g->attn_norm, width) &&
            qwen_graph_matmul(ctx, g->v, m, l->v, 5120, 1024, g->attn_norm, width);
        if (ok) ok = ds4_gpu_qwen_dspark_norm_rope_tensor(g->q_gate, m->map, m->size,
            l->q_norm->abs_offset, 32, width, position) &&
            ds4_gpu_qwen_dspark_norm_rope_tensor(g->k, m->map, m->size,
            l->k_norm->abs_offset, 8, width, position);
        if (ok) ok = ds4_gpu_qwen_dspark_attention_tensor(g->heads, g->q_gate, g->k, g->v,
            d->keys[i], d->values[i], position, width, d->attention_workspace) != 0;
        if (ok) ok = qwen_graph_matmul(ctx, g->attn_out, m, l->out, 4096, 5120, g->heads, width) &&
            ds4_gpu_add_tensor(g->after_attn, g->cur, g->attn_out, width*5120);
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->ffn_norm, g->after_attn, m->map, m->size,
            l->post_norm->abs_offset, 5120, width, 1e-6f) != 0;
        if (ok) ok = qwen_graph_matmul(ctx, g->ffn_gate, m, l->gate, 5120, 17408, g->ffn_norm, width) &&
            qwen_graph_matmul(ctx, g->ffn_up, m, l->up, 5120, 17408, g->ffn_norm, width) &&
            ds4_gpu_swiglu_tensor(g->ffn_mid, g->ffn_gate, g->ffn_up, width*17408, 0, 1) &&
            qwen_graph_matmul(ctx, g->ffn_out, m, l->down, 17408, 5120, g->ffn_mid, width) &&
            ds4_gpu_add_tensor(g->next, g->after_attn, g->ffn_out, width*5120);
        if (ok) { ds4_gpu_tensor *swap = g->cur; g->cur = g->next; g->next = swap; }
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mtp_target_rows, g->cur, m->map, m->size,
        w->norm->abs_offset, 5120, width, 1e-6f) &&
        qwen_graph_matmul(ctx, g->mtp_verify_logits, target, target_weights->output,
            5120, 248320, g->mtp_target_rows, width);
    *count = 0;
    int previous = first_token;
    for (unsigned i = 0; ok && i < width; ++i) {
        ok = ds4_gpu_embed_token_quant_tensor(d->markov, m->map, m->size,
            w->markov_w1->abs_offset, w->markov_w1->type, 248320, previous, 256) &&
            qwen_graph_matmul(ctx, d->bias, m, w->markov_w2, 256, 248320, d->markov, 1) &&
            ds4_gpu_add_tensor(d->corrected, d->base_rows[i], d->bias, 248320);
        if (ok && threshold > 0) {
            float logit = 0;
            ok = ds4_gpu_tensor_copy(d->confidence_input, 0, d->hidden_rows[i], 0, 5120*4) &&
                ds4_gpu_tensor_copy(d->confidence_input, 5120*4, d->markov, 0, 256*4) &&
                qwen_graph_matmul(ctx, d->confidence_value, m, w->confidence, 5376, 1, d->confidence_input, 1) &&
                ds4_gpu_tensor_read(d->confidence_value, 0, &logit, 4);
            if (ok) {
                logit += *(const float *)((const unsigned char *)m->map + w->confidence_bias->abs_offset);
                const float probability = 1.0f / (1.0f + expf(-logit));
                if (getenv("DS4_MTP_CYCLE_TRACE"))
                    fprintf(stderr,"QWEN_DSPARK_CONFIDENCE row=%u probability=%.6f threshold=%.6f\n",i,probability,threshold);
                if (probability < threshold) break;
            }
        }
        if (ok) ok = ds4_gpu_indexer_topk_tensor(g->tokens, d->corrected, 248320, 1, 1) &&
            ds4_gpu_tensor_read(g->tokens, 0, &previous, 4);
        if (ok) drafts[(*count)++] = previous;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok;
#else
    (void)g; (void)target; (void)target_weights; (void)first_token; (void)position;
    (void)width; (void)threshold; (void)drafts; (void)count; return false;
#endif
}
