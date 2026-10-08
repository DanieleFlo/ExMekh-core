/* Pinned Qwen DSpark v2 support binding; no global target shape mutation. */
typedef struct {
    ds4_tensor *input_norm, *post_norm, *q_norm, *k_norm;
    ds4_tensor *q, *k, *v, *out, *gate, *up, *down;
} ds4_qwen_dspark_layer;
typedef struct {
    ds4_tensor *fc, *hidden_norm, *norm, *markov_w1, *markov_w2;
    uint64_t fingerprint;
    ds4_tensor *confidence, *confidence_bias;
    ds4_qwen_dspark_layer layer[5];
} ds4_qwen_dspark_weights;

static ds4_tensor *qwen_dspark_weight(const ds4_model *m, const char *name,
        uint32_t in, uint32_t out, bool norm, bool *valid) {
    ds4_tensor *w = model_find_tensor(m, name);
    const bool type_ok = w && (norm ? w->type == DS4_TENSOR_F32 :
        (w->type == DS4_TENSOR_F32 || w->type == DS4_TENSOR_F16 ||
         w->type == DS4_TENSOR_Q8_0 || w->type == DS4_TENSOR_Q4_K));
    if (!type_ok || w->dim[0] != in || (out ? w->ndim != 2 || w->dim[1] != out : w->ndim != 1)) {
        fprintf(stderr, "ds4: invalid Qwen DSpark tensor %s\n", name);
        *valid = false; return NULL;
    }
    return w;
}

static bool qwen_dspark_weights_bind(ds4_qwen_dspark_weights *w, const ds4_model *m) {
    bool valid = DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN35 &&
        DS4_N_LAYER == 64 && DS4_N_EMBD == 5120 && DS4_N_VOCAB == 248320;
    ds4_str revision = {0}; uint32_t value = 0;
    if (!model_get_string(m, "general.source.revision", &revision) ||
        !ds4_streq(revision, "b9a5dbdf03bc999c6c73c426b19c2d9041cea393") ||
        !model_get_u32(m, "dspark.block_size", &value) || value != 7 ||
        !model_get_u32(m, "dspark.markov_rank", &value) || value != 256 ||
        !model_get_u32(m, "dspark.noise_token_id", &value) || value != 248070) {
        fprintf(stderr, "ds4: incompatible or unpinned Qwen DSpark metadata\n"); valid = false;
    }
    static const struct { const char *key; uint32_t expected; } integers[] = {
        {"dspark.training_block_size",16},{"qwen3-dspark.embedding_length",5120},
        {"qwen3-dspark.block_count",5},{"qwen3-dspark.feed_forward_length",17408},
        {"qwen3-dspark.attention.head_count",32},{"qwen3-dspark.attention.head_count_kv",8},
        {"qwen3-dspark.attention.key_length",128},{"qwen3-dspark.context_length",262144},
        {"qwen3-dspark.rope.scaling.original_context_length",8192}};
    static const struct { const char *key; float expected; } floats[] = {
        {"qwen3-dspark.attention.layer_norm_rms_epsilon",1e-6f},
        {"qwen3-dspark.rope.freq_base",1e7f},{"qwen3-dspark.rope.scaling.factor",32},
        {"qwen3-dspark.rope.scaling.beta_fast",32},{"qwen3-dspark.rope.scaling.beta_slow",1}};
    for (unsigned i=0;i<sizeof(integers)/sizeof(integers[0]);++i)
        if (!model_get_u32(m,integers[i].key,&value)||value!=integers[i].expected) valid=false;
    for (unsigned i=0;i<sizeof(floats)/sizeof(floats[0]);++i) {
        float f=0;
        if (!model_get_f32_compat(m,floats[i].key,&f)||f!=floats[i].expected) valid=false;
    }
    ds4_array_ref taps={0};
    const uint32_t expected_taps[]={5,19,33,47,61};
    if (!model_get_array(m,"dspark.target_layer_ids",&taps)||taps.type!=GGUF_VALUE_UINT32||taps.len!=5) valid=false;
    else {
        ds4_cursor c=cursor_at(m,taps.data_pos);
        for (unsigned i=0;i<5;++i) if (!cursor_u32(&c,&value)||value!=expected_taps[i]) valid=false;
    }
    if (!valid) {
        fprintf(stderr,"ds4: incompatible Qwen DSpark architecture, taps or rotary metadata\n");
        return false;
    }
    memset(w, 0, sizeof(*w));
    w->fingerprint = UINT64_C(14695981039346656037);
    const unsigned char *data = m->map;
    for (uint64_t i = 0; i < m->size; ++i)
        w->fingerprint = (w->fingerprint ^ data[i]) * UINT64_C(1099511628211);
#define DS4_QDSP_BIND(field, name, in, out, norm) \
    w->field = qwen_dspark_weight(m, name, in, out, norm, &valid)
    DS4_QDSP_BIND(fc, "fc.weight", 25600, 5120, false);
    DS4_QDSP_BIND(hidden_norm, "hidden_norm.weight", 5120, 0, true);
    DS4_QDSP_BIND(norm, "norm.weight", 5120, 0, true);
    DS4_QDSP_BIND(markov_w1, "markov_head.markov_w1.weight", 256, 248320, false);
    DS4_QDSP_BIND(markov_w2, "markov_head.markov_w2.weight", 256, 248320, false);
    DS4_QDSP_BIND(confidence, "confidence_head.proj.weight", 5376, 1, true);
    DS4_QDSP_BIND(confidence_bias, "confidence_head.proj.bias", 1, 0, true);
#undef DS4_QDSP_BIND
    for (unsigned i = 0; i < 5; ++i) {
        ds4_qwen_dspark_layer *l = &w->layer[i]; char name[128];
#define DS4_QDSP_LAYER(field, suffix, in, out, norm) do { \
        snprintf(name, sizeof(name), "layers.%u.%s.weight", i, suffix); \
        l->field = qwen_dspark_weight(m, name, in, out, norm, &valid); \
    } while (0)
        DS4_QDSP_LAYER(input_norm, "input_layernorm", 5120, 0, true);
        DS4_QDSP_LAYER(post_norm, "post_attention_layernorm", 5120, 0, true);
        DS4_QDSP_LAYER(q_norm, "self_attn.q_norm", 128, 0, true);
        DS4_QDSP_LAYER(k_norm, "self_attn.k_norm", 128, 0, true);
        DS4_QDSP_LAYER(q, "self_attn.q_proj", 5120, 4096, false);
        DS4_QDSP_LAYER(k, "self_attn.k_proj", 5120, 1024, false);
        DS4_QDSP_LAYER(v, "self_attn.v_proj", 5120, 1024, false);
        DS4_QDSP_LAYER(out, "self_attn.o_proj", 4096, 5120, false);
        DS4_QDSP_LAYER(gate, "mlp.gate_proj", 5120, 17408, false);
        DS4_QDSP_LAYER(up, "mlp.up_proj", 5120, 17408, false);
        DS4_QDSP_LAYER(down, "mlp.down_proj", 17408, 5120, false);
#undef DS4_QDSP_LAYER
    }
    return valid;
}
