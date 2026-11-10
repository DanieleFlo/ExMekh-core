/* Reduced Flash: PLE disk reads, GDN, attention and routed experts. The exact
 * same GPU scheduler/weights run resident and through the explicit SSD owner. */
#include "../ds4.c"
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x);                                        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
enum { BACKING = 32 << 20, CONTEXT = 16, ROWS = 5 };
static struct {
    FILE *file;
    uint8_t *map;
    uint64_t used;
    ds4_tensor tensors[256];
    unsigned count;
    ds4_model model;
    ds4_weights weights;
} fixture;
static ds4_tensor *matrix(unsigned cols, unsigned rows, float constant, float scale) {
    CHECK(fixture.count < 256);
    ds4_tensor *t = &fixture.tensors[fixture.count++];
    fixture.used = (fixture.used + 255) & ~UINT64_C(255);
    *t = (ds4_tensor){.ndim = 2,
                      .dim = {cols, rows},
                      .type = DS4_TENSOR_F32,
                      .elements = (uint64_t)cols * rows,
                      .bytes = (uint64_t)cols * rows * 4,
                      .abs_offset = fixture.used};
    fixture.used += t->bytes;
    CHECK(fixture.used < BACKING);
    float *p = (float *)(fixture.map + t->abs_offset);
    for (uint64_t i = 0; i < t->elements; i++)
        p[i] = constant + scale * ((int)((i * 13 + fixture.count * 7) % 31) - 15);
    return t;
}
static ds4_tensor *projection(unsigned cols, unsigned rows) { return matrix(cols, rows, 0, .002f); }
static ds4_tensor *norm(unsigned n) { return matrix(n, 1, 1, 0); }
static ds4_tensor *experts(unsigned cols, unsigned rows) {
    ds4_tensor *t = projection(cols, rows * DS4_N_EXPERT);
    t->ndim = 3;
    t->dim[1] = rows;
    t->dim[2] = DS4_N_EXPERT;
    return t;
}
static void create_fixture(void) {
    g_ds4_shape = DS4_SHAPE_QWEN4_MINI;
    g_ds4_shape.n_vocab = 32;
    g_ds4_shape.n_layer = 4;
    g_ds4_shape.n_nextn_predict = 0;
    g_ds4_shape.n_head_dim = 128;
    g_ds4_shape.n_value_dim = 128;
    g_ds4_shape.n_indexer_head = 1;
    g_ds4_shape.n_indexer_head_dim = 128;
    fixture.file = tmpfile();
    CHECK(fixture.file && !ftruncate(fileno(fixture.file), BACKING));
    fixture.map = mmap(NULL, BACKING, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(fixture.file), 0);
    CHECK(fixture.map != MAP_FAILED);
    unsigned e = DS4_N_EMBD, hc = e * DS4_N_HC, rank = DS4_N_HC_LOWRANK;
    ds4_weights *w = &fixture.weights;
    w->token_embd = projection(e, DS4_N_VOCAB);
    w->output = projection(e, DS4_N_VOCAB);
    w->output_hc_norm = norm(hc);
    w->output_hc_down = projection(hc, rank);
    w->output_hc_up = projection(rank, hc);
    ds4_tensor *ngram = matrix(DS4_N_PLE_HEAD_DIM, DS4_N_PLE_HEADS * 16, 0, 0);
    ngram->type = DS4_TENSOR_BF16;
    ngram->bytes = ngram->elements * 2;
    uint16_t *table = (uint16_t *)(fixture.map + ngram->abs_offset);
    for (uint64_t i = 0; i < ngram->elements; i++)
        table[i] = 0x3c80 + i % 16;
    w->ple_embd = ngram;
    for (unsigned i = 0; i < DS4_N_PLE_HEADS; i++) {
        g_ds4_qwen4_ple.head_vocab[i] = 16;
        g_ds4_qwen4_ple.head_offset[i] = i * 16;
    }
    for (unsigned i = 0; i < DS4_N_PLE_NGRAM; i++)
        g_ds4_qwen4_ple.multiplier[i] = 2 * i + 1;
    for (unsigned i = 0; i < DS4_N_LAYER; i++) {
        ds4_layer_weights *l = &w->layer[i];
        l->hc_attn_norm = norm(hc);
        l->hc_ffn_norm = norm(hc);
        l->hc_attn_down = projection(hc, rank);
        l->hc_ffn_down = projection(hc, rank);
        l->hc_attn_up = projection(rank, hc);
        l->hc_ffn_up = projection(rank, hc);
        l->hc_attn_inject = projection(hc, DS4_N_HC);
        l->hc_ffn_inject = projection(hc, DS4_N_HC);
        l->ffn_gate_inp = projection(e, DS4_N_EXPERT);
        l->ffn_gate_inp_shexp = projection(e, 1);
        l->ffn_gate_exps = experts(e, DS4_N_FF_EXP);
        l->ffn_up_exps = experts(e, DS4_N_FF_EXP);
        l->ffn_down_exps = experts(DS4_N_FF_EXP, e);
        l->ffn_gate_shexp = projection(e, DS4_N_FF_EXP);
        l->ffn_up_shexp = projection(e, DS4_N_FF_EXP);
        l->ffn_down_shexp = projection(DS4_N_FF_EXP, e);
        if (ds4_qwen4_layer_is_linear(i)) {
            l->lin_qkv = projection(e, DS4_N_LIN_CONV_DIM);
            l->lin_gate = projection(e, DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM);
            l->lin_alpha = projection(e, DS4_N_LIN_V_HEAD);
            l->lin_beta = projection(e, DS4_N_LIN_V_HEAD);
            l->lin_out = projection(DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM, e);
            l->lin_norm = norm(DS4_N_LIN_HEAD_DIM);
            l->lin_a = matrix(DS4_N_LIN_V_HEAD, 1, -.1f, 0);
            l->lin_dt_bias = matrix(DS4_N_LIN_V_HEAD, 1, -2, 0);
            l->lin_conv = matrix(DS4_N_LIN_CONV, DS4_N_LIN_CONV_DIM, .25f, 0);
        } else {
            unsigned q = DS4_N_HEAD * DS4_N_HEAD_DIM, kv = DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
            l->attn_q = projection(e, q * 2);
            l->attn_k = projection(e, kv);
            l->attn_v = projection(e, kv);
            l->attn_output = projection(q, e);
            l->attn_q_norm = norm(DS4_N_HEAD_DIM);
            l->attn_k_norm = norm(DS4_N_HEAD_DIM);
            l->indexer_q_proj = projection(e, 128);
            l->indexer_k_proj = projection(e, 128);
            l->indexer_q_norm = norm(128);
            l->indexer_k_norm = norm(128);
        }
        if (ds4_qwen4_layer_is_ple(i)) {
            l->ple_key = projection(e, hc);
            l->ple_value = projection(e, e);
            l->ple_norm_key = norm(hc);
            l->ple_norm_query = norm(hc);
            l->ple_norm_conv = norm(hc);
            l->ple_conv = matrix(DS4_N_PLE_CONV, hc, .25f, 0);
        }
    }
    fixture.model = (ds4_model){.map = fixture.map,
                                .size = fixture.used,
                                .fd = fileno(fixture.file),
                                .ngram_fd = fileno(fixture.file),
                                .ngram_tensor = ngram};
}
static void equal_tensor(const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t bytes) {
    if (!a && !b)
        return;
    CHECK(a && b && bytes <= ds4_gpu_tensor_bytes(a) && bytes <= ds4_gpu_tensor_bytes(b));
    void *left = malloc(bytes), *right = malloc(bytes);
    CHECK(left && right && ds4_gpu_tensor_read(a, 0, left, bytes) &&
          ds4_gpu_tensor_read(b, 0, right, bytes));
    CHECK(!memcmp(left, right, bytes));
    free(right);
    free(left);
}
static void equal_state(const ds4_qwen4_gpu_graph *a, const ds4_qwen4_gpu_graph *b) {
    CHECK(a->pos == b->pos && a->mrope_delta == b->mrope_delta &&
          !memcmp(a->ple_prev, b->ple_prev, sizeof(a->ple_prev)));
    equal_tensor(a->ple_hist, b->ple_hist, ds4_gpu_tensor_bytes(a->ple_hist));
    for (unsigned i = 0; i < DS4_N_LAYER; i++) {
        if (ds4_qwen4_layer_is_linear(i)) {
            equal_tensor(a->layer_lin_state[i], b->layer_lin_state[i],
                         ds4_gpu_tensor_bytes(a->layer_lin_state[i]));
            equal_tensor(a->layer_lin_hist[i], b->layer_lin_hist[i],
                         ds4_gpu_tensor_bytes(a->layer_lin_hist[i]));
        } else {
            uint64_t kv = (uint64_t)a->pos * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2;
            equal_tensor(a->layer_k_cache[i], b->layer_k_cache[i], kv);
            equal_tensor(a->layer_v_cache[i], b->layer_v_cache[i], kv);
            equal_tensor(a->layer_ik_cache[i], b->layer_ik_cache[i], (uint64_t)a->pos * 128 * 4);
            equal_tensor(a->layer_block_key[i], b->layer_block_key[i],
                         (uint64_t)(a->pos / 4) * 128 * 2);
        }
    }
}
static void replay(ds4_qwen4_gpu_graph *g, float logits[32]) {
    const int tokens[] = {1, 2, 3, 4, 2, 5, 1, 6};
    qwen4_graph_reset(g);
    CHECK(qwen4_graph_forward_tokens(g, &fixture.model, &fixture.weights, tokens, ROWS, logits,
                                     false));
    CHECK(qwen4_graph_forward_tokens(g, &fixture.model, &fixture.weights, tokens + ROWS, 8 - ROWS,
                                     logits, false));
}
static void check_verify_parity(ds4_qwen4_gpu_graph *reference, ds4_qwen4_gpu_graph *candidate) {
    const int tokens[] = {1, 2, 3, 4, 5};
    float expected[32], actual[3 * 32];
    qwen4_graph_reset(reference);
    qwen4_graph_reset(candidate);
    for (unsigned i = 0; i < 5; i++)
        CHECK(qwen4_graph_forward_token(reference, &fixture.model, &fixture.weights,
                                        tokens[i], expected));
    for (unsigned i = 0; i < 2; i++)
        CHECK(qwen4_graph_forward_token(candidate, &fixture.model, &fixture.weights,
                                        tokens[i], actual));
    candidate->mtp_cat = ds4_gpu_tensor_alloc(3ull * DS4_N_EMBD * DS4_N_HC * 4);
    CHECK(candidate->mtp_cat);
    CHECK(qwen4_graph_verify_tokens(candidate, &fixture.model, &fixture.weights, tokens + 2,
                                    3, actual));
    CHECK(!memcmp(expected, actual + 2 * 32, sizeof(expected)));
    equal_state(reference, candidate);
}
int main(void) {
    create_fixture();
    CHECK(ds4_gpu_init() && ds4_gpu_resources_refresh(2ull << 30));
    ds4_resource_budget cap;
    CHECK(!ds4_resources_get_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &cap));
    cap.limit = 128u << 20;
    cap.margin = 0;
    CHECK(!ds4_resources_set_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &cap));
    ds4_resource_totals baseline;
    ds4_resources_totals(ds4_gpu_resources(), 0, &baseline);
    CHECK(ds4_gpu_set_model_map(fixture.model.map, fixture.model.size));
    ds4_qwen4_gpu_graph reference = {0}, streamed = {0};
    CHECK(qwen4_graph_alloc(&reference, &fixture.model, &fixture.weights, CONTEXT, ROWS, false,
                            NULL, NULL, NULL, 0));
    CHECK(qwen4_graph_alloc(&streamed, &fixture.model, &fixture.weights, CONTEXT, ROWS, false, NULL,
                            NULL, NULL, 0));
    float expected[32], actual[32];
    replay(&reference, expected);
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_qwen4_stream_open(fixture.model.map, fixture.model.size, fixture.model.fd,
                                    500u << 10));
    replay(&streamed, actual);
    CHECK(!memcmp(expected, actual, sizeof(actual)));
    equal_state(&reference, &streamed);
    fixture.model.ngram_fd = -1;
    qwen4_graph_reset(&streamed);
    int token = 1;
    CHECK(!qwen4_graph_forward_tokens(&streamed, &fixture.model, &fixture.weights, &token, 1,
                                      actual, false));
    fixture.model.ngram_fd = fileno(fixture.file);
    replay(&streamed, actual);
    CHECK(!memcmp(expected, actual, sizeof(actual)));
    equal_state(&reference, &streamed);
    check_verify_parity(&reference, &streamed);
    CHECK(ds4_gpu_qwen4_stream_close());
    qwen4_graph_free(&streamed);
    qwen4_graph_free(&reference);
    CHECK(ds4_gpu_synchronize());
    ds4_resource_totals after;
    ds4_resources_totals(ds4_gpu_resources(), 0, &after);
    CHECK(after.roots == baseline.roots && after.views == baseline.views);
    ds4_gpu_cleanup();
    CHECK(!munmap(fixture.map, BACKING) && !fclose(fixture.file));
    puts("PASS: reduced Flash resident/SSD logits and full recurrent/KV/PLE state exact; n-gram "
         "failure and recovery");
}
