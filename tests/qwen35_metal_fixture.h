/* Bounded deterministic model for scheduler/state tests; no pretrained-model
 * accuracy claim. Include after ds4.c so fixtures can bind internal tensors. */
#ifndef DS4_TEST_QWEN35_METAL_FIXTURE_H
#define DS4_TEST_QWEN35_METAL_FIXTURE_H
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x);                                        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
enum { WIDTH = 256, VOCAB = 32, LAYERS = 4, CONTEXT = 16, PREFILL = 5, STEPS = 11 };
enum { FIXTURE_BYTES = 32u * 1024u * 1024u, DEVICE_CAP_BYTES = 128u * 1024u * 1024u };
typedef struct {
    ds4_tensor tensors[128];
    FILE *backing;
    unsigned tensor_count;
    uint64_t used;
    uint8_t *mapping;
    ds4_weights weights;
    ds4_model model;
} qwen35_fixture;

static ds4_tensor *fixture_tensor(qwen35_fixture *fixture, uint32_t type, uint64_t cols,
                                  uint64_t rows) {
    CHECK(fixture->tensor_count < sizeof(fixture->tensors) / sizeof(fixture->tensors[0]));
    ds4_tensor *tensor = &fixture->tensors[fixture->tensor_count++];
    *tensor = (ds4_tensor){.type = type, .ndim = 2, .dim = {cols, rows}, .elements = cols * rows};
    CHECK(tensor_nbytes(type, tensor->elements, &tensor->bytes));
    fixture->used = (fixture->used + 255) & ~UINT64_C(255);
    tensor->abs_offset = fixture->used;
    fixture->used += tensor->bytes;
    CHECK(fixture->used < FIXTURE_BYTES);
    return tensor;
}
static ds4_tensor *fixture_constant(qwen35_fixture *fixture, uint64_t cols, uint64_t rows,
                                    float scalar) {
    ds4_tensor *tensor = fixture_tensor(fixture, DS4_TENSOR_F32, cols, rows);
    float *values = (float *)(fixture->mapping + tensor->abs_offset);
    for (uint64_t i = 0; i < tensor->elements; i++)
        values[i] = scalar;
    return tensor;
}
static ds4_tensor *fixture_projection(qwen35_fixture *fixture, uint64_t cols, uint64_t rows) {
    CHECK(cols % 32 == 0);
    ds4_tensor *tensor = fixture_tensor(fixture, DS4_TENSOR_Q8_0, cols, rows);
    uint8_t *weights = fixture->mapping + tensor->abs_offset;
    /* Deterministic signed blocks; no full-model expansion in production. */
    for (uint64_t block = 0; block < tensor->elements / 32; block++) {
        uint16_t scale = 0x1800; /* 1/512 */
        memcpy(weights + block * 34, &scale, sizeof(scale));
        for (unsigned j = 0; j < 32; j++)
            weights[block * 34 + 2 + j] =
                (uint8_t)(int8_t)((int)((block * 13 + j * 7 + fixture->tensor_count * 11) % 31) -
                                  15);
    }
    return tensor;
}

static void fixture_weights(qwen35_fixture *fixture) {
    ds4_weights *w = &fixture->weights;
    w->token_embd = fixture_projection(fixture, WIDTH, VOCAB);
    w->output = fixture_projection(fixture, WIDTH, VOCAB);
    w->output_norm = fixture_constant(fixture, WIDTH, 1, 1);
    for (unsigned il = 0; il < LAYERS; il++) {
        ds4_layer_weights *l = &w->layer[il];
        l->attn_norm = fixture_constant(fixture, WIDTH, 1, 1);
        l->ffn_norm = fixture_constant(fixture, WIDTH, 1, 1);
        l->ffn_gate = fixture_projection(fixture, WIDTH, WIDTH);
        l->ffn_up = fixture_projection(fixture, WIDTH, WIDTH);
        l->ffn_down = fixture_projection(fixture, WIDTH, WIDTH);
        if (qwen_layer_is_recurrent(il)) {
            l->qwen_attn_qkv = fixture_projection(fixture, WIDTH, 10240);
            l->qwen_attn_gate = fixture_projection(fixture, WIDTH, 6144);
            l->qwen_ssm_alpha = fixture_projection(fixture, WIDTH, 48);
            l->qwen_ssm_beta = fixture_projection(fixture, WIDTH, 48);
            l->qwen_ssm_out = fixture_projection(fixture, 6144, WIDTH);
            l->qwen_ssm_a = fixture_constant(fixture, 48, 1, -.1f);
            l->qwen_ssm_dt = fixture_constant(fixture, 48, 1, -2);
            l->qwen_ssm_norm = fixture_constant(fixture, 128, 1, 1);
            l->qwen_ssm_conv1d = fixture_constant(fixture, 4, 10240, .25f);
        } else {
            l->qwen_attn_q = fixture_projection(fixture, WIDTH, 12288);
            l->qwen_attn_k = fixture_projection(fixture, WIDTH, 1024);
            l->qwen_attn_v = fixture_projection(fixture, WIDTH, 1024);
            l->attn_output = fixture_projection(fixture, 6144, WIDTH);
            l->qwen_attn_q_norm = fixture_constant(fixture, 256, 1, 1);
            l->qwen_attn_k_norm = fixture_constant(fixture, 256, 1, 1);
        }
    }
}

static void fixture_create(qwen35_fixture *fixture) {
    memset(fixture, 0, sizeof(*fixture));
    fixture->backing = tmpfile();
    CHECK(fixture->backing && !ftruncate(fileno(fixture->backing), FIXTURE_BYTES));
    fixture->mapping =
        mmap(NULL, FIXTURE_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(fixture->backing), 0);
    CHECK(fixture->mapping != MAP_FAILED);
    fixture_weights(fixture);
    fixture->model =
        (ds4_model){.map = fixture->mapping, .size = fixture->used, .fd = fileno(fixture->backing)};
}
static void fixture_destroy(qwen35_fixture *fixture) {
    CHECK(!munmap(fixture->mapping, FIXTURE_BYTES));
    CHECK(!fclose(fixture->backing));
    memset(fixture, 0, sizeof(*fixture));
}

static void configure_fixture_shape(void) {
    g_ds4_shape = DS4_SHAPE_QWEN38_27B;
    g_ds4_shape.n_layer = LAYERS;
    g_ds4_shape.n_embd = WIDTH;
    g_ds4_shape.n_ff_dense = WIDTH;
    g_ds4_shape.n_vocab = VOCAB;
}

static inline void fixture_mtp(qwen35_fixture *fixture, ds4_qwen_mtp_weights *mtp) {
    *mtp = (ds4_qwen_mtp_weights){0};
    mtp->token_embd = fixture->weights.token_embd;
    mtp->output = fixture->weights.output;
    mtp->block = fixture->weights.layer[LAYERS - 1];
    mtp->block.nextn_enorm = fixture_constant(fixture, WIDTH, 1, 1);
    mtp->block.nextn_hnorm = fixture_constant(fixture, WIDTH, 1, 1);
    mtp->block.nextn_eh_proj = fixture_projection(fixture, 2 * WIDTH, WIDTH);
    mtp->output_norm = fixture_constant(fixture, WIDTH, 1, 1);
    mtp->block.nextn_shared_head_norm = mtp->output_norm;
    fixture->model.size = fixture->used;
}
#endif
