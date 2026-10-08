typedef struct {
    const ds4_model *model;
    const ds4_qwen_dspark_weights *weights;
    uint32_t context, cap, width, frontier;
    ds4_gpu_tensor *features, *projected, *normalized, *k, *v, *half;
    ds4_gpu_tensor *markov, *bias, *corrected, *confidence_input, *confidence_value;
    ds4_gpu_tensor *base_rows[15], *hidden_rows[15];
    uint16_t *keys[5], *values[5];
    void *attention_workspace;
} ds4_qwen_dspark_graph;

static void qwen_dspark_graph_free(ds4_qwen_dspark_graph *d) {
    if (!d) return;
    (void)ds4_gpu_synchronize();
    ds4_gpu_tensor *owned[] = {d->features, d->projected, d->normalized, d->k, d->v, d->half,
        d->markov, d->bias, d->corrected, d->confidence_input, d->confidence_value};
    for (unsigned i = 0; i < sizeof(owned) / sizeof(owned[0]); ++i) ds4_gpu_tensor_free(owned[i]);
    for (unsigned i = 0; i < 15; ++i) {
        ds4_gpu_tensor_free(d->base_rows[i]); ds4_gpu_tensor_free(d->hidden_rows[i]);
    }
    for (unsigned i = 0; i < 5; ++i) { free(d->keys[i]); free(d->values[i]); }
#if defined(__linux__) && !defined(DS4_ROCM_BUILD)
    ds4_gpu_qwen_dspark_workspace_free(d->attention_workspace);
#endif
    free(d);
}
