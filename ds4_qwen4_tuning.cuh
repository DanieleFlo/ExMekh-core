#include "ds4_qwen4_runtime.h"

namespace qwen4_cuda {
/* Ordinary decode, multi-row verification, prefill. Legacy profiles do not
 * silently apply a single-row measurement to multi-row verification. */
struct hybrid_tuning {
    float fraction[DS4_QWEN4_WORKLOAD_COUNT];
    unsigned workers[DS4_QWEN4_WORKLOAD_COUNT];
    bool enabled[DS4_QWEN4_WORKLOAD_COUNT], valid;
};
static hybrid_tuning tuning;
static ds4_qwen4_workload tuning_workload_scope = DS4_QWEN4_WORKLOAD_AUTO;

static ds4_qwen4_workload tuning_set_workload(ds4_qwen4_workload workload) {
    const ds4_qwen4_workload previous = tuning_workload_scope;
    if (workload >= DS4_QWEN4_WORKLOAD_AUTO && workload < DS4_QWEN4_WORKLOAD_COUNT)
        tuning_workload_scope = workload;
    return previous;
}

static ds4_qwen4_workload tuning_phase(unsigned T) {
    if (tuning_workload_scope != DS4_QWEN4_WORKLOAD_AUTO) return tuning_workload_scope;
    return T > 8u ? DS4_QWEN4_WORKLOAD_PREFILL :
        T > 1u ? DS4_QWEN4_WORKLOAD_MTP : DS4_QWEN4_WORKLOAD_DECODE;
}

static void tuning_reset(void) {
    tuning = hybrid_tuning{};
    tuning_workload_scope = DS4_QWEN4_WORKLOAD_AUTO;
}

static void tuning_load(uint64_t key) {
    tuning = hybrid_tuning{};
    const char *path = getenv("DS4_CUDA_QWEN4_TUNING_PROFILE");
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_CALIBRATE_INFO", 0))
        fprintf(stderr, "ds4: Qwen tuning key=%016llx ctx=%u prefill=%u\n", (unsigned long long)key, cache_ctx, cache_prefill);
    if (!path || !*path) return;
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "ds4: Qwen tuning profile unavailable; keeping ordinary CUDA\n"); return; }
    char magic[32], label[16]; unsigned long long recorded;
    bool ok = fscanf(fp, "%31s %llx", magic, &recorded) == 2 && recorded == key;
    const bool legacy = ok && !strcmp(magic, "DS4QWEN4TUNE1");
    ok = ok && (legacy || !strcmp(magic, "DS4QWEN4TUNE2"));
    const char *names[] = {"decode", "mtp", "prefill"};
    const unsigned phases = legacy ? 2u : (unsigned)DS4_QWEN4_WORKLOAD_COUNT;
    for (unsigned i = 0; i < phases && ok; i++) {
        const ds4_qwen4_workload phase = legacy && i == 1u ? DS4_QWEN4_WORKLOAD_PREFILL : (ds4_qwen4_workload)i;
        unsigned enabled = 0;
        ok = fscanf(fp, "%15s %f %u %u", label, &tuning.fraction[phase], &tuning.workers[phase], &enabled) == 4 &&
             !strcmp(label, names[phase]) && std::isfinite(tuning.fraction[phase]) &&
             tuning.fraction[phase] >= 0 && tuning.fraction[phase] <= 1 && tuning.workers[phase] >= 1 &&
             tuning.workers[phase] <= 8 && enabled <= 1;
        tuning.enabled[phase] = enabled != 0;
    }
    int tail;
    do { tail = fgetc(fp); } while (tail == ' ' || tail == '\n' || tail == '\r' || tail == '\t');
    ok = ok && tail == EOF; fclose(fp);
    if (ok) tuning.valid = true;
    else fprintf(stderr, "ds4: incompatible Qwen tuning profile ignored; keeping ordinary CUDA\n");
}

/* 0 preserves the previous GPU scheduler, -1 is an invalid explicit option. */
static int tuning_config(unsigned T, float *fraction, unsigned *workers) {
    const ds4_qwen4_workload phase = tuning_phase(T);
    const char *env = getenv("DS4_CUDA_QWEN4_PCIE_FRAC");
    if (!env || !*env || !strcmp(env, "auto")) {
        if (!tuning.valid || !tuning.enabled[phase]) return 0;
        *fraction = tuning.fraction[phase]; *workers = tuning.workers[phase];
    } else {
        char *end = NULL; *fraction = strtof(env, &end);
        if (end == env || *end || !std::isfinite(*fraction) || *fraction < 0 || *fraction > 1) return -1;
        if (phase == DS4_QWEN4_WORKLOAD_PREFILL && !ds4_env_flag_enabled("DS4_CUDA_QWEN4_HYBRID_PREFILL", 0)) return 0;
        *workers = 4;
    }
    env = getenv("DS4_CUDA_QWEN4_CPU_WORKERS");
    if (env && *env) {
        char *end = NULL; const unsigned long n = strtoul(env, &end, 10);
        if (end == env || *end || n < 1 || n > 8) return -1;
        *workers = (unsigned)n;
    }
    return 1;
}
} // namespace qwen4_cuda
