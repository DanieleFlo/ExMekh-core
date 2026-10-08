/* Exercise the real loader and workload selector, including legacy profiles
 * and invalid input. This test needs no model, CUDA device or mock allocator. */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "../ds4_env.h"
namespace qwen4_cuda { static unsigned cache_ctx = 8192, cache_prefill = 1024; }
#include "../ds4_qwen4_tuning.cuh"
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)

int main() {
    using namespace qwen4_cuda;
    char path[] = "/tmp/ds4-tuning-XXXXXX";
    const int fd = mkstemp(path); CHECK(fd >= 0); close(fd);
    CHECK(!setenv("DS4_CUDA_QWEN4_TUNING_PROFILE", path, 1));
    unsetenv("DS4_CUDA_QWEN4_PCIE_FRAC"); unsetenv("DS4_CUDA_QWEN4_CPU_WORKERS");
    auto load = [&](const char *text) {
        FILE *fp = std::fopen(path,"w"); CHECK(fp);
        CHECK(std::fputs(text,fp) >= 0 && !std::fclose(fp));
        tuning_load(0x1234);
    };
    float fraction = -1; unsigned workers = 0;
    load("DS4QWEN4TUNE2 1234\ndecode 0.25 1 1\nmtp 0.75 4 1\nprefill 0.5 8 1\n");
    CHECK(tuning_config(1,&fraction,&workers)==1 && fraction==0.25f && workers==1);
    for (unsigned rows=2; rows<=8; ++rows)
        CHECK(tuning_config(rows,&fraction,&workers)==1 && fraction==0.75f && workers==4);
    CHECK(tuning_config(9,&fraction,&workers)==1 && fraction==0.5f && workers==8);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_MTP);
    CHECK(tuning_config(1,&fraction,&workers)==1 && fraction==0.75f && workers==4);
    CHECK(tuning_config(9,&fraction,&workers)==1 && fraction==0.75f && workers==4);
    const ds4_qwen4_workload outer = tuning_workload_scope;
    tuning_set_workload(DS4_QWEN4_WORKLOAD_PREFILL);
    for (unsigned rows=1; rows<=9; ++rows)
        CHECK(tuning_config(rows,&fraction,&workers)==1 && fraction==0.5f && workers==8);
    tuning_set_workload(outer);
    CHECK(tuning_config(1,&fraction,&workers)==1 && fraction==0.75f && workers==4);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_DECODE);
    CHECK(tuning_config(8,&fraction,&workers)==1 && fraction==0.25f && workers==1);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_AUTO);
    CHECK(tuning_set_workload(DS4_QWEN4_WORKLOAD_PREFILL) == DS4_QWEN4_WORKLOAD_AUTO);
    CHECK(tuning_set_workload((ds4_qwen4_workload)DS4_QWEN4_WORKLOAD_COUNT) == DS4_QWEN4_WORKLOAD_PREFILL);
    CHECK(tuning_phase(1) == DS4_QWEN4_WORKLOAD_PREFILL);
    CHECK(tuning_set_workload((ds4_qwen4_workload)-2) == DS4_QWEN4_WORKLOAD_PREFILL);
    CHECK(tuning_phase(9) == DS4_QWEN4_WORKLOAD_PREFILL);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_AUTO);
    CHECK(tuning_config(1,&fraction,&workers)==1 && fraction==0.25f && workers==1);
    load("DS4QWEN4TUNE1 1234\ndecode 0.25 1 1\nprefill 0.5 8 1\n");
    CHECK(tuning_config(1,&fraction,&workers)==1 && fraction==0.25f && workers==1);
    CHECK(tuning_config(2,&fraction,&workers)==0 && tuning_config(8,&fraction,&workers)==0);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_MTP);
    CHECK(tuning_config(1,&fraction,&workers)==0);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_PREFILL);
    CHECK(tuning_config(1,&fraction,&workers)==1 && fraction==0.5f && workers==8);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_AUTO);
    CHECK(tuning_config(9,&fraction,&workers)==1 && fraction==0.5f && workers==8);
    const char *invalid[] = {
        "DS4QWEN4TUNE2 1235\ndecode 0.25 1 1\nmtp 0.75 4 1\nprefill 0.5 8 1\n",
        "DS4QWEN4TUNE2 1234\ndecode 0.25 1 1\nmtp 0.75 4 1\n",
        "DS4QWEN4TUNE2 1234\ndecode 0.25 1 1\nmtp nan 4 1\nprefill 0.5 8 1\n",
        "DS4QWEN4TUNE2 1234\ndecode 0.25 1 1\nmtp 0.75 9 1\nprefill 0.5 8 1\n",
        "DS4QWEN4TUNE2 1234\ndecode 0.25 1 1\nmtp 0.75 4 2\nprefill 0.5 8 1\n",
        "DS4QWEN4TUNE2 1234\ndecode 0.25 1 1\nmtp 0.75 4 1\nprefill 0.5 8 1\ngarbage\n",
    };
    for (const char *text : invalid) {
        load(text);
        CHECK(!tuning.valid && tuning_config(1,&fraction,&workers)==0 &&
            tuning_config(8,&fraction,&workers)==0 && tuning_config(256,&fraction,&workers)==0);
    }
    CHECK(!setenv("DS4_CUDA_QWEN4_PCIE_FRAC","0.5",1));
    CHECK(!setenv("DS4_CUDA_QWEN4_CPU_WORKERS","2",1));
    CHECK(!setenv("DS4_CUDA_QWEN4_HYBRID_PREFILL","0",1));
    CHECK(tuning_config(1,&fraction,&workers)==1 && fraction==0.5f && workers==2);
    CHECK(tuning_config(8,&fraction,&workers)==1 && fraction==0.5f && workers==2);
    CHECK(tuning_config(256,&fraction,&workers)==0);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_PREFILL);
    for (unsigned rows=1; rows<=8; ++rows)
        CHECK(tuning_config(rows,&fraction,&workers)==0);
    CHECK(!setenv("DS4_CUDA_QWEN4_HYBRID_PREFILL","1",1));
    CHECK(tuning_config(1,&fraction,&workers)==1 && fraction==0.5f && workers==2);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_MTP);
    CHECK(!setenv("DS4_CUDA_QWEN4_HYBRID_PREFILL","0",1));
    CHECK(tuning_config(1,&fraction,&workers)==1 && fraction==0.5f && workers==2);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_AUTO);
    unlink(path);
    tuning_set_workload(DS4_QWEN4_WORKLOAD_PREFILL);
    tuning_reset();
    CHECK(!tuning.valid && tuning_phase(1) == DS4_QWEN4_WORKLOAD_DECODE);
    std::puts("PASS: decode/MTP/prefill separated, legacy/failure fallback and overrides");
}
