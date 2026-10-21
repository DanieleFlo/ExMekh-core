CC ?= cc
UNAME_S := $(shell uname -s)
DS4_API_HEADERS = ds4.h ds4_runtime.h ds4_resources.h ds4_executor.h

CPU_SUPPORT_OBJS = ds4_runtime.o ds4_executor.o ds4_bundle.o ds4_resources.o ds4_checkpoint.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
CPU_CORE_OBJS = ds4_cpu.o $(CPU_SUPPORT_OBJS)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif
SAMPLING_TEST := tests/test_sampling
GLM53_KDA_TEST := tests/test_glm53_kda
QWEN4_KERNEL_TEST := tests/test_qwen4_kernels
GLM53_KDA_ROCM_TEST := tests/test_glm53_kda_rocm

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
ROCM_SRCS := $(wildcard rocm/*.cuh)
DS4_TEST_MODEL ?= $(if $(wildcard gguf/Qwen3.8-27B-UD-Q4_K_S.gguf),gguf/Qwen3.8-27B-UD-Q4_K_S.gguf,ds4flash.gguf)
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_runtime.o ds4_executor.o ds4_bundle.o ds4_resources.o ds4_checkpoint.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o ds4_engram.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= $(shell if [ -x /usr/local/cuda/bin/nvcc ]; then \
	printf '%s' /usr/local/cuda; \
	elif command -v nvcc >/dev/null 2>&1; then \
	dirname "$$(dirname "$$(command -v nvcc)")"; \
	else \
	printf '%s' /usr/local/cuda; \
	fi)
NVCC ?= $(CUDA_HOME)/bin/nvcc
# Direct binary/test targets must select the local GPU just like cuda-generic.
CUDA_ARCH ?= native
ifneq ($(strip $(CUDA_ARCH)),)
ifneq ($(filter sm_120 sm_120a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1
else ifneq ($(filter sm_121 sm_121a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a -DDS4_CUDA_HAVE_MXF4=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
# Vendored llama.cpp mmq prefill tier (cuda/mmq/, see cuda/mmq/VENDOR.md).
MMQ_INCLUDES := -Icuda/mmq
MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
# Standalone CUDA test links use this object group as well as production links.
# Build AVX2 host arithmetic with the C compiler, not nvcc's intrinsic parser.
MMQ_OBJS += ds4_qwen4_cpu.o ds4_resources.o
CORE_OBJS = ds4.o ds4_runtime.o ds4_executor.o ds4_bundle.o ds4_checkpoint.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o ds4_engram.o $(MMQ_OBJS)
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1151
ROCM_HOST_CFLAGS ?= -fPIC
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt -lrocblas
ROCM_MMQ_Y ?= 64
ROCM_MMQ_FLAGS := $(ROCM_CFLAGS) -std=c++17 -DGGML_USE_HIP -DDS4_HIP_MMQ_Y=$(ROCM_MMQ_Y) $(MMQ_INCLUDES)
ROCM_MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.rocm.o cuda/mmq/ds4_mmq.rocm.o cuda/mmq/quantize.rocm.o cuda/mmq/mmid.rocm.o cuda/mmq/mmvq.rocm.o cuda/mmq/d2r_stubs.rocm.o
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test test-rocm test-qwen38 test-qwen38-perf test-constrained-json-live test-server-sampling-matrix-static test-jsonschemabench-safety test-metal-session-batch test-mmq-parity test-mxfp4-cuda test-mxfp4-rocm test-cuda-session-batch test-cuda-mixed-batch dspark-acceptance dspark-verify-depth mtp-verify-depth cpu cuda cuda-spark cuda-generic cuda-regression qwen-numerics perf-doctor perf-harness-test perf-fast-test perf-direction-ab perf-slow-test research-fetch research-index research-check docs-index docs-check strix-halo rocm test-glm53-kda-rocm

ifeq ($(UNAME_S),Darwin)
.PHONY: metal-decode-schedule-bench metal-prefill-variant-bench session-concurrency-bench check-mxfp4-half-lut
.PHONY: test-metal-moe-prefill test-metal-dense-mpp

all: ds4 ds4-server ds4-bench ds4-eval

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval"
	@echo "  make test         Build and run tests"
	@echo "  make test-qwen38-perf  Run the opt-in Qwen3.8 CLI/server MTP VRAM-safe context gate"
	@echo "  make test-constrained-json-live  Run opt-in live DSML/JSON Schema adversarial API tests"
	@echo "  make test-server-interactive  Run interactive protocol and text-constraint fixtures"
	@echo "  make test-interactive-api-live Run interactive HTTP tests (DS4_INTERACTIVE_BASE_URL)"
	@echo "  make test-server-sampling-matrix-static  Check the temperature/thinking/MTP server matrix fixtures"
	@echo "  make test-jsonschemabench-safety  Run the pinned external candidate-vs-oracle safety gate"
	@echo "  make metal-decode-schedule-bench  Build the balanced Metal decode schedule benchmark"
	@echo "  make session-concurrency-bench    Build the concurrency x context serving benchmark"
	@echo "  make metal-prefill-variant-bench  Build the balanced Metal prefill variant benchmark"
	@echo "  make check-mxfp4-half-lut  Verify the checked-in MXFP4 half LUT matches the generator"
	@echo "  make test-mxfp4-metal  Check the MXFP4 half LUT, then run Metal MXFP4 exactness tests"
	@echo "  make test-qwen4-kernels  Run the Qwen3.8 Metal kernel tests"
	@echo "  make test-qwen4-q2       Check exact low-bit decode and prefill tile parity"
	@echo "  make test-qwen4-vision  Compare the Qwen3.8 vision tower with HF (set DS4_QWEN4_SNAPSHOT, DS4_QWEN4_MMPROJ, DS4_QWEN4_IMAGE)"
	@echo "  make dspark-verify-depth  Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth  Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)


gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.c $(DS4_API_HEADERS) ds4_distributed.h ds4_tp.h $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(CC) $(QUALITY_CFLAGS) -I. -o $@ gguf-tools/quality-testing/score_official.c $(CORE_OBJS) rax.o ds4_gpu_args.o $(METAL_LDLIBS)

tests/test_metal_session_batch.o: tests/test_metal_session_batch.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_metal_session_batch.c

tests/test_metal_session_batch: tests/test_metal_session_batch.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

tests/test_metal_tp_spec.o: tests/test_metal_tp_spec.c $(DS4_API_HEADERS) ds4_tp.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_tp_spec: tests/test_metal_tp_spec.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

tests/test_metal_tp_cancel: tests/test_metal_tp_cancel.c $(DS4_API_HEADERS) ds4_tp.h $(CORE_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< $(CORE_OBJS) $(METAL_LDLIBS)

test-metal-session-batch: tests/test_metal_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_metal_session_batch

speed-bench/metal_decode_schedule_bench.o: speed-bench/metal_decode_schedule_bench.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_decode_schedule_bench: speed-bench/metal_decode_schedule_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-decode-schedule-bench: speed-bench/metal_decode_schedule_bench

speed-bench/metal_prefill_variant_bench.o: speed-bench/metal_prefill_variant_bench.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_prefill_variant_bench: speed-bench/metal_prefill_variant_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-prefill-variant-bench: speed-bench/metal_prefill_variant_bench

speed-bench/session_concurrency_bench.o: speed-bench/session_concurrency_bench.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/session_concurrency_bench: speed-bench/session_concurrency_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

session-concurrency-bench: speed-bench/session_concurrency_bench

tests/test_mxfp4_metal.o: tests/test_mxfp4_metal.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_mxfp4_metal: tests/test_mxfp4_metal.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

check-mxfp4-half-lut:
	python3 metal/generate_mxfp4_half_lut.py --check

test-mxfp4-metal: check-mxfp4-half-lut tests/test_mxfp4_metal
	./tests/test_mxfp4_metal

tests/test_metal_moe_prefill.o: tests/test_metal_moe_prefill.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_moe_prefill: tests/test_metal_moe_prefill.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-moe-prefill: tests/test_metal_moe_prefill
	./tests/test_metal_moe_prefill

tests/test_qwen4_moe_mm_specialize.o: tests/test_qwen4_moe_mm_specialize.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_qwen4_moe_mm_specialize: tests/test_qwen4_moe_mm_specialize.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: test-qwen4-moe-mm-specialize
test-qwen4-moe-mm-specialize: tests/test_qwen4_moe_mm_specialize
	./tests/test_qwen4_moe_mm_specialize

tests/test_qwen4_conv_parallel.o: tests/test_qwen4_conv_parallel.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_qwen4_conv_parallel: tests/test_qwen4_conv_parallel.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: test-qwen4-prefill-reuse
test-qwen4-prefill-reuse: tests/test_qwen4_conv_parallel tests/test_qwen4_moe_mm_specialize
	./tests/test_qwen4_conv_parallel
	./tests/test_qwen4_moe_mm_specialize

tests/test_q8_prefill_variants.o: tests/test_q8_prefill_variants.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_q8_prefill_variants: tests/test_q8_prefill_variants.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: test-q8-prefill-variants
test-q8-prefill-variants: tests/test_q8_prefill_variants
	./tests/test_q8_prefill_variants

tests/test_metal_ssd_experts.o: tests/test_metal_ssd_experts.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_ssd_experts: tests/test_metal_ssd_experts.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: test-metal-ssd-experts
test-metal-ssd-experts: tests/test_metal_ssd_experts
	./tests/test_metal_ssd_experts
	./tests/test_metal_ssd_experts --q4
	./tests/test_metal_ssd_experts --mxfp4

tests/test_metal_command_memory: tests/test_metal_command_memory.c ds4_gpu.h $(CORE_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< $(CORE_OBJS) $(METAL_LDLIBS)

.PHONY: test-metal-command-memory
test-metal-command-memory: tests/test_metal_command_memory
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory row
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory session
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory batch
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory big

tests/test_deepseek41_metal.o: tests/test_deepseek41_metal.c ds4_gpu.h ds4_deepseek41_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ $<

tests/test_deepseek41_metal: tests/test_deepseek41_metal.o $(CORE_OBJS)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(METAL_LDLIBS)

.PHONY: test-deepseek41-metal
test-deepseek41-metal: tests/test_deepseek41_metal
	./tests/test_deepseek41_metal

tests/test_deepseek41_graph.o: tests/test_deepseek41_graph.c ds4.c ds4_gpu.h ds4_engram.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_deepseek41_graph: tests/test_deepseek41_graph.o $(filter-out ds4.o,$(CORE_OBJS))
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(METAL_LDLIBS)

tests/test_deepseek41_prefill.o: tests/test_deepseek41_prefill.c ds4.c ds4_gpu.h ds4_engram.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_deepseek41_prefill: tests/test_deepseek41_prefill.o $(filter-out ds4.o,$(CORE_OBJS))
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(METAL_LDLIBS)

tests/test_metal_tp_bulk: tests/test_metal_tp_bulk.c ds4_gpu.h ds4_tp.h $(CORE_OBJS)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -o $@ $< $(CORE_OBJS) $(METAL_LDLIBS)

tests/test_deepseek41_cli.o: tests/test_deepseek41_cli.c ds4_cli.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -Wno-unused-function -I. -c -o $@ $<

tests/test_deepseek41_cli: tests/test_deepseek41_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

tests/test_metal_dense_mpp.o: tests/test_metal_dense_mpp.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_dense_mpp: tests/test_metal_dense_mpp.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-dense-mpp: tests/test_metal_dense_mpp
	./tests/test_metal_dense_mpp

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o ds4_prompt_prefix.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "  make rocm                Alias for make strix-halo"
	@echo "  make test-mxfp4-rocm     Build and run the synthetic ROCm MXFP4 MoE test"
	@echo "  make test-rocm           Core regression suite on ROCm-only hosts"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval"
	@echo "  make test                Build and run tests"
	@echo "  make test-qwen38-perf    Run the opt-in Qwen3.8 CLI/server MTP VRAM-safe context gate"
	@echo "  make test-constrained-json-live  Run opt-in live DSML/JSON Schema adversarial API tests"
	@echo "  make test-server-interactive  Run interactive protocol and text-constraint fixtures"
	@echo "  make test-interactive-api-live Run interactive HTTP tests (DS4_INTERACTIVE_BASE_URL)"
	@echo "  make test-server-sampling-matrix-static  Check the temperature/thinking/MTP server matrix fixtures"
	@echo "  make test-jsonschemabench-safety  Run the pinned external candidate-vs-oracle safety gate"
	@echo "  make dspark-verify-depth Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth    Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=sm_121

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH="$(CUDA_ARCH)"

strix-halo:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval \
		CORE_OBJS="ds4.o ds4_runtime.o ds4_executor.o ds4_bundle.o ds4_resources.o ds4_checkpoint.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o $(ROCM_MMQ_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

rocm: strix-halo

# Core regression suite for ROCm-only hosts: the CUDA-specific binaries
# (tests/test_sampling, the CUDA session/mixed-batch oracles) are not part
# of this target; run them through `make test` / `make cuda-regression` on
# CUDA hosts.  Everything else mirrors `make test`.
test-rocm:
	$(MAKE) -B ds4_test ds4-eval q4k-dot-test mxfp4-dot-test \
		test-session-state \
		tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args tests/test_prompt_prefix \
		ds4 ds4-server ds4-bench \
		CORE_OBJS="ds4.o ds4_runtime.o ds4_executor.o ds4_bundle.o ds4_resources.o ds4_checkpoint.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o $(ROCM_MMQ_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"
	./ds4-eval --self-test-extractors
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	bash tests/test_gpu_args_cli.sh
	./tests/test_prompt_prefix

ds4: ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)


gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c $(DS4_API_HEADERS)
	$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) $(ROCM_HOST_CFLAGS) -I. -c -o $@ $<

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_q8_scratch.o: tests/test_cuda_q8_scratch.cu cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Icuda/mmq -c -o $@ $<

tests/test_cuda_ssd_cache.o: tests/test_cuda_ssd_cache.c ds4_gpu.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_ssd_batch.o: tests/test_cuda_ssd_batch.c ds4_gpu.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_ssd_batch: tests/test_cuda_ssd_batch.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_tp.o: tests/test_cuda_tp.c ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_tp: tests/test_cuda_tp.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_tp_repack.o: tests/test_cuda_tp_repack.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_tp_repack: tests/test_cuda_tp_repack.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_ssd_repack.o: tests/test_cuda_ssd_repack.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Icuda/mmq -c -o $@ $<

tests/test_cuda_ssd_repack: tests/test_cuda_ssd_repack.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_ssd_cache: tests/test_cuda_ssd_cache.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-ssd-cache
test-cuda-ssd-cache: tests/test_cuda_ssd_cache
	./tests/test_cuda_ssd_cache

tests/test_cuda_q8_rows.o: tests/test_cuda_q8_rows.c ds4_gpu.h
	$(CC) $(QUALITY_CFLAGS) -I. -c -o $@ $<

tests/test_cuda_q8_rows: tests/test_cuda_q8_rows.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-q8-rows
test-cuda-q8-rows: tests/test_cuda_q8_rows
	./tests/test_cuda_q8_rows

tests/test_cuda_reductions.o: tests/test_cuda_reductions.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_reductions: tests/test_cuda_reductions.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-reductions
test-cuda-reductions: tests/test_cuda_reductions
	./tests/test_cuda_reductions

tests/test_cuda_shared.o: tests/test_cuda_shared.cu ds4_gpu.h ds4_deepseek41_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_shared: tests/test_cuda_shared.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-shared
test-cuda-shared: tests/test_cuda_shared
	./tests/test_cuda_shared

tests/test_deepseek41_prefill.o: tests/test_deepseek41_prefill.c ds4.c ds4_gpu.h ds4_engram.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_deepseek41_prefill: tests/test_deepseek41_prefill.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_q8_scratch: tests/test_cuda_q8_scratch.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-q8-scratch
test-cuda-q8-scratch: tests/test_cuda_q8_scratch
	./tests/test_cuda_q8_scratch

tests/test_cuda_dspark_moe.o: cuda/mmq/test/test_iq2_aligned_entry.cu cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Icuda/mmq -c -o $@ $<

tests/test_cuda_dspark_moe: tests/test_cuda_dspark_moe.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-dspark-moe
test-cuda-dspark-moe: tests/test_cuda_dspark_moe
	./tests/test_cuda_dspark_moe

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o ds4_prompt_prefix.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

qwen-numerics: tests/qwen_numerics_probe
	./tests/qwen_numerics_probe

tests/test_mxfp4_cuda: tests/test_mxfp4_cuda.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

test-mxfp4-cuda: tests/test_mxfp4_cuda
	./tests/test_mxfp4_cuda

tests/test_mmq_parity: cuda/mmq/test/test_mmq_parity.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

test-mmq-parity: tests/test_mmq_parity
	./tests/test_mmq_parity
tests/test_deepseek41_cuda.o: tests/test_deepseek41_metal.c ds4_gpu.h ds4_deepseek41_gpu.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_deepseek41_cuda: tests/test_deepseek41_cuda.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-deepseek41-cuda
test-deepseek41-cuda: tests/test_deepseek41_cuda
	./tests/test_deepseek41_cuda

tests/test_qwen4_cuda.o: tests/test_qwen4_kernels.c ds4_gpu.h ds4_gpu_mgpu.h ds4_qwen4_runtime.h $(DS4_API_HEADERS)
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_qwen4_cuda: tests/test_qwen4_cuda.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-qwen4-cuda
test-qwen4-cuda: tests/test_qwen4_cuda
	./tests/test_qwen4_cuda
endif

perf-doctor:
	python3 tools/perf_harness.py doctor

perf-harness-test:
	python3 -m unittest tests.test_perf_harness tests.test_jsonschemabench_subset -v

JSONSCHEMABENCH_MODEL ?= gguf/Qwen3.8-27B-UD-Q4_K_S.gguf
JSONSCHEMABENCH_PREFIX_STEPS ?= 8

test-jsonschemabench-safety:
	python3 -c "import jsonschema"
	python3 tools/perf_harness.py constrained-server \
		--model "$(JSONSCHEMABENCH_MODEL)" \
		--jsonschemabench-subset --jsonschemabench-tier safety \
		--jsonschemabench-check-unsupported \
		--jsonschemabench-prefix-steps "$(JSONSCHEMABENCH_PREFIX_STEPS)" \
		--constraint-mode compare_new_vs_oracle --context 4096 \
		--repetitions 2 --warmup 0 --baseline-run \
		--hypothesis "pinned JSONSchemaBench safety prefixes remain candidate-vs-oracle equivalent and unsupported schemas fail before inference"

perf-fast-test: perf-harness-test

PERF_MODEL ?= gguf/Qwen3.8-27B-UD-Q4_K_S.gguf
PERF_PROMPT ?= tests/long_context_story_prompt.txt

perf-direction-ab:
	@test -n "$(PERF_ID)" || (echo "set a unique PERF_ID, for example PERF_ID=score4-01"; exit 2)
	@test -n "$(PERF_CANDIDATE_ENV)" || (echo "set PERF_CANDIDATE_ENV=NAME=VALUE"; exit 2)
	python3 tools/perf_harness.py run --id "$(PERF_ID)-baseline" \
		--model "$(PERF_MODEL)" --prompt "$(PERF_PROMPT)" \
		--hypothesis "freeze direction baseline for $(PERF_ID)" \
		--metric gen_steady_tps --baseline-run
	python3 tools/perf_harness.py run --id "$(PERF_ID)-candidate" \
		--model "$(PERF_MODEL)" --prompt "$(PERF_PROMPT)" \
		--hypothesis "test $(PERF_CANDIDATE_ENV) for $(PERF_ID)" \
		--metric gen_steady_tps --env "$(PERF_CANDIDATE_ENV)" \
		--baseline "performance-results/$(PERF_ID)-baseline/experiment.json"

perf-slow-test: perf-fast-test qwen-numerics
	python3 -m unittest tests.test_qwen36_fixtures tests.test_qwen36_numerics tests.test_qwen38_compatibility -v

test-qwen38:
	python3 -m unittest tests.test_qwen38_compatibility tests.test_server_sampling_matrix.SamplingMatrixFixtureTests -v

test-qwen38-perf: ds4 ds4-server
	DS4_TEST_QWEN38_PERF=1 python3 -m unittest tests.test_qwen38_compatibility.Qwen38CompatibilityTests.test_cli_and_server_mtp_performance_gate -v

research-fetch:
	python3 tools/fetch_research_corpus.py --refresh
	python3 tools/build_research_indexes.py

research-index:
	python3 tools/build_research_indexes.py

research-check: research-index
	python3 -m unittest tests.test_research_corpus -v

docs-index: research-index
	python3 tools/build_docs_indexes.py

docs-check: docs-index
	python3 -m unittest tests.test_research_corpus tests.test_docs_wiki -v

ds4.o: ds4.c $(DS4_API_HEADERS) ds4_lifecycle.inc ds4_lifecycle_text.h ds4_checkpoint.h ds4_sha256.h ds4_resources.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_qwen.h ds4_env.h ds4_qwen4_runtime.h ds4_gpu_tp.h ds4_deepseek41_gpu.h ds4_linux_memory.h ds4_engram.h ds4_read_queue.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_image.o: ds4_image.c ds4_image.h third_party/iris/jpeg.h third_party/iris/png.h
	$(CC) $(CFLAGS) -c -o $@ ds4_image.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_engram.o: ds4_engram.c ds4_engram.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -c -o $@ ds4_engram.c

ds4_cli.o: ds4_cli.c $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h $(DS4_API_HEADERS) ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h $(DS4_API_HEADERS) ds4_ssd.h ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_prompt_prefix.o: ds4_prompt_prefix.c ds4_prompt_prefix.h $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -c -o $@ ds4_prompt_prefix.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h ds4_env.h ds4_qwen4_runtime.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h ds4_qwen4_runtime.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4_eval_cases.h $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_eval_cases.o: ds4_eval_cases.c ds4_eval_cases.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval_cases.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h $(DS4_API_HEADERS) ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h ds4_env.h ds4_qwen4_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -c -o $@ tests/ds4_test.c


ds4_server_test.o: ds4_server.c $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h ds4_env.h ds4_qwen4_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_SERVER_TEST -DDS4_NO_GPU -c -o $@ ds4_server.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

tests/qwen_numerics_probe.o: tests/qwen_numerics_probe.c ds4_gpu.h ds4_qwen.h
	$(CC) $(QUALITY_CFLAGS) -I. -c -o $@ tests/qwen_numerics_probe.c

tests/test_qwen_mtp_callback.o: tests/test_qwen_mtp_callback.c ds4.c $(DS4_API_HEADERS) ds4_gpu.h ds4_qwen.h ds4_env.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_qwen_mtp_callback: tests/test_qwen_mtp_callback.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_qwen_snapshot_drafts.o: tests/test_qwen_snapshot_drafts.c ds4.c $(DS4_API_HEADERS) ds4_gpu.h ds4_qwen.h ds4_env.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_qwen_snapshot_drafts: tests/test_qwen_snapshot_drafts.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_qwen_memory_lifetime.o: tests/test_qwen_memory_lifetime.c ds4.c $(DS4_API_HEADERS) ds4_gpu.h ds4_qwen.h ds4_env.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_qwen_memory_lifetime: tests/test_qwen_memory_lifetime.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_qwen4_tuning: tests/test_qwen4_tuning.cpp ds4_qwen4_tuning.cuh ds4_qwen4_runtime.h ds4_env.h
	$(CXX) -O2 -Wall -Wextra -std=c++17 -o $@ $<

tests/qwen_mtp_catchup_probe.o: tests/qwen_mtp_catchup_probe.c ds4.c $(DS4_API_HEADERS) ds4_gpu.h ds4_qwen.h ds4_env.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

ifneq ($(UNAME_S),Darwin)
tests/qwen_iq4_dequant_probe.o: tests/qwen_iq4_dequant_probe.cu ds4_cuda.cu ds4_env.h
	$(NVCC) $(NVCCFLAGS) -I. -c -o $@ $<

tests/qwen_iq4_dequant_probe: tests/qwen_iq4_dequant_probe.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/qwen_mtp_catchup_probe: tests/qwen_mtp_catchup_probe.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c $(DS4_API_HEADERS) ds4_lifecycle.inc ds4_lifecycle_text.h ds4_checkpoint.h ds4_sha256.h ds4_resources.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_gpu_tp.h ds4_qwen.h ds4_env.h ds4_qwen4_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h ds4_env.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4_eval_cases.h $(DS4_API_HEADERS) ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_metal.o: ds4_metal.m ds4_qwen.h $(DS4_API_HEADERS) ds4_gpu.h ds4_gpu_tp.h ds4_deepseek41_gpu.h ds4_image.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

tests/test_glm53_kda.o: tests/test_glm53_kda.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_glm53_kda.c

tests/test_glm53_vision_engine.o: tests/test_glm53_vision_engine.c $(DS4_API_HEADERS) ds4_image.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ tests/test_glm53_vision_engine.c

tests/test_glm53_vision_engine: tests/test_glm53_vision_engine.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_glm53_vision_prompt.o: tests/test_glm53_vision_prompt.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ tests/test_glm53_vision_prompt.c

tests/test_glm53_vision_prompt: tests/test_glm53_vision_prompt.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_deepseek4_vision_image.o: tests/test_deepseek4_vision_image.c ds4_image.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_deepseek4_vision_image: tests/test_deepseek4_vision_image.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/test_image_decode.o: tests/test_image_decode.c ds4_image.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_image_decode: tests/test_image_decode.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

ifeq ($(UNAME_S),Darwin)
$(GLM53_KDA_TEST): tests/test_glm53_kda.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
$(GLM53_KDA_TEST): tests/test_glm53_kda.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

.PHONY: test-glm53-kda
test-glm53-kda: $(GLM53_KDA_TEST)
	./$(GLM53_KDA_TEST)

ifeq ($(UNAME_S),Darwin)
tests/test_qwen4_kernels.o: tests/test_qwen4_kernels.c ds4_gpu.h $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen4_kernels.c

$(QWEN4_KERNEL_TEST): tests/test_qwen4_kernels.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

endif

tests/test_qwen4_vision.o: tests/test_qwen4_vision.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen4_vision.c

tests/test_qwen4_vision: tests/test_qwen4_vision.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

.PHONY: test-qwen4-kernels test-qwen4-q2 test-qwen4-vision
test-qwen4-kernels: $(QWEN4_KERNEL_TEST)
	./$(QWEN4_KERNEL_TEST)

test-qwen4-q2: $(QWEN4_KERNEL_TEST) tests/test_qwen4_moe_mm_specialize
	DS4_TEST_QWEN4_MV_EXACT=1 ./$(QWEN4_KERNEL_TEST)
	./tests/test_qwen4_moe_mm_specialize

# DS4_QWEN4_SNAPSHOT=<HF checkpoint dir> DS4_QWEN4_MMPROJ=<mmproj.gguf> DS4_QWEN4_IMAGE=<image>
test-qwen4-vision: tests/test_qwen4_vision
	@test -n "$(DS4_QWEN4_SNAPSHOT)" -a -n "$(DS4_QWEN4_MMPROJ)" -a -n "$(DS4_QWEN4_IMAGE)" || \
	  { echo "set DS4_QWEN4_SNAPSHOT, DS4_QWEN4_MMPROJ and DS4_QWEN4_IMAGE"; exit 1; }
	python3 tests/qwen4_vision_ref.py --snapshot "$(DS4_QWEN4_SNAPSHOT)" --mmproj "$(DS4_QWEN4_MMPROJ)" --image "$(DS4_QWEN4_IMAGE)"

tests/test_glm53_kda_rocm.o: tests/test_glm53_kda.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

$(GLM53_KDA_ROCM_TEST): tests/test_glm53_kda_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-glm53-kda-rocm: $(GLM53_KDA_ROCM_TEST)
	./$(GLM53_KDA_ROCM_TEST)

tests/test_glm_attention_rocm.o: tests/test_glm_attention.c $(DS4_API_HEADERS) ds4_gpu.h ds4_linux_memory.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_glm_attention_rocm: tests/test_glm_attention_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

.PHONY: test-glm-attention-rocm
test-glm-attention-rocm: tests/test_glm_attention_rocm
	./tests/test_glm_attention_rocm

tests/test_linux_memory: tests/test_linux_memory.c ds4_linux_memory.h
	$(CC) $(CFLAGS) -I. -o $@ $<

tests/test_rocm_memory: tests/test_rocm_memory.cu ds4_rocm_memory.h ds4_linux_memory.h
	$(HIPCC) $(ROCM_CFLAGS) -I. -o $@ $<

.PHONY: test-linux-memory test-rocm-memory
test-linux-memory: tests/test_linux_memory
	./tests/test_linux_memory

test-rocm-memory: tests/test_rocm_memory
	./tests/test_rocm_memory

tests/test_glm_attention.o: tests/test_glm_attention.c $(DS4_API_HEADERS) ds4_gpu.h ds4_linux_memory.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ $<

ifeq ($(UNAME_S),Darwin)
tests/test_glm_attention: tests/test_glm_attention.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
tests/test_glm_attention: tests/test_glm_attention.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

.PHONY: test-glm-attention
test-glm-attention: tests/test_glm_attention
	./tests/test_glm_attention

tests/test_ssd_cache: tests/test_ssd_cache.c ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_ssd_cache.c ds4_ssd.c

.PHONY: test-ssd-cache
test-ssd-cache: tests/test_ssd_cache
	./tests/test_ssd_cache

tests/test_engram: tests/test_engram.c ds4_engram.c ds4_engram.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -o $@ tests/test_engram.c ds4_engram.c $(LDLIBS)

.PHONY: test-engram
test-engram: tests/test_engram
	./tests/test_engram

tests/test_deepseek41_gguf.o: tests/test_deepseek41_gguf.c ds4.c $(DS4_API_HEADERS) ds4_engram.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -DDS4_NO_GPU -I. -c -o $@ $<

tests/test_deepseek41_gguf: tests/test_deepseek41_gguf.o ds4_engram.c $(CPU_SUPPORT_OBJS)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -o $@ $^ $(LDLIBS)

.PHONY: test-deepseek41-gguf
test-deepseek41-gguf: tests/test_deepseek41_gguf
	./tests/test_deepseek41_gguf

tests/test_qwen4_ngrams.o: tests/test_qwen4_ngrams.c ds4.c $(DS4_API_HEADERS)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_ngrams: tests/test_qwen4_ngrams.o $(CPU_SUPPORT_OBJS)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(LDLIBS)

.PHONY: test-qwen4-ngrams
test-qwen4-ngrams: tests/test_qwen4_ngrams
	./tests/test_qwen4_ngrams

tests/test_qwen4_ngram_state.o: tests/test_qwen4_ngram_state.c ds4.c $(DS4_API_HEADERS)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_ngram_state: tests/test_qwen4_ngram_state.o $(filter-out ds4.o,$(CORE_OBJS))
ifeq ($(UNAME_S),Darwin)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_qwen4_prefill.o: tests/test_qwen4_prefill.c $(DS4_API_HEADERS)
	$(CC) $(QUALITY_CFLAGS) -I. -c -o $@ $<

tests/test_qwen4_prefill: tests/test_qwen4_prefill.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

ds4.o ds4_cpu.o ds4_cpu_test_hooks.o ds4_cuda_test_hooks.o ds4_metal.o ds4_cuda.o ds4_rocm.o tests/test_qwen4_cuda.o tests/test_qwen4_kernels.o tests/test_qwen4_ngram_state.o: ds4_qwen4_vision.h
ds4.o ds4_cpu.o ds4_server.o ds4_server_cpu.o ds4_server_test.o ds4_test.o ds4_tp.o ds4_cpu_test_hooks.o ds4_test_hooks.o ds4_cuda_test_hooks.o tests/test_server_agentic_payload.o tests/test_qwen4_cuda.o: ds4_qwen4_runtime.h

ds4_cuda.o: ds4_cuda.cu ds4_read_queue.h ds4_qwen4_expert_cache.h ds4_qwen4_cpu.h ds4_qwen4_hybrid.cuh ds4_qwen4_transfer.cuh ds4_qwen4_profile.cuh ds4_qwen4_residency.cuh ds4_qwen4_kv.cuh ds4_qwen4_runtime.h ds4_qwen4_tuning.cuh ds4_qwen.h ds4_env.h ds4_gpu.h ds4_gpu_tp.h ds4_gpu_mgpu.h ds4_linux_memory.h ds4_deepseek41_gpu.h ds4_deepseek41_cuda.cuh ds4_glm53_vision_gpu.cuh ds4_deepseek4_vision_gpu.cuh ds4_qwen4_cuda.cuh ds4_image.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

ds4_cuda.o: ds4_qwen4_ranks.cuh
ds4_cuda.o: ds4_qwen4_ring.cuh


ds4_qwen4_cpu.o: ds4_qwen4_cpu.c ds4_qwen4_cpu.h ds4_qwen4_cpu_internal.h ds4_qwen4_cpu_sign_masks.inc ds4_iq2_tables_cuda.inc
	$(CC) $(QUALITY_CFLAGS) -ffp-contract=off -c -o $@ ds4_qwen4_cpu.c

# Vendored mmq pieces (see cuda/mmq/VENDOR.md).  ds4_mmq.cu transitively
# pulls in mmq.cuh which has heavy template instantiation -- each piece
# compiles in its own TU and links in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c -o $@ $<

ds4_rocm.o: ds4_rocm.cu ds4_rocm.h ds4_rocm_memory.h ds4_linux_memory.h ds4_gpu.h ds4_gpu_tp.h ds4_glm53_vision_gpu.cuh ds4_deepseek4_vision_gpu.cuh ds4_image.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS) ds4_qwen.h
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

cuda/mmq/ds4_ggml_stubs.rocm.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh cuda/mmq/vendors/hip.h ds4_rocm_memory.h ds4_linux_memory.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/ds4_mmq.rocm.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/quantize.rocm.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/mmid.rocm.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/mmvq.rocm.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/d2r_stubs.rocm.o: cuda/mmq/test/d2r_stubs.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

tests/test_mxfp4_rocm.o: tests/test_mxfp4_rocm.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_mxfp4_rocm: tests/test_mxfp4_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

tests/bench_mxfp4_rocm.o: tests/bench_mxfp4_rocm.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/bench_mxfp4_rocm: tests/bench_mxfp4_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-mxfp4-rocm: tests/test_mxfp4_rocm
	./tests/test_mxfp4_rocm

ds4_rocm_compat.o: ds4_rocm_compat.cu ds4_gpu.h ds4_gpu_tp.h ds4_gpu_mgpu.h ds4_gpu_args.h ds4_rocm_memory.h ds4_linux_memory.h ds4_qwen.h
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_compat.cu

ds4_rocm_unavailable.o: ds4_rocm_unavailable.cu
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_unavailable.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/qwen_numerics_probe: tests/qwen_numerics_probe.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/qwen_flash_quality_probe.o: tests/qwen_flash_quality_probe.c $(DS4_API_HEADERS)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ $<

tests/qwen_flash_quality_probe: tests/qwen_flash_quality_probe.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_layer_pack.o: tests/test_layer_pack.c ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_args.o: tests/test_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_cpu_test_hooks.o: ds4.c $(DS4_API_HEADERS) ds4_image.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h ds4_qwen.h ds4_env.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

ds4_test_hooks.o: ds4.c $(DS4_API_HEADERS) ds4_gpu.h ds4_gpu_mgpu.h ds4_qwen.h ds4_layer_pack.h ds4_env.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c $(DS4_API_HEADERS) ds4_gpu_mgpu.h ds4_qwen.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o ds4_cpu_test_hooks.o $(CPU_SUPPORT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_sampling.o: tests/test_sampling.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -fno-finite-math-only -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cpu_test_hooks.o $(CPU_SUPPORT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state.o: tests/test_session_state.c ds4.c $(DS4_API_HEADERS) ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -I. -c -o $@ $<

tests/test_session_state: tests/test_session_state.o $(CPU_SUPPORT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state_gpu.o: tests/test_session_state.c ds4.c $(DS4_API_HEADERS) ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -c -o $@ $<

tests/test_session_state_gpu: tests/test_session_state_gpu.o $(filter-out ds4.o,$(CORE_OBJS))
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_tp_commands.o: tests/test_tp_commands.c ds4_tp.c ds4_tp.h $(DS4_API_HEADERS) ds4_gpu_tp.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_tp_commands: tests/test_tp_commands.o $(filter-out ds4_tp.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_tp_rdma.o: tests/test_tp_rdma.c ds4_tp.c ds4_tp.h $(DS4_API_HEADERS) ds4_gpu_tp.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_tp_rdma: tests/test_tp_rdma.o $(filter-out ds4_tp.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_tp_link.o: tests/test_tp_link.c ds4_tp.h $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_tp_link: tests/test_tp_link.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_tp_tcp.o: tests/test_tp_tcp.c ds4_tp.c ds4_tp.h $(DS4_API_HEADERS) ds4_gpu_tp.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_tp_tcp: tests/test_tp_tcp.o $(filter-out ds4_tp.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-session-state
test-session-state: tests/test_session_state tests/test_tp_commands tests/test_tp_rdma tests/test_tp_tcp
	./tests/test_session_state
	./tests/test_tp_commands
	./tests/test_tp_rdma
	./tests/test_tp_tcp

ifneq ($(UNAME_S),Darwin)
tests/test_gpu_xdev.o: tests/test_gpu_xdev.c ds4_gpu.h ds4_gpu_mgpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_xdev: tests/test_gpu_xdev.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_model_cache.o: tests/test_gpu_model_cache.c ds4_gpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_cache: tests/test_gpu_model_cache.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_lookup_cache_strict.o: tests/test_gpu_lookup_cache_strict.c ds4_gpu.h ds4_gpu_mgpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_lookup_cache_strict: tests/test_gpu_lookup_cache_strict.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_cuda_test_hooks.o: ds4.c $(DS4_API_HEADERS) ds4_gpu.h ds4_gpu_mgpu.h ds4_gpu_tp.h ds4_layer_pack.h ds4_qwen.h ds4_env.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ ds4.c

tests/test_engine_mgpu_refusal.o: tests/test_engine_mgpu_refusal.c $(DS4_API_HEADERS) ds4_gpu_mgpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_refusal: tests/test_engine_mgpu_refusal.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_mgpu_runtime.o: tests/test_engine_mgpu_runtime.c $(DS4_API_HEADERS) ds4_gpu_mgpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_runtime: tests/test_engine_mgpu_runtime.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o $(filter-out ds4.o,$(CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_correctness.o: tests/test_engine_correctness.c $(DS4_API_HEADERS) ds4_gpu_mgpu.h ds4_qwen.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_correctness: tests/test_engine_correctness.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_session_batch.o: tests/test_cuda_session_batch.c $(DS4_API_HEADERS) ds4_gpu_args.h ds4_gpu_mgpu.h ds4_tp.h ds4_qwen.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_session_batch: tests/test_cuda_session_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o $(filter-out ds4.o,$(CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-session-batch: tests/test_cuda_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_session_batch

tests/test_cuda_mixed_batch.o: tests/test_cuda_mixed_batch.c $(DS4_API_HEADERS) ds4_gpu_args.h ds4_gpu_mgpu.h ds4_tp.h ds4_qwen.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_mixed_batch: tests/test_cuda_mixed_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o $(filter-out ds4.o,$(CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-mixed-batch: tests/test_cuda_mixed_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_mixed_batch

tests/test_agentic_checkpoint.o: tests/test_agentic_checkpoint.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_agentic_checkpoint: tests/test_agentic_checkpoint.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-agentic-checkpoint: tests/test_agentic_checkpoint
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_agentic_checkpoint
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o ds4_test_hooks.o $(filter-out ds4.o,$(CORE_OBJS))
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

ds4_server_test: ds4_server_test.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

AGENTIC_HEADERS = ds4_agentic_settings.h ds4_prompt_settings.h ds4_server_text.h
INTERACTIVE_HEADERS = $(AGENTIC_HEADERS) ds4_interactive_checkpoint.h ds4_interactive_settings.h ds4_edit_constraint.h ds4_edit_document.h ds4_interactive_edit.h ds4_interactive_config.h ds4_interactive_grammar.h ds4_interactive_runtime.h
ds4_server.o ds4_server_cpu.o ds4_server_test.o: $(INTERACTIVE_HEADERS)
ds4_test.o tests/test_server_agentic_payload.o tests/test_server_resident_frontier.o: $(INTERACTIVE_HEADERS)

tests/test_server_interactive.o: tests/test_server_interactive.c ds4_server.c $(INTERACTIVE_HEADERS)
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ $<

tests/test_server_interactive: tests/test_server_interactive.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-server-interactive
test-server-interactive: tests/test_server_interactive
	./tests/test_server_interactive

tests/test_server_memory_lifetime.o: tests/test_server_memory_lifetime.c ds4_server.c $(INTERACTIVE_HEADERS)
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ $<

tests/test_server_memory_lifetime: tests/test_server_memory_lifetime.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/bench_interactive_edit.o: tests/bench_interactive_edit.c ds4_server.c $(INTERACTIVE_HEADERS)
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ $<

tests/bench_interactive_edit: tests/bench_interactive_edit.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: bench-interactive-edit
bench-interactive-edit: tests/bench_interactive_edit
	./tests/bench_interactive_edit

.PHONY: test-interactive-api-live
test-interactive-api-live:
	@test -n "$$DS4_INTERACTIVE_BASE_URL" || (echo "set DS4_INTERACTIVE_BASE_URL to a running server"; exit 2)
	python3 -m unittest tests.test_interactive_api -v

tests/test_prompt_prefix.o: tests/test_prompt_prefix.c ds4_prompt_prefix.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_prompt_prefix: tests/test_prompt_prefix.o ds4_prompt_prefix.o
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: test-frontends
test-frontends: ds4_test
	./ds4_test --server

test: ds4_test ds4_server_test ds4-eval q4k-dot-test mxfp4-dot-test test-session-state test-linux-memory test-read-queue test-engram \
	tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args \
	tests/test_deepseek4_vision_image tests/test_image_decode tests/test_prompt_prefix tests/test_server_interactive tests/test_server_memory_lifetime $(SAMPLING_TEST) ds4 ds4-server ds4-bench
	./ds4-eval --validate-cases
	./ds4-eval --self-test-extractors
	./ds4_server_test
	./tests/test_server_interactive
	./tests/test_server_memory_lifetime
	python3 -m unittest tests.test_qwen38_compatibility tests.test_server_sampling_matrix.SamplingMatrixFixtureTests -v
	./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	bash tests/test_gpu_args_cli.sh
	./tests/test_prompt_prefix
	./tests/test_sampling
	./tests/test_deepseek4_vision_image
	./tests/test_image_decode

test-constrained-json-live:
	@if [ -z "$$DS4_CONSTRAINED_BASE_URL" ]; then \
		echo "error: set DS4_CONSTRAINED_BASE_URL, for example http://127.0.0.1:8080"; \
		exit 2; \
	fi
	@if [ "$${DS4_CONSTRAINED_SERVER_CTX:-22593}" != "22593" ]; then \
		echo "error: constrained live tests require ds4-server --ctx 22593"; \
		exit 2; \
	fi
	python3 -m unittest tests.test_constrained_json_api -v

test-server-sampling-matrix-static:
	python3 -m unittest tests.test_server_sampling_matrix.SamplingMatrixFixtureTests -v

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run ./download_model.sh ds4f-dspark or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: set DS4_TEST_MTP=FILE to a legacy support GGUF"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

tests/test_qwen35_mrope.o: tests/test_qwen35_mrope.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_qwen35_mrope: tests/test_qwen35_mrope.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-qwen35-mrope
test-qwen35-mrope: tests/test_qwen35_mrope
	./tests/test_qwen35_mrope

tests/test_server_agentic_payload.o: tests/test_server_agentic_payload.c ds4_server.c $(DS4_API_HEADERS) ds4_env.h ds4_qwen4_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -c -o $@ $<

tests/test_server_agentic_payload: tests/test_server_agentic_payload.o ds4_help.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_server_resident_frontier.o: tests/test_server_resident_frontier.c ds4_server.c $(DS4_API_HEADERS) ds4_env.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -c -o $@ $<

tests/test_server_resident_frontier: tests/test_server_resident_frontier.o ds4_help.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

.PHONY: test-server-resident-frontier
test-server-resident-frontier: tests/test_server_resident_frontier
	./tests/test_server_resident_frontier "$(DS4_TEST_MODEL)"

.PHONY: test-download-model
test-download-model:
	python3 tests/test_model_download.py

.PHONY: test-quality-api
# Only the scorer's JSON parser is needed; discard the unused engine entry point.
test-quality-api: tests/test_quality_api.c gguf-tools/quality-testing/score_official.c
	$(CC) $(QUALITY_CFLAGS) -I. -ffunction-sections -fdata-sections -o tests/test_quality_api tests/test_quality_api.c -Wl,$(if $(filter Darwin,$(UNAME_S)),-dead_strip,--gc-sections) -lm
	./tests/test_quality_api
	python3 tests/test_collect_official.py

ds4.o ds4_cpu.o ds4_server.o ds4_server_cpu.o \
ds4_test.o \
ds4_cpu_test_hooks.o ds4_cuda_test_hooks.o tests/test_session_state.o \
tests/test_session_state_gpu.o: ds4_tool_text.h

clean:
	rm -f tests/test_server_interactive tests/bench_interactive_edit
	rm -f tests/test_qwen_snapshot_drafts
	rm -f tests/test_qwen_memory_lifetime
	rm -f tests/test_server_memory_lifetime
	rm -f tests/qwen_flash_quality_probe
	rm -f tests/test_qwen4_tuning
	rm -f tests/test_qwen4_ranks tests/test_qwen4_cpu
	rm -f tests/test_qwen35_mrope tests/test_server_agentic_payload tests/test_qwen_mtp_callback tests/test_server_resident_frontier tests/test_agentic_checkpoint tests/qwen_numerics_probe tests/test_mmq_parity
	rm -f tests/test_qwen4_ngrams
	rm -f tests/test_qwen4_ngram_state
	rm -f tests/test_web_recovery
	rm -f tests/test_metal_ssd_experts
	rm -f tests/test_metal_command_memory
	rm -f tests/test_deepseek41_metal
	rm -f tests/test_deepseek41_cuda
	rm -f tests/test_cuda_q8_rows
	rm -f tests/test_cuda_reductions
	rm -f tests/test_cuda_shared
	rm -f tests/test_cuda_ssd_cache
	rm -f tests/test_cuda_ssd_batch
	rm -f tests/test_cuda_tp
	rm -f tests/test_cuda_tp_repack
	rm -f tests/test_cuda_ssd_repack
	rm -f tests/test_deepseek41_gguf
	rm -f tests/test_deepseek41_graph tests/test_deepseek41_cli
	rm -f tests/test_deepseek41_prefill
	rm -f tests/test_metal_tp_bulk
	rm -f tests/test_cuda_q8_scratch
	rm -f tests/test_cuda_dspark_moe
	rm -f tests/test_quality_api
	rm -f tests/test_linux_memory tests/test_rocm_memory
	rm -f tests/test_glm_attention tests/test_glm_attention_rocm
	rm -f tests/test_ssd_cache tests/test_engram
	rm -f tests/test_session_state tests/test_session_state_gpu tests/test_tp_commands
	rm -f tests/test_tp_rdma tests/test_tp_link tests/test_tp_tcp
	rm -f tests/test_metal_tp_spec
	rm -f tests/test_metal_tp_cancel
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4_cpu ds4_native ds4_server_test ds4_test gguf-tools/quality-testing/score_official gguf-tools/quality-testing/score_official.o speed-bench/metal_decode_schedule_bench speed-bench/metal_prefill_variant_bench speed-bench/*.o tests/test_q4k_dot tests/test_mxfp4_dot tests/test_mxfp4_metal tests/test_mxfp4_rocm tests/test_mxfp4_cuda tests/test_metal_session_batch tests/test_metal_moe_prefill tests/test_qwen4_moe_mm_specialize tests/test_qwen4_conv_parallel tests/test_q8_prefill_variants tests/test_metal_dense_mpp tests/test_glm53_kda tests/test_glm53_kda_rocm tests/test_glm53_vision_engine tests/test_glm53_vision_prompt tests/test_deepseek4_vision_image tests/test_prompt_prefix tests/test_gpu_xdev tests/test_gpu_model_cache tests/test_gpu_lookup_cache_strict tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness tests/test_sampling tests/test_cuda_session_batch tests/test_cuda_mixed_batch tests/*.o *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
	rm -f tests/test_image_decode
	rm -f tests/test_read_queue tests/test_qwen4_expert_cache tests/test_qwen4_kernels tests/test_qwen4_cuda tests/test_qwen4_vision tests/test_qwen4_prefill
	rm -f speed-bench/session_concurrency_bench

# The active tokenizer includes generated Unicode classes.
ds4.o ds4_cpu.o ds4_cpu_test_hooks.o: ds4_qwen4_unicode.inc

# Host policy regression; actual CUDA replacement is covered by forced-cache
# pressure runs of the Qwen4 kernel/model suites.
tests/test_qwen4_expert_cache: tests/test_qwen4_expert_cache.c ds4_qwen4_expert_cache.h
	$(CC) $(QUALITY_CFLAGS) -I. -o $@ $<

.PHONY: test-qwen4-expert-cache
tests/test_qwen4_cpu: tests/test_qwen4_cpu.c ds4_qwen4_cpu.c ds4_qwen4_cpu.h ds4_qwen4_cpu_internal.h ds4_qwen4_cpu_sign_masks.inc ds4_iq2_tables_cuda.inc
	$(CC) $(QUALITY_CFLAGS) -ffp-contract=off -o $@ tests/test_qwen4_cpu.c ds4_qwen4_cpu.c -lm

.PHONY: test-qwen4-cpu
test-qwen4-cpu: tests/test_qwen4_cpu
	./tests/test_qwen4_cpu

tests/test_qwen4_ranks: tests/test_qwen4_ranks.cpp ds4_qwen4_ranks.cuh ds4_qwen4_expert_cache.h
	$(CXX) -O2 -std=c++17 -Wall -Wextra -I. -o $@ tests/test_qwen4_ranks.cpp

test-qwen4-expert-cache: tests/test_qwen4_expert_cache
	./tests/test_qwen4_expert_cache

# Host queue lifetime and bounded concurrency; no model/GPU required.
tests/test_read_queue: tests/test_read_queue.c ds4_read_queue.h
	$(CC) $(QUALITY_CFLAGS) -I. -o $@ $< -pthread

.PHONY: test-read-queue
test-read-queue: tests/test_read_queue
	./tests/test_read_queue

.PHONY: test-product-text
test-product-text:
	python3 -m unittest tests.test_product_text_configuration -v

# Product text placement is part of the normal regression suite.
test: test-product-text

# Deterministic lifecycle baseline (CUDA target, run serially).
ifeq ($(UNAME_S),Linux)
tools/probe_suite_baseline.o: tools/probe_suite_baseline.c $(DS4_API_HEADERS)
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tools/probe_suite_baseline: tools/probe_suite_baseline.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

ds4_runtime.o: ds4_runtime.c ds4_runtime.h ds4_resources.h
ds4.o ds4_cpu.o ds4_test_hooks.o ds4_cpu_test_hooks.o ds4_cuda_test_hooks.o \
 tests/test_session_state.o tests/test_session_state_gpu.o: ds4_runtime.h

tests/test_runtime: tests/test_runtime.c ds4_runtime.c ds4_runtime.h ds4_resources.o
	$(CC) $(CFLAGS) -I. -o $@ tests/test_runtime.c ds4_runtime.c ds4_resources.o $(LDLIBS)

ifeq ($(UNAME_S),Linux)
tests/test_runtime_cuda.o: tests/test_runtime_cuda.c $(DS4_API_HEADERS) ds4_gpu.h ds4_gpu_mgpu.h ds4_runtime.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_runtime_cuda: tests/test_runtime_cuda.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

.PHONY: test-runtime
test-runtime: tests/test_runtime
	./tests/test_runtime

test: test-runtime

.PHONY: test-suite-contracts
test-suite-contracts:
	python3 -m unittest tests.test_suite_contracts tests.test_suite_baseline_manifests tests.test_suite_artifacts -v

test: test-suite-contracts

clean-suite-runtime:
	rm -f tests/test_runtime tests/test_runtime_cuda tests/test_resources tests/test_lifecycle_cuda tests/test_executor tests/test_executor_cuda tests/test_bundle tools/probe_suite_baseline tools/probe_suite_baseline.o

.PHONY: clean-suite-runtime
clean: clean-suite-runtime

# Native multimodal suite has an explicit numerical profile; never inherit
# legacy -ffast-math/--use_fast_math. CPU is a reference/debug backend only.
SUITE_CFLAGS = -O2 -g -fPIC -std=c11 -D_GNU_SOURCE -Wall -Wextra -fno-fast-math
SUITE_NVCCFLAGS = -O2 -g -lineinfo -Xcompiler -fPIC --fmad=false $(NVCC_ARCH_FLAGS)
SUITE_PYTHON ?= python3
CUDNN_HOME ?= $(CUDA_HOME)
SUITE_CUDA_LIBDIR ?= $(CUDA_HOME)/lib64
SUITE_CUDA_LDLIBS = -lm -pthread -L$(SUITE_CUDA_LIBDIR) -l:libcudart.so.13 -l:libcublas.so.13 -l:libcublasLt.so.13 -Wl,-rpath,$(abspath $(SUITE_CUDA_LIBDIR))
SUITE_CUDNN_LIBS = -L$(CUDNN_HOME)/lib -L$(CUDNN_HOME)/lib64 -l:libcudnn.so.9 -Wl,-rpath,$(abspath $(CUDNN_HOME)/lib)
SUITE_API_HEADERS = ds4_numeric.h ds4_bundle.h ds4_resources.h ds4_runtime.h

ds4_embedding_gemma2.o: ds4_embedding_gemma2.c ds4_embedding_gemma2.h ds4_gemma_tokenizer.h ds4_numeric_weights.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_gemma_tokenizer.o: ds4_gemma_tokenizer.c ds4_gemma_tokenizer.h ds4_embedding_gemma2.h ds4_sha256.h
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_qwen_image_reference.o: ds4_qwen_image_reference.c ds4_qwen_image_reference.h ds4_qwen_image_input.h ds4_image.h
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<

SUITE_OBJS = ds4_embedding_gemma2.o ds4_gemma_tokenizer.o ds4_qwen_image_reference.o ds4_qwen_image_pipeline.o ds4_executor.o ds4_qwen_image_attention.o ds4_qwen_image_dit.o ds4_qwen_image_vae.o ds4_qwen_image_rgba.o ds4_qwen_image_observer.o ds4_qwen_image_encoder.o ds4_qwen_image_input.o ds4_qwen_image_tokenizer.o ds4_unicode_nfc.o ds4_numeric.o ds4_numeric_weights.o ds4_bundle.o ds4_resources.o ds4_runtime.o ds4_image.o
ds4_qwen_image_pipeline.o: ds4_qwen_image_pipeline.c ds4_qwen_image_pipeline.h ds4_qwen_image_components.h ds4_qwen_image_reference.h ds4_sha256.h ds4_executor.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_qwen_image_encoder.o ds4_qwen_image_dit.o ds4_qwen_image_vae.o: ds4_qwen_image_components.h
tests/libds4_suite_cuda.so: $(filter-out ds4_resources.o,$(SUITE_OBJS)) tests/suite_resources.o ds4_numeric_cuda.o ds4_numeric_cudnn.o
	$(CC) -shared -o $@ $^ $(SUITE_CUDA_LDLIBS) $(SUITE_CUDNN_LIBS)
tests/libds4_suite_cpu.so: $(filter-out ds4_resources.o,$(SUITE_OBJS)) tests/suite_resources.o
	$(CC) -shared -o $@ $^ -lm -pthread
tests/probe_native_suite: tests/probe_native_suite.c $(SUITE_OBJS) ds4_numeric_cuda.o ds4_numeric_cudnn.o
	$(CC) $(SUITE_CFLAGS) -I. -o $@ $^ $(SUITE_CUDA_LDLIBS) $(SUITE_CUDNN_LIBS)
tests/probe_qwen_image_pipeline: tests/probe_qwen_image_pipeline.c $(SUITE_OBJS) ds4_numeric_cuda.o ds4_numeric_cudnn.o
	$(CC) $(SUITE_CFLAGS) -I. -o $@ $^ $(SUITE_CUDA_LDLIBS) $(SUITE_CUDNN_LIBS)
libds4-suite-cpu.a: $(SUITE_OBJS)
	$(AR) rcs $@ $^
libds4-suite-cuda.a: $(SUITE_OBJS) ds4_numeric_cuda.o ds4_numeric_cudnn.o
	$(AR) rcs $@ $^
tests/test_numeric_lifetime: tests/test_numeric_lifetime.c ds4_numeric.o ds4_resources.o
	$(CC) $(SUITE_CFLAGS) -I. -o $@ $^ -lm -pthread
.PHONY: test-numeric-lifetime clean-suite-numeric
test-numeric-lifetime: tests/test_numeric_lifetime
	./tests/test_numeric_lifetime
tests/test_qwen_image_input: tests/test_qwen_image_input.c ds4_qwen_image_input.o ds4_qwen_image_tokenizer.o ds4_unicode_nfc.o ds4_numeric.o ds4_numeric_weights.o ds4_resources.o ds4_image.o
	$(CC) $(SUITE_CFLAGS) -I. -o $@ $^ -lm -pthread
.PHONY: test-qwen-image-input
test-qwen-image-input: tests/test_qwen_image_input
	./tests/test_qwen_image_input
clean-suite-numeric:
	rm -f tests/probe_native_suite tests/probe_qwen_image_pipeline tests/libds4_suite_cpu.so tests/libds4_suite_cuda.so
	rm -f tests/test_qwen_image_output
	rm -f tests/probe_qwen_image_graph tests/probe_qwen_image_graph.o tests/libds4_qwen_image_graph_cpu.so
	rm -f $(filter-out ds4_bundle.o ds4_resources.o ds4_runtime.o ds4_image.o,$(SUITE_OBJS)) ds4_numeric_cuda.o ds4_numeric_cudnn.o libds4-suite-cpu.a libds4-suite-cuda.a tests/probe_qwen_image_encoder tests/probe_qwen_image_encoder.o tests/test_numeric_lifetime tests/libds4_numeric_cpu.so tests/libds4_numeric_cuda.so tests/suite_resources.o tests/libds4_image_suite.so tests/libds4_qwen_image_tokenizer.so tests/test_qwen_image_input
clean: clean-suite-numeric
ds4_qwen_image_encoder.o: ds4_qwen_image_encoder.c ds4_qwen_image_encoder.h ds4_qwen_image_observer.h ds4_qwen_image_input.h ds4_qwen_image_input_internal.h ds4_qwen_image_tokenizer.h ds4_numeric_weights.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_qwen_image_dit.o: ds4_qwen_image_dit.c ds4_qwen_image_dit.h ds4_qwen_image_dit_constants.h ds4_qwen_image_graph.h ds4_numeric_weights.h ds4_qwen_image_attention.h ds4_qwen_image_encoder.h ds4_qwen_image_observer.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_qwen_image_attention.o: ds4_qwen_image_attention.c ds4_qwen_image_attention.h ds4_qwen_image_graph.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_qwen_image_vae.o: ds4_qwen_image_vae.c ds4_qwen_image_vae.h ds4_qwen_image_vae_stats.h ds4_qwen_image_graph.h ds4_qwen_image_attention.h ds4_qwen_image_rgba_internal.h ds4_qwen_image_rgba.h ds4_qwen_image_observer.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_qwen_image_rgba.o: ds4_qwen_image_rgba.c ds4_qwen_image_rgba.h ds4_qwen_image_rgba_internal.h ds4_image.h third_party/iris/png.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_qwen_image_observer.o: ds4_qwen_image_observer.c ds4_qwen_image_observer.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
tests/test_qwen_image_output: tests/test_qwen_image_output.c ds4_qwen_image_rgba.o ds4_qwen_image_observer.o ds4_numeric.o ds4_resources.o ds4_image.o
	$(CC) $(SUITE_CFLAGS) -I. -o $@ $^ -lm -pthread
.PHONY: test-qwen-image-output
test-qwen-image-output: tests/test_qwen_image_output
	./tests/test_qwen_image_output
ds4_qwen_image_input.o: ds4_qwen_image_input.c ds4_qwen_image_input.h ds4_qwen_image_input_internal.h ds4_qwen_image_tokenizer.h ds4_image.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_numeric_weights.o: ds4_numeric_weights.c ds4_numeric_weights.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_qwen_image_tokenizer.o: ds4_qwen_image_tokenizer.c ds4_qwen_image_tokenizer.h ds4_sha256.h third_party/unicode/image_classes.inc ds4_unicode_nfc.h
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_unicode_nfc.o: ds4_unicode_nfc.c ds4_unicode_nfc.h third_party/unicode/nfc9.inc
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
tests/probe_qwen_image_encoder: tests/probe_qwen_image_encoder.c $(SUITE_OBJS) ds4_numeric_cuda.o ds4_numeric_cudnn.o
	$(CC) $(SUITE_CFLAGS) -I. -c -o tests/probe_qwen_image_encoder.o $<
	$(CC) -o $@ tests/probe_qwen_image_encoder.o $(filter %.o,$^) $(SUITE_CUDA_LDLIBS) $(SUITE_CUDNN_LIBS)
tests/probe_qwen_image_graph: tests/probe_qwen_image_graph.c $(SUITE_OBJS) ds4_numeric_cuda.o ds4_numeric_cudnn.o
	$(CC) $(SUITE_CFLAGS) -I. -c -o tests/probe_qwen_image_graph.o $<
	$(CC) -o $@ tests/probe_qwen_image_graph.o $(filter %.o,$^) $(SUITE_CUDA_LDLIBS) $(SUITE_CUDNN_LIBS)
tests/libds4_qwen_image_graph_cpu.so: $(filter-out ds4_resources.o,$(SUITE_OBJS)) tests/suite_resources.o
	$(CC) -shared -o $@ $^ -lm -pthread
.PHONY: test-qwen-image-graph
test-qwen-image-graph: tests/libds4_qwen_image_graph_cpu.so
	$(SUITE_PYTHON) -m unittest tests.test_qwen_image_graph -v
ds4_numeric.o: ds4_numeric.c ds4_numeric_internal.h $(SUITE_API_HEADERS)
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
ds4_numeric_cuda.o: ds4_numeric_cuda.cu ds4_numeric_internal.h $(SUITE_API_HEADERS)
	$(NVCC) $(SUITE_NVCCFLAGS) -c -o $@ $<
ds4_numeric_cudnn.o: ds4_numeric_cudnn.cu ds4_numeric_internal.h $(SUITE_API_HEADERS)
	$(NVCC) $(SUITE_NVCCFLAGS) -I$(CUDNN_HOME)/include -c -o $@ $<
tests/libds4_numeric_cpu.so: ds4_numeric.c ds4_resources.c ds4_numeric.h ds4_numeric_internal.h ds4_resources.h
	$(CC) $(SUITE_CFLAGS) -shared -I. -o $@ ds4_numeric.c ds4_resources.c -lm -pthread
tests/libds4_qwen_image_tokenizer.so: ds4_qwen_image_tokenizer.c ds4_qwen_image_tokenizer.h ds4_sha256.h third_party/unicode/image_classes.inc ds4_unicode_nfc.h ds4_unicode_nfc.c third_party/unicode/nfc9.inc
	$(CC) $(SUITE_CFLAGS) -shared -I. -o $@ ds4_qwen_image_tokenizer.c ds4_unicode_nfc.c
tests/libds4_image_suite.so: ds4_image.c ds4_image.h ds4_image_resize.inc
	$(CC) $(CFLAGS) -fPIC -shared -I. -o $@ ds4_image.c -lm
.PHONY: test-qwen-image-tokenizer
test-qwen-image-tokenizer: tests/libds4_qwen_image_tokenizer.so
	$(SUITE_PYTHON) -m unittest tests.test_qwen_image_tokenizer -v
tests/suite_resources.o: ds4_resources.c ds4_resources.h ds4_runtime.h
	$(CC) $(SUITE_CFLAGS) -c -o $@ $<
tests/libds4_numeric_cuda.so: ds4_numeric.o ds4_numeric_cuda.o ds4_numeric_cudnn.o tests/suite_resources.o
	$(CC) -shared -o $@ $^ $(SUITE_CUDA_LDLIBS) $(SUITE_CUDNN_LIBS)
.PHONY: test-suite-numeric test-suite-numeric-cuda
test-suite-numeric: tests/libds4_numeric_cpu.so
	$(SUITE_PYTHON) -m unittest tests.test_suite_numeric -v
test-suite-numeric-cuda: tests/libds4_numeric_cuda.so
	DS4_SUITE_NUMERIC_CUDA=1 $(SUITE_PYTHON) -m unittest tests.test_suite_numeric -v

.PHONY: test-embedding-gemma2 test-qwen-image-reference
test-embedding-gemma2: tests/libds4_suite_cpu.so
	$(SUITE_PYTHON) -m unittest tests.test_embedding_gemma2 -v
test-qwen-image-reference: tests/libds4_suite_cpu.so
	$(SUITE_PYTHON) -m unittest tests.test_qwen_image_reference -v

.PHONY: test-suite-api
test-suite-api: tests/libds4_suite_cpu.so
	$(SUITE_PYTHON) -m unittest tests.test_suite_api -v

# Native suite resource and checkpoint gates run without a model.
ds4_resources.o: ds4_resources.c ds4_resources.h ds4_runtime.h
	$(CC) $(CFLAGS) -c -o $@ $<
ds4_executor.o: ds4_executor.c ds4_executor.h ds4_resources.h ds4_runtime.h
	$(CC) $(CFLAGS) -c -o $@ $<
ds4_bundle.o: ds4_bundle.c ds4_bundle.h ds4_resources.h ds4_runtime.h ds4_sha256.h ds4_linux_memory.h
	$(CC) $(CFLAGS) -c -o $@ $<
tests/test_bundle: tests/test_bundle.c ds4_bundle.o ds4_resources.o
	$(CC) $(CFLAGS) -I. -o $@ $^ $(LDLIBS)
.PHONY: test-bundle
test-bundle: tests/test_bundle
	python3 -m unittest tests.test_suite_bundle -v
test: test-bundle
tests/test_executor: tests/test_executor.c ds4_executor.o ds4_resources.o ds4_runtime.o
	$(CC) $(CFLAGS) -I. -o $@ $^ $(LDLIBS)
.PHONY: test-executor
test-executor: tests/test_executor
	./tests/test_executor
test: test-executor
ds4_checkpoint.o: ds4_checkpoint.c ds4_checkpoint.h ds4_sha256.h ds4_runtime.h
	$(CC) $(CFLAGS) -c -o $@ $<
tests/test_resources: tests/test_resources.c ds4_resources.o ds4_checkpoint.o
	$(CC) $(CFLAGS) -I. -o $@ $^ $(LDLIBS)
.PHONY: test-resources
test-resources: tests/test_resources
	./tests/test_resources
test: test-resources

tests/test_lifecycle_cuda.o: tests/test_lifecycle_cuda.c $(DS4_API_HEADERS) ds4_resources.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<
tests/test_lifecycle_cuda: tests/test_lifecycle_cuda.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
tests/test_executor_cuda.o: tests/test_executor_cuda.c $(DS4_API_HEADERS) ds4_executor.h ds4_resources.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<
tests/test_executor_cuda: tests/test_executor_cuda.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4_cuda.o tests/qwen_iq4_dequant_probe.o: ds4_cuda_accounting.cuh ds4_resources.h
ds4_image.o: ds4_sha256.h ds4_image_resize.inc
ds4.o ds4_cpu.o ds4_test_hooks.o ds4_cpu_test_hooks.o ds4_cuda_test_hooks.o \
 tests/test_session_state.o tests/test_session_state_gpu.o \
 tests/test_qwen_mtp_callback.o tests/test_qwen_snapshot_drafts.o tests/test_qwen_memory_lifetime.o \
 tests/qwen_mtp_catchup_probe.o tests/test_qwen4_ngrams.o tests/test_qwen4_ngram_state.o \
 tests/test_deepseek41_gguf.o tests/test_deepseek41_graph.o tests/test_deepseek41_prefill.o: \
 ds4_lifecycle.inc ds4_lifecycle_text.h ds4_resources.h ds4_checkpoint.h ds4_sha256.h ds4_runtime.h ds4_executor.h

ds4_cli.o ds4_server.o ds4_bench.o ds4_eval.o \
 ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o \
 tests/test_lifecycle_cuda.o tests/test_runtime_cuda.o: ds4_executor.h
