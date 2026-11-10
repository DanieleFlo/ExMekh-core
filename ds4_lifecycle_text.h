#ifndef DS4_LIFECYCLE_TEXT_H
#define DS4_LIFECYCLE_TEXT_H

#define DS4_QWEN_VISION_BACKEND "ds4: Qwen3.8-27B vision requires CUDA or Metal\n"
#define DS4_QWEN_GRAPH_ALLOCATION_FAILED "ds4: %s graph allocation failed for context %u\n"
#define DS4_METAL_MEMORY_UNKNOWN "ds4: cannot determine the Metal memory budget\n"
#define DS4_METAL_MEMORY_REFUSED "ds4: Metal memory admission refused: model/state/runtime %.2f GiB, available budget %.2f GiB (%.2f GiB host reserve)\n"
#define DS4_METAL_STREAM_BUDGET "ds4: Metal SSD cache budget %.2f GiB (%llu bytes, %s); graph/aux/runtime %.2f GiB, host reserve %.2f GiB\n"
#define DS4_METAL_STREAM_BUDGET_AUTO "automatic"
#define DS4_METAL_STREAM_BUDGET_CAPPED "explicit ceiling"
#define DS4_METAL_STREAM_REBALANCED "ds4: Metal SSD cache adjusted %.2f -> %.2f GiB; future state/runtime %.2f GiB, host reserve %.2f GiB\n"
#define DS4_METAL_STREAM_CAPACITY_LOW "insufficient available RAM for Metal SSD inference"
#define DS4_FLASH_BACKEND_CONFIGURATION "ds4: Qwen Flash requires Metal or single-GPU CUDA (or --cpu --first-token-test); SSD streaming requires Metal. Tensor parallelism, pipeline execution, DSpark, external MTP models and power throttling are not supported\n"
#define DS4_FLASH_STREAM_CONFIGURATION "ds4: Flash Metal SSD streaming uses a byte-bounded layer cache; expert-count, preload and full-resident-layer options are not supported\n"
#define DS4_LIFECYCLE_RESTORE "unable to restore session; checkpoint retained"
#define DS4_LIFECYCLE_CHECKPOINT "unable to checkpoint session; live state retained"
#define DS4_LIFECYCLE_NOT_RESIDENT "session state or model weights are not resident"
#define DS4_LIFECYCLE_RESTORE_FAILED "ds4: checkpoint restore failed phase=%s status=%d ctx=%d%s%s\n"
#define DS4_LIFECYCLE_VISION_RESUME_FAILED "ds4: vision session resume failed status=%d ctx=%d tokens=%d\n"
#endif
