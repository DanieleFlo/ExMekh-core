#ifndef DS4_QWEN4_RUNTIME_H
#define DS4_QWEN4_RUNTIME_H
#include <stdint.h>

/* Placement work belongs to the complete forward, including short prefill
 * tails and single-row drafts. AUTO is only for tensor-only callers. */
typedef enum {
    DS4_QWEN4_WORKLOAD_AUTO = -1,
    DS4_QWEN4_WORKLOAD_DECODE = 0,
    DS4_QWEN4_WORKLOAD_MTP,
    DS4_QWEN4_WORKLOAD_PREFILL,
    DS4_QWEN4_WORKLOAD_COUNT,
} ds4_qwen4_workload;

typedef struct {
    uint64_t gate, up, down;
    uint32_t gu_type, down_type, experts, slots, embd, ff;
} ds4_gpu_qwen4_expert_layout;
typedef struct {
    uint64_t weight_hits, cpu_pairs, gpu_pairs, pcie_pairs;
    uint64_t ssd_bytes, ssd_reads, expert_h2d_bytes, kv_h2d_bytes, kv_hits, kv_misses;
    uint64_t host_weight_bytes, kv_host_bytes, staging_bytes, expert_vram_bytes;
    uint64_t kv_resident_bytes, kv_workspace_bytes, hybrid_workspace_bytes;
    uint64_t hybrid_host_bytes;
    uint64_t cpu_h2d_bytes, activation_d2h_bytes;
    uint64_t cpu_stage_jobs;
    double cpu_wait_seconds;
    double staging_wait_seconds, victim_select_seconds;
} ds4_gpu_qwen4_tier_stats;
#endif
