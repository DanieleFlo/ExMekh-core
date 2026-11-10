#ifndef DS4_NUMERIC_METAL_ARGS_H
#define DS4_NUMERIC_METAL_ARGS_H

/* Eight BF16 values per vector, sixteen vectors per moment chunk. */
#define DS4_NM_BF16_LAYERNORM_LEVELS 16
#define DS4_NM_BF16_LAYERNORM_MAX_WIDTH \
    (8u * 16u * (1u << DS4_NM_BF16_LAYERNORM_LEVELS))
#ifndef __METAL_VERSION__
#include <stdint.h>
typedef uint64_t ds4_nm_size;
typedef uint32_t ds4_nm_uint;
#else
typedef ulong ds4_nm_size;
typedef uint ds4_nm_uint;
#endif
/* One fixed ABI for suite primitives. n[0] is the output extent; storage types
 * follow buffer bindings. Geometry is private to each kernel/host pair. */
typedef struct {
    ds4_nm_size n[4];
    ds4_nm_uint t[8], u[32];
    float f[8];
    ds4_nm_size shape[8], stride[8];
} ds4_nm_args;
#endif
