#ifndef DS4_CHECKPOINT_H
#define DS4_CHECKPOINT_H
#include "ds4_runtime.h"
#include <stdio.h>

/* Private lifecycle envelope. The legacy family payload ABI stays unchanged.
 * All integers are little endian. Validation scans the entire file before
 * the caller allocates or mutates a graph. The caller owns the open FILE. */
#define DS4_CHECKPOINT_HEADER_BYTES 128u
ds4_status ds4_checkpoint_seal(FILE *, const uint8_t identity[32],
        const uint8_t config[32], uint64_t payload_bytes, const void *metadata, size_t metadata_bytes);
ds4_status ds4_checkpoint_validate(FILE *, const uint8_t identity[32],
        const uint8_t config[32], uint64_t *payload_bytes, void *metadata, size_t *metadata_bytes);
#endif
