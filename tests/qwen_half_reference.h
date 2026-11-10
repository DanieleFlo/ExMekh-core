#ifndef QWEN_HALF_REFERENCE_H
#define QWEN_HALF_REFERENCE_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Independent scalar storage decoder for the Qwen quantization oracles.
 * A zero exponent includes subnormals, not only signed zero. */
static inline float qwen_reference_f16_to_f32(uint16_t half) {
    const uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    const uint32_t exponent = (half >> 10) & 0x1fu;
    const uint32_t fraction = half & 0x3ffu;
    if (exponent == 0)
        return copysignf(ldexpf((float)fraction, -24), sign ? -1.0f : 1.0f);
    uint32_t bits = exponent == 31 ? sign | 0x7f800000u | (fraction << 13)
                                   : sign | ((exponent + 112u) << 23) | (fraction << 13);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}
#endif
