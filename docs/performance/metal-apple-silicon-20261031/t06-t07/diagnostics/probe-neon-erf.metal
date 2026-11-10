// FP32 expression of pinned ATen NEON erf and its Arm exp_u20 polynomial.
// PyTorch d0f2af8, vec128_float_neon.h. See third_party/pytorch/LICENSE.
inline float nm_neon_exp_u20(float x) {
#pragma clang fp contract(off)
    if (abs(x) > 0x1.5d5e2ap+6f)
        return nm_sleef_exp(x);
    float exponent = round(x * 0x1.715476p+0f);
    float r = fma(-exponent, 0x1.62e4p-1f, x);
    r = fma(-exponent, 0x1.7f7d1cp-20f, r);
    float scale = as_type<float>((uint(int(exponent)) << 23) + 0x3f800000u);
    float square = r * r;
    float p = fma(r, 0x1.0e4020p-7f, 0x1.573e2ep-5f);
    float q = fma(r, 0x1.555e66p-3f, 0x1.fffdb6p-2f);
    q = fma(p, square, q);
    p = 0x1.ffffecp-1f * r;
    float polynomial = fma(q, square, p);
    return fma(polynomial, scale, scale);
}
inline float nm_bf16_gelu_erf(float value) {
#pragma clang fp contract(off)
    float x = value * 0.70710678118654752440f;
    float t = 1.0f / fma(0.3275911f, abs(x), 1.0f);
    float r = fma(1.061405429f, t, -1.453152027f);
    r = fma(r, t, 1.421413741f);
    r = fma(r, t, -0.284496736f);
    r = fma(r, t, 0.254829592f);
    float e = nm_neon_exp_u20(-(x * x));
    float result = fma(t * -e, r, 1.0f);
    result = as_type<float>(as_type<uint>(result) ^ (as_type<uint>(x) & 0x80000000u));
    return value * 0.5f * (1.0f + result);
}
