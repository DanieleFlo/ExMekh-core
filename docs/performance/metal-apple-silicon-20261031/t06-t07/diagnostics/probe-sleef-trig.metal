// SLEEF xsinf/xcosf u35 small-range branch, same pinned Boost license as numeric_ordered.metal.
inline float nm_sleef_sin(float value) {
#pragma clang fp contract(off)
    if (!(abs(value) < 125.0f))
        return sin(value);
    if ((as_type<uint>(value) & 0x7fffffff) < 0x00800000)
        return value;
    int q = int(rint(value * 0.31830988618379067154f));
    float d = fma(float(q), -3.1414794921875f, value);
    d = fma(float(q), -0.00011315941810607910156f, d);
    d = fma(float(q), -1.9841872589410058936e-09f, d);
    float s = d * d;
    d = as_type<float>(as_type<uint>(d) ^ ((q & 1) ? 0x80000000u : 0u));
    float u = 2.6083159809786593541503e-06f;
    u = fma(u, s, -0.0001981069071916863322258f);
    u = fma(u, s, 0.00833307858556509017944336f);
    u = fma(u, s, -0.166666597127914428710938f);
    return s * (u * d) + d;
}
inline float nm_sleef_cos(float value) {
#pragma clang fp contract(off)
    if (!(abs(value) < 125.0f))
        return cos(value);
    int q = 2 * int(rint(value * 0.31830988618379067154f - 0.5f)) + 1;
    float d = fma(float(q), -3.1414794921875f * 0.5f, value);
    d = fma(float(q), -0.00011315941810607910156f * 0.5f, d);
    d = fma(float(q), -1.9841872589410058936e-09f * 0.5f, d);
    float s = d * d;
    d = as_type<float>(as_type<uint>(d) ^ ((q & 2) == 0 ? 0x80000000u : 0u));
    float u = 2.6083159809786593541503e-06f;
    u = fma(u, s, -0.0001981069071916863322258f);
    u = fma(u, s, 0.00833307858556509017944336f);
    u = fma(u, s, -0.166666597127914428710938f);
    return s * (u * d) + d;
}
