#include "ds4_image.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int hex_fingerprint(const uint8_t *fp, char *out, size_t cap) {
    if (cap < 65) return 0;
    for (int i = 0; i < 32; i++)
        snprintf(out + i * 2, cap - (size_t)i * 2, "%02x", fp[i]);
    return 1;
}

static int check_jpeg(const char *path, uint32_t width, uint32_t height,
                      const char *expected_fp) {
    ds4_image image = {0};
    char error[160] = {0};
    if (!ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "decode failed for %s: %s\n", path, error);
        return 0;
    }
    char got[65] = {0};
    int ok = hex_fingerprint(image.fingerprint, got, sizeof(got)) &&
             image.width == width && image.height == height &&
             strcmp(got, expected_fp) == 0;
    if (!ok) {
        fprintf(stderr, "%s: got %ux%u fp=%s, expected %ux%u fp=%s\n",
                path, image.width, image.height, got,
                width, height, expected_fp);
    }
    ds4_image_free(&image);
    return ok;
}

static int check_qwen_processor(void) {
    uint8_t rgb[64 * 80 * 3];
    for (unsigned y = 0; y < 64; y++)
        for (unsigned x = 0; x < 80; x++)
            for (unsigned c = 0; c < 3; c++)
                rgb[(y * 80 + x) * 3 + c] = (13 * x + 7 * y + 31 * c) % 256;
    ds4_image image = {.width = 80, .height = 64, .rgb = rgb};
    ds4_image_patches normalized = {0}, raw = {0};
    char error[160] = {0};
    int ok = ds4_image_preprocess_qwen4(&normalized, &image, 1, 128, error, sizeof(error)) &&
        ds4_image_preprocess_qwen3vl_raw(&raw, &image, 1, 128, error, sizeof(error));
    /* Python round(80/32) uses ties-to-even: both APIs must select width 64.
     * Pixel values frozen from Pillow's uint8 bicubic, before normalization. */
    ok = ok && normalized.content_width == 64 && normalized.content_height == 64 &&
        raw.content_width == 64 && raw.content_height == 64 &&
        normalized.patch_count == raw.patch_count;
    const unsigned indices[] = {0, 256, 512, 15, 256 + 7 * 16 + 11, 512 + 15 * 16 + 15};
    const float expected[] = {2, 33, 64, 238, 51, 156};
    for (unsigned i = 0; ok && i < sizeof(indices) / sizeof(indices[0]); i++)
        ok = raw.patches[indices[i]] == expected[i];
    for (size_t i = 0; ok && i < (size_t)raw.patch_count * 768; i++) {
        float value = (raw.patches[i] / 255.0f - 0.5f) / 0.5f;
        ok = normalized.patches[i] >= -1.0000002f && normalized.patches[i] <= 1.0000002f &&
            fabsf(normalized.patches[i] - value) < 2e-7f;
    }
    if (!ok) fprintf(stderr, "Qwen normalized/raw processor mismatch: %s\n", error);
    ds4_image_patches_free(&normalized); ds4_image_patches_free(&raw);
    return ok;
}

int main(void) {
    if (!check_qwen_processor()) return 1;
    /* 24x16 grayscale progressive JPEG. Unpatched Iris yields different
     * pixels; patched decode matches libjpeg-turbo djpeg bit-exactly. */
    if (!check_jpeg("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg",
                    24u, 16u,
                    "63aae8863e829170c5abc746c90a5ff3ad60e9c4312e2ffa5eefbfeaacd26bfc"))
        return 1;
    /* 64x48 4:2:0 progressive JPEG. Unpatched jpeg_load returns NULL.
     * After the ZRL fix plus main's chroma interpolation, decode matches
     * libjpeg-turbo djpeg bit-exactly. */
    if (!check_jpeg("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_420.jpg",
                    64u, 48u,
                    "d3be4d7078c41b6589942c82bd622ca8a3ed40adddee11cccf1de9ca1a096ba4"))
        return 1;
    return 0;
}
