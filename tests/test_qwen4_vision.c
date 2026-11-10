/* Dump the Qwen3.8 vision tower output for one image; tests/qwen4_vision_ref.py
 * compares it against the HF vision model */
#include "ds4.h"
#include "ds4_image.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Export the actual tower input so an independent CPU implementation can
 * separate encoder errors from decoder/resize differences (notably JPEG). */
static int dump_patches(const char *image_path, const char *out_path, uint32_t min_tokens,
                        uint32_t max_tokens) {
    ds4_image image = {0};
    ds4_image_patches patches = {0};
    char error[256] = {0};
    int ok =
        ds4_image_decode_file(&image, image_path, error, sizeof(error)) &&
        ds4_image_preprocess_qwen4(&patches, &image, min_tokens, max_tokens, error, sizeof(error));
    FILE *fp = ok ? fopen(out_path, "wb") : NULL;
    if (fp) {
        const uint32_t header[4] = {patches.patch_count, 768, patches.grid_height,
                                    patches.grid_width};
        const size_t count = (size_t)patches.patch_count * 768;
        ok = fwrite(header, sizeof(header), 1, fp) == 1 &&
             fwrite(patches.patches, sizeof(float), count, fp) == count;
        if (fclose(fp))
            ok = 0;
    } else
        ok = 0;
    if (!ok)
        fprintf(stderr, "vision patch dump failed: %s\n", error);
    ds4_image_patches_free(&patches);
    ds4_image_free(&image);
    return ok ? 0 : 1;
}

static int parse_tokens(const char *value, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(value, &end, 10);
    if (errno || end == value || *end || parsed == 0 || parsed > 16383u)
        return 0;
    *out = (uint32_t)parsed;
    return 1;
}

int main(int argc, char **argv) {
    uint32_t min_tokens = 64u, max_tokens = 1024u;
    if (argc < 4 || argc > 6 || (argc > 4 && !parse_tokens(argv[4], &min_tokens)) ||
        (argc > 5 && !parse_tokens(argv[5], &max_tokens)) || min_tokens > max_tokens) {
        fprintf(stderr,
                "usage: %s mmproj.gguf|--patches image out.bin [min_tokens max_tokens]\n"
                "token limits must satisfy 1 <= min <= max <= 16383\n",
                argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "--patches"))
        return dump_patches(argv[2], argv[3], min_tokens, max_tokens);
    return ds4_qwen4_vision_dump(argv[1], argv[2], argv[3], min_tokens, max_tokens) ? 0 : 1;
}
