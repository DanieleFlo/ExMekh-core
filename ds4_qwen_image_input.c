/* Owns preprocessing, immutable prepared input and its host lifetime. */
#include "ds4_image.h"
#include "ds4_qwen_image_input_internal.h"
#include <stdlib.h>
#include <string.h>
struct ds4_qwen_image_input {
  uint32_t *tokens, *positions; /* positions: [tokens,3] T,H,W */
  uint8_t *image_mask;
  uint32_t count, drop, images, patches;
  uint32_t left_padding; /* excluded before the 14 system tokens */
  uint32_t *grids;       /* [images,2] H,W, before spatial merge */
  float *pixels;         /* [patches,3,2,16,16], temporal image repeated */
  char *prompt;
  ds4_resource *account; /* private ownership record; do not modify */
};
#define TRY(expr)                                                              \
  do {                                                                         \
    status = (expr);                                                           \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
ds4_qwen_image_input_view
ds4_qwen_image_input_data(const ds4_qwen_image_input *in) {
  if (!in)
    return (ds4_qwen_image_input_view){0};
  return (ds4_qwen_image_input_view){
      in->tokens, in->positions, in->image_mask, in->count,
      in->drop,   in->images,    in->patches,    in->left_padding,
      in->grids,  in->pixels,    in->prompt};
}
void ds4_qwen_image_input_free(ds4_qwen_image_input *i) {
  if (i) {
    free(i->tokens);
    free(i->positions);
    free(i->image_mask);
    free(i->grids);
    free(i->pixels);
    free(i->prompt);
    if (i->account)
      ds4_resource_release(i->account);
    free(i);
  }
}
static ds4_status validate_input(const ds4_qwen_image_input *in) {
  if (!in || !in->tokens || !in->positions || !in->image_mask || !in->count ||
      in->count > 262144 || in->drop >= in->count ||
      in->left_padding > in->drop || in->drop - in->left_padding != 14 ||
      in->images > 1024 || (in->images && (!in->grids || !in->pixels)))
    return DS4_STATUS_INVALID;
  uint64_t patches = 0;
  for (uint32_t i = 0; i < in->images; i++) {
    uint32_t h = in->grids[2 * i], w = in->grids[2 * i + 1];
    if (h < 2 || w < 2 || h % 2 || w % 2)
      return DS4_STATUS_INVALID;
    patches += (uint64_t)h * w;
  }
  if (patches != in->patches || patches > in->count * 4ull)
    return DS4_STATUS_INVALID;
  uint32_t mask = 0;
  for (uint32_t i = 0; i < in->count; i++) {
    if (i < in->left_padding && (in->tokens[i] != 151643 || in->image_mask[i]))
      return DS4_STATUS_INVALID;
    if (in->tokens[i] >= 151936 || in->image_mask[i] > 1 ||
        in->image_mask[i] != (in->tokens[i] == 151655))
      return DS4_STATUS_INVALID;
    mask += in->image_mask[i];
  }
  return (uint64_t)mask * 4 == patches ? DS4_STATUS_OK : DS4_STATUS_INVALID;
}
static ds4_status prepare_reference(const ds4_qwen_image_reference *ref,
                                    ds4_image_patches *patches) {
  size_t pixels = (size_t)ref->width * ref->height;
  uint8_t *rgb = malloc(pixels * 3);
  if (!rgb) {
    return DS4_STATUS_NOMEM;
  }
  for (size_t p = 0; p < pixels; p++)
    for (size_t c = 0; c < 3; c++) {
      uint32_t a = ref->channels == 4 ? ref->pixels[p * 4 + 3] : 255;
      rgb[p * 3 + c] =
          (ref->pixels[p * ref->channels + c] * a + 255 * (255 - a) + 127) /
          255;
    }
  ds4_image image = {ref->width, ref->height, rgb, {0}};
  char error[128];
  int ok = ds4_image_preprocess_qwen3vl_raw(patches, &image, 64, 16384, error,
                                            sizeof(error));
  free(rgb);
  if (!ok) {
    return DS4_STATUS_INVALID;
  }
  return DS4_STATUS_OK;
}
static ds4_status prepare_positions(ds4_qwen_image_input *in) {
  /* The pinned model falls back to physical arange positions for text-only
   * inputs. Multimodal get_rope_index instead excludes left padding. BF16
   * RoPE is not numerically invariant to translating all positions. */
  uint32_t pos = in->images ? 0 : in->left_padding, image = 0;
  for (uint32_t t = 0; t < in->left_padding; t++)
    for (int d = 0; d < 3; d++)
      in->positions[3 * t + d] = in->images ? 0 : t;
  for (uint32_t t = in->left_padding; t < in->count;) {
    if (in->tokens[t] != 151655) {
      for (int d = 0; d < 3; d++)
        in->positions[3 * t + d] = pos;
      pos++;
      t++;
      continue;
    }
    if (image >= in->images) {
      return DS4_STATUS_INVALID;
    }
    uint32_t h = in->grids[2 * image] / 2, w = in->grids[2 * image + 1] / 2;
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++) {
        if (t >= in->count || in->tokens[t] != 151655) {
          return DS4_STATUS_INVALID;
        }
        in->positions[3 * t] = pos;
        in->positions[3 * t + 1] = pos + y;
        in->positions[3 * t + 2] = pos + x;
        in->image_mask[t++] = 1;
      }
    image++;
    pos += h > w ? h : w;
  }
  if (image != in->images) {
    return DS4_STATUS_INVALID;
  }

  return DS4_STATUS_OK;
}
static ds4_status prepare_pixels(ds4_qwen_image_input *in,
                                 const ds4_image_patches *patches) {
  if (in->patches) {
    in->pixels = malloc((size_t)in->patches * 1536 * 4);
    if (!in->pixels) {
      return DS4_STATUS_NOMEM;
    }
    size_t dest = 0;
    for (size_t i = 0; i < in->images; i++)
      for (uint32_t p = 0; p < patches[i].patch_count; p++)
        for (uint32_t c = 0; c < 3; c++)
          for (uint32_t time = 0; time < 2; time++) {
            for (uint32_t pixel = 0; pixel < 256; pixel++)
              in->pixels[dest + pixel] =
                  (patches[i].patches[(size_t)p * 768 + c * 256 + pixel] -
                   127.5f) /
                  127.5f;
            dest += 256;
          }
  }
  return DS4_STATUS_OK;
}

ds4_status ds4_qwen_image_input_create(
    ds4_numeric *num, const ds4_qwen_image_tokenizer *tokenizer,
    const char *prompt, const ds4_qwen_image_reference *refs, size_t count,
    uint32_t pad_to, ds4_qwen_image_input **out) {
  uint64_t owner = 0;
  ds4_resources *resources = ds4_numeric_resources(num, &owner);
  if (!resources || !tokenizer || !prompt || !out || (count && !refs) ||
      count > 1024)
    return DS4_STATUS_INVALID;
  *out = NULL;
  size_t prompt_bytes = strlen(prompt);
  if (prompt_bytes > 4u * 1024u * 1024u || pad_to > 262144)
    return DS4_STATUS_INVALID;
  uint64_t planned_patches = 0, temporary_peak = 0;
  for (size_t i = 0; i < count; i++) {
    const ds4_qwen_image_reference *ref = &refs[i];
    uint32_t width, height;
    if (!ref->pixels || ref->width > DS4_IMAGE_MAX_DIMENSION ||
        ref->height > DS4_IMAGE_MAX_DIMENSION ||
        (uint64_t)ref->width * ref->height > DS4_IMAGE_MAX_PIXELS ||
        (ref->channels != 3 && ref->channels != 4) ||
        !ds4_image_qwen3vl_plan(ref->width, ref->height, 64, 16384, &width,
                                &height))
      return DS4_STATUS_INVALID;
    uint64_t patches = (uint64_t)width * height / 256;
    planned_patches += patches;
    if (planned_patches > 262144ull * 4)
      return DS4_STATUS_INVALID;
    uint64_t temporary =
        (uint64_t)ref->width * ref->height * 6 + patches * 4096 + 1024u * 1024u;
    if (temporary > temporary_peak)
      temporary_peak = temporary;
  }
  /* Admission covers raw RGB, both resize axes, spatial/temporal patches,
   * tokenization/NFC and final input storage before any large host allocation.
   * Only the persistent input becomes allocated; the remaining lease is
   * returned after preprocessing, rather than reported as fictitious RSS. */
  uint64_t budget[DS4_MEMORY_COUNT] = {0};
  budget[DS4_MEMORY_HOST] =
      sizeof(ds4_qwen_image_input) + prompt_bytes * 64 +
      planned_patches * 9216 + temporary_peak + 8u * 1024u * 1024u +
      count * (sizeof(ds4_image_patches) + 12) + (uint64_t)pad_to * 17;
  ds4_reservation *lease = NULL;
  ds4_status status = ds4_resources_reserve(resources, owner, budget, &lease);
  if (status)
    return status;
  ds4_reservation_scope reservation_scope = ds4_reservation_enter(lease);
  ds4_qwen_image_input *in = calloc(1, sizeof(*in));
  if (!in) {
    ds4_reservation_leave(&reservation_scope);
    ds4_reservation_cancel(lease);
    return DS4_STATUS_NOMEM;
  }
  in->images = count;
  in->prompt = strdup(prompt);
  ds4_image_patches *patches = calloc(count ? count : 1, sizeof(*patches));
  uint32_t *image_tokens = calloc(count ? count : 1, 4);
  status = DS4_STATUS_NOMEM;
  in->grids = calloc(count ? count * 2 : 1, 4);
  if (!patches || !image_tokens || !in->grids || !in->prompt)
    goto cleanup;
  for (size_t i = 0; i < count; i++) {
    uint32_t width, height;
    if (!ds4_image_qwen3vl_plan(refs[i].width, refs[i].height, 64, 16384,
                                &width, &height)) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    in->grids[2 * i] = height / 16;
    in->grids[2 * i + 1] = width / 16;
    image_tokens[i] = (width / 32) * (height / 32);
  }
  {
    size_t n = 0, drop = 0;
    TRY(ds4_qwen_image_prompt(tokenizer, prompt, image_tokens, count,
                              &in->tokens, &n, &drop));
    in->left_padding = pad_to > n ? pad_to - n : 0;
    in->count = n + in->left_padding;
    in->drop = drop + in->left_padding;
    uint32_t *exact = malloc((size_t)in->count * 4);
    if (!exact) {
      status = DS4_STATUS_NOMEM;
      goto cleanup;
    }
    for (uint32_t i = 0; i < in->left_padding; i++)
      exact[i] = 151643;
    memcpy(exact + in->left_padding, in->tokens, n * 4);
    free(in->tokens);
    in->tokens = exact;
  }
  for (size_t i = 0; i < count; i++) {
    const ds4_qwen_image_reference *ref = &refs[i];
    TRY(prepare_reference(ref, &patches[i]));
    in->grids[2 * i] = patches[i].grid_height;
    in->grids[2 * i + 1] = patches[i].grid_width;
    image_tokens[i] = patches[i].image_token_count;
    if (in->patches > UINT32_MAX - patches[i].patch_count) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    in->patches += patches[i].patch_count;
  }
  in->positions = calloc((size_t)in->count * 3, 4);
  in->image_mask = calloc(in->count, 1);
  if (!in->positions || !in->image_mask) {
    status = DS4_STATUS_NOMEM;
    goto cleanup;
  }
  TRY(prepare_positions(in));
  TRY(prepare_pixels(in, patches));
  TRY(validate_input(in));
  ds4_resource_spec spec = {.owner = owner,
                            .location = DS4_MEMORY_HOST,
                            .device = -1,
                            .requested = sizeof(*in) + prompt_bytes + 1 +
                                         (count ? count * 2 : 1) * 4 +
                                         (uint64_t)in->count * 17 +
                                         (uint64_t)in->patches * 1536 * 4};
  spec.allocated = spec.live = spec.requested;
  TRY(ds4_resource_register(resources, &spec, &in->account));
  *out = in;
  in = NULL;
  status = DS4_STATUS_OK;
cleanup:
  if (patches) {
    for (size_t i = 0; i < count; i++)
      ds4_image_patches_free(&patches[i]);
  }
  free(patches);
  free(image_tokens);
  ds4_qwen_image_input_free(in);
  ds4_reservation_leave(&reservation_scope);
  ds4_reservation_cancel(lease);
  return status;
}
