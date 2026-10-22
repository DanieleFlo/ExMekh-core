#include "ds4_image.h"
#include "ds4_qwen_image_rgba_internal.h"
#include "third_party/iris/png.h"
#include <math.h>
#include <stdlib.h>
struct ds4_qwen_image_rgba {
  uint32_t width, height;
  ds4_resources *resources;
  uint64_t owner;
  ds4_resource *account;
  float pixels[];
};
uint64_t ds4_qwen_image_rgba_bytes(uint32_t width, uint32_t height) {
  uint64_t count = (uint64_t)width * height;
  if (!width || !height || width > DS4_IMAGE_MAX_DIMENSION ||
      height > DS4_IMAGE_MAX_DIMENSION || count > DS4_IMAGE_MAX_PIXELS)
    return 0;
  return sizeof(ds4_qwen_image_rgba) + count * 4 * sizeof(float);
}
ds4_status ds4_qwen_image_rgba_create(ds4_resources *resources, uint64_t owner,
                                      uint32_t width, uint32_t height,
                                      ds4_qwen_image_rgba **out,
                                      float **pixels) {
  if (out)
    *out = NULL;
  if (pixels)
    *pixels = NULL;
  uint64_t bytes = ds4_qwen_image_rgba_bytes(width, height);
  if (!out || !pixels || !bytes || bytes > SIZE_MAX)
    return DS4_STATUS_INVALID;
  ds4_resource *account = NULL;
  ds4_status status =
      ds4_resource_register_host(resources, owner, bytes, &account);
  if (status)
    return status;
  ds4_qwen_image_rgba *image = calloc(1, bytes);
  if (!image) {
    ds4_resource_release(account);
    return DS4_STATUS_NOMEM;
  }
  image->width = width;
  image->height = height;
  image->resources = resources;
  image->owner = owner;
  image->account = account;
  *pixels = image->pixels;
  *out = image;
  return DS4_STATUS_OK;
}
ds4_qwen_image_rgba_view
ds4_qwen_image_rgba_data(const ds4_qwen_image_rgba *image) {
  return image ? (ds4_qwen_image_rgba_view){image->width, image->height,
                                            image->pixels}
               : (ds4_qwen_image_rgba_view){0};
}
void ds4_qwen_image_rgba_free(ds4_qwen_image_rgba *image) {
  if (image) {
    ds4_resource_release(image->account);
    free(image);
  }
}
ds4_status ds4_qwen_image_rgba_opaque(ds4_qwen_image_rgba *image) {
  if (!image) return DS4_STATUS_INVALID;
  for (uint64_t i = 0; i < (uint64_t)image->width * image->height; i++)
    image->pixels[4 * i + 3] = 1;
  return DS4_STATUS_OK;
}
ds4_status ds4_qwen_image_rgba_save_png(const ds4_qwen_image_rgba *image,
                                        const char *path) {
  if (!image || !path)
    return DS4_STATUS_INVALID;
  /* The codec's uint8/raw/deflate copies are all covered before allocation. */
  uint64_t bytes[DS4_MEMORY_COUNT] = {0};
  bytes[DS4_MEMORY_HOST] =
      4 * ((uint64_t)image->width * image->height * 4 + image->height) + 65536;
  ds4_reservation *reservation = NULL;
  ds4_status status = ds4_resources_reserve(image->resources, image->owner,
                                            bytes, &reservation);
  if (status)
    return status;
  png_image *png = png_create(image->width, image->height, 4);
  if (!png) {
    ds4_reservation_cancel(reservation);
    return DS4_STATUS_NOMEM;
  }
  for (uint64_t i = 0; i < (uint64_t)image->width * image->height * 4; i++)
    png->data[i] =
        (uint8_t)lrintf(fminf(1, fmaxf(0, image->pixels[i])) * 255.f);
  int saved = png_save(png, path);
  png_free(png);
  ds4_reservation_cancel(reservation);
  return saved ? DS4_STATUS_INVALID : DS4_STATUS_OK;
}
