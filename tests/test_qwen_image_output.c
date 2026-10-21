#include "ds4_qwen_image_observer.h"
#include "ds4_qwen_image_rgba_internal.h"
#include "third_party/iris/png.h"
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
  bool cancel;
  unsigned calls;
} observer_state;
static bool cancelled(void *opaque) {
  return ((observer_state *)opaque)->cancel;
}
static ds4_status trace(void *opaque, const char *stage, uint32_t layer,
                        ds4_num_view v) {
  (void)stage;
  (void)layer;
  (void)v;
  observer_state *state = opaque;
  state->calls++;
  state->cancel = true;
  return DS4_STATUS_OK;
}
int main(void) {
  ds4_resources *resources = NULL;
  ds4_numeric *num = NULL;
  assert(!ds4_resources_create(&resources));
  assert(!ds4_numeric_cpu(&num, resources, 913));
  observer_state state = {0};
  ds4_qwen_image_observer observer = {&state, cancelled, trace};
  assert(ds4_qwen_image_observe(num, &observer, "ready", 0,
                                (ds4_num_view){0}) == DS4_STATUS_CANCELLED);
  assert(state.calls == 1);
  assert(ds4_qwen_image_observe(num, &observer, "ready", 0,
                                (ds4_num_view){0}) == DS4_STATUS_CANCELLED);
  assert(state.calls == 1);
  assert(!ds4_numeric_close(num));

  uint64_t bytes[DS4_MEMORY_COUNT] = {0};
  bytes[DS4_MEMORY_HOST] = 1024 * 1024;
  ds4_reservation *lease = NULL;
  assert(!ds4_resources_reserve(resources, 913, bytes, &lease));
  assert(!ds4_reservation_use(lease, 29));
  ds4_reservation_scope scope = ds4_reservation_enter(lease);
  ds4_qwen_image_rgba *image = NULL;
  float *pixels = NULL;
  assert(!ds4_qwen_image_rgba_create(resources, 913, 2, 1, &image, &pixels));
  const float expected[] = {1, 0, 0, 0, 0, 0.5f, 1, 1};
  for (unsigned i = 0; i < 8; i++)
    pixels[i] = expected[i];
  ds4_qwen_image_rgba_view view = ds4_qwen_image_rgba_data(image);
  assert(view.width == 2 && view.height == 1 && view.pixels[3] == 0 &&
         view.pixels[7] == 1);
  char path[] = "/tmp/ds4-rgba-XXXXXX";
  int fd = mkstemp(path);
  assert(fd >= 0);
  close(fd);
  assert(!ds4_qwen_image_rgba_save_png(image, path));
  png_image *png = png_load(path);
  assert(png && png->channels == 4 && png->width == 2 && png->height == 1);
  const unsigned char rgba[] = {255, 0, 0, 0, 0, 128, 255, 255};
  for (unsigned i = 0; i < 8; i++)
    assert(png->data[i] == rgba[i]);
  png_free(png);
  unlink(path);
  assert(ds4_qwen_image_rgba_save_png(image, "/dev/full") ==
         DS4_STATUS_INVALID);
  ds4_qwen_image_rgba_free(image);
  ds4_resource_totals totals;
  ds4_resources_totals(resources, 0, &totals);
  assert(!totals.roots &&
         totals.reserved[DS4_MEMORY_HOST] == bytes[DS4_MEMORY_HOST]);
  ds4_reservation_leave(&scope);
  ds4_resources_complete(resources, 29);
  assert(!ds4_reservation_cancel(lease));
  ds4_resource_budget denied = {.limit = 1};
  assert(!ds4_resources_set_budget(resources, DS4_MEMORY_HOST, &denied));
  image = (void *)1;
  pixels = (void *)1;
  assert(ds4_qwen_image_rgba_create(resources, 913, 2, 1, &image, &pixels) ==
         DS4_STATUS_NOMEM);
  assert(!image && !pixels);
  image = (void *)1;
  assert(ds4_qwen_image_rgba_create(resources, 913, 0, 1, &image, &pixels) ==
             DS4_STATUS_INVALID &&
         !image);
  assert(!ds4_resources_close(resources));
  puts("RGBA host lifetime, PNG alpha/errors, reservation accounting and "
       "observer cancellation: PASS");
}
