/* Small independent tokenizer asset: no model weights or GPU are needed to
 * verify prepared-input ownership, padding, validation and admission. */
#include "ds4_numeric_weights.h"
#include "ds4_qwen_image_input_internal.h"
#include "ds4_sha256.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void put32(uint8_t **out, uint32_t value) {
  for (unsigned i = 0; i < 4; i++)
    *(*out)++ = (uint8_t)(value >> (8 * i));
}
static void checksum(const void *data, size_t size, uint8_t out[32]) {
  ds4_sha256 state;
  ds4_sha256_init(&state);
  ds4_sha256_update(&state, data, size);
  ds4_sha256_final(&state, out);
}
static void write_tokenizer(char path[], uint8_t source[32]) {
  uint8_t content[4096] = {0}, *p = content + 72;
  memcpy(content, "DS4QTK1\0", 8);
  checksum("fixture", 7, source);
  memcpy(content + 8, source, 32);
  put32(&p, 1);
  put32(&p, 1);
  put32(&p, 14);
  put32(&p, 1);
  put32(&p, 14);
  for (unsigned i = 0; i < 256; i++)
    put32(&p, i);
  for (unsigned i = 0; i < 14; i++)
    put32(&p, 'S');
  put32(&p, 'E');
  const char *special = "<|image_pad|>";
  put32(&p, 151655);
  put32(&p, strlen(special));
  memcpy(p, special, strlen(special));
  p += strlen(special);
  put32(&p, 'a');
  put32(&p, 'b');
  put32(&p, 256);
  checksum(content + 72, p - content - 72, content + 40);
  int fd = mkstemp(path);
  assert(fd >= 0);
  assert(write(fd, content, p - content) == p - content);
  assert(!close(fd));
}
static void same_totals(ds4_resources *r, const ds4_resource_totals *before) {
  ds4_resource_totals after;
  ds4_resources_totals(r, 0, &after);
  assert(before->roots == after.roots && before->views == after.views);
  for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++) {
    assert(before->allocated[i] == after.allocated[i]);
    assert(before->reserved[i] == after.reserved[i]);
  }
}
int main(void) {
  char path[] = "/tmp/ds4-image-input-XXXXXX";
  uint8_t source[32];
  write_tokenizer(path, source);
  ds4_qwen_image_tokenizer *tokenizer = NULL;
  ds4_resources *r = NULL;
  ds4_numeric *num = NULL;
  assert(!ds4_qwen_image_tokenizer_open(&tokenizer, path, source));
  assert(!unlink(path));
  assert(!ds4_resources_create(&r));
  assert(!ds4_numeric_cpu(&num, r, 771));
  ds4_resource_totals baseline;
  ds4_resources_totals(r, 0, &baseline);
  ds4_qwen_image_input *input = NULL;
  char prompt[] = "ab";
  assert(!ds4_qwen_image_input_create(num, tokenizer, prompt, NULL, 0, 64,
                                      &input));
  ds4_qwen_image_input_view view = ds4_qwen_image_input_data(input);
  assert(view.count == 64 && view.drop == 62 && view.left_padding == 48);
  assert(view.tokens[62] == 256 && view.tokens[63] == 'E');
  for (unsigned i = 0; i < 64; i++)
    assert(view.positions[i * 3] == i);
  prompt[0] = 'x';
  assert(!strcmp(view.prompt, "ab"));
  ds4_qwen_image_input_free(input);
  input = NULL;
  same_totals(r, &baseline);
  assert(ds4_qwen_image_input_create(num, tokenizer, "<|image_pad|>", NULL, 0,
                                     0, &input) == DS4_STATUS_INVALID);
  assert(!input);
  same_totals(r, &baseline);
  assert(ds4_qwen_image_input_create(num, tokenizer, "ab", NULL, 0, 262145,
                                     &input) == DS4_STATUS_INVALID);
  assert(!input);
  uint8_t rgba[8 * 8 * 4] = {0};
  ds4_qwen_image_reference reference = {rgba, 8, 8, 4};
  assert(!ds4_qwen_image_input_create(num, tokenizer, "ab", &reference, 1, 128,
                                      &input));
  view = ds4_qwen_image_input_data(input);
  assert(view.images == 1 && view.patches == 256 && view.left_padding > 0);
  for (unsigned i = 0; i < view.left_padding; i++)
    assert(view.positions[i * 3] == 0);
  for (size_t i = 0; i < (size_t)view.patches * 1536; i++)
    assert(view.pixels[i] == 1.f);
  for (size_t i = 0; i < sizeof(rgba); i++)
    assert(rgba[i] == 0);
  rgba[0] = 255;
  assert(view.pixels[0] == 1.f);
  ds4_qwen_image_input_free(input);
  input = NULL;
  same_totals(r, &baseline);
  ds4_resource_budget original, denied;
  assert(!ds4_resources_get_budget(r, DS4_MEMORY_HOST, &original));
  denied = original;
  denied.limit = baseline.allocated[DS4_MEMORY_HOST] + 1;
  assert(!ds4_resources_set_budget(r, DS4_MEMORY_HOST, &denied));
  assert(ds4_qwen_image_input_create(num, tokenizer, "ab", NULL, 0, 0,
                                     &input) == DS4_STATUS_NOMEM);
  assert(!input);
  assert(!ds4_resources_set_budget(r, DS4_MEMORY_HOST, &original));
  same_totals(r, &baseline);
  /* Group storage delegates byte writes/reinterpretation to its numeric owner.
   */
  ds4_weight_backend backend = ds4_numeric_weight_backend(num);
  void *storage = NULL;
  assert(!backend.allocate(backend.context, 9, &storage));
  uint8_t byte = 73, readback = 0;
  assert(!backend.write(backend.context, storage, 9, &byte, 1));
  assert(backend.write(backend.context, storage, 10, &byte, 1) ==
         DS4_STATUS_INVALID);
  ds4_num_view words = ds4_num_data(storage);
  ds4_num_view tail = ds4_num_slice(words, 4, 1);
  uint8_t pair[2];
  assert(!ds4_num_read(num, tail, pair, 2));
  readback = pair[1];
  assert(readback == 73);
  assert(!ds4_numeric_weight_view(storage, 1, 1, DS4_NUM_F32).data);
  assert(ds4_numeric_weight_view(storage, 4, 1, DS4_NUM_F32).data);
  assert(!backend.release(backend.context, storage));
  same_totals(r, &baseline);
  ds4_qwen_image_tokenizer_close(tokenizer);
  assert(!ds4_numeric_close(num));
  assert(!ds4_resources_close(r));
  puts("prepared input ownership, padding, validation, admission and weight "
       "bridge: PASS");
  return 0;
}
