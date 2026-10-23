/* BF16 values captured from the frozen official QwenImage21Pipeline call:
 * t.expand(batch).to(latents.dtype) / 1000, before the time embedding. */
#include "ds4_qwen_image_graph.h"
#include <assert.h>
int main(void) {
  const float sigma[] = {1.f, .978453f, .95418f, .926626f, .89508f,
                         .845148f, .704534f, .414568f, 0};
  const float expected[] = {1.f, .98046875f, .95703125f, .9296875f,
                            .89453125f, .84375f, .703125f, .4140625f, 0};
  for (unsigned i = 0; i < sizeof(sigma) / sizeof(*sigma); i++)
    assert(qi_image_timestep(sigma[i]) == expected[i]);
  puts("Qwen Image pipeline BF16 timestep rounding: PASS");
}
