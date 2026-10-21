#include "ds4_qwen_image_observer.h"
ds4_status ds4_qwen_image_observe(ds4_numeric *num,
                                  const ds4_qwen_image_observer *observer,
                                  const char *stage, uint32_t layer,
                                  ds4_num_view view) {
  if (observer && observer->cancelled && observer->cancelled(observer->opaque))
    return DS4_STATUS_CANCELLED;
  if (observer && observer->trace) {
    ds4_status status = ds4_numeric_fence(num);
    if (status)
      return status;
    status = observer->trace(observer->opaque, stage, layer, view);
    if (status)
      return status;
  }
  return observer && observer->cancelled &&
                 observer->cancelled(observer->opaque)
             ? DS4_STATUS_CANCELLED
             : DS4_STATUS_OK;
}
