#include "ds4_unicode_nfc.h"
#include "third_party/unicode/nfc9.inc"
#include <stdlib.h>
#include <string.h>
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
static uint32_t combining_class(uint32_t cp) {
  size_t lo = 0, hi = COUNT(nfc_classes);
  while (lo < hi) {
    size_t m = (lo + hi) / 2;
    uint32_t code = nfc_classes[m] >> 8;
    if (code < cp)
      lo = m + 1;
    else
      hi = m;
  }
  return lo < COUNT(nfc_classes) && (nfc_classes[lo] >> 8) == cp
             ? nfc_classes[lo] & 255
             : 0;
}
static const nfc_decomposition *decomposition(uint32_t cp) {
  size_t lo = 0, hi = COUNT(nfc_decompositions);
  while (lo < hi) {
    size_t m = (lo + hi) / 2;
    if (nfc_decompositions[m].cp < cp)
      lo = m + 1;
    else
      hi = m;
  }
  return lo < COUNT(nfc_decompositions) && nfc_decompositions[lo].cp == cp
             ? &nfc_decompositions[lo]
             : NULL;
}
static uint32_t composition(uint32_t a, uint32_t b) {
  if (a >= 0x1100 && a < 0x1113 && b >= 0x1161 && b < 0x1176)
    return 0xac00 + ((a - 0x1100) * 21 + b - 0x1161) * 28;
  if (a >= 0xac00 && a < 0xd7a4 && (a - 0xac00) % 28 == 0 && b > 0x11a7 &&
      b < 0x11c3)
    return a + b - 0x11a7;
  uint64_t key = (uint64_t)a << 21 | b;
  size_t lo = 0, hi = COUNT(nfc_compositions);
  while (lo < hi) {
    size_t m = (lo + hi) / 2;
    if (nfc_compositions[m].key < key)
      lo = m + 1;
    else
      hi = m;
  }
  return lo < COUNT(nfc_compositions) && nfc_compositions[lo].key == key
             ? nfc_compositions[lo].value
             : 0;
}
ds4_status ds4_unicode_nfc(const char *text, char **out) {
  if (!text || !out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  size_t bytes = strlen(text);
  if (bytes > 4u * 1024u * 1024u)
    return DS4_STATUS_INVALID;
  uint32_t *points = malloc((bytes * 4 + 3) * sizeof(uint32_t));
  if (!points)
    return DS4_STATUS_NOMEM;
  size_t count = 0, offset = 0;
  ds4_status status = DS4_STATUS_INVALID;
  while (offset < bytes) {
    uint8_t first = (uint8_t)text[offset++];
    uint32_t cp = first;
    int extra = 0;
    uint32_t minimum = 0;
    if (first >= 0xf0 && first < 0xf5) {
      cp = first & 7;
      extra = 3;
      minimum = 0x10000;
    } else if (first >= 0xe0 && first < 0xf0) {
      cp = first & 15;
      extra = 2;
      minimum = 0x800;
    } else if (first >= 0xc2 && first < 0xe0) {
      cp = first & 31;
      extra = 1;
      minimum = 0x80;
    } else if (first >= 0x80)
      goto cleanup;
    if (bytes - offset < (size_t)extra)
      goto cleanup;
    for (int j = 0; j < extra; j++) {
      uint8_t next = (uint8_t)text[offset++];
      if ((next & 192) != 128)
        goto cleanup;
      cp = (cp << 6) | (next & 63);
    }
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
      goto cleanup;
    if (cp >= 0xac00 && cp < 0xd7a4) {
      uint32_t index = cp - 0xac00;
      points[count++] = 0x1100 + index / 588;
      points[count++] = 0x1161 + (index % 588) / 28;
      if (index % 28)
        points[count++] = 0x11a7 + index % 28;
    } else {
      const nfc_decomposition *d = decomposition(cp);
      if (d) {
        memcpy(points + count, nfc_decomposed + d->offset, d->length * 4);
        count += d->length;
      } else
        points[count++] = cp;
    }
  }
  /* Stable canonical ordering. Counting sort only the non-monotone runs,
   * avoiding quadratic behavior on long adversarial combining sequences. */
  for (size_t start = 0; start < count;) {
    if (!combining_class(points[start])) {
      start++;
      continue;
    }
    size_t end = start + 1;
    bool ordered = true;
    uint32_t last = combining_class(points[start]);
    while (end < count) {
      uint32_t c = combining_class(points[end]);
      if (!c)
        break;
      if (c < last)
        ordered = false;
      last = c;
      end++;
    }
    if (!ordered) {
      size_t positions[256] = {0};
      for (size_t i = start; i < end; i++)
        positions[combining_class(points[i])]++;
      size_t sum = 0;
      for (size_t c = 0; c < 256; c++) {
        size_t n = positions[c];
        positions[c] = sum;
        sum += n;
      }
      uint32_t *sorted = malloc((end - start) * 4);
      if (!sorted) {
        status = DS4_STATUS_NOMEM;
        goto cleanup;
      }
      for (size_t i = start; i < end; i++)
        sorted[positions[combining_class(points[i])]++] = points[i];
      memcpy(points + start, sorted, (end - start) * 4);
      free(sorted);
    }
    start = end;
  }
  {
    size_t length = 0, starter = SIZE_MAX;
    uint32_t last = 0;
    for (size_t i = 0; i < count; i++) {
      uint32_t cp = points[i], ccc = combining_class(cp);
      uint32_t joined = starter != SIZE_MAX && (!last || last < ccc)
                            ? composition(points[starter], cp)
                            : 0;
      if (joined)
        points[starter] = joined;
      else {
        if (!ccc)
          starter = length;
        points[length++] = cp;
        last = ccc;
      }
    }
    char *result = malloc(length * 4 + 1);
    if (!result) {
      status = DS4_STATUS_NOMEM;
      goto cleanup;
    }
    size_t n = 0;
    for (size_t i = 0; i < length; i++) {
      uint32_t cp = points[i];
      if (cp < 128)
        result[n++] = cp;
      else if (cp < 0x800) {
        result[n++] = 0xc0 | (cp >> 6);
        result[n++] = 0x80 | (cp & 63);
      } else if (cp < 0x10000) {
        result[n++] = 0xe0 | (cp >> 12);
        result[n++] = 0x80 | ((cp >> 6) & 63);
        result[n++] = 0x80 | (cp & 63);
      } else {
        result[n++] = 0xf0 | (cp >> 18);
        result[n++] = 0x80 | ((cp >> 12) & 63);
        result[n++] = 0x80 | ((cp >> 6) & 63);
        result[n++] = 0x80 | (cp & 63);
      }
    }
    result[n] = 0;
    *out = result;
    status = DS4_STATUS_OK;
  }
cleanup:
  free(points);
  return status;
}
