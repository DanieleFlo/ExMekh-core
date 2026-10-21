#include "ds4_qwen_image_tokenizer.h"
#include "ds4_sha256.h"
#include "ds4_unicode_nfc.h"
#include "third_party/unicode/image_classes.inc"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint32_t left, right, out, rank;
} merge;
typedef struct {
  uint32_t id, len;
  char *s;
} special;
struct ds4_qwen_image_tokenizer {
  uint8_t *file;
  uint32_t bytes[256], *prefix, *suffix, npre, nsuf, drop, mask, nspecial;
  merge *table;
  special *specials;
  size_t file_bytes;
};
uint64_t ds4_qwen_image_tokenizer_bytes(const ds4_qwen_image_tokenizer *t) {
  return t ? sizeof(*t) + t->file_bytes +
                 ((uint64_t)t->mask + 1) * sizeof(merge) +
                 (uint64_t)t->nspecial * sizeof(special)
           : 0;
}
typedef struct {
  uint32_t *v;
  size_t n, cap;
  bool limit;
} token_vec;
static bool push(token_vec *v, uint32_t id) {
  if (v->n == v->cap) {
    size_t cap = v->cap ? v->cap * 2 : 64;
    if (cap > 262144) {
      v->limit = true;
      return false;
    }
    void *p = realloc(v->v, cap * 4);
    if (!p)
      return false;
    v->v = p;
    v->cap = cap;
  }
  v->v[v->n++] = id;
  return true;
}
static uint32_t hash(uint32_t a, uint32_t b) {
  uint64_t x = ((uint64_t)a << 32) | b;
  x ^= x >> 33;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33;
  return x;
}
static uint32_t u32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}
static merge *find(const ds4_qwen_image_tokenizer *t, uint32_t a, uint32_t b) {
  uint32_t slot = hash(a, b) & t->mask;
  while (t->table[slot].rank) {
    merge *m = &t->table[slot];
    if (m->left == a && m->right == b)
      return m;
    slot = (slot + 1) & t->mask;
  }
  return NULL;
}
void ds4_qwen_image_tokenizer_close(ds4_qwen_image_tokenizer *t) {
  if (t) {
    free(t->table);
    free(t->specials);
    free(t->file);
    free(t);
  }
}
ds4_status ds4_qwen_image_tokenizer_open(ds4_qwen_image_tokenizer **out,
                                         const char *path,
                                         const uint8_t sha[32]) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!path || !sha)
    return DS4_STATUS_INVALID;
  FILE *f = fopen(path, "rb");
  if (!f)
    return DS4_STATUS_INVALID;
  if (fseek(f, 0, SEEK_END)) {
    fclose(f);
    return DS4_STATUS_INVALID;
  }
  long n = ftell(f);
  rewind(f);
  if (n < 1116 || n > 32 * 1024 * 1024) {
    fclose(f);
    return DS4_STATUS_INVALID;
  }
  ds4_qwen_image_tokenizer *t = calloc(1, sizeof(*t));
  if (!t) {
    fclose(f);
    return DS4_STATUS_NOMEM;
  }
  t->file = malloc(n);
  t->file_bytes = n;
  if (!t->file) {
    fclose(f);
    ds4_qwen_image_tokenizer_close(t);
    return DS4_STATUS_NOMEM;
  }
  bool read = fread(t->file, 1, n, f) == (size_t)n;
  if (fclose(f) || !read)
    goto invalid;
  uint8_t digest[32];
  ds4_sha256 checksum;
  ds4_sha256_init(&checksum);
  ds4_sha256_update(&checksum, t->file + 72, n - 72);
  ds4_sha256_final(&checksum, digest);
  if (memcmp(t->file, "DS4QTK1\0", 8) || memcmp(t->file + 8, sha, 32) ||
      memcmp(t->file + 40, digest, 32))
    goto invalid;
  uint8_t *p = t->file + 72, *end = t->file + n;
  uint32_t nm = u32(p);
  t->nspecial = u32(p + 4);
  t->npre = u32(p + 8);
  t->nsuf = u32(p + 12);
  t->drop = u32(p + 16);
  p += 20;
  if (nm > 1000000 || t->nspecial > 1024 || t->npre > 1024 || t->nsuf > 1024 ||
      t->drop > t->npre)
    goto invalid;
  for (int i = 0; i < 256; i++) {
    t->bytes[i] = u32(p);
    p += 4;
    if (t->bytes[i] >= 151936)
      goto invalid;
  }
  if ((uint64_t)(t->npre + t->nsuf) * 4 > (uint64_t)(end - p))
    goto invalid;
  /* Native host endian is checked here rather than aliasing unaligned data. */
  t->prefix = (uint32_t *)p;
  t->suffix = t->prefix + t->npre;
  for (uint32_t i = 0; i < t->npre + t->nsuf; i++) {
    uint32_t id = u32(p + 4 * i);
    if (id >= 151936)
      goto invalid;
    memcpy(p + 4 * i, &id, 4);
  }
  p += 4 * (t->npre + t->nsuf);
  if (t->nspecial) {
    t->specials = calloc(t->nspecial, sizeof(special));
    if (!t->specials)
      goto nomem;
  }
  for (uint32_t i = 0; i < t->nspecial; i++) {
    if (end - p < 8)
      goto invalid;
    special *s = &t->specials[i];
    s->id = u32(p);
    s->len = u32(p + 4);
    p += 8;
    if (s->id >= 151936 || !s->len || s->len > 4096 ||
        (uint64_t)(end - p) < s->len)
      goto invalid;
    s->s = (char *)p;
    p += s->len;
  }
  if ((uint64_t)nm * 12 != (uint64_t)(end - p))
    goto invalid;
  uint32_t cap = 1;
  while (cap < (uint64_t)nm * 2)
    cap *= 2;
  t->mask = cap - 1;
  t->table = calloc(cap, sizeof(merge));
  if (!t->table)
    goto nomem;
  for (uint32_t i = 0; i < nm; i++, p += 12) {
    uint32_t a = u32(p), b = u32(p + 4), c = u32(p + 8);
    if (a >= 151936 || b >= 151936 || c >= 151936 || find(t, a, b))
      goto invalid;
    uint32_t slot = hash(a, b) & t->mask;
    while (t->table[slot].rank)
      slot = (slot + 1) & t->mask;
    t->table[slot] = (merge){a, b, c, i + 1};
  }
  *out = t;
  return DS4_STATUS_OK;
nomem:
  ds4_qwen_image_tokenizer_close(t);
  return DS4_STATUS_NOMEM;
invalid:
  ds4_qwen_image_tokenizer_close(t);
  return DS4_STATUS_INVALID;
}
static bool urange(const ds4_image_urange *r, size_t n, uint32_t cp) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < r[mid].lo)
      hi = mid;
    else if (cp > r[mid].hi)
      lo = mid + 1;
    else
      return true;
  }
  return false;
}
#define CLASS(name, c)                                                         \
  urange(ds4_image_##name##_ranges,                                            \
         sizeof(ds4_image_##name##_ranges) /                                   \
             sizeof(ds4_image_##name##_ranges[0]),                             \
         c)
typedef struct {
  uint32_t cp;
  size_t next;
  bool letter, number, space;
} character;
static character at(const char *s, size_t n, size_t p) {
  character c = {0};
  if (p >= n) {
    c.next = p;
    return c;
  }
  uint8_t a = s[p++];
  uint32_t cp = a;
  int k = 0;
  if (a >= 0xf0) {
    cp = a & 7;
    k = 3;
  } else if (a >= 0xe0) {
    cp = a & 15;
    k = 2;
  } else if (a >= 0xc0) {
    cp = a & 31;
    k = 1;
  }
  while (k-- && p < n)
    cp = (cp << 6) | ((uint8_t)s[p++] & 63);
  c.cp = cp;
  c.next = p;
  c.letter = CLASS(letter, cp);
  c.number = CLASS(number, cp);
  c.space = CLASS(space, cp);
  return c;
}
/* Ordered Qwen2 regex, deliberately excluding combining marks from \p{L}.
 * Qwen35's otherwise similar splitter includes \p{M} and is not equivalent. */
static size_t piece_end(const char *s, size_t n, size_t pos) {
  character c = at(s, n, pos);
  size_t p = c.next;
  if (c.cp == '\'' && p < n) {
    const char *suffix[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (int i = 0; i < 7; i++) {
      size_t k = strlen(suffix[i]);
      if (p + k <= n) {
        size_t j = 0;
        while (j < k && tolower((unsigned char)s[p + j]) == suffix[i][j])
          j++;
        if (j == k)
          return p + k;
      }
    }
  }
  character next = at(s, n, p);
  if (c.letter ||
      (c.cp != '\r' && c.cp != '\n' && !c.number && p < n && next.letter)) {
    if (!c.letter)
      p = next.next;
    while (p < n) {
      character x = at(s, n, p);
      if (!x.letter)
        break;
      p = x.next;
    }
    return p;
  }
  if (c.number)
    return p;
  size_t pp = pos;
  character punct = c;
  if (c.cp == ' ') {
    pp = p;
    punct = next;
  }
  if (pp < n && !punct.space && !punct.letter && !punct.number) {
    p = punct.next;
    while (p < n) {
      character x = at(s, n, p);
      if (x.space || x.letter || x.number)
        break;
      p = x.next;
    }
    while (p < n && (s[p] == '\r' || s[p] == '\n'))
      p++;
    return p;
  }
  if (c.space) {
    p = pos;
    size_t newline = 0, last = pos, count = 0;
    while (p < n) {
      character x = at(s, n, p);
      if (!x.space)
        break;
      last = p;
      count++;
      p = x.next;
      if (x.cp == '\r' || x.cp == '\n')
        newline = p;
    }
    return newline ? newline : (count > 1 && p < n ? last : p);
  }
  return c.next;
}
static bool emit(const ds4_qwen_image_tokenizer *t, const char *s, size_t n,
                 token_vec *out) {
  uint32_t *ids = malloc(n * 4);
  if (!ids)
    return false;
  for (size_t i = 0; i < n; i++)
    ids[i] = t->bytes[(uint8_t)s[i]];
  size_t count = n;
  while (count > 1) {
    uint32_t best = UINT32_MAX, id = 0;
    size_t index = 0;
    for (size_t i = 0; i + 1 < count; i++) {
      merge *m = find(t, ids[i], ids[i + 1]);
      if (m && m->rank < best) {
        best = m->rank;
        id = m->out;
        index = i;
      }
    }
    if (best == UINT32_MAX)
      break;
    ids[index] = id;
    memmove(ids + index + 1, ids + index + 2, (count - index - 2) * 4);
    count--;
  }
  bool ok = true;
  for (size_t i = 0; i < count && ok; i++)
    ok = push(out, ids[i]);
  free(ids);
  return ok;
}
ds4_status ds4_qwen_image_tokenize(const ds4_qwen_image_tokenizer *t,
                                   const char *s, uint32_t **out,
                                   size_t *count) {
  if (out)
    *out = NULL;
  if (count)
    *count = 0;
  if (!t || !s || !out || !count)
    return DS4_STATUS_INVALID;
  char *normalized = NULL;
  ds4_status status = ds4_unicode_nfc(s, &normalized);
  if (status)
    return status;
  s = normalized;
  size_t n = strlen(s);
  if (n > 4 * 1024 * 1024) {
    free(normalized);
    return DS4_STATUS_INVALID;
  }
  token_vec v = {0};
  size_t pos = 0;
  while (pos < n) {
    special *best = NULL;
    for (uint32_t i = 0; i < t->nspecial; i++) {
      special *sp = &t->specials[i];
      if (sp->len <= n - pos && !memcmp(s + pos, sp->s, sp->len) &&
          (!best || sp->len > best->len))
        best = sp;
    }
    if (best) {
      if (!push(&v, best->id))
        goto fail;
      pos += best->len;
      continue;
    }
    size_t end = piece_end(s, n, pos);
    /* Added tokens split before pre-tokenization, including inside punctuation
     * runs. */
    for (size_t p = pos + 1; p < end; p++)
      for (uint32_t j = 0; j < t->nspecial; j++) {
        special *sp = &t->specials[j];
        if (sp->len <= n - p && !memcmp(s + p, sp->s, sp->len)) {
          end = p;
          break;
        }
      }
    if (!emit(t, s + pos, end - pos, &v))
      goto fail;
    pos = end;
  }
  *out = v.v;
  *count = v.n;
  free(normalized);
  return DS4_STATUS_OK;
fail:
  free(v.v);
  free(normalized);
  return v.limit ? DS4_STATUS_INVALID : DS4_STATUS_NOMEM;
}
ds4_status ds4_qwen_image_prompt(const ds4_qwen_image_tokenizer *t,
                                 const char *prompt, const uint32_t *images,
                                 size_t count, uint32_t **out, size_t *n,
                                 size_t *drop) {
  if (out)
    *out = NULL;
  if (n)
    *n = 0;
  if (drop)
    *drop = 0;
  if (!t || !prompt || !out || !n || !drop || (count && !images) ||
      count > 1024)
    return DS4_STATUS_INVALID;
  token_vec v = {0};
  for (uint32_t i = 0; i < t->npre; i++)
    if (!push(&v, t->prefix[i]))
      goto fail;
  for (size_t i = 0; i < count; i++) {
    if (!images[i]) {
      free(v.v);
      return DS4_STATUS_INVALID;
    }
    char tag[128];
    snprintf(tag, sizeof(tag), "%s<image%zu><|vision_start|>", i ? " " : "",
             i + 1);
    uint32_t *ids = NULL;
    size_t ni = 0;
    ds4_status status = ds4_qwen_image_tokenize(t, tag, &ids, &ni);
    if (status) {
      free(v.v);
      return status;
    }
    for (size_t j = 0; j < ni; j++)
      if (!push(&v, ids[j])) {
        free(ids);
        goto fail;
      }
    free(ids);
    for (uint32_t j = 0; j < images[i]; j++)
      if (!push(&v, 151655))
        goto fail;
    if (!push(&v, 151653))
      goto fail;
  }
  uint32_t *ids = NULL;
  size_t ni = 0;
  ds4_status status = ds4_qwen_image_tokenize(t, prompt, &ids, &ni);
  if (status) {
    free(v.v);
    return status;
  }
  if (!*prompt && !push(&v, t->bytes[32]))
    goto fail;
  for (size_t i = 0; i < ni; i++)
    if (!push(&v, ids[i])) {
      free(ids);
      goto fail;
    }
  free(ids);
  for (uint32_t i = 0; i < t->nsuf; i++)
    if (!push(&v, t->suffix[i]))
      goto fail;
  *out = v.v;
  *n = v.n;
  *drop = t->drop;
  return DS4_STATUS_OK;
fail:
  free(v.v);
  return v.limit ? DS4_STATUS_INVALID : DS4_STATUS_NOMEM;
}
