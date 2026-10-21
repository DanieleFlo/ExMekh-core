#include "ds4_gemma_tokenizer.h"
#include "ds4_sha256.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint32_t a, b, out, rank;
} pair;
typedef struct {
  uint32_t id, len;
  const char *text;
} special;
struct gemma_tokenizer {
  uint8_t *file;
  size_t size;
  gemma_config config;
  const uint8_t *chars;
  uint32_t nchars, mask, bytes[256];
  pair *pairs;
  special *specials;
  uint32_t nspecial;
  char prefixes[4][128];
};
static uint32_t read32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}
static uint32_t hash(uint32_t a, uint32_t b) {
  uint64_t x = ((uint64_t)a << 32) | b;
  x ^= x >> 33;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33;
  return (uint32_t)x;
}
static pair *lookup(const gemma_tokenizer *t, uint32_t a, uint32_t b) {
  uint32_t i = hash(a, b) & t->mask;
  while (t->pairs[i].rank) {
    if (t->pairs[i].a == a && t->pairs[i].b == b)
      return t->pairs + i;
    i = (i + 1) & t->mask;
  }
  return NULL;
}
void gemma_tokenizer_close(gemma_tokenizer *t) {
  if (t) {
    free(t->file);
    free(t->pairs);
    free(t->specials);
    free(t);
  }
}
uint64_t gemma_tokenizer_bytes(const gemma_tokenizer *t) {
  return t ? sizeof(*t) + t->size + ((uint64_t)t->mask + 1) * sizeof(pair) +
                 t->nspecial * sizeof(special)
           : 0;
}
const gemma_config *gemma_tokenizer_config(const gemma_tokenizer *t) {
  return t ? &t->config : NULL;
}
ds4_status gemma_tokenizer_open(gemma_tokenizer **out, const char *path,
                                const uint8_t hashes[96]) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!path || !hashes)
    return DS4_STATUS_INVALID;
  FILE *f = fopen(path, "rb");
  if (!f)
    return DS4_STATUS_INVALID;
  if (fseek(f, 0, SEEK_END)) {
    fclose(f);
    return DS4_STATUS_INVALID;
  }
  long size = ftell(f);
  rewind(f);
  if (size < 136 + 20 || size > 32 * 1024 * 1024) {
    fclose(f);
    return DS4_STATUS_INVALID;
  }
  gemma_tokenizer *t = calloc(1, sizeof(*t));
  if (!t) {
    fclose(f);
    return DS4_STATUS_NOMEM;
  }
  t->size = size;
  t->file = malloc(size);
  if (!t->file) {
    fclose(f);
    gemma_tokenizer_close(t);
    return DS4_STATUS_NOMEM;
  }
  bool ok = fread(t->file, 1, size, f) == (size_t)size;
  if (fclose(f) || !ok)
    goto invalid;
  uint8_t digest[32];
  ds4_sha256 sha;
  ds4_sha256_init(&sha);
  ds4_sha256_update(&sha, t->file + 136, size - 136);
  ds4_sha256_final(&sha, digest);
  if (memcmp(t->file, "DS4EGM1\0", 8) || memcmp(t->file + 8, hashes, 96) ||
      memcmp(t->file + 104, digest, 32))
    goto invalid;
  uint8_t *p = t->file + 136, *end = t->file + size;
  t->nchars = read32(p);
  uint32_t nm = read32(p + 4);
  t->nspecial = read32(p + 8);
  t->config.window = read32(p + 12);
  memcpy(&t->config.eps, p + 16, 4);
  p += 20;
  if (t->nchars > 262144 || nm > 1048576 || t->nspecial > 8192 ||
      t->config.window != 512 || t->config.eps != 1e-6f)
    goto invalid;
  for (uint32_t i = 0; i < 24; i++) {
    if (end - p < 16)
      goto invalid;
    gemma_layer *l = t->config.layers + i;
    l->dim = read32(p);
    l->kv = read32(p + 4);
    l->local = read32(p + 8);
    l->heads = read32(p + 12);
    p += 16;
    if (l->heads != 4 || l->local > 1 || l->dim != (l->local ? 256u : 512u) ||
        l->kv != (l->local ? 2u : 1u) || end - p < l->dim * 2)
      goto invalid;
    memcpy(l->frequency, p, l->dim * 2);
    p += l->dim * 2;
  }
  for (uint32_t i = 0; i < 4; i++) {
    if (end - p < 4)
      goto invalid;
    uint32_t n = read32(p);
    p += 4;
    if (!n || n >= 128 || (uint64_t)(end - p) < n || memchr(p, 0, n))
      goto invalid;
    memcpy(t->prefixes[i], p, n);
    p += n;
  }
  if ((uint64_t)(end - p) < 1024 + (uint64_t)t->nchars * 8)
    goto invalid;
  for (uint32_t i = 0; i < 256; i++) {
    t->bytes[i] = read32(p);
    p += 4;
    if (t->bytes[i] >= 262144)
      goto invalid;
  }
  t->chars = p;
  p += t->nchars * 8;
  for (uint32_t i = 0; i < t->nchars; i++)
    if (read32(t->chars + 8 * i + 4) >= 262144 ||
        (i && read32(t->chars + 8 * (i - 1)) >= read32(t->chars + 8 * i)))
      goto invalid;
  t->specials = calloc(t->nspecial, sizeof(special));
  if (!t->specials)
    goto nomem;
  for (uint32_t i = 0; i < t->nspecial; i++) {
    if (end - p < 8)
      goto invalid;
    special *s = t->specials + i;
    s->id = read32(p);
    s->len = read32(p + 4);
    p += 8;
    if (s->id >= 262144 || !s->len || s->len > 4096 ||
        (uint64_t)(end - p) < s->len)
      goto invalid;
    s->text = (char *)p;
    p += s->len;
  }
  if ((uint64_t)(end - p) != (uint64_t)nm * 12)
    goto invalid;
  uint32_t cap = 1;
  while (cap < (uint64_t)nm * 2)
    cap *= 2;
  t->mask = cap - 1;
  t->pairs = calloc(cap, sizeof(pair));
  if (!t->pairs)
    goto nomem;
  for (uint32_t i = 0; i < nm; i++, p += 12) {
    uint32_t a = read32(p), b = read32(p + 4), c = read32(p + 8);
    if (a >= 262144 || b >= 262144 || c >= 262144)
      goto invalid;
    if (lookup(t, a, b))
      continue; /* First occurrence defines BPE rank. */
    uint32_t j = hash(a, b) & t->mask;
    while (t->pairs[j].rank)
      j = (j + 1) & t->mask;
    t->pairs[j] = (pair){a, b, c, i + 1};
  }
  *out = t;
  return DS4_STATUS_OK;
nomem:
  gemma_tokenizer_close(t);
  return DS4_STATUS_NOMEM;
invalid:
  gemma_tokenizer_close(t);
  return DS4_STATUS_INVALID;
}
typedef struct {
  uint32_t id, prev, next, version;
} node;
typedef struct {
  uint32_t pos, next, version, rank, out;
} edge;
typedef struct {
  edge *v;
  size_t n;
} heap;
static bool less(edge a, edge b) {
  return a.rank < b.rank || (a.rank == b.rank && a.pos < b.pos);
}
static void push(heap *h, edge e) {
  size_t i = h->n++;
  while (i && less(e, h->v[(i - 1) / 2])) {
    h->v[i] = h->v[(i - 1) / 2];
    i = (i - 1) / 2;
  }
  h->v[i] = e;
}
static edge pop(heap *h) {
  edge r = h->v[0], e = h->v[--h->n];
  size_t i = 0;
  while (i * 2 + 1 < h->n) {
    size_t j = i * 2 + 1;
    if (j + 1 < h->n && less(h->v[j + 1], h->v[j]))
      j++;
    if (!less(h->v[j], e))
      break;
    h->v[i] = h->v[j];
    i = j;
  }
  h->v[i] = e;
  return r;
}
static void offer(const gemma_tokenizer *t, node *nodes, uint32_t i, heap *h) {
  if (i == UINT32_MAX || nodes[i].next == UINT32_MAX)
    return;
  uint32_t j = nodes[i].next;
  pair *p = lookup(t, nodes[i].id, nodes[j].id);
  if (p)
    push(h, (edge){i, j, nodes[i].version, p->rank, p->out});
}
static uint32_t character(const gemma_tokenizer *t, uint32_t cp) {
  uint32_t lo = 0, hi = t->nchars;
  while (lo < hi) {
    uint32_t mid = (lo + hi) / 2, v = read32(t->chars + 8 * mid);
    if (v < cp)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo < t->nchars && read32(t->chars + 8 * lo) == cp
             ? read32(t->chars + 8 * lo + 4)
             : UINT32_MAX;
}
static ds4_status segment(const gemma_tokenizer *t, const uint8_t *s,
                          size_t bytes, uint32_t *result, uint32_t *count) {
  if (!bytes)
    return DS4_STATUS_OK;
  node *a = calloc(bytes, sizeof(node));
  edge *edges = malloc((3 * bytes + 1) * sizeof(edge));
  if (!a || !edges) {
    free(a);
    free(edges);
    return DS4_STATUS_NOMEM;
  }
  ds4_status status = DS4_STATUS_OK;
  uint32_t n = 0;
  for (size_t i = 0; i < bytes;) {
    size_t begin = i;
    uint32_t cp = s[i++], k = 0;
    if (cp >= 0xc2 && cp <= 0xdf) {
      cp &= 31;
      k = 1;
    } else if (cp >= 0xe0 && cp <= 0xef) {
      cp &= 15;
      k = 2;
    } else if (cp >= 0xf0 && cp <= 0xf4) {
      cp &= 7;
      k = 3;
    } else if (cp >= 128) {
      status = DS4_STATUS_INVALID;
      goto done;
    }
    for (uint32_t j = 0; j < k; j++) {
      if (i == bytes || (s[i] & 0xc0) != 0x80) {
        status = DS4_STATUS_INVALID;
        goto done;
      }
      cp = (cp << 6) | (s[i++] & 63);
    }
    if ((k == 1 && cp < 128) || (k == 2 && cp < 2048) ||
        (k == 3 && cp < 65536) || cp > 0x10ffff ||
        (cp >= 0xd800 && cp <= 0xdfff)) {
      status = DS4_STATUS_INVALID;
      goto done;
    }
    if (cp == 32)
      cp = 0x2581;
    uint32_t id = character(t, cp);
    if (id != UINT32_MAX)
      a[n++].id = id;
    else
      for (size_t j = begin; j < i; j++)
        a[n++].id = t->bytes[s[j]];
  }
  heap h = {edges, 0};
  for (uint32_t i = 0; i < n; i++) {
    a[i].prev = i ? i - 1 : UINT32_MAX;
    a[i].next = i + 1 < n ? i + 1 : UINT32_MAX;
  }
  for (uint32_t i = 0; i < n; i++)
    offer(t, a, i, &h);
  while (h.n) {
    edge e = pop(&h);
    uint32_t i = e.pos, j = e.next;
    if (a[i].version != e.version || a[i].next != j)
      continue;
    pair *current = lookup(t, a[i].id, a[j].id);
    if (!current || current->rank != e.rank)
      continue;
    a[i].id = e.out;
    a[i].next = a[j].next;
    a[i].version++;
    a[j].version++;
    if (a[i].next != UINT32_MAX)
      a[a[i].next].prev = i;
    offer(t, a, a[i].prev, &h);
    offer(t, a, i, &h);
  }
  for (uint32_t i = 0; i != UINT32_MAX; i = a[i].next) {
    if (*count >= 8191) {
      status = DS4_STATUS_INVALID;
      goto done;
    }
    result[(*count)++] = a[i].id;
  }
done:
  free(a);
  free(edges);
  return status;
}
ds4_status gemma_tokenize(const gemma_tokenizer *t, const char *text,
                          uint32_t task, uint32_t **out, uint32_t *count) {
  if (out)
    *out = NULL;
  if (count)
    *count = 0;
  if (!t || !text || !out || !count || task > DS4_GEMMA_SIMILARITY)
    return DS4_STATUS_INVALID;
  size_t n = strnlen(text, 1024 * 1024 + 1);
  if (n > 1024 * 1024)
    return DS4_STATUS_INVALID;
  const char *prefix = task ? t->prefixes[task - 1] : "";
  /* A caller-supplied task/title is already formatted. Never apply it twice. */
  if (!strncmp(text, "task: ", 6) ||
      (!strncmp(text, "title: ", 7) && strstr(text, " | text: ")))
    prefix = "";
  size_t pn = strlen(prefix);
  char *input = malloc(n + pn + 1);
  uint32_t *ids = malloc(8192 * 4);
  if (!input || !ids) {
    free(input);
    free(ids);
    return DS4_STATUS_NOMEM;
  }
  memcpy(input, prefix, pn);
  memcpy(input + pn, text, n + 1);
  n += pn;
  uint32_t used = 1;
  ids[0] = 2;
  ds4_status status = DS4_STATUS_OK;
  size_t start = 0;
  for (size_t i = 0; i < n;) {
    special *best = NULL;
    for (uint32_t j = 0; j < t->nspecial; j++) {
      special *s = t->specials + j;
      if (s->len <= n - i && (!best || s->len > best->len) &&
          !memcmp(input + i, s->text, s->len))
        best = s;
    }
    if (!best) {
      i++;
      continue;
    }
    status = segment(t, (uint8_t *)input + start, i - start, ids, &used);
    if (status)
      goto done;
    if (used >= 8191) {
      status = DS4_STATUS_INVALID;
      goto done;
    }
    ids[used++] = best->id;
    i += best->len;
    start = i;
  }
  status = segment(t, (uint8_t *)input + start, n - start, ids, &used);
  if (status)
    goto done;
  ids[used++] = 1;
  *out = ids;
  *count = used;
  ids = NULL;
done:
  free(input);
  free(ids);
  return status;
}
