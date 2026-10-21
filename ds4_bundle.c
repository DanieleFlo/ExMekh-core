#define _POSIX_C_SOURCE 200809L
#include "ds4_bundle.h"
#include "ds4_sha256.h"
#include <fcntl.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include "ds4_linux_memory.h"
#endif

typedef struct { int fd; void *map; ds4_resource *account; struct stat identity; } mapping;
struct ds4_bundle {
    char *model, *revision, *profile;
    ds4_bundle_file *files;
    ds4_bundle_component *components;
    ds4_weight_tensor *tensors;
    mapping *maps;
    size_t file_count, component_count, tensor_count, groups;
    ds4_resource *metadata;
    ds4_resources *resources;
    uint64_t owner;
};
struct ds4_weight_group {
    ds4_bundle *bundle;
    uint32_t component, group;
    uint64_t bytes;
    uint64_t *offsets;
    ds4_weight_backend backend;
    ds4_resource *metadata;
    void *storage;
    bool resident;
};
typedef struct { const uint8_t *data; size_t size, pos; ds4_status status; } reader;
static uint64_t integer(reader *r, unsigned bytes) {
    if (r->status) return 0;
    if (r->pos > r->size || bytes > r->size - r->pos) { r->status = DS4_STATUS_INVALID; return 0; }
    uint64_t n = 0;
    for (unsigned i = 0; i < bytes; i++) n |= (uint64_t)r->data[r->pos++] << (8u*i);
    return n;
}
static char *string(reader *r) {
    uint64_t n = integer(r, 4);
    if (r->status) return NULL;
    if (!n || n > 4096 || n > r->size - r->pos || memchr(r->data + r->pos, 0, (size_t)n)) {
        r->status = DS4_STATUS_INVALID; return NULL;
    }
    char *s = malloc((size_t)n + 1);
    if (!s) { r->status = DS4_STATUS_NOMEM; return NULL; }
    memcpy(s, r->data + r->pos, (size_t)n); s[n] = 0; r->pos += (size_t)n;
    return s;
}
static bool safe_path(const char *s) {
    if (!s || !*s || *s == '/' || s[strlen(s)-1] == '/' || strchr(s, '\\') || strchr(s, ':')) return false;
    for (const char *p = s; *p;) {
        const char *end = strchr(p, '/'); size_t n = end ? (size_t)(end-p) : strlen(p);
        if (!n || (n == 1 && p[0] == '.') || (n == 2 && p[0] == '.' && p[1] == '.')) return false;
        p += n; if (*p) p++;
    }
    return true;
}
static bool pinned_revision(const char *s) {
    if (!s || strlen(s) != 40) return false;
    for (size_t i = 0; i < 40; i++)
        if ((s[i] < '0' || s[i] > '9') && (s[i] < 'a' || s[i] > 'f')) return false;
    return true;
}
static void reconcile_host(ds4_resources *resources) {
#ifdef __linux__
    uint64_t available=0;
    if (ds4_linux_nonmovable_memory(&available))
        (void)ds4_resources_reconcile(resources,DS4_MEMORY_HOST,available,true);
#else
    (void)resources;
#endif
}
static bool file_hash(int fd, uint64_t bytes, const uint8_t expected[32]) {
    ds4_sha256 sha; ds4_sha256_init(&sha);
    uint8_t buffer[65536], hash[32];
    uint64_t offset = 0;
    while (offset < bytes) {
        size_t n = bytes-offset < sizeof(buffer) ? (size_t)(bytes-offset) : sizeof(buffer);
        ssize_t got = pread(fd, buffer, n, (off_t)offset);
        if (got <= 0) return false;
        ds4_sha256_update(&sha, buffer, (size_t)got); offset += (uint64_t)got;
    }
    ds4_sha256_final(&sha, hash);
    return !memcmp(hash, expected, 32);
}
static bool mapping_unchanged(const mapping *m) {
    struct stat now;
    const struct stat *old = &m->identity;
    if (fstat(m->fd, &now) || now.st_size != old->st_size ||
        now.st_mtime != old->st_mtime || now.st_ctime != old->st_ctime) return false;
#ifdef __APPLE__
    return now.st_mtimespec.tv_nsec == old->st_mtimespec.tv_nsec && now.st_ctimespec.tv_nsec == old->st_ctimespec.tv_nsec;
#else
    return now.st_mtim.tv_nsec == old->st_mtim.tv_nsec && now.st_ctim.tv_nsec == old->st_ctim.tv_nsec;
#endif
}
static bool bundle_unchanged(const ds4_bundle *b) {
    for (size_t i = 0; i < b->file_count; i++)
        if (!mapping_unchanged(&b->maps[i])) return false;
    return true;
}
static bool tensor_layout(const ds4_weight_tensor *t) {
    /* These are GGML storage IDs, not model-level quantization labels. */
    static const uint32_t blocks[40][2] = {
        [0]={1,4}, [1]={1,2}, [2]={32,18}, [3]={32,20}, [6]={32,22}, [7]={32,24},
        [8]={32,34}, [9]={32,36}, [10]={256,84}, [11]={256,110}, [12]={256,144},
        [13]={256,176}, [14]={256,210}, [15]={256,292}, [16]={256,66}, [17]={256,74},
        [18]={256,98}, [19]={256,50}, [20]={32,18}, [21]={256,110}, [22]={256,82},
        [23]={256,136}, [24]={1,1}, [25]={1,2}, [26]={1,4}, [27]={1,8}, [28]={1,8},
        [29]={256,56}, [30]={1,2}, [39]={32,17}};
    if (t->type >= 40 || !blocks[t->type][0] || t->block_elements != blocks[t->type][0] ||
        t->block_bytes != blocks[t->type][1] || t->rank > DS4_WEIGHT_MAX_DIMS ||
        !t->alignment || (t->alignment & (t->alignment-1)) || t->offset % t->alignment) return false;
    if (!t->rank) return t->block_elements == 1 && t->bytes == t->block_bytes;
    if (!t->shape[0] || t->shape[0] % t->block_elements || t->stride[0] != t->block_bytes) return false;
    uint64_t extent = t->shape[0] / t->block_elements;
    if (extent > UINT64_MAX / t->block_bytes) return false;
    extent *= t->block_bytes;
    for (unsigned i = 1; i < t->rank; i++) {
        if (!t->shape[i] || t->stride[i] != extent || t->shape[i] > UINT64_MAX / extent) return false;
        extent *= t->shape[i];
    }
    return extent == t->bytes;
}

ds4_status ds4_bundle_close(ds4_bundle *b) {
    if (!b) return DS4_STATUS_OK;
    if (b->groups) return DS4_STATUS_BUSY;
    for (size_t i = 0; b->files && i < b->file_count; i++) {
        if (b->maps && b->maps[i].map) munmap(b->maps[i].map, (size_t)b->files[i].bytes);
        if (b->maps && b->maps[i].fd >= 0) close(b->maps[i].fd);
        if (b->maps) (void)ds4_resource_release(b->maps[i].account);
        free((void *)b->files[i].path); free((void *)b->files[i].revision); free((void *)b->files[i].format);
    }
    for (size_t i = 0; b->components && i < b->component_count; i++) {
        free((void *)b->components[i].id); free((void *)b->components[i].revision); free((void *)b->components[i].compute_dtype);
    }
    for (size_t i = 0; b->tensors && i < b->tensor_count; i++) free((void *)b->tensors[i].name);
    free(b->model); free(b->revision); free(b->profile); free(b->files);
    free(b->components); free(b->tensors); free(b->maps);
    (void)ds4_resource_release(b->metadata);
    if (b->resources) reconcile_host(b->resources);
    free(b);
    return DS4_STATUS_OK;
}

ds4_status ds4_bundle_open(ds4_bundle **out, const char *index_path, const char *root,
                          ds4_resources *resources, uint64_t owner) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    const uint16_t endian = 1;
    if (*(const uint8_t *)&endian != 1) return DS4_STATUS_UNSUPPORTED;
    if (!index_path || !root || !root[0] || strlen(root) > 4096 || !resources || !owner) return DS4_STATUS_INVALID;
    FILE *fp = fopen(index_path, "rb");
    if (!fp) return DS4_STATUS_INVALID;
    uint8_t header[64], hash[32];
    if (fread(header, 1, 64, fp) != 64 || memcmp(header, "DS4BND\1\0", 8)) { fclose(fp); return DS4_STATUS_INVALID; }
    reader h = {header, 64, 8, DS4_STATUS_OK};
    uint64_t abi = integer(&h, 8), bytes = integer(&h, 8);
    h.pos = 56;
    if (abi != DS4_SUITE_ABI || integer(&h, 8) || !bytes || bytes > 64u*1024u*1024u) { fclose(fp); return DS4_STATUS_INVALID; }
    /* Bound the directory, duplicated strings, array tables and hash/read
     * workspace before any allocation. This scales with metadata, not weights. */
    uint64_t working[DS4_MEMORY_COUNT] = {[DS4_MEMORY_HOST]=16u*bytes+sizeof(ds4_bundle)+73728u};
    reconcile_host(resources);
    ds4_reservation *lease = NULL;
    ds4_status admission = ds4_resources_reserve(resources, owner, working, &lease);
    if (admission) { fclose(fp); return admission; }
    uint8_t *data = malloc((size_t)bytes);
    if (!data) { fclose(fp); ds4_reservation_cancel(lease); return DS4_STATUS_NOMEM; }
    bool read_ok = fread(data, 1, (size_t)bytes, fp) == bytes && fgetc(fp) == EOF;
    if (fclose(fp)) read_ok = false;
    ds4_sha256 sha; ds4_sha256_init(&sha); ds4_sha256_update(&sha, data, (size_t)bytes); ds4_sha256_final(&sha, hash);
    if (!read_ok || memcmp(hash, header+24, 32)) { free(data); ds4_reservation_cancel(lease); return DS4_STATUS_INVALID; }
    ds4_bundle *b = calloc(1, sizeof(*b));
    if (!b) { free(data); ds4_reservation_cancel(lease); return DS4_STATUS_NOMEM; }
    b->resources = resources;
    b->owner = owner;
    reader r = {data, (size_t)bytes, 0, DS4_STATUS_OK};
    b->model = string(&r); b->revision = string(&r); b->profile = string(&r);
    b->file_count = (size_t)integer(&r, 4); b->component_count = (size_t)integer(&r, 4); b->tensor_count = (size_t)integer(&r, 4);
    ds4_status status = DS4_STATUS_INVALID;
    if (r.status || !pinned_revision(b->revision) || !b->file_count || !b->component_count || !b->tensor_count ||
        b->file_count > bytes/48 || b->component_count > bytes/12 || b->tensor_count > bytes/64) goto fail;
    b->files = calloc(b->file_count, sizeof(*b->files)); b->maps = calloc(b->file_count, sizeof(*b->maps));
    if (b->maps) for (size_t i = 0; i < b->file_count; i++) b->maps[i].fd = -1;
    b->components = calloc(b->component_count, sizeof(*b->components)); b->tensors = calloc(b->tensor_count, sizeof(*b->tensors));
    if (!b->files || !b->maps || !b->components || !b->tensors) { status = DS4_STATUS_NOMEM; goto fail; }
    for (size_t i = 0; i < b->file_count; i++) {
        ds4_bundle_file *f = &b->files[i];
        f->path = string(&r); f->revision = string(&r); f->format = string(&r); f->bytes = integer(&r, 8);
        for (unsigned j = 0; j < 32; j++) f->sha256[j] = (uint8_t)integer(&r, 1);
        if (r.status || !pinned_revision(f->revision) || !safe_path(f->path) || !f->bytes || f->bytes > SIZE_MAX ||
            (strcmp(f->format, "gguf") && strcmp(f->format, "safetensors") && strcmp(f->format, "asset"))) goto fail;
        for (size_t j = 0; j < i; j++) if (!strcmp(f->path, b->files[j].path)) goto fail;
    }
    for (size_t i = 0; i < b->component_count; i++) {
        ds4_bundle_component *c = &b->components[i];
        c->id = string(&r); c->revision = string(&r); c->compute_dtype = string(&r);
        if (r.status || !pinned_revision(c->revision) ||
            (strcmp(c->compute_dtype, "float32") && strcmp(c->compute_dtype, "bfloat16"))) goto fail;
        for (size_t j = 0; j < i; j++) if (!strcmp(c->id, b->components[j].id)) goto fail;
    }
    for (size_t i = 0; i < b->tensor_count; i++) {
        ds4_weight_tensor *t = &b->tensors[i];
        t->name = string(&r); t->file = (uint32_t)integer(&r, 4); t->component = (uint32_t)integer(&r, 4);
        t->group = (uint32_t)integer(&r, 4); t->type = (uint32_t)integer(&r, 4); t->rank = (uint32_t)integer(&r, 4);
        t->block_elements = (uint32_t)integer(&r, 4); t->block_bytes = (uint32_t)integer(&r, 4); t->alignment = (uint32_t)integer(&r, 4);
        t->offset = integer(&r, 8); t->bytes = integer(&r, 8);
        if (t->rank > DS4_WEIGHT_MAX_DIMS) goto fail;
        for (unsigned j = 0; j < t->rank; j++) t->shape[j] = integer(&r, 8);
        for (unsigned j = 0; j < t->rank; j++) t->stride[j] = integer(&r, 8);
        if (r.status || t->file >= b->file_count || t->component >= b->component_count || !tensor_layout(t) ||
            strcmp(b->files[t->file].revision, b->components[t->component].revision) ||
            t->offset > b->files[t->file].bytes || t->bytes > b->files[t->file].bytes-t->offset ||
            !strcmp(b->files[t->file].format, "asset")) goto fail;
        for (size_t j = 0; j < i; j++) if (t->component == b->tensors[j].component && !strcmp(t->name, b->tensors[j].name)) goto fail;
    }
    if (r.status || r.pos != r.size) goto fail;
    for (size_t i = 0; i < b->file_count; i++) {
        const ds4_bundle_file *f = &b->files[i];
        size_t length = strlen(root)+strlen(f->path)+2;
        char *path = malloc(length);
        if (!path) { status = DS4_STATUS_NOMEM; goto fail; }
        snprintf(path, length, "%s/%s", root, f->path);
        b->maps[i].fd = open(path, O_RDONLY); free(path);
        mapping *m = &b->maps[i];
        if (m->fd < 0 || fstat(m->fd, &m->identity) || m->identity.st_size < 0 ||
            (uint64_t)m->identity.st_size != f->bytes || !file_hash(m->fd, f->bytes, f->sha256) ||
            !mapping_unchanged(m)) goto fail;
        ds4_resource_spec spec = {.owner=owner, .location=DS4_MEMORY_MMAP, .requested=f->bytes,
            .allocated=f->bytes, .live=f->bytes, .device=-1, .authoritative=true};
        status = ds4_resource_register(resources, &spec, &b->maps[i].account);
        if (status) goto fail;
        void *map = mmap(NULL, (size_t)f->bytes, PROT_READ, MAP_PRIVATE, b->maps[i].fd, 0);
        if (map == MAP_FAILED) { status = DS4_STATUS_NOMEM; goto fail; }
        b->maps[i].map = map;
        status = DS4_STATUS_INVALID;
    }
    uint64_t metadata_bytes = sizeof(*b) + b->file_count*(sizeof(*b->files)+sizeof(*b->maps)) +
        b->component_count*sizeof(*b->components) + b->tensor_count*sizeof(*b->tensors) +
        strlen(b->model)+strlen(b->revision)+strlen(b->profile)+3;
    for (size_t i=0; i<b->file_count; i++)
        metadata_bytes += strlen(b->files[i].path)+strlen(b->files[i].revision)+strlen(b->files[i].format)+3;
    for (size_t i=0; i<b->component_count; i++)
        metadata_bytes += strlen(b->components[i].id)+strlen(b->components[i].revision)+strlen(b->components[i].compute_dtype)+3;
    for (size_t i=0; i<b->tensor_count; i++) metadata_bytes += strlen(b->tensors[i].name)+1;
    ds4_resource_spec metadata = {.owner=owner, .location=DS4_MEMORY_HOST,
        .requested=metadata_bytes, .allocated=metadata_bytes, .live=metadata_bytes, .authoritative=true, .device=-1};
    status = ds4_reservation_commit(lease, &metadata, &b->metadata);
    if (status) goto fail;
    free(data); (void)ds4_reservation_cancel(lease); *out = b; return DS4_STATUS_OK;
fail:
    if (r.status) status = r.status;
    free(data); ds4_bundle_close(b); (void)ds4_reservation_cancel(lease); return status;
}

const char *ds4_bundle_model(const ds4_bundle *b) { return b ? b->model : NULL; }
const char *ds4_bundle_revision(const ds4_bundle *b) { return b ? b->revision : NULL; }
const char *ds4_bundle_profile(const ds4_bundle *b) { return b ? b->profile : NULL; }
size_t ds4_bundle_tensor_count(const ds4_bundle *b) { return b ? b->tensor_count : 0; }
size_t ds4_bundle_component_count(const ds4_bundle *b) { return b ? b->component_count : 0; }
size_t ds4_bundle_file_count(const ds4_bundle *b) { return b ? b->file_count : 0; }
const ds4_weight_tensor *ds4_bundle_tensor(const ds4_bundle *b, size_t i) { return b && i < b->tensor_count ? &b->tensors[i] : NULL; }
const ds4_bundle_component *ds4_bundle_component_at(const ds4_bundle *b, size_t i) { return b && i < b->component_count ? &b->components[i] : NULL; }
const ds4_bundle_file *ds4_bundle_file_at(const ds4_bundle *b, size_t i) { return b && i < b->file_count ? &b->files[i] : NULL; }
const void *ds4_bundle_file_data(const ds4_bundle *b, size_t i) { return b && i < b->file_count ? b->maps[i].map : NULL; }
const void *ds4_bundle_tensor_data(const ds4_bundle *b, size_t i) {
    const ds4_weight_tensor *t = ds4_bundle_tensor(b, i);
    return t ? (const uint8_t *)b->maps[t->file].map + t->offset : NULL;
}
ds4_status ds4_bundle_validate_component(const ds4_bundle *b, size_t component, const ds4_weight_tensor *expected, size_t count) {
    if (!b || component >= b->component_count || !expected || !count) return DS4_STATUS_INVALID;
    size_t found = 0;
    for (size_t i = 0; i < b->tensor_count; i++) {
        const ds4_weight_tensor *t = &b->tensors[i]; if (t->component != component) continue;
        bool matched = false;
        for (size_t j = 0; j < count; j++) {
            const ds4_weight_tensor *e = &expected[j];
            if (!e->name || strcmp(e->name, t->name)) continue;
            if (matched || e->type != t->type || e->rank != t->rank || e->group != t->group ||
                memcmp(e->shape, t->shape, t->rank*sizeof(uint64_t))) return DS4_STATUS_INVALID;
            matched = true;
        }
        if (!matched) return DS4_STATUS_INVALID;
        found++;
    }
    return found == count ? DS4_STATUS_OK : DS4_STATUS_INVALID;
}
int ds4_bundle_dump(const ds4_bundle *b, FILE *fp) {
    if (!b || !fp) return -1;
    for (size_t i = 0; i < b->tensor_count; i++) {
        const ds4_weight_tensor *t = &b->tensors[i];
        if (fprintf(fp, "%s\t%s\t%u\t%u\t%" PRIu64 "\t%" PRIu64, b->components[t->component].id,
            t->name, t->group, t->type, t->offset, t->bytes) < 0) return -1;
        for (unsigned j = 0; j < t->rank; j++) if (fprintf(fp, "\t%" PRIu64, t->shape[j]) < 0) return -1;
        if (fputc('\n', fp) == EOF) return -1;
    }
    return 0;
}

static ds4_status weight_group_create(ds4_weight_group **out, ds4_bundle *b, uint32_t component,
        uint32_t group, const size_t *indices, size_t count, const ds4_weight_backend *backend) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    if (!b || component >= b->component_count || !backend || !backend->allocate || !backend->write ||
        !backend->release || !backend->quiesce) return DS4_STATUS_INVALID;
    if (b->tensor_count > (SIZE_MAX-sizeof(ds4_weight_group))/sizeof(uint64_t)) return DS4_STATUS_INVALID;
    uint64_t metadata_bytes = sizeof(ds4_weight_group)+b->tensor_count*sizeof(uint64_t);
    ds4_resource_spec metadata = {.owner=b->owner, .location=DS4_MEMORY_HOST, .device=-1,
        .requested=metadata_bytes, .allocated=metadata_bytes, .live=metadata_bytes};
    ds4_resource *account = NULL;
    ds4_status status = ds4_resource_register(b->resources, &metadata, &account);
    if (status) return status;
    ds4_weight_group *g = calloc(1, sizeof(*g));
    if (!g) { ds4_resource_release(account); return DS4_STATUS_NOMEM; }
    g->metadata = account;
    g->offsets = calloc(b->tensor_count, sizeof(uint64_t));
    if (!g->offsets) { ds4_resource_release(account); free(g); return DS4_STATUS_NOMEM; }
    g->bundle=b; g->component=component; g->group=group; g->backend=*backend;
    for (size_t i=0;i<b->tensor_count;i++) g->offsets[i]=UINT64_MAX;
    for (size_t j=0;j<count;j++) {
        if(indices[j]>=b->tensor_count || b->tensors[indices[j]].component!=component) goto invalid;
        for(size_t k=0;k<j;k++) if(indices[k]==indices[j]) goto invalid;
    }
    for (size_t i = 0; i < b->tensor_count; i++) {
        const ds4_weight_tensor *t=&b->tensors[i];
        bool selected=!indices && t->component==component && t->group==group;
        for(size_t j=0;j<count;j++) if(indices[j]==i) selected=true;
        if (!selected) continue;
        uint64_t alignment = t->alignment > 32 ? t->alignment : 32;
        if (g->bytes > UINT64_MAX-(alignment-1)) goto invalid;
        g->bytes = (g->bytes+alignment-1) & ~(alignment-1); g->offsets[i]=g->bytes;
        if (t->bytes > UINT64_MAX-g->bytes) goto invalid;
        g->bytes += t->bytes;
    }
    if (!g->bytes) goto invalid;
    b->groups++; *out=g; return DS4_STATUS_OK;
invalid:
    ds4_resource_release(g->metadata); free(g->offsets); free(g); return DS4_STATUS_INVALID;
}
ds4_status ds4_weight_group_create(ds4_weight_group **out, ds4_bundle *b, uint32_t component,
                                  uint32_t group, const ds4_weight_backend *backend) {
    return weight_group_create(out,b,component,group,NULL,0,backend);
}
ds4_status ds4_weight_group_create_selected(ds4_weight_group **out, ds4_bundle *b, uint32_t component,
        const size_t *indices, size_t count, const ds4_weight_backend *backend) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    if(!indices||!count) return DS4_STATUS_INVALID;
    return weight_group_create(out,b,component,0,indices,count,backend);
}
uint64_t ds4_weight_group_bytes(const ds4_weight_group *g) { return g ? g->bytes : 0; }
bool ds4_weight_group_resident(const ds4_weight_group *g) { return g && g->resident; }
ds4_status ds4_weight_group_load(ds4_weight_group *g) {
    if (!g) return DS4_STATUS_INVALID;
    if (g->resident) return DS4_STATUS_OK;
    if (g->storage) return DS4_STATUS_NOT_READY; /* A failed upload needs drain/release. */
    if (!bundle_unchanged(g->bundle)) return DS4_STATUS_INVALID;
    ds4_status status = g->backend.allocate(g->backend.context, g->bytes, &g->storage);
    if (status) return status;
    if (!g->storage) return DS4_STATUS_NOMEM;
    for (size_t i=0; i<g->bundle->tensor_count && !status; i++) {
        const ds4_weight_tensor *t=&g->bundle->tensors[i];
        if (g->offsets[i]!=UINT64_MAX)
            status=g->backend.write(g->backend.context, g->storage, g->offsets[i], ds4_bundle_tensor_data(g->bundle,i), t->bytes);
    }
    if (!status) status=g->backend.quiesce(g->backend.context);
    if (!status && !bundle_unchanged(g->bundle)) status=DS4_STATUS_INVALID;
    if (!status) g->resident=true;
    return status;
}
ds4_status ds4_weight_group_release(ds4_weight_group *g) {
    if (!g) return DS4_STATUS_INVALID;
    if (!g->storage) return DS4_STATUS_OK;
    /* A failed drain/free leaves storage for recovery, not usable weights. */
    g->resident=false;
    ds4_status status=g->backend.quiesce(g->backend.context);
    if (!status) status=g->backend.release(g->backend.context,g->storage);
    if (!status) { g->storage=NULL; g->resident=false; }
    return status;
}
ds4_status ds4_weight_group_close(ds4_weight_group *g) {
    if (!g) return DS4_STATUS_OK;
    ds4_status status=ds4_weight_group_release(g);
    if (status) return status;
    g->bundle->groups--; ds4_resource_release(g->metadata); free(g->offsets); free(g); return DS4_STATUS_OK;
}
ds4_status ds4_weight_group_tensor(const ds4_weight_group *g, const char *name, const void **storage, uint64_t *offset) {
    if (!g || !name || !storage || !offset) return DS4_STATUS_INVALID;
    if (!g->resident) return DS4_STATUS_NOT_RESIDENT;
    for (size_t i=0; i<g->bundle->tensor_count; i++) {
        const ds4_weight_tensor *t=&g->bundle->tensors[i];
        if (g->offsets[i]!=UINT64_MAX && !strcmp(t->name,name)) {
            *storage=g->storage; *offset=g->offsets[i]; return DS4_STATUS_OK;
        }
    }
    return DS4_STATUS_INVALID;
}
