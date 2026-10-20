#define _POSIX_C_SOURCE 200809L
#include "ds4_checkpoint.h"
#include "ds4_sha256.h"
#include <limits.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void put64(uint8_t *p, uint64_t n) {
    for (unsigned i = 0; i < 8; i++) p[i] = (uint8_t)(n >> (8u * i));
}
static uint64_t get64(const uint8_t *p) {
    uint64_t n = 0;
    for (unsigned i = 0; i < 8; i++) n |= (uint64_t)p[i] << (8u * i);
    return n;
}
static bool digest(FILE *fp, uint64_t bytes, uint8_t hash[32]) {
    ds4_sha256 sha; ds4_sha256_init(&sha);
    uint8_t chunk[65536];
    while (bytes) {
        size_t n = bytes < sizeof(chunk) ? (size_t)bytes : sizeof(chunk);
        if (fread(chunk, 1, n, fp) != n) return false;
        ds4_sha256_update(&sha, chunk, n); bytes -= n;
    }
    ds4_sha256_final(&sha, hash); return true;
}
ds4_status ds4_checkpoint_seal(FILE *fp, const uint8_t identity[32],
        const uint8_t config[32], uint64_t payload, const void *metadata, size_t bytes) {
    if (!fp || !identity || !config || (bytes && !metadata) ||
        payload > INT64_MAX - DS4_CHECKPOINT_HEADER_BYTES ||
        bytes > INT64_MAX - DS4_CHECKPOINT_HEADER_BYTES - payload) return DS4_STATUS_INVALID;
    if (fseeko(fp, DS4_CHECKPOINT_HEADER_BYTES + (off_t)payload, SEEK_SET) ||
        (bytes && fwrite(metadata, 1, bytes, fp) != bytes) || fflush(fp)) return DS4_STATUS_NOMEM;
    uint8_t h[DS4_CHECKPOINT_HEADER_BYTES] = {'D','S','4','C','K','P',1,0};
    put64(h + 8, DS4_SUITE_ABI); put64(h + 16, payload); put64(h + 24, bytes);
    memcpy(h + 32, identity, 32); memcpy(h + 64, config, 32);
    if (fseeko(fp, DS4_CHECKPOINT_HEADER_BYTES, SEEK_SET) || !digest(fp, payload + bytes, h + 96) ||
        fseeko(fp, 0, SEEK_SET) || fwrite(h, 1, sizeof(h), fp) != sizeof(h) ||
        fflush(fp) ||
        ftruncate(fileno(fp), DS4_CHECKPOINT_HEADER_BYTES + (off_t)payload + (off_t)bytes) ||
        fsync(fileno(fp))) return DS4_STATUS_NOMEM;
    return DS4_STATUS_OK;
}
ds4_status ds4_checkpoint_validate(FILE *fp, const uint8_t identity[32],
        const uint8_t config[32], uint64_t *payload, void *metadata, size_t *bytes) {
    if (!fp || !identity || !config || !payload || !bytes) return DS4_STATUS_INVALID;
    uint8_t h[DS4_CHECKPOINT_HEADER_BYTES], hash[32];
    struct stat st;
    if (fseeko(fp, 0, SEEK_SET) || fread(h, 1, sizeof(h), fp) != sizeof(h) ||
        memcmp(h, "DS4CKP\1\0", 8) || get64(h + 8) != DS4_SUITE_ABI ||
        memcmp(h + 32, identity, 32) || memcmp(h + 64, config, 32) ||
        fstat(fileno(fp), &st) || st.st_size < DS4_CHECKPOINT_HEADER_BYTES) return DS4_STATUS_INVALID;
    uint64_t body = get64(h + 16), aux = get64(h + 24);
    uint64_t extent = (uint64_t)st.st_size - DS4_CHECKPOINT_HEADER_BYTES;
    if (body > extent || aux != extent - body || aux > SIZE_MAX ||
        !digest(fp, extent, hash) || memcmp(hash, h + 96, 32)) return DS4_STATUS_INVALID;
    size_t capacity = *bytes; *bytes = (size_t)aux;
    if ((aux && !metadata) || aux > capacity) return DS4_STATUS_NOMEM;
    if (fseeko(fp, DS4_CHECKPOINT_HEADER_BYTES + (off_t)body, SEEK_SET) ||
        (aux && fread(metadata, 1, (size_t)aux, fp) != aux) ||
        fseeko(fp, DS4_CHECKPOINT_HEADER_BYTES, SEEK_SET)) return DS4_STATUS_INVALID;
    *payload = body; return DS4_STATUS_OK;
}
