/* NIST SHA-256 vectors and arbitrary streaming boundaries, no model required. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4_sha256.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void digest_hex(ds4_sha256 *sha, char hex[65]) {
    uint8_t bytes[32];
    ds4_sha256_final(sha, bytes);
    for (unsigned i = 0; i < 32; i++)
        snprintf(hex + i * 2, 3, "%02x", bytes[i]);
}
static void check_vector(const char *data, size_t len, const char *expected) {
    for (size_t stride = 1; stride <= 65; stride++) {
        ds4_sha256 sha;
        ds4_sha256_init(&sha);
        ds4_sha256_update(&sha, NULL, 0);
        for (size_t offset = 0; offset < len;) {
            size_t bytes = len - offset < stride ? len - offset : stride;
            ds4_sha256_update(&sha, data + offset, bytes);
            offset += bytes;
        }
        char hex[65];
        digest_hex(&sha, hex);
        assert(!strcmp(hex, expected));
    }
}
int main(int argc, char **argv) {
    check_vector("", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check_vector("abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const char *long_vector = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    check_vector(long_vector, strlen(long_vector),
                 "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    ds4_sha256 sha;
    ds4_sha256_init(&sha);
    char block[1000], hex[65];
    memset(block, 'a', sizeof(block));
    for (unsigned i = 0; i < 1000; i++)
        ds4_sha256_update(&sha, block, sizeof(block));
    digest_hex(&sha, hex);
    assert(!strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
    puts("PASS: NIST SHA-256 vectors, zero-length updates and 1..65-byte streaming chunks");
    if (argc >= 2) {
        FILE *file = fopen(argv[1], "rb");
        assert(file);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        ds4_sha256_init(&sha);
        uint8_t data[65536];
        size_t n;
        uint64_t bytes = 0;
        while ((n = fread(data, 1, sizeof(data), file))) {
            ds4_sha256_update(&sha, data, n);
            bytes += n;
        }
        assert(!ferror(file) && !fclose(file));
        digest_hex(&sha, hex);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) * 1e-9;
        printf("SHA256_FILE bytes=%llu seconds=%.6f digest=%s\n",
               (unsigned long long)bytes, elapsed, hex);
        if (argc == 3)
            assert(!strcmp(hex, argv[2]));
    }
    return 0;
}
