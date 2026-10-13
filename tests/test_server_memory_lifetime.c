#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

/* A much shorter replacement frontier must not keep the former large host
 * allocation alive. Nearby sizes still reuse storage to avoid allocator churn. */
static void test_frontier_storage_shrinks(void) {
    ds4_session_snapshot state = {0};
    char err[128] = {0};
    FILE *fp = resident_kv_frontier_open_write(&state, 2 * 1048576, err, sizeof(err));
    TEST_ASSERT(fp != NULL);
    if (fp) fclose(fp);
    uint64_t large_cap = state.cap;
    fp = resident_kv_frontier_open_write(&state, 1048576, err, sizeof(err));
    TEST_ASSERT(fp != NULL);
    if (fp) fclose(fp);
    TEST_ASSERT(state.cap == large_cap);
    fp = resident_kv_frontier_open_write(&state, 65536, err, sizeof(err));
    TEST_ASSERT(fp != NULL);
    if (fp) {
        const uint8_t value = 0xa5;
        TEST_ASSERT(fseek(fp, 65535, SEEK_SET) == 0);
        TEST_ASSERT(fwrite(&value, 1, 1, fp) == 1);
        TEST_ASSERT(fclose(fp) == 0);
        TEST_ASSERT(state.ptr[65535] == value && state.ptr[65536] == 0);
    }
    TEST_ASSERT(state.cap == 65537);
    ds4_session_snapshot_free(&state);
    TEST_ASSERT(!state.ptr && !state.len && !state.cap);
}

int main(void) {
    test_frontier_storage_shrinks();
    test_resident_frontier_memory_stream_preserves_last_byte();
    if (test_failures) return 1;
    fprintf(stderr, "PASS: host frontier allocation lifetime and payload boundaries\n");
    return 0;
}
