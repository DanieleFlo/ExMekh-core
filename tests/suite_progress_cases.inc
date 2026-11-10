/* Real socket stream while an independent producer simulates slow compute.
 * No model, GPU or backend-specific header is needed for these contracts. */
typedef enum { SLOW_SUCCESS, SLOW_CANCELLED, SLOW_FAILED } slow_result;
typedef struct { server *s; job *j; slow_result result; } slow_operation;
static void *slow_image_operation(void *opaque) {
    slow_operation *run = opaque;
    ds4_operation_progress tokens = {.phase = "generation", .unit = "token",
        .completed_units = 7, .total_units = 0, .overall_fraction = -1};
    if (run->result == SLOW_SUCCESS)
        suite_progress_begin_items(run->j, 0, 3, 2);
    suite_progress_update(run->j, &tokens);
    usleep(350000);
    if (run->result == SLOW_CANCELLED) {
        assert(suite_cancel(run->j));
        suite_status_error(run->s, run->j, DS4_STATUS_CANCELLED);
    } else if (run->result == SLOW_FAILED) {
        suite_error(run->s, run->j, 400, "invalid_request", "request_cancelled", NULL);
    } else {
        assert(!suite_cancel(run->j));
        suite_progress_update(run->j, &(ds4_operation_progress){
            .phase = "generation", .overall_fraction = .75, .estimated = true});
        assert(run->j->suite_request->progress.overall_fraction == .5);
        suite_send(run->s, run->j, 200, "{\"created\":1,\"data\":[],\"ds4\":{}}");
    }
    job_complete(run->j);
    return NULL;
}
static void test_progress_contract(void) {
    static const char *const terminal_event[] = {
        [SLOW_SUCCESS] = "event: operation.completed",
        [SLOW_CANCELLED] = "event: operation.cancelled",
        [SLOW_FAILED] = "event: operation.failed",
    };
    for (slow_result result = SLOW_SUCCESS; result <= SLOW_FAILED; result++) {
        int pair[2]; assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, pair));
        set_client_socket_nonblocking(pair[0]);
        suite_http h = {.stream = true, .started = now_sec(),
            .progress = {.overall_fraction = -1, .item_count = 1}};
        h.deadline = h.started + (result == SLOW_CANCELLED ? .1 : 5);
        h.path = "/v1/images/generations";
        suite_id(h.request_id, "req-");
        job j = {.fd = pair[0], .suite_request = &h}; server s = {0};
        pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
        slow_operation run = {&s, &j, result}; pthread_t worker;
        assert(!pthread_create(&worker, NULL, slow_image_operation, &run));
        suite_stream_wait(&s, &j, .05);
        pthread_join(worker, NULL);
        shutdown(pair[0], SHUT_WR);
        buf wire = {0}; char block[4096]; ssize_t bytes;
        while ((bytes = recv(pair[1], block, sizeof(block), 0)) > 0)
            buf_append(&wire, block, bytes);
        assert(strstr(wire.ptr, "Content-Type: text/event-stream"));
        char correlation[96];
        snprintf(correlation, sizeof(correlation), "x-request-id: %s\r\n", h.request_id);
        assert(strstr(wire.ptr, correlation));
        const char *heartbeat = strstr(wire.ptr, "event: operation.heartbeat");
        const char *end = strstr(wire.ptr, terminal_event[result]);
        assert(heartbeat && end && heartbeat < end);
        assert(strstr(wire.ptr, "\"unit\":\"token\""));
        assert(strstr(wire.ptr, "\"total_units\":null"));
        assert(strstr(wire.ptr, "\"overall_fraction\":null"));
        if (result == SLOW_SUCCESS) {
            assert(strstr(wire.ptr, "\"item_index\":0,\"item_count\":3"));
            assert(strstr(wire.ptr, "\"active_items\":2"));
            assert(strstr(end, "\"overall_fraction\":1,\"overall_kind\":\"work\""));
            assert(strstr(end, "\"result\":{\"created\":1,\"data\":[],\"ds4\":{}}}\n\n"));
        }
        assert(!strstr(end + 7, "event: operation."));
        free(h.terminal.body); buf_free(&wire);
        pthread_cond_destroy(&j.cv); pthread_mutex_destroy(&j.mu);
        close(pair[0]); close(pair[1]);
    }
    /* A chat job can be rejected when a previous suite operation needs recovery. */
    int pair[2]; assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, pair));
    server s = {0}; job chat = {.fd = pair[0]};
    suite_unavailable(&s, &chat);
    char response[512] = {0};
    assert(recv(pair[1], response, sizeof(response) - 1, 0) > 0);
    assert(strstr(response, "503"));
    close(pair[0]); close(pair[1]);

    bool stream = false; char type[256] = {0}, key[129] = {0};
    const char accept[] = "Accept: text/event-stream\r\n";
    assert(suite_read_headers(accept, sizeof(accept)-1, type, sizeof(type), key, sizeof(key), &stream));
    assert(stream);
}
