#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(1); }
}
int main() {
    const size_t bytes = 64u * 1048576u;
    void *host = NULL, *device = NULL;
    cudaEvent_t start, end; cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    check(cudaMallocHost(&host, bytes)); check(cudaMalloc(&device, bytes));
    memset(host, 17, bytes);
    check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
    printf("{\"gpu\":\"%s\",\"bytes\":%zu,\"warmup\":2,\"samples\":[", prop.name, bytes);
    for (int direction = 0; direction < 2; direction++) for (int sample = -2; sample < 5; sample++) {
        check(cudaEventRecord(start));
        for (int repeat = 0; repeat < 8; repeat++)
            check(cudaMemcpyAsync(direction ? host : device, direction ? device : host, bytes,
                                  direction ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice));
        check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        float ms; check(cudaEventElapsedTime(&ms, start, end));
        if (sample >= 0) printf("%s{\"direction\":\"%s\",\"GiB_s\":%.6f}",
            direction || sample ? "," : "", direction ? "D2H" : "H2D", bytes * 8.0 / 1073741824.0 / (ms / 1000.0));
    }
    puts("]}");
    check(cudaFree(device)); check(cudaFreeHost(host));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
}
