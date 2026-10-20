/* Host-side allocation accounting only. Native CUDA calls are captured here
 * before the macros below, so tensor owners can use the untracked primitive
 * and publish one root with reference-counted views instead. */
static ds4_resources *g_tensor_resources;
static pthread_mutex_t g_tensor_resources_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_cuda_account_events_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_cuda_account_event;
static bool g_cuda_account_multidevice;
static cudaError_t cuda_account_synchronize(void) {
    pthread_mutex_lock(&g_cuda_account_events_mutex);
    cudaError_t status = cudaDeviceSynchronize();
    if (status == cudaSuccess && g_tensor_resources)
        ds4_resources_complete(g_tensor_resources, g_cuda_account_event);
    pthread_mutex_unlock(&g_cuda_account_events_mutex);
    return status;
}
static ds4_status cuda_tensor_account_release(ds4_resource *account) {
    ds4_status status = ds4_resource_release(account);
    if (status == DS4_STATUS_BUSY && cuda_account_synchronize() == cudaSuccess)
        status = ds4_resource_release(account);
    return status;
}
extern "C" ds4_resources *ds4_gpu_resources(void) {
    pthread_mutex_lock(&g_tensor_resources_mutex);
    if (!g_tensor_resources && ds4_resources_create(&g_tensor_resources) == DS4_STATUS_OK) {
        size_t free_bytes = 0, total = 0;
        if (!g_cuda_account_multidevice && cudaMemGetInfo(&free_bytes, &total) == cudaSuccess) {
            ds4_resource_budget budget = {total, 0, free_bytes, true};
            (void)ds4_resources_set_budget(g_tensor_resources, DS4_MEMORY_DEVICE, &budget);
        }
        uint64_t host_free = 0;
        if (ds4_linux_nonmovable_memory(&host_free)) {
            ds4_resource_budget budget = {UINT64_MAX, 0, host_free, true};
            (void)ds4_resources_set_budget(g_tensor_resources, DS4_MEMORY_HOST, &budget);
        }
    }
    ds4_resources *r = g_tensor_resources;
    pthread_mutex_unlock(&g_tensor_resources_mutex);
    return r;
}
static cudaError_t cuda_untracked_malloc(void **p, size_t bytes) { return cudaMalloc(p, bytes); }
static cudaError_t cuda_untracked_managed(void **p, size_t bytes) { return cudaMallocManaged(p, bytes); }
static cudaError_t cuda_untracked_free(void *p) { return cudaFree(p); }
typedef struct cuda_account_entry {
    void *ptr;
    ds4_resource *account;
    struct cuda_account_entry *next;
} cuda_account_entry;
static cuda_account_entry *g_cuda_accounts;
static pthread_mutex_t g_cuda_accounts_mutex = PTHREAD_MUTEX_INITIALIZER;
static cudaError_t cuda_account_allocate(void **out, size_t bytes, ds4_memory_location location,
                                         unsigned flags, uint64_t owner) {
    cuda_account_entry *entry = (cuda_account_entry *)calloc(1, sizeof(*entry));
    if (!entry) return cudaErrorMemoryAllocation;
    ds4_resource_spec spec = {};
    spec.owner = owner;
    spec.requested = spec.allocated = spec.live = bytes;
    spec.location = location; spec.device = -1;
    if (location == DS4_MEMORY_DEVICE || location == DS4_MEMORY_MANAGED) (void)cudaGetDevice(&spec.device);
    if (ds4_resource_register(ds4_gpu_resources(), &spec, &entry->account) != DS4_STATUS_OK) {
        free(entry); return cudaErrorMemoryAllocation;
    }
    cudaError_t status = location == DS4_MEMORY_DEVICE ? cudaMalloc(out, bytes) :
        location == DS4_MEMORY_MANAGED ? cudaMallocManaged(out, bytes, flags) : cudaHostAlloc(out, bytes, flags);
    if (status != cudaSuccess || !*out) {
        (void)ds4_resource_release(entry->account); free(entry);
        (void)cudaGetLastError();
        if (location == DS4_MEMORY_DEVICE && !g_cuda_account_multidevice) {
            size_t free_bytes = 0, total = 0;
            if (cudaMemGetInfo(&free_bytes, &total) == cudaSuccess)
                (void)ds4_resources_reconcile(ds4_gpu_resources(), DS4_MEMORY_DEVICE, free_bytes, true);
        } else if (location == DS4_MEMORY_PINNED) {
            uint64_t host_free = 0;
            if (ds4_linux_nonmovable_memory(&host_free))
                (void)ds4_resources_reconcile(ds4_gpu_resources(), DS4_MEMORY_HOST, host_free, true);
        }
        return status;
    }
    entry->ptr = *out;
    pthread_mutex_lock(&g_cuda_accounts_mutex);
    entry->next = g_cuda_accounts; g_cuda_accounts = entry;
    pthread_mutex_unlock(&g_cuda_accounts_mutex);
    return cudaSuccess;
}
static cudaError_t cuda_account_managed(void **out, size_t bytes, unsigned flags = cudaMemAttachGlobal) {
    return cuda_account_allocate(out, bytes, DS4_MEMORY_MANAGED, flags, DS4_RESOURCE_OWNER_RUNTIME);
}
static cudaError_t cuda_account_free(void *ptr, bool host) {
    if (!ptr) return cudaSuccess;
    pthread_mutex_lock(&g_cuda_accounts_mutex);
    cuda_account_entry **link = &g_cuda_accounts;
    while (*link && (*link)->ptr != ptr) link = &(*link)->next;
    cuda_account_entry *entry = *link;
    if (entry) *link = entry->next;
    pthread_mutex_unlock(&g_cuda_accounts_mutex);
    cudaError_t status = host ? cudaFreeHost(ptr) : cudaFree(ptr);
    if (entry) {
        if (status == cudaSuccess) { (void)ds4_resource_release(entry->account); free(entry); }
        else {
            pthread_mutex_lock(&g_cuda_accounts_mutex);
            entry->next = g_cuda_accounts; g_cuda_accounts = entry;
            pthread_mutex_unlock(&g_cuda_accounts_mutex);
        }
    }
    if (!host && status == cudaSuccess && g_n_gpus <= 1) {
        size_t free_bytes = 0, total = 0;
        if (cudaMemGetInfo(&free_bytes, &total) == cudaSuccess)
            (void)ds4_resources_reconcile(ds4_gpu_resources(), DS4_MEMORY_DEVICE, free_bytes, true);
    }
    if (host && status == cudaSuccess) {
        uint64_t host_free = 0;
        if (ds4_linux_nonmovable_memory(&host_free))
            (void)ds4_resources_reconcile(ds4_gpu_resources(), DS4_MEMORY_HOST, host_free, true);
    }
    return status;
}
#define cudaMalloc(ptr_, bytes_) cuda_account_allocate((void **)(ptr_), (bytes_), DS4_MEMORY_DEVICE, 0, DS4_RESOURCE_OWNER_RUNTIME)
#define cudaMallocHost(ptr_, bytes_) cuda_account_allocate((void **)(ptr_), (bytes_), DS4_MEMORY_PINNED, cudaHostAllocDefault, DS4_RESOURCE_OWNER_RUNTIME)
#define cudaHostAlloc(ptr_, bytes_, flags_) cuda_account_allocate((void **)(ptr_), (bytes_), DS4_MEMORY_PINNED, (flags_), DS4_RESOURCE_OWNER_RUNTIME)
#define cuda_model_malloc(ptr_, bytes_) cuda_account_allocate((void **)(ptr_), (bytes_), DS4_MEMORY_DEVICE, 0, DS4_RESOURCE_OWNER_LEGACY_WEIGHTS)
#define cuda_model_malloc_host(ptr_, bytes_) cuda_account_allocate((void **)(ptr_), (bytes_), DS4_MEMORY_PINNED, cudaHostAllocDefault, DS4_RESOURCE_OWNER_LEGACY_WEIGHTS)
#define cudaMallocManaged(ptr_, ...) cuda_account_managed((void **)(ptr_), __VA_ARGS__)
#define cudaFree(ptr_) cuda_account_free((ptr_), false)
#define cudaFreeHost(ptr_) cuda_account_free((ptr_), true)
