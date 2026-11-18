/* common.cu - helpers shared by all allreduce methods. Do not change per method. */
#include "allreduce.h"
#include <stdio.h>
#include <stdlib.h>

#define AR_MAX_GPUS 8
#define AR_SLOTS 4

static int    g_k = 0;
static int    g_p2p[AR_MAX_GPUS][AR_MAX_GPUS];
static float *g_scratch[AR_MAX_GPUS][AR_SLOTS];
static size_t g_scratch_bytes[AR_MAX_GPUS][AR_SLOTS];

void ar_setup(int k) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (k > count || k > AR_MAX_GPUS) {
        fprintf(stderr, "requested %d GPUs, %d available (max %d)\n", k, count, AR_MAX_GPUS);
        exit(1);
    }
    g_k = k;
    for (int i = 0; i < k; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < k; j++) {
            g_p2p[i][j] = (i == j);
            if (i == j) continue;
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
            if (can) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e = cudaSuccess; }
                CUDA_CHECK(e);
            }
            g_p2p[i][j] = can;
        }
    }
    fprintf(stderr, "P2P matrix (row i can access column j):\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) fprintf(stderr, " %d", g_p2p[i][j]);
        fprintf(stderr, "\n");
    }
}

int ar_can_p2p(int i, int j) { return g_p2p[i][j]; }

float *ar_scratch(int dev, int slot, size_t bytes) {
    if (dev < 0 || dev >= AR_MAX_GPUS || slot < 0 || slot >= AR_SLOTS) {
        fprintf(stderr, "ar_scratch: bad dev/slot\n"); exit(1);
    }
    if (g_scratch_bytes[dev][slot] < bytes) {
        CUDA_CHECK(cudaSetDevice(dev));
        /* Growing a buffer must not race with in-flight work that uses the old one. */
        if (g_scratch[dev][slot]) { CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaFree(g_scratch[dev][slot])); }
        CUDA_CHECK(cudaMalloc(&g_scratch[dev][slot], bytes));
        g_scratch_bytes[dev][slot] = bytes;
    }
    return g_scratch[dev][slot];
}

__global__ void k_add(float *dst, const float *src, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] += src[i];
}

__global__ void k_scale(float *x, float a, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        x[i] *= a;
}

static int grid_for(size_t n) {
    size_t b = (n + 255) / 256;
    return (int)(b > 4096 ? 4096 : (b ? b : 1));
}

void ar_add(int dev, float *dst, const float *src, size_t n, cudaStream_t s) {
    if (!n) return;
    CUDA_CHECK(cudaSetDevice(dev));
    k_add<<<grid_for(n), 256, 0, s>>>(dst, src, n);
    CUDA_CHECK(cudaGetLastError());
}

void ar_scale(int dev, float *x, float a, size_t n, cudaStream_t s) {
    if (!n) return;
    CUDA_CHECK(cudaSetDevice(dev));
    k_scale<<<grid_for(n), 256, 0, s>>>(x, a, n);
    CUDA_CHECK(cudaGetLastError());
}

static cudaEvent_t g_bar_ev[AR_MAX_GPUS];

void ar_barrier(int k, cudaStream_t *streams) {
    for (int i = 0; i < k; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        if (!g_bar_ev[i]) CUDA_CHECK(cudaEventCreateWithFlags(&g_bar_ev[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(g_bar_ev[i], streams[i]));
    }
    for (int i = 0; i < k; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < k; j++)
            if (j != i) CUDA_CHECK(cudaStreamWaitEvent(streams[i], g_bar_ev[j], 0));
    }
}

void ar_teardown(void) {
    for (int d = 0; d < AR_MAX_GPUS; d++)
        for (int s = 0; s < AR_SLOTS; s++)
            if (g_scratch[d][s]) {
                cudaSetDevice(d);
                cudaFree(g_scratch[d][s]);
                g_scratch[d][s] = NULL; g_scratch_bytes[d][s] = 0;
            }
    (void)g_k;
}

void ar_todo(const char *name) {
    fprintf(stderr, "allreduce method '%s' is not implemented yet (see allreduce/%s.cu)\n", name, name);
    exit(2);
}
