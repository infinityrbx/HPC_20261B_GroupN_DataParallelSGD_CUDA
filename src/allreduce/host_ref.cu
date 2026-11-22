/*
 * host_ref.cu - REFERENCE allreduce (correctness baseline, not one of the compared methods).
 * Copies every buffer to host memory, averages on the CPU, copies back.
 * It is simple and obviously correct, so it is used to validate the other
 * methods and as the "no GPU-to-GPU communication" baseline. It blocks the host.
 */
#include "allreduce.h"
#include <stdio.h>
#include <stdlib.h>

static float *h_buf = NULL;   /* pinned, k * n floats */
static size_t h_cap = 0;

void allreduce_host_ref(float **grads, size_t n, int k, cudaStream_t *streams) {
    size_t need = (size_t)k * n;
    if (h_cap < need) {
        if (h_buf) CUDA_CHECK(cudaFreeHost(h_buf));
        CUDA_CHECK(cudaMallocHost(&h_buf, need * sizeof(float)));
        h_cap = need;
    }
    for (int i = 0; i < k; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpyAsync(h_buf + (size_t)i * n, grads[i], n * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
    }
    for (int i = 0; i < k; i++) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaStreamSynchronize(streams[i])); }

    float inv = 1.0f / k;
    for (size_t j = 0; j < n; j++) {
        float s = 0.0f;
        for (int i = 0; i < k; i++) s += h_buf[(size_t)i * n + j];
        h_buf[j] = s * inv;
    }
    for (int i = 0; i < k; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpyAsync(grads[i], h_buf, n * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
    }
    /* h_buf is reused on the next call; make sure the uploads are done first. */
    for (int i = 0; i < k; i++) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaStreamSynchronize(streams[i])); }
}
