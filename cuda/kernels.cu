#include "kernels.cuh"
#include "../allreduce/allreduce.h"
#include <float.h>
#include <stdio.h>

/* ---------------- Matrix multiply ---------------------------------------- */

/* Shared-memory tiled matmul with optional transposes. Each block computes a
 * TILE x TILE tile of C; each thread one element. */
__global__ void k_matmul_tiled(const float *__restrict__ A, const float *__restrict__ B,
                               float *__restrict__ C, int M, int N, int K, int tA, int tB) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        int ak = t + threadIdx.x;               /* column of op(A) loaded by this thread */
        int bk = t + threadIdx.y;               /* row of op(B) loaded by this thread    */
        float a = 0.0f, b = 0.0f;
        if (row < M && ak < K) a = tA ? A[(size_t)ak * M + row] : A[(size_t)row * K + ak];
        if (col < N && bk < K) b = tB ? B[(size_t)col * K + bk] : B[(size_t)bk * N + col];
        As[threadIdx.y][threadIdx.x] = a;
        Bs[threadIdx.y][threadIdx.x] = b;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

__global__ void k_matmul_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    C[(size_t)row * N + col] = acc;
}

void gpu_matmul(const float *A, const float *B, float *C, int M, int N, int K,
                int transA, int transB, cudaStream_t s) {
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    k_matmul_tiled<<<grid, block, 0, s>>>(A, B, C, M, N, K, transA, transB);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_matmul_naive(const float *A, const float *B, float *C, int M, int N, int K, cudaStream_t s) {
    dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    k_matmul_naive<<<grid, block, 0, s>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

/* ---------------- Element-wise / row-wise ------------------------------- */

__global__ void k_bias_relu(float *Z, const float *b, float *A, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float z = Z[i] + b[i % cols];
        Z[i] = z;
        if (A) A[i] = z > 0.0f ? z : 0.0f;
    }
}

static int grid1d(size_t n) {
    size_t g = (n + 255) / 256;
    return (int)(g > 4096 ? 4096 : (g ? g : 1));
}

void gpu_bias_relu(float *Z, const float *b, float *A, int rows, int cols, cudaStream_t s) {
    k_bias_relu<<<grid1d((size_t)rows * cols), 256, 0, s>>>(Z, b, A, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_bias(float *Z, const float *b, int rows, int cols, cudaStream_t s) {
    k_bias_relu<<<grid1d((size_t)rows * cols), 256, 0, s>>>(Z, b, NULL, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

/* One thread per row (C = 10 classes is tiny). */
__global__ void k_softmax_xent(const float *Z, const float *Y, float *dZ, float scale,
                               float *loss, int *correct, int rows, int C) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *z = Z + (size_t)r * C;
    float m = -FLT_MAX; int arg = 0;
    for (int j = 0; j < C; j++) if (z[j] > m) { m = z[j]; arg = j; }
    float sum = 0.0f;
    for (int j = 0; j < C; j++) sum += expf(z[j] - m);
    if (!Y) return;
    const float *y = Y + (size_t)r * C;
    float l = 0.0f; int hit = 0;
    for (int j = 0; j < C; j++) {
        float p = expf(z[j] - m) / sum;
        if (y[j] > 0.5f) { l = -logf(p + 1e-12f); hit = (arg == j); }
        if (dZ) dZ[(size_t)r * C + j] = (p - y[j]) * scale;
    }
    if (loss) atomicAdd(loss, l);
    if (correct && hit) atomicAdd(correct, 1);
}

void gpu_softmax_xent(const float *Z, const float *Y, float *dZ, float scale,
                      float *loss, int *correct, int rows, int C, cudaStream_t s) {
    k_softmax_xent<<<(rows + 127) / 128, 128, 0, s>>>(Z, Y, dZ, scale, loss, correct, rows, C);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_relu_backward(float *dA, const float *Z, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (Z[i] <= 0.0f) dA[i] = 0.0f;
}

void gpu_relu_backward(float *dA, const float *Z, size_t n, cudaStream_t s) {
    k_relu_backward<<<grid1d(n), 256, 0, s>>>(dA, Z, n);
    CUDA_CHECK(cudaGetLastError());
}

/* One thread per column; rows is the (small) local batch size. */
__global__ void k_colsum(const float *M, float *out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float s = 0.0f;
    for (int r = 0; r < rows; r++) s += M[(size_t)r * cols + c];
    out[c] = s;
}

void gpu_colsum(const float *M, float *out, int rows, int cols, cudaStream_t s) {
    k_colsum<<<(cols + 127) / 128, 128, 0, s>>>(M, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sgd(float *p, const float *g, float lr, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] -= lr * g[i];
}

void gpu_sgd(float *p, const float *g, float lr, size_t n, cudaStream_t s) {
    k_sgd<<<grid1d(n), 256, 0, s>>>(p, g, lr, n);
    CUDA_CHECK(cudaGetLastError());
}
