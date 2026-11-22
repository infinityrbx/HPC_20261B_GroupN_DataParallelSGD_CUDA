/* kernels.cuh - hand-written CUDA kernels for the MLP (no cuBLAS). */
#ifndef KERNELS_CUH
#define KERNELS_CUH
#include <cuda_runtime.h>
#include <stddef.h>

#define TILE 16

/* C (M x N) = op(A) * op(B); op(A) is M x K, op(B) is K x N. Row-major.
 * transA: A is stored K x M.  transB: B is stored N x K. */
void gpu_matmul(const float *A, const float *B, float *C, int M, int N, int K,
                int transA, int transB, cudaStream_t s);
/* Naive (one thread per output, no shared memory). For the matmul experiment. */
void gpu_matmul_naive(const float *A, const float *B, float *C, int M, int N, int K, cudaStream_t s);

/* Z[i,j] += b[j]; A = relu(Z)  (Z: rows x cols) */
void gpu_bias_relu(float *Z, const float *b, float *A, int rows, int cols, cudaStream_t s);
/* Z[i,j] += b[j] */
void gpu_bias(float *Z, const float *b, int rows, int cols, cudaStream_t s);
/* Row-wise softmax of Z (rows x C). If Y != NULL: dZ = (P - Y) * scale,
 * *loss += sum(-log P[y]), *correct += #argmax hits. dZ may be NULL (eval). */
void gpu_softmax_xent(const float *Z, const float *Y, float *dZ, float scale,
                      float *loss, int *correct, int rows, int C, cudaStream_t s);
/* dA[i] = Z[i] > 0 ? dA[i] : 0 */
void gpu_relu_backward(float *dA, const float *Z, size_t n, cudaStream_t s);
/* out[j] = sum_i M[i,j]  (M: rows x cols) */
void gpu_colsum(const float *M, float *out, int rows, int cols, cudaStream_t s);
/* p -= lr * g */
void gpu_sgd(float *p, const float *g, float lr, size_t n, cudaStream_t s);

#endif
