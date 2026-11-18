/*
 * mlp_seq.c - Sequential C baseline: 784 -> H (ReLU) -> 10 (softmax) MLP trained
 * with mini-batch SGD on MNIST. This is the correctness reference for every
 * GPU version (same init, same shuffling, same math).
 *
 * Model (row-major, B = batch size):
 *   Z1 = X W1 + b1        X: B x 784, W1: 784 x H
 *   A1 = relu(Z1)
 *   Z2 = A1 W2 + b2       W2: H x 10
 *   P  = softmax(Z2)
 *   L  = -(1/B) sum log P[y]              (paper eq. 2, averaged = eq. 4)
 * Backward:
 *   dZ2 = (P - Y) / B
 *   dW2 = A1^T dZ2,  db2 = colsum(dZ2)
 *   dA1 = dZ2 W2^T,  dZ1 = dA1 * (Z1 > 0)
 *   dW1 = X^T dZ1,   db1 = colsum(dZ1)
 * Update: w <- w - lr * g                  (paper eq. 3)
 */
#include "../common/config.h"
#include "../common/csv.h"
#include "../common/mnist.h"
#include "../common/params.h"
#include "../common/timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* C (MxN) = op(A) op(B), op = transpose if flag set. A is MxK after op, B is KxN after op.
 * lda/ldb are the row lengths of A and B as stored. */
static void matmul(const float *A, const float *B, float *C, int M, int N, int K,
                   int transA, int transB) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) C[(size_t)i * N + j] = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = transA ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
            if (a == 0.0f) continue;
            float *c = C + (size_t)i * N;
            if (!transB) {
                const float *b = B + (size_t)k * N;
                for (int j = 0; j < N; j++) c[j] += a * b[j];
            } else {
                for (int j = 0; j < N; j++) c[j] += a * B[(size_t)j * K + k];
            }
        }
    }
}

typedef struct {
    int B, H;
    float *Z1, *A1, *Z2, *P, *dZ2, *dA1;
} Work;

/* Forward pass. Returns summed (not averaged) cross-entropy loss; counts correct. */
static double forward(const float *p, const ParamLayout *L, const float *X, const float *Y,
                      int B, Work *w, int *correct) {
    int H = L->H, C = CFG_CLASSES;
    matmul(X, p + L->W1, w->Z1, B, H, CFG_INPUT, 0, 0);
    for (int i = 0; i < B; i++)
        for (int j = 0; j < H; j++) {
            float z = w->Z1[(size_t)i * H + j] + p[L->b1 + j];
            w->Z1[(size_t)i * H + j] = z;
            w->A1[(size_t)i * H + j] = z > 0 ? z : 0;
        }
    matmul(w->A1, p + L->W2, w->Z2, B, C, H, 0, 0);
    double loss = 0;
    for (int i = 0; i < B; i++) {
        float *z = w->Z2 + i * C, *pr = w->P + i * C;
        float m = -INFINITY; int arg = 0;
        for (int j = 0; j < C; j++) { z[j] += p[L->b2 + j]; if (z[j] > m) { m = z[j]; arg = j; } }
        float s = 0;
        for (int j = 0; j < C; j++) { pr[j] = expf(z[j] - m); s += pr[j]; }
        for (int j = 0; j < C; j++) {
            pr[j] /= s;
            if (Y && Y[i * C + j] > 0.5f) { loss -= log(pr[j] + 1e-12); if (correct && arg == j) (*correct)++; }
        }
    }
    return loss;
}

static void backward(const float *p, float *g, const ParamLayout *L, const float *X,
                     const float *Y, int B, Work *w) {
    int H = L->H, C = CFG_CLASSES;
    for (int i = 0; i < B * C; i++) w->dZ2[i] = (w->P[i] - Y[i]) / B;
    matmul(w->A1, w->dZ2, g + L->W2, H, C, B, 1, 0);              /* dW2 = A1^T dZ2 */
    for (int j = 0; j < C; j++) {
        float s = 0; for (int i = 0; i < B; i++) s += w->dZ2[i * C + j];
        g[L->b2 + j] = s;
    }
    matmul(w->dZ2, p + L->W2, w->dA1, B, H, C, 0, 1);              /* dA1 = dZ2 W2^T */
    for (size_t i = 0; i < (size_t)B * H; i++) if (w->Z1[i] <= 0) w->dA1[i] = 0;   /* dZ1 */
    matmul(X, w->dA1, g + L->W1, CFG_INPUT, H, B, 1, 0);           /* dW1 = X^T dZ1 */
    for (int j = 0; j < H; j++) {
        float s = 0; for (int i = 0; i < B; i++) s += w->dA1[(size_t)i * H + j];
        g[L->b1 + j] = s;
    }
}

int main(int argc, char **argv) {
    Options o;
    parse_options(argc, argv, &o);
    Dataset train, test;
    if (mnist_load(o.data_dir, "train", &train) || mnist_load(o.data_dir, "t10k", &test)) return 1;

    ParamLayout L = param_layout(o.hidden);
    float *p = malloc(L.total * sizeof(float)), *g = malloc(L.total * sizeof(float));
    init_params(p, &L, o.seed);

    int B = o.batch, H = o.hidden;
    Work w = { B, H,
        malloc((size_t)B * H * 4), malloc((size_t)B * H * 4), malloc((size_t)B * 10 * 4),
        malloc((size_t)B * 10 * 4), malloc((size_t)B * 10 * 4), malloc((size_t)B * H * 4) };
    float *X = malloc((size_t)B * CFG_INPUT * 4), *Y = malloc((size_t)B * 10 * 4);
    int *idx = malloc(train.n * sizeof(int));
    for (int i = 0; i < train.n; i++) idx[i] = i;
    uint32_t rng = o.seed ^ 0xA5A5A5A5u;
    FILE *lf = o.loss_log ? fopen(o.loss_log, "w") : NULL;

    int steps = train.n / B;          /* drop the remainder batch */
    double total_ms = 0;
    for (int ep = 0; ep < o.epochs; ep++) {
        shuffle_indices(idx, train.n, &rng);
        double t0 = now_ms(), loss = 0;
        for (int s = 0; s < steps; s++) {
            gather_batch(&train, idx, s * B, B, X, Y);
            loss += forward(p, &L, X, Y, B, &w, NULL) / B;
            backward(p, g, &L, X, Y, B, &w);
            for (size_t i = 0; i < L.total; i++) p[i] -= o.lr * g[i];
        }
        double ms = now_ms() - t0;
        total_ms += ms;
        fprintf(stderr, "epoch %d  loss %.5f  time %.2f s\n", ep + 1, loss / steps, ms / 1e3);
        if (lf) fprintf(lf, "%d,%.8f\n", ep + 1, loss / steps);
    }

    /* Test accuracy, in chunks of B */
    int correct = 0;
    for (int s = 0; s < test.n; s += B) {
        int cnt = test.n - s < B ? test.n - s : B;
        int *tidx = malloc(cnt * sizeof(int));
        for (int i = 0; i < cnt; i++) tidx[i] = s + i;
        gather_batch(&test, tidx, 0, cnt, X, Y);
        forward(p, &L, X, Y, cnt, &w, &correct);
        free(tidx);
    }
    double acc = (double)correct / test.n;
    fprintf(stderr, "test accuracy %.4f\n", acc);
    csv_write(o.csv, "seq", 1, L.total * sizeof(float), -1, -1, total_ms / 1e3 / o.epochs, 0, acc);

    if (lf) fclose(lf);
    return 0;
}
