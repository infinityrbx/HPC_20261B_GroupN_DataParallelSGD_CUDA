/*
 * train_dp.cu - Synchronous data-parallel SGD on k GPUs (shared training loop).
 *
 * Every GPU holds a full copy of the parameters and processes B/k samples of each
 * global mini-batch. After backward, gradients are averaged with the selected
 * allreduce method, then every GPU applies the same SGD update, so all copies
 * stay identical (checked at the end).
 *
 *   ./train_dp --gpus 2 --method ring             # one allreduce per step
 *   ./train_dp --gpus 2 --method ring --overlap   # per-layer buckets, overlapped
 *   ./train_dp --gpus 1                           # single-GPU baseline
 *
 * Overlap mode: layer-2 gradients (W2,b2) are ready first in backward.
 * Their allreduce is launched on the comm stream while layer-1 backward still runs
 * on the compute stream; then layer-1 gradients (W1,b1) are reduced. Improving this
 * (more/smaller buckets, chunking W1, ...) is future work.
 *
 * Output: one CSV row: method,k,msg_bytes,-1,-1,epoch_s,comm_pct,test_acc
 *   comm_pct = exposed communication time (end of backward -> end of allreduce,
 *              measured on GPU 0 with cudaEvents) / step time * 100
 */
#include "../allreduce/allreduce.h"
#include "../common/config.h"
#include "../common/csv.h"
#include "../common/mnist.h"
#include "../common/params.h"
#include "../common/timer.h"
#include "kernels.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GPUS 8

typedef struct {
    int dev, B;                       /* local batch */
    float *p, *g;                     /* flat params / grads (same layout as CPU) */
    float *X, *Y, *Z1, *A1, *Z2, *dZ2, *dA1;
    float *loss; int *correct;        /* device accumulators */
    cudaStream_t comp, comm;
    cudaEvent_t ev_l2_ready, ev_bwd_done, ev_comm_done;
} Replica;

static void replica_init(Replica *r, int dev, int B, const ParamLayout *L, const float *h_init) {
    memset(r, 0, sizeof(*r));
    r->dev = dev; r->B = B;
    int H = L->H, C = CFG_CLASSES;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaMalloc(&r->p, L->total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->g, L->total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(r->p, h_init, L->total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&r->X,   (size_t)B * CFG_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->Y,   (size_t)B * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->Z1,  (size_t)B * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->A1,  (size_t)B * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->dA1, (size_t)B * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->Z2,  (size_t)B * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->dZ2, (size_t)B * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->correct, sizeof(int)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&r->comp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&r->comm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&r->ev_l2_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&r->ev_bwd_done));
    CUDA_CHECK(cudaEventCreate(&r->ev_comm_done));
}

/* Forward + loss for `rows` samples already in r->X / r->Y. */
static void forward(Replica *r, const ParamLayout *L, int rows, int with_grad) {
    int H = L->H, C = CFG_CLASSES;
    gpu_matmul(r->X, r->p + L->W1, r->Z1, rows, H, CFG_INPUT, 0, 0, r->comp);
    gpu_bias_relu(r->Z1, r->p + L->b1, r->A1, rows, H, r->comp);
    gpu_matmul(r->A1, r->p + L->W2, r->Z2, rows, C, H, 0, 0, r->comp);
    gpu_bias(r->Z2, r->p + L->b2, rows, C, r->comp);
    gpu_softmax_xent(r->Z2, r->Y, with_grad ? r->dZ2 : NULL, 1.0f / rows,
                     r->loss, r->correct, rows, C, r->comp);
}

/* Backward for layer 2: dW2 = A1^T dZ2, db2 = colsum(dZ2). */
static void backward_l2(Replica *r, const ParamLayout *L) {
    int H = L->H, C = CFG_CLASSES, B = r->B;
    gpu_matmul(r->A1, r->dZ2, r->g + L->W2, H, C, B, 1, 0, r->comp);
    gpu_colsum(r->dZ2, r->g + L->b2, B, C, r->comp);
}

/* Backward for layer 1: dA1 = dZ2 W2^T, relu mask, dW1 = X^T dZ1, db1 = colsum(dZ1). */
static void backward_l1(Replica *r, const ParamLayout *L) {
    int H = L->H, C = CFG_CLASSES, B = r->B;
    gpu_matmul(r->dZ2, r->p + L->W2, r->dA1, B, H, C, 0, 1, r->comp);
    gpu_relu_backward(r->dA1, r->Z1, (size_t)B * H, r->comp);
    gpu_matmul(r->X, r->dA1, r->g + L->W1, CFG_INPUT, H, B, 1, 0, r->comp);
    gpu_colsum(r->dA1, r->g + L->b1, B, H, r->comp);
}

int main(int argc, char **argv) {
    Options o;
    parse_options(argc, argv, &o);
    int k = o.gpus;
    if (k < 1 || k > MAX_GPUS) { fprintf(stderr, "--gpus must be 1..%d\n", MAX_GPUS); return 1; }
    if (o.batch % k) { fprintf(stderr, "global batch %d must be divisible by k=%d\n", o.batch, k); return 1; }
    if (allreduce_select(o.method)) return 1;

    Dataset train, test;
    if (mnist_load(o.data_dir, "train", &train) || mnist_load(o.data_dir, "t10k", &test)) return 1;

    ParamLayout L = param_layout(o.hidden);
    float *h_init = (float *)malloc(L.total * sizeof(float));
    init_params(h_init, &L, o.seed);            /* same init as the sequential baseline */

    ar_setup(k);
    int Bl = o.batch / k;
    Replica R[MAX_GPUS];
    for (int i = 0; i < k; i++) replica_init(&R[i], i, Bl, &L, h_init);

    /* Pinned host staging for the global batch. */
    float *hX, *hY;
    CUDA_CHECK(cudaMallocHost(&hX, (size_t)o.batch * CFG_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&hY, (size_t)o.batch * CFG_CLASSES * sizeof(float)));

    float *grads_all[MAX_GPUS], *grads_l1[MAX_GPUS], *grads_l2[MAX_GPUS];
    cudaStream_t comm[MAX_GPUS];
    for (int i = 0; i < k; i++) {
        grads_all[i] = R[i].g;                 /* whole buffer             */
        grads_l1[i]  = R[i].g + L.W1;          /* W1 | b1   (n = L.W2)     */
        grads_l2[i]  = R[i].g + L.W2;          /* W2 | b2   (n = total-W2) */
        comm[i] = R[i].comm;
    }
    size_t n_l1 = L.W2, n_l2 = L.total - L.W2;

    int *idx = (int *)malloc(train.n * sizeof(int));
    for (int i = 0; i < train.n; i++) idx[i] = i;
    uint32_t rng = o.seed ^ 0xA5A5A5A5u;       /* same shuffling as the sequential baseline */
    FILE *lf = o.loss_log ? fopen(o.loss_log, "w") : NULL;

    int steps = train.n / o.batch;
    double total_ms = 0, comm_ms_total = 0, step_ms_total = 0;

    fprintf(stderr, "method=%s k=%d overlap=%d H=%d params=%zu (%.2f MB) local_batch=%d\n",
            allreduce_selected_name(), k, o.overlap, o.hidden, L.total, L.total * 4 / 1e6, Bl);

    for (int ep = 0; ep < o.epochs; ep++) {
        shuffle_indices(idx, train.n, &rng);
        for (int i = 0; i < k; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaMemsetAsync(R[i].loss, 0, sizeof(float), R[i].comp));
        }
        double t_ep = now_ms();

        for (int s = 0; s < steps; s++) {
            gather_batch(&train, idx, s * o.batch, o.batch, hX, hY);
            double t_step = now_ms();

            /* Phase A: copy shard, forward, layer-2 backward on every GPU. */
            for (int i = 0; i < k; i++) {
                Replica *r = &R[i];
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaMemcpyAsync(r->X, hX + (size_t)i * Bl * CFG_INPUT,
                                           (size_t)Bl * CFG_INPUT * sizeof(float), cudaMemcpyHostToDevice, r->comp));
                CUDA_CHECK(cudaMemcpyAsync(r->Y, hY + (size_t)i * Bl * CFG_CLASSES,
                                           (size_t)Bl * CFG_CLASSES * sizeof(float), cudaMemcpyHostToDevice, r->comp));
                forward(r, &L, Bl, 1);
                backward_l2(r, &L);
                CUDA_CHECK(cudaEventRecord(r->ev_l2_ready, r->comp));
            }
            if (o.overlap) {
                for (int i = 0; i < k; i++) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaStreamWaitEvent(R[i].comm, R[i].ev_l2_ready, 0)); }
                allreduce(grads_l2, n_l2, k, comm);           /* overlaps with phase B */
            }

            /* Phase B: layer-1 backward. */
            for (int i = 0; i < k; i++) {
                CUDA_CHECK(cudaSetDevice(i));
                backward_l1(&R[i], &L);
                CUDA_CHECK(cudaEventRecord(R[i].ev_bwd_done, R[i].comp));
            }
            for (int i = 0; i < k; i++) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaStreamWaitEvent(R[i].comm, R[i].ev_bwd_done, 0)); }
            if (o.overlap) allreduce(grads_l1, n_l1, k, comm);
            else           allreduce(grads_all, L.total, k, comm);

            /* Phase C: update after communication finished. */
            for (int i = 0; i < k; i++) {
                Replica *r = &R[i];
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaEventRecord(r->ev_comm_done, r->comm));
                CUDA_CHECK(cudaStreamWaitEvent(r->comp, r->ev_comm_done, 0));
                gpu_sgd(r->p, r->g, o.lr, L.total, r->comp);
            }
            for (int i = 0; i < k; i++) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaStreamSynchronize(R[i].comp)); }

            float exposed = 0;
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaEventElapsedTime(&exposed, R[0].ev_bwd_done, R[0].ev_comm_done));
            comm_ms_total += exposed > 0 ? exposed : 0;
            step_ms_total += now_ms() - t_step;
        }
        double ms = now_ms() - t_ep;
        total_ms += ms;

        float loss_sum = 0, l;
        for (int i = 0; i < k; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaMemcpy(&l, R[i].loss, sizeof(float), cudaMemcpyDeviceToHost));
            loss_sum += l;
        }
        double mean_loss = loss_sum / ((double)steps * o.batch);
        fprintf(stderr, "epoch %d  loss %.5f  time %.3f s\n", ep + 1, mean_loss, ms / 1e3);
        if (lf) fprintf(lf, "%d,%.8f\n", ep + 1, mean_loss);
    }

    /* All replicas must hold identical parameters. */
    float *h_p0 = (float *)malloc(L.total * sizeof(float)), *h_pi = (float *)malloc(L.total * sizeof(float));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(h_p0, R[0].p, L.total * sizeof(float), cudaMemcpyDeviceToHost));
    double max_diff = 0;
    for (int i = 1; i < k; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(h_pi, R[i].p, L.total * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < L.total; j++) { double d = fabs(h_p0[j] - h_pi[j]); if (d > max_diff) max_diff = d; }
    }
    fprintf(stderr, "max parameter difference between replicas: %.3g %s\n", max_diff,
            max_diff < 1e-4 ? "(OK)" : "(MISMATCH: allreduce result differs between GPUs)");

    /* Test accuracy on GPU 0, in chunks of the local batch. */
    Replica *r0 = &R[0];
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemsetAsync(r0->correct, 0, sizeof(int), r0->comp));
    int *tidx = (int *)malloc(Bl * sizeof(int));
    for (int s = 0; s < test.n; s += Bl) {
        int cnt = test.n - s < Bl ? test.n - s : Bl;
        for (int i = 0; i < cnt; i++) tidx[i] = s + i;
        gather_batch(&test, tidx, 0, cnt, hX, hY);
        CUDA_CHECK(cudaMemcpyAsync(r0->X, hX, (size_t)cnt * CFG_INPUT * sizeof(float), cudaMemcpyHostToDevice, r0->comp));
        CUDA_CHECK(cudaMemcpyAsync(r0->Y, hY, (size_t)cnt * CFG_CLASSES * sizeof(float), cudaMemcpyHostToDevice, r0->comp));
        forward(r0, &L, cnt, 0);
        CUDA_CHECK(cudaStreamSynchronize(r0->comp));   /* hX is reused next iteration */
    }
    int correct = 0;
    CUDA_CHECK(cudaMemcpy(&correct, r0->correct, sizeof(int), cudaMemcpyDeviceToHost));
    double acc = (double)correct / test.n;
    double comm_pct = step_ms_total > 0 ? 100.0 * comm_ms_total / step_ms_total : 0;
    fprintf(stderr, "test accuracy %.4f   exposed comm %.1f%% of step time\n", acc, comm_pct);

    char name[64];
    snprintf(name, sizeof name, "%s%s", allreduce_selected_name(), o.overlap ? "+overlap" : "");
    if (k == 1) snprintf(name, sizeof name, "gpu1");
    csv_write(o.csv, name, k, L.total * sizeof(float), -1, -1, total_ms / 1e3 / o.epochs, comm_pct, acc);

    if (lf) fclose(lf);
    ar_teardown();
    return 0;
}
