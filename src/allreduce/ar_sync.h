/*
 * ar_sync.h - small cross-GPU ordering helpers used by tree.cu, ring.cu and rd.cu.
 *
 * One event per GPU and per translation unit. ar_mark(i) records GPU i's event on
 * streams[i]; ar_after(i, j) makes streams[i] wait for the LAST mark of GPU j.
 * cudaStreamWaitEvent captures the event's state at the moment it is called, so the
 * same event can be re-recorded later as long as every wait on the old mark has
 * already been enqueued. All three methods enqueue strictly in that order.
 */
#ifndef AR_SYNC_H
#define AR_SYNC_H
#include "allreduce.h"
#include <stdio.h>
#include <stdlib.h>

#define AR_SYNC_MAX 8
static cudaEvent_t ar_ev_[AR_SYNC_MAX];

static inline void ar_mark(int i, cudaStream_t *streams) {
    CUDA_CHECK(cudaSetDevice(i));
    if (!ar_ev_[i]) CUDA_CHECK(cudaEventCreateWithFlags(&ar_ev_[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ar_ev_[i], streams[i]));
}

static inline void ar_after(int i, int j, cudaStream_t *streams) {
    if (i == j) return;
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaStreamWaitEvent(streams[i], ar_ev_[j], 0));
}

/* Copy n floats from GPU src_dev to GPU dst_dev, enqueued on the receiver's stream. */
static inline void ar_copy(float *dst, int dst_dev, const float *src, int src_dev,
                           size_t n, cudaStream_t s) {
    if (!n) return;
    CUDA_CHECK(cudaSetDevice(dst_dev));
    CUDA_CHECK(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, n * sizeof(float), s));
}
#endif
