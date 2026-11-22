/*
 * rd.cu - recursive doubling.
 *
 * Let p be the largest power of two <= k. GPUs p..k-1 are "extra":
 *   Fold:   extra GPU r sends its buffer to GPU r - p, which adds it in.
 *   Rounds: for d = 1, 2, 4, ..., p/2, GPU i < p exchanges its WHOLE buffer with
 *           GPU i XOR d: copy partner -> own scratch, wait until the partner has
 *           also finished copying, then ar_add the scratch into its own buffer.
 *           Both partners compute a + b and b + a, which are the same in IEEE
 *           arithmetic, so all GPUs keep bitwise identical values.
 *   Scale:  every GPU i < p multiplies by 1/k.
 *   Unfold: extra GPU r copies the result back from GPU r - p.
 * Cost: log2(p) * (alpha + n*beta + n*gamma), plus 2(alpha + n*beta) when k is not
 * a power of two. Few rounds, so it suits SMALL (latency-bound) messages; for large
 * messages every round moves the whole buffer, unlike ring.
 * Not done yet: recursive halving + doubling (Rabenseifner), the bandwidth-friendly
 * variant: 2 log2(k) alpha + 2 (k-1)/k n beta.
 */
#include "ar_sync.h"

void allreduce_rd(float **grads, size_t n, int k, cudaStream_t *streams) {
    int p = 1;
    while (p * 2 <= k) p *= 2;
    float *scratch[AR_SYNC_MAX];
    for (int i = 0; i < p; i++) scratch[i] = ar_scratch(i, 0, n * sizeof(float));

    /* Fold the extra GPUs into their partners */
    for (int r = p; r < k; r++) ar_mark(r, streams);
    for (int r = p; r < k; r++) {
        int q = r - p;
        ar_after(q, r, streams);
        ar_copy(scratch[q], q, grads[r], r, n, streams[q]);
        ar_add(q, grads[q], scratch[q], n, streams[q]);
    }

    for (int d = 1; d < p; d *= 2) {
        /* partner's buffer is final for this round */
        for (int i = 0; i < p; i++) ar_mark(i, streams);
        for (int i = 0; i < p; i++) {
            ar_after(i, i ^ d, streams);
            ar_copy(scratch[i], i, grads[i ^ d], i ^ d, n, streams[i]);
        }
        /* partner finished reading our buffer before we add into it */
        for (int i = 0; i < p; i++) ar_mark(i, streams);
        for (int i = 0; i < p; i++) {
            ar_after(i, i ^ d, streams);
            ar_add(i, grads[i], scratch[i], n, streams[i]);
        }
    }

    for (int i = 0; i < p; i++) ar_scale(i, grads[i], 1.0f / k, n, streams[i]);

    /* Unfold: extra GPUs fetch the result, then their partners wait for that read */
    for (int r = p; r < k; r++) ar_mark(r - p, streams);
    for (int r = p; r < k; r++) {
        ar_after(r, r - p, streams);
        ar_copy(grads[r], r, grads[r - p], r - p, n, streams[r]);
    }
    for (int r = p; r < k; r++) ar_mark(r, streams);
    for (int r = p; r < k; r++) ar_after(r - p, r, streams);
}
