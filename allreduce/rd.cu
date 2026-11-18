/*
 * rd.cu - MEMBER 3: recursive doubling (and optionally recursive halving).
 *
 * Recursive doubling (k a power of two; handle other k by folding the extra GPUs
 * into partners first):
 *   for d = 1, 2, 4, ..., k/2:  GPU i exchanges its WHOLE buffer with GPU i XOR d
 *   and adds it (copy partner -> scratch, then ar_add). Both partners must read
 *   the other's buffer BEFORE either one adds into its own -> use a scratch copy
 *   and events. Finally ar_scale by 1/k.
 *   Cost: log2(k) * (alpha + n*beta)  -> best for SMALL messages (latency bound).
 * Recursive halving + doubling (Rabenseifner) is the bandwidth-friendly variant:
 *   halve the exchanged part each round (reduce-scatter), then double it back
 *   (allgather). Cost: 2 log2(k) alpha + 2 (k-1)/k n beta.
 */
#include "allreduce.h"
#include <stdio.h>
#include <stdlib.h>

void allreduce_rd(float **grads, size_t n, int k, cudaStream_t *streams) {
    (void)grads; (void)n; (void)k; (void)streams;
    ar_todo("rd");
}
