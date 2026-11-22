/*
 * tree.cu - the paper's method (Hegde & Usmani, Sec. 5):
 *           reduce all gradients to GPU 0 ("driver"), average, broadcast back.
 *
 * Suggested steps:
 *   1. ar_barrier(k, streams) (or finer events) so GPU 0 may read every grads[j].
 *   2. For j = 1..k-1: copy grads[j] -> scratch on GPU 0 (cudaMemcpyPeerAsync on
 *      streams[0]) and ar_add(0, grads[0], scratch, n, streams[0]).
 *      Variant to compare: a binary tree (log2 k rounds of pairwise reduce).
 *   3. ar_scale(0, grads[0], 1.0f / k, n, streams[0]).
 *   4. Broadcast: copy grads[0] -> grads[j] for every j (make streams[j] wait for
 *      an event recorded on streams[0] after step 3).
 * Expected cost (alpha-beta model): about 2(k-1)(alpha + n*beta) for the flat
 * version; compare with the paper's O(L k) + O(k p / B).
 */
#include "allreduce.h"
#include <stdio.h>
#include <stdlib.h>

void allreduce_tree(float **grads, size_t n, int k, cudaStream_t *streams) {
    (void)grads; (void)n; (void)k; (void)streams;
    ar_todo("tree");
}
