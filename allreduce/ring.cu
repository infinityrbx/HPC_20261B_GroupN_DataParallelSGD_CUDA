/*
 * ring.cu - ring allreduce = reduce-scatter + allgather.
 *
 * Split the n floats into k chunks (handle n % k != 0: last chunk is shorter).
 *   Reduce-scatter, step s = 0..k-2:
 *     GPU i sends chunk (i - s) mod k to GPU (i+1) mod k, which adds it into its
 *     own copy of that chunk (peer copy into scratch, then ar_add).
 *     After k-1 steps GPU i owns the fully summed chunk (i + 1) mod k.
 *   Scale the owned chunk by 1/k.
 *   Allgather, step s = 0..k-2: pass the finished chunks around the ring (copy only).
 * Each step needs ordering: GPU i+1 must not read a chunk before GPU i finished
 * writing it -> record an event per step and cudaStreamWaitEvent on the receiver.
 * Optimization to try: pipeline each chunk in smaller pieces so copies and adds
 * overlap (that is the "chunked pipelining" in the plan).
 * Expected cost: 2(k-1) alpha + 2 (k-1)/k * n * beta.
 */
#include "allreduce.h"
#include <stdio.h>
#include <stdlib.h>

void allreduce_ring(float **grads, size_t n, int k, cudaStream_t *streams) {
    (void)grads; (void)n; (void)k; (void)streams;
    ar_todo("ring");
}
