/*
 * ring.cu - ring allreduce = reduce-scatter + allgather.
 *
 * The n floats are split into k chunks; chunk c has n/k floats plus one more if
 * c < n % k, so any n works (chunks can be empty when n < k).
 *   Reduce-scatter, step s = 0..k-2: GPU i sends chunk (i - s) mod k to GPU
 *     r = (i+1) mod k, which copies it into its scratch and adds it into its own
 *     copy of that chunk. After k-1 steps GPU i owns the fully summed chunk (i+1) mod k.
 *   Scale the owned chunk by 1/k.
 *   Allgather, step s = 0..k-2: GPU i sends chunk (i + 1 - s) mod k to GPU r, which
 *     copies it straight into its own buffer (no add).
 *
 * Ordering: before every step each GPU waits for BOTH ring neighbours to finish the
 * previous step. Waiting for the left neighbour makes the chunk it sends final;
 * waiting for the right neighbour makes sure it finished reading our buffer before
 * we write into it. Within one step the chunk a GPU writes and the chunk its right
 * neighbour reads are always different, so no further ordering is needed.
 * The same neighbour wait after the last step protects the caller's next use of grads.
 *
 * Copies are enqueued on the receiver's stream, so each GPU's scratch is only ever
 * touched by its own stream. Each chunk is summed on one GPU and then copied, so all
 * GPUs end with bitwise the same values.
 *
 * Expected cost: 2(k-1) alpha + 2 (k-1)/k * n * beta (+ (k-1)/k * n * gamma for the adds).
 * Not done yet: splitting each chunk into smaller pieces to pipeline copies and adds.
 */
#include "ar_sync.h"

static inline size_t chunk_off(size_t n, int k, int c) {
    size_t base = n / k, rem = n % k;
    return (size_t)c * base + ((size_t)c < rem ? (size_t)c : rem);
}
static inline size_t chunk_len(size_t n, int k, int c) {
    return n / k + ((size_t)c < n % k ? 1 : 0);
}
static inline int mod(int a, int k) { return ((a % k) + k) % k; }

/* Each GPU waits for its left and right ring neighbours' last mark. */
static void neighbour_sync(int k, cudaStream_t *streams) {
    for (int i = 0; i < k; i++) ar_mark(i, streams);
    for (int i = 0; i < k; i++) {
        ar_after(i, mod(i - 1, k), streams);
        ar_after(i, mod(i + 1, k), streams);
    }
}

void allreduce_ring(float **grads, size_t n, int k, cudaStream_t *streams) {
    float *scratch[AR_SYNC_MAX];
    size_t maxc = chunk_len(n, k, 0);
    for (int i = 0; i < k; i++) scratch[i] = ar_scratch(i, 0, (maxc ? maxc : 1) * sizeof(float));

    /* Reduce-scatter */
    for (int s = 0; s < k - 1; s++) {
        neighbour_sync(k, streams);
        for (int i = 0; i < k; i++) {
            int r = mod(i + 1, k), c = mod(i - s, k);
            size_t off = chunk_off(n, k, c), len = chunk_len(n, k, c);
            ar_copy(scratch[r], r, grads[i] + off, i, len, streams[r]);
            ar_add(r, grads[r] + off, scratch[r], len, streams[r]);
        }
    }
    /* GPU i now owns chunk (i+1) mod k: average it */
    for (int i = 0; i < k; i++) {
        int c = mod(i + 1, k);
        ar_scale(i, grads[i] + chunk_off(n, k, c), 1.0f / k, chunk_len(n, k, c), streams[i]);
    }
    /* Allgather */
    for (int s = 0; s < k - 1; s++) {
        neighbour_sync(k, streams);
        for (int i = 0; i < k; i++) {
            int r = mod(i + 1, k), c = mod(i + 1 - s, k);
            size_t off = chunk_off(n, k, c), len = chunk_len(n, k, c);
            ar_copy(grads[r] + off, r, grads[i] + off, i, len, streams[r]);
        }
    }
    neighbour_sync(k, streams);
}
