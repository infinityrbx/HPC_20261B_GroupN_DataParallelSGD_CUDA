/*
 * bench_p2p.cu - Measures the alpha-beta(-gamma) parameters of the machine, so the
 *                cost models in the allreduce files can be checked against the
 *                measurements of bench_allreduce (idea borrowed from pdl-bench).
 *
 *   ./bench_p2p [--src 0] [--dst 1] [--min 4] [--max 536870912] [--csv results/p2p.csv]
 *
 * For each message size (bytes, doubling from --min to --max):
 *   p2p : ping-pong. GPU src copies n bytes to GPU dst with cudaMemcpyPeerAsync,
 *         then dst copies them back. One-way time = round trip / 2.
 *         Fitting  t(n) = alpha + beta * n  gives latency alpha and 1/bandwidth beta.
 *         Works with and without peer access (without P2P the driver stages the
 *         copy through host memory, which shows up as a larger beta).
 *   add : time of ar_add() on n/4 floats on GPU src (the reduction kernel every
 *         method uses).  t(n) = alpha_add + gamma * n  gives gamma.
 * Timing is the same as bench_allreduce: host timer around the call + stream
 * synchronize, CFG_BENCH_WARMUP untimed runs, median of CFG_BENCH_REPEATS.
 *
 * CSV rows (same columns as everywhere):
 *   p2p,2,msg_bytes,one_way_ms,GB/s,-1,-1,-1
 *   add,1,msg_bytes,time_ms,GB/s,-1,-1,-1
 * Here the busbw_GBs column holds the plain bandwidth bytes/time (no ring factor).
 * scripts/analyze.py fits alpha, beta, gamma from these rows.
 */
#include "../allreduce/allreduce.h"
#include "../common/config.h"
#include "../common/csv.h"
#include "../common/timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double median_of(double *t, int n) {
    qsort(t, n, sizeof(double), cmp_double);
    return t[n / 2];
}

int main(int argc, char **argv) {
    const char *csv = NULL;
    int src = 0, dst = 1;
    size_t minb = 4, maxb = (size_t)512 << 20;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--src") && i + 1 < argc) src = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dst") && i + 1 < argc) dst = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min") && i + 1 < argc) minb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--max") && i + 1 < argc) maxb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else { fprintf(stderr, "usage: %s [--src G] [--dst G] [--min B] [--max B] [--csv F]\n", argv[0]); return 1; }
    }
    if (minb < 4) minb = 4;
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    int need = (src > dst ? src : dst) + 1;
    if (src == dst || src < 0 || dst < 0 || need > count) {
        fprintf(stderr, "need two different GPUs (src=%d dst=%d, %d visible)\n", src, dst, count);
        return 1;
    }
    ar_setup(need);                                    /* enables peer access where possible */
    fprintf(stderr, "GPU %d -> GPU %d, peer access %s\n", src, dst,
            ar_can_p2p(src, dst) && ar_can_p2p(dst, src) ? "ON" : "OFF (copies staged through host)");

    float *a, *b, *c;
    cudaStream_t s_src, s_dst;
    CUDA_CHECK(cudaSetDevice(src));
    CUDA_CHECK(cudaMalloc(&a, maxb));
    CUDA_CHECK(cudaMalloc(&c, maxb));
    CUDA_CHECK(cudaMemset(a, 0, maxb));
    CUDA_CHECK(cudaMemset(c, 0, maxb));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_src, cudaStreamNonBlocking));
    CUDA_CHECK(cudaSetDevice(dst));
    CUDA_CHECK(cudaMalloc(&b, maxb));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_dst, cudaStreamNonBlocking));

    double t[CFG_BENCH_REPEATS];
    fprintf(stderr, "%12s %14s %10s %14s %10s\n", "bytes", "p2p_1way_ms", "GB/s", "add_ms", "GB/s");
    for (size_t bytes = minb; bytes <= maxb; bytes *= 2) {
        /* ping-pong: src -> dst on s_src, then dst -> src on s_dst (ordered by sync). */
        for (int rep = -CFG_BENCH_WARMUP; rep < CFG_BENCH_REPEATS; rep++) {
            double t0 = now_ms();
            CUDA_CHECK(cudaMemcpyPeerAsync(b, dst, a, src, bytes, s_src));
            CUDA_CHECK(cudaStreamSynchronize(s_src));
            CUDA_CHECK(cudaMemcpyPeerAsync(a, src, b, dst, bytes, s_dst));
            CUDA_CHECK(cudaStreamSynchronize(s_dst));
            if (rep >= 0) t[rep] = (now_ms() - t0) / 2;
        }
        double p2p = median_of(t, CFG_BENCH_REPEATS);

        /* reduction kernel: c += a on n floats. */
        size_t n = bytes / sizeof(float);
        for (int rep = -CFG_BENCH_WARMUP; rep < CFG_BENCH_REPEATS; rep++) {
            double t0 = now_ms();
            ar_add(src, c, a, n, s_src);
            CUDA_CHECK(cudaStreamSynchronize(s_src));
            if (rep >= 0) t[rep] = now_ms() - t0;
        }
        double add = median_of(t, CFG_BENCH_REPEATS);

        double bw_p2p = bytes / (p2p * 1e-3) / 1e9, bw_add = bytes / (add * 1e-3) / 1e9;
        fprintf(stderr, "%12zu %14.5f %10.3f %14.5f %10.3f\n", bytes, p2p, bw_p2p, add, bw_add);
        csv_write(csv, "p2p", 2, bytes, p2p, bw_p2p, -1, -1, -1);
        csv_write(csv, "add", 1, bytes, add, bw_add, -1, -1, -1);
    }

    CUDA_CHECK(cudaSetDevice(src));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaSetDevice(dst));
    CUDA_CHECK(cudaFree(b));
    ar_teardown();
    return 0;
}
