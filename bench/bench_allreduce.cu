/*
 * bench_allreduce.cu - Raw allreduce benchmark + correctness test for ANY method.
 *
 *   ./bench_allreduce --method ring --gpus 2 [--min 4096] [--max 536870912] [--csv results.csv]
 *   ./bench_allreduce --method ring --gpus 2 --check-only     # correctness test only
 *
 * For each message size (bytes, doubling from --min to --max):
 *   1. correctness: GPU r holds f(i) + r; after allreduce every GPU must hold
 *      f(i) + (k-1)/2 (checked on every element, relative tol CFG_CHECK_TOL)
 *   2. CFG_BENCH_WARMUP untimed runs, then CFG_BENCH_REPEATS timed runs
 *      (host timer around the call + synchronize all streams), median reported
 *   3. CSV row: method,k,msg_bytes,time_ms,busbw_GBs,-1,-1,-1
 *      busbw = (bytes / time) * 2(k-1)/k   (same definition as nccl-tests)
 */
#include "../src/allreduce/allreduce.h"
#include "../src/common/config.h"
#include "../src/common/csv.h"
#include "../src/common/timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GPUS 8

static float f_of(size_t i) { return (float)(i % 1021) * 0.001f; }

__global__ void k_fill(float *x, size_t n, float rank) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        x[i] = (float)(i % 1021) * 0.001f + rank;
}

static void fill_all(float **g, size_t n, int k, cudaStream_t *st) {
    for (int r = 0; r < k; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        k_fill<<<1024, 256, 0, st[r]>>>(g[r], n, (float)r);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void sync_all(int k, cudaStream_t *st) {
    for (int r = 0; r < k; r++) { CUDA_CHECK(cudaSetDevice(r)); CUDA_CHECK(cudaStreamSynchronize(st[r])); }
}

/* Returns number of wrong elements (prints the first few). */
static size_t check(float **g, size_t n, int k, float *host) {
    size_t bad = 0;
    float off = (k - 1) * 0.5f;
    for (int r = 0; r < k; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemcpy(host, g[r], n * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i++) {
            float want = f_of(i) + off;
            if (fabsf(host[i] - want) > CFG_CHECK_TOL * fmaxf(1.0f, fabsf(want))) {
                if (bad < 5) fprintf(stderr, "  GPU %d elem %zu: got %g want %g\n", r, i, host[i], want);
                bad++;
            }
        }
    }
    return bad;
}

int main(int argc, char **argv) {
    const char *method = "host_ref", *csv = NULL;
    int k = 2, check_only = 0;
    size_t minb = 4096, maxb = (size_t)512 << 20;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--method") && i + 1 < argc) method = argv[++i];
        else if (!strcmp(argv[i], "--gpus")   && i + 1 < argc) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min")    && i + 1 < argc) minb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--max")    && i + 1 < argc) maxb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--csv")    && i + 1 < argc) csv = argv[++i];
        else if (!strcmp(argv[i], "--check-only")) check_only = 1;
        else { fprintf(stderr, "usage: %s --method NAME --gpus K [--min B] [--max B] [--csv F] [--check-only]\n", argv[0]); return 1; }
    }
    if (k < 2 || k > MAX_GPUS) { fprintf(stderr, "--gpus must be 2..%d\n", MAX_GPUS); return 1; }
    if (allreduce_select(method)) return 1;
    ar_setup(k);

    size_t maxn = maxb / sizeof(float);
    float *g[MAX_GPUS];
    cudaStream_t st[MAX_GPUS];
    for (int r = 0; r < k; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&g[r], maxn * sizeof(float)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&st[r], cudaStreamNonBlocking));
    }
    float *host = (float *)malloc(maxn * sizeof(float));
    double times[CFG_BENCH_REPEATS];
    int failed = 0;

    /* Odd sizes too, so methods that assume n % k == 0 get caught. */
    size_t odd[] = { 1, 7, 1000003 };
    for (size_t t = 0; t < sizeof(odd) / sizeof(odd[0]); t++) {
        if (odd[t] > maxn) continue;
        fill_all(g, odd[t], k, st);
        allreduce(g, odd[t], k, st);
        sync_all(k, st);
        size_t bad = check(g, odd[t], k, host);
        fprintf(stderr, "check n=%zu: %s\n", odd[t], bad ? "FAIL" : "ok");
        failed |= bad != 0;
    }

    fprintf(stderr, "%12s %12s %12s %8s\n", "bytes", "time_ms", "busbw_GB/s", "check");
    for (size_t bytes = minb; bytes <= maxb; bytes *= 2) {
        size_t n = bytes / sizeof(float);
        fill_all(g, n, k, st);
        allreduce(g, n, k, st);
        sync_all(k, st);
        size_t bad = check(g, n, k, host);
        failed |= bad != 0;
        if (check_only) { fprintf(stderr, "%12zu %12s %12s %8s\n", bytes, "-", "-", bad ? "FAIL" : "ok"); continue; }

        for (int w = 0; w < CFG_BENCH_WARMUP; w++) allreduce(g, n, k, st);
        sync_all(k, st);
        for (int rep = 0; rep < CFG_BENCH_REPEATS; rep++) {
            double t0 = now_ms();
            allreduce(g, n, k, st);
            sync_all(k, st);
            times[rep] = now_ms() - t0;
        }
        qsort(times, CFG_BENCH_REPEATS, sizeof(double), cmp_double);
        double med = times[CFG_BENCH_REPEATS / 2];
        double busbw = (bytes / (med * 1e-3)) * (2.0 * (k - 1) / k) / 1e9;
        fprintf(stderr, "%12zu %12.4f %12.3f %8s\n", bytes, med, busbw, bad ? "FAIL" : "ok");
        csv_write(csv, allreduce_selected_name(), k, bytes, med, busbw, -1, -1, -1);
    }

    fprintf(stderr, failed ? "CORRECTNESS: FAILED\n" : "CORRECTNESS: all sizes passed\n");
    ar_teardown();
    return failed ? 3 : 0;
}
