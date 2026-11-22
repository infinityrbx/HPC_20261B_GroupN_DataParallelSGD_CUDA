/*
 * sim_allreduce.cpp - GPU-free check of tree, ring and rd (make simcheck, needs only g++).
 *
 * The methods are compiled against tests/mock/cuda_runtime.h: every copy, add, scale,
 * event record and event wait is queued on its stream, and the queues are then run
 * in a RANDOM order that respects only stream order and event waits, like real GPUs
 * running concurrently. A missing event wait shows up as a wrong or non-identical
 * result in some of the random schedules. Checked for k = 2..8, odd n, two
 * back-to-back calls (as in --overlap), and the caller overwriting
 * the gradients right after the call. Each operation runs atomically, so this does
 * not model two copies overlapping inside one operation, and it says nothing about
 * speed or about real P2P behaviour: the GPU checks are still required.
 */
#include "allreduce.h"
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <vector>
#include <cmath>
using namespace std;

struct Op { int wait_token = -1; function<void()> fn; };
struct SimStream { deque<Op> q; };
struct SimEvent { int token = -1; };
static vector<char> done_tokens;
static mt19937 rng;

cudaError_t cudaSetDevice(int) { return 0; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned) { *e = new SimEvent(); return 0; }
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
    int t = done_tokens.size(); done_tokens.push_back(0); e->token = t;
    s->q.push_back({-1, [t] { done_tokens[t] = 1; }}); return 0;
}
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned) {
    if (e->token >= 0) s->q.push_back({e->token, nullptr}); return 0;
}
cudaError_t cudaMemcpyPeerAsync(void *d, int, const void *src, int, size_t b, cudaStream_t s) {
    s->q.push_back({-1, [=] { memmove(d, src, b); }}); return 0;
}
// helpers normally in common.cu
static vector<vector<float>> scratch_mem(8 * 4);
float *ar_scratch(int dev, int slot, size_t bytes) {
    auto &v = scratch_mem[dev * 4 + slot];
    if (v.size() * 4 < bytes) v.assign(bytes / 4 + 1, NAN);
    return v.data();
}
void ar_add(int, float *d, const float *s, size_t n, cudaStream_t st) {
    st->q.push_back({-1, [=] { for (size_t i = 0; i < n; i++) d[i] += s[i]; }});
}
void ar_scale(int, float *x, float a, size_t n, cudaStream_t st) {
    st->q.push_back({-1, [=] { for (size_t i = 0; i < n; i++) x[i] *= a; }});
}

static void run_all(vector<SimStream> &S) {
    for (;;) {
        vector<int> ready; bool any = false;
        for (int i = 0; i < (int)S.size(); i++) if (!S[i].q.empty()) {
            any = true; auto &o = S[i].q.front();
            if (o.wait_token < 0 || done_tokens[o.wait_token]) ready.push_back(i);
        }
        if (!any) return;
        if (ready.empty()) { fprintf(stderr, "DEADLOCK\n"); exit(5); }
        int i = ready[rng() % ready.size()];
        Op o = S[i].q.front(); S[i].q.pop_front();
        if (o.fn) o.fn();
    }
}

int main() {
    struct M { const char *name; allreduce_fn fn; } methods[] = {
        {"tree", allreduce_tree}, {"ring", allreduce_ring}, {"rd", allreduce_rd}};
    size_t sizes[] = {1, 2, 3, 7, 8, 13, 64, 1000, 1003};
    int fails = 0, runs = 0;
    for (auto &m : methods) for (int k = 2; k <= 8; k++) for (size_t n : sizes) for (int seed = 0; seed < 30; seed++) {
        rng.seed(seed * 7919 + k * 31 + n);
        vector<SimStream> S(k); vector<cudaStream_t> st(k);
        for (int i = 0; i < k; i++) st[i] = &S[i];
        // two buffers per GPU, reduced back to back (like --overlap)
        vector<vector<float>> A(k, vector<float>(n)), B(k, vector<float>(n)), outA(k, vector<float>(n)), outB(k, vector<float>(n));
        for (int i = 0; i < k; i++) for (size_t e = 0; e < n; e++) { A[i][e] = (float)((i + 1) * 3 + e % 5); B[i][e] = (float)((i + 2) * 5 + e % 3); }
        vector<float *> pa(k), pb(k);
        for (int i = 0; i < k; i++) { pa[i] = A[i].data(); pb[i] = B[i].data(); }
        m.fn(pa.data(), n, k, st.data());
        m.fn(pb.data(), n, k, st.data());
        // caller's next use: snapshot, then overwrite the gradients
        for (int i = 0; i < k; i++) {
            float *a = pa[i], *b = pb[i], *oa = outA[i].data(), *ob = outB[i].data();
            S[i].q.push_back({-1, [=] { memcpy(oa, a, n * 4); memcpy(ob, b, n * 4);
                                         for (size_t e = 0; e < n; e++) { a[e] = -1e9f; b[e] = -1e9f; } }});
        }
        run_all(S); runs++;
        bool ok = true;
        for (size_t e = 0; e < n && ok; e++) {
            double wa = 0, wb = 0;
            for (int i = 0; i < k; i++) { wa += (i + 1) * 3 + e % 5; wb += (i + 2) * 5 + e % 3; }
            wa /= k; wb /= k;
            for (int i = 0; i < k; i++) {
                if (fabs(outA[i][e] - wa) > 1e-5 * fmax(1, fabs(wa)) || fabs(outB[i][e] - wb) > 1e-5 * fmax(1, fabs(wb))) ok = false;
                if (memcmp(&outA[i][e], &outA[0][e], 4) || memcmp(&outB[i][e], &outB[0][e], 4)) ok = false; // bitwise equal replicas
            }
        }
        if (!ok) { if (fails < 10) fprintf(stderr, "FAIL %s k=%d n=%zu seed=%d\n", m.name, k, n, seed); fails++; }
    }
    printf("%d runs, %d failures\n", runs, fails);
    return fails != 0;
}
