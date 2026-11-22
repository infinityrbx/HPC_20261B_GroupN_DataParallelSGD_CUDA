// Host-only mock of the few CUDA runtime calls the allreduce methods use.
// Every call is queued on its stream; sim.cpp runs the queues in a random
// order that respects only stream order and event waits, to catch races.
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorPeerAccessAlreadyEnabled = 1 };
enum { cudaEventDisableTiming = 2, cudaMemcpyDeviceToHost = 0, cudaMemcpyHostToDevice = 1 };
struct SimStream; struct SimEvent;
typedef SimStream *cudaStream_t;
typedef SimEvent *cudaEvent_t;
inline const char *cudaGetErrorString(cudaError_t) { return "mock"; }
cudaError_t cudaSetDevice(int);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaMemcpyPeerAsync(void *, int, const void *, int, size_t, cudaStream_t);
