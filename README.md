# HPC_20261B_GroupN_DataParallelSGD_CUDA

Topic 10 (Multi-GPU Communication Optimization), based on Hegde & Usmani,
*Parallel and Distributed Deep Learning* (Stanford). We train an MLP on MNIST with
synchronous data-parallel SGD on k GPUs and compare **hand-written allreduce
methods** for averaging gradients. The paper only analysed this cost
(O(L·k) + O(k·p/B)) in theory; we measure it.

> Rename the folder/repo to `HPC_20261B_Group<N>_DataParallelSGD_CUDA` with your group number.

## Who implements what

| File | Owner | Method |
|---|---|---|
| `allreduce/host_ref.cu` | given | Reference: copy to CPU, average, copy back. Used to validate the others |
| `allreduce/tree.cu` | member 1 | The paper's method: reduce to GPU 0, then broadcast |
| `allreduce/ring.cu` | member 2 | Ring: reduce-scatter + allgather, chunked pipelining |
| `allreduce/rd.cu` | member 3 | Recursive doubling / halving |
| `cuda/train_dp.cu` (`--overlap`) | member 4 | Overlap gradient communication with backprop |

Everything else is the **shared base**. Change it only by agreement with the group,
because changes affect every method's numbers.

## The interface (read `allreduce/allreduce.h`)

```c
void allreduce_<name>(float** grads, size_t n, int k, cudaStream_t* streams);
```
* `grads[i]` is on GPU i; afterwards every GPU holds the **average**.
* Enqueue on `streams[i]` and return (no device-wide syncs).
* Cross-GPU ordering is your job (`ar_barrier()` or your own events).
* Use `ar_scratch()`, `ar_add()`, `ar_scale()`. No NCCL, cuBLAS or MPI collectives.
* Register the method in `allreduce/registry.cu` (already done for tree, ring and rd).

## Fixed settings (`common/config.h`)

seed 42, global batch 256 (split evenly over k), lr 0.1, hidden H = 512, 10 epochs,
benchmark 5 warm-up + 20 timed runs (median). **All final numbers use these
defaults and the same machine.** CLI flags exist only for exploring.

## Build

```bash
make data          # download MNIST into ./data
make seq           # sequential C baseline (gcc only)
make ARCH=sm_75    # everything; sm_75 = T4 (Kaggle/Colab), sm_80 = A100
```

## Run

```bash
./bin/mlp_seq                                   # sequential baseline (CPU)
./bin/train_dp --gpus 1                         # single GPU
./bin/train_dp --gpus 2 --method ring           # data-parallel, plain
./bin/train_dp --gpus 2 --method ring --overlap # data-parallel, overlapped
./bin/bench_allreduce --method ring --gpus 2    # raw allreduce, 4 KB .. 512 MB
./bin/bench_allreduce --method ring --gpus 2 --check-only
bash scripts/run_experiments.sh 2 "tree ring rd"   # the final comparison
```

## Correctness checks (every method must pass all three)

1. `bench_allreduce --check-only`: result equals the exact average on every
   element for many sizes, including odd n (1, 7, 1000003).
2. `train_dp` prints the max parameter difference between replicas. It must be ~0.
3. Loss curves (`--loss-log`) for k GPUs match the 1-GPU and sequential curves
   (same global batch and seed) up to float rounding.

## Output (same CSV columns everywhere)

```
method,k,msg_bytes,time_ms,busbw_GBs,epoch_s,comm_pct,test_acc
```
`-1` means "not applicable". `busbw = bytes/time × 2(k−1)/k` (same definition as
nccl-tests). `comm_pct` = exposed communication (end of backward to end of
allreduce on GPU 0) as a % of step time.

## Layout

```
common/     config.h (fixed settings), MNIST loader, params/CLI, timer, CSV
seq/        mlp_seq.c: sequential C MLP (correctness reference)
cuda/       kernels.cu (tiled matmul, softmax, relu, ...), train_dp.cu (training loop)
allreduce/  allreduce.h (interface), common.cu (helpers), registry.cu, one file per method
bench/      bench_allreduce.cu: correctness + latency/bandwidth benchmark
scripts/    get_mnist.sh, run_experiments.sh
```

## Status

* `mlp_seq` was built and run with the fixed settings: **96.2 % test accuracy**, about 4.9 s per epoch
  (on one CPU core of the build machine). This is the reference every GPU version must match.
* **First GPU run on Kaggle (2 × Tesla T4, CUDA 13.0)**: everything builds; `host_ref` passes all
  correctness checks; seq, 1-GPU and 2-GPU training give the same loss to ~1e-5 and 92.7 % test
  accuracy after 2 epochs. 1 GPU: 0.24 s/epoch; 2 GPUs with `host_ref`: 0.41 s/epoch (64 % of
  the step is communication). That is the baseline to beat.
* **Kaggle T4s have NO peer access** (P2P matrix all 0, topology PHB). `cudaMemcpyPeerAsync` still
  works there but is staged through host memory by the driver, and kernels cannot read another
  GPU's memory directly. Design your method so it works with and without P2P.
* Re-run the check any time: `kaggle kernels push -p scripts/kaggle --accelerator NvidiaTeslaT4`
  (set your username in `scripts/kaggle/kernel-metadata.json` first).
