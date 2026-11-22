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
./bin/train_dp --gpus 2 --method ring --per-layer  # per-layer buckets, no overlap
./bin/train_dp --gpus 2 --method ring --phase-csv results/phases.csv  # + time per phase
./bin/bench_allreduce --method ring --gpus 2    # raw allreduce, 4 KB .. 512 MB
./bin/bench_allreduce --method ring --gpus 2 --check-only
./bin/bench_p2p --csv results/p2p.csv           # alpha, beta (copy) and gamma (ar_add)
bash scripts/run_experiments.sh 2 "tree ring rd"   # the final comparison + report
python3 scripts/analyze.py results              # report only (results/report.md, fig_*.png)
```

## Correctness checks (every method must pass all three)

1. `bench_allreduce --check-only`: result equals the exact average on every
   element for many sizes, including odd n (1, 7, 1000003). Exit 3 on failure.
2. `train_dp` compares the replicas at the end. Above 1e-4 (or NaN) it writes no
   results and exits with status 4, so `run_experiments.sh` stops.
3. Loss curves (`--loss-log`) for k GPUs match the 1-GPU and sequential curves
   (same global batch and seed) up to float rounding. Checked automatically:
   `python3 scripts/analyze.py --check-loss results/loss_gpu1.csv results/loss_ring_k2.csv`
   (exit 1 if any curve differs by more than 1e-4).

## Output (same CSV columns everywhere)

```
method,k,msg_bytes,time_ms,busbw_GBs,epoch_s,comm_pct,test_acc
```
`-1` means "not applicable". `busbw = bytes/time × 2(k−1)/k` (same definition as
nccl-tests). `comm_pct` = exposed communication (end of backward to end of
allreduce on GPU 0) as a % of step time. In `bench_p2p` rows (`p2p`, `add`) the
`busbw_GBs` column is the plain bandwidth bytes/time.

Two extra files (not the shared format):
* `--phase-csv` (train_dp): mean ms per step of `data, h2d, fwd, bwd, comm, upd`, the
  step time and the compute imbalance. Each phase is timed with cudaEvents on every GPU
  and the **maximum over GPUs** is kept, because a synchronous step ends with its slowest GPU.
* `--loss-log`: `epoch,loss` per epoch.

## Analysis (`scripts/analyze.py`)

Needs only Python 3 (matplotlib for the figures). From whatever CSVs exist in `results/`:

* **Machine parameters**: fits `t = α + βn` to the `bench_p2p` ping-pong (GPU→GPU copy)
  and `t = α_add + γn` to the `ar_add` kernel. The fit minimizes relative error so both
  small (latency) and large (bandwidth) messages count.
* **Model vs measured**: predicts every allreduce size with the cost models below and
  reports the % error per size. Where measured time is well above the model, look for
  contention (several copies sharing one PCIe link, or staging through host memory when
  P2P is off): that gap is a finding for the report.

  | method | model (n bytes, k GPUs) |
  |---|---|
  | tree (flat) | 2(k−1)(α + nβ) + (k−1)(α_add + nγ) |
  | ring | 2(k−1)(α + nβ/k) + (k−1)(α_add + nγ/k) |
  | rd | ⌈log₂k⌉(α + nβ + α_add + nγ), +2 rounds if k is not a power of 2 |

* **Training**: mean ± 95% CI (Student t) over `REPEATS` runs of epoch time, speedup and
  efficiency vs 1 GPU, exposed comm %, accuracy.
* **Phases**: where the step time goes, per method (stacked bars).
* **Loss check**: every `loss_*.csv` against the 1-GPU curve (seq against 1 GPU).
* Output: `results/report.md` and `results/fig_{allreduce,alpha_beta,phases,speedup,loss}.png`.

## Layout

```
common/     config.h (fixed settings), MNIST loader, params/CLI, timer, CSV
seq/        mlp_seq.c: sequential C MLP (correctness reference)
cuda/       kernels.cu (tiled matmul, softmax, relu, ...), train_dp.cu (training loop)
allreduce/  allreduce.h (interface), common.cu (helpers), registry.cu, one file per method
bench/      bench_allreduce.cu: correctness + latency/bandwidth benchmark
            bench_p2p.cu: GPU-to-GPU ping-pong and ar_add timing (alpha, beta, gamma)
scripts/    get_mnist.sh, run_experiments.sh, analyze.py (report + figures)
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
