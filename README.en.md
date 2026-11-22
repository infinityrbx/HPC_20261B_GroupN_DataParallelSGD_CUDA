# Data-parallel SGD on CUDA with hand-written allreduce: results

_Tiếng Việt: [README.md](README.md)_

Topic 10 (Multi-GPU Communication Optimization), based on Hegde & Usmani, *Parallel and
Distributed Deep Learning* (Stanford). An MLP is trained on MNIST with synchronous data-parallel
SGD on k GPUs, and the gradients are averaged with hand-written allreduce methods (no NCCL,
cuBLAS or MPI). The paper analyses the communication cost (O(L·k) + O(k·p/B)) only in theory;
this project measures it.

All numbers below come from Kaggle, 2× Tesla T4, on 2026-11-22. Raw logs and CSVs are in
[`runs/kaggle-2026-11-22/`](runs/kaggle-2026-11-22/) (training and allreduce) and
[`runs/kaggle-profile-2026-11-22/`](runs/kaggle-profile-2026-11-22/) (kernel profile).

## Summary

| Question | Answer |
|---|---|
| Are the GPU allreduce methods correct? | Yes. tree, ring and rd pass every check; replicas stay bitwise identical; loss matches 1 GPU to 3.4e-6; 92.70 % test accuracy after 2 epochs. |
| Is a GPU faster than a CPU for training? | Yes. 1 T4: 0.208 s per epoch; sequential C on one CPU core: 7.99 s. About **38×**. |
| Is matrix multiplication the bottleneck? | Yes. 99.7 % of the FLOPs and **94.5–95.5 % of GPU kernel time** (nsys). |
| How much does communication cost? | About **half of every 2-GPU step** (47–50 % exposed communication) with the GPU methods, 68.5 % through the CPU. One allreduce of the 1.63 MB gradient takes 0.36–0.39 ms in the benchmark. |
| Do 2 GPUs beat 1? | No, not at this model size. In steady state (epoch 2) 2 GPUs take 0.231–0.234 s per epoch against 0.208 s on 1 GPU, because communication eats the compute saved by splitting the batch. |
| Which allreduce wins? | tree at 4–8 KB, recursive doubling from 16 KB to 2 MB, ring from 4 MB up. All reach 4.5–4.7 GB/s. |

## Setup

| Item | Value |
|---|---|
| Hardware | Kaggle, 2× Tesla T4 (70 W, sm_75), driver 580.178, CUDA 13.0 |
| Interconnect | PCIe through the host bridge (topology PHB); P2P access on in this run |
| Model | MLP 784–512–10, 407,050 parameters (gradient 1.63 MB fp32) |
| Training | seed 42, global batch 256 (128 per GPU at k = 2), lr 0.1, 2 epochs |
| Benchmark | 4 KB to 64 MB, 5 warm-up + 20 timed runs, median; busbw = bytes/time × 2(k−1)/k |

Methods (one file each in `src/allreduce/`):

| Method | How it works |
|---|---|
| `host_ref` | Copy every gradient to the CPU, average, copy back. Reference for correctness |
| `tree` | The paper's method: reduce to GPU 0, average, broadcast |
| `ring` | Reduce-scatter, then allgather (no chunk pipelining) |
| `rd` | Recursive doubling; extra GPUs are folded in when k is not a power of two |
| `--overlap` | Start the allreduce of layer 2 while layer 1's backward is still running |

## 1. Correctness

| Check | host_ref | tree | ring | rd |
|---|---|---|---|---|
| `make simcheck` (mocked CUDA, random stream orders, k = 2..8) | n/a | pass | pass | pass |
| `bench_allreduce --check-only`, all sizes incl. odd n | pass | pass | pass | pass |
| Max parameter difference between replicas | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| Loss vs 1 GPU, max abs diff (tol 1e-4) | 3.36e-6 | 3.36e-6 | 3.36e-6 | 3.36e-6 |
| Test accuracy, 2 epochs | 92.70 % | 92.70 % | 92.70 % | 92.70 % |

The sequential C version also matches 1 GPU (max loss diff 3.67e-6, 92.70 %).

## 2. GPU vs CPU

| Version | Epoch time | Speedup |
|---|---|---|
| Sequential C, one CPU core (`mlp_seq`) | 7.99 s | 1× |
| 1× T4 (`train_dp --gpus 1`), epoch 2 | 0.208 s | ~38× |

Both reach the same loss (0.55714, then 0.30289) and 92.70 % test accuracy.

## 3. Matrix multiplication is the compute bottleneck

The model is an MLP, but a fully connected layer is one matmul and a convolution becomes one
matmul after im2col, so the same argument covers CNNs.

**FLOP count** (`scripts/flops.py`, H = 512, batch 256): the five matmuls per step are
418.9 MFLOP; everything else (bias + ReLU, softmax, ReLU backward, bias gradients, SGD update)
is 1.4 MFLOP. Matmul is **99.7 % of the FLOPs**, at every hidden size. For comparison, one
3×3 conv layer with 32→64 channels on a 28×28 image is a (784×288)·(288×64) matmul,
28.9 MFLOP per image.

**Measured time** (nsys, one epoch on 1 T4, [`kernel_share.csv`](runs/kaggle-profile-2026-11-22/kernel_share.csv)):

| Kernel | H = 512 | H = 1024 | H = 2048 |
|---|---|---|---|
| `k_matmul_tiled` | **94.5 %** | **95.0 %** | **95.5 %** |
| `k_sgd` | 1.6 % | 2.4 % | 2.5 % |
| `k_colsum` | 1.8 % | 0.9 % | 0.5 % |
| `k_bias_relu` | 0.9 % | 0.8 % | 0.8 % |
| `k_softmax_xent` | 0.8 % | 0.5 % | 0.2 % |
| `k_relu_backward` | 0.4 % | 0.4 % | 0.5 % |
| Total kernel time | 220.7 ms | 368.1 ms | 715.2 ms |

The time share is a little below the FLOP share because the element-wise kernels are
memory-bound: they do almost no arithmetic but still read and write every element. The tiled
matmul does about 106 GFLOP per epoch (training plus test-set forward passes) in 208.5 ms of kernel
time, about 510 GFLOP/s or 6 % of the T4's FP32 peak, so a faster matmul would cut every step's
compute directly.

## 4. Communication cost and speedup on 2 GPUs

`train_dp` reports the epoch time and the exposed communication (end of backward to end of
allreduce on GPU 0) as a share of the step:

| Config | Mean of 2 epochs (s) | Epoch 2 only (s) | Exposed comm |
|---|---|---|---|
| 1 GPU | 0.267 | 0.208 | 0.4 % |
| host_ref | 0.463 | 0.436 | 68.5 % |
| host_ref + overlap | 0.451 | 0.452 | 71.3 % |
| tree | 0.243 | 0.234 | 46.8 % |
| tree + overlap | 0.234 | 0.233 | 49.9 % |
| ring | 0.250 | 0.231 | 50.3 % |
| ring + overlap | 0.234 | 0.234 | 50.3 % |
| rd | 0.251 | 0.233 | 48.4 % |
| rd + overlap | 0.232 | 0.231 | 49.9 % |

What this shows:

- About half of every 2-GPU step is communication. In the benchmark (section 5), one allreduce
  of the model's 1.63 MB gradient takes 0.36–0.39 ms, so communication costs about as much as
  the compute saved by splitting the batch.
- Doing the reduction on the GPUs instead of the CPU halves the epoch time (0.46 s → 0.23–0.25 s)
  and lowers exposed communication from 68.5 % to about 50 %.
- `--overlap` does not lower the exposed-communication share (still about 50 %); a two-layer MLP
  has little backward work left to hide communication behind. Its 2-epoch mean is a little lower,
  but on epoch 2 alone the difference is under 2 %, within noise.
- In steady state (epoch 2), 2 GPUs are about 10 % slower than 1 GPU. The 2-epoch mean shows
  1.06–1.15× instead, but only because the 1-GPU run's first epoch includes warm-up (0.326 s,
  then 0.208 s).

## 5. Raw allreduce: which method wins at which size

Median time per allreduce, k = 2, ms (`allreduce.csv`); bold is fastest:

| Size | tree | ring | rd |
|---|---|---|---|
| 4 KB | **0.031** | 0.053 | 0.034 |
| 8 KB | **0.031** | 0.054 | 0.035 |
| 16 KB | 0.041 | 0.053 | **0.035** |
| 64 KB | 0.053 | 0.060 | **0.037** |
| 256 KB | 0.101 | 0.100 | **0.076** |
| 1 MB | 0.261 | 0.258 | **0.238** |
| 2 MB | 0.486 | 0.481 | **0.456** |
| 4 MB | 0.971 | **0.927** | 0.955 |
| 16 MB | 3.776 | **3.608** | 3.750 |
| 64 MB | 15.009 | **14.301** | 14.981 |

Bus bandwidth at 64 MB: tree 4.47, ring 4.69, rd 4.48 GB/s. A one-way peer copy reaches 9.9 GB/s.

- **Small messages are latency-bound.** Ring has a floor of about 0.053 ms because it runs more
  dependent steps; tree and rd start at about 0.031–0.035 ms.
- **Medium messages (16 KB–2 MB) favour recursive doubling**, by up to 30 % at 64 KB and 5–10 % at
  1–2 MB. The model's 1.63 MB gradient is in this range.
- **Large messages are bandwidth-bound and favour ring**, but only by 4–5 %.
- **All three reach about half the link rate.** Tree has to send the data one way and back, so
  half is its limit. Ring and rd could use both directions at once but do not get further, which
  suggests the two directions do not overlap and ring would need chunk pipelining.
- With k = 2, the three algorithms move the same amount of data. Their textbook differences only
  show at k ≥ 4.

## Limitations

- **One run, 2 epochs.** Differences between tree, ring and rd in training (under 4 % on epoch
  time) are within run-to-run noise. Final numbers should use the default 10 epochs, report epoch
  1 separately, and repeat each run at least 3 times.
- **Only 2 GPUs.** Kaggle offers at most 2× T4, so the O(L·k) + O(k·p/B) scaling in k cannot be
  tested.
- **P2P access varies between Kaggle machines.** This run had peer access on. An earlier Kaggle
  machine had none (P2P matrix all 0); there, peer copies are staged through host memory and the
  numbers are not comparable (1 GPU 0.24 s/epoch, `host_ref` 0.41 s/epoch with 64 %
  communication). Always record the P2P matrix with the results.
- **T4 power throttling.** The T4 is a 70 W passively cooled card that lowers its clocks under
  sustained load. Clocks were not logged during the runs, so throttling cannot be ruled out.
- **Code version.** The 2-GPU training log was built from the `ltnaphixay/allreduce-cuda` branch of
  `infinityrbx/High-Performance-Computing`. Only numbers that this repository's code also
  produces (epoch time, exposed communication, accuracy, allreduce benchmark) are reported here.

## Build and run

```bash
make data          # download MNIST into ./data
make seq           # sequential C baseline (gcc only)
make ARCH=sm_75    # everything; sm_75 = T4, sm_80 = A100
make simcheck      # check tree/ring/rd under random stream orders (g++ only, no GPU)

./bin/mlp_seq                                     # sequential baseline (CPU)
./bin/train_dp --gpus 1                           # single GPU
./bin/train_dp --gpus 2 --method ring [--overlap] # data-parallel
./bin/bench_allreduce --method ring --gpus 2 [--check-only]
bash scripts/run_experiments.sh 2 "tree ring rd"  # the full comparison
python3 scripts/flops.py --hidden 512             # FLOP count, matmul vs the rest
```

On Kaggle (set your username in `kernel-metadata.json` first):

```bash
kaggle kernels push -p scripts/kaggle/check --accelerator NvidiaTeslaT4    # build, check, train, benchmark
kaggle kernels push -p scripts/kaggle/profile --accelerator NvidiaTeslaT4  # nsys kernel-time profile
```

Fixed settings live in `src/common/config.h`. CSV columns everywhere:
`method,k,msg_bytes,time_ms,busbw_GBs,epoch_s,comm_pct,test_acc` (`-1` = not applicable;
`comm_pct` = exposed communication as a % of step time).

## Layout

```
src/
  common/     config.h (fixed settings), MNIST loader, params/CLI, timer, CSV
  seq/        mlp_seq.c: sequential C MLP (correctness reference)
  cuda/       kernels.cu (tiled matmul, softmax, relu, ...), train_dp.cu (training loop)
  allreduce/  allreduce.h (interface), ar_sync.h, common.cu, registry.cu, one file per method
bench/        bench_allreduce.cu: correctness + latency/bandwidth benchmark
tests/        sim_allreduce.cpp + mock CUDA runtime (make simcheck)
scripts/      run_experiments.sh, flops.py, kernel_share.py, get_mnist.sh
  kaggle/     check/ (build, check, train, benchmark), profile/ (nsys kernel profile)
runs/         raw Kaggle logs and CSVs
```
