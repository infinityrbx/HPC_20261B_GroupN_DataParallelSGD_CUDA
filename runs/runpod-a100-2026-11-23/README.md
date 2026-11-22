# RunPod 4× A100-SXM4-80GB, 2026-11-23

Pod: Secure Cloud, US-MD-1, $8.76/h. Built from commit efb9b5e plus the build fix in the next commit
(duplicate AR_DISABLE_P2P block removed from `ar_barrier`).
Command: `MODES="p2p host" bash scripts/scaling.sh sm_80`

- `p2p/`: peer access on, GPU-to-GPU copies over NVLink (NV12 on every pair, all-to-all).
- `host/`: `AR_DISABLE_P2P=1`, peer access off, copies staged through host memory over PCIe.

`nvidia-smi topo -m` (GPU rows):

```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV12    NV12    NV12    0-63            0
GPU1    NV12     X      NV12    NV12    64-127          1
GPU2    NV12    NV12     X      NV12    64-127          1
GPU3    NV12    NV12    NV12     X      64-127          1
```

P2P matrix at k=4: all 1 in `p2p`, identity in `host`.
Raw logs: topo.txt, clocks.csv, run.out, and p2p/ and host/ scaling.log and scaling.csv.
