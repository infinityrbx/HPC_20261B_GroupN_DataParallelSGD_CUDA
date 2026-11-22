# Final run, Kaggle 2x Tesla T4, 2026-11-22 (kernel hpc-allreduce-final, version 2)

Commit 3fd2b8a, fixed settings (10 epochs, seed 42, global batch 256, lr 0.1, H = 512), 3 repeats per
configuration. `make simcheck`: 5670 runs, 0 failures. `bench_allreduce --check-only`: all sizes
passed for host_ref, tree, ring and rd. Replicas bitwise identical in every 2-GPU run.
Produced by `scripts/summarize_runs.py results/final.log results/clocks.csv`; median over repeats,
min–max in brackets. Raw `final.log`, `clocks.csv` and the 31 loss logs are kept outside the repo.

| Config | Runs | Mean epoch (s) | Steady epoch, 2..E (s) | Speedup vs 1 GPU (steady) | Exposed comm (%) | Test acc | Max loss diff vs 1 GPU | Replicas |
|---|---|---|---|---|---|---|---|---|
| seq (CPU) | 1 | 8.170 | 8.190 | 0.02× | – | 96.16 | 1.0e-05 | – |
| 1 GPU | 3 | 0.200 (0.200–0.210) | 0.199 (0.199–0.200) | 1.00× | 0.5 (0.5–0.5) | 96.16 (96.16–96.16) | 0.0e+00 | 0 |
| host_ref | 3 | 0.423 (0.421–0.428) | 0.420 (0.420–0.420) | 0.47× | 74.9 (73.5–75.0) | 96.14 (96.14–96.14) | 2.0e-05 | 0 |
| host_ref + overlap | 3 | 0.438 (0.433–0.438) | 0.433 (0.431–0.436) | 0.46× | 71.6 (71.6–71.7) | 96.14 (96.14–96.14) | 2.0e-05 | 0 |
| tree | 3 | 0.228 (0.226–0.229) | 0.225 (0.225–0.227) | 0.88× | 50.9 (50.8–51.5) | 96.14 (96.14–96.14) | 2.0e-05 | 0 |
| tree + overlap | 3 | 0.227 (0.227–0.228) | 0.227 (0.226–0.227) | 0.88× | 50.4 (50.0–50.5) | 96.14 (96.14–96.14) | 2.0e-05 | 0 |
| ring | 3 | 0.226 (0.223–0.230) | 0.225 (0.223–0.225) | 0.88× | 49.8 (49.7–50.4) | 96.14 (96.14–96.14) | 2.0e-05 | 0 |
| ring + overlap | 3 | 0.227 (0.227–0.228) | 0.227 (0.226–0.228) | 0.88× | 50.9 (50.6–51.0) | 96.14 (96.14–96.14) | 2.0e-05 | 0 |
| rd | 3 | 0.229 (0.224–0.229) | 0.228 (0.224–0.229) | 0.87× | 50.1 (49.5–50.7) | 96.14 (96.14–96.14) | 2.0e-05 | 0 |
| rd + overlap | 3 | 0.225 (0.224–0.226) | 0.224 (0.224–0.225) | 0.89× | 49.7 (49.7–49.8) | 96.14 (96.14–96.14) | 2.0e-05 | 0 |

GPU clocks under load (samples with power.draw >= 30 W, from `clocks.csv`, 1 sample per second):

| GPU | Samples | SM clock MHz (min / median / max) | Power W (min / median / max) | Temp °C (min / median / max) | clocks_event_reasons.active |
|---|---|---|---|---|---|
| 0 | 105 | 300 / 1590 / 1590 | 32.6 / 48.4 / 72.5 | 54 / 70 / 82 | none ×97, SW power cap (0x4) ×5, idle (0x1) ×3 |
| 1 | 96 | 300 / 1590 / 1590 | 33.7 / 48.0 / 68.3 | 57 / 70 / 80 | none ×93, idle (0x1) ×3 |
