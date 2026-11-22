# Batch-size sweep, Kaggle 2x Tesla T4, 2026-11-23 (kernel hpc-allreduce-sweep, version 1)

Commit fbfbf96. H = 512, lr 0.1, 5 epochs, 3 repeats per point; P2P access on (matrix all 1).
Rows in `sweep.csv` are in run order: for each batch size 256, 1024, 2048, 4096, 8192, three
rounds of (gpu1, tree, ring, rd). Every 2-GPU run ended with bitwise-identical replicas.
Produced by `scripts/summarize_sweep.py results/sweep.log`; steady epoch = median of epochs 2..5,
median over repeats (min–max in brackets). Raw `sweep.log` and `clocks.csv` are in `raw/`.

| Global batch | Steps/epoch | Config | Steady epoch (s) | Time per step (ms) | Speedup vs 1 GPU | Exposed comm (%) | Test acc (%) |
|---|---|---|---|---|---|---|---|
| 256 | 234 | 1 GPU | 0.199 (0.199–0.200) | 0.85 | 1.00× | 0.5 | 94.68 |
| 256 | 234 | rd | 0.226 (0.225–0.227) | 0.97 | 0.88× | 49.9 | 94.70 |
| 256 | 234 | ring | 0.225 (0.224–0.226) | 0.96 | 0.88× | 49.7 | 94.70 |
| 256 | 234 | tree | 0.228 (0.228–0.232) | 0.97 | 0.87× | 54.4 | 94.70 |
| 1024 | 58 | 1 GPU | 0.190 (0.189–0.192) | 3.27 | 1.00× | 0.1 | 91.71 |
| 1024 | 58 | rd | 0.141 (0.141–0.142) | 2.43 | 1.34× | 22.8 | 91.71 |
| 1024 | 58 | ring | 0.142 (0.140–0.143) | 2.45 | 1.33× | 22.2 | 91.71 |
| 1024 | 58 | tree | 0.143 (0.143–0.148) | 2.47 | 1.33× | 35.7 | 91.71 |
| 2048 | 29 | 1 GPU | 0.196 (0.194–0.196) | 6.76 | 1.00× | 0.1 | 90.30 |
| 2048 | 29 | rd | 0.129 (0.129–0.129) | 4.43 | 1.53× | 13.3 | 90.30 |
| 2048 | 29 | ring | 0.128 (0.127–0.128) | 4.41 | 1.53× | 12.9 | 90.30 |
| 2048 | 29 | tree | 0.131 (0.129–0.133) | 4.52 | 1.50× | 27.3 | 90.30 |
| 4096 | 14 | 1 GPU | 0.198 (0.197–0.198) | 14.11 | 1.00× | 0.0 | 87.93 |
| 4096 | 14 | rd | 0.123 (0.121–0.124) | 8.82 | 1.60× | 7.0 | 87.93 |
| 4096 | 14 | ring | 0.124 (0.120–0.124) | 8.89 | 1.59× | 6.6 | 87.93 |
| 4096 | 14 | tree | 0.128 (0.124–0.129) | 9.11 | 1.55× | 22.4 | 87.93 |
| 8192 | 7 | 1 GPU | 0.205 (0.204–0.207) | 29.29 | 1.00× | 0.0 | 85.16 |
| 8192 | 7 | rd | 0.129 (0.128–0.132) | 18.43 | 1.59× | 4.6 | 85.16 |
| 8192 | 7 | ring | 0.131 (0.131–0.133) | 18.64 | 1.57× | 4.6 | 85.16 |
| 8192 | 7 | tree | 0.132 (0.131–0.136) | 18.86 | 1.55× | 20.9 | 85.16 |

GPU clocks under load (power >= 30 W):

| GPU | Samples | SM MHz (min / median / max) | Power W (min / median / max) | Temp °C (min / median / max) | Clock event reasons |
|---|---|---|---|---|---|
| 0 | 91 | 1290 / 1590 / 1590 | 36.2 / 52.9 / 75.0 | 53 / 65 / 73 | none ×60, SW power cap ×31 |
| 1 | 85 | 585 / 1590 / 1590 | 30.2 / 39.5 / 78.2 | 51 / 61 / 68 | none ×70, SW power cap ×8, idle ×7 |
