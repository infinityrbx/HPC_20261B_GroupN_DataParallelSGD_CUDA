#!/usr/bin/env bash
# Run the FINAL comparison with the fixed settings. Use the same machine for every method.
#   bash scripts/run_experiments.sh 2 "tree ring rd"      # k=2, three methods
# Results are appended to results/allreduce.csv and results/training.csv.
set -euo pipefail
K="${1:-2}"
METHODS="${2:-host_ref tree ring rd}"
REPEATS="${REPEATS:-5}"          # training runs per configuration (report the median)
mkdir -p results
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee results/gpus_k${K}.txt
nvidia-smi topo -m >> results/gpus_k${K}.txt 2>/dev/null || true

# Baselines
./bin/mlp_seq --csv results/training.csv --loss-log results/loss_seq.csv
./bin/train_dp --gpus 1 --csv results/training.csv --loss-log results/loss_gpu1.csv

for m in $METHODS; do
  # Experiment B: raw allreduce (correctness + latency/bandwidth vs message size)
  ./bin/bench_allreduce --method "$m" --gpus "$K" --csv results/allreduce.csv
  # Experiment C: training, plain and overlapped
  for r in $(seq "$REPEATS"); do
    ./bin/train_dp --gpus "$K" --method "$m" --csv results/training.csv \
                   --loss-log "results/loss_${m}_k${K}.csv"
    ./bin/train_dp --gpus "$K" --method "$m" --overlap --csv results/training.csv \
                   --loss-log "results/loss_${m}_overlap_k${K}.csv"
  done
done
echo "done -> results/"
