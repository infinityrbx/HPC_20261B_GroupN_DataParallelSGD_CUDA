#!/usr/bin/env bash
# Run the FINAL comparison with the fixed settings. Same machine for everyone.
#   bash scripts/run_experiments.sh 2 "tree ring rd"      # k=2, three methods
#   REPEATS=5 bash scripts/run_experiments.sh 2           # default methods, 5 training runs each
# Results are appended to results/*.csv, then scripts/analyze.py writes
# results/report.md (mean +- 95% CI, speedup, alpha-beta model vs measured,
# per-phase breakdown, loss-curve check) and results/fig_*.png.
#
# A method that is not implemented yet (exit 2) is skipped. A method that fails a
# correctness check (bench exit 3, replica mismatch exit 4) stops the script:
# numbers from a wrong method must not end up in the report.
set -euo pipefail
K="${1:-2}"
METHODS="${2:-host_ref tree ring rd}"
REPEATS="${REPEATS:-5}"          # training runs per configuration
R=results
mkdir -p "$R"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee "$R/gpus_k${K}.txt"
nvidia-smi topo -m >> "$R/gpus_k${K}.txt" 2>/dev/null || true

# run CMD...: exit 2 (not implemented) -> return 2; any other failure -> stop the script.
run() {
  local rc=0
  "$@" || rc=$?
  if [ "$rc" -eq 2 ]; then return 2; fi
  if [ "$rc" -ne 0 ]; then echo "FAILED (exit $rc): $*" >&2; exit "$rc"; fi
}

# Experiment A: machine parameters (alpha, beta, gamma) for the cost models.
./bin/bench_p2p --csv "$R/p2p.csv"

# Baselines (the sequential run is slow and fully deterministic: once is enough)
./bin/mlp_seq --csv "$R/training.csv" --loss-log "$R/loss_seq.csv"
for r in $(seq "$REPEATS"); do
  run ./bin/train_dp --gpus 1 --csv "$R/training.csv" --phase-csv "$R/phases.csv" --loss-log "$R/loss_gpu1.csv"
done

for m in $METHODS; do
  # Experiment B: raw allreduce (correctness + latency/bandwidth vs message size)
  if ! run ./bin/bench_allreduce --method "$m" --gpus "$K" --csv "$R/allreduce.csv"; then
    echo "== skipping '$m': not implemented yet" >&2
    continue
  fi
  # Experiment C: training.  plain   = one flat allreduce after backward
  #                          layer   = per-layer buckets after backward (no overlap)
  #                          overlap = per-layer buckets overlapped with backward
  # layer vs overlap separates the effect of bucketing from the effect of overlap.
  for r in $(seq "$REPEATS"); do
    run ./bin/train_dp --gpus "$K" --method "$m" --csv "$R/training.csv" --phase-csv "$R/phases.csv" \
                       --loss-log "$R/loss_${m}_k${K}.csv"
    run ./bin/train_dp --gpus "$K" --method "$m" --per-layer --csv "$R/training.csv" --phase-csv "$R/phases.csv" \
                       --loss-log "$R/loss_${m}_layer_k${K}.csv"
    run ./bin/train_dp --gpus "$K" --method "$m" --overlap --csv "$R/training.csv" --phase-csv "$R/phases.csv" \
                       --loss-log "$R/loss_${m}_overlap_k${K}.csv"
  done
done

# Tables, model check, loss-curve check, figures. Exits 1 if a loss curve differs.
python3 scripts/analyze.py "$R"
echo "done -> $R/report.md"
