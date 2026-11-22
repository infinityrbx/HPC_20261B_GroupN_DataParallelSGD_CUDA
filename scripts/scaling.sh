#!/usr/bin/env bash
# Scaling test on one multi-GPU machine (k = 1, 2, 4, 8 as available), e.g. a RunPod pod.
#   bash scripts/scaling.sh sm_80        # A100 = sm_80, A40/A6000 = sm_86, L4/4090 = sm_89, H100 = sm_90
# Writes results/scaling.log, results/scaling.csv (training), results/allreduce_k.csv (benchmark),
# results/scaling_summary.md. Takes about 10-15 minutes.
set -uo pipefail
ARCH="${1:-sm_80}"
EPOCHS=5; REPEATS=3; BATCHES="256 4096"; METHODS="tree ring rd"
mkdir -p results
exec > >(tee results/scaling.log) 2>&1
NGPU=$(nvidia-smi -L | wc -l)
KS=$(for k in 2 4 8; do [ "$k" -le "$NGPU" ] && echo "$k"; done)
echo "GPUs: $NGPU, k values: 1 $KS"
git log -1 --format='commit %H %s'
nvidia-smi; nvidia-smi topo -m; nvcc --version
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv -l 1 > results/clocks.csv & CLK=$!
make data && make ARCH="$ARCH" && make simcheck
for k in $KS; do
  for m in $METHODS; do
    ./bin/bench_allreduce --method "$m" --gpus "$k" --check-only --max 67108864
    ./bin/bench_allreduce --method "$m" --gpus "$k" --max 268435456 --csv results/allreduce_k.csv
  done
done
for B in $BATCHES; do
  for r in $(seq "$REPEATS"); do
    ./bin/train_dp --gpus 1 --batch "$B" --epochs "$EPOCHS" --csv results/scaling.csv
    for k in $KS; do for m in $METHODS; do
      ./bin/train_dp --gpus "$k" --method "$m" --batch "$B" --epochs "$EPOCHS" --csv results/scaling.csv
    done; done
  done
done
kill $CLK
python3 scripts/summarize_sweep.py results/scaling.log > results/scaling_summary.md
python3 scripts/summarize_runs.py /dev/null results/clocks.csv >> results/scaling_summary.md
cat results/scaling_summary.md
echo "done -> results/"
