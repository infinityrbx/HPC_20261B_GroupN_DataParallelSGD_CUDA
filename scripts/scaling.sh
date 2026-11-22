#!/usr/bin/env bash
# Scaling test on one multi-GPU machine (k = 1, 2, 4, 8 as available), e.g. a RunPod pod.
#   bash scripts/scaling.sh sm_80        # A100 = sm_80, A40/A6000 = sm_86, L4/4090 = sm_89, H100 = sm_90
#   MODES="p2p host" bash scripts/scaling.sh sm_80   # also rerun with peer access off (AR_DISABLE_P2P=1)
# p2p  = peer access on (NVLink on SXM boards, PCIe P2P otherwise)
# host = peer access off, GPU-to-GPU copies staged through host memory over PCIe
# Writes results/<mode>/{scaling.log,scaling.csv,allreduce_k.csv,scaling_summary.md}
# and results/topo.txt, results/clocks.csv. About 10-15 minutes per mode.
set -uo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
ARCH="${1:-sm_80}"
MODES="${MODES:-p2p}"
EPOCHS=5; REPEATS=3; BATCHES="256 4096"; METHODS="tree ring rd"
mkdir -p results
NGPU=$(nvidia-smi -L | wc -l)
KS=$(for k in 2 4 8; do [ "$k" -le "$NGPU" ] && echo "$k"; done)
{ echo "GPUs: $NGPU, k values: 1 $KS, modes: $MODES"
  git log -1 --format='commit %H %s'
  nvidia-smi; nvidia-smi topo -m; nvidia-smi nvlink --status; nvcc --version; } 2>&1 | tee results/topo.txt
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv -l 1 > results/clocks.csv & CLK=$!
make data && make ARCH="$ARCH" && make simcheck || { kill $CLK; exit 1; }
for mode in $MODES; do
  OUT="results/$mode"; mkdir -p "$OUT"
  if [ "$mode" = host ]; then export AR_DISABLE_P2P=1; else unset AR_DISABLE_P2P; fi
  {
    echo "=== mode $mode (AR_DISABLE_P2P=${AR_DISABLE_P2P:-0}) ==="
    for k in $KS; do
      for m in $METHODS; do
        ./bin/bench_allreduce --method "$m" --gpus "$k" --check-only --max 67108864
        ./bin/bench_allreduce --method "$m" --gpus "$k" --max 268435456 --csv "$OUT/allreduce_k.csv"
      done
    done
    for B in $BATCHES; do
      for r in $(seq "$REPEATS"); do
        ./bin/train_dp --gpus 1 --batch "$B" --epochs "$EPOCHS" --csv "$OUT/scaling.csv"
        for k in $KS; do for m in $METHODS; do
          ./bin/train_dp --gpus "$k" --method "$m" --batch "$B" --epochs "$EPOCHS" --csv "$OUT/scaling.csv"
        done; done
      done
    done
  } 2>&1 | tee "$OUT/scaling.log"
  python3 scripts/summarize_sweep.py "$OUT/scaling.log" > "$OUT/scaling_summary.md"
  cat "$OUT/scaling_summary.md"
done
unset AR_DISABLE_P2P
kill $CLK
python3 scripts/summarize_runs.py /dev/null results/clocks.csv > results/clocks_summary.md
cat results/clocks_summary.md
echo "done -> results/"
