# First GPU check on Kaggle: build, correctness test, short training runs.
# Push with:  kaggle kernels push -p scripts/kaggle   (after editing kernel-metadata.json)
import os, subprocess

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    print(f"[exit {r.returncode}]", flush=True)
    return r.returncode

sh("nvidia-smi; nvidia-smi topo -m; nvcc --version")
sh("git clone --depth 1 https://github.com/infinityrbx/HPC_20261B_GroupN_DataParallelSGD_CUDA repo")
os.chdir("repo")
sh("make data && make ARCH=sm_75")
sh("./bin/mlp_seq --epochs 2 --loss-log loss_seq.csv")
sh("./bin/train_dp --gpus 1 --epochs 2 --loss-log loss_gpu1.csv --phase-csv phases.csv")
ngpu = int(subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
print(f"GPUs visible: {ngpu}", flush=True)
if ngpu >= 2:
    sh("./bin/bench_p2p --max 67108864 --csv p2p.csv")
    sh("./bin/bench_allreduce --method host_ref --gpus 2 --check-only --max 67108864")
    sh("./bin/train_dp --gpus 2 --method host_ref --epochs 2 --loss-log loss_k2.csv --phase-csv phases.csv")
    sh("./bin/train_dp --gpus 2 --method host_ref --per-layer --epochs 2 --phase-csv phases.csv")
    sh("./bin/train_dp --gpus 2 --method host_ref --overlap --epochs 2 --phase-csv phases.csv")
    sh("./bin/bench_allreduce --method tree --gpus 2 --check-only --max 4096 || true")
print("\n== per-epoch loss: seq, gpu1, k2 ==", flush=True)
sh("for f in loss_seq.csv loss_gpu1.csv loss_k2.csv; do echo $f; cat $f; done")
sh("python3 scripts/analyze.py --check-loss loss_gpu1.csv loss_seq.csv loss_k2.csv")
sh("cat phases.csv p2p.csv 2>/dev/null || true")
