# First GPU check on Kaggle: build, correctness test, short training runs.
# Push with:  kaggle kernels push -p scripts/kaggle/check   (after editing kernel-metadata.json)
import os, subprocess

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    print(f"[exit {r.returncode}]", flush=True)
    return r.returncode

sh("nvidia-smi; nvidia-smi topo -m; nvcc --version")
# Repo and branch the Kaggle job builds from.
REPO_URL = "https://github.com/infinityrbx/HPC_20261B_GroupN_DataParallelSGD_CUDA"
BRANCH = "main"
sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} repo")
os.chdir("repo")
sh("make data && make ARCH=sm_75")
sh("./bin/mlp_seq --epochs 2 --loss-log loss_seq.csv")
sh("./bin/train_dp --gpus 1 --epochs 2 --loss-log loss_gpu1.csv")
ngpu = int(subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
print(f"GPUs visible: {ngpu}", flush=True)
if ngpu >= 2:
    sh("./bin/bench_allreduce --method host_ref --gpus 2 --check-only --max 67108864")
    sh("./bin/train_dp --gpus 2 --method host_ref --epochs 2 --loss-log loss_k2.csv")
    sh("./bin/train_dp --gpus 2 --method host_ref --overlap --epochs 2")
    for m in ("tree", "ring", "rd"):
        sh(f"./bin/bench_allreduce --method {m} --gpus 2 --check-only --max 67108864")
        sh(f"./bin/train_dp --gpus 2 --method {m} --epochs 2 --loss-log loss_{m}_k2.csv")
        sh(f"./bin/train_dp --gpus 2 --method {m} --overlap --epochs 2")
        sh(f"./bin/bench_allreduce --method {m} --gpus 2 --max 67108864 --csv allreduce.csv")
print("\n== per-epoch loss: compare by eye with loss_gpu1.csv ==", flush=True)
sh("for f in loss_seq.csv loss_gpu1.csv loss_k2.csv loss_tree_k2.csv loss_ring_k2.csv loss_rd_k2.csv; do echo $f; cat $f; done")
sh("cat allreduce.csv 2>/dev/null || true")
