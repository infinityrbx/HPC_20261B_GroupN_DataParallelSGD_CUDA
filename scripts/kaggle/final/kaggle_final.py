# Final comparison on Kaggle 2x T4 with the fixed settings (10 epochs) and repeats.
# Push with:  kaggle kernels push -p scripts/kaggle/final --accelerator NvidiaTeslaT4
# (set your username in kernel-metadata.json first).
# Output: results/ with training.csv, allreduce.csv, loss logs, clocks.csv, final.log and summary.md.
import os, subprocess

REPEATS = 3
METHODS = ("host_ref", "tree", "ring", "rd")
LOG = None

def sh(cmd):
    line = f"\n$ {cmd}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for out in p.stdout:
        print(out, end="", flush=True); LOG.write(out); LOG.flush()
    rc = p.wait()
    print(f"[exit {rc}]", flush=True); LOG.write(f"[exit {rc}]\n"); LOG.flush()
    return rc

REPO_URL = "https://github.com/infinityrbx/HPC_20261B_GroupN_DataParallelSGD_CUDA"
subprocess.run(f"git clone --depth 1 -b main {REPO_URL} repo", shell=True, check=True)
os.chdir("repo")
os.makedirs("results", exist_ok=True)
LOG = open("results/final.log", "w")

sh("git log -1 --format='commit %H %s'")
sh("nvidia-smi; nvidia-smi topo -m; nvcc --version")
# Log clocks, power, temperature and throttle reasons once per second for the whole run.
clocks = subprocess.Popen(
    "nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
    "clocks_throttle_reasons.active --format=csv -l 1 > results/clocks.csv", shell=True)

sh("make data && make ARCH=sm_75 && make simcheck")
sh("./bin/mlp_seq --csv results/training.csv --loss-log results/loss_seq.csv")
for r in range(1, REPEATS + 1):
    sh(f"./bin/train_dp --gpus 1 --csv results/training.csv --loss-log results/loss_gpu1_r{r}.csv")
for m in METHODS:
    sh(f"./bin/bench_allreduce --method {m} --gpus 2 --check-only --max 67108864")
    for r in range(1, REPEATS + 1):
        sh(f"./bin/train_dp --gpus 2 --method {m} --csv results/training.csv --loss-log results/loss_{m}_r{r}.csv")
        sh(f"./bin/train_dp --gpus 2 --method {m} --overlap --csv results/training.csv --loss-log results/loss_{m}_overlap_r{r}.csv")
    sh(f"./bin/bench_allreduce --method {m} --gpus 2 --max 67108864 --csv results/allreduce.csv")

clocks.terminate()
LOG.close()
subprocess.run("python3 scripts/summarize_runs.py results/final.log results/clocks.csv > results/summary.md", shell=True)
subprocess.run("cat results/summary.md; cat results/allreduce.csv", shell=True)
