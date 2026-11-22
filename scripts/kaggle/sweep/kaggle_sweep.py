# Batch-size sweep on Kaggle 2x T4: where does data parallelism start to pay off?
# The gradient is always 1.63 MB, so communication per step is fixed, while compute per step
# grows with the batch. Runs 1 GPU and tree/ring/rd on 2 GPUs at several global batch sizes.
# Push with:  kaggle kernels push -p scripts/kaggle/sweep --accelerator NvidiaTeslaT4
import os, subprocess

BATCHES = (256, 1024, 2048, 4096, 8192)
METHODS = ("tree", "ring", "rd")
EPOCHS, REPEATS = 5, 3
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
LOG = open("results/sweep.log", "w")

sh("git log -1 --format='commit %H %s'")
sh("nvidia-smi; nvidia-smi topo -m")
clocks = subprocess.Popen(
    "nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
    "clocks_throttle_reasons.active --format=csv -l 1 > results/clocks.csv", shell=True)
sh("make data && make ARCH=sm_75")
for B in BATCHES:
    for r in range(REPEATS):
        sh(f"./bin/train_dp --gpus 1 --batch {B} --epochs {EPOCHS} --csv results/sweep.csv")
        for m in METHODS:
            sh(f"./bin/train_dp --gpus 2 --method {m} --batch {B} --epochs {EPOCHS} --csv results/sweep.csv")
clocks.terminate()
LOG.close()
subprocess.run("python3 scripts/summarize_sweep.py results/sweep.log > results/sweep_summary.md; "
               "python3 scripts/summarize_runs.py /dev/null results/clocks.csv >> results/sweep_summary.md; "
               "cat results/sweep_summary.md", shell=True)
