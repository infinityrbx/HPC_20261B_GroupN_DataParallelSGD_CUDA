# Kernel-time profile on Kaggle: what share of GPU time is matrix multiplication?
# Push with:  kaggle kernels push -p scripts/kaggle/profile --accelerator NvidiaTeslaT4
# (set your username in kernel-metadata.json first).
# Profiles 1-GPU training for one epoch at H = 512 (the fixed setting), 1024 and 2048,
# with nsys if it is installed, otherwise ncu. Prints a per-kernel table and the
# matmul share, and writes kernel_share.csv.
import os, shutil, subprocess

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    print(f"[exit {r.returncode}]", flush=True)
    return r.returncode

sh("nvidia-smi; nvcc --version")
REPO_URL = "https://github.com/infinityrbx/HPC_20261B_GroupN_DataParallelSGD_CUDA"
BRANCH = "main"
sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} repo")
os.chdir("repo")
sh("make data && make ARCH=sm_75")

def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    for root in ("/usr/local/cuda/bin", "/opt/nvidia"):
        for d, _, files in os.walk(root):
            if name in files:
                return os.path.join(d, name)
    return None

nsys, ncu = find_tool("nsys"), find_tool("ncu")
print(f"nsys: {nsys}\nncu:  {ncu}", flush=True)
if not nsys and not ncu:
    raise SystemExit("Neither nsys nor ncu found; cannot profile.")

for H in (512, 1024, 2048):
    run = f"./bin/train_dp --gpus 1 --epochs 1 --hidden {H}"
    if nsys:
        sh(f"{nsys} profile -o prof_h{H} --force-overwrite true -t cuda {run}")
        sh(f"{nsys} stats -r cuda_gpu_kern_sum -f csv -o prof_h{H} prof_h{H}.nsys-rep")
        sh(f"python3 scripts/kernel_share.py --nsys prof_h{H}_cuda_gpu_kern_sum.csv --hidden {H} --csv kernel_share.csv")
    else:
        # ncu serialises kernels, so it is slow; 3000 launches covers about one epoch.
        sh(f"{ncu} --csv --metrics gpu__time_duration.sum -c 3000 {run} > ncu_h{H}.csv")
        sh(f"python3 scripts/kernel_share.py --ncu ncu_h{H}.csv --hidden {H} --csv kernel_share.csv")

sh("python3 scripts/flops.py --hidden 512 1024 2048")
sh("cat kernel_share.csv")
