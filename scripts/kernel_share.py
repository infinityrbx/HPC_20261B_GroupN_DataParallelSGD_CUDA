#!/usr/bin/env python3
"""Sum GPU kernel time by kernel from an nsys or ncu CSV and report the matmul share.

  kernel_share.py --nsys prof_cuda_gpu_kern_sum.csv --hidden 512 [--csv out.csv]
  kernel_share.py --ncu  ncu.csv                   --hidden 512 [--csv out.csv]
"""
import argparse, csv, io, os, re
from collections import defaultdict

def short(name):
    m = re.search(r"(k_\w+)", name)
    return m.group(1) if m else name.split("(")[0][:40]

def read_rows(path):
    # nsys/ncu may print banner lines before the CSV header.
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"') or "Name" in l), 0)
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))

def from_nsys(path):
    t = defaultdict(float); n = defaultdict(int)
    for r in read_rows(path):
        name = r.get("Name", "")
        if not name:
            continue
        t[short(name)] += float(r["Total Time (ns)"]) / 1e6
        n[short(name)] += int(r.get("Instances", 0) or 0)
    return t, n

def from_ncu(path):
    t = defaultdict(float); n = defaultdict(int)
    for r in read_rows(path):
        if r.get("Metric Name") != "gpu__time_duration.sum":
            continue
        v = float(r["Metric Value"].replace(",", ""))
        unit = r.get("Metric Unit", "nsecond")
        scale = {"nsecond": 1e-6, "usecond": 1e-3, "msecond": 1.0, "second": 1e3}.get(unit, 1e-6)
        k = short(r["Kernel Name"])
        t[k] += v * scale; n[k] += 1
    return t, n

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--nsys"); g.add_argument("--ncu")
ap.add_argument("--hidden", type=int, required=True)
ap.add_argument("--csv", help="append one summary row per kernel to this CSV")
a = ap.parse_args()

t, n = from_nsys(a.nsys) if a.nsys else from_ncu(a.ncu)
total = sum(t.values())
mm = sum(v for k, v in t.items() if "matmul" in k)
print(f"H = {a.hidden}: total kernel time {total:.2f} ms")
print(f"{'kernel':<22}{'calls':>8}{'ms':>10}{'share':>9}")
for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
    print(f"{k:<22}{n[k]:>8}{v:>10.2f}{100*v/total:>8.1f}%")
print(f"MATMUL SHARE (H = {a.hidden}): {100*mm/total:.1f}% of GPU kernel time")

if a.csv:
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["hidden", "kernel", "calls", "time_ms", "share_pct"])
        for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
            w.writerow([a.hidden, k, n[k], f"{v:.4f}", f"{100*v/total:.2f}"])
