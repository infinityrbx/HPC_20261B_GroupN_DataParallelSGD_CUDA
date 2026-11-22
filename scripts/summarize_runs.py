#!/usr/bin/env python3
"""Summarize a log of mlp_seq / train_dp runs (e.g. results/final.log) as Markdown.

  summarize_runs.py final.log [clocks.csv]

For each configuration (method, k, overlap) over its repeats:
  mean epoch   = mean of all epochs (includes the first, warm-up epoch)
  steady epoch = median of epochs 2..E
  comm %, test accuracy, final loss, and the max loss difference to the 1-GPU run.
Reported values are the median over repeats, with min-max in brackets.
"""
import csv, re, statistics as st, sys
from collections import defaultdict

runs, cur = [], None
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("$ ./bin/mlp_seq"):
        cur = {"cfg": "seq (CPU)", "ep": [], "loss": []}; runs.append(cur); continue
    m = re.match(r"method=(\S+) k=(\d+) overlap=(\d)", line)
    if m:
        name, k, ov = m.group(1), int(m.group(2)), m.group(3) == "1"
        cfg = "1 GPU" if k == 1 else f"{name}{' + overlap' if ov else ''}"
        cur = {"cfg": cfg, "ep": [], "loss": []}; runs.append(cur); continue
    if cur is None:
        continue
    m = re.match(r"epoch (\d+)\s+loss ([\d.]+)\s+time ([\d.]+) s", line)
    if m:
        cur["ep"].append(float(m.group(3))); cur["loss"].append(float(m.group(2))); continue
    m = re.search(r"test accuracy ([\d.]+)(?:\s+exposed comm ([\d.]+)%)?", line)
    if m:
        cur["acc"] = float(m.group(1))
        if m.group(2): cur["comm"] = float(m.group(2))
    m = re.search(r"max parameter difference between replicas: (\S+)", line)
    if m: cur["repl"] = m.group(1)

by = defaultdict(list)
for r in runs:
    if r["ep"]:
        by[r["cfg"]].append(r)

def fmt(vals, f="{:.3f}"):
    if not vals: return "–"
    med = st.median(vals)
    return f.format(med) if len(vals) == 1 else f"{f.format(med)} ({f.format(min(vals))}–{f.format(max(vals))})"

base_loss = by["1 GPU"][0]["loss"] if "1 GPU" in by else None
steady1 = st.median([st.median(r["ep"][1:]) for r in by.get("1 GPU", [])]) if "1 GPU" in by else None

print("| Config | Runs | Mean epoch (s) | Steady epoch, 2..E (s) | Speedup vs 1 GPU (steady) | Exposed comm (%) | Test acc | Max loss diff vs 1 GPU | Replicas |")
print("|---|---|---|---|---|---|---|---|---|")
for cfg, rs in by.items():
    mean_ep = [sum(r["ep"]) / len(r["ep"]) for r in rs]
    steady = [st.median(r["ep"][1:]) for r in rs if len(r["ep"]) > 1]
    sp = f"{steady1 / st.median(steady):.2f}×" if steady1 and steady else "–"
    comm = [r["comm"] for r in rs if "comm" in r]
    acc = [100 * r["acc"] for r in rs if "acc" in r]
    diff = "–"
    if base_loss:
        d = [max(abs(a - b) for a, b in zip(r["loss"], base_loss)) for r in rs]
        diff = f"{max(d):.1e}"
    repl = ", ".join(sorted({r.get("repl", "–") for r in rs}))
    print(f"| {cfg} | {len(rs)} | {fmt(mean_ep)} | {fmt(steady)} | {sp} | {fmt(comm, '{:.1f}')} | {fmt(acc, '{:.2f}')} | {diff} | {repl} |")

if len(sys.argv) > 2:
    sm, pw, tp, thr = [], [], [], defaultdict(int)
    try:
        for row in csv.DictReader(open(sys.argv[2]), skipinitialspace=True):
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            try:
                sm.append(float(row["clocks.current.sm [MHz]"].split()[0]))
                pw.append(float(row["power.draw [W]"].split()[0]))
                tp.append(float(row["temperature.gpu"]))
            except (KeyError, ValueError, IndexError):
                pass
            for k, v in row.items():
                if k.startswith("clocks_throttle_reasons.active"):
                    thr[v] += 1
    except FileNotFoundError:
        pass
    if sm:
        print(f"\nGPU clocks during the run ({len(sm)} samples): SM {min(sm):.0f}–{max(sm):.0f} MHz "
              f"(median {st.median(sm):.0f}), power {min(pw):.0f}–{max(pw):.0f} W, "
              f"temperature {min(tp):.0f}–{max(tp):.0f} °C.")
        print("Throttle-reason bitmask counts: " + ", ".join(f"{k}: {v}" for k, v in sorted(thr.items())))
