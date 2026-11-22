#!/usr/bin/env python3
"""Summarize a batch-size sweep log (results/sweep.log from scripts/kaggle/sweep) as Markdown.

For each global batch B and configuration: steady epoch time = median of epochs 2..E, median over
repeats (min-max in brackets), time per step, speedup vs 1 GPU at the same B, exposed comm %, accuracy.
"""
import re, statistics as st, sys
from collections import defaultdict

runs, cur = [], None
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"method=(\S+) k=(\d+) overlap=(\d).*local_batch=(\d+)", line)
    if m:
        name, k, ov, bl = m.group(1), int(m.group(2)), m.group(3) == "1", int(m.group(4))
        cfg = "1 GPU" if k == 1 else f"{name}{' + overlap' if ov else ''} (k={k})"
        cur = {"cfg": cfg, "B": bl * k, "ep": []}; runs.append(cur); continue
    if cur is None:
        continue
    m = re.match(r"epoch (\d+)\s+loss [\d.]+\s+time ([\d.]+) s", line)
    if m:
        cur["ep"].append(float(m.group(2))); continue
    m = re.search(r"test accuracy ([\d.]+)\s+exposed comm ([\d.]+)%", line)
    if m:
        cur["acc"], cur["comm"] = float(m.group(1)), float(m.group(2))

by = defaultdict(list)
for r in runs:
    if len(r["ep"]) > 1:
        by[(r["B"], r["cfg"])].append(st.median(r["ep"][1:]))
info = defaultdict(dict)
for r in runs:
    if "comm" in r:
        info[(r["B"], r["cfg"])].setdefault("comm", []).append(r["comm"])
        info[(r["B"], r["cfg"])].setdefault("acc", []).append(100 * r["acc"])

print("| Global batch | Steps/epoch | Config | Steady epoch (s) | Time per step (ms) | Speedup vs 1 GPU | Exposed comm (%) | Test acc (%) |")
print("|---|---|---|---|---|---|---|---|")
for B in sorted({b for b, _ in by}):
    steps = 60000 // B
    base = st.median(by[(B, "1 GPU")]) if (B, "1 GPU") in by else None
    for (b, cfg), v in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] != "1 GPU", kv[0][1])):
        if b != B:
            continue
        med = st.median(v)
        rng = f" ({min(v):.3f}–{max(v):.3f})" if len(v) > 1 else ""
        sp = f"{base / med:.2f}×" if base else "–"
        comm = st.median(info[(b, cfg)].get("comm", [float("nan")]))
        acc = st.median(info[(b, cfg)].get("acc", [float("nan")]))
        print(f"| {B} | {steps} | {cfg} | {med:.3f}{rng} | {1000 * med / steps:.2f} | {sp} | {comm:.1f} | {acc:.2f} |")
