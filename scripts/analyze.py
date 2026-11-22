#!/usr/bin/env python3
"""
analyze.py - turn results/*.csv into tables, an alpha-beta model check and figures.
Ideas borrowed from pdl-bench: mean +- 95% CI over repeats, speedup vs a baseline,
fitted Hockney alpha-beta(-gamma) model vs measured allreduce time, per-phase
breakdown, and an automatic sequential <-> 1 GPU <-> k GPU loss-curve check.

  python3 scripts/analyze.py results                  # everything -> results/report.md, results/fig_*.png
  python3 scripts/analyze.py --check-loss REF.csv A.csv [B.csv ...] [--tol 1e-4]
                                                      # exit 1 if any curve differs from REF by > tol

Inputs (all optional, whatever exists is used):
  results/allreduce.csv   bench_allreduce rows
  results/p2p.csv         bench_p2p rows (method p2p / add) -> alpha, beta, gamma
  results/training.csv    train_dp / mlp_seq rows (one per repeat)
  results/phases.csv      train_dp --phase-csv rows
  results/loss_*.csv      per-epoch loss logs (epoch,loss)

Only the Python standard library is required; figures need matplotlib.

Cost models (n = message bytes, k = GPUs; alpha, beta from p2p ping-pong; alpha_a,
gamma from the ar_add kernel). The final ar_scale is the same for every method
and is left out.
  tree (flat, as in tree.cu): 2(k-1)(alpha + n beta) + (k-1)(alpha_a + n gamma)
  ring:                       2(k-1)(alpha + n/k beta) + (k-1)(alpha_a + n/k gamma)
  rd (recursive doubling):    r (alpha + n beta + alpha_a + n gamma),  r = ceil(log2 k)
                              (+2 rounds if k is not a power of two)
The models assume transfers do not slow each other down. When several copies
share one PCIe link (or are staged through host memory because P2P is off),
measured time is above the model: that gap is a result worth explaining.
"""
import csv
import glob
import math
import os
import sys
from collections import defaultdict

# Two-sided 95% Student t critical values, df = 1..30 (beyond: 1.96).
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
       2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
       2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci(xs):
    """Mean and half-width of the 95% confidence interval (0 if fewer than 2 values)."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t = T95[n - 2] if n - 1 <= len(T95) else 1.96
    return m, t * sd / math.sqrt(n)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            if r.get("method") == "method":       # header repeated after a concatenation
                continue
            rows.append(r)
        return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fit_linear(points):
    """Fit t = a + b*n minimizing RELATIVE error, so that both the latency-bound
    small sizes and the bandwidth-bound large sizes are fitted (plain least
    squares would ignore the small sizes). points: [(n, t)]. Returns (a, b)."""
    pts = [(n, t) for n, t in points if t > 0]
    if len(pts) < 2:
        return None
    sw = swn = swnn = swt = swnt = 0.0
    for n, t in pts:
        w = 1.0 / (t * t)
        sw += w; swn += w * n; swnn += w * n * n; swt += w * t; swnt += w * n * t
    det = sw * swnn - swn * swn
    if det == 0:
        return None
    b = (sw * swnt - swn * swt) / det
    a = (swt - b * swn) / sw
    return a, b


def model_ms(method, n, k, P):
    """Predicted allreduce time in ms, or None if there is no model for `method`."""
    if P is None:
        return None
    al, be, aa, ga = P["alpha"], P["beta"], P["alpha_a"], P["gamma"]
    if method == "tree":
        return 2 * (k - 1) * (al + n * be) + (k - 1) * (aa + n * ga)
    if method == "ring":
        return 2 * (k - 1) * (al + n / k * be) + (k - 1) * (aa + n / k * ga)
    if method == "rd":
        r = math.ceil(math.log2(k)) if k > 1 else 0
        if k & (k - 1):
            r += 2
        return r * (al + n * be + aa + n * ga)
    return None


def read_loss(path):
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    out.append((int(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    return out


def compare_loss(ref, other):
    """Max |loss difference| over the common epochs (inf if no common epoch)."""
    a, b = dict(ref), dict(other)
    common = sorted(set(a) & set(b))
    if not common:
        return float("inf"), 0
    return max(abs(a[e] - b[e]) for e in common), len(common)


def fmt_bytes(n):
    x = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if x < 1024 or unit == "GB":
            return f"{x:g} {unit}"
        x /= 1024


# --------------------------------------------------------------------------- sections

def section_params(res, rep):
    rows = read_csv(os.path.join(res, "p2p.csv"))
    if not rows:
        rep.append("## Machine parameters\n\nNo `p2p.csv` yet: run `./bin/bench_p2p --csv results/p2p.csv`.\n")
        return None
    p2p = defaultdict(list)
    add = defaultdict(list)
    for r in rows:
        if r["method"] == "p2p":
            p2p[int(r["msg_bytes"])].append(fnum(r["time_ms"]))
        elif r["method"] == "add":
            add[int(r["msg_bytes"])].append(fnum(r["time_ms"]))
    p2p_pts = [(n, median(ts)) for n, ts in sorted(p2p.items())]
    add_pts = [(n, median(ts)) for n, ts in sorted(add.items())]
    f1, f2 = fit_linear(p2p_pts), fit_linear(add_pts)
    if not f1 or not f2:
        rep.append("## Machine parameters\n\nNot enough points in `p2p.csv` to fit.\n")
        return None
    P = {"alpha": f1[0], "beta": f1[1], "alpha_a": f2[0], "gamma": f2[1],
         "p2p_pts": p2p_pts, "add_pts": add_pts}
    err = [abs(P["alpha"] + P["beta"] * n - t) / t * 100 for n, t in p2p_pts]
    rep.append("## Machine parameters (fitted from bench_p2p)\n")
    rep.append("| parameter | value | meaning |\n|---|---|---|")
    rep.append(f"| alpha | {P['alpha'] * 1e3:.2f} us | latency of one GPU-to-GPU copy |")
    rep.append(f"| beta | {P['beta'] * 1e9 / 1e3:.4f} us/KB ({1 / (P['beta'] * 1e6):.2f} GB/s) | time per byte = 1 / bandwidth |")
    rep.append(f"| alpha_add | {P['alpha_a'] * 1e3:.2f} us | launch latency of ar_add |")
    rep.append(f"| gamma | {P['gamma'] * 1e9 / 1e3:.5f} us/KB ({1 / (P['gamma'] * 1e6):.1f} GB/s) | time per byte of the reduction kernel |")
    rep.append(f"\nFit quality (copy): median relative error {median(err):.1f}%, max {max(err):.1f}%.\n")
    return P


def section_allreduce(res, rep, P):
    rows = read_csv(os.path.join(res, "allreduce.csv"))
    if not rows:
        rep.append("## Raw allreduce\n\nNo `allreduce.csv` yet.\n")
        return {}
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], int(r["k"]), int(r["msg_bytes"]))].append((fnum(r["time_ms"]), fnum(r["busbw_GBs"])))
    data = {key: (median([t for t, _ in v]), median([b for _, b in v])) for key, v in g.items()}
    rep.append("## Raw allreduce: measured vs alpha-beta model\n")
    rep.append("Median over the 20 timed runs of bench_allreduce (and over re-runs, if the CSV has several). "
               "`model` uses the fitted parameters above; `error` = (measured − model) / model.\n")
    rep.append("| method | k | size | measured ms | busbw GB/s | model ms | error |\n|---|---|---|---|---|---|---|")
    errs = defaultdict(list)
    for (m, k, n), (t, bw) in sorted(data.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        pred = model_ms(m, n, k, P)
        if pred:
            e = (t - pred) / pred * 100
            errs[(m, k)].append(e)
            ps, es = f"{pred:.4f}", f"{e:+.0f}%"
        else:
            ps, es = "-", "-"
        rep.append(f"| {m} | {k} | {fmt_bytes(n)} | {t:.4f} | {bw:.2f} | {ps} | {es} |")
    if errs:
        rep.append("\nModel error summary (median |error| over sizes):\n")
        for (m, k), es in sorted(errs.items()):
            rep.append(f"* {m}, k={k}: {median([abs(e) for e in es]):.0f}%")
    # winner per size
    best = defaultdict(lambda: (None, float("inf")))
    for (m, k, n), (t, _) in data.items():
        if m != "host_ref" and t < best[(k, n)][1]:
            best[(k, n)] = (m, t)
    if best:
        rep.append("\nFastest method per size (host_ref excluded):\n")
        for (k, n), (m, t) in sorted(best.items()):
            rep.append(f"* k={k}, {fmt_bytes(n)}: **{m}** ({t:.4f} ms)")
    rep.append("")
    return data


def section_training(res, rep):
    rows = read_csv(os.path.join(res, "training.csv"))
    if not rows:
        rep.append("## Training\n\nNo `training.csv` yet.\n")
        return {}
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], int(r["k"]))].append(r)
    base = None
    for key in (("gpu1", 1),):
        if key in g:
            base = mean_ci([fnum(r["epoch_s"]) for r in g[key]])[0]
    rep.append("## Training (mean ± 95% CI over repeats)\n")
    rep.append("Speedup is relative to the 1-GPU run with the same global batch "
               "(strong scaling, same samples in the same order).\n")
    rep.append("| method | k | runs | epoch s | speedup vs 1 GPU | efficiency | exposed comm % | test acc |\n|---|---|---|---|---|---|---|---|")
    out = {}
    for (m, k), rs in sorted(g.items(), key=lambda x: (x[0][1], x[0][0])):
        ep = mean_ci([fnum(r["epoch_s"]) for r in rs])
        cp = mean_ci([fnum(r["comm_pct"]) for r in rs])
        acc = mean_ci([fnum(r["test_acc"]) for r in rs])
        sp = base / ep[0] if base and m != "seq" else float("nan")
        # CI of the speedup from the relative CI of the epoch time (first order).
        sp_ci = sp * ep[1] / ep[0] if ep[0] and not math.isnan(sp) else float("nan")
        out[(m, k)] = (ep, sp, sp_ci, cp)
        sps = "-" if math.isnan(sp) else f"{sp:.2f} ± {sp_ci:.2f}"
        eff = "-" if math.isnan(sp) else f"{sp / k * 100:.0f}%"
        cps = "-" if m == "seq" else f"{cp[0]:.1f} ± {cp[1]:.1f}"
        rep.append(f"| {m} | {k} | {len(rs)} | {ep[0]:.4f} ± {ep[1]:.4f} | {sps} | {eff} | {cps} | {acc[0] * 100:.2f}% |")
    if any(len(v) < 3 for v in g.values()):
        rep.append("\nSome configurations have fewer than 3 runs: their CI is wide or 0. Use REPEATS=5 for final numbers.")
    rep.append("")
    return out


PHASES = ["data_ms", "h2d_ms", "fwd_ms", "bwd_ms", "comm_ms", "upd_ms"]


def section_phases(res, rep):
    rows = read_csv(os.path.join(res, "phases.csv"))
    if not rows:
        return {}
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], int(r["k"]))].append(r)
    rep.append("## Where the time goes (mean ms per step; max over GPUs per step)\n")
    rep.append("`comm` is exposed communication (after backward). In `+overlap` runs part of the "
               "communication hides inside `bwd`. `imbalance` = slowest GPU compute / mean GPU compute.\n")
    rep.append("| method | k | " + " | ".join(p[:-3] for p in PHASES) + " | step | comm share | imbalance |")
    rep.append("|---|---|" + "---|" * (len(PHASES) + 3))
    out = {}
    for (m, k), rs in sorted(g.items(), key=lambda x: (x[0][1], x[0][0])):
        mv = {p: sum(fnum(r[p]) for r in rs) / len(rs) for p in PHASES + ["step_ms", "imbalance"]}
        out[(m, k)] = mv
        share = mv["comm_ms"] / mv["step_ms"] * 100 if mv["step_ms"] else 0
        rep.append(f"| {m} | {k} | " + " | ".join(f"{mv[p]:.3f}" for p in PHASES)
                   + f" | {mv['step_ms']:.3f} | {share:.0f}% | {mv['imbalance']:.3f} |")
    rep.append("")
    return out


def section_loss(res, rep, tol):
    ref_path = os.path.join(res, "loss_gpu1.csv")
    seq_path = os.path.join(res, "loss_seq.csv")
    files = sorted(glob.glob(os.path.join(res, "loss_*.csv")))
    if not files:
        return True
    rep.append(f"## Equivalence check: loss curves (tolerance {tol:g})\n")
    rep.append("Synchronous data-parallel SGD with the same global batch and seed must give the same "
               "loss as 1 GPU and as the sequential C code, up to float rounding.\n")
    rep.append("| curve | reference | epochs | max abs diff | result |\n|---|---|---|---|---|")
    ok = True
    curves = {}
    for f in files:
        curves[f] = read_loss(f)
    for f in files:
        name = os.path.basename(f)
        for ref in (ref_path, seq_path):
            if f == ref or ref not in curves:
                continue
            d, n = compare_loss(curves[ref], curves[f])
            good = d <= tol
            ok &= good
            rep.append(f"| {name} | {os.path.basename(ref)} | {n} | {d:.2e} | {'PASS' if good else '**FAIL**'} |")
            break     # each curve against the 1-GPU run (or seq if there is no 1-GPU run)
    rep.append("")
    return ok


# --------------------------------------------------------------------------- figures

def figures(res, P, ar, tr, ph):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures (pip install matplotlib)", file=sys.stderr)
        return []
    made = []
    colors = {"host_ref": "#888888", "tree": "#2a6fdb", "ring": "#e0702b", "rd": "#2e9e5b"}

    if ar:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for (m, k) in sorted({(m, k) for m, k, _ in ar}):
            pts = sorted((n, t, bw) for (mm, kk, n), (t, bw) in ar.items() if mm == m and kk == k)
            ns = [p[0] for p in pts]
            c = colors.get(m)
            ax1.plot(ns, [p[1] for p in pts], "o-", ms=3, color=c, label=f"{m} k={k}")
            ax2.plot(ns, [p[2] for p in pts], "o-", ms=3, color=c, label=f"{m} k={k}")
            pred = [model_ms(m, n, k, P) for n in ns]
            if all(pred):
                ax1.plot(ns, pred, "--", color=c, alpha=0.6, label=f"{m} k={k} model")
        ax1.set(xscale="log", yscale="log", xlabel="message size (bytes)", ylabel="time (ms)",
                title="Allreduce time: measured (solid) vs model (dashed)")
        ax2.set(xscale="log", xlabel="message size (bytes)", ylabel="bus bandwidth (GB/s)",
                title="Allreduce bus bandwidth")
        for a in (ax1, ax2):
            a.grid(True, which="both", alpha=0.3)
        ax1.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(res, "fig_allreduce.png"); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    if P:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ns = [n for n, _ in P["p2p_pts"]]
        ax.plot(ns, [t for _, t in P["p2p_pts"]], "o", ms=3, label="p2p one-way (measured)")
        ax.plot(ns, [P["alpha"] + P["beta"] * n for n in ns], "-", label="alpha + beta n")
        ax.plot(ns, [t for _, t in P["add_pts"]], "s", ms=3, label="ar_add (measured)")
        ax.plot(ns, [P["alpha_a"] + P["gamma"] * n for n in ns], "-", label="alpha_add + gamma n")
        ax.set(xscale="log", yscale="log", xlabel="bytes", ylabel="ms", title="Fitted machine parameters")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(res, "fig_alpha_beta.png"); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    if ph:
        keys = sorted(ph, key=lambda x: (x[1], x[0]))
        fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(keys) + 2), 4))
        bottom = [0.0] * len(keys)
        pal = ["#bbbbbb", "#8fb8e8", "#2a6fdb", "#1b3f80", "#e0702b", "#2e9e5b"]
        for p_name, col in zip(PHASES, pal):
            vals = [ph[k][p_name] for k in keys]
            ax.bar(range(len(keys)), vals, bottom=bottom, color=col, label=p_name[:-3])
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{m}\nk={k}" for m, k in keys], fontsize=8)
        ax.set(ylabel="ms per step", title="Step time breakdown (max over GPUs)")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        p = os.path.join(res, "fig_phases.png"); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    if tr:
        keys = [k for k in sorted(tr, key=lambda x: (x[1], x[0])) if not math.isnan(tr[k][1]) and k[1] > 1]
        if keys:
            fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(keys) + 2), 4))
            ax.bar(range(len(keys)), [tr[k][1] for k in keys], yerr=[tr[k][2] for k in keys], capsize=3,
                   color=[colors.get(k[0].split("+")[0], "#666") for k in keys])
            for kk in sorted({k[1] for k in keys}):
                ax.axhline(kk, ls=":", color="k", alpha=0.4)
            ax.axhline(1, color="k", lw=0.8)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels([f"{m}\nk={k}" for m, k in keys], fontsize=8)
            ax.set(ylabel="speedup vs 1 GPU", title="Training speedup (mean ± 95% CI; dotted = ideal)")
            fig.tight_layout()
            p = os.path.join(res, "fig_speedup.png"); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    files = sorted(glob.glob(os.path.join(res, "loss_*.csv")))
    if files:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for f in files:
            c = read_loss(f)
            if c:
                ax.plot([e for e, _ in c], [l for _, l in c], "o-", ms=3, label=os.path.basename(f)[5:-4])
        ax.set(xlabel="epoch", ylabel="training loss", title="Loss curves (should overlap)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(res, "fig_loss.png"); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    return made


# --------------------------------------------------------------------------- main

def main(argv):
    tol = 1e-4
    if "--tol" in argv:
        i = argv.index("--tol")
        tol = float(argv[i + 1])
        del argv[i:i + 2]
    if argv and argv[0] == "--check-loss":
        if len(argv) < 3:
            print(__doc__); return 2
        ref = read_loss(argv[1])
        bad = 0
        for f in argv[2:]:
            d, n = compare_loss(ref, read_loss(f))
            good = d <= tol and n > 0
            bad += not good
            print(f"{'PASS' if good else 'FAIL'}  {f} vs {argv[1]}: max |diff| {d:.2e} over {n} epochs (tol {tol:g})")
        return 1 if bad else 0
    if len(argv) != 1 or argv[0].startswith("-"):
        print(__doc__); return 2
    res = argv[0]
    rep = ["# Results report", "", f"Generated by `scripts/analyze.py` from `{res}/`.", ""]
    P = section_params(res, rep)
    ar = section_allreduce(res, rep, P)
    tr = section_training(res, rep)
    ph = section_phases(res, rep)
    ok = section_loss(res, rep, tol)
    figs = figures(res, P, ar, tr, ph)
    if figs:
        rep.append("## Figures\n")
        rep += [f"![{os.path.basename(f)}]({os.path.basename(f)})" for f in figs]
    out = os.path.join(res, "report.md")
    with open(out, "w") as f:
        f.write("\n".join(rep) + "\n")
    print(f"wrote {out}" + (f" and {len(figs)} figures" if figs else ""))
    if not ok:
        print("WARNING: at least one loss curve differs from the reference (see report)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
