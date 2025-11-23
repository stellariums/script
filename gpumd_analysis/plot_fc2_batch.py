#!/usr/bin/env python3
import argparse, os
import numpy as np
from glob import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+")
    p.add_argument("--pattern", default="output*.txt")
    p.add_argument("-o", "--output", default="fc2_compare.png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true")
    p.add_argument("--size", type=float, default=40.0)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--limit", type=int)
    return p.parse_args()


def read_fc2(path):
    idxs = []
    xs = []
    ys = []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    n = len(lines)
    i = 0
    while i < n:
        ln = lines[i]
        if not ln:
            i += 1
            continue
        if ln.startswith("j"):
            parts = ln.split()
            idx = int(parts[1]) if len(parts) > 1 else None
            i += 1
            mat_rows = []
            while i < n and len(mat_rows) < 3:
                if not lines[i]:
                    i += 1
                    continue
                row_vals = [float(x) for x in lines[i].split()]
                if len(row_vals) >= 3:
                    mat_rows.append(row_vals[:3])
                i += 1
            if len(mat_rows) == 3:
                M = np.array(mat_rows, dtype=np.float64)
                d = np.array([abs(M[0,0]), abs(M[1,1]), abs(M[2,2])])
                o = np.array([abs(M[0,1]), abs(M[0,2]), abs(M[1,0]), abs(M[1,2]), abs(M[2,0]), abs(M[2,1])])
                x = float(np.mean(d))
                y = float(np.mean(o))
                idxs.append(idx if idx is not None else len(idxs))
                xs.append(x)
                ys.append(y)
        else:
            i += 1
    return idxs, np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def main():
    args = parse_args()
    files = []
    if args.inputs:
        for p in args.inputs:
            files.extend(sorted(glob(p)))
    else:
        files = sorted(glob(args.pattern))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("no input files matched")
        return
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, fpath in enumerate(files):
        idxs, xs, ys = read_fc2(fpath)
        if args.limit is not None and args.limit > 0:
            xs = xs[:args.limit]
            ys = ys[:args.limit]
        color = cmap(k % 10)
        ax.scatter(xs, ys, s=args.size, alpha=args.alpha, color=color, label=os.path.basename(fpath))
        print("points", len(xs), fpath)
    ax.set_xlabel("Mean |diag| of 3 diagonal entries")
    ax.set_ylabel("Mean |off-diag| of 6 entries")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print("saved", args.output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()