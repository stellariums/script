#!/usr/bin/env python3
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="output.txt")
    p.add_argument("-o", "--output", default="fc2_scatter.png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true")
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
    if not os.path.exists(args.input):
        print("missing input:", args.input)
        return
    idxs, xs, ys = read_fc2(args.input)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, s=40, alpha=0.8)
    ax.set_xlabel("Mean |diag| of 3 diagonal entries")
    ax.set_ylabel("Mean |off-diag| of 6 entries")
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print("points", len(xxs := xs))
    print("saved", args.output)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()