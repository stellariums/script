#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
from ase.io import read
from calorine.calculators import CPUNEP
import multiprocessing as mp
 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-s","--structure", default="dump.xyz")
    p.add_argument("-m","--model", default="nep.txt")
    p.add_argument("-i","--index", type=int, default=0)
    p.add_argument("--cell", nargs=3, type=float)
    p.add_argument("--atom-index", type=int)
    p.add_argument("--indices", nargs="+", type=int)
    p.add_argument("--random", action="store_true")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("-o","--output", default="fc2_one_atom.txt")
    p.add_argument("--nprocs", type=int, default=0)
    return p.parse_args()

 
_g_base = None
_g_calc = None
_g_delta = None

def _init_worker(model, base_atoms, delta):
    global _g_base, _g_calc, _g_delta
    _g_base = base_atoms
    _g_calc = CPUNEP(model)
    _g_delta = delta

def compute_col_for_alpha(task):
    idx, alpha = task
    atoms = _g_base.copy()
    pos = atoms.get_positions()
    pos[idx, alpha] += _g_delta
    atoms.set_positions(pos)
    atoms.calc = _g_calc
    _g_calc.set_atoms(atoms)
    f_plus = atoms.get_forces()
    pos[idx, alpha] -= 2.0 * _g_delta
    atoms.set_positions(pos)
    _g_calc.set_atoms(atoms)
    f_minus = atoms.get_forces()
    col = -(f_plus[idx, :] - f_minus[idx, :]) / (2.0 * _g_delta)
    return idx, alpha, col

def main():
    args = parse_args()
    if not os.path.exists(args.structure):
        print("missing structure file:", args.structure); sys.exit(1)
    if not os.path.exists(args.model):
        print("missing model file:", args.model); sys.exit(1)
    structure = read(args.structure, index=args.index)
    print("natoms", len(structure))
    if args.cell:
        structure.set_cell([[args.cell[0],0,0],[0,args.cell[1],0],[0,0,args.cell[2]]], scale_atoms=True)
        structure.pbc = True
    calculator = CPUNEP(args.model)
    structure.calc = calculator
    n = len(structure)
    if args.indices:
        sel = [int(x) for x in args.indices]
    elif args.atom_index is not None:
        sel = [int(args.atom_index)]
    else:
        k = max(1, min(args.count, n)) if args.random or args.count else 1
        sel = list(np.random.choice(n, size=k, replace=False))
    sel = sorted(sel)
    for idx in sel:
        if idx < 0 or idx >= n:
            print("invalid atom index:", idx); sys.exit(1)
    print("selected atoms", sel)
    tasks = []
    for idx in sel:
        for a in range(3):
            tasks.append((idx, a))
    procs = args.nprocs if args.nprocs > 0 else min(len(tasks), (mp.cpu_count() or 1))
    if procs > 1:
        results = []
        done = 0
        total = len(tasks)
        with mp.Pool(processes=procs, initializer=_init_worker, initargs=(args.model, structure, args.delta)) as pool:
            for r in pool.imap_unordered(compute_col_for_alpha, tasks):
                results.append(r)
                done += 1
                print(f"progress {done}/{total}")
    else:
        results = []
        total = len(tasks)
        calculator = CPUNEP(args.model)
        for i, (idx_task, a_task) in enumerate(tasks, 1):
            s_plus = structure.copy()
            pos = s_plus.get_positions()
            pos[idx_task, a_task] += args.delta
            s_plus.set_positions(pos)
            s_plus.calc = calculator
            calculator.set_atoms(s_plus)
            f_plus = s_plus.get_forces()
            pos[idx_task, a_task] -= 2.0 * args.delta
            s_plus.set_positions(pos)
            calculator.set_atoms(s_plus)
            f_minus = s_plus.get_forces()
            col = -(f_plus[idx_task, :] - f_minus[idx_task, :]) / (2.0 * args.delta)
            results.append((idx_task, a_task, col))
            print(f"progress {i}/{total}")
    K = {idx: np.zeros((3, 3), dtype=np.float64) for idx in sel}
    for idx, alpha, col in results:
        K[idx][:, alpha] = col
    with open(args.output, "w") as tf:
        for idx in sel:
            tf.write(f"j {idx}\n")
            tf.write(f"{K[idx][0,0]:.16e} {K[idx][0,1]:.16e} {K[idx][0,2]:.16e}\n")
            tf.write(f"{K[idx][1,0]:.16e} {K[idx][1,1]:.16e} {K[idx][1,2]:.16e}\n")
            tf.write(f"{K[idx][2,0]:.16e} {K[idx][2,1]:.16e} {K[idx][2,2]:.16e}\n\n")
    print("fc2_one_atom_txt written to", args.output)
    return

if __name__ == "__main__":
    main()