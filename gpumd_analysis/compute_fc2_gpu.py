#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
from ase.io import read
from calorine.calculators import CPUNEP, GPUNEP
import multiprocessing as mp
import time
 

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
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--gpu-dir", type=str)
    p.add_argument("--gpu-concurrency", type=int, default=1)
    return p.parse_args()

 
_g_base = None
_g_calc = None
_g_delta = None

def _init_worker(model, base_atoms, delta, use_gpu, gpu_index, gpu_dir):
    global _g_base, _g_calc, _g_delta
    _g_base = base_atoms
    if use_gpu:
        dir_worker = gpu_dir
        if dir_worker:
            try:
                dir_worker = os.path.join(dir_worker, f"worker-{os.getpid()}")
                os.makedirs(dir_worker, exist_ok=True)
            except Exception:
                dir_worker = gpu_dir
        _g_calc = GPUNEP(model, directory=dir_worker, gpu_identifier_index=gpu_index)
    else:
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
    t0 = time.perf_counter()
    if not os.path.exists(args.structure):
        print("missing structure file:", args.structure); sys.exit(1)
    if not os.path.exists(args.model):
        print("missing model file:", args.model); sys.exit(1)
    structure = read(args.structure, index=args.index)
    print("natoms", len(structure))
    if args.cell:
        structure.set_cell([[args.cell[0],0,0],[0,args.cell[1],0],[0,0,args.cell[2]]], scale_atoms=True)
        structure.pbc = True
    calc_cls = GPUNEP if args.gpu else CPUNEP
    gpu_idx = None if (args.gpu and args.gpu_index is not None and args.gpu_index < 0) else args.gpu_index
    calculator_boot = GPUNEP(args.model, directory=args.gpu_dir, gpu_identifier_index=gpu_idx) if args.gpu else CPUNEP(args.model)
    structure.calc = calculator_boot
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
    if args.gpu:
        procs = max(1, args.gpu_concurrency)
    if procs > 1:
        results = []
        done = 0
        total = len(tasks)
        with mp.Pool(processes=procs, initializer=_init_worker, initargs=(args.model, structure, args.delta, args.gpu, gpu_idx, args.gpu_dir)) as pool:
            for r in pool.imap_unordered(compute_col_for_alpha, tasks):
                results.append(r)
                done += 1
                print(f"progress {done}/{total}")
    else:
        results = []
        total = len(tasks)
        calculator = calculator_boot
        pos0 = structure.get_positions()
        for i, (idx_task, a_task) in enumerate(tasks, 1):
            pos = pos0.copy()
            pos[idx_task, a_task] += args.delta
            structure.set_positions(pos)
            structure.calc = calculator
            calculator.set_atoms(structure)
            f_plus = structure.get_forces()
            pos[idx_task, a_task] -= 2.0 * args.delta
            structure.set_positions(pos)
            calculator.set_atoms(structure)
            f_minus = structure.get_forces()
            col = -(f_plus[idx_task, :] - f_minus[idx_task, :]) / (2.0 * args.delta)
            results.append((idx_task, a_task, col))
            print(f"progress {i}/{total}")
        structure.set_positions(pos0)
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
    dt = time.perf_counter() - t0
    print(f"total elapsed {dt:.3f} s")
    return

if __name__ == "__main__":
    main()