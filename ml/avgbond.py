from __future__ import annotations

# ----------------------------------------------------------------------
# 本脚本用途：
#   批量遍历指定根目录下的子文件夹，自动查找其中的 dump.xyz 结构文件，
#   读取每个结构文件的指定帧（extxyz 格式），利用 ASE 的 neighborlist
#   功能计算该帧中所有成键原子对的键长，并输出：
#       1) 平均键长 avg_bond_length
#       2) 键长方差 var_bond_length
#   结果写入 CSV 文件，便于后续统计分析或画图。脚本支持 CPU 多进程并行，
#   并在终端显示整体处理进度条。
#
# 基本用法（在含有子目录的根目录下执行）：
#   python avgbond.py --root . --frame 9 --cutoff-mult 1.2 --output bond.csv
# 或显式指定并行进程数：
#   python avgbond.py --root . --frame 9 --cutoff-mult 1.2 --output bond.csv --workers 4
#
# 参数说明：
#   --root        根目录，脚本会遍历该目录下的一级子目录，在其中递归查找 dump.xyz
#   --frame       读取第几帧（从 1 开始计数，例如 9 表示第 9 帧）
#   --cutoff-mult 传递给 ASE natural_cutoffs 的倍率，用于确定成键截断
#   --output      输出 CSV 文件名，默认 bond.csv
#   --workers     并行进程数（默认使用所有可用 CPU 核心；1 表示串行）
#
# 输出 CSV 格式：
#   path,avg_bond_length,var_bond_length
#   其中 path 为相对于 root 的相对路径，avg_bond_length 为平均键长（Å），
#   var_bond_length 为键长方差（Å²）。当某个结构无法定义任何键长或计算失败时，
#   对应的数值会写成空字符串，便于后续按是否有效进行筛选。
# ----------------------------------------------------------------------

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    from ase.io import iread
    from ase.neighborlist import natural_cutoffs, neighbor_list
except ImportError as e:
    raise SystemExit("缺少依赖：请先安装 ase（例如：pip install ase）") from e


def read_nth_frame_extxyz(xyz_path: Path, n: int):
    for idx, atoms in enumerate(iread(str(xyz_path), format="extxyz")):
        if idx == n:
            return atoms
    raise IndexError(f"{xyz_path} 帧数不足：需要第 {n + 1} 帧")


def bond_lengths(atoms, cutoff_mult: float):
    cutoffs = natural_cutoffs(atoms, mult=cutoff_mult)
    i, j, d = neighbor_list("ijd", atoms, cutoffs)
    if len(d) == 0:
        return np.array([], dtype=float)
    mask = i < j
    d = d[mask]
    if len(d) == 0:
        return np.array([], dtype=float)
    return np.asarray(d, dtype=float)


def average_bond_length(atoms, cutoff_mult: float) -> float:
    d = bond_lengths(atoms, cutoff_mult=cutoff_mult)
    if len(d) == 0:
        return math.nan
    return float(np.mean(d))


def _process_single(args: tuple[Path, Path, int, float]) -> tuple[str, float, float]:
    xyz_path, root, frame_index, cutoff_mult = args
    try:
        atoms = read_nth_frame_extxyz(xyz_path, frame_index)
        d = bond_lengths(atoms, cutoff_mult=cutoff_mult)
        if len(d) == 0:
            avg = math.nan
            var = math.nan
        else:
            avg = float(np.mean(d))
            var = float(np.var(d))
    except Exception:
        avg = math.nan
        var = math.nan
    rel = str(xyz_path.resolve().relative_to(root))
    return rel, avg, var


def iter_structure_files(root: Path):
    for top in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        for p in sorted(top.rglob("dump.xyz"), key=lambda x: str(x)):
            yield p


def _print_progress(done: int, total: int) -> None:
    if total <= 0:
        return
    frac = done / total
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    percent = int(frac * 100)
    print(
        f"\r进度: [{bar}] {percent:3d}% ({done}/{total})",
        end="",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(Path.cwd()),
        help="大文件夹根目录，默认当前目录",
    )
    ap.add_argument(
        "--frame",
        type=int,
        default=9,
        help="读取第几帧（从 1 开始计数，默认 9）",
    )
    ap.add_argument(
        "--cutoff-mult",
        type=float,
        default=1.2,
        help="ASE natural_cutoffs 的倍率（默认 1.2）",
    )
    ap.add_argument(
        "--output",
        default="bond.csv",
        help="输出 CSV 文件名（默认 bond.csv）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数（默认使用所有可用 CPU 核心）",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    frame_index = int(args.frame) - 1
    if frame_index < 0:
        raise SystemExit("--frame 必须 >= 1")

    xyz_files = list(iter_structure_files(root))

    rows: list[tuple[str, float, float]] = []
    if not xyz_files:
        out_path = (root / args.output).resolve()
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "avg_bond_length", "var_bond_length"])
        print(f"未找到结构文件，已写出空文件：{out_path}")
        return 0

    tasks = [(p, root, frame_index, float(args.cutoff_mult)) for p in xyz_files]
    total = len(tasks)
    done = 0

    if args.workers is None or args.workers <= 0:
        max_workers = os.cpu_count() or 1
    else:
        max_workers = max(1, int(args.workers))

    if max_workers == 1 or len(tasks) == 1:
        for task in tasks:
            rows.append(_process_single(task))
            done += 1
            _print_progress(done, total)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for rel, avg, var in executor.map(_process_single, tasks):
                rows.append((rel, avg, var))
                done += 1
                _print_progress(done, total)

    if total > 0:
        print()

    out_path = (root / args.output).resolve()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "avg_bond_length", "var_bond_length"])
        for path, avg, var in rows:
            if (
                avg is None
                or var is None
                or not (math.isfinite(avg) and math.isfinite(var))
            ):
                w.writerow([path, "", ""])
            else:
                w.writerow([path, avg, var])

    print(f"写入完成：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
