from __future__ import annotations

"""
使用 ASE 计算多个结构的平均键角，并将结果写入 CSV 文件。

用法示例（在包含子目录的根目录下执行）::

    python avgangle.py --root . --frame 9 --cutoff 1.7 --output avgangle.csv --workers 4

脚本会在 ``root`` 目录下找到所有子文件夹中的 ``dump.xyz`` 文件，
对每个结构读取指定帧，计算该帧中所有三体键角的平均值。
"""

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    from ase.neighborlist import neighbor_list
except ImportError as e:
    raise SystemExit("缺少依赖：请先安装 ase（例如：pip install ase）") from e

from avgbond import iter_structure_files, read_nth_frame_extxyz


def compute_all_angles(atoms, cutoff: float) -> np.ndarray:
    """
    计算给定结构中所有三体键角（单位：度）。

    参数
    ----
    atoms:
        ASE Atoms 对象。
    cutoff:
        近邻截断半径，单位为 Å。

    返回
    ----
    np.ndarray
        所有三体键角，单位为度。如果没有可用的角，则返回长度为 0 的数组。
    """
    # 使用 ASE 的邻居列表获得 i-j 原子对和最小镜像位移向量
    i_idx, j_idx, dvec = neighbor_list("ijD", atoms, cutoff)

    n_atoms = len(atoms)
    # 为每个原子构建邻居列表：neighbors[i] = [(j1, vec1), (j2, vec2), ...]
    neighbors = [[] for _ in range(n_atoms)]
    for a, b, v in zip(i_idx, j_idx, dvec):
        neighbors[a].append((b, v))

    angles: list[float] = []

    # 对每个中心原子 i，取其所有邻居两两组合 (j, k)
    for center in range(n_atoms):
        neigh = neighbors[center]
        m = len(neigh)
        if m < 2:
            # 邻居不足两个，无法构成角
            continue

        for p in range(m):
            v1 = neigh[p][1]
            n1 = float(np.linalg.norm(v1))
            if n1 == 0.0:
                # 距离为 0 的异常情况，跳过以避免数值问题
                continue
            for q in range(p + 1, m):
                v2 = neigh[q][1]
                n2 = float(np.linalg.norm(v2))
                if n2 == 0.0:
                    continue

                # cos(theta) = v1·v2 / (|v1||v2|)
                cosang = float(np.dot(v1, v2) / (n1 * n2))
                # 数值误差可能导致略微超出 [-1, 1]，需要裁剪
                cosang = float(np.clip(cosang, -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cosang)))
                angles.append(ang)

    if not angles:
        return np.array([], dtype=float)
    return np.array(angles, dtype=float)


def average_bond_angle(atoms, cutoff: float) -> float:
    """
    计算给定结构中所有键角的平均值（单位：度）。

    当结构中无法定义任何键角时，返回 ``math.nan``。
    """
    angles = compute_all_angles(atoms, cutoff=cutoff)
    if angles.size == 0:
        return math.nan
    return float(np.mean(angles))


def _process_single(
    args: Tuple[Path, Path, int, float],
) -> Tuple[str, float, float]:
    """
    单个结构文件的处理函数，可用于多进程调用。

    参数
    ----
    args:
        (xyz_path, root, frame_index, cutoff) 元组。

    返回
    ----
    (rel_path, avg_angle, var_angle):
        rel_path 为相对于 root 的路径字符串，
        avg_angle 为平均键角（度），var_angle 为键角方差（度²）。
    """
    xyz_path, root, frame_index, cutoff = args
    try:
        atoms = read_nth_frame_extxyz(xyz_path, frame_index)
        angles = compute_all_angles(atoms, cutoff=cutoff)
        if angles.size == 0:
            avg = math.nan
            var = math.nan
        else:
            avg = float(np.mean(angles))
            var = float(np.var(angles))
    except Exception:
        avg = math.nan
        var = math.nan

    rel = str(xyz_path.resolve().relative_to(root))
    return rel, avg, var


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
    """命令行入口：批量计算平均键角并写入 CSV 文件。"""
    ap = argparse.ArgumentParser(
        description="批量计算多个结构的平均键角，并输出为 CSV 文件。"
    )
    ap.add_argument(
        "--root",
        default=str(Path.cwd()),
        help="大文件夹根目录，默认当前目录。",
    )
    ap.add_argument(
        "--frame",
        type=int,
        default=9,
        help="读取第几帧（从 1 开始计数，默认 9）。",
    )
    ap.add_argument(
        "--cutoff",
        type=float,
        default=1.7,
        help="近邻截断半径，单位 Å，用于确定成键（默认 1.7）。",
    )
    ap.add_argument(
        "--output",
        default="avgangle.csv",
        help="输出 CSV 文件名（默认 avgangle.csv）。",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数（默认使用所有可用 CPU 核心）。",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    frame_index = int(args.frame) - 1
    if frame_index < 0:
        raise SystemExit("--frame 必须 >= 1")

    xyz_files = list(iter_structure_files(root))

    rows: list[tuple[str, float, float]] = []
    if not xyz_files:
        # 没有找到任何结构文件时，也生成一个只包含表头的空 CSV
        out_path = (root / args.output).resolve()
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "avg_bond_angle_deg", "var_bond_angle_deg2"])
        print(f"未找到结构文件，已写出空文件：{out_path}")
        return 0

    tasks = [(p, root, frame_index, float(args.cutoff)) for p in xyz_files]
    total = len(tasks)
    done = 0

    # 根据用户指定或系统可用 CPU 数量确定并行进程数
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
        writer = csv.writer(f)
        writer.writerow(["path", "avg_bond_angle_deg", "var_bond_angle_deg2"])
        for path_str, avg, var in rows:
            if (
                avg is None
                or var is None
                or not (math.isfinite(avg) and math.isfinite(var))
            ):
                writer.writerow([path_str, "", ""])
            else:
                writer.writerow([path_str, avg, var])

    print(f"写入完成：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

