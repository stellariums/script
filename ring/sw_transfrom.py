#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
扩展 XYZ 输出（保留输入前两行）+ 体素哈希近邻搜索（避免大内存）
- 找到距离 < cutoff 的相邻 C-C 对（体素哈希）
- 随机选出互不重叠的 npairs 对
- 以每对中点为中心，绕 z 轴旋转 angle 度（默认 90°）
- 旋转后用坐标包围盒计算 Lattice（正交晶胞：diag=[Lx,Ly,Lz]），仅用于日志打印
- **写出文件的前两行与输入 .xyz 文件完全一致（逐字保留）**
- 每行原子字段仍为：species  x y z  mass  vx vy vz
"""

import argparse
import math
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
from ase.io import read
from ase.data import atomic_numbers, atomic_masses

random_number = random.randint(1, 10000)

# ---------------------------
# 近邻搜索：体素哈希（内存稳定）
# ---------------------------
def build_voxel_grid(positions: np.ndarray, cell: float) -> Dict[Tuple[int, int, int], List[int]]:
    keys = np.floor(positions / cell).astype(np.int64)  # (N,3)
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, key in enumerate(map(tuple, keys)):
        grid.setdefault(key, []).append(idx)
    return grid

def neighbor_keys(key: Tuple[int, int, int]) -> Iterable[Tuple[int, int, int]]:
    x, y, z = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (x + dx, y + dy, z + dz)

def find_adjacent_pairs_grid(positions: np.ndarray, cutoff: float) -> List[Tuple[int, int, float]]:
    """返回所有 (i, j, d)，i<j 且 0<d<cutoff。"""
    cutoff2 = cutoff * cutoff
    grid = build_voxel_grid(positions, cutoff)
    pairs: List[Tuple[int, int, float]] = []

    for key, idx_list in grid.items():
        cand: List[int] = []
        for nk in neighbor_keys(key):
            if nk in grid:
                cand.extend(grid[nk])
        if not cand:
            continue
        cand_arr = np.asarray(cand, dtype=np.int64)

        for i in idx_list:
            j_arr = cand_arr[cand_arr > i]
            if j_arr.size == 0:
                continue
            diff = positions[j_arr] - positions[i]          # (M,3)
            d2 = np.einsum("ij,ij->i", diff, diff)          # (M,)
            mask = (d2 > 0.0) & (d2 < cutoff2)
            if not np.any(mask):
                continue
            js = j_arr[mask]
            ds = np.sqrt(d2[mask])
            pairs.extend(zip([i]*js.size, js.tolist(), ds.tolist()))
    return pairs

# ---------------------------
# 选对 + 旋转
# ---------------------------
def select_disjoint_pairs(
    pairs: List[Tuple[int, int, float]], n_select: int, seed: int = 42
) -> List[Tuple[int, int, float]]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    used = set()
    chosen = []
    for a, b, dist in shuffled:
        if a in used or b in used:
            continue
        chosen.append((a, b, dist))
        used.add(a); used.add(b)
        if len(chosen) >= n_select:
            break
    return chosen

def rotate_about_z_midpoint(positions: np.ndarray, idx_a: int, idx_b: int, angle_deg: float):
    """以 a、b 中点为中心，绕 z 轴旋转 angle_deg（度）。仅更新 a、b 坐标。"""
    angle = math.radians(angle_deg)
    cosA = math.cos(angle); sinA = math.sin(angle)
    R = np.array([[cosA, -sinA, 0.0],
                  [sinA,  cosA, 0.0],
                  [0.0,   0.0,  1.0]], dtype=float)
    pa = positions[idx_a]
    pb = positions[idx_b]
    m = 0.5 * (pa + pb)
    positions[idx_a] = (R @ (pa - m)) + m
    positions[idx_b] = (R @ (pb - m)) + m

# ---------------------------
# Lattice 计算（由旋转后的包围盒得到）
# ---------------------------
def compute_lattice_from_bbox(positions: np.ndarray) -> Tuple[float, float, float]:
    """返回 (Lx, Ly, Lz)，为坐标的包围盒尺寸。"""
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    Lx, Ly, Lz = (maxs - mins).tolist()
    eps = 1e-12
    return (max(Lx, eps), max(Ly, eps), max(Lz, eps))

# ---------------------------
# I/O 辅助：读取并保留输入的前两行
# ---------------------------
def read_first_two_lines(path: str, encoding: str = "utf-8") -> Tuple[str, str]:
    """
    逐字读取输入 .xyz 的前两行（含换行符去除），用于原样写回输出文件。
    如果文件不足两行，将抛出异常。
    """
    with open(path, "r", encoding=encoding, newline="") as f:
        line1 = f.readline()
        line2 = f.readline()
    if line1 == "" or line2 == "":
        raise ValueError("输入 .xyz 文件行数不足两行，无法保持原样头部。")
    # 去除末尾换行，写出时统一由 writer 控制换行
    return line1.rstrip("\r\n"), line2.rstrip("\r\n")

# ---------------------------
# 扩展 XYZ 手写输出（保留输入前两行）
# ---------------------------
def write_extxyz_preserve_header(
    path: str,
    header_line1: str,
    header_line2: str,
    symbols: List[str],
    positions: np.ndarray,
    masses: np.ndarray,
    velocities: np.ndarray,
    float_fmt_pos: str = "{:.6f}",
    float_fmt_mass: str = "{:.6f}",
    float_fmt_vel: str = "{:.6f}",
    encoding: str = "utf-8",
):
    """
    写扩展 XYZ：
    第 1 行：原样复制输入文件的第一行
    第 2 行：原样复制输入文件的第二行
    后续每行：species  x y z  mass  vx vy vz
    """
    assert positions.shape[0] == len(symbols) == masses.shape[0] == velocities.shape[0]

    with open(path, "w", encoding=encoding, newline="\n") as f:
        f.write(f"{header_line1}\n")
        f.write(f"{header_line2}\n")
        for s, (x, y, z), m, (vx, vy, vz) in zip(symbols, positions, masses, velocities):
            f.write(
                "{} {} {} {} {} {} {} {}\n".format(
                    s,
                    float_fmt_pos.format(x),
                    float_fmt_pos.format(y),
                    float_fmt_pos.format(z),
                    float_fmt_mass.format(m),
                    float_fmt_vel.format(vx),
                    float_fmt_vel.format(vy),
                    float_fmt_vel.format(vz),
                )
            )

# ---------------------------
# 主程序
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Rotate randomly selected adjacent C-C pairs around z-axis, then write Extended XYZ while preserving the first two header lines from the input."
    )
    ap.add_argument("input", help="输入 .xyz 文件（推荐只含 C）")
    ap.add_argument("output", help="输出 .xyz 文件（前两行与输入一致）")
    ap.add_argument("--cutoff", type=float, default=1.85, help="相邻判定距离（Angstrom），默认 1.85")
    ap.add_argument("--npairs", type=int, default=100, help="要旋转的互不重叠 C-C 对数量，默认 100")
    ap.add_argument("--angle", type=float, default=90.0, help="绕 z 轴旋转角度（度），默认 90")
    ap.add_argument("--seed", type=int, default=random_number, help="随机种子，默认 42")
    ap.add_argument("--encoding", type=str, default="utf-8", help="输入/输出编码（默认 utf-8）")
    args = ap.parse_args()

    # 读取输入文件的前两行（原样保留）
    header_line1, header_line2 = read_first_two_lines(args.input, encoding=args.encoding)

    # 用 ASE 读取坐标/速度等
    atoms = read(args.input, format="xyz")
    symbols = atoms.get_chemical_symbols()
    if not all(sym == "C" for sym in symbols):
        print("[WARN] 检测到非 C 的元素标签，将继续处理。")

    pos = atoms.get_positions().astype(float, copy=True)  # (N,3)

    # 找邻居 & 选对
    print("开始近邻搜索（体素哈希）...")
    pairs = find_adjacent_pairs_grid(pos, args.cutoff)
    print(f"找到 {len(pairs)} 对距离 < {args.cutoff:.3f} Angstrom 的相邻原子。")

    chosen = select_disjoint_pairs(pairs, args.npairs, seed=args.seed)
    if len(chosen) < args.npairs:
        print(f"[INFO] 可用的互不重叠对只有 {len(chosen)} 对，将旋转这些。")
    else:
        print(f"已选取 {len(chosen)} 对用于旋转。")

    # 旋转
    for k, (a, b, dist) in enumerate(chosen, 1):
        rotate_about_z_midpoint(pos, a, b, args.angle)
        print(f"第 {k:02d} 对：({a}, {b})，原始距离 {dist:.3f} Angstrom 已旋转 {args.angle}°")

    # 质量（从 ASE 数据库）
    Zc = atomic_numbers["C"]
    mass_C = float(atomic_masses[Zc])  # 12.011...
    masses = np.full((pos.shape[0],), mass_C, dtype=float)

    # 速度：若原子不含速度则全 0
    v = atoms.get_velocities()
    if v is None or v.shape != pos.shape:
        velocities = np.zeros_like(pos)
    else:
        velocities = v.astype(float, copy=True)

    # 旋转后计算 Lattice（包围盒），仅用于日志输出
    Lx, Ly, Lz = compute_lattice_from_bbox(pos)

    # 写扩展 XYZ（前两行与输入一致）
    write_extxyz_preserve_header(
        path=args.output,
        header_line1=header_line1,
        header_line2=header_line2,
        symbols=symbols,
        positions=pos,
        masses=masses,
        velocities=velocities,
        encoding=args.encoding,
    )
    print(f"已写出到: {args.output}")
    print(f"[INFO] 旋转后包围盒 Lattice=({Lx:.6f}, {Ly:.6f}, {Lz:.6f})（仅日志输出；文件头部未改动）")

if __name__ == "__main__":
    main()
